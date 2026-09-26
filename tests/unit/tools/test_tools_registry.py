"""``PolicyToolRegistry``: pipeline order, policy and failures (§4.9)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from tools_testkit import ListAudit, call

from aivtube.contracts.events import ToolExecuted, ToolRejected, ToolRequested
from aivtube.contracts.llm import ToolCall, ToolSpec
from aivtube.contracts.safety import FilterResult, Verdict
from aivtube.contracts.tools import RateLimit, Tool, ToolContext, ToolPolicy, ToolResult
from aivtube.contracts.types import Platform
from aivtube.infra import SupervisedTasks
from aivtube.memory import OpsDb
from aivtube.testing.fakes import (
    ECHO_SPEC,
    FakeChannelActions,
    FakeClock,
    FakeEventBus,
    FakeSafetyGate,
    FakeTool,
)
from aivtube.tools import UNAVAILABLE, PolicyToolRegistry, parse_tool_arguments

BLOCKED = "คำต้องห้าม"
EMPTY_SPEC = ToolSpec("ping", "No arguments.", {"type": "object", "properties": {}})
NESTED_SPEC = ToolSpec(
    "poll",
    "Nested free text.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 20},
            "choices": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["title"],
        "additionalProperties": False,
    },
)


def body(res: ToolResult) -> dict[str, Any]:
    out = json.loads(res.content)
    assert isinstance(out, dict)
    return out


class Harness:
    def __init__(
        self,
        clock: FakeClock,
        events: FakeEventBus,
        make_ctx: Callable[..., ToolContext],
        tools: Sequence[Tool],
        **kw: Any,
    ) -> None:
        self.clock = clock
        self.events = events
        self.audit = ListAudit()
        self.gate = kw.pop("gate", FakeSafetyGate([BLOCKED]))
        self.reg = PolicyToolRegistry(
            tools,
            gate=self.gate,
            ops=kw.pop("ops", self.audit),
            bus=events,
            clock=clock,
            enabled_by_character=kw.pop(
                "enabled", {"pailin": [t.spec.name for t in tools], "mali": ["echo"]}
            ),
            **kw,
        )
        self.ctx = make_ctx()


@pytest.fixture
def harness(
    clock: FakeClock, events: FakeEventBus, make_ctx: Callable[..., ToolContext]
) -> Callable[..., Harness]:
    def factory(*tools: Tool, **kw: Any) -> Harness:
        return Harness(clock, events, make_ctx, tools or (FakeTool(ECHO_SPEC),), **kw)

    return factory


# --- specs ------------------------------------------------------------------------------------------
def test_specs_are_static_per_character_and_byte_stable(harness: Callable[..., Harness]) -> None:
    tools = (FakeTool(ECHO_SPEC), FakeTool(EMPTY_SPEC), FakeTool(NESTED_SPEC))
    h = harness(*tools, enabled={"pailin": ["poll", "echo", "nope"], "mali": ["ping"]})
    first = h.reg.specs("pailin")
    assert [s.name for s in first] == ["echo", "poll"]  # registration order, unknown ignored
    assert [s.name for s in h.reg.specs("mali")] == ["ping"]
    assert h.reg.specs("stranger") == ()
    h.reg.set_enabled("echo", False)
    h.reg.set_mode("off")
    assert h.reg.specs("pailin") is first
    again = harness(*tools, enabled={"pailin": ["echo", "poll"]}).reg.specs("pailin")
    dump = [json.dumps([s.name, s.description, s.parameters], ensure_ascii=False) for s in first]
    assert dump == [
        json.dumps([s.name, s.description, s.parameters], ensure_ascii=False) for s in again
    ]


def test_constructor_validation(harness: Callable[..., Harness]) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        harness(FakeTool(ECHO_SPEC), FakeTool(ECHO_SPEC))
    with pytest.raises(ValueError, match="object"):
        harness(FakeTool(ToolSpec("x", "", {"type": "string"})))
    with pytest.raises(Exception, match="not of type"):
        harness(FakeTool(ToolSpec("x", "", {"type": "object", "properties": 5})))
    bad = FakeTool(ToolSpec("x", "", {"type": "object"}))
    bad.arg_direction = "sideways"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="arg_direction"):
        harness(bad)
    h = harness()
    with pytest.raises(KeyError):
        h.reg.set_enabled("nope", True)
    with pytest.raises(ValueError):
        h.reg.set_mode("maybe")  # type: ignore[arg-type]


# --- parsing and validation ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "arguments", "expected"),
    [
        ('{"text": "ok"}', None, {"text": "ok"}),
        ("{'text': 'single'}", None, {"text": "single"}),
        ('{"text": "trailing",}', None, {"text": "trailing"}),
        ('{"text": "no brace"', None, {"text": "no brace"}),
        ('```json\n{"text": "fenced"}\n```', None, {"text": "fenced"}),
        ("", None, {}),
        ("", {"text": "from arguments"}, {"text": "from arguments"}),
        ('{"text": "raw wins"}', {"text": "provider copy"}, {"text": "raw wins"}),
    ],
)
def test_parse_tool_arguments(raw: str, arguments: Any, expected: dict[str, Any]) -> None:
    tc = ToolCall(id="c", name="echo", arguments=arguments, raw_arguments=raw, extra=None)
    assert parse_tool_arguments(tc) == (expected, None)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('{"text": "ตัดกลางคำ', "cut off"),
        ('{"text": "escaped \\" quote', "cut off"),
        ("[1, 2]", "not a valid JSON object"),
        ("hello", "not a valid JSON object"),
        ('"just a string"', "not a valid JSON object"),
    ],
)
def test_parse_tool_arguments_refusals(raw: str, reason: str) -> None:
    tc = ToolCall(id="c", name="echo", arguments={"text": "x"}, raw_arguments=raw, extra=None)
    args, why = parse_tool_arguments(tc)
    assert args is None and why is not None and reason in why


async def test_validation_messages_do_not_echo_values(harness: Callable[..., Harness]) -> None:
    tool = FakeTool(NESTED_SPEC)
    h = harness(tool)
    secret = "ข้อความยาวมากเกินไปมากๆๆๆๆๆๆๆ"
    res = await h.reg.execute(call("poll", {"title": secret, "choices": [1], "x": 2}), h.ctx)
    assert not res.ok and body(res)["error"] == "invalid arguments"
    details = body(res)["details"]
    assert "unexpected argument(s): x" in details
    assert "title: maxLength is 20" in details
    assert "choices/0: must be of type string" in details
    assert secret not in res.content
    res = await h.reg.execute(call("poll", {}), h.ctx)
    assert body(res)["details"] == ["missing required argument(s): title"]
    assert not tool.calls and h.gate.arg_checks == []


# --- pipeline order ------------------------------------------------------------------------------------
async def test_pipeline_order(harness: Callable[..., Harness]) -> None:
    tool = FakeTool(ECHO_SPEC)
    h = harness(tool)
    h.reg.set_enabled("echo", False)
    steps = [
        (call("nope", {}), "unknown"),
        (call("echo", None, raw='{"text": "ตัด'), "invalid"),  # parse before anything else
        (call("echo", {"text": 7}), "invalid"),  # schema before the filter
        (call("echo", {"text": f"พูด {BLOCKED}"}), "blocked"),  # filter before "enabled"
        (call("echo", {"text": "ปกติ"}), "unavailable"),  # disabled: listed but unavailable
    ]
    for tc, verdict in steps:
        res = await h.reg.execute(tc, h.ctx)
        assert not res.ok
        assert h.audit.verdicts[-1] == verdict, (tc, h.audit.verdicts)
    assert "unavailable" in body(await h.reg.execute(call("echo", {"text": "x"}), h.ctx))["error"]
    # the filter only ran for calls that got past validation
    assert [c[1] for c in h.gate.arg_checks] == [(f"พูด {BLOCKED}",), ("ปกติ",), ("x",)]
    assert all(c[0] == "tool" for c in h.gate.arg_checks)
    assert not tool.calls
    h.reg.set_enabled("echo", True)
    ok = await h.reg.execute(call("echo", {"text": "สวัสดี"}), h.ctx)
    assert ok.ok and tool.calls == [{"text": "สวัสดี"}]
    assert h.audit.verdicts[-1] == "ok"
    assert len(h.audit.rows) == 7  # every call audited
    assert len(h.events.of_type(ToolRequested)) == 7
    assert len(h.events.of_type(ToolRejected)) == 6
    assert [e.ok for e in h.events.of_type(ToolExecuted)] == [True]


async def test_tools_outside_the_characters_list_are_unknown(
    harness: Callable[..., Harness], make_ctx: Callable[..., ToolContext]
) -> None:
    ping = FakeTool(EMPTY_SPEC)
    h = harness(FakeTool(ECHO_SPEC), ping, enabled={"pailin": ["echo"]})
    res = await h.reg.execute(call("ping", {}), h.ctx)
    assert not res.ok and body(res)["error"] == "unknown tool 'ping'" and not ping.calls


async def test_blocked_arguments_never_reach_audit_or_result(
    harness: Callable[..., Harness],
) -> None:
    tool = FakeTool(NESTED_SPEC)
    h = harness(tool)
    res = await h.reg.execute(call("poll", {"title": "โพล", "choices": ["a", BLOCKED]}), h.ctx)
    assert not res.ok and "blocked" in body(res)["error"]
    assert BLOCKED not in res.content
    row = h.audit.rows[-1]
    assert BLOCKED not in row["args"] and "_blocked_sha256" in row["args"]
    assert h.gate.arg_checks[-1][1] == ("โพล", "a", BLOCKED)  # nested values are checked


class MaskingGate(FakeSafetyGate):
    """Masks 'http…' links; fails or stalls on demand."""

    def __init__(self, *, fail: bool = False, stall: FakeClock | None = None) -> None:
        super().__init__([BLOCKED])
        self.fail = fail
        self.stall = stall

    async def check_args(
        self,
        direction: Literal["tool", "memory", "game"],
        texts: Sequence[str],
        *,
        character: str,
    ) -> FilterResult:
        if self.fail:
            raise RuntimeError("classifier down")
        if self.stall is not None:
            await self.stall.sleep(60)
        res = await super().check_args(direction, texts, character=character)
        if res.verdict is not Verdict.PASS:
            return res
        joined = "\n".join(texts)
        if "http" not in joined:
            return res
        masked = " ".join("[ลิงก์]" if w.startswith("http") else w for w in joined.split(" "))
        return FilterResult(Verdict.MASK, masked, "tier0", rule="url")


async def test_masked_arguments_are_rewritten(harness: Callable[..., Harness]) -> None:
    tool = FakeTool(NESTED_SPEC)
    h = harness(tool, gate=MaskingGate())
    res = await h.reg.execute(
        call("poll", {"title": "ดู http://x.y", "choices": ["ok", "http://a.b"]}), h.ctx
    )
    assert res.ok
    assert tool.calls == [{"title": "ดู [ลิงก์]", "choices": ["ok", "[ลิงก์]"]}]
    assert "http" not in h.audit.rows[-1]["args"]


async def test_filter_failure_or_stall_fails_closed(
    harness: Callable[..., Harness], clock: FakeClock
) -> None:
    tool = FakeTool(ECHO_SPEC)
    h = harness(tool, gate=MaskingGate(fail=True))
    res = await h.reg.execute(call("echo", {"text": "hi"}), h.ctx)
    assert not res.ok and h.audit.verdicts == ["blocked"]
    h = harness(tool, gate=MaskingGate(stall=clock), filter_timeout_s=2.0)
    task = asyncio.create_task(h.reg.execute(call("echo", {"text": "hi"}), h.ctx))
    await clock.run_for(2.5)
    res = await task
    assert not res.ok and h.audit.verdicts == ["blocked"] and not tool.calls


# --- policy ------------------------------------------------------------------------------------------------
async def test_mode_off_and_dry_run(harness: Callable[..., Harness], tmp_path: Any) -> None:
    tool = FakeTool(ECHO_SPEC)
    ops = OpsDb(tmp_path / "ops.db")
    h = harness(tool, ops=ops, mode="dry_run")
    try:
        assert h.reg.mode == "dry_run"
        res = await h.reg.execute(call("echo", {"text": "ลองดู"}), h.ctx)
        assert res.ok and body(res)["dry_run"] is True and not tool.calls
        assert h.events.of_type(ToolExecuted)[-1].dry_run
        h.reg.set_mode("off")
        res = await h.reg.execute(call("echo", {"text": "x"}), h.ctx)
        assert not res.ok and UNAVAILABLE in res.content
        h.reg.set_mode("live")
        assert (await h.reg.execute(call("echo", {"text": "x"}), h.ctx)).ok
        rows = await ops.audit_rows("tool_audit")
        assert [(r["verdict"], r["dry_run"]) for r in rows] == [
            ("dry_run", 1),
            ("off", 0),
            ("ok", 0),
        ]
        assert json.loads(rows[0]["args"]) == {"text": "ลองดู"}
        assert rows[0]["character"] == "pailin" and rows[0]["turn_id"] == "t-1"
    finally:
        await ops.aclose()


async def test_required_capabilities(
    harness: Callable[..., Harness], make_ctx: Callable[..., ToolContext]
) -> None:
    tool = FakeTool(EMPTY_SPEC, ToolPolicy(requires=frozenset({"timeout"})))
    h = harness(tool)
    res = await h.reg.execute(call("ping", {}), h.ctx)
    assert not res.ok and body(res) == {"ok": False, "error": UNAVAILABLE, "needs": ["timeout"]}
    ctx = make_ctx(channels={Platform.TWITCH: FakeChannelActions(capabilities={"send"})})
    assert not (await h.reg.execute(call("ping", {}), ctx)).ok
    ctx = make_ctx(channels={Platform.TWITCH: FakeChannelActions(capabilities={"timeout"})})
    assert (await h.reg.execute(call("ping", {}), ctx)).ok


async def test_rate_limit(harness: Callable[..., Harness], clock: FakeClock) -> None:
    tool = FakeTool(EMPTY_SPEC, ToolPolicy(rate_limit=RateLimit(2, 60.0)))
    h = harness(tool)
    assert (await h.reg.execute(call("ping", {}), h.ctx)).ok
    clock.advance(10)
    assert (await h.reg.execute(call("ping", {}), h.ctx)).ok
    limited = await h.reg.execute(call("ping", {}), h.ctx)
    assert not limited.ok and body(limited)["retry_in_s"] == 50
    h.reg.set_mode("dry_run")  # a dry run reports the limit too
    assert not (await h.reg.execute(call("ping", {}), h.ctx)).ok
    h.reg.set_mode("live")
    clock.advance(50)
    assert (await h.reg.execute(call("ping", {}), h.ctx)).ok
    assert len(tool.calls) == 3
    assert h.audit.verdicts == ["ok", "ok", "rate_limited", "rate_limited", "ok"]


async def test_approval(harness: Callable[..., Harness], clock: FakeClock) -> None:
    asked: list[tuple[str, ToolCall]] = []
    answer: dict[str, Any] = {"value": True}

    async def approve(character: str, tc: ToolCall) -> bool:
        asked.append((character, tc))
        if answer["value"] == "hang":
            await asyncio.Event().wait()
        if answer["value"] == "boom":
            raise RuntimeError("panel down")
        return bool(answer["value"])

    tool = FakeTool(ECHO_SPEC, ToolPolicy(requires_approval=True, side_effect=True))
    h = harness(tool, approval=approve)
    assert (await h.reg.execute(call("echo", {"text": "ขออนุญาต"}), h.ctx)).ok
    assert h.audit.rows[-1]["approved_by"] == "operator"
    assert asked[-1][0] == "pailin" and asked[-1][1].arguments == {"text": "ขออนุญาต"}
    answer["value"] = False
    res = await h.reg.execute(call("echo", {"text": "ไม่ให้"}), h.ctx)
    assert not res.ok and body(res)["detail"] == "denied by the operator"
    answer["value"] = "boom"
    assert not (await h.reg.execute(call("echo", {"text": "พัง"}), h.ctx)).ok
    # default deny after 20 s without an answer (FakeClock)
    answer["value"] = "hang"
    task = asyncio.create_task(h.reg.execute(call("echo", {"text": "รอ"}), h.ctx))
    await clock.run_for(19.0)
    assert not task.done()
    await clock.run_for(1.5)
    res = await task
    assert not res.ok and body(res)["detail"] == "no answer within 20 s"
    assert [c["text"] for c in tool.calls] == ["ขออนุญาต"]
    assert h.audit.verdicts == ["ok", "denied", "denied", "denied"]
    # without an approval queue, approval-gated tools are always denied
    h2 = harness(tool)
    assert not (await h2.reg.execute(call("echo", {"text": "x"}), h2.ctx)).ok


# --- execution failures --------------------------------------------------------------------------------------
async def test_timeout_is_a_result_and_never_retried(
    harness: Callable[..., Harness], clock: FakeClock
) -> None:
    calls = 0

    async def slow(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        nonlocal calls
        calls += 1
        await ctx.clock.sleep(30)
        return ToolResult(True, "{}")

    tool = FakeTool(EMPTY_SPEC, ToolPolicy(side_effect=True, timeout_s=3.0), handler=slow)
    h = harness(tool)
    task = asyncio.create_task(h.reg.execute(call("ping", {}), h.ctx))
    await clock.run_for(3.5)
    res = await task
    assert not res.ok and body(res)["error"] == "timed out after 3 s"
    assert calls == 1 and h.audit.verdicts == ["timeout"]
    assert h.events.of_type(ToolExecuted)[-1].ok is False


async def test_timeout_is_capped(harness: Callable[..., Harness], clock: FakeClock) -> None:
    async def slow(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        await ctx.clock.sleep(10_000)
        return ToolResult(True, "{}")

    tool = FakeTool(EMPTY_SPEC, ToolPolicy(timeout_s=5_000.0), handler=slow)
    h = harness(tool, timeout_max_s=600.0)
    task = asyncio.create_task(h.reg.execute(call("ping", {}), h.ctx))
    await clock.run_for(601)
    assert "600 s" in (await task).content


async def test_exceptions_and_bad_results(harness: Callable[..., Harness]) -> None:
    broken = FakeTool(ECHO_SPEC, ToolPolicy(side_effect=True), raises=ValueError("secret detail"))
    h = harness(broken)
    res = await h.reg.execute(call("echo", {"text": "x"}), h.ctx)
    assert not res.ok and body(res)["error"] == "failed: ValueError"
    assert "secret detail" not in res.content and res.note is not None
    assert "secret detail" in res.note
    assert len(broken.calls) == 1

    async def nothing(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        return "not a result"  # type: ignore[return-value]

    h = harness(FakeTool(ECHO_SPEC, handler=nothing))
    res = await h.reg.execute(call("echo", {"text": "x"}), h.ctx)
    assert not res.ok and h.audit.verdicts == ["failed"]


async def test_tool_level_failure_is_audited_as_error(harness: Callable[..., Harness]) -> None:
    tool = FakeTool(ECHO_SPEC, result=ToolResult(False, '{"ok": false, "error": "slot_empty"}'))
    h = harness(tool)
    res = await h.reg.execute(call("echo", {"text": "x"}), h.ctx)
    assert not res.ok and h.audit.verdicts == ["error"]


async def test_cancellation_propagates_and_is_audited(
    harness: Callable[..., Harness], clock: FakeClock, events: FakeEventBus
) -> None:
    started = asyncio.Event()

    async def hang(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    tasks = SupervisedTasks(clock, events, on_critical_failure=lambda *a: None)
    h = harness(FakeTool(EMPTY_SPEC, handler=hang), tasks=tasks)
    task = asyncio.create_task(h.reg.execute(call("ping", {}), h.ctx))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await clock.run_until_idle()
    assert h.audit.verdicts == ["cancelled"]
    await tasks.aclose()
    # without a supervisor the cancellation is still propagated (and only logged)
    started.clear()
    h2 = harness(FakeTool(EMPTY_SPEC, handler=hang))
    task = asyncio.create_task(h2.reg.execute(call("ping", {}), h2.ctx))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h2.audit.rows == []


async def test_audit_failure_does_not_break_the_call(harness: Callable[..., Harness]) -> None:
    class Broken:
        async def log_tool(self, **row: Any) -> None:
            raise RuntimeError("disk full")

    h = harness(ops=Broken())
    assert (await h.reg.execute(call("echo", {"text": "x"}), h.ctx)).ok
    h = harness(ops=None)
    assert (await h.reg.execute(call("echo", {"text": "x"}), h.ctx)).ok


def test_snapshot(harness: Callable[..., Harness]) -> None:
    h = harness()
    h.reg.set_enabled("echo", False)
    assert h.reg.snapshot() == {"mode": "live", "tools": {"echo": {"enabled": False}}}
    assert not h.reg.is_enabled("echo")


async def test_internal_errors_become_results(harness: Callable[..., Harness]) -> None:
    tool = FakeTool(EMPTY_SPEC)
    h = harness(tool)
    tool.policy = SimpleNamespace(requires=None)  # type: ignore[assignment]  # a broken tool
    res = await h.reg.execute(call("ping", {}), h.ctx)
    assert not res.ok and body(res)["error"] == "failed: internal error"
    assert h.audit.verdicts == ["failed"] and not tool.calls


def test_deeply_nested_arguments_are_refused() -> None:
    deep = "[" * 100_000
    tc = ToolCall(id="c", name="echo", arguments=None, raw_arguments=deep, extra=None)
    args, why = parse_tool_arguments(tc)
    assert args is None and why is not None
