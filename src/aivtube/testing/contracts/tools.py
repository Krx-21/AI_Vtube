"""Contract suite for ``ToolRegistry`` (§3.11, §4.9)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

from aivtube.contracts.llm import ToolCall, ToolSpec
from aivtube.contracts.safety import SafetyGate
from aivtube.contracts.tools import Tool, ToolContext, ToolPolicy, ToolRegistry, ToolResult
from aivtube.contracts.types import Stimulus, StimulusKind
from aivtube.testing.contracts._base import AsyncCase, _Cases, check, maybe_await
from aivtube.testing.fakes.avatar import FakeAvatarSink
from aivtube.testing.fakes.bus import FakeEventBus
from aivtube.testing.fakes.clock import RealClock
from aivtube.testing.fakes.memory import FakeMemoryStore
from aivtube.testing.fakes.safety import FakeSafetyGate
from aivtube.testing.fakes.speech import FakeSpeechOutput
from aivtube.testing.fakes.tools import ECHO_SPEC, FakeTool

__all__ = ["BLOCKED_ARG", "make_tool_context", "tool_registry_suite"]

BLOCKED_ARG = "คำต้องห้าม"

SLOW_SPEC = ToolSpec(
    name="slow",
    description="Never finishes in time (test tool).",
    parameters={"type": "object", "properties": {}},
)
BROKEN_SPEC = ToolSpec(
    name="broken",
    description="Always raises (test tool).",
    parameters={"type": "object", "properties": {}},
)


def make_tool_context(character: str = "pailin") -> ToolContext:
    """A ``ToolContext`` wired to fakes, for registry and tool tests."""
    clock = RealClock()
    bus = FakeEventBus(clock)
    stim = Stimulus(
        id="s-1", kind=StimulusKind.VOICE, character=character, text="ทดสอบ", created=clock.now()
    )
    return ToolContext(
        character=character,
        turn_id="t-1",
        stimulus=stim,
        memory=FakeMemoryStore(character, clock),
        speech=FakeSpeechOutput(bus, clock),
        avatar=FakeAvatarSink(),
        channels={},
        bus=bus,
        clock=clock,
    )


def _call(name: str, args: dict[str, object] | None, raw: str | None = None) -> ToolCall:
    raw_args = raw if raw is not None else json.dumps(args, ensure_ascii=False)
    return ToolCall(
        id=f"call-{name}", name=name, arguments=args, raw_arguments=raw_args, extra=None
    )


def tool_registry_suite(
    factory: Callable[[Sequence[Tool], SafetyGate], ToolRegistry | Awaitable[ToolRegistry]],
    *,
    character: str = "pailin",
) -> list[AsyncCase]:
    """``factory(tools, gate)`` builds a registry exposing exactly ``tools`` (all enabled for
    ``character``, live mode, no approvals) and filtering free-text arguments with ``gate``,
    which blocks ``BLOCKED_ARG``."""
    cases = _Cases("tool_registry")

    async def setup() -> tuple[ToolRegistry, dict[str, FakeTool], ToolContext]:
        tools = {
            "echo": FakeTool(ECHO_SPEC),
            "slow": FakeTool(SLOW_SPEC, ToolPolicy(timeout_s=0.1), delay_s=5.0),
            "broken": FakeTool(BROKEN_SPEC, raises=RuntimeError("boom")),
        }
        gate = FakeSafetyGate([BLOCKED_ARG])
        reg = await maybe_await(factory(list(tools.values()), gate))
        return reg, tools, make_tool_context(character)

    def refused(res: ToolResult) -> bool:
        return isinstance(res, ToolResult) and res.ok is False and isinstance(res.content, str)

    @cases
    async def specs_are_stable_and_ordered() -> None:
        reg, _, _ = await setup()
        first = reg.specs(character)
        check([s.name for s in first] == ["echo", "slow", "broken"], f"order {first}")
        reg.set_enabled("echo", False)
        check(reg.specs(character) == first, "specs() changed after set_enabled (cache-stable)")

    @cases
    async def valid_call_runs_the_tool() -> None:
        reg, tools, ctx = await setup()
        res = await reg.execute(_call("echo", {"text": "สวัสดี"}), ctx)
        check(res.ok, f"valid call refused: {res!r}")
        check(tools["echo"].calls == [{"text": "สวัสดี"}], f"tool saw {tools['echo'].calls}")

    @cases
    async def invalid_arguments_are_refused() -> None:
        reg, tools, ctx = await setup()
        check(refused(await reg.execute(_call("echo", {}), ctx)), "missing required arg accepted")
        check(refused(await reg.execute(_call("echo", {"text": 5}), ctx)), "wrong type accepted")
        bad_json = _call("echo", None, raw='{"text": "ไม่ปิด')
        check(refused(await reg.execute(bad_json, ctx)), "unparseable arguments accepted")
        check(not tools["echo"].calls, "the tool ran with invalid arguments")

    @cases
    async def filtered_arguments_are_refused() -> None:
        reg, tools, ctx = await setup()
        res = await reg.execute(_call("echo", {"text": f"พูดว่า {BLOCKED_ARG}"}), ctx)
        check(refused(res), "blocked argument text accepted")
        check(not tools["echo"].calls, "the tool ran with blocked arguments")

    @cases
    async def disabled_tool_is_unavailable() -> None:
        reg, tools, ctx = await setup()
        reg.set_enabled("echo", False)
        res = await reg.execute(_call("echo", {"text": "hi"}), ctx)
        check(refused(res) and "unavailable" in res.content, f"disabled tool: {res!r}")
        check(not tools["echo"].calls, "a disabled tool ran")
        reg.set_enabled("echo", True)
        check((await reg.execute(_call("echo", {"text": "hi"}), ctx)).ok, "re-enable failed")

    @cases
    async def unknown_tool_is_refused() -> None:
        reg, _, ctx = await setup()
        check(refused(await reg.execute(_call("nope", {}), ctx)), "unknown tool accepted")

    @cases
    async def modes_dry_run_and_off() -> None:
        reg, tools, ctx = await setup()
        reg.set_mode("dry_run")
        await reg.execute(_call("echo", {"text": "ลองดู"}), ctx)
        check(not tools["echo"].calls, "dry_run executed the tool")
        reg.set_mode("off")
        check(refused(await reg.execute(_call("echo", {"text": "x"}), ctx)), "off mode ran")
        reg.set_mode("live")
        check((await reg.execute(_call("echo", {"text": "x"}), ctx)).ok, "live mode refused")

    @cases
    async def failures_become_results() -> None:
        reg, _, ctx = await setup()
        slow = await asyncio.wait_for(reg.execute(_call("slow", {}), ctx), 3.0)
        check(refused(slow), f"timeout not reported: {slow!r}")
        broken = await reg.execute(_call("broken", {}), ctx)
        check(refused(broken), f"exception not reported: {broken!r}")

    return cases.items
