"""ToolFlow: call order, follow-up rules, extra_content stripping (§4.9)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from brain_testkit import stim

from aivtube.brain.tool_flow import UNAVAILABLE_RESULT, ToolFlow
from aivtube.contracts.llm import ToolSpec
from aivtube.contracts.tools import ToolContext, ToolPolicy, ToolResult
from aivtube.contracts.types import StimulusKind
from aivtube.testing.fakes import (
    FakeClock,
    FakeEventBus,
    FakeMemoryStore,
    FakeSpeechOutput,
    FakeTool,
    FakeToolRegistry,
    tool_call,
)

SPEC_A = ToolSpec("a", "tool a", {"type": "object", "properties": {"n": {"type": "integer"}}})
SPEC_B = ToolSpec("b", "tool b", {"type": "object", "properties": {"n": {"type": "integer"}}})


def ctx(clock: FakeClock, bus: FakeEventBus) -> ToolContext:
    return ToolContext(
        character="pailin",
        turn_id="t1",
        stimulus=stim(StimulusKind.VOICE),
        memory=FakeMemoryStore(clock=clock),
        speech=FakeSpeechOutput(bus, clock),
        avatar=None,
        channels={},
        bus=bus,
        clock=clock,
    )


def handler(name: str, order: list[str]) -> Any:
    async def run(args: Mapping[str, Any], c: ToolContext) -> ToolResult:
        order.append(f"{name}{args['n']}")
        return ToolResult(True, json.dumps({"tool": name, "n": args["n"]}))

    return run


async def test_results_keep_call_order(fake_clock: FakeClock, fake_bus: FakeEventBus) -> None:
    order: list[str] = []
    reg = FakeToolRegistry(
        [
            FakeTool(SPEC_A, handler=handler("a", order)),
            FakeTool(SPEC_B, handler=handler("b", order)),
        ]
    )
    flow = ToolFlow(reg, clock=fake_clock)
    calls = [tool_call("b", {"n": 1}), tool_call("a", {"n": 2}), tool_call("b", {"n": 3})]
    results = await flow.run(calls, ctx(fake_clock, fake_bus))
    assert order == ["b1", "a2", "b3"]
    assert [c.id for c, _ in results] == [c.id for c in calls]
    msgs = flow.tool_messages(
        {"role": "assistant", "content": "", "tool_calls": []}, results, keep_extra=False
    )
    assert [m["role"] for m in msgs] == ["assistant", "tool", "tool", "tool"]
    assert [json.loads(m["content"])["n"] for m in msgs[1:]] == [1, 2, 3]
    assert [m["tool_call_id"] for m in msgs[1:]] == [c.id for c in calls]
    assert all(isinstance(m["content"], str) for m in msgs)


async def test_failures_become_results(fake_clock: FakeClock, fake_bus: FakeEventBus) -> None:
    reg = FakeToolRegistry([FakeTool(SPEC_A, raises=RuntimeError("boom"))])
    flow = ToolFlow(reg, clock=fake_clock)
    [(_, res)] = await flow.run([tool_call("a", {"n": 1})], ctx(fake_clock, fake_bus))
    assert not res.ok and "failed" in res.content
    [(_, unknown)] = await flow.run([tool_call("zzz", {})], ctx(fake_clock, fake_bus))
    assert not unknown.ok
    assert json.loads(UNAVAILABLE_RESULT.content) == {"ok": False, "error": "unavailable now"}


def test_follow_up_rules() -> None:
    reg = FakeToolRegistry([FakeTool(SPEC_A), FakeTool(SPEC_B, ToolPolicy(follow_up=True))])
    flow = ToolFlow(
        reg,
        follow_up_tools=ToolFlow.follow_up_names([FakeTool(SPEC_B, ToolPolicy(follow_up=True))]),
    )
    ok = ToolResult(True, "{}")
    a = [(tool_call("a", {"n": 1}), ok)]
    b = [(tool_call("b", {"n": 1}), ok)]
    # tool-only reply (nothing spoken): exactly one follow-up
    assert flow.needs_follow_up(a, spoke=False, rounds=1)
    assert not flow.needs_follow_up(a, spoke=False, rounds=2)  # never more than max_rounds
    # the reply spoke: follow up only for a follow_up tool
    assert not flow.needs_follow_up(a, spoke=True)
    assert flow.needs_follow_up(b, spoke=True)
    assert not flow.needs_follow_up(b, spoke=True, rounds=2)
    # no tool results: nothing to follow up
    assert not flow.needs_follow_up([], spoke=False)
    single = ToolFlow(reg, max_rounds=1)
    assert not single.needs_follow_up(a, spoke=False)


def test_keep_extra_false_strips_extra_content() -> None:
    flow = ToolFlow(FakeToolRegistry())
    call = tool_call("a", {"n": 1}, extra={"google": {"thought_signature": "sig"}})
    assistant = {
        "role": "assistant",
        "content": "ok",
        "extra_content": {"x": 1},
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": "a", "arguments": call.raw_arguments},
                "extra_content": {"google": {"thought_signature": "sig"}},
            }
        ],
    }
    results = [(call, ToolResult(True, "{}"))]
    stripped = flow.tool_messages(assistant, results, keep_extra=False)[0]
    assert "extra_content" not in stripped
    assert "extra_content" not in stripped["tool_calls"][0]
    assert stripped["tool_calls"][0]["function"]["arguments"] == call.raw_arguments  # verbatim
    kept = flow.tool_messages(assistant, results, keep_extra=True)[0]
    assert kept["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert "extra_content" in assistant["tool_calls"][0]  # the input is not modified
