"""``ToolApprovalQueue``: the panel's answer to ``requires_approval`` tools (§4.9)."""

from __future__ import annotations

import asyncio
import json

import pytest
from panel_testkit import FakeOps, RecordingBrain, tool_context

from aivtube.contracts.control import OpCommand, OpKind
from aivtube.contracts.events import Alert, ToolExecuted, ToolRejected
from aivtube.contracts.llm import ToolCall
from aivtube.contracts.tools import ToolPolicy
from aivtube.panel.approvals import ToolApprovalQueue
from aivtube.panel.control import CoreControl
from aivtube.testing.fakes import (
    ECHO_SPEC,
    FakeClock,
    FakeEventBus,
    FakeLLM,
    FakeLLMRouter,
    FakeSafetyGate,
    FakeSpeechOutput,
    FakeTaskSupervisor,
    FakeTool,
    FakeToolRegistry,
)
from aivtube.tools import PolicyToolRegistry


def _call(text: str = "ขออนุญาต") -> ToolCall:
    args = {"text": text}
    return ToolCall(
        id="c1", name="echo", arguments=args, raw_arguments=json.dumps(args), extra=None
    )


async def _until_pending(clock: FakeClock, queue: ToolApprovalQueue, n: int = 1) -> None:
    await clock.run_until(lambda: len(queue.pending()) >= n, within=5.0)


async def test_approve_and_deny_resolve_the_waiting_call() -> None:
    clock, bus = FakeClock(), FakeEventBus()
    queue = ToolApprovalQueue(clock=clock, bus=bus)
    waiting = asyncio.create_task(queue("pailin", _call()))
    await _until_pending(clock, queue)
    [request] = queue.snapshot()
    assert request["tool"] == "echo" and request["character"] == "pailin"
    assert request["args"] == {"text": "ขออนุญาต"}
    [alert] = bus.of_type(Alert)
    assert alert.level == "warn" and "echo" in alert.message and alert.character == "pailin"
    assert queue.resolve(request["id"], True)
    assert await waiting is True
    assert queue.pending() == [] and not queue.resolve(request["id"], False)

    denied = asyncio.create_task(queue("pailin", _call("ไม่ให้")))
    await _until_pending(clock, queue)
    assert queue.resolve(queue.pending()[0].id, False)
    assert await denied is False
    assert (queue.approved, queue.denied) == (1, 1)


async def test_cancelled_or_expired_waits_leave_the_queue() -> None:
    clock = FakeClock()
    queue = ToolApprovalQueue(clock=clock, timeout_s=30.0)
    cancelled = asyncio.create_task(queue("pailin", _call()))
    await _until_pending(clock, queue)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert queue.pending() == []
    expiring = asyncio.create_task(queue("pailin", _call()))
    await _until_pending(clock, queue)
    await clock.run_for(31.0)
    assert await expiring is False
    assert queue.pending() == [] and queue.expired == 2


async def test_a_full_queue_denies_at_once_and_long_args_are_shortened() -> None:
    clock = FakeClock()
    queue = ToolApprovalQueue(clock=clock, max_pending=1)
    first = asyncio.create_task(queue("pailin", _call("x" * 1000)))
    await _until_pending(clock, queue)
    assert await queue("pailin", _call()) is False
    shown = queue.snapshot()[0]["args"]["text"]
    assert len(shown) == 301 and shown.endswith("…")
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    with pytest.raises(ValueError):
        ToolApprovalQueue(clock=clock, max_pending=0)


async def test_real_registry_waits_for_the_panel_approve_command() -> None:
    """``PolicyToolRegistry`` → queue → ``CoreControl`` APPROVE → the tool runs (or not)."""
    clock, bus = FakeClock(), FakeEventBus()
    queue = ToolApprovalQueue(clock=clock, bus=bus)
    tool = FakeTool(ECHO_SPEC, ToolPolicy(requires_approval=True, side_effect=True))
    registry = PolicyToolRegistry(
        [tool],
        gate=FakeSafetyGate(),
        ops=None,
        bus=bus,
        clock=clock,
        enabled_by_character={"pailin": ["echo"]},
        approval=queue,
        approval_timeout_s=20.0,
    )
    brain = RecordingBrain()
    ops = FakeOps()
    tasks = FakeTaskSupervisor(clock)
    control = CoreControl(
        {"pailin": brain},
        router=FakeLLMRouter([FakeLLM([], name="local-30b", clock=clock)], clock=clock),
        registry=FakeToolRegistry(),
        speech=FakeSpeechOutput(bus, clock),
        memory_by_char={},
        safety=FakeSafetyGate(),
        ops=ops,
        bus=bus,
        clock=clock,
        restart=_no_restart,
        tasks=tasks,
        approvals=queue,
    )
    ctx = tool_context(bus, clock)
    run = asyncio.create_task(registry.execute(_call("ขอเปิดเพลง"), ctx))
    await _until_pending(clock, queue)
    [pending] = control.snapshot()["approvals"]
    result = await control.execute(
        OpCommand(OpKind.APPROVE, {"request": pending["id"], "approved": True})
    )
    assert result.ok, result.detail
    assert (await run).ok and tool.calls == [{"text": "ขอเปิดเพลง"}]
    assert bus.of_type(ToolExecuted)[-1].ok
    assert brain.commands == []  # tool approvals never reach the brain

    run = asyncio.create_task(registry.execute(_call("อีกครั้ง"), ctx))
    await _until_pending(clock, queue)
    request_id = queue.pending()[0].id
    denied = await control.execute(
        OpCommand(OpKind.APPROVE, {"request": request_id, "approved": False})
    )
    assert denied.ok
    assert not (await run).ok and bus.of_type(ToolRejected)[-1].tool == "echo"
    late = await control.execute(OpCommand(OpKind.APPROVE, {"request": request_id}))
    assert not late.ok and "no pending" in late.detail
    # other APPROVE shapes (e.g. the M2 review gate) still go to the brain
    assert (await control.execute(OpCommand(OpKind.APPROVE, {"utt_id": "u1"}))).ok
    assert brain.kinds() == [OpKind.APPROVE]
    await clock.run_until_idle()
    assert [r["command"] for r in ops.rows] == ["approve"] * 4
    await tasks.aclose()


async def test_the_registry_deadline_denies_and_clears_the_request() -> None:
    clock, bus = FakeClock(), FakeEventBus()
    queue = ToolApprovalQueue(clock=clock, bus=bus)
    tool = FakeTool(ECHO_SPEC, ToolPolicy(requires_approval=True))
    registry = PolicyToolRegistry(
        [tool],
        gate=FakeSafetyGate(),
        ops=None,
        bus=bus,
        clock=clock,
        enabled_by_character={"pailin": ["echo"]},
        approval=queue,
        approval_timeout_s=20.0,
    )
    run = asyncio.create_task(registry.execute(_call(), tool_context(bus, clock)))
    await _until_pending(clock, queue)
    await clock.run_for(20.5)
    result = await run
    assert not result.ok and "no answer within 20 s" in result.content
    assert queue.pending() == [] and queue.expired == 1 and tool.calls == []


async def _no_restart(component: str) -> None:
    return None
