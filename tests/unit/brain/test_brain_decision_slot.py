"""``DecisionSlot`` cancellation semantics (§4.1)."""

from __future__ import annotations

import asyncio

import pytest

from aivtube.brain.decision import DecisionResult, DecisionSlot
from aivtube.testing.fakes import FakeClock, FakeTaskSupervisor


async def _decision(clock: FakeClock, turn: str, seconds: float, log: list[str]) -> DecisionResult:
    log.append(f"start:{turn}")
    try:
        await clock.sleep(seconds)
    finally:
        log.append(f"end:{turn}")
    return DecisionResult(turn_id=turn)


async def test_ok_outcome(fake_clock: FakeClock) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))
    log: list[str] = []
    task = asyncio.create_task(slot.run("t1", _decision(fake_clock, "t1", 1.0, log)))
    await fake_clock.run_until_idle()
    assert slot.busy and slot.current_turn == "t1"
    await fake_clock.run_for(1.0)
    outcome = await task
    assert outcome.status == "ok" and outcome.result == DecisionResult(turn_id="t1")
    assert not slot.busy


async def test_cancelled_child_returns_aborted_and_does_not_raise(fake_clock: FakeClock) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))
    log: list[str] = []
    task = asyncio.create_task(slot.run("t1", _decision(fake_clock, "t1", 10.0, log)))
    await fake_clock.run_until_idle()
    assert slot.cancel("preempt_critical") is True
    outcome = await task
    assert outcome.status == "aborted" and outcome.reason == "preempt_critical"
    assert log == ["start:t1", "end:t1"]
    assert slot.cancel("again") is False


async def test_failed_child_returns_failed(fake_clock: FakeClock) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))

    async def boom() -> DecisionResult:
        raise RuntimeError("bad")

    outcome = await slot.run("t1", boom())
    assert outcome.status == "failed" and outcome.reason == "RuntimeError: bad"
    assert isinstance(outcome.error, RuntimeError)


async def test_cancelling_the_brain_task_cancels_the_child_and_propagates(
    fake_clock: FakeClock,
) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))
    log: list[str] = []
    brain = asyncio.create_task(slot.run("t1", _decision(fake_clock, "t1", 10.0, log)))
    await fake_clock.run_until_idle()
    brain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await brain
    await fake_clock.run_until_idle()
    assert log == ["start:t1", "end:t1"]  # the child was cancelled too
    assert not slot.busy


async def test_at_most_one_decision_in_flight(fake_clock: FakeClock) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))
    log: list[str] = []
    first = asyncio.create_task(slot.run("t1", _decision(fake_clock, "t1", 1.0, log)))
    second = asyncio.create_task(slot.run("t2", _decision(fake_clock, "t2", 1.0, log)))
    await fake_clock.run_until_idle()
    assert log == ["start:t1"]
    await fake_clock.run_for(2.5)
    assert (await first).status == "ok" and (await second).status == "ok"
    assert log == ["start:t1", "end:t1", "start:t2", "end:t2"]


async def test_cancelled_while_waiting_for_the_lock_closes_the_coroutine(
    fake_clock: FakeClock,
) -> None:
    slot = DecisionSlot(FakeTaskSupervisor(fake_clock))
    log: list[str] = []
    first = asyncio.create_task(slot.run("t1", _decision(fake_clock, "t1", 1.0, log)))
    await fake_clock.run_until_idle()
    waiting = asyncio.create_task(slot.run("t2", _decision(fake_clock, "t2", 1.0, log)))
    await fake_clock.run_until_idle()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await fake_clock.run_for(1.0)
    assert (await first).status == "ok"
    assert "start:t2" not in log
