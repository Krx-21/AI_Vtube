"""SupervisedTasks: backoff sequence, crash-loop breaker, critical tasks, track()."""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest

from aivtube.contracts.events import Alert, ComponentRestarted, HealthChanged
from aivtube.contracts.infra import TaskSupervisor
from aivtube.contracts.types import HealthState
from aivtube.infra import AsyncEventBus, SupervisedTasks, backoff_delay


class StepClock:
    """Sleeps return at once and advance time by the requested amount."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def wall(self) -> float:
        return 1.7e9 + self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)


class Harness:
    def __init__(self) -> None:
        self.clock = StepClock()
        self.bus = AsyncEventBus(self.clock)
        self.events = self.bus.subscribe(name="test", maxsize=10_000)
        self.critical: list[tuple[str, BaseException]] = []
        self.sup = SupervisedTasks(
            self.clock, self.bus, on_critical_failure=lambda n, e: self.critical.append((n, e))
        )

    def health(self, name: str) -> HealthState:
        return next(h.state for h in self.sup.status() if h.component == name)


async def settle(predicate: object, timeout: float = 2.0) -> None:
    async def wait() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout)


def test_backoff_delay_doubles_up_to_the_cap() -> None:
    assert [backoff_delay(n, (0.5, 30.0)) for n in range(1, 10)] == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        30.0,
        30.0,
        30.0,
    ]


async def test_implements_the_protocol() -> None:
    assert isinstance(Harness().sup, TaskSupervisor)


async def test_backoff_sequence_and_restart_events() -> None:
    h = Harness()
    runs = 0
    forever = asyncio.Event()

    async def flaky() -> None:
        nonlocal runs
        runs += 1
        if runs <= 8:
            raise RuntimeError(f"boom {runs}")
        await forever.wait()

    h.sup.spawn("flaky", flaky, breaker=(100, 1000.0))
    await settle(lambda: runs == 9)
    assert h.clock.sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert h.health("flaky") is HealthState.OK
    restarts = [e for e in h.events.drain() if isinstance(e, ComponentRestarted)]
    assert [e.count for e in restarts] == list(range(1, 9))
    await h.sup.aclose()
    assert h.health("flaky") is HealthState.DOWN


async def test_crash_loop_breaker_marks_failed() -> None:
    h = Harness()
    runs = 0

    async def crash() -> None:
        nonlocal runs
        runs += 1
        raise ValueError("always")

    h.sup.spawn("crashy", crash)  # default breaker (6, 120 s): 5 restarts, then FAILED
    await settle(lambda: h.health("crashy") is HealthState.FAILED)
    assert runs == 6
    assert h.clock.sleeps == [0.5, 1.0, 2.0, 4.0, 8.0]
    events = h.events.drain()
    failed = [
        e for e in events if isinstance(e, HealthChanged) and e.health.state is HealthState.FAILED
    ]
    assert failed and "crash loop" in failed[-1].health.detail
    assert any(isinstance(e, Alert) and e.level == "error" for e in events)
    assert not h.critical
    await h.sup.aclose()


async def test_failures_outside_the_window_do_not_trip_the_breaker() -> None:
    h = Harness()
    runs = 0

    async def slow_crash() -> None:
        nonlocal runs
        runs += 1
        h.clock.t += 70.0  # each run lasts 70 s, so only two failures fit in 120 s
        if runs < 10:
            raise ValueError("again")
        await asyncio.Event().wait()

    h.sup.spawn("slow", slow_crash, breaker=(3, 120.0))
    await settle(lambda: runs == 10)
    assert h.health("slow") is HealthState.OK
    assert set(h.clock.sleeps) == {0.5}  # a run of >= backoff max resets the backoff
    await h.sup.aclose()


async def test_critical_task_failure_calls_the_callback() -> None:
    h = Harness()

    async def brain() -> None:
        raise RuntimeError("brain died")

    h.sup.spawn("brain", brain, critical=True)
    await settle(lambda: bool(h.critical))
    name, exc = h.critical[0]
    assert name == "brain" and str(exc) == "brain died"
    assert h.health("brain") is HealthState.FAILED
    assert h.clock.sleeps == []  # critical tasks are not restarted


async def test_critical_task_that_returns_is_a_failure() -> None:
    h = Harness()

    async def returns() -> None:
        return None

    h.sup.spawn("ipc", returns, critical=True)
    await settle(lambda: bool(h.critical))
    assert "exited" in str(h.critical[0][1])


async def test_restart_policies() -> None:
    h = Harness()
    done = 0

    async def once() -> None:
        nonlocal done
        done += 1

    async def fail() -> None:
        raise OSError("nope")

    h.sup.spawn("once", once)  # on_error: a clean exit is final
    h.sup.spawn("never", fail, restart="never")
    await settle(
        lambda: h.health("once") is HealthState.DOWN and h.health("never") is HealthState.FAILED
    )
    assert done == 1

    runs = 0

    async def always() -> None:
        nonlocal runs
        runs += 1
        if runs >= 3:
            await asyncio.Event().wait()

    h.sup.spawn("always", always, restart="always")
    await settle(lambda: runs == 3)
    await h.sup.aclose()


async def test_manual_restart_clears_failed_state() -> None:
    h = Harness()
    runs = 0

    async def task() -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise RuntimeError("first run fails")
        await asyncio.Event().wait()

    h.sup.spawn("vts", task, restart="never")
    await settle(lambda: h.health("vts") is HealthState.FAILED)
    await h.sup.restart("vts")
    await settle(lambda: runs == 2)
    assert h.health("vts") is HealthState.OK
    with pytest.raises(KeyError):
        await h.sup.restart("missing")
    with pytest.raises(ValueError):
        h.sup.spawn("vts", task)  # still running
    await h.sup.aclose()


async def test_track_keeps_a_strong_reference() -> None:
    h = Harness()
    gate = asyncio.Event()
    finished: list[bool] = []

    async def job() -> None:
        await gate.wait()
        finished.append(True)

    h.sup.track(job(), name="fire-and-forget")  # the returned task is deliberately dropped
    await asyncio.sleep(0)
    gc.collect()
    assert h.sup.tracked == 1
    gate.set()
    await settle(lambda: finished == [True])
    await asyncio.sleep(0)
    assert h.sup.tracked == 0


async def test_track_logs_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    h = Harness()

    async def bad() -> None:
        raise KeyError("lost")

    with caplog.at_level(logging.ERROR, logger="aivtube.tasks"):
        task = h.sup.track(bad(), name="preempt-stop")
        await asyncio.wait({task})
        await asyncio.sleep(0)
    assert h.sup.tracked_errors == 1
    assert any("preempt-stop" in r.getMessage() and r.exc_info for r in caplog.records)
    assert task.get_name() == "preempt-stop"


async def test_track_accepts_futures_and_ignores_cancellation() -> None:
    h = Harness()
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    task = h.sup.track(fut, name="future")
    fut.set_result(5)
    assert await task == 5
    slow = h.sup.track(asyncio.sleep(10), name="slow")
    slow.cancel()
    await asyncio.wait({slow})
    assert h.sup.tracked_errors == 0


async def test_aclose_cancels_everything_within_the_timeout() -> None:
    h = Harness()

    async def forever() -> None:
        await asyncio.Event().wait()

    h.sup.spawn("a", forever)
    tracked = h.sup.track(forever(), name="t")
    await asyncio.sleep(0)
    await h.sup.aclose(timeout=1.0)
    assert tracked.cancelled()
    assert h.health("a") is HealthState.DOWN
    with pytest.raises(RuntimeError):
        h.sup.spawn("late", forever)
