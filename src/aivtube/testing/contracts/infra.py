"""Contract suites for ``Clock``, ``EventBus`` and ``TaskSupervisor`` (§3.3)."""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import AsyncIterator, Awaitable, Callable

from aivtube.contracts.events import Alert, Event, LatencyMark, StateChanged
from aivtube.contracts.infra import Clock, EventBus, Overflow, Subscription, TaskSupervisor
from aivtube.contracts.types import Health
from aivtube.testing.contracts._base import AsyncCase, _Cases, check, maybe_await

__all__ = ["clock_suite", "event_bus_suite", "task_supervisor_suite"]


def clock_suite(
    factory: Callable[[], Clock],
    *,
    advance: Callable[[Clock, float], Awaitable[None]] | None = None,
) -> list[AsyncCase]:
    """``advance(clock, dt)`` drives a fake clock; leave it ``None`` for real clocks."""
    cases = _Cases("clock")

    async def _sleep_and_drive(clock: Clock, seconds: float) -> None:
        task = asyncio.ensure_future(clock.sleep(seconds))
        if advance is not None:
            await asyncio.sleep(0)
            await advance(clock, seconds)
        await asyncio.wait_for(task, 5.0)

    @cases
    async def now_is_monotonic() -> None:
        clock = factory()
        a = clock.now()
        b = clock.now()
        check(isinstance(a, float) and b >= a, "now() must be a non-decreasing float")

    @cases
    async def wall_is_epoch_seconds() -> None:
        check(factory().wall() > 1.5e9, "wall() must be Unix epoch seconds")

    @cases
    async def sleep_zero_returns() -> None:
        await asyncio.wait_for(factory().sleep(0), 1.0)

    @cases
    async def sleep_advances_now() -> None:
        clock = factory()
        t0 = clock.now()
        await _sleep_and_drive(clock, 0.05)
        # asyncio timers may fire up to one clock tick early on Windows (15.6 ms)
        check(clock.now() - t0 >= 0.05 - 0.02, "sleep(0.05) returned too early")

    @cases
    async def shorter_sleep_wakes_first() -> None:
        clock = factory()
        order: list[str] = []

        async def nap(tag: str, s: float) -> None:
            await clock.sleep(s)
            order.append(tag)

        tasks = [
            asyncio.ensure_future(nap("long", 0.08)),
            asyncio.ensure_future(nap("short", 0.02)),
        ]
        if advance is not None:
            await asyncio.sleep(0)
            await advance(clock, 0.1)
        await asyncio.wait_for(asyncio.gather(*tasks), 5.0)
        check(order == ["short", "long"], f"wake order {order}")

    return cases.items


async def _next(it: AsyncIterator[Event], timeout: float = 1.0) -> Event:  # noqa: ASYNC109
    return await asyncio.wait_for(it.__anext__(), timeout)


async def _no_more(it: AsyncIterator[Event], timeout: float = 0.1) -> bool:  # noqa: ASYNC109
    try:
        await asyncio.wait_for(it.__anext__(), timeout)
    except (TimeoutError, StopAsyncIteration):
        return True
    return False


def event_bus_suite(factory: Callable[[], EventBus]) -> list[AsyncCase]:
    cases = _Cases("event_bus")

    def sub(
        bus: EventBus, *types: type[Event], **kw: object
    ) -> tuple[Subscription, AsyncIterator[Event]]:
        s = bus.subscribe(*types, name="contract", **kw)  # type: ignore[arg-type]
        return s, s.__aiter__()

    @cases
    async def delivers_matching_type() -> None:
        bus = factory()
        s, it = sub(bus, Alert)
        bus.publish(Alert(level="info", message="สวัสดี"))
        ev = await _next(it)
        check(isinstance(ev, Alert) and ev.message == "สวัสดี", f"got {ev!r}")
        s.close()

    @cases
    async def no_types_means_all_events() -> None:
        bus = factory()
        s, it = sub(bus)
        bus.publish(Alert(level="info", message="a"))
        bus.publish(LatencyMark(stage="x"))
        got = [await _next(it), await _next(it)]
        check([type(e) for e in got] == [Alert, LatencyMark], f"got {got!r}")
        s.close()

    @cases
    async def type_filter_excludes_others() -> None:
        bus = factory()
        s, it = sub(bus, LatencyMark)
        bus.publish(Alert(level="info", message="skip me"))
        bus.publish(LatencyMark(stage="keep"))
        ev = await _next(it)
        check(isinstance(ev, LatencyMark), f"filter leaked {ev!r}")
        s.close()

    @cases
    async def zero_ts_is_stamped_and_nonzero_kept() -> None:
        bus = factory()
        s, it = sub(bus, LatencyMark)
        bus.publish(LatencyMark(stage="a"))
        bus.publish(LatencyMark(ts=123.5, stage="b"))
        a, b = await _next(it), await _next(it)
        check(a.ts > 0.0, "ts=0 must be stamped with the clock")
        check(b.ts == 123.5, "a non-zero ts must be kept")
        s.close()

    @cases
    async def drop_oldest_keeps_newest() -> None:
        bus = factory()
        s, it = sub(bus, LatencyMark, maxsize=3, overflow=Overflow.DROP_OLDEST)
        for i in range(5):
            bus.publish(LatencyMark(stage=str(i)))
        got = [await _next(it) for _ in range(3)]
        check([e.stage for e in got] == ["2", "3", "4"], f"got {[e.stage for e in got]}")  # type: ignore[attr-defined]
        check(s.dropped == 2, f"dropped={s.dropped}")
        s.close()

    @cases
    async def drop_newest_keeps_oldest() -> None:
        bus = factory()
        s, it = sub(bus, LatencyMark, maxsize=3, overflow=Overflow.DROP_NEWEST)
        for i in range(5):
            bus.publish(LatencyMark(stage=str(i)))
        got = [await _next(it) for _ in range(3)]
        check([e.stage for e in got] == ["0", "1", "2"], f"got {[e.stage for e in got]}")  # type: ignore[attr-defined]
        check(s.dropped == 2, f"dropped={s.dropped}")
        s.close()

    @cases
    async def every_subscriber_gets_a_copy() -> None:
        bus = factory()
        s1, it1 = sub(bus, StateChanged)
        s2, it2 = sub(bus, StateChanged)
        bus.publish(StateChanged(old="idle", new="deciding"))
        check(isinstance(await _next(it1), StateChanged), "first subscriber missed it")
        check(isinstance(await _next(it2), StateChanged), "second subscriber missed it")
        s1.close()
        s2.close()

    @cases
    async def publish_without_subscribers_is_fine() -> None:
        bus = factory()
        bus.publish(Alert(level="warn", message="nobody listens"))
        bus.publish(Event())

    @cases
    async def publish_threadsafe_delivers() -> None:
        bus = factory()
        s, it = sub(bus, Alert)
        t = threading.Thread(
            target=bus.publish_threadsafe, args=(Alert(level="info", message="t"),)
        )
        t.start()
        t.join()
        ev = await _next(it, 2.0)
        check(isinstance(ev, Alert) and ev.message == "t", f"got {ev!r}")
        s.close()

    @cases
    async def closed_subscription_gets_nothing_new() -> None:
        bus = factory()
        s, it = sub(bus, Alert)
        s.close()
        bus.publish(Alert(level="info", message="late"))
        check(await _no_more(it), "a closed subscription still received events")

    return cases.items


def task_supervisor_suite(factory: Callable[[], TaskSupervisor]) -> list[AsyncCase]:
    """Backoffs in these cases are tiny (10-20 ms) so they run in real time."""
    cases = _Cases("task_supervisor")
    fast = {"backoff": (0.01, 0.02), "breaker": (50, 60.0)}

    @cases
    async def track_returns_task_with_result() -> None:
        sup = await maybe_await(factory())

        async def work() -> int:
            await asyncio.sleep(0)
            return 7

        task = sup.track(work(), name="work")
        check(await asyncio.wait_for(task, 2.0) == 7, "tracked task result lost")
        await sup.aclose(1.0)

    @cases
    async def track_keeps_a_strong_reference() -> None:
        sup = await maybe_await(factory())
        gate = asyncio.Event()
        done = asyncio.Event()

        async def work() -> None:
            await gate.wait()
            done.set()

        sup.track(work(), name="orphan")  # the caller drops its reference on purpose
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
        gate.set()
        await asyncio.wait_for(done.wait(), 2.0)
        await sup.aclose(1.0)

    @cases
    async def tracked_exception_does_not_propagate() -> None:
        sup = await maybe_await(factory())

        async def boom() -> None:
            raise RuntimeError("tracked failure")

        sup.track(boom(), name="boom")
        for _ in range(5):
            await asyncio.sleep(0)
        await sup.aclose(1.0)

    @cases
    async def on_error_restarts() -> None:
        sup = await maybe_await(factory())
        calls: list[int] = []
        ok = asyncio.Event()

        async def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("first run fails")
            ok.set()
            await asyncio.Event().wait()

        sup.spawn("flaky", flaky, restart="on_error", **fast)  # type: ignore[arg-type]
        await asyncio.wait_for(ok.wait(), 3.0)
        check(len(calls) == 2, f"expected one restart, saw {len(calls)} runs")
        names = [h.component for h in sup.status()]
        check(any("flaky" in n for n in names), f"status() does not list the task: {names}")
        check(all(isinstance(h, Health) for h in sup.status()), "status() must return Health")
        await sup.aclose(1.0)

    @cases
    async def never_does_not_restart() -> None:
        sup = await maybe_await(factory())
        calls: list[int] = []

        async def once() -> None:
            calls.append(1)
            raise RuntimeError("no restart please")

        sup.spawn("once", once, restart="never", **fast)  # type: ignore[arg-type]
        await asyncio.sleep(0.15)
        check(len(calls) == 1, f"restart='never' ran {len(calls)} times")
        await sup.aclose(1.0)

    @cases
    async def aclose_cancels_running_tasks() -> None:
        sup = await maybe_await(factory())
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def forever() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        sup.spawn("forever", forever, restart="always", **fast)  # type: ignore[arg-type]
        await asyncio.wait_for(started.wait(), 2.0)
        await asyncio.wait_for(sup.aclose(1.0), 3.0)
        check(cancelled.is_set(), "aclose() did not cancel the running task")

    return cases.items
