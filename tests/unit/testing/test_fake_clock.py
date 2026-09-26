"""FakeClock: deterministic sleeps, stepping and cancellation."""

from __future__ import annotations

import asyncio

import pytest

from aivtube.testing.fakes import FakeClock, run_until_idle


async def test_sleep_waits_for_advance() -> None:
    clock = FakeClock(start=10.0)
    done: list[float] = []

    async def nap() -> None:
        await clock.sleep(1.0)
        done.append(clock.now())

    task = asyncio.ensure_future(nap())
    await clock.run_until_idle()
    assert clock.pending == 1 and not done
    clock.advance(0.5)
    await clock.run_until_idle()
    assert not done
    clock.advance(0.5)
    await task
    assert done == [11.0]


async def test_run_for_steps_timer_by_timer() -> None:
    clock = FakeClock(start=0.0)
    seen: list[tuple[str, float]] = []

    async def chain() -> None:
        await clock.sleep(0.1)
        seen.append(("a", clock.now()))
        await clock.sleep(0.2)  # scheduled inside the window: must still fire
        seen.append(("b", clock.now()))

    task = asyncio.ensure_future(chain())
    await clock.run_for(1.0)
    assert seen == [("a", 0.1), ("b", pytest.approx(0.3))]
    assert clock.now() == 1.0
    await task


async def test_advance_wakes_everything_due_at_once() -> None:
    clock = FakeClock(start=0.0)
    seen: list[float] = []

    async def nap(s: float) -> None:
        await clock.sleep(s)
        seen.append(clock.now())

    tasks = [asyncio.ensure_future(nap(s)) for s in (0.3, 0.1, 0.2)]
    await run_until_idle()
    clock.advance(0.25)
    await run_until_idle()
    assert seen == [0.25, 0.25]  # woken together; they observe the final time
    clock.advance(1.0)
    await asyncio.gather(*tasks)


async def test_cancelled_sleep_is_skipped() -> None:
    clock = FakeClock()
    task = asyncio.ensure_future(clock.sleep(5.0))
    await run_until_idle()
    task.cancel()
    await run_until_idle()
    assert clock.pending == 0 and clock.next_deadline() is None
    clock.advance(10.0)  # must not raise on the cancelled future


async def test_zero_sleep_yields_without_timer() -> None:
    clock = FakeClock()
    await asyncio.wait_for(clock.sleep(0), 1.0)
    assert clock.pending == 0 and clock.sleeps == [0]


async def test_run_until_predicate_and_timeout() -> None:
    clock = FakeClock(start=0.0)
    flag: list[bool] = []

    async def later() -> None:
        await clock.sleep(0.5)
        flag.append(True)

    task = asyncio.ensure_future(later())
    await clock.run_until(lambda: bool(flag), within=2.0, step=0.1)
    assert 0.5 <= clock.now() <= 0.6
    await task
    with pytest.raises(TimeoutError):
        await clock.run_until(lambda: False, within=0.3, step=0.1)


def test_wall_tracks_now_and_no_going_back() -> None:
    clock = FakeClock(start=5.0, wall_start=1_700_000_000.0)
    clock.advance(2.5)
    assert clock.wall() == 1_700_000_002.5
    with pytest.raises(ValueError):
        clock.set(1.0)
    with pytest.raises(ValueError):
        clock.advance(-1.0)


async def test_shared_fixtures(fake_clock: FakeClock, fake_bus: object, bus: object) -> None:
    from aivtube.contracts.events import Alert
    from aivtube.contracts.infra import EventBus

    assert isinstance(bus, EventBus) and isinstance(fake_bus, EventBus)
    sub = bus.subscribe(Alert, name="fixture")
    bus.publish(Alert(level="info", message="hi"))
    ev = await sub.__aiter__().__anext__()
    assert isinstance(ev, Alert) and ev.ts > 0
    sub.close()
    fake_clock.advance(1.0)
    fake_bus.publish(Alert(level="info", message="x"))  # type: ignore[attr-defined]
    assert fake_bus.history[-1].ts == fake_clock.now()  # type: ignore[attr-defined]
