"""AsyncEventBus: overflow policies, publish never raises, threadsafe delivery, ts stamping."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from aivtube.contracts.events import (
    Alert,
    Event,
    LatencyMark,
    UserSpeechEnded,
    UserSpeechStarted,
)
from aivtube.contracts.infra import EventBus, Overflow, Subscription
from aivtube.infra import AsyncEventBus, FlightRecorder


class FixedClock:
    def __init__(self, t: float = 42.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def wall(self) -> float:
        return 1.7e9

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)


def mark(i: int) -> LatencyMark:
    return LatencyMark(stage=f"s{i}", ts=float(i + 1))


def test_implements_the_protocols() -> None:
    bus = AsyncEventBus(FixedClock())
    assert isinstance(bus, EventBus)
    assert isinstance(bus.subscribe(name="x"), Subscription)


async def test_drop_oldest_keeps_the_newest_and_counts_drops() -> None:
    bus = AsyncEventBus(FixedClock())
    sub = bus.subscribe(LatencyMark, name="slow", maxsize=3)
    for i in range(5):
        bus.publish(mark(i))
    assert sub.dropped == 2
    assert [e.stage for e in sub.drain()] == ["s2", "s3", "s4"]  # type: ignore[attr-defined]


async def test_drop_newest_keeps_the_oldest() -> None:
    bus = AsyncEventBus(FixedClock())
    sub = bus.subscribe(LatencyMark, name="slow", maxsize=3, overflow=Overflow.DROP_NEWEST)
    for i in range(5):
        bus.publish(mark(i))
    assert sub.dropped == 2
    assert [e.stage for e in sub.drain()] == ["s0", "s1", "s2"]  # type: ignore[attr-defined]


async def test_a_full_subscriber_does_not_affect_others() -> None:
    bus = AsyncEventBus(FixedClock())
    slow = bus.subscribe(name="slow", maxsize=1)
    fast = bus.subscribe(name="fast", maxsize=100)
    for i in range(10):
        bus.publish(mark(i))
    assert slow.dropped == 9 and fast.dropped == 0 and fast.pending == 10
    stats = {s["name"]: s for s in bus.subscribers()}
    assert stats["slow"]["dropped"] == 9 and stats["fast"]["pending"] == 10


async def test_publish_never_raises() -> None:
    class BrokenFlight(FlightRecorder):
        def record(self, ev: Event) -> None:
            raise RuntimeError("disk on fire")

    bus = AsyncEventBus(FixedClock(), flight=BrokenFlight())
    sub = bus.subscribe(name="all")
    bus.publish("not an event")  # type: ignore[arg-type]
    bus.publish(None)  # type: ignore[arg-type]
    bus.publish(mark(1))  # still delivered although the flight recorder fails
    bus.publish_threadsafe(42)  # type: ignore[arg-type]
    assert bus.errors >= 4
    assert [type(e) for e in sub.drain()] == [LatencyMark]


async def test_publish_stamps_ts_when_zero() -> None:
    clock = FixedClock(123.5)
    bus = AsyncEventBus(clock)
    sub = bus.subscribe(name="all")
    bus.publish(LatencyMark(stage="a"))
    bus.publish(LatencyMark(stage="b", ts=7.0))
    first, second = sub.drain()
    assert first.ts == 123.5 and second.ts == 7.0


async def test_type_filtering_and_subclasses() -> None:
    bus = AsyncEventBus(FixedClock())
    speech = bus.subscribe(UserSpeechStarted, UserSpeechEnded, name="speech")
    everything = bus.subscribe(Event, name="everything")
    bus.publish(UserSpeechStarted(barge=False))
    bus.publish(Alert(level="info", message="hi"))
    bus.publish(UserSpeechEnded(audio_s=1.0))
    assert [type(e) for e in speech.drain()] == [UserSpeechStarted, UserSpeechEnded]
    assert everything.pending == 3
    with pytest.raises(TypeError):
        bus.subscribe(int, name="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bus.subscribe(name="bad", maxsize=0)


async def test_async_iteration_waits_and_close_ends_it() -> None:
    bus = AsyncEventBus(FixedClock())
    sub = bus.subscribe(name="reader")
    got: list[Event] = []

    async def reader() -> None:
        async for ev in sub:
            got.append(ev)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    bus.publish(mark(1))
    bus.publish(mark(2))
    await asyncio.sleep(0.01)
    assert len(got) == 2
    sub.close()
    await asyncio.wait_for(task, 1.0)
    bus.publish(mark(3))  # closed subscriptions receive nothing
    assert len(got) == 2 and sub.closed


async def test_publish_threadsafe_delivers_from_another_thread() -> None:
    bus = AsyncEventBus(FixedClock())
    sub = bus.subscribe(name="reader", maxsize=1000)
    loop_thread = threading.get_ident()
    n = 200

    def producer() -> None:
        for i in range(n):
            bus.publish_threadsafe(mark(i))
        bus.publish(mark(n))  # plain publish from a foreign thread is routed safely too

    thread = threading.Thread(target=producer)
    thread.start()
    received: list[Any] = []

    async def collect() -> None:
        async for ev in sub:
            received.append(ev)
            if len(received) == n + 1:
                return

    await asyncio.wait_for(collect(), 5.0)
    thread.join()
    assert [e.stage for e in received] == [f"s{i}" for i in range(n + 1)]
    assert threading.get_ident() == loop_thread
    assert bus.lost == 0


def test_without_any_event_loop_publish_delivers_synchronously() -> None:
    bus = AsyncEventBus(FixedClock())
    sub = bus.subscribe(name="sync")
    bus.publish(mark(1))
    bus.publish_threadsafe(mark(2))
    assert sub.pending == 2


def test_threadsafe_publish_after_the_loop_closed_is_counted_not_raised() -> None:
    bus = AsyncEventBus(FixedClock())

    async def bind() -> None:
        bus.subscribe(name="x")

    asyncio.run(bind())
    bus.publish_threadsafe(mark(1))
    assert bus.lost == 1


async def test_flight_recorder_sees_every_published_event() -> None:
    flight = FlightRecorder(maxlen=10)
    bus = AsyncEventBus(FixedClock(), flight=flight)
    for i in range(3):
        bus.publish(mark(i))
    assert [e.stage for e in flight.events()] == ["s0", "s1", "s2"]  # type: ignore[attr-defined]


async def test_bus_close_ends_all_subscriptions() -> None:
    bus = AsyncEventBus(FixedClock())
    a, b = bus.subscribe(name="a"), bus.subscribe(name="b")
    bus.close()
    assert a.closed and b.closed and bus.subscribers() == []
