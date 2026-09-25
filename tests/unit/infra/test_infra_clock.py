"""SystemClock and deadline()/DeadlineExceeded."""

from __future__ import annotations

import asyncio
import math
import pickle
import time

import pytest

from aivtube.contracts.infra import Clock
from aivtube.infra import DeadlineExceeded, SystemClock, deadline


class InstantClock:
    """A fake-ish clock whose sleeps finish on the next loop iteration."""

    def now(self) -> float:
        return 0.0

    def wall(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)


class NeverClock(InstantClock):
    async def sleep(self, seconds: float) -> None:
        await asyncio.Event().wait()


async def test_system_clock() -> None:
    clock = SystemClock()
    assert isinstance(clock, Clock)
    a = clock.now()
    await clock.sleep(0.01)
    b = clock.now()
    assert b > a and abs(b - time.perf_counter()) < 1.0
    assert abs(clock.wall() - time.time()) < 1.0
    await clock.sleep(-1)  # negative sleeps are clamped


async def test_deadline_raises_with_its_label() -> None:
    with pytest.raises(DeadlineExceeded) as info:
        async with deadline(0.02, what="llm connect"):
            await asyncio.sleep(5)
    assert info.value.what == "llm connect"
    assert info.value.seconds == 0.02
    assert isinstance(info.value, TimeoutError)
    assert "llm connect" in str(info.value)
    again = pickle.loads(pickle.dumps(info.value))
    assert again.what == "llm connect"


async def test_deadline_passes_when_in_time() -> None:
    async with deadline(1.0, what="fast"):
        await asyncio.sleep(0)
    async with deadline(math.inf, what="unbounded"):
        await asyncio.sleep(0)


async def test_inner_timeout_error_is_not_relabelled() -> None:
    with pytest.raises(TimeoutError) as info:
        async with deadline(5.0, what="outer"):
            raise TimeoutError("inner")
    assert not isinstance(info.value, DeadlineExceeded)


async def test_deadline_driven_by_an_injected_clock() -> None:
    with pytest.raises(DeadlineExceeded) as info:
        async with deadline(30.0, what="stt decode", clock=InstantClock()):
            await asyncio.Event().wait()
    assert info.value.what == "stt decode"
    assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]


async def test_injected_clock_deadline_does_not_fire_early() -> None:
    async with deadline(30.0, what="x", clock=NeverClock()):
        for _ in range(5):
            await asyncio.sleep(0)


async def test_outer_cancellation_is_not_converted() -> None:
    async def body() -> None:
        async with deadline(30.0, what="x", clock=NeverClock()):
            await asyncio.Event().wait()

    task = asyncio.create_task(body())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_system_clock_takes_the_fast_path() -> None:
    with pytest.raises(DeadlineExceeded):
        async with deadline(0.01, what="fast path", clock=SystemClock()):
            await asyncio.sleep(1)
