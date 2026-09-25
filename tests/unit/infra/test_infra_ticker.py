"""PrecisionTicker: posts to the loop thread, coalesces under lag, stays precise."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from aivtube.infra import PrecisionTicker


async def test_callbacks_run_on_the_loop_thread() -> None:
    loop = asyncio.get_running_loop()
    threads: list[int] = []
    times: list[float] = []

    def cb(t: float) -> None:
        threads.append(threading.get_ident())
        times.append(t)

    ticker = PrecisionTicker(100.0, cb, loop, name="test-ticker")
    ticker.start()
    ticker.start()  # idempotent
    await asyncio.sleep(0.2)
    ticker.stop()
    assert not ticker.running
    count = len(times)
    assert count >= 5
    assert set(threads) == {threading.get_ident()}
    assert times == sorted(times)
    await asyncio.sleep(0.05)
    assert len(times) == count  # stopped means stopped
    stats = ticker.jitter_stats()
    assert stats["ticks"] >= count and stats["hz"] == 100.0


async def test_a_blocked_loop_coalesces_ticks() -> None:
    loop = asyncio.get_running_loop()
    calls: list[float] = []
    ticker = PrecisionTicker(200.0, calls.append, loop)
    ticker.start()
    await asyncio.sleep(0.02)
    time.sleep(0.1)  # block the loop: ~20 ticks elapse, at most one may be queued
    before = len(calls)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    after_block = len(calls) - before
    ticker.stop()
    assert after_block <= 2
    assert ticker.coalesced >= 10


async def test_callback_errors_are_contained() -> None:
    loop = asyncio.get_running_loop()

    def bad(t: float) -> None:
        raise RuntimeError("driver bug")

    ticker = PrecisionTicker(100.0, bad, loop)
    ticker.start()
    await asyncio.sleep(0.1)
    ticker.stop()
    assert ticker.errors >= 3


async def test_set_hz_and_validation() -> None:
    loop = asyncio.get_running_loop()
    ticker = PrecisionTicker(60.0, lambda t: None, loop)
    ticker.set_hz(30.0)
    assert ticker.hz == pytest.approx(30.0)
    with pytest.raises(ValueError):
        ticker.set_hz(0)
    with pytest.raises(ValueError):
        PrecisionTicker(-1, lambda t: None, loop)


async def test_slow_rate_stops_promptly() -> None:
    loop = asyncio.get_running_loop()
    ticker = PrecisionTicker(1.0, lambda t: None, loop)
    ticker.start()
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    ticker.stop()
    assert time.perf_counter() - started < 0.5


@pytest.mark.timing
async def test_60hz_p95_jitter_under_5ms() -> None:
    loop = asyncio.get_running_loop()
    ticker = PrecisionTicker(60.0, lambda t: None, loop)
    ticker.start()
    await asyncio.sleep(1.5)
    ticker.stop()
    stats = ticker.jitter_stats()
    assert stats["ticks"] >= 80
    assert stats["p95_ms"] < 5.0, stats
