"""``SystemClock`` and ``deadline()`` (ARCHITECTURE.md §2.5, invariant I2).

All timestamps are ``time.perf_counter()``: on Windows 3.12 ``time.monotonic()`` and
``loop.time()`` tick at ~15.6 ms, which is too coarse for audio markers and latency traces.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aivtube.contracts.infra import Clock

__all__ = ["DeadlineExceeded", "SystemClock", "deadline"]


class SystemClock:
    """The real clock: ``now`` = perf_counter, ``wall`` = time.time, ``sleep`` = asyncio.sleep.

    ``asyncio.sleep`` follows the loop clock (coarse on Windows); use it for timers of 50 ms
    or more and ``PrecisionTicker`` for anything faster.
    """

    __slots__ = ()

    def now(self) -> float:
        return time.perf_counter()

    def wall(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class DeadlineExceeded(TimeoutError):
    """An awaited operation ran past its deadline. ``what`` names the operation."""

    def __init__(self, what: str, seconds: float) -> None:
        super().__init__(f"{what}: deadline of {seconds:g} s exceeded")
        self.what = what
        self.seconds = seconds

    def __reduce__(self) -> tuple[type[DeadlineExceeded], tuple[str, float]]:
        return (DeadlineExceeded, (self.what, self.seconds))


@asynccontextmanager
async def deadline(seconds: float, *, what: str, clock: Clock | None = None) -> AsyncIterator[None]:
    """Cancel the body after ``seconds`` and raise ``DeadlineExceeded(what)`` instead.

    With no ``clock`` (or a ``SystemClock``) this is ``asyncio.timeout``. With another Clock
    (e.g. a FakeClock) the deadline fires when ``clock.sleep(seconds)`` returns, so tests can
    drive it deterministically. A ``TimeoutError`` raised by the body itself passes through
    unchanged. ``math.inf`` means no deadline.
    """
    if math.isinf(seconds):
        yield
        return
    if clock is None or isinstance(clock, SystemClock):
        try:
            async with asyncio.timeout(seconds) as cm:
                yield
        except TimeoutError as exc:
            if cm.expired():
                raise DeadlineExceeded(what, seconds) from exc
            raise
        return

    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("deadline() must be used inside a task")
    fired = False

    async def _watch() -> None:
        nonlocal fired
        await clock.sleep(seconds)
        fired = True
        task.cancel()

    watcher = asyncio.get_running_loop().create_task(_watch(), name=f"deadline:{what}")
    try:
        yield
    except asyncio.CancelledError:
        if fired and task.uncancel() == 0:
            raise DeadlineExceeded(what, seconds) from None
        raise
    else:
        if fired:  # the body swallowed our cancellation; do not leak it outwards
            task.uncancel()
    finally:
        watcher.cancel()
