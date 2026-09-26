"""Deterministic and real clocks for tests (ARCHITECTURE.md §2.5, §10)."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["FakeClock", "RealClock", "run_until_idle"]


class RealClock:
    """``Clock`` backed by ``time.perf_counter`` (a stand-in until ``infra.SystemClock`` exists)."""

    def now(self) -> float:
        return time.perf_counter()

    def wall(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


@dataclass(order=True)
class _Timer:
    deadline: float
    seq: int
    future: asyncio.Future[None] = field(compare=False)


async def run_until_idle(max_rounds: int = 10_000) -> None:
    """Yield to the running loop until no callbacks are ready (all tasks are blocked).

    Blocked on real I/O or threads does not count as busy, so this returns even if a socket
    read is pending. Raises ``RuntimeError`` if the loop never settles (a busy loop).
    """
    loop = asyncio.get_running_loop()
    ready = getattr(loop, "_ready", None)  # CPython BaseEventLoop internals; checked defensively
    idle_rounds = 0
    for _ in range(max_rounds):
        await asyncio.sleep(0)
        if ready is None:
            idle_rounds += 1
            if idle_rounds >= 50:
                return
            continue
        if not ready:
            idle_rounds += 1
            if idle_rounds >= 2:  # two consecutive empty rounds: nothing left to run
                return
        else:
            idle_rounds = 0
    raise RuntimeError("event loop did not become idle")


class FakeClock:
    """A ``Clock`` whose time only moves when a test moves it.

    ``sleep()`` parks the caller until the clock is advanced past its deadline. ``advance()`` is
    synchronous and wakes every due sleeper at once (they observe the final time); the async
    ``run_for()`` steps timer by timer and lets the loop settle between steps, so woken code
    sees the exact deadline and can schedule new timers inside the window.
    """

    def __init__(self, start: float = 1000.0, *, wall_start: float = 1_760_000_000.0) -> None:
        self._t = float(start)
        self._start = float(start)
        self._wall_start = float(wall_start)
        self._timers: list[_Timer] = []
        self._seq = itertools.count()
        self.sleeps: list[float] = []  # every requested sleep duration, for assertions

    # --- Clock protocol ---------------------------------------------------------------------
    def now(self) -> float:
        return self._t

    def wall(self) -> float:
        return self._wall_start + (self._t - self._start)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._timers, _Timer(self._t + seconds, next(self._seq), fut))
        await fut  # cancellation leaves a cancelled future in the heap; it is skipped later

    # --- test controls ----------------------------------------------------------------------
    @property
    def pending(self) -> int:
        """Number of sleepers still waiting."""
        return sum(1 for t in self._timers if not t.future.done())

    def next_deadline(self) -> float | None:
        self._drop_dead()
        return self._timers[0].deadline if self._timers else None

    def set(self, t: float) -> None:
        """Jump to absolute time ``t`` (never backwards), waking due sleepers."""
        if t < self._t:
            raise ValueError(f"FakeClock cannot go backwards ({t} < {self._t})")
        self.advance(t - self._t)

    def advance(self, dt: float) -> None:
        """Move time forward by ``dt`` and wake every sleeper whose deadline has passed."""
        if dt < 0:
            raise ValueError("dt must be >= 0")
        target = self._t + dt
        while self._timers and self._timers[0].deadline <= target:
            timer = heapq.heappop(self._timers)
            if not timer.future.done():
                timer.future.set_result(None)
        self._t = target

    async def run_until_idle(self) -> None:
        """Let every runnable task run until all of them are blocked."""
        await run_until_idle()

    async def run_for(self, dt: float) -> None:
        """Advance ``dt`` seconds one timer at a time, settling the loop after each wake-up."""
        if dt < 0:
            raise ValueError("dt must be >= 0")
        target = self._t + dt
        await run_until_idle()
        while True:
            self._drop_dead()
            if not self._timers or self._timers[0].deadline > target:
                break
            deadline = self._timers[0].deadline
            self._t = max(self._t, deadline)
            while self._timers and self._timers[0].deadline <= deadline:
                timer = heapq.heappop(self._timers)
                if not timer.future.done():
                    timer.future.set_result(None)
            await run_until_idle()
        self._t = target
        await run_until_idle()

    async def run_until(
        self, predicate: Callable[[], bool], *, within: float = 60.0, step: float = 0.01
    ) -> None:
        """Advance in ``step`` increments until ``predicate()`` is true (at most ``within``
        fake seconds, else ``TimeoutError``)."""
        end = self._t + within
        await run_until_idle()
        while not predicate():
            if self._t >= end:
                raise TimeoutError(f"condition not met within {within} fake seconds")
            await self.run_for(step)

    def _drop_dead(self) -> None:
        while self._timers and self._timers[0].future.done():
            heapq.heappop(self._timers)
