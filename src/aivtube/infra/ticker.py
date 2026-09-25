"""``PrecisionTicker``: periodic callbacks faster than 20 ms (§2.5), e.g. the 60 Hz VTS driver.

A daemon thread sleeps with ``time.sleep`` (a high-resolution waitable timer on Windows since
Python 3.11) against an absolute perf_counter schedule, then posts ``callback(t)`` to the
event loop with ``call_soon_threadsafe``. If the loop has not run the previous tick yet, the
new one is coalesced (skipped and counted) so a lagging loop never builds a backlog.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from collections.abc import Callable

__all__ = ["PrecisionTicker"]

log = logging.getLogger("aivtube.ticker")


class PrecisionTicker:
    """Calls ``callback(t)`` on ``loop`` at ``hz``; ``t`` is the perf_counter tick time."""

    def __init__(
        self,
        hz: float,
        callback: Callable[[float], None],
        loop: asyncio.AbstractEventLoop,
        *,
        name: str = "ticker",
        window: int = 600,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self._period = 1.0 / hz
        self._callback = callback
        self._loop = loop
        self._name = name
        self._clock = clock
        self._sleep = sleep
        self._jitter: collections.deque[float] = collections.deque(maxlen=window)
        self._delivery: collections.deque[float] = collections.deque(maxlen=window)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending = False
        self.ticks = 0
        self.coalesced = 0
        self.late = 0  # ticks more than one period behind schedule (schedule was reset)
        self.errors = 0

    @property
    def hz(self) -> float:
        return 1.0 / self._period

    def set_hz(self, hz: float) -> None:
        """Change the rate (e.g. drop to 30 Hz); applies from the next tick."""
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self._period = 1.0 / hz

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        next_t = self._clock() + self._period
        while not self._stop.is_set():
            delay = next_t - self._clock()
            if delay > 0.05:  # long periods: stay responsive to stop(), then sleep precisely
                if self._stop.wait(delay - 0.02):
                    break
                delay = next_t - self._clock()
            if delay > 0:
                self._sleep(delay)
            if self._stop.is_set():
                break
            now = self._clock()
            self._jitter.append(now - next_t)
            self.ticks += 1
            if self._pending:
                self.coalesced += 1
            else:
                self._pending = True
                try:
                    self._loop.call_soon_threadsafe(self._fire, now)
                except RuntimeError:  # the loop is closed
                    log.debug("ticker %r stopping: loop closed", self._name)
                    break
            next_t += self._period
            if now - next_t > self._period:  # fell far behind: resync instead of bursting
                self.late += 1
                next_t = now + self._period

    def _fire(self, t: float) -> None:
        self._pending = False
        self._delivery.append(self._clock() - t)
        try:
            self._callback(t)
        except Exception:
            self.errors += 1
            if self.errors <= 10 or self.errors % 1000 == 0:
                log.exception("ticker %r callback failed (error #%d)", self._name, self.errors)

    def jitter_stats(self) -> dict[str, float]:
        """Wake-up jitter of the thread and loop delivery delay over the recent window (ms)."""
        jit = sorted(abs(j) * 1000.0 for j in self._jitter)
        dly = sorted(d * 1000.0 for d in self._delivery)
        return {
            "hz": self.hz,
            "ticks": float(self.ticks),
            "coalesced": float(self.coalesced),
            "late": float(self.late),
            "p50_ms": _pct(jit, 0.50),
            "p95_ms": _pct(jit, 0.95),
            "max_ms": jit[-1] if jit else 0.0,
            "delivery_p50_ms": _pct(dly, 0.50),
            "delivery_p95_ms": _pct(dly, 0.95),
        }


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return sorted_values[index]
