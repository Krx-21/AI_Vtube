"""``LoopLagMonitor`` (§2.4): warn at 250 ms of event-loop lag, dump all stacks at 2 s.

A coroutine measures how late ``sleep(interval)`` wakes up. A loop blocked for 2 s cannot
notice its own stall until it is over, so a watchdog thread checks the coroutine's heartbeat
and calls ``faulthandler.dump_traceback(all_threads=True)`` while the loop is still stuck;
the dump then shows the code that is blocking it.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import IO, Any

from aivtube.contracts.infra import Clock
from aivtube.infra.clock import SystemClock
from aivtube.infra.metrics import MetricsRegistry

__all__ = ["LoopLagMonitor"]

log = logging.getLogger("aivtube.lag")


class LoopLagMonitor:
    """Run ``await monitor.run()`` as a supervised task on the loop being watched."""

    def __init__(
        self,
        interval_s: float = 0.1,
        warn_s: float = 0.25,
        dump_s: float = 2.0,
        *,
        clock: Clock | None = None,
        metrics: MetricsRegistry | None = None,
        on_lag: Callable[[float], None] | None = None,
        dump_file: IO[str] | None = None,
        watchdog: bool = True,
    ) -> None:
        if not 0 < interval_s < warn_s < dump_s:
            raise ValueError("need 0 < interval_s < warn_s < dump_s")
        self.interval_s = interval_s
        self.warn_s = warn_s
        self.dump_s = dump_s
        self._clock = clock or SystemClock()
        self._metrics = metrics
        self._on_lag = on_lag
        self._dump_file = dump_file
        self._watchdog = watchdog
        self._beat = time.perf_counter()
        self._stop = threading.Event()
        self.last_lag_s = 0.0
        self.max_lag_s = 0.0
        self.warnings = 0
        self.dumps = 0

    async def run(self) -> None:
        thread: threading.Thread | None = None
        self._beat = time.perf_counter()
        self._stop.clear()
        if self._watchdog:
            thread = threading.Thread(target=self._watch, name="loop-lag-watchdog", daemon=True)
            thread.start()
        try:
            while True:
                start = self._clock.now()
                await self._clock.sleep(self.interval_s)
                self._beat = time.perf_counter()
                self._record(max(0.0, self._clock.now() - start - self.interval_s))
        finally:
            self._stop.set()
            if thread is not None:
                thread.join(timeout=1.0)

    def _record(self, lag: float) -> None:
        self.last_lag_s = lag
        self.max_lag_s = max(self.max_lag_s, lag)
        if self._metrics is not None:
            self._metrics.set("loop_lag_ms", lag * 1000.0)
            self._metrics.set("loop_lag_max_ms", self.max_lag_s * 1000.0)
        if lag >= self.warn_s:
            self.warnings += 1
            log.warning("event loop lagged %.0f ms (something blocked it)", lag * 1000.0)
            if self._on_lag is not None:
                try:
                    self._on_lag(lag)
                except Exception:
                    log.exception("on_lag callback failed")

    def _watch(self) -> None:
        dumped = False
        poll = min(self.interval_s, 0.1)
        while not self._stop.wait(poll):
            stalled = time.perf_counter() - self._beat
            if stalled >= self.dump_s and not dumped:
                dumped = True
                self.dumps += 1
                self._dump(stalled)
            elif stalled < self.dump_s:
                dumped = False

    def _dump(self, stalled: float) -> None:
        target: Any = self._dump_file
        if target is None:
            from aivtube.infra.logging import fault_file

            target = fault_file() or sys.stderr
        try:
            target.write(f"\n=== event loop stalled for {stalled:.1f} s; all thread stacks ===\n")
            target.flush()
            faulthandler.dump_traceback(file=target, all_threads=True)
        except (OSError, ValueError, AttributeError):  # e.g. a captured stream with no fileno
            log.error("event loop stalled for %.1f s (stack dump unavailable)", stalled)
            return
        log.error("event loop stalled for %.1f s; stacks dumped", stalled)
