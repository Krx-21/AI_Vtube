"""``MetricsRegistry``: thread-safe counters and gauges, sampled at 1 Hz by the panel (§2.10)."""

from __future__ import annotations

import threading

__all__ = ["MetricsRegistry"]


class MetricsRegistry:
    """Counters (``inc``) and gauges (``set``) in one flat namespace of floats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + n

    def set(self, name: str, v: float) -> None:
        with self._lock:
            self._values[name] = float(v)

    def get(self, name: str, default: float = 0.0) -> float:
        with self._lock:
            return self._values.get(name, default)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def reset(self, prefix: str = "") -> None:
        """Drop every metric whose name starts with ``prefix`` (all of them by default)."""
        with self._lock:
            for name in [k for k in self._values if k.startswith(prefix)]:
                del self._values[name]
