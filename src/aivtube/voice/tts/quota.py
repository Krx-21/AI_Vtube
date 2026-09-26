"""Token bucket for quota-limited TTS backends (Azure F0: 20 transactions / 60 s, §11 R15)."""

from __future__ import annotations

import threading
from collections.abc import Callable

from aivtube.contracts.voice import QuotaSpec

__all__ = ["TokenBucket"]


class TokenBucket:
    """``capacity`` tokens refilled continuously at ``capacity / per_s`` tokens per second.

    ``try_take()`` never waits: an empty bucket means "skip this backend for this segment".
    """

    def __init__(self, capacity: int, per_s: float, clock: Callable[[], float]) -> None:
        if capacity <= 0 or per_s <= 0:
            raise ValueError("capacity and per_s must be positive")
        self.capacity = int(capacity)
        self.per_s = float(per_s)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = threading.Lock()

    @classmethod
    def from_quota(cls, quota: QuotaSpec, clock: Callable[[], float]) -> TokenBucket:
        return cls(quota.max_requests, quota.per_s, clock)

    def _refill(self) -> None:
        now = self._clock()
        if now > self._last:
            rate = self.capacity / self.per_s
            self._tokens = min(float(self.capacity), self._tokens + (now - self._last) * rate)
        self._last = max(self._last, now)

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def try_take(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def refund(self) -> None:
        """Give a token back (the request never reached the service)."""
        with self._lock:
            self._tokens = min(float(self.capacity), self._tokens + 1.0)
