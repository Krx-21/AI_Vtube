"""The idle timer (ARCHITECTURE.md §4.13).

The timer starts when playback ends: 25 s ± 5 s. Any speech or decision cancels it. Chat
activity doubles the interval (up to 120 s); the interval returns to 25 s once an idle turn
actually fires or chat has been quiet for ``max_s``. The brain suppresses it while the streamer
speaks (``on_activity`` at speech start, ``on_playback_end`` again at speech end). Idle turns
are written to history and the prompt lists the last 5 topics so she does not repeat herself.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import Priority, Rank, Stimulus, StimulusKind

__all__ = ["IDLE_TEXT", "IdleScheduler"]

IDLE_TEXT = "ไม่มีใครคุยด้วยอยู่ หาเรื่องใหม่มาชวนคุยสั้นๆ"


class IdleScheduler:
    """Computes the idle deadline; ``Arbiter.next`` returns ``None`` once it passes."""

    def __init__(
        self,
        clock: Clock,
        *,
        after_s: float = 25.0,
        jitter_s: float = 5.0,
        max_s: float = 120.0,
        rng: random.Random | None = None,
    ) -> None:
        if after_s <= 0 or max_s < after_s or jitter_s < 0:
            raise ValueError("need 0 < after_s <= max_s and jitter_s >= 0")
        self._clock = clock
        self.after_s = after_s
        self.jitter_s = min(jitter_s, after_s * 0.9)
        self.max_s = max_s
        self._rng = rng or random.Random()
        self._interval = after_s
        self._armed_at: float | None = None
        self._jitter = 0.0
        self._last_chat: float | None = None

    @property
    def interval(self) -> float:
        """The current base interval (doubles with chat activity)."""
        return self._interval

    def on_playback_end(self, t: float) -> None:
        """Arm the timer (playback ended, or the streamer stopped talking)."""
        if self._last_chat is not None and t - self._last_chat >= self.max_s:
            self._interval = self.after_s
        self._armed_at = t
        self._jitter = self._rng.uniform(-self.jitter_s, self.jitter_s)

    def on_activity(self, t: float) -> None:
        """Speech or a decision: cancel the timer."""
        self._armed_at = None

    def on_chat_activity(self, t: float) -> None:
        """Chat is active: back off ×2 (up to ``max_s``); an armed deadline moves out."""
        self._last_chat = t
        self._interval = min(self.max_s, self._interval * 2.0)

    def deadline(self) -> float | None:
        if self._armed_at is None:
            return None
        return self._armed_at + max(0.0, self._interval + self._jitter)

    def make_stimulus(self, character: str, recent_topics: Sequence[str] = ()) -> Stimulus:
        """The IDLE stimulus; firing resets the interval and disarms the timer."""
        now = self._clock.now()
        self._armed_at = None
        self._interval = self.after_s
        topics = tuple(t for t in recent_topics if t.strip())[-5:]
        return Stimulus(
            id=f"idle-{uuid.uuid4().hex[:8]}",
            kind=StimulusKind.IDLE,
            character=character,
            text=IDLE_TEXT,
            created=now,
            priority=Priority.LOW,
            rank=Rank.IDLE,
            ttl_s=None,
            source="idle",
            payload={"topics": list(topics)},
        )
