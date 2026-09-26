"""``NullSink``: the ``AvatarSink`` used when ``avatar.sink = "none"`` (and in ``run --safe``).

It accepts everything and renders nothing. ``connected`` is true while ``run()`` is active so
callers never wait on it; ``health()`` reports DISABLED and ``trigger`` always fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from aivtube.contracts.types import Health, HealthState

__all__ = ["NullSink"]


class NullSink:
    def __init__(self, name: str = "null", *, component: str | None = None) -> None:
        self.name = name
        self.component = component or f"avatar:{name}"
        self.frames = 0
        self.last_values: dict[str, float] = {}
        self.emotion = "neutral"
        self._running = False
        self._stop = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._running

    async def run(self) -> None:
        self._running = True
        self._stop.clear()
        try:
            await self._stop.wait()
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()

    def set_params(self, values: Mapping[str, float]) -> None:
        self.frames += 1
        self.last_values.update(values)

    async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None:
        self.emotion = emotion

    async def trigger(self, hotkey: str) -> bool:
        return False

    async def move(
        self,
        *,
        rotation: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        size: float = 0.0,
        seconds: float = 0.25,
        relative: bool = True,
    ) -> None:
        return None

    def health(self) -> Health:
        return Health(self.component, HealthState.DISABLED, "avatar output disabled (sink = none)")
