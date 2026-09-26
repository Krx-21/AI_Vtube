"""Avatar fakes: ``FakeAvatarSink`` and ``FakeAvatarDriver`` (§3.7)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from aivtube.contracts.avatar import AvatarState, LipTrack
from aivtube.contracts.types import Health, HealthState

__all__ = ["FakeAvatarDriver", "FakeAvatarSink"]


class FakeAvatarSink:
    """``AvatarSink`` that records everything. ``run()`` "connects" and waits for ``stop()``;
    ``disconnect()`` simulates the renderer closing. ``trigger`` succeeds for known hotkeys."""

    def __init__(
        self,
        name: str = "fake",
        *,
        hotkeys: Iterable[str] = ("happy", "wave", "spin"),
        max_in_flight: int = 8,
    ) -> None:
        self.name = name
        self.hotkeys = {h.casefold() for h in hotkeys}
        self.max_in_flight = max_in_flight
        self.params: list[dict[str, float]] = []
        self.current: dict[str, float] = {}
        self.emotions: list[tuple[str, float]] = []
        self.triggered: list[str] = []
        self.moves: list[dict[str, Any]] = []
        self.runs = 0
        self._connected = False
        self._stop = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._connected

    async def run(self) -> None:
        self.runs += 1
        self._stop.clear()
        self._connected = True
        try:
            await self._stop.wait()
        finally:
            self._connected = False

    def stop(self) -> None:
        self._stop.set()

    def disconnect(self) -> None:
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def set_params(self, values: Mapping[str, float]) -> None:
        if not self._connected:
            return  # fire-and-forget: silently dropped while disconnected
        snapshot = {k: float(v) for k, v in values.items()}
        self.params.append(snapshot)
        self.current.update(snapshot)

    async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None:
        self.emotions.append((emotion, fade_s))

    async def trigger(self, hotkey: str) -> bool:
        ok = self._connected and hotkey.casefold() in self.hotkeys
        if ok:
            self.triggered.append(hotkey)
        return ok

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
        self.moves.append(
            dict(rotation=rotation, x=x, y=y, size=size, seconds=seconds, relative=relative)
        )

    def health(self) -> Health:
        state = HealthState.OK if self._connected else HealthState.DOWN
        return Health(f"avatar:{self.name}", state)


class FakeAvatarDriver:
    """``AvatarDriver`` that records lip tracks, cuts, states and emotions."""

    def __init__(self) -> None:
        self.tracks: list[LipTrack] = []
        self.cuts: list[tuple[str, float]] = []
        self.states: list[AvatarState] = []
        self.emotions: list[str | None] = []
        self._stop = asyncio.Event()

    @property
    def state(self) -> AvatarState | None:
        return self.states[-1] if self.states else None

    def on_lip_track(self, track: LipTrack) -> None:
        self.tracks.append(track)

    def on_cut(self, utt_id: str, t: float) -> None:
        self.cuts.append((utt_id, t))

    def set_state(self, state: AvatarState) -> None:
        self.states.append(state)

    def set_emotion(self, emotion: str | None) -> None:
        self.emotions.append(emotion)

    async def run(self) -> None:
        await self._stop.wait()

    def stop(self) -> None:
        self._stop.set()
