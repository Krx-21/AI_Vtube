"""Avatar protocols: AvatarSink, AvatarDriver and LipTrack (ARCHITECTURE.md §3.7).

``LipTrack`` is not an event: the voice worker's lip tracks go straight to the driver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.types import Health

__all__ = ["AvatarDriver", "AvatarSink", "AvatarState", "LipTrack"]

AvatarState: TypeAlias = Literal["idle", "listening", "thinking", "speaking", "paused"]


@dataclass(frozen=True, slots=True)
class LipTrack:
    """Mouth values for one segment, sampled at ``fps`` from perf_counter time ``t0``."""

    utt_id: str
    seq: int
    t0: float
    fps: int
    mouth: tuple[float, ...]  # MouthOpen per frame, 0..1
    form: tuple[float, ...]  # MouthForm per frame, 0..1 (0.5 neutral)
    final: bool = False


@runtime_checkable
class AvatarSink(Protocol):
    """``VTSSink`` | ``BrowserSink`` (M6) | ``NullSink`` | ``FakeSink``."""

    name: str

    @property
    def connected(self) -> bool: ...

    async def run(self) -> None:
        """Connect, authenticate and reconnect forever."""
        ...

    def set_params(self, values: Mapping[str, float]) -> None:
        """Fire-and-forget parameter injection; at most 8 requests in flight."""
        ...

    async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None: ...

    async def trigger(self, hotkey: str) -> bool: ...

    async def move(
        self,
        *,
        rotation: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        size: float = 0.0,
        seconds: float = 0.25,
        relative: bool = True,
    ) -> None: ...

    def health(self) -> Health: ...


@runtime_checkable
class AvatarDriver(Protocol):
    def on_lip_track(self, track: LipTrack) -> None: ...

    def on_cut(self, utt_id: str, t: float) -> None:
        """Close the mouth at perf_counter time ``t`` (speech was cut)."""
        ...

    def set_state(self, state: AvatarState) -> None: ...

    def set_emotion(self, emotion: str | None) -> None: ...

    async def run(self) -> None: ...
