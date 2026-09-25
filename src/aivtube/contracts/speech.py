"""Core-side speech output: SpeechOutput, VoicePolicy, TTSConstraints (ARCHITECTURE.md §3.5).

The core never touches audio. It hands already-filtered ``Segment`` objects to a
``SpeechOutput`` (``BusSpeechOutput`` | ``ConsoleSpeechOutput`` | ``FakeSpeechOutput``).
Results come back as ``SegmentStarted``/``SegmentDone``/``UtteranceDone`` events; lip tracks
go straight to the ``AvatarDriver``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.types import Segment

__all__ = [
    "BargePolicy",
    "EchoMode",
    "MicMode",
    "SpeechOutput",
    "StopMode",
    "TTSConstraints",
    "VoicePolicy",
]

MicMode: TypeAlias = Literal["open", "ptt", "deafened"]
BargePolicy: TypeAlias = Literal["interrupt", "duck_only", "off"]
EchoMode: TypeAlias = Literal["auto", "aec", "energy_dtd", "half_duplex", "none"]
StopMode: TypeAlias = Literal["now", "after_segment"]


@dataclass(frozen=True, slots=True)
class VoicePolicy:
    mic_mode: MicMode = "open"
    ptt_active: bool = False
    barge_in: BargePolicy = "interrupt"
    echo_mode: EchoMode = "auto"
    listening: bool = True


@dataclass(frozen=True, slots=True)
class TTSConstraints:
    """Chunk sizes the reply pipeline must respect for the character's current TTS backend."""

    first_min_chars: int
    min_chars: int
    max_chars: int
    backend: str
    identity: str


@runtime_checkable
class SpeechOutput(Protocol):
    def ready(self) -> bool: ...

    def constraints(self, character: str) -> TTSConstraints: ...

    async def begin(
        self,
        utt_id: str,
        character: str,
        *,
        filler_after_s: float | None = None,
        gate_open: bool = True,
    ) -> None:
        """Open an utterance. ``gate_open=False`` holds playback until ``open_gate`` (M2)."""
        ...

    async def segment(self, seg: Segment) -> bool:
        """Queue a segment; ``False`` means backpressure. Never unfiltered text (I7)."""
        ...

    async def open_gate(self, utt_id: str) -> None:
        """Release a held utterance (M2 speculation / review mode)."""
        ...

    async def stop(
        self, utt_id: str | None, mode: StopMode, reason: str, fade_ms: int = 30
    ) -> None:
        """Stop ``utt_id`` (``None``: whatever is playing) now or after the current segment."""
        ...

    async def duck(self, gain: float, ramp_ms: int = 30) -> None: ...

    async def mute(self, on: bool) -> None: ...

    async def set_policy(self, policy: VoicePolicy) -> None: ...

    async def set_voice_rate(self, character: str, percent: int) -> None: ...

    async def play_canned(self, key: str, character: str) -> None:
        """Play a cached phrase: ``"filtered"``, fillers, the brain-freeze line."""
        ...
