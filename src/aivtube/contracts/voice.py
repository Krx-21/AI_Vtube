"""Voice-worker protocols: audio I/O, AEC, VAD, endpointing, STT and TTS (ARCHITECTURE.md §3.4).

Used only inside the voice worker; the core talks to speech through ``SpeechOutput``
(``contracts.speech``). numpy is imported for the PCM type aliases only. Nothing here imports
``sounddevice`` or any other native audio stack: ``AudioBackend`` is the injectable subset of
``sounddevice`` that ``FakeSD`` implements.

PCM is float32 in [-1, 1] (``F32``) or int16 (``I16``), always with an explicit sample rate.
Every ``t``/``t0`` is ``time.perf_counter()`` seconds.

The dataclasses that carry arrays (``VadPartial``, ``VadEnd``, ``AudioChunk``) use identity
equality (``eq=False``): comparing numpy arrays element-wise inside a generated ``__eq__``
would raise.
"""

from __future__ import annotations

import collections
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

from aivtube.contracts.types import Transcript, VoiceSpec

__all__ = [
    "F32",
    "I16",
    "AudioBackend",
    "AudioChunk",
    "AudioIn",
    "AudioOut",
    "EchoCanceller",
    "Endpointer",
    "EndpointerConfig",
    "MarkCallback",
    "PhraseCache",
    "QuotaSpec",
    "SpeechRecognizer",
    "TTSBackend",
    "TTSUnavailable",
    "TranscriptPostProcessor",
    "VadEnd",
    "VadEvent",
    "VadPartial",
    "VadStart",
    "VoiceActivityDetector",
    "WordMark",
]

F32: TypeAlias = npt.NDArray[np.float32]
I16: TypeAlias = npt.NDArray[np.int16]
MarkCallback: TypeAlias = Callable[[bool, float], None]
"""``(heard, t_audible)``; called on the notifier thread when a marker reaches the DAC."""


@runtime_checkable
class AudioBackend(Protocol):
    """The subset of ``sounddevice`` we use; ``FakeSD`` implements it for tests."""

    def query_hostapis(self, index: int | None = None) -> Any: ...

    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> Any: ...

    def OutputStream(self, **kw: Any) -> Any: ...  # sounddevice naming

    def InputStream(self, **kw: Any) -> Any: ...  # sounddevice naming

    def WasapiSettings(self, **kw: Any) -> Any: ...  # sounddevice naming


@runtime_checkable
class AudioOut(Protocol):
    """Always-open player. ``reference`` holds post-gain ``(t, block)`` pairs for AEC."""

    sample_rate: int
    output_latency_s: float
    reference: collections.deque[tuple[float, F32]]
    stats: Mapping[str, int]

    def start(self) -> None: ...

    def play(self, pcm: F32, sample_rate: int) -> None: ...

    def mark(self, cb: MarkCallback) -> None:
        """Fire ``cb(True, t)`` when everything queued so far is audible, or ``cb(False, t)``
        if it is cancelled first."""
        ...

    def cancel(self, fade_ms: float = 30.0) -> float:
        """Fade out and drop the queue; pending marks fire ``False``. Returns seconds dropped."""
        ...

    def set_gain(self, gain: float, ramp_ms: float = 20.0) -> None: ...

    def is_speaking(self, tail_s: float = 0.25) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioIn(Protocol):
    sample_rate: int
    block_samples: int
    stats: Mapping[str, int]

    def start(self, on_frame: Callable[[F32, float], None]) -> None:
        """Deliver ``(block, t_capture)`` from the capture thread."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class EchoCanceller(Protocol):
    def feed_reference(self, block: F32) -> None: ...

    def process(self, mic_block: F32) -> F32: ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    sample_rate: int  # 16000
    frame_samples: int  # 512

    def reset(self) -> None: ...

    def prob(self, frame: F32) -> float: ...


@dataclass(frozen=True, slots=True)
class EndpointerConfig:
    threshold: float = 0.5
    neg_threshold: float = 0.35
    barge_threshold: float = 0.6
    min_speech_ms: int = 250
    end_silence_ms: int = 600
    preroll_ms: int = 300
    max_segment_s: float = 15.0
    max_turn_s: float = 60.0
    particle_endpointing: bool = False
    final_particle_ms: int = 420
    continuation_ms: int = 900


@dataclass(frozen=True, slots=True)
class VadStart:
    t: float
    barge: bool


@dataclass(frozen=True, slots=True, eq=False)
class VadPartial:
    """Forced split (``max_segment_s``): decode it, but the turn continues. Never an end of turn."""

    t: float
    audio: F32


@dataclass(frozen=True, slots=True, eq=False)
class VadEnd:
    """Real end of utterance; ``audio`` includes the pre-roll."""

    t: float
    audio: F32


VadEvent: TypeAlias = VadStart | VadPartial | VadEnd


@runtime_checkable
class Endpointer(Protocol):
    def push(self, frame16k: F32, t: float, ai_speaking: bool) -> list[VadEvent]: ...

    def set_tail_hint(self, text: str) -> None:
        """M2 particle endpointing: the latest partial text of the current turn."""
        ...

    def reset(self) -> None: ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    name: str
    sample_rate: int

    def warmup(self) -> None: ...

    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript:
        """Blocking; call it on the STT thread only. ``quick`` marks a barge-in decode."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class TranscriptPostProcessor(Protocol):
    def __call__(self, t: Transcript, recent_tts_text: str) -> Transcript | None:
        """Return a corrected transcript, or ``None`` to drop it (for example, echo)."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class AudioChunk:
    pcm: I16
    sample_rate: int


@dataclass(frozen=True, slots=True)
class WordMark:
    text: str
    offset_s: float  # from the start of this segment's audio
    duration_s: float


class TTSUnavailable(Exception):
    """The backend cannot synthesise right now (timeout, quota, network, no audio)."""


@dataclass(frozen=True, slots=True)
class QuotaSpec:
    max_requests: int
    per_s: float


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    normalizer: Literal["cloud", "local"]
    quota: QuotaSpec | None

    async def warmup(self) -> None: ...

    def synth(
        self,
        text: str,
        voice: VoiceSpec,
        *,
        first_audio_timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[AudioChunk | WordMark]:
        """Stream audio and word marks; raises ``TTSUnavailable`` on failure or timeout."""
        ...

    async def aclose(self) -> None: ...


@runtime_checkable
class PhraseCache(Protocol):
    def get(self, key: tuple[str, ...]) -> AudioChunk | None: ...

    def put(self, key: tuple[str, ...], audio: AudioChunk) -> None: ...
