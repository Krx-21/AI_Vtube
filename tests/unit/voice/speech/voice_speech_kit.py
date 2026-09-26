"""Test helpers for the voice speech tests: synthetic MP3, fake edge stream, fake Azure SDK.

Everything here is license-clean and offline: the MP3 is encoded on the fly from a synthetic
"vowel" signal with PyAV's libmp3lame, and split into ~720-byte messages like edge-tts sends.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np

SR = 24000


def vowel(seconds: float, sr: int = SR, f0: float = 220.0, amp: float = 0.4) -> np.ndarray:
    """A voiced, syllable-like test signal (harmonics with a 4 Hz envelope), float64."""
    t = np.arange(int(seconds * sr)) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 5))
    return amp * env * x / 1.6


def encode_mp3(x: np.ndarray, sr: int = SR, bitrate: int = 48000) -> bytes:
    """Raw MPEG-2 Layer III frames (no ID3), like the edge-tts audio stream."""
    import av

    cc = av.CodecContext.create("libmp3lame", "w")
    cc.sample_rate = sr
    cc.layout = "mono"
    cc.format = "s32p"
    cc.bit_rate = bitrate
    cc.open()
    fs = cc.frame_size
    out = b""
    for i in range(0, len(x), fs):
        blk = np.asarray(x[i : i + fs], np.float64)
        if blk.size < fs:
            blk = np.pad(blk, (0, fs - blk.size))
        arr = (np.clip(blk, -1, 1) * 2147483000).astype(np.int32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s32p", layout="mono")
        frame.sample_rate = sr
        frame.pts = i
        for pkt in cc.encode(frame):
            out += bytes(pkt)
    for pkt in cc.encode(None):
        out += bytes(pkt)
    return out


def decode_whole(data: bytes) -> np.ndarray:
    """Reference: decode the whole MP3 buffer at once."""
    import av

    ctx = av.CodecContext.create("mp3", "r")
    out = []
    for pkt in ctx.parse(data):
        for fr in ctx.decode(pkt):
            out.append(fr.to_ndarray().reshape(-1))
    for pkt in ctx.parse(None):
        for fr in ctx.decode(pkt):
            out.append(fr.to_ndarray().reshape(-1))
    for fr in ctx.decode(None):
        out.append(fr.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16)


def edge_messages(
    text: str, mp3: bytes, *, chunk: int = 720, cps: float = 12.5, marks_first: int = 3
) -> list[dict[str, Any]]:
    """edge-tts ``stream()`` messages: most WordBoundary metadata first, then audio chunks."""
    words = [w for w in text.split(" ") if w]
    marks: list[dict[str, Any]] = []
    pos = 0
    for w in words:
        marks.append(
            {
                "type": "WordBoundary",
                "offset": int(pos / cps * 1e7),
                "duration": int(len(w) / cps * 1e7),
                "text": w,
            }
        )
        pos += len(w) + 1
    audio = [{"type": "audio", "data": mp3[i : i + chunk]} for i in range(0, len(mp3), chunk)]
    msgs = marks[:marks_first] + audio[:3] + marks[marks_first:] + audio[3:]
    return msgs


@dataclass
class ScriptedCommunicate:
    """Stand-in for ``edge_tts.Communicate``: replays messages with optional delays."""

    text: str
    voice: str
    kwargs: dict[str, Any]
    messages: Sequence[dict[str, Any]]
    delays: Sequence[float] = ()
    sleep: Callable[[float], Any] = asyncio.sleep
    fail_after: int | None = None
    exc: BaseException | None = None
    closed: bool = False
    yielded: int = 0

    async def _gen(self) -> AsyncIterator[dict[str, Any]]:
        try:
            for i, msg in enumerate(self.messages):
                if i < len(self.delays) and self.delays[i]:
                    await self.sleep(self.delays[i])
                if self.fail_after is not None and i >= self.fail_after:
                    raise self.exc or RuntimeError("scripted edge failure")
                self.yielded += 1
                yield msg
            if self.fail_after is not None and self.fail_after >= len(self.messages):
                raise self.exc or RuntimeError("scripted edge failure")
        finally:
            self.closed = True

    def stream(self) -> AsyncIterator[dict[str, Any]]:
        return self._gen()


@dataclass
class CommunicateFactory:
    """Builds ``ScriptedCommunicate`` objects and records them."""

    script: Callable[[str], Sequence[dict[str, Any]]]
    delays: Sequence[float] = ()
    sleep: Callable[[float], Any] = asyncio.sleep
    fail_after: int | None = None
    exc: BaseException | None = None
    made: list[ScriptedCommunicate] = field(default_factory=list)

    def __call__(self, text: str, voice: str, **kwargs: Any) -> ScriptedCommunicate:
        c = ScriptedCommunicate(
            text,
            voice,
            kwargs,
            self.script(text),
            self.delays,
            self.sleep,
            self.fail_after,
            self.exc,
        )
        self.made.append(c)
        return c


# --- fake Azure SDK ---------------------------------------------------------------------------


class _Signal:
    def __init__(self) -> None:
        self.handlers: list[Callable[[Any], None]] = []

    def connect(self, cb: Callable[[Any], None]) -> None:
        self.handlers.append(cb)

    def fire(self, evt: Any) -> None:
        for cb in list(self.handlers):
            cb(evt)


class FakeAzureSdk:
    """The subset of ``azure.cognitiveservices.speech`` that ``AzureTTSBackend`` uses.

    ``speak_ssml_async`` starts a thread that emits ``synthesizing`` events (PCM bytes, split
    at odd sizes to test carrying), word boundaries and ``synthesis_completed``. Knobs:
    ``delay_s`` before the first event, ``cancel`` to emit ``synthesis_canceled``, ``silent`` to
    emit nothing at all.
    """

    SpeechSynthesisOutputFormat = SimpleNamespace(Raw24Khz16BitMonoPcm="raw24k")
    SpeechSynthesisBoundaryType = SimpleNamespace(Word="word", Punctuation="punct")

    def __init__(self, *, seconds: float = 0.6, delay_s: float = 0.0) -> None:
        self.seconds = seconds
        self.delay_s = delay_s
        self.cancel = False
        self.silent = False
        self.ssml: list[str] = []
        self.synths: list[Any] = []
        self.opened = 0
        self.stopped = 0
        self.formats: list[Any] = []
        sdk = self

        class SpeechConfig:
            def __init__(self, subscription: str, region: str) -> None:
                self.subscription, self.region = subscription, region

            def set_speech_synthesis_output_format(self, fmt: Any) -> None:
                sdk.formats.append(fmt)

        class SpeechSynthesizer:
            def __init__(self, speech_config: Any, audio_config: Any) -> None:
                assert audio_config is None
                self.synthesizing = _Signal()
                self.synthesis_word_boundary = _Signal()
                self.synthesis_completed = _Signal()
                self.synthesis_canceled = _Signal()
                sdk.synths.append(self)

            def speak_ssml_async(self, ssml: str) -> Any:
                sdk.ssml.append(ssml)
                threading.Thread(target=sdk._emit, args=(self,), daemon=True).start()
                return SimpleNamespace(get=lambda: None)

            def stop_speaking_async(self) -> Any:
                sdk.stopped += 1
                return SimpleNamespace(get=lambda: None)

        class Connection:
            @staticmethod
            def from_speech_synthesizer(syn: Any) -> Any:
                def open_(persistent: bool) -> None:
                    sdk.opened += 1

                return SimpleNamespace(open=open_, close=lambda: None)

        self.SpeechConfig = SpeechConfig
        self.SpeechSynthesizer = SpeechSynthesizer
        self.Connection = Connection

    def _emit(self, syn: Any) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.silent:
            return
        if self.cancel:
            details = SimpleNamespace(reason="Error", error_details="quota exceeded")
            syn.synthesis_canceled.fire(
                SimpleNamespace(result=SimpleNamespace(cancellation_details=details))
            )
            return
        for i, word in enumerate(("สวัสดี", "ค่ะ")):
            syn.synthesis_word_boundary.fire(
                SimpleNamespace(
                    text=word,
                    audio_offset=int(i * 0.3 * 1e7),
                    duration=timedelta(seconds=0.25),
                    boundary_type="word",
                )
            )
        syn.synthesis_word_boundary.fire(
            SimpleNamespace(text="!", audio_offset=0, duration=timedelta(0), boundary_type="punct")
        )
        pcm = (np.sin(np.arange(int(self.seconds * SR)) * 0.05) * 8000).astype("<i2").tobytes()
        cuts = [0, 1001, 4801, 9601, len(pcm)]  # odd split points
        for a, b in itertools.pairwise(cuts):
            syn.synthesizing.fire(SimpleNamespace(result=SimpleNamespace(audio_data=pcm[a:b])))
        syn.synthesis_completed.fire(SimpleNamespace(result=SimpleNamespace(audio_data=pcm)))


async def wait_real(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    """Poll in real time (callbacks from other threads)."""
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > end:
            raise TimeoutError("condition not met in time")
        await asyncio.sleep(0.005)
