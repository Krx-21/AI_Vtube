"""Speech-recognition fakes (ARCHITECTURE.md §3.4, §10)."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence

import numpy as np

from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32
from aivtube.testing.fakes.audio import dominant_frequency

__all__ = ["FakePostProcessor", "FakeRecognizer"]


def _parse_hz(key: str) -> float:
    k = key.strip().lower().removesuffix("hz").strip()
    return float(k)


class FakeRecognizer:
    """``SpeechRecognizer`` with scripted output.

    - ``script`` as a sequence: the texts are returned in order, then ``""``.
    - ``script`` as a mapping of tone frequency (``"440"`` or ``"440hz"``) to text: the
      dominant frequency of the audio (see ``marker_tone``) selects the sentence, within
      ``tolerance_hz``; unknown or silent audio gives ``""``.

    ``transcribe`` blocks for ``delay_s`` (``quick_delay_s`` for quick decodes) like a real
    decoder on the STT thread. ``fail_times`` makes the first N calls raise ``RuntimeError``.
    """

    def __init__(
        self,
        script: Mapping[str, str] | Sequence[str],
        delay_s: float = 0.0,
        *,
        quick_delay_s: float | None = None,
        name: str = "fake",
        tolerance_hz: float = 15.0,
        fail_times: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.delay_s = delay_s
        self.quick_delay_s = delay_s if quick_delay_s is None else quick_delay_s
        self.tolerance_hz = tolerance_hz
        self.fail_times = fail_times
        self._tones: dict[float, str] | None = None
        self._texts: list[str] = []
        if isinstance(script, Mapping):
            self._tones = {_parse_hz(k): v for k, v in script.items()}
        else:
            self._texts = list(script)
        self._lock = threading.Lock()
        self._next = 0
        self.calls: list[tuple[int, bool]] = []  # (samples, quick)
        self.warmed = False
        self.closed = False

    def warmup(self) -> None:
        self.warmed = True

    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript:
        t0 = time.perf_counter()
        x = np.asarray(pcm16k, np.float32).reshape(-1)
        with self._lock:
            if self.closed:
                raise RuntimeError(f"recognizer {self.name!r} is closed")
            self.calls.append((int(x.size), quick))
            fail = len(self.calls) <= self.fail_times
        delay = self.quick_delay_s if quick else self.delay_s
        if delay > 0:
            time.sleep(delay)
        if fail:
            raise RuntimeError(f"{self.name}: scripted failure")
        text = self._text_for(x)
        return Transcript(
            text=text,
            is_final=True,
            audio_s=x.size / self.sample_rate,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            engine=self.name,
        )

    def close(self) -> None:
        self.closed = True

    def _text_for(self, x: F32) -> str:
        if self._tones is not None:
            f = dominant_frequency(x, self.sample_rate)
            if f is None:
                return ""
            best = min(self._tones, key=lambda hz: abs(hz - f), default=None)
            if best is None or abs(best - f) > self.tolerance_hz:
                return ""
            return self._tones[best]
        with self._lock:
            i = self._next
            self._next += 1
        return self._texts[i] if i < len(self._texts) else ""


class FakePostProcessor:
    """``TranscriptPostProcessor``: optional alias replacement and echo drop, plus a call log."""

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]] | None = None,
        *,
        drop_if_in_tts: bool = False,
        min_audio_s: float = 0.0,
    ) -> None:
        self.aliases = {v: k for k, vs in (aliases or {}).items() for v in vs}
        self.drop_if_in_tts = drop_if_in_tts
        self.min_audio_s = min_audio_s
        self.calls: list[tuple[Transcript, str]] = []

    def __call__(self, t: Transcript, recent_tts_text: str) -> Transcript | None:
        self.calls.append((t, recent_tts_text))
        if t.audio_s < self.min_audio_s or not t.text.strip():
            return None
        if self.drop_if_in_tts and t.text.strip() and t.text.strip() in recent_tts_text:
            return None
        text = t.text
        for wrong, right in self.aliases.items():
            text = text.replace(wrong, right)
        if text == t.text:
            return t
        return Transcript(text, t.is_final, t.audio_s, t.latency_ms, t.engine)
