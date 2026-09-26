"""PyThaiASR 2.1 (Typhoon RT fp32 ONNX, pure onnxruntime) as a fallback recognizer.

``pythaiasr`` is imported lazily and the model is loaded on first use (``warmup``), because
constructing it downloads ~460 MB to ``~/pythaiasr-data`` when the files are missing; run
``aivtube setup`` beforehand so that never happens on stream.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32
from aivtube.voice.stt.sherpa import to_mono_f32

__all__ = ["PyThaiAsrRecognizer"]

SAMPLE_RATE = 16000


def _default_loader(model_dir: Path | None, device: str) -> Any:
    from pythaiasr.typhoon import FastConformerRNNT

    return FastConformerRNNT(model_dir=str(model_dir) if model_dir else None, device=device)


class PyThaiAsrRecognizer:
    """``SpeechRecognizer`` over ``pythaiasr.typhoon.FastConformerRNNT``.

    ``loader(model_dir, device)`` builds the model (tests inject a fake); the model must offer
    ``transcribe(audio, sample_rate=16000) -> str``.
    """

    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        device: str = "cpu",
        name: str = "pythaiasr",
        loader: Callable[[Path | None, str], Any] | None = None,
    ) -> None:
        self.name = name
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self._loader = loader or _default_loader
        self._model: Any = None
        self._closed = False
        self._lock = threading.Lock()

    def _ensure(self) -> Any:
        if self._closed:
            raise RuntimeError(f"recognizer {self.name!r} is closed")
        if self._model is None:
            self._model = self._loader(self.model_dir, self.device)
        return self._model

    def warmup(self) -> None:
        with self._lock:
            self._ensure()
        self.transcribe(np.zeros(SAMPLE_RATE, np.float32))

    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript:
        t0 = time.perf_counter()
        x = to_mono_f32(pcm16k)
        with self._lock:
            model = self._ensure()
            result = model.transcribe(x, sample_rate=SAMPLE_RATE)
        text = result.get("text", "") if isinstance(result, dict) else result
        return Transcript(
            text=str(text or "").strip(),
            is_final=True,
            audio_s=x.size / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            engine=self.name,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._model = None
