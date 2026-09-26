"""Typhoon ASR Realtime (int8 ONNX) in sherpa-onnx ``OfflineRecognizer`` on the CPU (STT brief B).

One recognizer per process; decode only on the STT thread (``decode_stream`` releases the
GIL). The model directory holds ``encoder[.int8].onnx``, ``decoder[.int8].onnx``,
``joiner[.int8].onnx`` and ``tokens.txt`` from our NeMo export (CC-BY-4.0, Typhoon).
``sherpa_onnx`` is imported lazily, so the voice worker imports without it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32

__all__ = ["SherpaTyphoonRT", "find_transducer_files", "to_mono_f32"]

SAMPLE_RATE = 16000


def to_mono_f32(pcm: npt.ArrayLike) -> F32:
    """Contiguous 1-D float32 in [-1, 1] (int16 is scaled)."""
    x = np.asarray(pcm)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    x = x.astype(np.float32, copy=False).reshape(-1)
    return np.ascontiguousarray(x)


def find_transducer_files(model_dir: Path) -> dict[str, Path]:
    """Locate encoder/decoder/joiner (int8 preferred) and tokens.txt; ``FileNotFoundError``."""
    found: dict[str, Path] = {}
    for part in ("encoder", "decoder", "joiner"):
        for name in (f"{part}.int8.onnx", f"{part}.onnx"):
            if (model_dir / name).is_file():
                found[part] = model_dir / name
                break
        else:
            raise FileNotFoundError(f"{part}[.int8].onnx not found in {model_dir}")
    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        raise FileNotFoundError(f"tokens.txt not found in {model_dir}")
    found["tokens"] = tokens
    return found


class SherpaTyphoonRT:
    """``SpeechRecognizer``: Typhoon RT through sherpa-onnx (nemo_transducer, greedy search).

    ``quick=True`` (barge-in confirmation) decodes only the last ``quick_max_s`` seconds.
    Audio shorter than ``min_decode_s`` is padded with trailing silence before decoding, which
    keeps the FastConformer's 8x subsampling happy; ``audio_s`` reports the real length.
    ``factory`` replaces ``sherpa_onnx.OfflineRecognizer.from_transducer`` (tests).
    """

    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        model_dir: Path | str,
        *,
        num_threads: int = 2,
        provider: str = "cpu",
        name: str = "typhoon_rt",
        quick_max_s: float = 1.2,
        min_decode_s: float = 0.4,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.quick_max_s = quick_max_s
        self.min_decode_s = min_decode_s
        files = find_transducer_files(self.model_dir)
        if factory is None:
            import sherpa_onnx

            factory = sherpa_onnx.OfflineRecognizer.from_transducer
        self._rec: Any = factory(
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            tokens=str(files["tokens"]),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider=provider,
        )
        self._lock = threading.Lock()  # sherpa streams are not meant for concurrent decodes
        self.warmed = False

    def warmup(self) -> None:
        """Decode 1 s and a quick 0.8 s of silence so the first real decode is fast."""
        self.transcribe(np.zeros(SAMPLE_RATE, np.float32))
        self.transcribe(np.zeros(int(0.8 * SAMPLE_RATE), np.float32), quick=True)
        self.warmed = True

    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript:
        t0 = time.perf_counter()
        x = to_mono_f32(pcm16k)
        audio_s = x.size / SAMPLE_RATE
        if quick and self.quick_max_s > 0:
            x = x[-int(self.quick_max_s * SAMPLE_RATE) :]
        need = int(self.min_decode_s * SAMPLE_RATE)
        if x.size < need:
            x = np.concatenate([x, np.zeros(need - x.size, np.float32)])
        with self._lock:
            rec = self._rec
            if rec is None:
                raise RuntimeError(f"recognizer {self.name!r} is closed")
            stream = rec.create_stream()
            stream.accept_waveform(SAMPLE_RATE, x)
            rec.decode_stream(stream)
            text = str(stream.result.text).strip()
        return Transcript(
            text=text,
            is_final=True,
            audio_s=audio_s,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            engine=self.name,
        )

    def close(self) -> None:
        with self._lock:
            self._rec = None
