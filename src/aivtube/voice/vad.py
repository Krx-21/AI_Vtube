"""Voice activity detectors (ARCHITECTURE.md §0 STT/VAD, §3.4 ``VoiceActivityDetector``).

- ``SileroOrtVAD``: the torch-free Silero v6 wrapper (numpy + onnxruntime, one thread). Each
  call takes a 512-sample 16 kHz frame; the graph input is the previous frame's last 64
  samples followed by the frame (``[1, 576]``), plus the recurrent state ``[2, 1, 128]``.
- ``SherpaSileroVAD``: the same model on sherpa-onnx's bundled onnxruntime (the S3 coexistence
  fallback). Its per-frame ``is_speech_detected`` becomes a probability of 0 or 1; the
  endpointer's pre-roll ring covers its small onset lag.
- ``EnergyVAD``: dependency-free, dBFS mapped to a pseudo-probability.

Never ``pip install silero-vad``: it pulls in torch. onnxruntime and sherpa-onnx are imported
lazily, when a detector is built.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from aivtube.contracts.voice import F32, VoiceActivityDetector

__all__ = [
    "EnergyVAD",
    "ModelChecksumError",
    "SherpaSileroVAD",
    "SileroOrtVAD",
    "VadBackend",
    "make_vad",
    "sha256_file",
]

log = logging.getLogger("aivtube.voice.vad")

VadBackend = Literal["silero_ort", "silero_sherpa", "energy"]

_SR: Final = 16000
_FRAME: Final = 512
_CONTEXT: Final = 64


class ModelChecksumError(ValueError):
    """A model file does not match its pinned sha256."""


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file (blocking: call it off the event loop)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _frame(frame: F32) -> F32:
    x = np.asarray(frame, dtype=np.float32).reshape(-1)
    if x.size != _FRAME:
        raise ValueError(f"VAD frames are {_FRAME} samples at {_SR} Hz, got {x.size}")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x


class SileroOrtVAD:
    """Silero VAD v5/v6 ONNX on onnxruntime (CPU, one intra-op and one inter-op thread)."""

    sample_rate: int = _SR
    frame_samples: int = _FRAME

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        self._sess: Any = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        names = {i.name for i in self._sess.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise ValueError(f"{onnx_path} is not a Silero v5/v6 VAD model (inputs {names})")
        self._sr = np.array(_SR, dtype=np.int64)
        self._x = np.zeros((1, _CONTEXT + _FRAME), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._x.fill(0.0)

    def prob(self, frame: F32) -> float:
        x = self._x
        x[0, _CONTEXT:] = _frame(frame)
        out, state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = np.asarray(state, dtype=np.float32)
        x[0, :_CONTEXT] = x[0, _FRAME:]  # context for the next frame: this frame's last 64
        p = float(np.asarray(out).reshape(-1)[0])
        if not math.isfinite(p):
            self.reset()
            return 0.0
        return min(1.0, max(0.0, p))


class SherpaSileroVAD:
    """Silero on sherpa-onnx's own runtime; ``prob`` is 1.0 while sherpa detects speech."""

    sample_rate: int = _SR
    frame_samples: int = _FRAME

    def __init__(self, onnx_path: Path, *, threshold: float = 0.5) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(onnx_path)
        cfg.silero_vad.threshold = float(threshold)
        cfg.silero_vad.min_silence_duration = 0.0  # the endpointer does the smoothing
        cfg.silero_vad.min_speech_duration = 0.0
        cfg.silero_vad.max_speech_duration = 3600.0
        cfg.silero_vad.window_size = _FRAME
        cfg.sample_rate = _SR
        cfg.num_threads = 1
        cfg.provider = "cpu"
        self._vad: Any = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

    def reset(self) -> None:
        self._vad.reset()

    def prob(self, frame: F32) -> float:
        vad = self._vad
        vad.accept_waveform(_frame(frame))
        p = 1.0 if vad.is_speech_detected() else 0.0
        while not vad.empty():  # we never use sherpa's segments; keep its buffer small
            vad.pop()
        return p


class EnergyVAD:
    """Maps frame dBFS to a pseudo-probability: 0.5 at ``threshold_dbfs``, ±``width_db``/2."""

    sample_rate: int = _SR
    frame_samples: int = _FRAME

    def __init__(self, threshold_dbfs: float = -42.0, *, width_db: float = 12.0) -> None:
        if width_db <= 0:
            raise ValueError("width_db must be positive")
        self.threshold_dbfs = threshold_dbfs
        self.width_db = width_db

    def reset(self) -> None:
        return None

    def prob(self, frame: F32) -> float:
        x = _frame(frame)
        db = 10.0 * math.log10(float(np.dot(x, x)) / x.size + 1e-12)
        return min(1.0, max(0.0, (db - self.threshold_dbfs) / self.width_db + 0.5))


def make_vad(
    backend: VadBackend,
    model: Path,
    sha256: str,
    *,
    threshold: float = 0.5,
    energy_threshold_dbfs: float = -42.0,
) -> VoiceActivityDetector:
    """Build the configured detector (blocking: reads and hashes the model).

    ``sha256`` (lowercase hex; ``""`` skips the check) must match the model file, otherwise
    ``ModelChecksumError``. The energy backend needs no model.
    """
    if backend == "energy":
        return EnergyVAD(energy_threshold_dbfs)
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"VAD model not found: {path}")
    if sha256:
        actual = sha256_file(path)
        if actual != sha256.strip().lower():
            raise ModelChecksumError(
                f"{path} has sha256 {actual}, expected {sha256}; re-download it (aivtube setup)"
            )
    if backend == "silero_ort":
        return SileroOrtVAD(path)
    if backend == "silero_sherpa":
        return SherpaSileroVAD(path, threshold=threshold)
    raise ValueError(f"unknown VAD backend {backend!r}")
