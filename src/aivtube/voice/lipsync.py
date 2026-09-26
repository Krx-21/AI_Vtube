"""Streaming lip-sync: TTS PCM → MouthOpen / MouthForm tracks at ``fps`` (§3.7, avatar brief).

We own the TTS audio, so the mouth is computed from the PCM itself (no loopback). Per frame of
``sample_rate / fps`` samples:

- **MouthOpen**: RMS → dBFS (+ ``gain_db``) → normalised between the noise floor ``floor_db``
  and ``ceil_db`` → ``gamma`` → one-pole attack/release smoothing.
- **MouthForm** (heuristic spectral tilt): log ratio of the 1.8–4 kHz band ("i/e", spread) to the
  250–900 Hz band ("o/u", round) mapped to 0..1 (0.5 neutral), held at 0.5 while unvoiced and
  smoothed. It is not phoneme-accurate; it only breaks the "flapping jaw" look.

Frame ``k`` always covers samples ``[round(k*sr/fps), round((k+1)*sr/fps))`` of the stream, so
feeding the PCM in arbitrary chunks gives exactly the frames of a whole-buffer analysis. Index
``k`` corresponds to playback time ``t0 + k/fps`` where ``t0`` is the segment's audible start;
the 40 ms lead is applied by the avatar driver, not here.

Pure numpy; about 1 ms of CPU per second of 24 kHz audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["LipSyncAnalyzer", "LipSyncConfig", "pcm_to_f32"]

_EPS = 1e-12
F32Array = npt.NDArray[np.float32]
_Spectral = tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]


@dataclass(frozen=True, slots=True)
class LipSyncConfig:
    """Tuning of the mouth tracks (defaults from the avatar brief)."""

    fps: int = 60
    floor_db: float = -45.0  # at or below: mouth closed (noise gate)
    ceil_db: float = -12.0  # at or above: fully open
    gamma: float = 0.7  # < 1 opens the mouth more on quiet syllables
    attack_ms: float = 30.0
    release_ms: float = 80.0
    gain_db: float = 0.0  # input gain normalisation for quiet/loud voices
    max_open: float = 1.0
    form_gain: float = 0.35  # MouthForm swing around 0.5
    form_attack_ms: float = 60.0
    form_release_ms: float = 120.0
    voiced_level: float = 0.05  # below this mouth level, MouthForm returns to neutral

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.ceil_db <= self.floor_db:
            raise ValueError("ceil_db must be above floor_db")
        if min(self.attack_ms, self.release_ms, self.form_attack_ms, self.form_release_ms) <= 0:
            raise ValueError("smoothing times must be positive")


def _coef(fps: int, tau_ms: float) -> float:
    """One-pole coefficient ``1 - exp(-1000 / (fps * tau_ms))``."""
    return 1.0 - math.exp(-1000.0 / (fps * tau_ms))


def pcm_to_f32(pcm: npt.ArrayLike) -> F32Array:
    """Mono float32 in [-1, 1] from int16 or float PCM (multi-channel is averaged)."""
    x = np.asarray(pcm)
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[0] > x.shape[1] else x.mean(axis=0)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).reshape(-1)
    return x.astype(np.float32, copy=False).reshape(-1)


class LipSyncAnalyzer:
    """Incremental mouth analysis for one segment at a time; call ``reset()`` between segments."""

    def __init__(self, cfg: LipSyncConfig | None = None) -> None:
        self.cfg = cfg or LipSyncConfig()
        self._a_up = _coef(self.cfg.fps, self.cfg.attack_ms)
        self._a_dn = _coef(self.cfg.fps, self.cfg.release_ms)
        self._f_up = _coef(self.cfg.fps, self.cfg.form_attack_ms)
        self._f_dn = _coef(self.cfg.fps, self.cfg.form_release_ms)
        self._spectral: dict[tuple[int, int], _Spectral] = {}
        self.reset()

    @property
    def fps(self) -> int:
        return self.cfg.fps

    @property
    def frames(self) -> int:
        """Frames produced since the last ``reset()``."""
        return self._frame

    def reset(self) -> None:
        self._sr: int | None = None
        self._buf = np.zeros(0, np.float32)
        self._consumed = 0  # absolute index of self._buf[0] in the stream
        self._frame = 0
        self._mouth = 0.0
        self._form = 0.5

    def feed(self, pcm: npt.ArrayLike, sample_rate: int) -> tuple[F32Array, F32Array]:
        """Analyse more PCM; returns the ``(mouth, form)`` frames it completed (maybe empty)."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self._sr is None:
            self._sr = int(sample_rate)
        elif sample_rate != self._sr:
            raise ValueError(f"sample rate changed from {self._sr} to {sample_rate}; reset() first")
        x = pcm_to_f32(pcm)
        if x.size:
            self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        return self._run(final=False)

    def flush(self) -> tuple[F32Array, F32Array]:
        """Emit the last partial frame (if any samples are left) at the end of a segment."""
        if self._sr is None:
            return _empty(), _empty()
        return self._run(final=True)

    # --- internals ------------------------------------------------------------------------
    def _bounds(self, k: int) -> int:
        assert self._sr is not None
        return round(k * self._sr / self.cfg.fps)

    def _run(self, *, final: bool) -> tuple[F32Array, F32Array]:
        frames: list[F32Array] = []
        end_abs = self._consumed + self._buf.size
        while True:
            lo, hi = self._bounds(self._frame), self._bounds(self._frame + 1)
            if hi > end_abs:
                if final and lo < end_abs:
                    frames.append(self._buf[lo - self._consumed : end_abs - self._consumed])
                    self._frame += 1
                    lo = end_abs
                break
            frames.append(self._buf[lo - self._consumed : hi - self._consumed])
            self._frame += 1
        start = self._bounds(self._frame) if not final else end_abs
        start = min(start, end_abs)
        if start > self._consumed:
            self._buf = self._buf[start - self._consumed :].copy()
            self._consumed = start
        if not frames:
            return _empty(), _empty()
        levels, tilts = self._features(frames)
        return self._smooth(levels, tilts)

    def _features(self, frames: list[F32Array]) -> tuple[F32Array, F32Array]:
        cfg = self.cfg
        assert self._sr is not None
        n = len(frames)
        levels = np.empty(n, np.float32)
        tilts = np.empty(n, np.float32)
        by_len: dict[int, list[int]] = {}
        for i, fr in enumerate(frames):
            by_len.setdefault(fr.size, []).append(i)
        span = cfg.ceil_db - cfg.floor_db
        for size, idx in by_len.items():
            block = np.stack([frames[i] for i in idx]).astype(np.float64)
            rms = np.sqrt(np.mean(block * block, axis=1) + _EPS)
            db = 20.0 * np.log10(rms) + cfg.gain_db
            lvl = np.clip((db - cfg.floor_db) / span, 0.0, 1.0) ** cfg.gamma
            if size >= 8:
                window, lo_mask, hi_mask = self._spectral_setup(size)
                spec = np.abs(np.fft.rfft(block * window, axis=1)) ** 2
                lo = spec[:, lo_mask].sum(axis=1) + _EPS
                hi = spec[:, hi_mask].sum(axis=1) + _EPS
                form = np.clip(0.5 + cfg.form_gain * (np.log10(hi / lo) + 1.0), 0.0, 1.0)
            else:
                form = np.full(len(idx), 0.5)
            form = np.where(lvl > cfg.voiced_level, form, 0.5)
            levels[idx] = lvl
            tilts[idx] = form
        return levels, tilts

    def _spectral_setup(self, size: int) -> _Spectral:
        """Hann window and band masks for a frame length (cached)."""
        assert self._sr is not None
        key = (size, self._sr)
        got = self._spectral.get(key)
        if got is None:
            freqs = np.fft.rfftfreq(size, 1.0 / self._sr)
            lo_mask = (freqs >= 250.0) & (freqs < 900.0)
            hi_mask = (freqs >= 1800.0) & (freqs < 4000.0)
            got = (np.hanning(size), lo_mask, hi_mask)
            self._spectral[key] = got
        return got

    def _smooth(self, levels: F32Array, tilts: F32Array) -> tuple[F32Array, F32Array]:
        mouth = np.empty(levels.size, np.float32)
        form = np.empty(levels.size, np.float32)
        m, f = self._mouth, self._form
        for i in range(levels.size):
            v = float(levels[i])
            m += (self._a_up if v > m else self._a_dn) * (v - m)
            u = float(tilts[i])
            f += (self._f_up if u > f else self._f_dn) * (u - f)
            mouth[i] = min(1.0, max(0.0, m * self.cfg.max_open))
            form[i] = min(1.0, max(0.0, f))
        self._mouth, self._form = m, f
        return mouth, form


def _empty() -> F32Array:
    return np.zeros(0, np.float32)
