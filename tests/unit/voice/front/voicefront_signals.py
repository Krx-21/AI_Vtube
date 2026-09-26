"""Signal and clock helpers for the voice front-end tests (imported by conftest and tests).

``Synth.speech`` makes a speech-like signal (glottal harmonics shaped by moving formants,
170–270 ms syllables with a short noise consonant) that the real Silero v6 model scores above
0.9 for nearly every 32 ms frame, while pure tones and silence score ~0.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from aivtube.contracts.voice import F32

_VOWELS = np.array(
    [
        (730, 1090, 2440),
        (270, 2290, 3010),
        (300, 870, 2240),
        (530, 1840, 2480),
        (570, 840, 2410),
        (660, 1720, 2410),
    ],
    dtype=np.float64,
)
_BANDWIDTHS = np.array([90.0, 110.0, 170.0])


class ManualClock:
    """A thread-safe settable ``perf_counter`` stand-in: ``clock()`` returns ``clock.t``."""

    def __init__(self, t: float = 100.0) -> None:
        self._t = t
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, dt: float) -> float:
        with self._lock:
            self._t += dt
            return self._t

    def set(self, t: float) -> None:
        with self._lock:
            self._t = t


@dataclass(frozen=True)
class Synth:
    """Deterministic test signals (float32 mono)."""

    def silence(self, seconds: float, sr: int = 16000) -> F32:
        return np.zeros(round(seconds * sr), np.float32)

    def noise(self, seconds: float, sr: int = 16000, amp: float = 0.001, seed: int = 0) -> F32:
        rng = np.random.default_rng(seed)
        return (amp * rng.standard_normal(round(seconds * sr))).astype(np.float32)

    def tone(self, seconds: float, freq: float = 440.0, sr: int = 16000, amp: float = 0.3) -> F32:
        t = np.arange(round(seconds * sr), dtype=np.float64) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def speech(self, seconds: float, *, seed: int = 0, f0: float = 100.0, amp: float = 0.3) -> F32:
        """Speech-like audio at 16 kHz (resample it for other rates)."""
        sr = 16000
        rng = np.random.default_rng(seed)
        n_total = round(seconds * sr)
        out = np.zeros(n_total)
        pos, phase = 0, 0.0
        harmonics = np.arange(1, int(7000 / f0))
        while pos < n_total:
            syl = int(rng.uniform(0.17, 0.27) * sr)
            n = min(syl, n_total - pos)
            tt = np.arange(n) / sr
            f_contour = f0 * (1.15 - 0.3 * tt / (syl / sr))
            ph = phase + 2 * np.pi * np.cumsum(f_contour) / sr
            phase = float(ph[-1])
            fa = _VOWELS[rng.integers(len(_VOWELS))]
            fb = _VOWELS[rng.integers(len(_VOWELS))]
            mix = tt / (syl / sr)
            y = np.zeros(n)
            for h in harmonics:
                fh = h * f0
                ga = np.sum(1 / (1 + ((fh - fa) / _BANDWIDTHS) ** 2))
                gb = np.sum(1 / (1 + ((fh - fb) / _BANDWIDTHS) ** 2))
                y += (ga * (1 - mix) + gb * mix + 0.01) / np.sqrt(h) * np.sin(h * ph)
            env = np.clip(np.minimum(tt / 0.025, (syl / sr - tt) / 0.04), 0.0, 1.0)
            y *= np.minimum(1.0, env)
            c = min(int(0.04 * sr), n)
            nz = np.diff(rng.standard_normal(c), prepend=0.0)
            peak = float(np.max(np.abs(y))) or 1.0
            y[:c] = y[:c] * np.linspace(0, 1, c) + 0.15 * peak * nz / (np.max(np.abs(nz)) or 1.0)
            out[pos : pos + n] = y
            pos += syl
        peak = float(np.max(np.abs(out))) or 1.0
        return (amp * out / peak).astype(np.float32)

    def up48(self, x16: F32) -> F32:
        """16 kHz → 48 kHz with soxr (what a 48 kHz mic would deliver)."""
        import soxr

        return np.asarray(soxr.resample(x16, 16000, 48000), dtype=np.float32)
