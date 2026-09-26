"""Echo handling for streamers on speakers (ARCHITECTURE.md §4.7 step 6; audio brief).

- ``WebRtcAEC`` (alias ``LiveKitAEC``): WebRTC AEC3 through ``livekit.rtc.AudioProcessingModule``
  (lazy import, ``aec`` extra). Frames are exactly 10 ms of int16; the reference is the
  player's post-gain output (``StreamingPlayer.reference``), fed before the matching mic frames.
- ``EnergyDTD``: a Geigel-style energy double-talk gate on the VAD probability; the automatic
  fallback when livekit cannot be imported. It does not clean the audio.
- ``HalfDuplexGate``: ignore the mic while she speaks (bad rooms, loud game audio).
- ``NullAEC``: a pass-through canceller.

``make_echo_canceller`` picks one from ``audio.echo_mode`` and ``audio.headphones``.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np

from aivtube.contracts.speech import EchoMode
from aivtube.contracts.voice import F32, EchoCanceller

__all__ = [
    "EnergyDTD",
    "HalfDuplexGate",
    "LiveKitAEC",
    "NullAEC",
    "WebRtcAEC",
    "make_echo_canceller",
]

log = logging.getLogger("aivtube.voice.aec")

_APM_RATES: Final = frozenset({8000, 16000, 32000, 48000})


def _to_int16(x: F32) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


class WebRtcAEC:
    """WebRTC AEC3 (+ noise suppression and high-pass) via livekit's AudioProcessingModule.

    Mono, ``rate`` in {8, 16, 32, 48} kHz. Use it from one thread (the mic thread): call
    ``feed_reference`` with the player blocks that became audible, then ``process`` the mic
    block. Audio that is not a whole number of 10 ms frames is carried over (reference) or
    passed through unprocessed (mic tail). Raises ``ImportError`` when livekit is missing.
    """

    def __init__(
        self,
        rate: int = 48000,
        *,
        ns: bool = True,
        hpf: bool = True,
        agc: bool = False,
        delay_hint_ms: int = 0,
    ) -> None:
        if rate not in _APM_RATES:
            raise ValueError(f"AEC rate must be one of {sorted(_APM_RATES)}, not {rate}")
        from livekit import rtc

        self._rtc: Any = rtc
        self.rate = rate
        self.frame = rate // 100
        self.delay_hint_ms = int(delay_hint_ms)
        self._apm: Any = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=ns,
            high_pass_filter=hpf,
            auto_gain_control=agc,
        )
        self._rev = np.zeros(0, np.float32)
        self.frames_processed = 0
        self.reverse_frames = 0
        self.passthrough_samples = 0

    def _frame(self, pcm: np.ndarray) -> Any:
        return self._rtc.AudioFrame(pcm.tobytes(), self.rate, 1, self.frame)

    def feed_reference(self, block: F32) -> None:
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if self._rev.size:
            x = np.concatenate([self._rev, x])
        n = self.frame
        whole = x.size - x.size % n
        for i in range(0, whole, n):
            self._apm.process_reverse_stream(self._frame(_to_int16(x[i : i + n])))
            self.reverse_frames += 1
        self._rev = x[whole:].copy()

    def process(self, mic_block: F32) -> F32:
        x = np.asarray(mic_block, dtype=np.float32).reshape(-1)
        out = x.copy()
        n = self.frame
        whole = x.size - x.size % n
        for i in range(0, whole, n):
            frame = self._frame(_to_int16(x[i : i + n]))
            self._apm.set_stream_delay_ms(self.delay_hint_ms)  # required before each frame
            self._apm.process_stream(frame)  # in place
            out[i : i + n] = np.frombuffer(frame.data, dtype=np.int16) / np.float32(32768.0)
            self.frames_processed += 1
        self.passthrough_samples += x.size - whole
        return out


LiveKitAEC = WebRtcAEC


class NullAEC:
    """``EchoCanceller`` that changes nothing (headphones, tests)."""

    def feed_reference(self, block: F32) -> None:
        return None

    def process(self, mic_block: F32) -> F32:
        return np.asarray(mic_block, dtype=np.float32)


class EnergyDTD:
    """Geigel-style energy double-talk detector for the VAD probability.

    While she is audible, the mic RMS must exceed ``k × coupling × ref_rms`` for a frame to
    count as the streamer. ``coupling`` (speaker→mic gain) is learned from frames where only
    she is audible (``vad_p < learn_below``): a peak hold with a slow decay. ``ref_rms`` is the
    loudest reference block audible in the last ``window_s`` seconds (it covers the output,
    acoustic and input delays).
    """

    def __init__(
        self,
        k: float = 2.0,
        window_s: float = 0.35,
        *,
        learn_below: float = 0.2,
        decay: float = 0.995,
        min_ref_rms: float = 1e-3,
    ) -> None:
        if k <= 0 or window_s <= 0 or not 0 < decay <= 1:
            raise ValueError("k and window_s must be positive, decay in (0, 1]")
        self.k = k
        self.window_s = window_s
        self.learn_below = learn_below
        self.decay = decay
        self.min_ref_rms = min_ref_rms
        self._coupling = 0.0
        self.gated = 0

    @property
    def coupling(self) -> float:
        return self._coupling

    def reset(self) -> None:
        self._coupling = 0.0
        self.gated = 0

    def gate(self, mic_rms: float, ref_rms: float, vad_p: float) -> float:
        """Return ``vad_p``, or 0.0 when the mic energy is explained by her echo."""
        if ref_rms <= self.min_ref_rms:
            return vad_p
        if vad_p < self.learn_below:
            self._coupling = max(self._coupling * self.decay, mic_rms / ref_rms)
        if mic_rms < self.k * self._coupling * ref_rms:
            self.gated += 1
            return 0.0
        return vad_p


class HalfDuplexGate:
    """Ignore the mic while she is audible (``echo_mode = "half_duplex"``)."""

    def __init__(self) -> None:
        self.gated = 0

    def gate(self, ai_speaking: bool, vad_p: float) -> float:
        if ai_speaking:
            self.gated += 1
            return 0.0
        return vad_p


def make_echo_canceller(
    mode: EchoMode,
    headphones: bool,
    *,
    rate: int = 48000,
    delay_hint_ms: int = 0,
) -> tuple[EchoCanceller | None, EchoMode]:
    """Build the canceller for ``mode`` and return it with the mode actually in effect.

    ``auto`` means ``none`` with headphones and ``aec`` with speakers. ``aec`` falls back to
    ``energy_dtd`` when livekit cannot be imported or fails to start. ``energy_dtd``,
    ``half_duplex`` and ``none`` need no canceller (the front-end gates the VAD instead).
    """
    effective: EchoMode = ("none" if headphones else "aec") if mode == "auto" else mode
    if effective != "aec":
        return None, effective
    try:
        return WebRtcAEC(rate, delay_hint_ms=delay_hint_ms), "aec"
    except ImportError as exc:
        log.warning("livekit is not installed (%s); echo mode falls back to energy_dtd", exc)
    except Exception:
        log.exception("WebRTC AEC failed to start; echo mode falls back to energy_dtd")
    return None, "energy_dtd"
