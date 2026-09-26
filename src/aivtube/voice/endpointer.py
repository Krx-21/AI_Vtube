"""Silence-based endpointing over per-frame VAD probabilities (ARCHITECTURE.md §3.4, §4.7, §5).

``SileroEndpointer`` turns 32 ms VAD frames into ``VadStart`` / ``VadPartial`` / ``VadEnd``:

- **Start**: ``min_speech_ms`` of consecutive frames at or above ``threshold`` (or the stricter
  ``barge_threshold`` while she is speaking). ``VadStart.t`` is the onset: the capture time of
  the first speech frame. ``barge`` tells whether she was speaking.
- **Continue**: a frame counts as silence only below ``neg_threshold`` (hysteresis).
- **End**: ``end_silence_ms`` of silence. ``VadEnd.t`` is the end of the last speech frame (the
  streamer's last phoneme, where the §5 latency budget starts). The audio holds
  ``preroll_ms`` from before the onset, the speech, and ``tail_pad_ms`` of the trailing silence.
- **Forced split**: a segment that reaches ``max_segment_s`` is emitted as ``VadPartial`` at
  the quietest frame of its last ``split_search_ms`` (a pause if there is one). The turn goes on;
  the partials plus the final ``VadEnd`` audio are contiguous.
- **Max turn**: ``max_turn_s`` after the onset the turn ends with ``VadEnd`` even mid-speech.
- **M2 particle endpointing** (``particle_endpointing``): a tail hint ending in a Thai final
  particle shortens the end silence to ``final_particle_ms``; a dangling conjunction extends it
  to ``continuation_ms``. Otherwise ``set_tail_hint`` is ignored.

It is a plain state machine with no clock or threads: the front-end calls it on the mic thread.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from aivtube.contracts.voice import (
    F32,
    EndpointerConfig,
    VadEnd,
    VadEvent,
    VadPartial,
    VadStart,
    VoiceActivityDetector,
)

__all__ = ["CONTINUATION_TAILS", "FINAL_PARTICLES", "SileroEndpointer"]

# Provisional lists for the M2 heuristic; to be calibrated on the streamer's speech (spike S7).
FINAL_PARTICLES: Final[tuple[str, ...]] = (
    "ครับ",
    "ค่ะ",
    "คะ",
    "นะ",
    "น่ะ",
    "จ้ะ",
    "จ้า",
    "ไหม",
    "มั้ย",
    "เหรอ",
    "หรอ",
    "สิ",
    "ล่ะ",
    "เลย",
    "แล้ว",
)
CONTINUATION_TAILS: Final[tuple[str, ...]] = (
    "และ",
    "แล้วก็",
    "แต่",
    "ก็",
    "ที่",
    "ว่า",
    "เพราะ",
    "ถ้า",
    "คือ",
    "หรือ",
    "กับ",
    "เช่น",
)
_EPS: Final = 1e-9
_TRAILING: Final = " \t\r\n.,!?…~ๆ\"'"


@dataclass(slots=True)
class _Seg:
    frames: list[F32]
    probs: list[float]


class SileroEndpointer:
    """``Endpointer`` over any ``VoiceActivityDetector`` (Silero by default).

    ``push`` runs the VAD on the frame; ``push_prob`` takes a probability the caller already
    computed (the front-end uses it to apply echo gates to the VAD output). ``flush`` ends a
    turn in progress at once (push-to-talk release).
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        cfg: EndpointerConfig,
        *,
        frame_ms: int = 32,
        tail_pad_ms: int = 200,
        split_search_ms: int = 1000,
    ) -> None:
        vad_ms = 1000.0 * vad.frame_samples / vad.sample_rate
        if abs(vad_ms - frame_ms) > 0.5:
            raise ValueError(f"frame_ms={frame_ms} but the VAD uses {vad_ms:g} ms frames")
        self.vad = vad
        self._n = int(vad.frame_samples)
        self._sr = int(vad.sample_rate)
        self._frame_s = self._n / self._sr
        self._frame_ms = 1000.0 * self._frame_s
        self._tail_pad = max(0, round(tail_pad_ms * self._sr / 1000))
        self._split_search = max(1, math.ceil(split_search_ms / self._frame_ms - _EPS))
        self._cfg = cfg
        self._state: Literal["idle", "speech"] = "idle"
        self._ring: collections.deque[tuple[F32, float]] = collections.deque()
        self._run = 0
        self._run_start_t = 0.0
        self._prefix: F32 = np.zeros(0, np.float32)
        self._seg = _Seg([], [])
        self._turn_start_t = 0.0
        self._seg_start_t = 0.0
        self._silence_run = 0
        self._last_speech_end_t = 0.0
        self._last_speech_idx = -1
        self._hint = ""
        self.set_config(cfg)

    # --- configuration --------------------------------------------------------------------------
    @property
    def config(self) -> EndpointerConfig:
        return self._cfg

    def set_config(self, cfg: EndpointerConfig) -> None:
        """Swap thresholds and timings (e.g. an echo-mode barge threshold); keeps the state."""
        if not 0.0 <= cfg.neg_threshold <= cfg.threshold <= 1.0:
            raise ValueError("need 0 <= neg_threshold <= threshold <= 1")
        if not 0.0 < cfg.barge_threshold <= 1.0:
            raise ValueError("barge_threshold must be in (0, 1]")
        if cfg.end_silence_ms <= 0 or cfg.max_segment_s <= 0 or cfg.max_turn_s <= 0:
            raise ValueError("end_silence_ms, max_segment_s and max_turn_s must be positive")
        if cfg.min_speech_ms < 0 or cfg.preroll_ms < 0:
            raise ValueError("min_speech_ms and preroll_ms must be >= 0")
        self._cfg = cfg
        self._min_speech = max(1, math.ceil(cfg.min_speech_ms / self._frame_ms - _EPS))
        self._preroll = round(cfg.preroll_ms * self._sr / 1000)
        keep = math.ceil(cfg.preroll_ms / self._frame_ms - _EPS) + self._min_speech + 1
        if self._ring.maxlen != keep:
            self._ring = collections.deque(self._ring, maxlen=keep)

    # --- state ----------------------------------------------------------------------------------
    @property
    def in_speech(self) -> bool:
        return self._state == "speech"

    @property
    def tail_hint(self) -> str:
        return self._hint

    def set_tail_hint(self, text: str) -> None:
        """M2: the latest partial text of the turn; ignored unless particle endpointing is on."""
        if self._cfg.particle_endpointing:
            self._hint = text

    def reset(self) -> None:
        """Forget everything (no events), including the VAD's recurrent state."""
        self.vad.reset()
        self._state = "idle"
        self._ring.clear()
        self._run = 0
        self._clear_turn()

    def flush(self, t: float) -> list[VadEvent]:
        """End the turn in progress now (push-to-talk released); ``[]`` when idle."""
        if self._state != "speech":
            self._ring.clear()
            self._run = 0
            return []
        end_t = self._last_speech_end_t if self._last_speech_idx >= 0 else t
        return [self._finish(end_t, trim=True, seed_ring=False)]

    # --- frames ---------------------------------------------------------------------------------
    def push(self, frame16k: F32, t: float, ai_speaking: bool) -> list[VadEvent]:
        """Feed one frame (``t`` = capture time of its first sample)."""
        x = self._check(frame16k)
        return self._step(x, self.vad.prob(x), t, ai_speaking)

    def push_prob(self, frame16k: F32, p: float, t: float, ai_speaking: bool) -> list[VadEvent]:
        """Like ``push`` with a VAD probability computed (and possibly gated) by the caller."""
        return self._step(self._check(frame16k), float(p), t, ai_speaking)

    def _check(self, frame: F32) -> F32:
        x = np.array(frame, dtype=np.float32).reshape(-1)  # copy: callers reuse buffers
        if x.size != self._n:
            raise ValueError(f"endpointer frames are {self._n} samples, got {x.size}")
        return x

    def _step(self, x: F32, p: float, t: float, ai_speaking: bool) -> list[VadEvent]:
        cfg = self._cfg
        if self._state == "idle":
            on = cfg.barge_threshold if ai_speaking else cfg.threshold
            self._ring.append((x, p))
            if p >= on:
                if self._run == 0:
                    self._run_start_t = t
                self._run += 1
            else:
                self._run = 0
            if self._run >= self._min_speech:
                self._begin()
                return [VadStart(t=self._turn_start_t, barge=ai_speaking)]
            return []

        seg = self._seg
        seg.frames.append(x)
        seg.probs.append(p)
        t_end = t + self._frame_s
        if p < cfg.neg_threshold:
            self._silence_run += 1
        else:
            self._silence_run = 0
            self._last_speech_end_t = t_end
            self._last_speech_idx = len(seg.frames) - 1
        if self._silence_run >= self._end_frames():
            return [self._finish(self._last_speech_end_t, trim=True, seed_ring=True)]
        if t_end - self._turn_start_t >= cfg.max_turn_s - _EPS:
            return [self._finish(t_end, trim=False, seed_ring=False)]
        if self._silence_run == 0 and t_end - self._seg_start_t >= cfg.max_segment_s - _EPS:
            return [self._split(t_end)]
        return []

    # --- transitions ----------------------------------------------------------------------------
    def _begin(self) -> None:
        items = list(self._ring)
        run = self._run
        before, speech = items[:-run], items[-run:]
        prefix = np.zeros(0, np.float32)
        if self._preroll and before:
            prefix = np.concatenate([f for f, _ in before])[-self._preroll :]
        self._prefix = prefix
        self._seg = _Seg([f for f, _ in speech], [p for _, p in speech])
        self._turn_start_t = self._seg_start_t = self._run_start_t
        self._silence_run = 0
        self._last_speech_idx = len(speech) - 1
        self._last_speech_end_t = self._run_start_t + run * self._frame_s
        self._ring.clear()
        self._run = 0
        self._hint = ""
        self._state = "speech"

    def _finish(self, t: float, *, trim: bool, seed_ring: bool) -> VadEnd:
        seg = self._seg
        parts: list[F32] = [self._prefix]
        if trim:
            keep = self._last_speech_idx + 1
            parts.extend(seg.frames[:keep])
            if self._tail_pad and keep < len(seg.frames):
                parts.append(np.concatenate(seg.frames[keep:])[: self._tail_pad])
        else:
            parts.extend(seg.frames)
        audio = np.concatenate(parts).astype(np.float32, copy=False)
        self._state = "idle"
        self._ring.clear()
        if seed_ring:  # trailing silence becomes pre-roll material for a quick follow-up
            tail = list(zip(seg.frames, seg.probs, strict=True))
            for item in tail[-(self._ring.maxlen or 1) :]:
                self._ring.append(item)
        self._run = 0
        self._clear_turn()
        return VadEnd(t=t, audio=audio)

    def _split(self, t_end: float) -> VadPartial:
        seg = self._seg
        n = len(seg.frames)
        lo = max(0, n - self._split_search)
        window = np.asarray(seg.probs[lo:], dtype=np.float64)
        # split after the quietest frame; on ties the latest one, so steady speech splits at
        # the limit itself
        k = lo + (window.size - 1 - int(np.argmin(window[::-1])))
        audio = np.concatenate([self._prefix, *seg.frames[: k + 1]]).astype(np.float32)
        t_split = t_end - (n - 1 - k) * self._frame_s
        self._prefix = np.zeros(0, np.float32)
        self._seg = _Seg(seg.frames[k + 1 :], seg.probs[k + 1 :])
        self._seg_start_t = t_split
        self._last_speech_idx -= k + 1
        return VadPartial(t=t_split, audio=audio)

    def _clear_turn(self) -> None:
        self._prefix = np.zeros(0, np.float32)
        self._seg = _Seg([], [])
        self._silence_run = 0
        self._last_speech_idx = -1
        self._hint = ""

    def _end_frames(self) -> int:
        cfg = self._cfg
        ms = float(cfg.end_silence_ms)
        if cfg.particle_endpointing and self._hint:
            tail = self._hint.rstrip(_TRAILING)
            if tail.endswith(CONTINUATION_TAILS):
                ms = max(ms, float(cfg.continuation_ms))
            elif tail.endswith(FINAL_PARTICLES):
                ms = float(cfg.final_particle_ms)
        return max(1, math.ceil(ms / self._frame_ms - _EPS))
