"""Mic front-end: [AEC] → 48k→16k → 512-sample frames → VAD → endpointer → events.

ARCHITECTURE.md §2.4 (the mic thread), §4.7 (echo handling, barge-in), §3.4. ``VoiceFrontEnd``
runs on the mic thread (``MicCapture``'s consumer calls ``feed``). It:

1. feeds the player's newly audible post-gain blocks to the echo canceller, then cancels the
   mic block (the reference always precedes the echo it explains);
2. resamples to 16 kHz with a streaming soxr resampler and keeps the capture-time mapping, so
   every VAD frame gets the perf_counter time of its first sample;
3. keeps a 1 s ring of 16 kHz audio for the barge-in quick decode (``recent_audio``);
4. gates the VAD probability while she is audible (``energy_dtd`` / ``half_duplex``) and runs
   the endpointer, which uses its barge threshold while she speaks;
5. hands each ``VadEvent`` to ``on_event`` on the mic thread (``post_to_loop`` bridges to
   asyncio with ``call_soon_threadsafe``).

Listening controls (``set_enabled``, ``set_ptt``, ``set_mic_mode``, ``apply_policy``) may be
called from any thread; they take effect at the next ``feed`` on the mic thread. While not
listening no events are emitted; releasing push-to-talk ends a turn in progress at once.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any, Final, TypeVar

import numpy as np

from aivtube.contracts.speech import MicMode, VoicePolicy
from aivtube.contracts.voice import F32, AudioOut, EchoCanceller, Endpointer, VadEvent
from aivtube.voice.aec import EnergyDTD, HalfDuplexGate
from aivtube.voice.endpointer import SileroEndpointer

__all__ = ["VAD_RATE", "VoiceFrontEnd", "post_to_loop"]

log = logging.getLogger("aivtube.voice.frontend")

VAD_RATE: Final = 16000
_FRAME: Final = 512
_MAX_REF: Final = 100  # reference blocks fed per mic block at most (1 s of 10 ms blocks)

E = TypeVar("E")


def post_to_loop(
    loop: asyncio.AbstractEventLoop, handler: Callable[[E], object]
) -> Callable[[E], None]:
    """Wrap ``handler`` so another thread can call it: it runs on ``loop``.

    Events posted after the loop has closed are dropped quietly.
    """

    def post(item: E) -> None:
        try:
            loop.call_soon_threadsafe(handler, item)
        except RuntimeError:  # loop closed during shutdown
            log.debug("dropped %r: event loop closed", item)

    return post


class VoiceFrontEnd:
    """The mic pipeline (see the module docstring). Not thread-safe except where noted."""

    def __init__(
        self,
        endpointer: Endpointer,
        *,
        player: AudioOut | None,
        aec: EchoCanceller | None,
        dtd: EnergyDTD | None,
        in_rate: int = 48000,
        on_event: Callable[[VadEvent], None],
        half_duplex: bool = False,
        ring_s: float = 1.0,
        speaking_tail_s: float = 0.25,
    ) -> None:
        if in_rate <= 0:
            raise ValueError("in_rate must be positive")
        self._ep = endpointer
        self._player = player
        self._on_event = on_event
        self._in_rate = int(in_rate)
        self._ratio = self._in_rate / VAD_RATE
        self._tail_s = speaking_tail_s
        self._rs: Any = None
        if self._in_rate != VAD_RATE:
            import soxr

            self._rs = soxr.ResampleStream(
                self._in_rate, VAD_RATE, 1, dtype="float32", quality="HQ"
            )
        self._pending: F32 = np.zeros(0, np.float32)
        self._n_in = 0  # input samples fed to the resampler
        self._n16 = 0  # 16 kHz index of the first sample in _pending (after the resampler)
        # echo handling (swapped atomically by configure_echo, applied on the mic thread)
        self._aec: EchoCanceller | None = aec
        self._dtd: EnergyDTD | None = dtd
        self._half: HalfDuplexGate | None = HalfDuplexGate() if half_duplex else None
        self._echo_update: tuple[EchoCanceller | None, EnergyDTD | None, bool] | None = None
        self._ref_last: F32 | None = None
        self._ref_rs: Any = None
        self._aec_failed = False
        # listening state: requested (any thread, under _ctl) vs applied (mic thread)
        self._ctl = threading.Lock()
        self._want_enabled = True
        self._want_ptt_mode = False
        self._want_ptt_held = False
        self._listening = True
        # 16 kHz ring for the barge-in quick decode
        self._ring = np.zeros(max(_FRAME, round(ring_s * VAD_RATE)), np.float32)
        self._ring_pos = 0
        self._ring_fill = 0
        self._ring_lock = threading.Lock()
        self._errors_logged = 0
        self.stats: Counter[str] = Counter()

    # --- controls (any thread) ----------------------------------------------------------------
    @property
    def listening(self) -> bool:
        """Whether VAD events are being produced (as last applied on the mic thread)."""
        return self._listening

    def set_enabled(self, enabled: bool) -> None:
        """Listen or not (``deafened`` / ``listening=False``); a turn in progress is dropped."""
        with self._ctl:
            self._want_enabled = bool(enabled)

    def set_ptt(self, held: bool) -> None:
        """Push-to-talk key state; only matters in ``ptt`` mode. Release ends the turn."""
        with self._ctl:
            self._want_ptt_held = bool(held)

    def set_mic_mode(self, mode: MicMode) -> None:
        with self._ctl:
            self._want_enabled = mode != "deafened"
            self._want_ptt_mode = mode == "ptt"

    def apply_policy(self, policy: VoicePolicy) -> None:
        """Apply the listening part of a ``VoicePolicy`` (mic mode, PTT, listening)."""
        with self._ctl:
            self._want_ptt_mode = policy.mic_mode == "ptt"
            self._want_ptt_held = policy.ptt_active
            self._want_enabled = policy.listening and policy.mic_mode != "deafened"

    def configure_echo(
        self, *, aec: EchoCanceller | None, dtd: EnergyDTD | None, half_duplex: bool
    ) -> None:
        """Swap the echo handling (takes effect at the next ``feed``)."""
        with self._ctl:
            self._echo_update = (aec, dtd, half_duplex)

    def recent_audio(self, seconds: float) -> F32:
        """The last ``seconds`` (at most the ring size) of 16 kHz post-AEC audio; any thread."""
        with self._ring_lock:
            n = min(max(0, round(seconds * VAD_RATE)), self._ring_fill)
            if n == 0:
                return np.zeros(0, np.float32)
            end = self._ring_pos
            start = end - n
            if start >= 0:
                return self._ring[start:end].copy()
            return np.concatenate([self._ring[start:], self._ring[:end]])

    # --- the mic thread ---------------------------------------------------------------------------
    def feed(self, frame48k: F32, t_capture: float) -> None:
        """Process one mic block (any length) captured at ``t_capture`` (first sample)."""
        x = np.asarray(frame48k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        self._apply_controls(t_capture)
        self.stats["blocks"] += 1
        if self._aec is not None and not self._aec_failed:
            x = self._cancel_echo(x)
        n0 = self._n_in
        self._n_in += x.size
        if self._rs is not None:
            y = np.asarray(self._rs.resample_chunk(x), dtype=np.float32).reshape(-1)
        else:
            y = x
        if y.size:
            self._ring_write(y)
        buf = np.concatenate([self._pending, y]) if self._pending.size else y
        k = 0
        speaking: bool | None = None
        while buf.size - k >= _FRAME:
            frame = buf[k : k + _FRAME]
            t = t_capture + ((self._n16 + k) * self._ratio - n0) / self._in_rate
            k += _FRAME
            if not self._listening:
                continue
            if speaking is None:
                speaking = self._ai_speaking()
            self._frame(frame, t, speaking)
        self._n16 += k
        self._pending = buf[k:].copy() if k else buf.copy()

    def _apply_controls(self, t: float) -> None:
        with self._ctl:  # one consistent snapshot of what the other threads asked for
            update, self._echo_update = self._echo_update, None
            enabled, ptt_mode = self._want_enabled, self._want_ptt_mode
            want = enabled and (not ptt_mode or self._want_ptt_held)
        if update is not None:
            self._aec, self._dtd, half = update
            self._half = HalfDuplexGate() if half else None
            self._aec_failed = False
            self._ref_last = None
        if want == self._listening:
            return
        self._listening = want
        if want:
            self._ep.reset()  # start clean: no stale VAD state from before
            return
        if enabled and ptt_mode and isinstance(self._ep, SileroEndpointer):
            for ev in self._ep.flush(t):  # PTT released mid-turn: end it now
                self._emit(ev)
        self._ep.reset()

    def _ai_speaking(self) -> bool:
        return self._player is not None and self._player.is_speaking(self._tail_s)

    def _frame(self, frame: F32, t: float, speaking: bool) -> None:
        self.stats["frames"] += 1
        ep = self._ep
        gate = speaking and (self._half is not None or self._dtd is not None)
        try:
            if gate and isinstance(ep, SileroEndpointer):
                p = ep.vad.prob(frame)
                if self._half is not None:
                    p = self._half.gate(True, p)
                elif self._dtd is not None:
                    mic_rms = math.sqrt(float(np.dot(frame, frame)) / frame.size)
                    ref_rms = self._ref_rms_max(t - self._dtd.window_s, t + _FRAME / VAD_RATE)
                    p = self._dtd.gate(mic_rms, ref_rms, p)
                if p == 0.0:
                    self.stats["gated"] += 1
                events = ep.push_prob(frame, p, t, speaking)
            else:
                events = ep.push(frame, t, speaking)
        except Exception:
            self._log_error("VAD/endpointer failed")
            return
        for ev in events:
            self._emit(ev)

    def _emit(self, ev: VadEvent) -> None:
        self.stats[type(ev).__name__] += 1
        try:
            self._on_event(ev)
        except Exception:
            self._log_error("VAD event handler failed")

    def _log_error(self, what: str) -> None:
        self.stats["errors"] += 1
        if self._errors_logged < 3 or self.stats["errors"] % 1000 == 0:
            self._errors_logged += 1
            log.exception("%s (%d so far)", what, self.stats["errors"])

    # --- echo -------------------------------------------------------------------------------------
    def _new_reference(self) -> list[F32]:
        """Player blocks appended since the last call, oldest first (at most ``_MAX_REF``).

        Blocks are matched by identity, not time: ``AudioOut`` does not promise strictly
        increasing reference timestamps.
        """
        player = self._player
        if player is None:
            return []
        snap = tuple(player.reference)  # one atomic copy: the audio thread keeps appending
        last = self._ref_last
        fresh: list[F32] = []
        for _, blk in reversed(snap):
            if blk is last or len(fresh) >= _MAX_REF:
                break
            fresh.append(blk)
        if fresh:
            self._ref_last = snap[-1][1]
            fresh.reverse()
        return fresh

    def _cancel_echo(self, x: F32) -> F32:
        aec = self._aec
        assert aec is not None
        try:
            ref = self._new_reference()
            if ref and self._player is not None and self._player.sample_rate != self._in_rate:
                ref = [self._resample_reference(np.concatenate(ref))]
            for blk in ref:
                aec.feed_reference(blk)
            return np.asarray(aec.process(x), dtype=np.float32).reshape(-1)
        except Exception:
            self._aec_failed = True
            self.stats["aec_errors"] += 1
            log.exception("echo canceller failed; continuing without AEC")
            return x

    def _resample_reference(self, ref: F32) -> F32:
        assert self._player is not None
        if self._ref_rs is None:
            import soxr

            self._ref_rs = soxr.ResampleStream(
                self._player.sample_rate, self._in_rate, 1, dtype="float32", quality="HQ"
            )
        return np.asarray(self._ref_rs.resample_chunk(ref), dtype=np.float32).reshape(-1)

    def _ref_rms_max(self, t0: float, t1: float) -> float:
        """Loudest reference block audible within ``[t0, t1]``."""
        player = self._player
        if player is None:
            return 0.0
        best = 0.0
        for t, blk in reversed(tuple(player.reference)):
            if t1 < t:
                continue
            if blk.size:
                block_s = blk.size / player.sample_rate
                if t + block_s < t0:
                    break
                best = max(best, math.sqrt(float(np.dot(blk, blk)) / blk.size))
        return best

    # --- ring -------------------------------------------------------------------------------------
    def _ring_write(self, y: F32) -> None:
        ring = self._ring
        size = ring.size
        if y.size >= size:
            y = y[-size:]
        with self._ring_lock:
            pos = self._ring_pos
            first = min(y.size, size - pos)
            ring[pos : pos + first] = y[:first]
            if first < y.size:
                ring[: y.size - first] = y[first:]
            self._ring_pos = (pos + y.size) % size
            self._ring_fill = min(size, self._ring_fill + y.size)
