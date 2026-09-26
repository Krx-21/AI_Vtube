"""``FakeSpeechOutput``: an in-process voice worker with an audible timeline (§3.5, §10)."""

from __future__ import annotations

import asyncio
import collections
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aivtube.contracts.events import SegmentDone, SegmentStarted, UtteranceDone
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.speech import StopMode, TTSConstraints, VoicePolicy
from aivtube.contracts.types import Segment

__all__ = ["DEFAULT_CANNED", "FakeSpeechOutput"]

DEFAULT_CANNED: Mapping[str, str] = {
    "filtered": "Filtered.",
    "filler": "อืม…",
    "brain_freeze": "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ",
    "thanks": "ขอบคุณมากนะคะ",
}


@dataclass
class _Utt:
    utt_id: str
    character: str
    gate_open: bool
    filler_after_s: float | None
    began: float
    segments: collections.deque[Segment] = field(default_factory=collections.deque)
    heard: list[str] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    cut: asyncio.Event = field(default_factory=asyncio.Event)
    last_seen: bool = False
    filtered: bool = False
    stop_after_segment: bool = False
    stop_reason: str | None = None
    playing: bool = False
    started_any: bool = False
    filler_played: bool = False
    done: bool = False
    canned: str | None = None  # set for play_canned pseudo-utterances (no events)


class FakeSpeechOutput:
    """``SpeechOutput`` that "plays" segments in fake or real time and publishes the results.

    Each segment lasts ``len(text) / chars_per_s`` seconds (scaled by ``set_voice_rate``), paced
    by ``clock.sleep``. It publishes ``SegmentStarted``/``SegmentDone``/``UtteranceDone`` with the
    utterance's character, and appends ``(t_audible, text)`` to ``audible`` for everything heard
    (segments, canned phrases and fillers). Utterances play one after another in ``begin`` order;
    at most ``max_queued`` segments may wait per utterance (``segment`` returns ``False`` beyond
    that). ``stop(None, ...)`` stops every open utterance. ``calls`` logs every method call.
    ``UtteranceDone.heard_text`` is the plain concatenation of the segments' heard texts (the
    chunker is lossless, so segments carry their own spacing); a cut segment is truncated at
    the last space it reached.
    """

    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        *,
        chars_per_s: float = 12.5,
        constraints: TTSConstraints | None = None,
        max_queued: int = 8,
        backend: str = "fake",
        canned: Mapping[str, str] | None = None,
    ) -> None:
        self.bus = bus
        self.clock = clock
        self.chars_per_s = chars_per_s
        self.max_queued = max_queued
        self.backend = backend
        self.canned = dict(DEFAULT_CANNED if canned is None else canned)
        self._constraints = constraints or TTSConstraints(
            first_min_chars=8, min_chars=40, max_chars=160, backend=backend, identity="fake"
        )
        self.audible: list[tuple[float, str]] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.dropped: list[Segment] = []
        self.policy = VoicePolicy()
        self.rates: dict[str, int] = {}
        self.gain = 1.0
        self.muted = False
        self.is_ready = True
        self._utts: collections.deque[_Utt] = collections.deque()
        self._by_id: dict[str, _Utt] = {}
        self._last_stopped: dict[str, _Utt] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # --- SpeechOutput protocol --------------------------------------------------------------
    def ready(self) -> bool:
        return self.is_ready

    def constraints(self, character: str) -> TTSConstraints:
        return self._constraints

    async def begin(
        self,
        utt_id: str,
        character: str,
        *,
        filler_after_s: float | None = None,
        gate_open: bool = True,
    ) -> None:
        self.calls.append(("begin", (utt_id, character, filler_after_s, gate_open)))
        if utt_id in self._by_id:
            raise ValueError(f"utterance {utt_id!r} already begun")
        utt = _Utt(utt_id, character, gate_open, filler_after_s, self.clock.now())
        self._by_id[utt_id] = utt
        self._utts.append(utt)
        self._ensure_player()
        self._wake.set()

    async def segment(self, seg: Segment) -> bool:
        self.calls.append(("segment", (seg,)))
        utt = self._by_id.get(seg.utt_id)
        if utt is None or utt.done or utt.stop_after_segment or utt.last_seen:
            self.dropped.append(seg)
            return True  # accepted and dropped, like the worker for a stopped utterance
        if len(utt.segments) >= self.max_queued:
            return False
        utt.segments.append(seg)
        if seg.last:
            utt.last_seen = True
        utt.wake.set()
        return True

    async def open_gate(self, utt_id: str) -> None:
        self.calls.append(("open_gate", (utt_id,)))
        utt = self._by_id.get(utt_id)
        if utt is not None:
            utt.gate_open = True
            utt.wake.set()

    async def stop(
        self, utt_id: str | None, mode: StopMode, reason: str, fade_ms: int = 30
    ) -> None:
        self.calls.append(("stop", (utt_id, mode, reason, fade_ms)))
        targets = [u for u in self._utts if not u.done]
        if utt_id is not None:
            targets = [u for u in targets if u.utt_id == utt_id]
        for utt in targets:
            if utt.canned is not None:  # canned phrases stop too (FREEZE silences everything)
                utt.cut.set()
                if utt is not self._utts[0]:
                    utt.done = True
                continue
            self._last_stopped[utt.character] = utt
            utt.stop_reason = reason
            utt.segments.clear()
            if utt.playing:
                if mode == "now":
                    utt.cut.set()
                else:
                    utt.stop_after_segment = True
            else:
                self._finish(utt, cancelled=True, reason=reason)
            utt.wake.set()
        self._wake.set()

    async def duck(self, gain: float, ramp_ms: int = 30) -> None:
        self.calls.append(("duck", (gain, ramp_ms)))
        self.gain = gain

    async def mute(self, on: bool) -> None:
        self.calls.append(("mute", (on,)))
        self.muted = on

    async def set_policy(self, policy: VoicePolicy) -> None:
        self.calls.append(("set_policy", (policy,)))
        self.policy = policy

    async def set_voice_rate(self, character: str, percent: int) -> None:
        self.calls.append(("set_voice_rate", (character, percent)))
        self.rates[character] = percent

    async def play_canned(self, key: str, character: str) -> None:
        self.calls.append(("play_canned", (key, character)))
        if key == "filtered":  # the reply pipeline stops the utterance, then plays this
            stopped = self._last_stopped.get(character)
            if stopped is not None and not stopped.done:
                stopped.filtered = True
            else:
                for u in reversed(self._utts):
                    if u.character == character and u.canned is None and not u.done:
                        u.filtered = True
                        break
        pseudo = _Utt(f"canned:{key}:{len(self.calls)}", character, True, None, self.clock.now())
        pseudo.canned = self.canned.get(key, key)
        self._utts.append(pseudo)
        self._ensure_player()
        self._wake.set()

    # --- test helpers -----------------------------------------------------------------------
    def set_ready(self, ready: bool) -> None:
        self.is_ready = ready

    def heard_texts(self) -> list[str]:
        return [text for _, text in self.audible]

    @property
    def idle(self) -> bool:
        return not any(not u.done for u in self._utts)

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # --- playback ---------------------------------------------------------------------------
    def _ensure_player(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._player(), name="fake-speech-player"
            )

    def _cps(self, character: str) -> float:
        return self.chars_per_s * (1.0 + self.rates.get(character, 0) / 100.0)

    async def _player(self) -> None:
        while True:
            while not self._utts:
                self._wake.clear()
                await self._wake.wait()
            utt = self._utts[0]
            if not utt.done:
                if utt.canned is not None:
                    await self._play_canned(utt)
                else:
                    await self._play_utt(utt)
            if self._utts and self._utts[0] is utt:
                self._utts.popleft()
            self._by_id.pop(utt.utt_id, None)

    async def _play_canned(self, utt: _Utt) -> None:
        text = utt.canned or ""
        if not self.muted:
            self.audible.append((self.clock.now(), text))
        await self._wait(utt.cut, len(text) / self._cps(utt.character))
        utt.done = True

    async def _play_utt(self, utt: _Utt) -> None:
        while not utt.done:
            if utt.stop_after_segment or utt.cut.is_set():
                self._finish(utt, cancelled=True, reason=utt.stop_reason)
                return
            if not utt.gate_open or not utt.segments:
                if utt.segments or not utt.last_seen:
                    timeout = self._filler_timeout(utt)
                    utt.wake.clear()
                    woke = await self._wait(utt.wake, timeout)
                    if not woke and timeout is not None:
                        self._play_filler(utt)
                    continue
                self._finish(utt, cancelled=False, reason=None)
                return
            seg = utt.segments.popleft()
            await self._play_segment(utt, seg)
            if seg.kind == "filtered":
                utt.filtered = True
            if utt.cut.is_set():
                self._finish(utt, cancelled=True, reason=utt.stop_reason)
                return
            if seg.last:
                self._finish(utt, cancelled=utt.stop_after_segment, reason=utt.stop_reason)
                return

    def _filler_timeout(self, utt: _Utt) -> float | None:
        if utt.filler_after_s is None or utt.started_any or utt.filler_played:
            return None
        return max(0.0, utt.began + utt.filler_after_s - self.clock.now())

    def _play_filler(self, utt: _Utt) -> None:
        utt.filler_played = True
        if not self.muted:
            self.audible.append((self.clock.now(), self.canned.get("filler", "อืม…")))

    async def _play_segment(self, utt: _Utt, seg: Segment) -> None:
        cps = self._cps(utt.character)
        duration = len(seg.text) / cps if cps > 0 else 0.0
        silent = self.muted
        t0 = self.clock.now()
        utt.playing = True
        utt.started_any = True
        self.bus.publish(
            SegmentStarted(
                ts=t0,
                character=utt.character,
                utt_id=seg.utt_id,
                seq=seg.seq,
                t_audible=t0,
                duration_s=duration,
                backend=self.backend,
                silent=silent,
                caption=seg.caption,
                emotion=seg.emotion,
            )
        )
        if not silent:
            self.audible.append((t0, seg.text))
        completed = await self._wait(utt.cut, duration) is False
        utt.playing = False
        if completed:
            heard_text = seg.text
        else:
            n = int((self.clock.now() - t0) * cps)
            heard_text = seg.text[:n]
            if n < len(seg.text) and " " in heard_text:
                heard_text = heard_text[: heard_text.rfind(" ")]
            heard_text = heard_text.rstrip()
        if silent:
            heard_text = ""
        utt.heard.append(heard_text)
        self.bus.publish(
            SegmentDone(
                ts=self.clock.now(),
                character=utt.character,
                utt_id=seg.utt_id,
                seq=seg.seq,
                heard=completed and not silent,
                heard_text=heard_text,
            )
        )

    def _finish(self, utt: _Utt, *, cancelled: bool, reason: str | None) -> None:
        if utt.done:
            return
        utt.done = True
        self.bus.publish(
            UtteranceDone(
                ts=self.clock.now(),
                character=utt.character,
                utt_id=utt.utt_id,
                heard_text="".join(utt.heard),  # lossless: segments carry their own spacing
                cancelled=cancelled,
                reason=reason if cancelled else None,
                filtered=utt.filtered,
            )
        )

    async def _wait(self, event: asyncio.Event, delay: float | None) -> bool:
        """Wait for ``event`` or ``delay`` clock seconds; ``True`` if the event won."""
        if event.is_set():
            return True
        waiter = asyncio.ensure_future(event.wait())
        tasks: set[asyncio.Future[Any]] = {waiter}
        if delay is not None:
            tasks.add(asyncio.ensure_future(self.clock.sleep(delay)))
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
        return waiter in done
