"""``ConsoleSpeechOutput``: text mode's "voice" (§3.5, §9 ``run --text``).

It prints each segment's caption when it "starts" and keeps it "audible" for
``len(text) / chars_per_s`` seconds of clock time, publishing the same events as the voice
worker path: ``SegmentStarted`` / ``SegmentDone`` / ``UtteranceDone`` with character and turn.
Semantics follow the worker: utterances play in ``begin`` order with canned phrases queued
between them, at most ``max_queued`` segments wait per utterance (``segment`` returns
``False`` beyond that), a closed gate holds playback, ``stop(now)`` cuts the current segment
(heard text cut back to the last whole word), ``stop(after_segment)`` lets it finish, MUTE
prints nothing and reports silent segments. A filler (``อืม…``) is printed when an utterance
that asked for one has nothing to say ``filler_after_s`` after ``begin``.

Invariant I7 holds as for ``BusSpeechOutput``: ``segment`` takes only gate-built ``Segment``
objects; ``i7_check`` is the test hook.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TextIO

from aivtube.contracts.events import Event, SegmentDone, SegmentStarted, UtteranceDone
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.speech import StopMode, TTSConstraints, VoicePolicy
from aivtube.contracts.types import Segment
from aivtube.speech._common import (
    SegmentCheck,
    TurnDirectory,
    UtteranceBook,
    UttRecord,
    check_segment,
    truncate_at_space,
)

__all__ = ["CONSOLE_CANNED", "ConsoleSpeechOutput"]

log = logging.getLogger("aivtube.speech.console")

CONSOLE_CANNED: Mapping[str, str] = {
    "filtered": "Filtered.",
    "filler": "อืม…",
    "brain_freeze": "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ",
    "thanks": "ขอบคุณมากนะคะ",
}


@dataclass(eq=False)
class _Play:
    rec: UttRecord | None  # None: a canned phrase
    character: str = ""
    gate_open: bool = True
    filler_after_s: float | None = None
    text: str = ""  # canned phrases only
    queue: collections.deque[Segment] = field(default_factory=collections.deque)
    last_seen: bool = False
    stop_after_segment: bool = False
    stop_reason: str | None = None
    playing: bool = False
    started: bool = False
    filler_done: bool = False
    finished: bool = False
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    cut: asyncio.Event = field(default_factory=asyncio.Event)


class ConsoleSpeechOutput:
    """Prints captions with simulated speaking time (see the module docstring)."""

    def __init__(
        self,
        out: TextIO,
        bus: EventBus,
        clock: Clock,
        *,
        chars_per_s: float = 12.5,
        constraints: TTSConstraints | None = None,
        max_queued: int = 8,
        canned: Mapping[str, str] | None = None,
        names: Mapping[str, str] | None = None,
        tasks: TaskSupervisor | None = None,
        i7_check: SegmentCheck | None = None,
    ) -> None:
        if chars_per_s <= 0:
            raise ValueError("chars_per_s must be positive")
        self._out = out
        self._bus = bus
        self._clock = clock
        self.chars_per_s = chars_per_s
        self._constraints = constraints or TTSConstraints(
            first_min_chars=8, min_chars=40, max_chars=160, backend="console", identity="console"
        )
        self.max_queued = max_queued
        self.canned = dict(CONSOLE_CANNED if canned is None else canned)
        self.names = dict(names or {})
        self._tasks = tasks
        self._i7 = i7_check
        self._book = UtteranceBook()
        self._turns = TurnDirectory(bus)
        self._plays: collections.deque[_Play] = collections.deque()
        self._by_utt: dict[str, _Play] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None
        self._closed = False
        self.muted = False
        self.gain = 1.0
        self.policy = VoicePolicy()
        self.rates: dict[str, int] = {}

    # --- SpeechOutput ---------------------------------------------------------------------
    def ready(self) -> bool:
        return not self._closed

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
        rec = self._book.begin(utt_id, character, self._clock.now())
        play = _Play(rec, character, gate_open=gate_open, filler_after_s=filler_after_s)
        self._by_utt[utt_id] = play
        self._plays.append(play)
        self._kick()

    async def segment(self, seg: Segment) -> bool:
        check_segment(seg, self._i7)
        play = self._by_utt.get(seg.utt_id)
        if play is None or play.finished or play.stop_after_segment or play.last_seen:
            return True  # accepted and dropped: the utterance is not open
        if len(play.queue) >= self.max_queued:
            return False
        assert play.rec is not None
        play.rec.segments[seg.seq] = seg
        if seg.kind == "filtered":
            play.rec.filtered = True
        play.queue.append(seg)
        play.last_seen = seg.last
        play.wake.set()
        return True

    async def open_gate(self, utt_id: str) -> None:
        play = self._by_utt.get(utt_id)
        if play is not None:
            play.gate_open = True
            play.wake.set()

    async def stop(
        self, utt_id: str | None, mode: StopMode, reason: str, fade_ms: int = 30
    ) -> None:
        for play in list(self._plays):
            if play.finished:
                continue
            if play.rec is None:  # canned phrases stop only with stop(None)
                if utt_id is None:
                    play.finished = True
                    play.cut.set()
                continue
            if utt_id is not None and play.rec.utt_id != utt_id:
                continue
            play.stop_reason = reason
            play.rec.stopped = True
            if reason == "filtered":
                play.rec.filtered = True
            play.queue.clear()
            if play.playing and mode == "after_segment":
                play.stop_after_segment = True
            elif play.playing:
                play.cut.set()
            else:
                self._finish(play, cancelled=True)
            play.wake.set()
        self._kick()

    async def duck(self, gain: float, ramp_ms: int = 30) -> None:
        self.gain = min(1.0, max(0.0, float(gain)))

    async def mute(self, on: bool) -> None:
        self.muted = bool(on)

    async def set_policy(self, policy: VoicePolicy) -> None:
        self.policy = policy

    async def set_voice_rate(self, character: str, percent: int) -> None:
        self.rates[character] = int(percent)

    async def play_canned(self, key: str, character: str) -> None:
        if key == "filtered":
            self._book.mark_filtered(character)
        self._plays.append(_Play(None, character=character, text=self.canned.get(key, key)))
        self._kick()

    # --- extras ---------------------------------------------------------------------------
    def bind_turn(self, utt_id: str, turn_id: str | None) -> None:
        self._turns.bind(utt_id, turn_id)

    @property
    def idle(self) -> bool:
        return all(p.finished for p in self._plays)

    async def aclose(self) -> None:
        self._closed = True
        self._turns.close()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # --- the player ---------------------------------------------------------------------------
    def _kick(self) -> None:
        if self._closed:
            return
        self._wake.set()
        if self._task is None or self._task.done():
            coro = self._player()
            if self._tasks is not None:
                self._task = self._tasks.track(coro, name="console-speech")
            else:
                self._task = asyncio.get_running_loop().create_task(coro, name="console-speech")

    async def _player(self) -> None:
        while not self._closed:
            while self._plays and self._plays[0].finished:
                done = self._plays.popleft()
                if done.rec is not None:
                    self._by_utt.pop(done.rec.utt_id, None)
            if not self._plays:
                self._wake.clear()
                await self._wake.wait()
                continue
            play = self._plays[0]
            if play.rec is None:
                await self._play_canned(play)
            else:
                await self._play_utterance(play)

    async def _play_canned(self, play: _Play) -> None:
        if not self.muted:
            self._print(play.character, play.text)
        await self._wait(play.cut, len(play.text) / self._cps(play.character))
        play.finished = True

    async def _play_utterance(self, play: _Play) -> None:
        rec = play.rec
        assert rec is not None
        while not play.finished:
            if play.stop_after_segment or play.cut.is_set():
                self._finish(play, cancelled=True)
                return
            if not play.gate_open or not play.queue:
                if play.queue or not play.last_seen:
                    timeout = self._filler_timeout(play)
                    play.wake.clear()
                    woke = await self._wait(play.wake, timeout)
                    if not woke and timeout is not None:
                        play.filler_done = True
                        if not self.muted:
                            self._print(rec.character, self.canned.get("filler", "อืม…"))
                    continue
                self._finish(play, cancelled=False)
                return
            seg = play.queue.popleft()
            await self._play_segment(play, seg)
            if play.cut.is_set():
                self._finish(play, cancelled=True)
                return
            if seg.last:
                self._finish(play, cancelled=play.stop_after_segment)
                return

    def _filler_timeout(self, play: _Play) -> float | None:
        rec = play.rec
        if rec is None or play.filler_after_s is None or play.started or play.filler_done:
            return None
        return max(0.0, rec.began + play.filler_after_s - self._clock.now())

    async def _play_segment(self, play: _Play, seg: Segment) -> None:
        rec = play.rec
        assert rec is not None
        caption = seg.caption if seg.caption.strip() else seg.text
        cps = self._cps(rec.character)
        duration = len(seg.text) / cps
        silent = self.muted
        t0 = self._clock.now()
        play.playing = play.started = True
        self._publish(
            SegmentStarted(
                ts=t0,
                character=rec.character,
                turn_id=self._turn(rec),
                utt_id=rec.utt_id,
                seq=seg.seq,
                t_audible=t0,
                duration_s=duration,
                backend="console",
                silent=silent,
                caption=seg.caption,
                emotion=seg.emotion,
            )
        )
        if not silent:
            self._print(rec.character, caption)
        completed = not await self._wait(play.cut, duration)
        play.playing = False
        if silent:
            heard_text = ""
        elif completed:
            heard_text = caption
        else:
            elapsed = self._clock.now() - t0
            heard_text = truncate_at_space(caption, int(elapsed * cps))
        rec.heard[seg.seq] = heard_text
        self._publish(
            SegmentDone(
                ts=self._clock.now(),
                character=rec.character,
                turn_id=self._turn(rec),
                utt_id=rec.utt_id,
                seq=seg.seq,
                heard=completed and not silent,
                heard_text=heard_text,
            )
        )

    def _finish(self, play: _Play, *, cancelled: bool) -> None:
        rec = play.rec
        play.finished = True
        self._wake.set()
        if rec is None or not self._book.close(rec):
            return
        self._publish(
            UtteranceDone(
                ts=self._clock.now(),
                character=rec.character,
                turn_id=self._turn(rec),
                utt_id=rec.utt_id,
                heard_text=rec.heard_text(),
                cancelled=cancelled,
                reason=play.stop_reason if cancelled else None,
                filtered=rec.filtered,
            )
        )

    # --- helpers ------------------------------------------------------------------------------
    def _cps(self, character: str) -> float:
        return self.chars_per_s * max(0.1, 1.0 + self.rates.get(character, 0) / 100.0)

    def _turn(self, rec: UttRecord) -> str | None:
        if rec.turn_id is None:
            rec.turn_id = self._turns.lookup(rec.utt_id)
        return rec.turn_id

    def _print(self, character: str, text: str) -> None:
        name = self.names.get(character, character) or "?"
        line = f"{name}: {text}\n"
        try:
            self._out.write(line)
            self._out.flush()
        except UnicodeEncodeError:  # a console code page without Thai
            try:
                self._out.write(line.encode("ascii", "replace").decode("ascii"))
                self._out.flush()
            except (OSError, ValueError):
                pass
        except (OSError, ValueError):
            pass

    def _publish(self, event: Event) -> None:
        try:
            self._bus.publish(event)
        except Exception:
            log.exception("publishing %s failed", type(event).__name__)

    async def _wait(self, event: asyncio.Event, delay: float | None) -> bool:
        """Wait for ``event`` or ``delay`` clock seconds; ``True`` if the event won."""
        if event.is_set():
            return True
        waiter = asyncio.ensure_future(event.wait())
        tasks: set[asyncio.Future[Any]] = {waiter}
        if delay is not None:
            tasks.add(asyncio.ensure_future(self._clock.sleep(delay)))
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
        return waiter in done
