"""Worker-side speech queue: segments → TTS → player, with markers, heard text and lip tracks.

This is the voice worker's half of ``SpeechOutput`` (§3.5, §4.6, Appendix A). It is transport
agnostic: results go to a ``SpeechCallbacks`` object (``segment_started``, ``segment_done``,
``utterance_done``, ``lip_track``); ``IpcSpeechCallbacks`` maps them onto IPC messages.

**Order.** Utterances play one after another in ``begin`` order; canned phrases are queued
between them. At most ``max_queued`` (8) segments may wait per utterance (``segment`` returns
``False`` beyond that).

**Prefetch.** At most ``max_in_flight`` (2) segments, counted from the one playing, are being
synthesised or held ready: segment N+1 is synthesised while N plays. Audio of the segment that
is playing is pushed to the player as it is decoded (streaming first audio).

**Markers.** Before a segment's first sample a start marker is queued in the player, after its
last sample an end marker. The start marker's audible time is ``SegmentStarted.t_audible`` and
the lip tracks' ``t0``; the end marker completes the segment (``heard=True``). Marker callbacks
come from the player's notifier thread and are moved onto the loop.

**Heard text.** A segment cut while playing is reported with its text truncated at the last
word mark that was completely heard (by time, when the backend gave no marks). A fully heard
segment reports its caption (which keeps the chunker's spacing); the utterance's heard text is
the concatenation.

**Stop.** ``stop(now)`` fades the player out (one block) and drops everything of the target
utterances; ``stop(after_segment)`` lets the playing segment finish and drops the rest. A cut
made by someone else (the local barge-in reflex calling ``player.cancel``) is detected from
the failed markers and ends the utterance with reason ``"cut"``.

**Filler.** If an utterance began with ``filler_after_s`` and has no audio that long after
``begin``, a cached filler (``อืม…``) plays, at most once per ``filler_min_interval_s``. It is
never part of the heard text and never synthesised on demand.

**Silent segments.** Captions-only segments (the voice identity failed) and segments while
muted are "played" silently for ``len(text) / 12.5`` seconds and reported ``silent=True``.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from aivtube.contracts import ipc
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.infra import Clock, TaskSupervisor
from aivtube.contracts.speech import StopMode, TTSConstraints
from aivtube.contracts.types import Segment
from aivtube.contracts.voice import AudioChunk, AudioOut, PhraseCache, WordMark
from aivtube.infra.clock import SystemClock, deadline
from aivtube.voice.lipsync import LipSyncAnalyzer
from aivtube.voice.tts.router import CAPTIONS, SynthStream, TTSRouter, UtteranceTTS

__all__ = [
    "DEFAULT_CANNED",
    "DEFAULT_FILLERS",
    "IpcSpeechCallbacks",
    "SpeechCallbacks",
    "SpeechQueue",
    "truncate_heard",
]

log = logging.getLogger("aivtube.voice.speech")

DEFAULT_CANNED: Mapping[str, str] = {
    "filtered": "Filtered.",
    "brain_freeze": "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ",
    "thanks": "ขอบคุณมากนะคะ",
}
DEFAULT_FILLERS: tuple[str, ...] = ("อืม…", "เอ่อ…", "แป๊บนะ…")
_TOL_S = 0.02


class SpeechCallbacks(Protocol):
    """Where the queue reports (called on the event loop)."""

    def segment_started(
        self,
        utt: str,
        seq: int,
        t_audible: float,
        duration_s: float | None,
        backend: str,
        silent: bool,
    ) -> None: ...

    def segment_done(self, utt: str, seq: int, heard: bool, heard_text: str) -> None: ...

    def utterance_done(
        self, utt: str, heard_text: str, cancelled: bool, reason: str | None
    ) -> None: ...

    def lip_track(self, track: LipTrack) -> None: ...


class IpcSpeechCallbacks:
    """``SpeechCallbacks`` → Appendix A messages through ``send(type, data)``."""

    def __init__(self, send: Callable[[str, Mapping[str, Any]], None], *, digits: int = 3) -> None:
        self._send = send
        self._digits = digits

    def segment_started(
        self,
        utt: str,
        seq: int,
        t_audible: float,
        duration_s: float | None,
        backend: str,
        silent: bool,
    ) -> None:
        self._send(
            ipc.SEG_STARTED,
            {
                "utt": utt,
                "seq": seq,
                "t_audible": t_audible,
                "duration_s": None if duration_s is None else max(0.0, duration_s),
                "backend": backend,
                "silent": silent,
            },
        )

    def segment_done(self, utt: str, seq: int, heard: bool, heard_text: str) -> None:
        self._send(ipc.SEG_DONE, {"utt": utt, "seq": seq, "heard": heard, "heard_text": heard_text})

    def utterance_done(
        self, utt: str, heard_text: str, cancelled: bool, reason: str | None
    ) -> None:
        self._send(
            ipc.UTT_DONE,
            {"utt": utt, "heard_text": heard_text, "cancelled": cancelled, "reason": reason},
        )

    def lip_track(self, track: LipTrack) -> None:
        d = self._digits
        self._send(
            ipc.LIP_TRACK,
            {
                "utt": track.utt_id,
                "seq": track.seq,
                "t0": track.t0,
                "fps": track.fps,
                "mouth": [min(1.0, max(0.0, round(v, d))) for v in track.mouth],
                "form": [min(1.0, max(0.0, round(v, d))) for v in track.form],
                "final": track.final,
            },
        )


def truncate_heard(text: str, marks: Sequence[WordMark], heard_s: float, total_s: float) -> str:
    """``text`` cut after the last word completely heard within ``heard_s`` seconds."""
    if heard_s >= total_s - _TOL_S:
        return text
    if heard_s <= 0.0 or not text:
        return ""
    if marks:
        pos = cut = 0
        for m in marks:
            idx = text.find(m.text, pos)
            if idx < 0:
                continue
            if m.offset_s + m.duration_s > heard_s + _TOL_S:
                break
            cut = pos = idx + len(m.text)
        return text[:cut].rstrip()
    n = int(len(text) * heard_s / total_s) if total_s > 0 else 0
    prefix = text[:n]
    mid_word = n < len(text) and not text[n].isspace()
    if mid_word and " " in prefix.strip():  # drop the partly heard word
        prefix = prefix[: prefix.rstrip().rfind(" ")]
    return prefix.rstrip()


def _caption_for(seg: Segment, spoken: str, complete: bool) -> str:
    """Report the caption (the LLM's own text with its spacing) wherever it maps 1:1."""
    caption = seg.caption if seg.caption.strip() else seg.text
    if complete:
        return caption
    if caption.strip() == seg.text.strip() and spoken:
        lead = len(caption) - len(caption.lstrip())
        return caption[: lead + len(spoken)]
    return spoken


@dataclass(eq=False)
class _Seg:
    seg: Segment
    utt: _Utt
    index: int
    lip: LipSyncAnalyzer
    state: str = "queued"  # queued | synth | ready | playing | done | dropped
    task: asyncio.Task[None] | None = None
    stream: SynthStream | None = None
    chunks: list[AudioChunk] = field(default_factory=list)
    marks: list[WordMark] = field(default_factory=list)
    audio_s: float = 0.0
    synth_done: bool = False
    backend: str = ""
    captions: bool = False
    deferred: bool = False
    muted: bool = False
    mouth: list[float] = field(default_factory=list)
    form: list[float] = field(default_factory=list)
    lip_sent: int = 0
    lip_final: bool = False
    t_start: float | None = None
    pushed: int = 0
    pushed_s: float = 0.0
    end_marked: bool = False
    finished: bool = False
    gen: int = 0
    muted_at: float | None = None
    silent: bool = False
    silent_muted: bool = False
    silent_duration: float = 0.0


@dataclass(eq=False)
class _Utt:
    id: str
    character: str
    gate_open: bool
    filler_after_s: float | None
    began: float
    tts: UtteranceTTS
    segs: list[_Seg] = field(default_factory=list)
    next_play: int = 0
    last_seen: bool = False
    stop_after_segment: bool = False
    stop_reason: str | None = None
    done: bool = False
    heard: list[str] = field(default_factory=list)
    filler_played: bool = False
    started: bool = False


@dataclass(eq=False)
class _Canned:
    key: str
    character: str
    text: str
    lip_id: str
    audio: AudioChunk | None = None
    marks: tuple[WordMark, ...] = ()
    ready: bool = False
    failed: bool = False
    done: bool = False
    task: asyncio.Task[None] | None = None
    t_start: float | None = None
    gen: int = 0


class SpeechQueue:
    """Per-utterance speech queue of the voice worker (see the module docstring).

    Create it on the event loop that will run it. ``callbacks`` receives the results; for IPC
    pass ``send=`` instead (wrapped in ``IpcSpeechCallbacks``). ``supervisor`` (optional) tracks
    the internal tasks; without one the queue keeps strong references itself. ``set_gain``
    (``(gain, ramp_ms)``) replaces ``player.set_gain`` for MUTE and ducking, so the worker can
    route them through the barge-in controller, which owns the player gain.
    """

    def __init__(
        self,
        *,
        player: AudioOut,
        tts: TTSRouter,
        lipsync_factory: Callable[[], LipSyncAnalyzer] = LipSyncAnalyzer,
        callbacks: SpeechCallbacks | None = None,
        send: Callable[[str, Mapping[str, Any]], None] | None = None,
        cache: PhraseCache | None = None,
        clock: Clock | None = None,
        supervisor: TaskSupervisor | None = None,
        set_gain: Callable[[float, float], None] | None = None,
        max_in_flight: int = 2,
        max_queued: int = 8,
        filler_min_interval_s: float = 30.0,
        captions_cps: float = 12.5,
        canned: Mapping[str, str] | None = None,
        fillers: Sequence[str] = DEFAULT_FILLERS,
        recent_window_s: float = 10.0,
    ) -> None:
        if callbacks is None:
            if send is None:
                raise ValueError("SpeechQueue needs callbacks or send")
            callbacks = IpcSpeechCallbacks(send)
        self._cb = callbacks
        self._player = player
        self._tts = tts
        if cache is not None and tts.cache is None:
            tts.cache = cache
        self._lip_factory = lipsync_factory
        self._clock: Clock = clock or SystemClock()
        self._supervisor = supervisor
        self._set_gain: Callable[[float, float], None] = set_gain or player.set_gain
        self.max_in_flight = max(1, max_in_flight)
        self.max_queued = max_queued
        self.filler_min_interval_s = filler_min_interval_s
        self.captions_cps = captions_cps
        self.canned = dict(DEFAULT_CANNED if canned is None else canned)
        self.fillers = tuple(fillers)
        self.recent_window_s = recent_window_s
        self._utts: dict[str, _Utt] = {}
        self._finished: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._order: collections.deque[_Utt | _Canned] = collections.deque()
        self._recent: collections.deque[tuple[float, str]] = collections.deque(maxlen=64)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._player_task: asyncio.Task[None] | None = None
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gen = 0
        self._muted = False
        self._duck = 1.0
        self._last_filler = float("-inf")
        self._filler_turn = 0
        self._audible_until = float("-inf")
        self._current: _Seg | _Canned | None = None
        self._closed = False
        self.stats: collections.Counter[str] = collections.Counter()

    # --- SpeechOutput-like API ------------------------------------------------------------
    def constraints(self, character: str) -> TTSConstraints:
        return self._tts.constraints(character)

    @property
    def idle(self) -> bool:
        """Nothing open, queued or playing."""
        return not self._order and self._current is None

    async def begin(
        self, utt: str, character: str, *, filler_after_s: float | None, gate_open: bool
    ) -> None:
        """Open an utterance; its voice identity is chosen now and kept to the end."""
        if utt in self._utts or utt in self._finished:
            raise ValueError(f"utterance {utt!r} already begun")
        self._ensure_running()
        u = _Utt(
            id=utt,
            character=character,
            gate_open=gate_open,
            filler_after_s=filler_after_s,
            began=self._clock.now(),
            tts=self._tts.begin_utterance(character),
        )
        self._utts[utt] = u
        self._order.append(u)
        self._kick()

    async def segment(self, seg: Segment) -> bool:
        """Queue a segment; ``False`` = backpressure. Segments of closed utterances are dropped."""
        u = self._utts.get(seg.utt_id)
        if u is None or u.done or u.stop_after_segment or u.last_seen:
            self.stats["dropped_segments"] += 1
            log.debug("dropping segment %s/%s (utterance not open)", seg.utt_id, seg.seq)
            return True
        if any(s.seg.seq == seg.seq for s in u.segs):
            return True  # duplicate delivery
        waiting = sum(1 for s in u.segs[u.next_play :] if s.state not in ("playing", "dropped"))
        if waiting >= self.max_queued:
            self.stats["busy"] += 1
            return False
        s = _Seg(seg=seg, utt=u, index=len(u.segs), lip=self._lip_factory())
        if self._muted:
            s.muted = True
        u.segs.append(s)
        if seg.last:
            u.last_seen = True
        self._schedule()
        self._kick()
        return True

    async def open_gate(self, utt: str) -> None:
        u = self._utts.get(utt)
        if u is not None and not u.gate_open:
            u.gate_open = True
            self._schedule()
            self._kick()

    async def stop(self, utt: str | None, mode: StopMode, reason: str, fade_ms: int = 30) -> None:
        """Stop one utterance (``None``: everything, canned phrases included)."""
        self._stop(utt, mode, reason, fade_ms)

    async def duck(self, gain: float, ramp_ms: int = 30) -> None:
        self._duck = min(1.0, max(0.0, float(gain)))
        if not self._muted:
            self._set_gain(self._duck, float(ramp_ms))

    def set_muted(self, on: bool) -> None:
        """MUTE: output gain 0; new segments are dropped (reported silent) until unmuted."""
        if on == self._muted:
            return
        self._muted = on
        self._set_gain(0.0 if on else self._duck, 20.0)
        if on and isinstance(self._current, _Seg) and self._current.muted_at is None:
            self._current.muted_at = self._clock.now()
        for u in self._utts.values():
            for s in u.segs[u.next_play :]:
                if s.state in ("queued",):
                    s.muted = on
        self._kick()

    async def mute(self, on: bool) -> None:
        self.set_muted(on)

    async def set_voice_rate(self, character: str, percent: int) -> None:
        self._tts.set_rate(character, percent)

    async def play_canned(self, key: str, character: str) -> None:
        """Queue a canned phrase (``filtered``, ``filler``, ``brain_freeze`` …) after what plays."""
        if self._muted:
            self.stats["canned_muted"] += 1
            return
        self._ensure_running()
        if key == "filler":
            text = self._pick_filler(character) or (self.fillers[0] if self.fillers else "")
        else:
            text = self.canned.get(key, key)
        if not text.strip():
            return
        self._gen += 1
        item = _Canned(key=key, character=character, text=text, lip_id=f"canned-{key}-{self._gen}")
        hit = self._tts.cached_phrase(character, text)
        if hit is not None:
            item.audio, item.marks = hit
            item.ready = True
        else:
            log.info("canned phrase %r is not cached; synthesising it now", text)
            item.task = self._spawn(self._synth_canned(item), name=f"canned:{key}")
        self._order.append(item)
        self.stats["canned"] += 1
        self._kick()

    def recent_tts_text(self, window_s: float | None = None) -> str:
        """What she said in the last ``window_s`` seconds (default ``recent_window_s``, 10 s);
        feeds the STT echo filter and the barge-in echo check."""
        cutoff = self._clock.now() - (self.recent_window_s if window_s is None else window_s)
        return " ".join(text for t, text in self._recent if t >= cutoff)

    async def drop_all(self, *, max_wait_s: float = 30.0) -> None:
        """Link lost (I1): finish the current segment, then drop everything and go quiet."""
        for item in list(self._order):
            if isinstance(item, _Canned) and item is not self._current:
                self._drop_canned(item)
        self._stop(None, "after_segment", "link_lost", 30, keep_current_canned=True)
        with contextlib.suppress(Exception):
            async with deadline(max_wait_s, what="drop_all", clock=self._clock):
                while not self.idle:
                    await self._wait(0.05)

    async def aclose(self) -> None:
        """Stop everything (reported as cancelled) and end the internal tasks."""
        if self._closed:
            return
        self._stop(None, "now", "shutdown", 10)
        self._closed = True
        self._kick()
        tasks = [t for t in (self._player_task, *self._tasks) if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2.0):
                    await asyncio.gather(*tasks, return_exceptions=True)

    # --- tasks and waking -----------------------------------------------------------------
    def _spawn(self, coro: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
        if self._supervisor is not None:
            task: asyncio.Task[None] = self._supervisor.track(coro, name=name)
        else:
            task = asyncio.get_running_loop().create_task(coro, name=name)
            task.add_done_callback(self._task_done)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("speech task %s failed", task.get_name(), exc_info=task.exception())

    def _ensure_running(self) -> None:
        if self._closed:
            raise RuntimeError("SpeechQueue is closed")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._wake = asyncio.Event()
        if self._player_task is None or self._player_task.done():
            self._player_task = self._spawn(self._player_loop(), name="speech-player")

    def _kick(self) -> None:
        if self._wake is not None:
            self._wake.set()

    async def _wait(self, wait_s: float | None) -> bool:
        """Wait for a state change or ``wait_s`` clock seconds; ``True`` if woken."""
        assert self._wake is not None
        if self._wake.is_set():
            self._wake.clear()
            return True
        waiter = asyncio.ensure_future(self._wake.wait())
        tasks: set[asyncio.Future[Any]] = {waiter}
        if wait_s is not None:
            tasks.add(asyncio.ensure_future(self._clock.sleep(max(0.0, wait_s))))
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
        if waiter in done:
            self._wake.clear()
            return True
        return False

    def _threadsafe(self, fn: Callable[..., None], *args: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):  # loop closed at shutdown
            loop.call_soon_threadsafe(fn, *args)

    # --- synthesis ------------------------------------------------------------------------
    def _schedule(self) -> None:
        """Start synthesis for the first ``max_in_flight`` segments from the playing one."""
        if self._closed:
            return
        budget = self.max_in_flight
        for item in self._order:
            if budget <= 0:
                return
            if isinstance(item, _Canned) or item.done:
                continue
            u = item
            for s in u.segs[u.next_play :]:
                if budget <= 0:
                    return
                if s.state == "dropped":
                    continue
                budget -= 1
                if s.state == "queued":
                    if s.muted:
                        s.state, s.synth_done, s.backend = "ready", True, "muted"
                        continue
                    s.state = "synth"
                    s.task = self._spawn(self._synth(s), name=f"tts:{u.id}:{s.seg.seq}")
                elif s.deferred and u.gate_open and s.state == "ready" and not s.chunks:
                    s.deferred = s.synth_done = False
                    s.state = "synth"
                    s.task = self._spawn(self._synth(s), name=f"tts:{u.id}:{s.seg.seq}")

    async def _synth(self, s: _Seg) -> None:
        u = s.utt
        stream = u.tts.synth(s.seg.text, first=s.index == 0, speculative=not u.gate_open)
        s.stream = stream
        try:
            async for item in stream:
                if s.state == "dropped":
                    return
                if isinstance(item, WordMark):
                    s.marks.append(item)
                    continue
                if item.pcm.size == 0:
                    continue
                s.chunks.append(item)
                s.audio_s += item.pcm.size / item.sample_rate
                if not s.backend and stream.backend:
                    s.backend = stream.backend
                mouth, form = s.lip.feed(item.pcm, item.sample_rate)
                s.mouth.extend(mouth.tolist())
                s.form.extend(form.tolist())
                self._emit_lip(s)
                self._kick()
            mouth, form = s.lip.flush()
            s.mouth.extend(mouth.tolist())
            s.form.extend(form.tolist())
        except Exception:
            log.exception("synthesis of %s/%s failed", u.id, s.seg.seq)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
            s.backend = stream.backend or (CAPTIONS if stream.captions or not s.chunks else "")
            s.captions = stream.captions or (not s.chunks and not stream.deferred)
            s.deferred = stream.deferred
            s.synth_done = True
            if s.state == "synth":
                s.state = "ready"
            self.stats["fallback_segments"] += int(bool(stream.captions))
            self._emit_lip(s)
            self._schedule()
            self._kick()

    async def _synth_canned(self, item: _Canned) -> None:
        utt = self._tts.begin_utterance(item.character)
        stream = utt.synth(item.text, first=True)
        chunks: list[np.ndarray] = []
        marks: list[WordMark] = []
        rate = 24000
        try:
            async for x in stream:
                if isinstance(x, WordMark):
                    marks.append(x)
                else:
                    chunks.append(x.pcm)
                    rate = x.sample_rate
        except Exception:
            log.exception("synthesis of canned phrase %r failed", item.text)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
            if chunks:
                item.audio = AudioChunk(np.concatenate(chunks).astype(np.int16), rate)
                item.marks = tuple(marks)
                item.ready = True
            else:
                item.failed = True
            self._kick()

    # --- lip tracks -----------------------------------------------------------------------
    def _emit_lip(self, s: _Seg) -> None:
        if s.t_start is None or s.lip_final or s.finished:
            return
        final = s.synth_done
        start, end = s.lip_sent, len(s.mouth)
        if end <= start and not final:
            return
        fps = s.lip.fps
        track = LipTrack(
            utt_id=s.utt.id,
            seq=s.seg.seq,
            t0=s.t_start + start / fps,
            fps=fps,
            mouth=tuple(s.mouth[start:end]),
            form=tuple(s.form[start:end]),
            final=final,
        )
        s.lip_sent = end
        s.lip_final = final
        self.stats["lip_tracks"] += 1
        self._call(self._cb.lip_track, track)

    def _lip_whole(self, lip_id: str, seq: int, audio: AudioChunk, t0: float) -> None:
        lip = self._lip_factory()
        m1, f1 = lip.feed(audio.pcm, audio.sample_rate)
        m2, f2 = lip.flush()
        mouth = np.concatenate([m1, m2]).tolist()
        form = np.concatenate([f1, f2]).tolist()
        if mouth:
            track = LipTrack(lip_id, seq, t0, lip.fps, tuple(mouth), tuple(form), final=True)
            self._call(self._cb.lip_track, track)

    # --- callbacks ------------------------------------------------------------------------
    def _call(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            log.exception("speech callback %s failed", getattr(fn, "__name__", fn))

    def _finish(self, u: _Utt, *, cancelled: bool, reason: str | None) -> None:
        if u.done:
            return
        u.done = True
        for s in u.segs:
            if s.state not in ("done", "playing"):
                s.state = "dropped"
            if s.task is not None and not s.task.done() and s.state == "dropped":
                s.task.cancel()
        self._utts.pop(u.id, None)
        self._finished[u.id] = None
        while len(self._finished) > 256:
            self._finished.popitem(last=False)
        self.stats["utterances"] += 1
        self.stats["cancelled"] += int(cancelled)
        self._call(
            self._cb.utterance_done,
            u.id,
            "".join(u.heard),
            cancelled,
            reason if cancelled else None,
        )
        self._kick()

    def _segment_done(self, s: _Seg, *, heard: bool, spoken: str, complete: bool) -> None:
        if s.finished:
            return
        s.finished = True
        s.state = "done"
        text = _caption_for(s.seg, spoken, complete) if (spoken or complete) else ""
        s.utt.heard.append(text)
        if s.task is not None and not s.task.done():
            s.task.cancel()
        self._call(self._cb.segment_done, s.utt.id, s.seg.seq, heard, text)
        self._kick()

    # --- marker handlers (moved onto the loop) --------------------------------------------
    def _on_seg_start(self, s: _Seg, gen: int, heard: bool, t: float) -> None:
        if s.gen != gen or s.finished or s.t_start is not None:
            return
        if not heard:
            self._external_cut(s.utt, t)
            return
        s.t_start = t
        self._recent.append((t, s.seg.text))
        duration = s.audio_s if s.synth_done else None
        self._call(
            self._cb.segment_started, s.utt.id, s.seg.seq, t, duration, s.backend or "", False
        )
        self._emit_lip(s)

    def _on_seg_end(self, s: _Seg, gen: int, heard: bool, t: float) -> None:
        if s.gen != gen or s.finished:
            return
        if not heard:
            self._external_cut(s.utt, t)
            return
        self._audible_until = max(self._audible_until, t)
        if s.muted_at is not None and s.t_start is not None:
            spoken = truncate_heard(s.seg.text, s.marks, s.muted_at - s.t_start, s.audio_s)
            self._segment_done(s, heard=False, spoken=spoken, complete=False)
        else:
            self._segment_done(s, heard=True, spoken=s.seg.text, complete=True)

    def _on_canned_mark(self, item: _Canned, gen: int, start: bool, heard: bool, t: float) -> None:
        if item.gen != gen or item.done:
            return
        if not heard:
            item.done = True
            self._kick()
            return
        if start:
            item.t_start = t
            self._recent.append((t, item.text))
            if item.audio is not None:
                self._lip_whole(item.lip_id, 0, item.audio, t)
        else:
            self._audible_until = max(self._audible_until, t)
            item.done = True
            self._kick()

    def _external_cut(self, u: _Utt, t: float) -> None:
        """The player was cancelled by someone else (barge-in reflex, device loss)."""
        if u.done:
            return
        s = self._current if isinstance(self._current, _Seg) else None
        if s is not None and s.utt is u and not s.finished:
            if s.t_start is not None:
                total = s.audio_s if s.synth_done else float("inf")
                spoken = truncate_heard(s.seg.text, s.marks, max(0.0, t - s.t_start), total)
                self._segment_done(s, heard=False, spoken=spoken, complete=False)
            else:
                s.finished = True
                s.state = "done"
        self._finish(u, cancelled=True, reason="cut")

    # --- stop -----------------------------------------------------------------------------
    def _stop(
        self,
        utt: str | None,
        mode: StopMode,
        reason: str,
        fade_ms: int,
        *,
        keep_current_canned: bool = False,
    ) -> None:
        targets = [u for u in list(self._utts.values()) if utt is None or u.id == utt]
        head = self._order[0] if self._order else None
        cut_player = False
        if utt is None:
            for item in list(self._order):
                if isinstance(item, _Canned) and not (
                    keep_current_canned and item is self._current
                ):
                    if item is head and mode == "now":
                        cut_player = True
                    self._drop_canned(item)
        for u in targets:
            u.stop_reason = reason
            playing = self._current if isinstance(self._current, _Seg) else None
            if playing is not None and (playing.utt is not u or playing.finished):
                playing = None
            for s in u.segs[u.next_play :]:
                if s is playing or s.state in ("done", "dropped"):
                    continue
                s.state = "dropped"
                if s.task is not None and not s.task.done():
                    s.task.cancel()
            if mode == "now":
                if head is u and (playing is not None or u.filler_played):
                    cut_player = True
                if playing is not None:
                    self._cut_segment(playing, fade_ms)
                    cut_player = False  # _cut_segment cancelled the player
                self._finish(u, cancelled=True, reason=reason)
            elif playing is not None:
                u.stop_after_segment = True
                u.segs = u.segs[: playing.index + 1]
            else:
                self._finish(u, cancelled=True, reason=reason)
        if cut_player:
            self._player.cancel(float(fade_ms))
        self._kick()

    def _cut_segment(self, s: _Seg, fade_ms: int) -> None:
        dropped = self._player.cancel(float(fade_ms))
        s.gen = -1  # the failed markers of this segment are stale now
        if s.t_start is None:  # never became audible
            s.finished = True
            s.state = "done"
            if s.task is not None and not s.task.done():
                s.task.cancel()
            return
        if s.silent:
            shown = max(0.0, self._clock.now() - s.t_start)
            spoken = (
                "" if s.silent_muted else truncate_heard(s.seg.text, (), shown, s.silent_duration)
            )
        else:
            heard_s = max(0.0, min(s.pushed_s - dropped, s.audio_s))
            if s.muted_at is not None:
                heard_s = min(heard_s, s.muted_at - s.t_start)
            total = s.audio_s if s.synth_done else float("inf")
            spoken = truncate_heard(s.seg.text, s.marks, heard_s, total)
        self._segment_done(s, heard=False, spoken=spoken, complete=False)

    def _drop_canned(self, item: _Canned) -> None:
        item.done = True
        item.gen = -1
        if item.task is not None and not item.task.done():
            item.task.cancel()
        with contextlib.suppress(ValueError):
            if item is not self._current:
                self._order.remove(item)

    # --- the player -----------------------------------------------------------------------
    async def _player_loop(self) -> None:
        while not self._closed:
            if not self._order:
                await self._wait(None)
                continue
            item = self._order[0]
            try:
                if isinstance(item, _Canned):
                    await self._play_canned(item)
                else:
                    await self._play_utt(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("speech player failed")
                if isinstance(item, _Utt):
                    self._finish(item, cancelled=True, reason="error")
                else:
                    item.done = True
            finally:
                self._current = None
            if self._order and self._order[0] is item:
                self._order.popleft()
            self._schedule()

    async def _play_canned(self, item: _Canned) -> None:
        try:
            async with deadline(10.0, what="canned synthesis", clock=self._clock):
                while not item.done and not item.ready and not item.failed:
                    await self._wait(None)
        except TimeoutError:  # DeadlineExceeded
            log.warning("canned phrase %r was not ready in time; skipped", item.text)
            item.failed = True
        if item.done or item.failed or item.audio is None or self._muted:
            item.done = True
            return
        self._current = item
        self._gen += 1
        item.gen = gen = self._gen
        self._player.mark(
            lambda h, t: self._threadsafe(self._on_canned_mark, item, gen, True, h, t)
        )
        self._player.play(_to_f32(item.audio.pcm), item.audio.sample_rate)
        self._player.mark(
            lambda h, t: self._threadsafe(self._on_canned_mark, item, gen, False, h, t)
        )
        while not item.done:
            await self._wait(None)

    async def _play_utt(self, u: _Utt) -> None:
        while not u.done:
            self._schedule()
            if u.stop_after_segment:
                self._finish(u, cancelled=True, reason=u.stop_reason)
                return
            s = u.segs[u.next_play] if u.next_play < len(u.segs) else None
            if s is None:
                if u.last_seen:
                    self._finish(u, cancelled=False, reason=None)
                    return
                await self._wait(self._filler_timeout(u))
                self._maybe_filler(u)
                continue
            if s.state == "dropped":
                u.next_play += 1
                continue
            if not u.gate_open or (not s.chunks and not s.synth_done) or s.deferred:
                await self._wait(self._filler_timeout(u))
                self._maybe_filler(u)
                continue
            self._current = s
            u.started = True
            try:
                if s.chunks and not s.muted and not self._muted:
                    await self._play_audio_segment(s)
                else:
                    await self._play_silent_segment(s)
            finally:
                self._current = None
            u.next_play += 1
            if s.seg.last and not u.done:
                self._finish(
                    u,
                    cancelled=u.stop_after_segment,
                    reason=u.stop_reason if u.stop_after_segment else None,
                )
                return

    async def _play_audio_segment(self, s: _Seg) -> None:
        s.state = "playing"
        self._gen += 1
        s.gen = gen = self._gen
        self._player.mark(lambda h, t: self._threadsafe(self._on_seg_start, s, gen, h, t))
        while not s.finished:
            while s.pushed < len(s.chunks) and not s.finished:
                chunk = s.chunks[s.pushed]
                s.pushed += 1
                self._player.play(_to_f32(chunk.pcm), chunk.sample_rate)
                s.pushed_s += chunk.pcm.size / chunk.sample_rate
            if s.synth_done and s.pushed >= len(s.chunks) and not s.end_marked and not s.finished:
                s.end_marked = True
                self._player.mark(lambda h, t: self._threadsafe(self._on_seg_end, s, gen, h, t))
            if s.finished or s.utt.done:
                break
            if not s.end_marked:
                await self._wait(None)
                continue
            played = self._clock.now() - s.t_start if s.t_start is not None else 0.0
            limit = max(0.0, s.audio_s - played) + self._player.output_latency_s + 2.0
            if not await self._wait(limit) and not s.finished and not s.utt.done:
                log.warning("no end marker for %s/%s; assuming it played", s.utt.id, s.seg.seq)
                self._segment_done(s, heard=True, spoken=s.seg.text, complete=True)

    async def _play_silent_segment(self, s: _Seg) -> None:
        s.state = "playing"
        u = s.utt
        muted = s.muted or self._muted
        text = s.seg.caption if s.seg.caption.strip() else s.seg.text
        duration = len(text) / self.captions_cps if self.captions_cps > 0 else 0.0
        now = self._clock.now()
        t0 = max(now, self._audible_until)
        s.silent, s.silent_muted, s.silent_duration, s.t_start = True, muted, duration, t0
        backend = "muted" if muted else (s.backend or CAPTIONS)
        self._call(self._cb.segment_started, u.id, s.seg.seq, t0, duration, backend, True)
        if not muted:
            self._recent.append((t0, s.seg.text))
        end = t0 + duration
        while not s.finished and not u.done:
            remaining = end - self._clock.now()
            if remaining <= 0:
                break
            await self._wait(remaining)
        if s.finished:
            return
        if u.done:  # stopped (now) while showing the caption
            shown = max(0.0, self._clock.now() - t0)
            spoken = "" if muted else truncate_heard(s.seg.text, (), shown, duration)
            self._segment_done(s, heard=False, spoken=spoken, complete=False)
            return
        self._audible_until = max(self._audible_until, end)
        if muted:
            self._segment_done(s, heard=False, spoken="", complete=False)
        else:  # the captions overlay showed the whole segment
            self._segment_done(s, heard=True, spoken=s.seg.text, complete=True)

    # --- filler ---------------------------------------------------------------------------
    def _filler_timeout(self, u: _Utt) -> float | None:
        if not self._filler_eligible(u):
            return None
        assert u.filler_after_s is not None
        return max(0.0, u.began + u.filler_after_s - self._clock.now())

    def _filler_eligible(self, u: _Utt) -> bool:
        nxt = u.segs[u.next_play] if u.next_play < len(u.segs) else None
        ready = nxt is not None and bool(nxt.chunks)
        return (
            u.filler_after_s is not None
            and not u.filler_played
            and not u.started
            and not ready
            and u.gate_open
            and not u.done
            and not self._muted
            and bool(self.fillers)
            and self._clock.now() - self._last_filler >= self.filler_min_interval_s
            and (not self._order or self._order[0] is u)
        )

    def _maybe_filler(self, u: _Utt) -> None:
        if not self._filler_eligible(u):
            return
        assert u.filler_after_s is not None
        if self._clock.now() < u.began + u.filler_after_s:
            return
        u.filler_played = True  # decided either way: at most one attempt per utterance
        text = self._pick_filler(u.character)
        hit = self._tts.cached_phrase(u.character, text) if text else None
        if hit is None:
            self.stats["filler_missing"] += 1
            return
        audio, _marks = hit
        self._last_filler = self._clock.now()
        self.stats["fillers"] += 1
        lip_id = u.id

        def on_start(heard: bool, t: float) -> None:
            if heard:
                self._threadsafe(self._filler_started, lip_id, audio, text, t)

        self._player.mark(on_start)
        self._player.play(_to_f32(audio.pcm), audio.sample_rate)
        self._player.mark(lambda h, t: self._threadsafe(self._filler_ended, h, t))

    def _filler_started(self, lip_id: str, audio: AudioChunk, text: str, t: float) -> None:
        self._recent.append((t, text))
        self._lip_whole(lip_id, 0, audio, t)

    def _filler_ended(self, heard: bool, t: float) -> None:
        if heard:
            self._audible_until = max(self._audible_until, t)

    def _pick_filler(self, character: str) -> str:
        """The next cached filler (rotating), or ``""`` when none is cached."""
        n = len(self.fillers)
        for i in range(n):
            text = self.fillers[(self._filler_turn + i) % n]
            if self._tts.cached_phrase(character, text) is not None:
                self._filler_turn = (self._filler_turn + i + 1) % n
                return text
        return ""


def _to_f32(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype == np.int16:
        return pcm.astype(np.float32) / 32768.0
    return pcm.astype(np.float32, copy=False)
