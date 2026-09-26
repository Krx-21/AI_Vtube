"""The serial ``Brain`` (ARCHITECTURE.md §4.1, §4.5, §2.11; parity §0.1; invariants §2.7).

One ``Brain`` per character, on the core's event loop only. A decision is one LLM generation
plus at most one tool follow-up round, and decisions are serial::

    while True:
        await gate.wait_open()                       # not PAUSED; the voice side is ready
        stim = await arbiter.next(idle_deadline=…, user_speaking=…)
        stim = stim or idle.make_stimulus(character)
        ctx = arbiter.drain_context(stim)            # everything queued since the last decision
        outcome = await slot.run(turn.id, self._decide(turn))   # the ONLY path to a decision
        await self._after(turn, outcome)

:meth:`Brain.submit` (sync, never blocks) queues a stimulus and applies the pure preemption
table (``decide_preemption``, §4.4): it may cancel the decision in flight and stop speech. A
stimulus arriving mid-decision is merged into the next one; an aborted decision whose speech was
never audible is put back and merged into the restart (§4.3b).

``_decide``: gather context (recall, viewer facts and new memories under a 50 ms deadline),
build the prompt (assistant history text frozen at first render), ``speech.begin`` (filler after
1.2 s for voice turns), stream the reply, then run tools with at most one follow-up (``/u2``).
History is written by ``_after``, in the brain task, so a cancelled decision never leaves half a
turn behind.

**Speech tracking.** ``SegmentStarted`` makes an utterance audible (state SPEAKING),
``SegmentDone`` carries heard text, ``UtteranceDone`` gives the exact heard text; the history line
is updated (or, once frozen, corrected in the next tail). After a ``now`` cut the brain waits up
to 150 ms for ``UtteranceDone`` so the heard text is exact to the word.

**Operator** (:meth:`Brain.control`, §2.11): FREEZE (sync fast path; cancel the LLM, stop speech
within 150 ms, PAUSED, chat and tools off, avatar neutral; only RESUME undoes it), SKIP, MUTE,
RESUME (inbox cleared except SUPPORT), GO_LIVE, CHAT_INTAKE, SAY, DIRECT, MIC_MODE, PTT,
LLM_USE/LLM_ROLLBACK, STRICT, FILTER_RELOAD, MUTE_USER and END_STREAM. Other kinds answer
``OpResult(ok=False, "not implemented …")``.

**Failures.** A 30 s watchdog cancels a hung decision and dumps the flight recorder. When every
provider fails before saying anything, the brain plays the canned "brain freeze" line once per
outage and raises a panel alarm (the real router may already have spoken it: ``canned``). When
the voice worker goes down (``HealthChanged`` of ``voice``), the decision in flight is cancelled
and open utterances are closed as ``voice_restart`` keeping only the heard text; nothing is
decided until ``SpeechOutput.ready()`` again.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Final, cast

from aivtube.brain._util import wait_future
from aivtube.brain.arbiter import Arbiter, MergedContext, is_say
from aivtube.brain.background import BackgroundJobs
from aivtube.brain.decision import DecisionOutcome, DecisionResult, DecisionSlot
from aivtube.brain.history import History
from aivtube.brain.idle import IdleScheduler
from aivtube.brain.intake import Intake
from aivtube.brain.preempt import BrainSnapshot, decide_preemption
from aivtube.brain.prompt import PromptBuilder, TurnContext
from aivtube.brain.reply import ReplyPipeline, ReplyProgress, ReplyResult
from aivtube.brain.state import BrainState, BrainStateMachine
from aivtube.brain.tool_flow import UNAVAILABLE_RESULT, ToolFlow, assistant_message_copy
from aivtube.contracts.avatar import AvatarDriver, AvatarSink
from aivtube.contracts.chat import ChannelActions
from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.events import (
    Alert,
    BargeInConfirmed,
    DecisionAborted,
    DecisionStarted,
    EmotionChanged,
    Event,
    HealthChanged,
    SegmentDone,
    SegmentStarted,
    UserSpeechEnded,
    UserSpeechStarted,
    UserTranscript,
    UtteranceDone,
    UtteranceStarted,
)
from aivtube.contracts.infra import Clock, EventBus, Subscription, TaskSupervisor
from aivtube.contracts.llm import ChatRequest, LLMRouter, ToolCall
from aivtube.contracts.memory import MemoryItem, MemoryStore
from aivtube.contracts.safety import SafetyGate, Verdict
from aivtube.contracts.speech import SpeechOutput, StopMode
from aivtube.contracts.tools import ToolContext, ToolRegistry, ToolResult
from aivtube.contracts.types import (
    ChatMessage,
    HealthState,
    Platform,
    Segment,
    Stimulus,
    StimulusKind,
)
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.infra.trace import TurnTraceRecorder
from aivtube.text.chunker import ChunkerConfig, ThaiSpeechChunker, no_word_split

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig, CharacterConfig

__all__ = ["Brain", "BrainDeps", "BrainSettings", "brain_cfg"]

log = logging.getLogger("aivtube.brain")

#: Abort reasons that drop the decision's stimuli. Any other abort with nothing audible yet
#: (a restart, a preemption, a barge-in) puts them back, merged into the next decision (§4.3b).
_DROP: Final = frozenset({"operator_freeze", "operator_skip", "watchdog", "cancelled"})
_CHATTY: Final = frozenset({StimulusKind.CHAT, StimulusKind.MENTION, StimulusKind.SUPPORT})
_FILTERED_NOTE: Final = "(ประโยคก่อนหน้าถูกกรอง)"
_STALE_UTTERANCE_S: Final = 90.0
#: Health states of the voice component that mean its utterances are gone (§2.8).
_VOICE_LOST: Final = frozenset({HealthState.DOWN, HealthState.FAILED, HealthState.STARTING})


@dataclass(frozen=True, slots=True)
class BrainSettings:
    """Brain knobs, read from the flat ``cfg`` mapping (see :func:`brain_cfg`)."""

    decision_watchdog_s: float = 30.0
    max_tool_rounds: int = 2
    user_speaking_watchdog_s: float = 20.0
    opener_repeat_ratio: float = 0.3
    critical_cut_wait_ms: int = 150
    filler_after_s: float = 1.2
    temperature: float = 0.6
    strict_temperature: float = 0.4
    max_reply_tokens: int = 256
    recall_k: int = 2
    viewer_facts_per_viewer: int = 3
    viewer_facts_max_viewers: int = 5
    gather_timeout_s: float = 0.05
    auto_live: bool = False
    speech_timeout_s: float = 2.0
    freeze_timeout_s: float = 0.15
    fast_timeout_s: float = 0.5
    end_stream_wait_s: float = 5.0
    ready_poll_s: float = 0.25
    filter_reload_timeout_s: float = 10.0
    flight_dump_timeout_s: float = 10.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> BrainSettings:
        if not cfg:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})


def brain_cfg(app: AppConfig, character: CharacterConfig | None = None) -> dict[str, Any]:
    """The flat ``cfg`` mapping for ``BrainDeps`` / ``Arbiter`` / ``Intake`` from the config."""
    b, mic = app.brain, app.mic
    out: dict[str, Any] = b.model_dump(exclude={"budget"})
    out.update(
        filler_after_s=app.tts.filler_after_s,
        temperature=app.llm.temperature,
        strict_temperature=app.llm.strict_temperature,
        max_reply_tokens=app.llm.max_reply_tokens,
        recall_k=app.memory.recall_k,
        viewer_facts_per_viewer=app.memory.viewer_facts_per_viewer,
        viewer_facts_max_viewers=app.memory.viewer_facts_max_viewers,
        auto_live=app.app.auto_live,
        mic_mode=mic.mode,
        addressing=mic.addressing,
        followup_window_s=mic.followup_window_s,
        read_aloud_dedupe=mic.read_aloud_dedupe,
        read_aloud_ratio=mic.read_aloud_ratio,
        read_aloud_window_s=mic.read_aloud_window_s,
        read_aloud_min_chars=mic.read_aloud_min_chars,
        max_msg_chars=min(300, app.chat.max_msg_chars),
        chat_intake=app.chat.enabled,
        mute_user_s=app.safety.auto_strict_mute_s,
    )
    return out


@dataclass
class BrainDeps:
    """Everything a :class:`Brain` needs (modules.json ``brain.loop``).

    Extras beyond modules.json (all optional): ``avatar_sink`` and ``channels`` for the tools'
    ``ToolContext``, ``intake`` (built from the other deps when ``None``), ``dump_flight``
    (``FlightRecorder.adump`` bound to the dump folder; called on a watchdog trip) and
    ``subscribe`` (``False``: the caller feeds events through :meth:`Brain.on_event`).
    """

    character: CharacterConfig
    clock: Clock
    bus: EventBus
    tasks: TaskSupervisor
    arbiter: Arbiter
    prompt: PromptBuilder
    router: LLMRouter
    reply: ReplyPipeline
    tool_flow: ToolFlow
    registry: ToolRegistry
    history: History
    background: BackgroundJobs
    memory: MemoryStore
    speech: SpeechOutput
    safety: SafetyGate
    idle: IdleScheduler
    avatar: AvatarDriver | None
    trace: TurnTraceRecorder
    cfg: Mapping[str, Any]
    avatar_sink: AvatarSink | None = None
    channels: Mapping[Platform, ChannelActions] = field(default_factory=dict)
    intake: Intake | None = None
    dump_flight: Callable[[str], Awaitable[Any]] | None = None
    subscribe: bool = True


@dataclass(eq=False)
class _Utt:
    """One utterance this brain began (until ``UtteranceDone``)."""

    utt_id: str
    turn: _Turn
    voice: bool
    began: float
    progress: ReplyProgress = field(default_factory=ReplyProgress)
    started: dict[int, str] = field(default_factory=dict)  # seq -> caption (audible)
    heard: dict[int, str] = field(default_factory=dict)  # seq -> heard text (SegmentDone)
    audible: bool = False
    stop: StopMode | None = None
    done: UtteranceDone | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    history_id: int | None = None
    last_activity: float = 0.0

    def tentative(self) -> tuple[str, bool]:
        """Heard text and interrupted flag as far as known now (§4.1): segments heard, plus
        the one playing when a stop after it is guaranteed."""
        parts: list[str] = []
        playing_counts = self.stop != "now"
        for seq in sorted(self.started):
            if seq in self.heard:
                parts.append(self.heard[seq])
            elif playing_counts:
                parts.append(self.started[seq])
        sent = [s.seq for s in self.progress.sent if s.text.strip()]
        unstarted = any(seq not in self.started for seq in sent)
        interrupted = self.stop is not None and (
            self.stop == "now" or unstarted or not self.progress.ended
        )
        return "".join(parts), interrupted


@dataclass(eq=False)
class _Round:
    utt: _Utt
    result: ReplyResult | None = None
    tools: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    assistant_message: Mapping[str, Any] | None = None


@dataclass(eq=False)
class _Turn:
    id: str
    ctx: MergedContext
    started: float
    rounds: list[_Round] = field(default_factory=list)
    tctx: TurnContext | None = None
    decision_done: bool = False
    finished: bool = False
    outcome: str = "ok"

    @property
    def primary(self) -> Stimulus:
        return self.ctx.primary

    @property
    def utts(self) -> list[_Utt]:
        return [r.utt for r in self.rounds]

    def audible(self) -> bool:
        return any(u.audible for u in self.utts)

    def sent_anything(self) -> bool:
        return any(any(s.text.strip() for s in u.progress.sent) for u in self.utts)

    def emitted_text(self) -> str:
        return "".join(u.progress.emitted_text for u in self.utts)


class Brain:
    """The serial decision loop for one character; see the module docstring."""

    def __init__(self, deps: BrainDeps) -> None:
        self.deps = deps
        self.character = str(deps.character.id)
        self.cfg = BrainSettings.from_mapping(deps.cfg)
        self._clock = deps.clock
        self._bus = deps.bus
        self._tasks = deps.tasks
        self.arbiter: Arbiter = deps.arbiter
        self.slot = DecisionSlot(deps.tasks)
        self.states = BrainStateMachine(character=self.character, bus=deps.bus, avatar=deps.avatar)
        self.intake = deps.intake or Intake(
            deps.character,
            gate=deps.safety,
            window=deps.arbiter.window,
            submit=self.submit,
            note=self._note,
            clock=deps.clock,
            bus=deps.bus,
            cfg=deps.cfg,
            withdraw=deps.arbiter.remove,
            tasks=deps.tasks,
        )
        self.live = False
        self.paused = False
        self.muted = False
        self.ending = False
        self.decisions = 0
        self._strict_local = False
        self._chat_before_freeze: bool | None = None
        self._user_since: float | None = None
        self._open = asyncio.Event()
        self._utts: dict[str, _Utt] = {}
        self._current: _Turn | None = None
        self._last_filtered = False
        self._carry_notes: list[str] = []
        self._freeze_played = False
        self._openers: collections.deque[str] = collections.deque(maxlen=20)
        self._topics: collections.deque[str] = collections.deque(maxlen=5)
        self._filtered_chatters: collections.deque[tuple[float, frozenset[tuple[str, str]]]] = (
            collections.deque(maxlen=20)
        )
        self._sub: Subscription | None = None
        self._prepared = False

    # --- public state ---------------------------------------------------------------------
    @property
    def state(self) -> BrainState:
        return self.states.state

    def user_speaking(self) -> bool:
        """The streamer is talking (``vad.start`` … ``vad.end``, 20 s watchdog)."""
        since = self._user_since
        if since is None:
            return False
        if self._clock.now() - since > self.cfg.user_speaking_watchdog_s:
            self._user_since = None
            self.states.set_user_speaking(False)
            return False
        return True

    def snapshot(self) -> BrainSnapshot:
        current = self._current
        open_utts = list(self._utts.values())
        return BrainSnapshot(
            state=self.state,
            user_speaking=self.user_speaking(),
            decision_turn=self.slot.current_turn,
            decision_rank=current.primary.rank if current is not None else None,
            utterance=open_utts[-1].utt_id if open_utts else None,
            audible=any(u.audible for u in open_utts),
            paused=self.paused,
        )

    def strict(self) -> bool:
        try:
            gate_strict = self.deps.safety.strict_mode(self.character)
        except Exception:
            gate_strict = False
        return gate_strict or self._strict_local

    # --- intake ----------------------------------------------------------------------------
    def submit(self, s: Stimulus) -> None:
        """Queue a stimulus and apply the preemption table (sync; never blocks)."""
        self.arbiter.push(s)
        act = decide_preemption(s, self.snapshot())
        if act.cancel_decision:
            self.slot.cancel(act.reason)
        if act.speech_stop is not None:
            for utt in list(self._utts.values()):
                self._stop_utt(utt, act.speech_stop, act.reason)
        if s.kind is StimulusKind.VOICE:
            self.deps.idle.on_activity(self._clock.now())

    def on_chat(self, m: ChatMessage) -> None:
        """Chat and alert ingest (platform sources, ``/api/event``, the console)."""
        self.intake.on_chat(m)
        self.deps.idle.on_chat_activity(self._clock.now())

    def on_auto_strict(self, character: str) -> None:
        """``LayeredSafetyGate`` auto-strict hook: hide the chatters whose messages fed the
        recently filtered decisions (§2.11)."""
        if character != self.character:
            return
        horizon = self._clock.now() - 300.0
        for t, users in self._filtered_chatters:
            if t >= horizon:
                for platform, user_id in users:
                    self.intake.mute_user(platform=platform, user_id=user_id)
        self._bus.publish(
            Alert(
                level="warn",
                message="เปิดโหมดเข้มงวดอัตโนมัติ (ถูกกรองหลายครั้ง) ซ่อนแชทของคนที่เกี่ยวข้อง 10 นาที",
                character=self.character,
            )
        )

    # --- the loop --------------------------------------------------------------------------
    async def run(self) -> None:
        """The decision loop (a critical supervised task)."""
        pump: asyncio.Task[Any] | None = None
        if self.deps.subscribe:  # subscribe before any await, so no event is lost
            self._sub = self._bus.subscribe(
                UserSpeechStarted,
                UserSpeechEnded,
                UserTranscript,
                BargeInConfirmed,
                SegmentStarted,
                SegmentDone,
                UtteranceDone,
                HealthChanged,
                name=f"brain:{self.character}",
                maxsize=4096,
            )
        try:
            await self._prepare()
            self._current = None  # a previous run() may have died mid-decision
            if self._sub is not None:
                pump = self._tasks.track(
                    self._pump(self._sub), name=f"brain:{self.character}:events"
                )
            while True:
                if pump is not None and pump.done():
                    raise RuntimeError("the brain's event pump stopped")
                await self._wait_open()
                stim = await self.arbiter.next(
                    idle_deadline=self._idle_deadline,
                    user_speaking=self.user_speaking,
                    speaking=self._speaking,
                    accept=self._accept,
                )
                if (
                    self.paused
                    or not self._speech_ready()
                    or (stim is not None and not self._accept(stim))
                ):
                    if stim is not None:  # frozen or voice lost while it was picked: put back
                        self.arbiter.restore(MergedContext(stim))
                    continue
                if stim is None:
                    if self._idle_deadline() is None:
                        continue
                    stim = self.deps.idle.make_stimulus(self.character, tuple(self._topics))
                ctx = self.arbiter.drain_context(stim)
                turn = _Turn(id=f"t-{uuid.uuid4().hex[:10]}", ctx=ctx, started=self._clock.now())
                outcome = await self._run_decision(turn)
                try:
                    await self._after(turn, outcome)
                except asyncio.CancelledError:
                    raise
                except Exception:  # bookkeeping must never stop the loop
                    log.exception("brain %s: after-decision step failed", self.character)
                    self._current = None
                    self._settled()
        finally:
            if pump is not None:
                pump.cancel()
            if self._sub is not None:
                self._sub.close()
                self._sub = None

    @property
    def ready(self) -> bool:
        """History and the prompt epoch are loaded (the loop takes stimuli)."""
        return self._prepared

    async def _prepare(self) -> None:
        if self._prepared:
            return
        try:
            epoch = self.deps.background.current_epoch()
        except RuntimeError:
            epoch = await self.deps.background.load()
        await self.deps.history.load(epoch.id)
        self._prepared = True
        if self.state == "booting":
            self.states.set("pre_show", reason="start")
        if not self.paused:  # a FREEZE during start-up stays in force
            self._open.set()
        if self.cfg.auto_live:
            self._go_live()

    async def _run_decision(self, turn: _Turn) -> DecisionOutcome:
        self._current = turn
        self.decisions += 1
        self.deps.background.cancel_active()
        self.deps.idle.on_activity(self._clock.now())
        watchdog = self._tasks.track(
            self._watchdog(turn.id), name=f"brain:{self.character}:watchdog"
        )
        try:
            self._refresh_state()
            return await self.slot.run(turn.id, self._decide(turn))
        finally:
            watchdog.cancel()

    async def _watchdog(self, turn_id: str) -> None:
        await self._clock.sleep(self.cfg.decision_watchdog_s)
        if self.slot.current_turn == turn_id:
            log.error("brain %s: decision %s hung; cancelling it", self.character, turn_id)
            self.slot.cancel("watchdog")

    async def _wait_open(self) -> None:
        while True:
            if not self._open.is_set():
                await self._open.wait()
                continue
            if self._speech_ready():
                return
            await self._clock.sleep(self.cfg.ready_poll_s)

    def _speech_ready(self) -> bool:
        try:
            return bool(self.deps.speech.ready())
        except Exception:
            return False

    def _accept(self, s: Stimulus) -> bool:
        if self.paused:
            return False
        if self.ending:
            return s.kind is StimulusKind.OPERATOR
        if not self.live:
            return s.kind in (StimulusKind.OPERATOR, StimulusKind.VOICE)
        return True

    def _idle_deadline(self) -> float | None:
        if not self.live or self.paused or self.ending or self._utts or self.user_speaking():
            return None
        return self.deps.idle.deadline()

    def _speaking(self) -> bool:
        """Her audio is still open (queued or playing); stale utterances are dropped."""
        now = self._clock.now()
        for utt in list(self._utts.values()):
            if utt.turn.decision_done and now - utt.last_activity > _STALE_UTTERANCE_S:
                log.warning(
                    "brain %s: no events for utterance %s; closing it", self.character, utt.utt_id
                )
                heard, interrupted = utt.tentative()
                self._utterance_done(
                    UtteranceDone(
                        character=self.character,
                        utt_id=utt.utt_id,
                        heard_text=heard,
                        cancelled=interrupted,
                        reason="stale",
                        filtered=utt.progress.filtered,
                    )
                )
        return bool(self._utts)

    # --- one decision ----------------------------------------------------------------------
    async def _decide(self, turn: _Turn) -> DecisionResult:
        primary = turn.primary
        tr = self.deps.trace
        tr.begin(turn.id, primary.kind.value, self.character)
        tr.mark(turn.id, "stimulus_in", primary.created)
        for key, stage in (("t_vad_end", "vad_end"), ("t_stt_final", "stt_final")):
            t = primary.payload.get(key)
            if isinstance(t, int | float) and not isinstance(t, bool):
                tr.mark(turn.id, stage, float(t))
        tr.mark(turn.id, "decision_start")
        try:
            provider = self.deps.router.active()
        except Exception:
            provider = "?"
        self._bus.publish(
            DecisionStarted(
                character=self.character,
                turn_id=turn.id,
                stimulus_id=primary.id,
                merged_ids=turn.ctx.merged_ids,
                provider=provider,
            )
        )
        if is_say(primary):
            return await self._say(turn)
        await self._settle(u for u in self._utts.values() if u.stop == "now")
        strict = self.strict()
        tctx = await self._gather(turn)
        turn.tctx = tctx
        req = self._build(turn, tctx, strict)
        voice = primary.kind is StimulusKind.VOICE
        utt_id = f"u-{turn.id[2:]}"
        tool_rounds = 0
        for k in range(1, self.deps.tool_flow.max_rounds + 1):
            rid = utt_id if k == 1 else f"{utt_id}/u{k}"
            rnd = await self._speak_round(turn, req, rid, voice and k == 1)
            result = rnd.result
            assert result is not None
            if k == 1:
                await self._check_provider(turn, result)
            done = result.done
            if not result.tool_calls or done is None or result.filtered or result.stalled:
                break
            message = self._assistant_message(rnd, result)
            rnd.assistant_message = message
            if strict or self.paused:
                rnd.tools = [(c, UNAVAILABLE_RESULT) for c in result.tool_calls]
                break
            rnd.tools = await self.deps.tool_flow.run(result.tool_calls, self._tool_context(turn))
            tool_rounds += 1
            if not self.deps.tool_flow.needs_follow_up(rnd.tools, result.spoke, rounds=k):
                break
            keep_extra = "gemini" in done.provider.casefold()
            follow = self.deps.tool_flow.tool_messages(message, rnd.tools, keep_extra=keep_extra)
            last_round = k + 1 >= self.deps.tool_flow.max_rounds
            req = dataclasses.replace(
                req,
                messages=(*req.messages, *follow),
                tool_choice="none" if last_round else req.tool_choice,
            )
        return DecisionResult(
            turn_id=turn.id,
            utt_id=utt_id,
            emitted_text=turn.emitted_text(),
            spoke=turn.sent_anything(),
            filtered=any(u.progress.filtered for u in turn.utts),
            provider=next(
                (r.result.done.provider for r in turn.rounds if r.result and r.result.done), None
            ),
            tool_rounds=tool_rounds,
        )

    async def _speak_round(
        self, turn: _Turn, req: ChatRequest, utt_id: str, filler: bool
    ) -> _Round:
        utt = await self._begin(turn, utt_id, filler)
        rnd = _Round(utt)
        turn.rounds.append(rnd)
        rnd.result = await self.deps.reply.run(
            character=self.character,
            utt_id=utt_id,
            turn_id=turn.id,
            events=self.deps.router.stream(req),
            progress=utt.progress,
        )
        utt.last_activity = self._clock.now()
        return rnd

    async def _begin(self, turn: _Turn, utt_id: str, filler: bool) -> _Utt:
        voice = turn.primary.kind is StimulusKind.VOICE
        utt = _Utt(utt_id, turn, voice, self._clock.now(), last_activity=self._clock.now())
        self._utts[utt_id] = utt
        try:
            async with deadline(self.cfg.speech_timeout_s, what="speech.begin", clock=self._clock):
                await self.deps.speech.begin(
                    utt_id,
                    self.character,
                    filler_after_s=self.cfg.filler_after_s if filler else None,
                )
        except BaseException:
            # it may be open on the voice side anyway (utterances play in begin order, so an
            # open one would block every later one): stop it, idempotently
            self._utts.pop(utt_id, None)
            self._tasks.track(self._stop(utt_id, "now", "begin_failed"), name="speech-stop")
            raise
        self._bus.publish(
            UtteranceStarted(
                character=self.character,
                turn_id=turn.id,
                utt_id=utt_id,
                stimulus_id=turn.primary.id,
            )
        )
        return utt

    async def _check_provider(self, turn: _Turn, result: ReplyResult) -> None:
        """The brain-freeze line when no provider could answer, once per outage (§2.8)."""
        done = result.done
        if done is not None and done.provider != "canned":
            self._freeze_played = False  # a real provider answered: the outage is over
            return
        spoken_by_router = done is not None  # the router's canned provider said the line
        if not spoken_by_router and not (result.provider_failed and not result.sent):
            return
        if self._freeze_played:
            return
        self._freeze_played = True
        self._bus.publish(
            Alert(
                level="error",
                message="LLM ไม่ตอบทุกตัว: เล่นประโยค 'สมองค้าง' แล้ว ตรวจสอบ llama-server",
                character=self.character,
                turn_id=turn.id,
            )
        )
        if spoken_by_router:
            return
        try:
            async with deadline(
                self.cfg.speech_timeout_s, what="brain-freeze line", clock=self._clock
            ):
                await self.deps.speech.play_canned("brain_freeze", self.character)
        except DeadlineExceeded:
            log.error("brain %s: the brain-freeze line did not start", self.character)

    def _assistant_message(self, rnd: _Round, result: ReplyResult) -> dict[str, Any]:
        """``Done.assistant_message`` with the spoken (gated) text as content; tool calls are
        kept verbatim (byte-stable arguments, provider extras)."""
        assert result.done is not None
        msg = assistant_message_copy(result.done.assistant_message, keep_extra=True)
        msg["content"] = rnd.utt.progress.emitted_text
        calls = msg.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != len(result.tool_calls):
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments},
                    **({"extra_content": dict(c.extra)} if c.extra else {}),
                }
                for c in result.tool_calls
            ]
        return msg

    def _tool_context(self, turn: _Turn) -> ToolContext:
        return ToolContext(
            character=self.character,
            turn_id=turn.id,
            stimulus=self._tool_stimulus(turn.ctx),
            memory=self.deps.memory,
            speech=self.deps.speech,
            avatar=self.deps.avatar_sink,
            channels=self.deps.channels,
            bus=self._bus,
            clock=self._clock,
        )

    @staticmethod
    def _tool_stimulus(ctx: MergedContext) -> Stimulus:
        """The stimulus tools see. A decision that merged chat content is CHAT-kind, so
        memory writes it triggers are quarantined (§6); its payload carries the chat block so
        ``remember(about=<viewer>)`` resolves the viewer."""
        primary = ctx.primary
        messages: list[ChatMessage] = []
        if ctx.chat is not None:
            messages.extend((*ctx.chat.must_ack, *ctx.chat.candidates))
        chatty = False
        for s in (*ctx.stimuli, *ctx.support):
            if s.kind in _CHATTY:
                chatty = True
            m = s.payload.get("message")
            if isinstance(m, ChatMessage):
                messages.append(m)
        chatty = chatty or bool(messages)
        payload = {**primary.payload, "chat": ctx.chat, "messages": tuple(messages)}
        kind = primary.kind
        if chatty and kind not in _CHATTY:
            kind = StimulusKind.CHAT
        return dataclasses.replace(primary, kind=kind, payload=payload)

    # --- context and prompt ----------------------------------------------------------------
    async def _gather(self, turn: _Turn) -> TurnContext:
        """Recall, viewer facts and new memories under one short deadline (§4.1 step 1)."""
        ctx = turn.ctx
        memory = self.deps.memory
        found: dict[str, list[MemoryItem]] = {"recall": [], "viewers": [], "new": []}
        query = " ".join(s.text for s in ctx.stimuli if s.text.strip()).strip()[:300]
        users = _chat_users(ctx)[: self.cfg.viewer_facts_max_viewers]

        async def fetch() -> None:
            found["new"] = [m for m in await memory.pending_since_epoch() if m.status == "active"]
            if self.cfg.recall_k > 0 and len(query) >= 3:
                hits = await memory.search(query, k=self.cfg.recall_k + 4)
                new_ids = {m.id for m in found["new"]}
                found["recall"] = [
                    m
                    for m in hits
                    if m.kind != "core" and m.status == "active" and m.id not in new_ids
                ][: self.cfg.recall_k]
            if users and self.cfg.viewer_facts_per_viewer > 0:
                facts = await memory.viewer_facts(
                    users, limit=self.cfg.viewer_facts_per_viewer * len(users)
                )
                found["viewers"] = [m for m in facts if m.status == "active"]

        try:
            async with deadline(self.cfg.gather_timeout_s, what="recall", clock=self._clock):
                await fetch()
        except DeadlineExceeded:
            log.debug(
                "brain %s: recall skipped (over %.0f ms)",
                self.character,
                self.cfg.gather_timeout_s * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("brain %s: recall failed: %r", self.character, exc)
        if found["new"]:
            self.deps.background.request_epoch_rebuild("memory")
        return TurnContext(
            stimulus=turn.primary,
            merged=ctx,
            recall=tuple(found["recall"]),
            viewer_facts=tuple(found["viewers"]),
            new_memories=tuple(found["new"]),
            game_state=None,
            notes=tuple(self._notes()),
            now_local=time.strftime("%H:%M", time.localtime(self._clock.wall())),
        )

    def _notes(self) -> list[str]:
        notes: list[str] = self._carry_notes
        self._carry_notes = []
        correction = self.deps.history.pending_correction()
        if correction:
            notes.append(correction)
        if self._last_filtered:
            notes.append(_FILTERED_NOTE)
            self._last_filtered = False
        opener = self._repeated_opener()
        if opener:
            notes.append(f"อย่าขึ้นต้นประโยคด้วย “{opener}” ซ้ำอีก")
        return notes

    def _repeated_opener(self) -> str | None:
        if len(self._openers) < 5:
            return None
        top, count = collections.Counter(self._openers).most_common(1)[0]
        if top and count >= 3 and count / len(self._openers) > self.cfg.opener_repeat_ratio:
            return top
        return None

    def _build(self, turn: _Turn, tctx: TurnContext, strict: bool) -> ChatRequest:
        history = self.deps.history
        for utt in self._utts.values():  # frozen at first render: heard so far (§4.1)
            if utt.history_id is not None:
                heard, interrupted = utt.tentative()
                history.mark_heard(
                    utt.history_id,
                    heard,
                    interrupted=interrupted,
                    filtered=utt.progress.filtered,
                    final=False,
                )
        req = self.deps.prompt.build(
            self.deps.background.current_epoch(),
            history.turns(),
            tctx,
            purpose="speak",
            temperature=self.cfg.strict_temperature if strict else self.cfg.temperature,
            max_tokens=self.cfg.max_reply_tokens,
            tool_choice="none" if strict else "auto",
            turn_id=turn.id,
        )
        history.freeze_all()
        self.deps.trace.mark(turn.id, "prompt_built")
        if self.deps.prompt.last_stats.get("needs_compaction"):
            self.deps.background.request_epoch_rebuild("history")
        return req

    # --- operator SAY ----------------------------------------------------------------------
    async def _say(self, turn: _Turn) -> DecisionResult:
        """SAY bypasses the LLM; the output gate only warns (§4.2)."""
        text = turn.primary.text
        try:
            async with deadline(1.0, what="SAY check", clock=self._clock):
                res = await self.deps.safety.check_output(
                    text, character=self.character, prev_tail=""
                )
            if res.verdict in (Verdict.BLOCK, Verdict.DROP):
                self._bus.publish(
                    Alert(
                        level="warn",
                        message=f"ข้อความ SAY มีคำที่ตัวกรองจับได้ ({res.category or res.rule}) แต่พูดตามคำสั่ง",
                        character=self.character,
                    )
                )
        except DeadlineExceeded:
            log.warning("brain %s: SAY output check timed out", self.character)
        utt_id = f"u-{turn.id[2:]}"
        utt = await self._begin(turn, utt_id, filler=False)
        turn.rounds.append(_Round(utt))
        cfg = ChunkerConfig.from_constraints(self.deps.speech.constraints(self.character))
        chunker = ThaiSpeechChunker(cfg, word_tokenize=no_word_split, clock=self._clock)
        chunks = [c for c in (*chunker.feed(text), *chunker.flush()) if c.strip()]
        normalize = self.deps.reply.normalizer
        progress = utt.progress
        for i, chunk in enumerate(chunks):
            seg = Segment(
                utt_id=utt_id,
                seq=i,
                text=normalize(chunk),
                caption=chunk,
                last=i == len(chunks) - 1,
                kind="operator",
            )
            if not await self._queue_segment(seg):
                break
            progress.sent.append(seg)
            progress.emitted.append(chunk)
        else:
            if not chunks:
                await self._queue_segment(Segment(utt_id, 0, "", "", last=True, kind="operator"))
            progress.ended = True
        return DecisionResult(
            turn_id=turn.id,
            utt_id=utt_id,
            emitted_text=progress.emitted_text,
            spoke=bool(progress.sent),
        )

    async def _queue_segment(self, seg: Segment) -> bool:
        started = self._clock.now()
        while True:
            try:
                async with deadline(
                    self.cfg.speech_timeout_s, what="speech.segment", clock=self._clock
                ):
                    if await self.deps.speech.segment(seg):
                        return True
            except DeadlineExceeded:
                return False
            if self._clock.now() - started > 30.0:
                return False
            await self._clock.sleep(0.05)

    # --- after a decision ------------------------------------------------------------------
    async def _after(self, turn: _Turn, outcome: DecisionOutcome) -> None:
        turn.decision_done = True
        self._current = None
        reason = outcome.reason or ""
        if outcome.status == "aborted":
            turn.outcome = f"aborted:{reason}"
            self._bus.publish(
                DecisionAborted(character=self.character, turn_id=turn.id, reason=reason)
            )
            ran_tools = any(r.tools for r in turn.rounds)
            if reason not in _DROP and not turn.audible() and not ran_tools:
                # §4.3b: nothing was heard (and no tool ran, which must never run twice):
                # merge everything into the restarted decision
                for utt in turn.utts:
                    self._stop_utt(utt, "now", reason)
                self.arbiter.restore(turn.ctx)
                if turn.tctx is not None:  # its tail notes were never heard: keep them
                    self._carry_notes = [*turn.tctx.notes, *self._carry_notes]
                self._maybe_finish_turn(turn)
                self._settled()
                return
            if reason == "watchdog":
                self._watchdog_tripped(turn)
        elif outcome.status == "failed":
            turn.outcome = "failed"
            log.error(
                "brain %s: decision %s failed: %s",
                self.character,
                turn.id,
                reason,
                exc_info=outcome.error,
            )
            self._bus.publish(
                DecisionAborted(
                    character=self.character, turn_id=turn.id, reason=f"error: {reason}"
                )
            )
        for utt in turn.utts:  # a decision that did not end its utterance: close it
            if utt.done is None and utt.stop is None and not utt.progress.ended:
                self._stop_utt(utt, "after_segment", reason or "decision_end")
        await self._settle(u for u in turn.utts if u.stop == "now")
        if self._worth_recording(turn, outcome):
            await self._record(turn)
        self._bookkeep(turn)
        self._maybe_finish_turn(turn)
        self._settled()

    def _worth_recording(self, turn: _Turn, outcome: DecisionOutcome) -> bool:
        if turn.sent_anything() or any(r.tools for r in turn.rounds):
            return True
        if outcome.status != "ok":
            return False
        first = turn.rounds[0].result if turn.rounds else None
        if first is not None and first.provider_failed and first.done is None:
            return False  # nothing could answer: not a turn
        return bool(turn.rounds)

    async def _record(self, turn: _Turn) -> None:
        """Append the compact user line, then each round's assistant line and tool results."""
        history = self.deps.history
        tctx = turn.tctx or TurnContext(turn.primary, turn.ctx)
        await history.append_user(
            self.deps.prompt.compact_user(tctx), turn.primary, turn_ref=turn.id
        )
        for rnd in turn.rounds:
            utt = rnd.utt
            progress = utt.progress
            done = rnd.result.done if rnd.result is not None else None
            calls_json: str | None = None
            if rnd.tools and rnd.assistant_message is not None:
                calls = rnd.assistant_message.get("tool_calls")
                if calls:
                    calls_json = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
            extra_json: str | None = None
            if done is not None:
                extra = {
                    k: v
                    for k, v in done.assistant_message.items()
                    if k not in ("role", "content", "tool_calls")
                }
                if extra:
                    extra_json = json.dumps(extra, ensure_ascii=False, default=str)
            final = utt.done
            filtered = progress.filtered or (final is not None and final.filtered)
            hid = await history.append_assistant(
                turn_ref=utt.utt_id,
                emitted=progress.emitted_text,
                heard=_heard_text(utt, final) if final is not None else None,
                interrupted=final.cancelled if final is not None else False,
                filtered=filtered,
                provider=done.provider if done is not None else None,
                tool_calls=calls_json,
                provider_extra=extra_json,
            )
            if utt.done is None:
                utt.history_id = hid
            elif final is None:  # UtteranceDone arrived while the line was being stored
                self._apply_heard(hid, utt, utt.done)
            for call, result in rnd.tools:
                await history.append_tool(result.content, turn_ref=call.id)

    def _bookkeep(self, turn: _Turn) -> None:
        text = turn.emitted_text().strip()
        filtered = any(u.progress.filtered for u in turn.utts)
        if filtered:
            self._last_filtered = True
            if self.deps.avatar is not None:
                with contextlib.suppress(Exception):
                    self.deps.avatar.set_emotion("neutral")
            users = frozenset(_chat_users(turn.ctx))
            if users:
                self._filtered_chatters.append((self._clock.now(), users))
        if text and not is_say(turn.primary):
            self._openers.append(TurnTraceRecorder.opener(text))
            self._topics.append(" ".join(text.split())[:60])
            self.deps.trace.set(turn.id, opener=TurnTraceRecorder.opener(text))
        self.deps.trace.set(turn.id, outcome=turn.outcome)

    def _watchdog_tripped(self, turn: _Turn) -> None:
        self._bus.publish(
            Alert(
                level="error",
                message="การตัดสินใจค้างเกิน 30 วินาที ถูกยกเลิกแล้ว (บันทึก flight recorder)",
                character=self.character,
                turn_id=turn.id,
            )
        )
        if self.deps.dump_flight is not None:
            self._tasks.track(self._dump_flight("watchdog"), name="flight-dump")

    async def _dump_flight(self, why: str) -> None:
        assert self.deps.dump_flight is not None
        try:
            async with deadline(
                self.cfg.flight_dump_timeout_s, what="flight dump", clock=self._clock
            ):
                await self.deps.dump_flight(why)
        except DeadlineExceeded:
            log.error("brain %s: the flight dump timed out", self.character)

    def _maybe_finish_turn(self, turn: _Turn) -> None:
        if turn.finished or not turn.decision_done:
            return
        if any(u.done is None for u in turn.utts):
            return
        turn.finished = True
        self.deps.trace.set(turn.id, outcome=turn.outcome)
        self.deps.trace.finish(turn.id)

    def _settled(self) -> None:
        """After a decision or an utterance: free the arbiter, arm idle, refresh the state."""
        if not self._utts and not self.slot.busy:
            now = self._clock.now()
            self.arbiter.mark_free(now)
            if self.live and not self.paused and not self.user_speaking():
                self.deps.idle.on_playback_end(now)
        self._refresh_state()
        self.arbiter.poke()

    async def _settle(self, utts: Iterable[_Utt]) -> None:
        """Wait up to 150 ms for ``UtteranceDone`` of utterances cut ``now`` (§4.1)."""
        pending = [u for u in utts if u.done is None]
        if not pending:
            return
        waiter = asyncio.ensure_future(asyncio.gather(*(u.finished.wait() for u in pending)))
        try:
            await wait_future(waiter, self._clock, self.cfg.critical_cut_wait_ms / 1000.0)
        finally:
            waiter.cancel()  # never awaited here: that could swallow our own cancellation

    # --- speech control ----------------------------------------------------------------------
    def _stop_utt(self, utt: _Utt, mode: StopMode, reason: str) -> None:
        if utt.done is not None or utt.stop == "now" or (utt.stop == mode):
            return
        utt.stop = mode
        self._tasks.track(self._stop(utt.utt_id, mode, reason), name="speech-stop")

    async def _stop(self, utt_id: str | None, mode: StopMode, reason: str) -> None:
        try:
            async with deadline(self.cfg.fast_timeout_s, what="speech.stop", clock=self._clock):
                await self.deps.speech.stop(utt_id, mode, reason)
        except DeadlineExceeded:
            log.error("brain %s: speech.stop(%s, %s) timed out", self.character, utt_id, mode)

    # --- events ------------------------------------------------------------------------------
    async def _pump(self, sub: Subscription) -> None:
        async for ev in sub:
            try:
                self.on_event(ev)
            except Exception:
                log.exception("brain %s: handling %s failed", self.character, type(ev).__name__)

    def on_event(self, ev: Event) -> None:
        """Handle one bus event (the pump calls this; tests and a custom wiring may too)."""
        if isinstance(ev, SegmentStarted):
            self._segment_started(ev)
        elif isinstance(ev, SegmentDone):
            utt = self._utts.get(ev.utt_id)
            if utt is not None:
                utt.heard[ev.seq] = ev.heard_text
                utt.last_activity = self._clock.now()
                if ev.heard:
                    self.deps.trace.mark(utt.turn.id, "last_audible", ev.ts or None)
        elif isinstance(ev, UtteranceDone):
            self._utterance_done(ev)
        elif isinstance(ev, HealthChanged):
            health = ev.health
            if health.component.split(":", 1)[0] == "voice" and health.state in _VOICE_LOST:
                self._voice_lost(health.state.value)
        elif ev.character not in (None, self.character):
            return
        elif isinstance(ev, UserSpeechStarted):
            self._user_since = self._clock.now()
            self.states.set_user_speaking(True)
            self.deps.idle.on_activity(self._user_since)
            self.arbiter.poke()
        elif isinstance(ev, UserSpeechEnded):
            self._user_since = None
            self.states.set_user_speaking(False)
            self._settled()
        elif isinstance(ev, UserTranscript):
            self.intake.on_transcript(ev)
        elif isinstance(ev, BargeInConfirmed):
            self._barge_in(ev)

    def _segment_started(self, ev: SegmentStarted) -> None:
        utt = self._utts.get(ev.utt_id)
        if utt is None:
            return
        utt.started[ev.seq] = ev.caption
        utt.last_activity = self._clock.now()
        if not ev.caption.strip():
            return
        if not utt.audible:
            utt.audible = True
            self.deps.trace.mark(utt.turn.id, "first_audible", ev.t_audible or None)
            self.deps.trace.set(utt.turn.id, tts_backend=ev.backend)
        if ev.emotion:
            if self.deps.avatar is not None:
                with contextlib.suppress(Exception):
                    self.deps.avatar.set_emotion(ev.emotion)
            self._bus.publish(
                EmotionChanged(character=self.character, turn_id=utt.turn.id, emotion=ev.emotion)
            )
        self._refresh_state()

    def _utterance_done(self, ev: UtteranceDone) -> None:
        utt = self._utts.pop(ev.utt_id, None)
        if utt is None:
            return
        utt.done = ev
        utt.finished.set()
        now = self._clock.now()
        if utt.history_id is not None:
            self._apply_heard(utt.history_id, utt, ev)
        if ev.cancelled and self.deps.avatar is not None:
            with contextlib.suppress(Exception):
                self.deps.avatar.on_cut(ev.utt_id, now)
        if utt.voice:
            self.intake.mark_spoke_to_streamer(now)
        self.deps.trace.mark(utt.turn.id, "done", ev.ts or None)
        self._maybe_finish_turn(utt.turn)
        self._settled()

    def _apply_heard(self, history_id: int, utt: _Utt, ev: UtteranceDone) -> None:
        audit = self.deps.history.mark_heard(
            history_id,
            _heard_text(utt, ev),
            interrupted=ev.cancelled,
            filtered=ev.filtered or utt.progress.filtered,
            final=True,
        )
        if audit is not None:
            self._tasks.track(self.deps.history.persist_audit(audit), name="history-heard")

    def _voice_lost(self, why: str) -> None:
        """The voice worker went down: its queue is gone. Cancel the decision in flight (its
        stimuli come back if nothing was heard) and close every open utterance, keeping only
        what was heard (§2.8 ``voice_restart``)."""
        if not self._utts and not self.slot.busy:
            return
        log.warning("brain %s: voice worker %s; closing open utterances", self.character, why)
        if self.slot.busy:
            self.slot.cancel("voice_restart")
        for utt in list(self._utts.values()):
            utt.stop = "now"
            heard, _ = utt.tentative()
            self._utterance_done(
                UtteranceDone(
                    character=self.character,
                    utt_id=utt.utt_id,
                    heard_text=heard,
                    cancelled=True,
                    reason="voice_restart",
                    filtered=utt.progress.filtered,
                )
            )

    def _barge_in(self, ev: BargeInConfirmed) -> None:
        """A confirmed barge-in: treated as CRITICAL (§4.7 step 3)."""
        if self.slot.busy:
            self.slot.cancel("barge_in")
        for utt in list(self._utts.values()):
            self._stop_utt(utt, "now", "barge_in")

    def _refresh_state(self) -> None:
        target: BrainState
        if self.paused:
            target = "paused"
        elif self._current is not None:
            target = "speaking" if self._current.audible() else "deciding"
        elif self._utts:
            target = "speaking" if any(u.audible for u in self._utts.values()) else "deciding"
        else:
            target = "idle" if self.live else "pre_show"
        self.states.set(target)

    async def _note(self, text: str) -> None:
        await self.deps.history.append_note(text, source="voice")

    # --- operator control --------------------------------------------------------------------
    async def control(self, cmd: OpCommand) -> OpResult:
        """Run one operator command for this character; never raises except cancellation."""
        t0 = self._clock.now()
        handler = self._handlers().get(cmd.kind)
        if handler is None:
            return OpResult(False, f"not implemented in the brain: {cmd.kind.value}")
        try:
            result = await handler(cmd)
        except asyncio.CancelledError:
            raise
        except DeadlineExceeded as exc:
            result = OpResult(False, f"timeout: {exc.what}")
        except (ValueError, KeyError, TypeError) as exc:
            result = OpResult(False, str(exc))
        except Exception as exc:
            log.exception("brain %s: %s failed", self.character, cmd.kind.value)
            result = OpResult(False, f"{type(exc).__name__}: {exc}")
        return OpResult(result.ok, result.detail, round((self._clock.now() - t0) * 1000.0, 2))

    def _handlers(self) -> Mapping[OpKind, Callable[[OpCommand], Awaitable[OpResult]]]:
        return {
            OpKind.FREEZE: self._op_freeze,
            OpKind.SKIP: self._op_skip,
            OpKind.MUTE: self._op_mute,
            OpKind.UNMUTE: self._op_mute,
            OpKind.RESUME: self._op_resume,
            OpKind.GO_LIVE: self._op_go_live,
            OpKind.CHAT_INTAKE: self._op_chat_intake,
            OpKind.SAY: self._op_operator_text,
            OpKind.DIRECT: self._op_operator_text,
            OpKind.MIC_MODE: self._op_mic_mode,
            OpKind.PTT: self._op_ptt,
            OpKind.LLM_USE: self._op_llm_use,
            OpKind.LLM_ROLLBACK: self._op_llm_rollback,
            OpKind.STRICT: self._op_strict,
            OpKind.FILTER_RELOAD: self._op_filter_reload,
            OpKind.MUTE_USER: self._op_mute_user,
            OpKind.END_STREAM: self._op_end_stream,
        }

    def freeze_now(self) -> None:
        """FREEZE's synchronous part (I4 fast path): PAUSED, cancel the LLM, chat and tools
        off, avatar neutral. :meth:`control` then stops speech within 150 ms."""
        if not self.paused:
            self._chat_before_freeze = self.intake.chat_intake
        self.paused = True
        self._open.clear()
        self.intake.set_chat_intake(False)
        self.slot.cancel("operator_freeze")
        self.deps.background.cancel_active()
        for utt in self._utts.values():
            utt.stop = "now"
        self._refresh_state()
        self.arbiter.poke()

    async def _op_freeze(self, cmd: OpCommand) -> OpResult:
        self.freeze_now()
        targets: list[str | None] = [None] if cmd.character is None else list(self._utts)
        async with deadline(
            self.cfg.freeze_timeout_s, what="freeze speech.stop", clock=self._clock
        ):
            for utt_id in targets:
                await self.deps.speech.stop(utt_id, "now", "operator_freeze")
        return OpResult(True, "frozen")

    async def _op_skip(self, cmd: OpCommand) -> OpResult:
        targets = list(self._utts.values())
        if not targets:
            return OpResult(False, "nothing is playing")
        current = self._current
        if current is not None and any(u.turn is current for u in targets):
            self.slot.cancel("operator_skip")
        async with deadline(self.cfg.fast_timeout_s, what="skip speech.stop", clock=self._clock):
            for utt in targets:
                utt.stop = "now"
                await self.deps.speech.stop(utt.utt_id, "now", "operator_skip")
        return OpResult(True, f"skipped {len(targets)}")

    async def _op_mute(self, cmd: OpCommand) -> OpResult:
        on = cmd.kind is OpKind.MUTE
        async with deadline(self.cfg.fast_timeout_s, what="speech.mute", clock=self._clock):
            await self.deps.speech.mute(on)
        self.muted = on
        return OpResult(True, "muted" if on else "unmuted")

    async def _op_resume(self, cmd: OpCommand) -> OpResult:
        if not self.paused:
            return OpResult(True, "not paused")
        self.arbiter.clear(keep=frozenset({StimulusKind.SUPPORT}))
        self.paused = False
        restore = self._chat_before_freeze
        self._chat_before_freeze = None
        self.intake.set_chat_intake(self.live if restore is None else restore)
        self._open.set()
        self._settled()
        return OpResult(True, "resumed")

    def _go_live(self) -> None:
        self.live = True
        self.ending = False
        chat_on = bool(self.intake.cfg.chat_intake)
        if self.paused:
            self._chat_before_freeze = chat_on
        else:
            self.intake.set_chat_intake(chat_on)
        self._settled()

    async def _op_go_live(self, cmd: OpCommand) -> OpResult:
        self._go_live()
        return OpResult(True, "live")

    async def _op_chat_intake(self, cmd: OpCommand) -> OpResult:
        on = bool(cmd.args.get("on", True))
        if self.paused:
            self._chat_before_freeze = on
        else:
            self.intake.set_chat_intake(on)
        return OpResult(True, "chat on" if on else "chat off")

    async def _op_operator_text(self, cmd: OpCommand) -> OpResult:
        if self.paused:
            return OpResult(False, "paused: RESUME first")
        text = cmd.args.get("text")
        if not isinstance(text, str) or not text.strip():
            return OpResult(False, f"{cmd.kind.value} needs a non-empty 'text'")
        stim = self.intake.on_operator("say" if cmd.kind is OpKind.SAY else "direct", text)
        return OpResult(True, stim.id)

    async def _op_mic_mode(self, cmd: OpCommand) -> OpResult:
        mode = cmd.args.get("mode")
        self.intake.set_mic_mode(cast(Any, mode))
        return OpResult(True, str(mode))

    async def _op_ptt(self, cmd: OpCommand) -> OpResult:
        self.intake.set_ptt(bool(cmd.args.get("active")))
        return OpResult(True, "ptt down" if self.intake.ptt_active else "ptt up")

    async def _op_llm_use(self, cmd: OpCommand) -> OpResult:
        name = cmd.args.get("name")
        if not isinstance(name, str) or not name:
            return OpResult(False, "llm_use needs a provider 'name'")
        try:
            self.deps.router.promote(name)
        except KeyError:
            return OpResult(False, f"unknown provider {name!r}")
        except Exception as exc:  # e.g. ConsentRequired for a cloud provider
            return OpResult(False, str(exc))
        return OpResult(True, f"active at the next decision: {name}")

    async def _op_llm_rollback(self, cmd: OpCommand) -> OpResult:
        before = self.deps.router.active()
        self.deps.router.rollback()
        after = self.deps.router.active()
        return OpResult(True, f"{before} -> {after}" if after != before else "nothing to roll back")

    async def _op_strict(self, cmd: OpCommand) -> OpResult:
        on = bool(cmd.args.get("on", True))
        setter = getattr(self.deps.safety, "set_strict", None)
        if callable(setter):
            try:
                setter(self.character, on, reason="operator")
            except TypeError:
                setter(self.character, on)
            self._strict_local = False
        else:
            self._strict_local = on
        return OpResult(True, "strict on" if on else "strict off")

    async def _op_filter_reload(self, cmd: OpCommand) -> OpResult:
        async with deadline(
            self.cfg.filter_reload_timeout_s, what="filter reload", clock=self._clock
        ):
            try:
                await asyncio.to_thread(self.deps.safety.reload)
            except Exception as exc:  # FilterListError: the old lists stay active
                return OpResult(False, f"filter lists not reloaded: {exc}")
        return OpResult(True, "filters reloaded")

    async def _op_mute_user(self, cmd: OpCommand) -> OpResult:
        args = cmd.args
        seconds = args.get("seconds")
        ok = self.intake.mute_user(
            platform=str(args["platform"]) if args.get("platform") else None,
            user_id=str(args["user_id"]) if args.get("user_id") else None,
            name=str(args["name"]) if args.get("name") else None,
            seconds=float(seconds) if isinstance(seconds, int | float) else None,
        )
        return OpResult(ok, "muted" if ok else "mute_user needs 'user_id' or 'name'")

    async def _op_end_stream(self, cmd: OpCommand) -> OpResult:
        """§2.9: stop chat intake, stop new decisions, let the utterance finish (≤ 5 s), then
        queue the episode summary (it runs in the background)."""
        self.intake.set_chat_intake(False)
        self.ending = True
        self.live = False
        self.arbiter.poke()
        end = self._clock.now() + self.cfg.end_stream_wait_s
        while (self.slot.busy or self._utts) and self._clock.now() < end:
            await self._clock.sleep(0.1)
        self._tasks.track(self._end_session(), name=f"brain:{self.character}:end-session")
        self.ending = False
        self._settled()
        return OpResult(True, "stream ended; episode summary queued")

    async def _end_session(self) -> None:
        try:
            await self.deps.background.end_session()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("brain %s: ending the session failed", self.character)


def _letters(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _heard_text(utt: _Utt, ev: UtteranceDone) -> str:
    """The heard text of a finished utterance. When all of it was heard, the brain's own
    emitted text is exact (spacing included); otherwise the voice side's truncation counts."""
    emitted = utt.progress.emitted_text
    if not ev.cancelled and _letters(ev.heard_text) == _letters(emitted):
        return emitted
    return ev.heard_text


def _chat_users(ctx: MergedContext) -> list[tuple[str, str]]:
    """``(platform, user_id)`` of the chat authors in a decision, in order, deduplicated."""
    out: dict[tuple[str, str], None] = {}
    messages: list[ChatMessage] = []
    if ctx.chat is not None:
        messages.extend((*ctx.chat.must_ack, *ctx.chat.candidates))
    for s in (*ctx.stimuli, *ctx.support):
        m = s.payload.get("message")
        if isinstance(m, ChatMessage):
            messages.append(m)
    for m in messages:
        out.setdefault((m.user.platform.value, m.user.id), None)
    return list(out)
