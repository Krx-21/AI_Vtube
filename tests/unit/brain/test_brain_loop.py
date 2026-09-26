"""Brain: the serial loop, merging, preemption, state machine, operator control (§4.1–§4.13)."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from brain_testkit import BrainRig, run_real, stim

from aivtube.contracts.control import OpCommand, OpKind
from aivtube.contracts.events import (
    Alert,
    BargeInConfirmed,
    DecisionAborted,
    DecisionStarted,
    Event,
    Filtered,
    HealthChanged,
    StateChanged,
    UserSpeechEnded,
    UserSpeechStarted,
    UtteranceDone,
)
from aivtube.contracts.llm import ChatRequest, Done, LLMEvent, TextDelta
from aivtube.contracts.memory import Turn
from aivtube.contracts.types import Health, HealthState, Priority, Rank, Stimulus, StimulusKind
from aivtube.testing.fakes import (
    FakeClock,
    FakeEventBus,
    FakeLLM,
    FakeReply,
    FakeSpeechOutput,
    FakeTool,
    make_chat_message,
    tool_call,
)


def roles(turns: list[Turn]) -> list[str]:
    return [t.role for t in turns if not (t.role == "note" and t.source == "heard")]


async def test_voice_turn_end_to_end(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock, script=[FakeReply("สวัสดี", "[happy] สวัสดีค่ะ วันนี้เป็นยังไงบ้าง")])
    await rig.start()
    try:
        rig.say("สวัสดีไพลิน")
        await fake_clock.run_until(lambda: rig.speech.heard_texts() != [], within=5)
        await rig.idle_out()
        assert len(rig.llm.requests) == 1
        req = rig.llm.requests[0]
        assert req.slot == 0 and req.purpose == "speak"
        assert "[สตรีมเมอร์] “สวัสดีไพลิน”" in rig.last_user()
        assert rig.speech.heard_texts() == ["สวัสดีค่ะ", "วันนี้เป็นยังไงบ้าง"]
        # history: compact user line, then the heard assistant text
        turns = rig.history.turns()
        assert roles(turns) == ["user", "assistant"]
        assert turns[0].text == "[สตรีมเมอร์] สวัสดีไพลิน"
        assert turns[1].heard_text == "สวัสดีค่ะ วันนี้เป็นยังไงบ้าง"
        # state machine: pre_show → idle → deciding → speaking → idle; avatar poses follow
        states = [(e.old, e.new) for e in rig.bus.of_type(StateChanged)]
        assert states[:5] == [
            ("booting", "pre_show"),
            ("pre_show", "idle"),
            ("idle", "deciding"),
            ("deciding", "speaking"),
            ("speaking", "idle"),
        ]
        assert "thinking" in rig.avatar.states and "speaking" in rig.avatar.states
        assert rig.avatar.emotions[:1] == ["happy"]
        [done] = rig.bus.of_type(UtteranceDone)
        assert not done.cancelled
        assert rig.llm.max_in_flight == 1
        trace = rig.trace.recent(1)[0]
        assert trace["kind"] == "voice" and trace["ttfa_ms"] is not None
        assert {"decision_start", "prompt_built", "first_audible", "done"} <= set(trace["stages"])
    finally:
        await rig.stop()


async def test_pre_show_accepts_only_operator_and_voice(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock, script=[FakeReply("", "โอเคค่ะ", repeat=True)])
    await rig.start(live=False)
    try:
        assert rig.brain.state == "pre_show"
        rig.brain.on_chat(make_chat_message("ไพลินอยู่ไหม", user="tom", clock=fake_clock))
        await fake_clock.run_for(10)
        assert rig.llm.requests == []  # chat intake is off before Go Live
        rig.say("ทดสอบเสียง")
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 1, within=5)
        await rig.idle_out()
        assert rig.brain.state == "pre_show"
        await fake_clock.run_for(60)
        assert len(rig.llm.requests) == 1  # no idle turns before Go Live
        assert len(rig.bus.of_type(DecisionStarted)) == 1
    finally:
        await rig.stop()


async def test_run_propagates_cancellation(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock, script=[FakeReply("", "ยาวมาก " * 30)])
    await rig.start()
    rig.say("เล่าอะไรยาวๆ หน่อย")
    await fake_clock.run_until(lambda: rig.brain.slot.busy, within=5)
    rig.task.cancel()
    results = await asyncio.gather(rig.task, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    await fake_clock.run_until_idle()
    assert rig.llm.in_flight == 0 and rig.llm.closed_early == 1
    await rig.stop()


# --- helpers ----------------------------------------------------------------------------------


class TimedLLM(FakeLLM):
    """FakeLLM that records when each request started (clock time)."""

    def __init__(self, script: list[FakeReply], clock: FakeClock, **kw: Any) -> None:
        super().__init__(script, clock=clock, **kw)
        self._clock = clock
        self.times: list[float] = []

    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        self.times.append(self._clock.now())
        return super().stream(req)


def mk(
    clock: FakeClock,
    kind: StimulusKind,
    text: str,
    *,
    priority: Priority | None = None,
    rank: Rank | None = None,
    payload: dict[str, Any] | None = None,
    speaker: str | None = None,
) -> Stimulus:
    return stim(
        kind,
        text,
        created=clock.now(),
        priority=priority,
        rank=rank,
        payload=payload,
        speaker=speaker,
    )


LONG = "วันนี้ไพลินอยากเล่าเรื่องยาวมากๆ ให้ทุกคนฟัง เรื่องแมวที่บ้านที่ชอบนอนบนคีย์บอร์ด แล้วก็กดปุ่มมั่วไปหมดเลย จนเกมพังไปสองรอบ"


def stops(rig: BrainRig) -> list[tuple[Any, ...]]:
    return [args for name, args in rig.speech.calls if name == "stop"]


# --- merging (§4.3) -----------------------------------------------------------------------------


async def test_chat_mid_decision_is_merged_into_the_next_one(fake_clock: FakeClock) -> None:
    rig = BrainRig(
        fake_clock,
        script=[
            FakeReply("เล่าเรื่องแมว", "แมวชื่อส้มค่ะ"),
            FakeReply("", "หวัดดีจ้า", repeat=True),
        ],
    )
    await rig.start()
    try:
        rig.say("เล่าเรื่องแมวหน่อย")
        await fake_clock.run_until(lambda: rig.brain.slot.busy, within=5)
        rig.brain.on_chat(make_chat_message("สวัสดีครับทุกคน", user="tom", clock=fake_clock))
        await fake_clock.run_for(0.1)
        assert rig.brain.slot.busy and rig.llm.closed_early == 0  # not interrupted
        await rig.idle_out()
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=10)
        await rig.idle_out()
        assert "สวัสดีครับทุกคน" not in rig.last_user(0)
        assert "สวัสดีครับทุกคน" in rig.last_user(1) and '<chat untrusted="true">' in rig.last_user(1)
        assert rig.llm.closed_early == 0 and rig.llm.max_in_flight == 1
    finally:
        await rig.stop()


async def test_mention_and_voice_mid_decision_merge(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock, script=[FakeReply("", "ค่ะ", repeat=True, ttft_s=1.0)])
    await rig.start()
    try:
        rig.say("เรื่องแรก")
        await fake_clock.run_until(lambda: rig.brain.slot.busy, within=5)
        rig.brain.on_chat(make_chat_message("ไพลินกินข้าวยัง", user="ann", clock=fake_clock))
        rig.say("เรื่องที่สอง")  # same rank as the decision in flight: merge, no abort
        await fake_clock.run_for(0.2)
        assert rig.llm.closed_early == 0
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=20)
        tail = rig.last_user(1)
        assert "เรื่องที่สอง" in tail and "ไพลินกินข้าวยัง" in tail  # both in one decision
        assert "เรื่องแรก" not in tail
        await rig.idle_out()
        assert len(rig.llm.requests) == 2 and rig.llm.max_in_flight == 1
    finally:
        await rig.stop()


# --- priorities while SPEAKING (§4.4) -----------------------------------------------------------


async def speaking_rig(clock: FakeClock, *, decision_running: bool = False) -> BrainRig:
    llm = TimedLLM(
        [FakeReply("เล่ายาว", LONG), FakeReply("", "รับทราบค่ะ", repeat=True)],
        clock,
        ttft_s=0.2,
        tok_s=45.0 if decision_running else 1000.0,
    )
    rig = BrainRig(clock, llm=llm)
    await rig.start()
    rig.say("เล่ายาวๆ หน่อย")
    await clock.run_until(
        lambda: rig.brain.state == "speaking" and rig.brain.slot.busy is decision_running,
        within=10,
    )
    return rig


def first_done(rig: BrainRig) -> UtteranceDone:
    return rig.bus.of_type(UtteranceDone)[0]


async def test_low_while_speaking_waits_for_utterance_done(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        rig.brain.submit(mk(fake_clock, StimulusKind.MENTION, "ไพลินชอบสีอะไร", speaker="bo"))
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=60)
        assert stops(rig) == []  # LOW never stops speech
        done = first_done(rig)
        assert not done.cancelled
        assert rig.llm.times[1] >= done.ts  # decided only after UtteranceDone
    finally:
        await rig.stop()


async def test_medium_while_speaking_stops_after_segment_then_decides(
    fake_clock: FakeClock,
) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        support = mk(
            fake_clock,
            StimulusKind.SUPPORT,
            "เป็นกำลังใจให้นะ",
            speaker="rich",
            payload={"msg_kind": "donation", "amount": 50, "currency": "THB"},
        )
        rig.brain.submit(support)
        await fake_clock.run_until_idle()
        [(utt, mode, reason, _)] = stops(rig)
        assert (mode, reason) == ("after_segment", "preempt_medium")
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=60)
        done = first_done(rig)
        assert done.cancelled and done.utt_id == utt
        assert rig.llm.times[1] >= done.ts  # the current segment finished first
        # history: the cut reply is recorded as heard, marked interrupted
        await rig.idle_out()
        assistant = next(t for t in rig.history.turns() if t.role == "assistant")
        assert (
            assistant.interrupted
            and assistant.heard_text
            and LONG.startswith(assistant.heard_text.strip()[:10])
        )
    finally:
        await rig.stop()


async def test_high_while_speaking_starts_the_next_decision_at_once(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        rig.say("หยุดก่อน ฟังนี่")
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=10)
        [(_, mode, reason, _)] = stops(rig)
        assert (mode, reason) == ("after_segment", "preempt_high")
        assert not rig.bus.of_type(UtteranceDone)  # in parallel with the current segment
        # the new prompt renders the first reply as heard so far (frozen), marked interrupted
        history_msgs = [m for m in rig.llm.requests[1].messages if m["role"] == "assistant"]
        assert history_msgs[-1]["content"].endswith("[ถูกขัดจังหวะ]")
        await rig.idle_out()
    finally:
        await rig.stop()


async def test_critical_while_speaking_cuts_now(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock, decision_running=True)
    try:
        await fake_clock.run_for(0.5)
        rig.brain.submit(mk(fake_clock, StimulusKind.VOICE, "หยุดเดี๋ยวนี้", priority=Priority.CRITICAL))
        await fake_clock.run_until_idle()
        assert [(m, r) for _, m, r, _ in stops(rig)] == [("now", "preempt_critical")]
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=5)
        assert rig.llm.closed_early == 1  # the streaming LLM was cancelled
        done = first_done(rig)
        assert done.cancelled
        assert rig.llm.times[1] - done.ts < 0.2  # decided at once
        await rig.idle_out()
    finally:
        await rig.stop()


# --- priorities while DECIDING with nothing audible (§4.3 a/b) ----------------------------------


async def deciding_rig(clock: FakeClock) -> BrainRig:
    llm = TimedLLM(
        [FakeReply("ช้ามาก", "ตอบช้าค่ะ", ttft_s=3.0), FakeReply("", "ได้เลยค่ะ", repeat=True)],
        clock,
        ttft_s=0.2,
    )
    rig = BrainRig(clock, llm=llm)
    await rig.start()
    rig.brain.submit(mk(clock, StimulusKind.MENTION, "ไพลินตอบช้ามากเลย", speaker="amy"))
    await clock.run_until(lambda: rig.brain.slot.busy and rig.llm.requests, within=10)
    return rig


async def test_low_and_medium_while_deciding_merge(fake_clock: FakeClock) -> None:
    rig = await deciding_rig(fake_clock)
    try:
        rig.brain.submit(mk(fake_clock, StimulusKind.MENTION, "ไพลินอยู่ไหม", speaker="bo"))
        rig.brain.submit(
            mk(fake_clock, StimulusKind.SUPPORT, "", speaker="rich", payload={"msg_kind": "sub"})
        )
        await fake_clock.run_for(0.5)
        assert rig.llm.closed_early == 0 and stops(rig) == []
        await fake_clock.run_until(lambda: len(rig.llm.requests) >= 2, within=30)
        assert rig.llm.closed_early == 0
    finally:
        await rig.stop()


async def test_better_rank_high_aborts_and_restarts_merged(fake_clock: FakeClock) -> None:
    rig = await deciding_rig(fake_clock)
    try:
        rig.say("ไพลิน ฟังทางนี้ก่อน")  # VOICE: HIGH and a strictly better rank than MENTION
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=5)
        assert rig.llm.closed_early == 1
        tail = rig.last_user(1)
        assert "ฟังทางนี้ก่อน" in tail and "ไพลินตอบช้ามากเลย" in tail  # restarted merged
        assert rig.speech.heard_texts() == [] or "ตอบช้าค่ะ" not in rig.speech.heard_texts()
        aborted = rig.bus.of_type(DecisionAborted)
        assert [a.reason for a in aborted] == ["restart_merged"]
        await rig.idle_out()
        assert "ตอบช้าค่ะ" not in "".join(rig.speech.heard_texts())
        # nothing of the aborted decision is in history; the restart is one turn
        assert roles(rig.history.turns()) == ["user", "assistant"]
    finally:
        await rig.stop()


async def test_same_priority_worse_rank_does_not_abort(fake_clock: FakeClock) -> None:
    rig = await deciding_rig(fake_clock)
    try:
        rig.brain.submit(
            mk(fake_clock, StimulusKind.GAME_CONTEXT, "บอสโผล่มา", priority=Priority.HIGH)
        )
        await fake_clock.run_for(0.5)
        assert rig.llm.closed_early == 0 and not rig.bus.of_type(DecisionAborted)
    finally:
        await rig.stop()


async def test_critical_while_deciding_aborts_and_restarts(fake_clock: FakeClock) -> None:
    rig = await deciding_rig(fake_clock)
    try:
        rig.brain.submit(mk(fake_clock, StimulusKind.VOICE, "ด่วนเลย", priority=Priority.CRITICAL))
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=5)
        assert [a.reason for a in rig.bus.of_type(DecisionAborted)] == ["restart_critical"]
        assert "ด่วนเลย" in rig.last_user(1) and "ไพลินตอบช้ามากเลย" in rig.last_user(1)
    finally:
        await rig.stop()


# --- FREEZE / SKIP / operator control (§2.11) ---------------------------------------------------


async def test_freeze_cancels_llm_and_stops_speech_within_150ms(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock, decision_running=True)
    try:
        t0 = fake_clock.now()
        result = await rig.brain.control(OpCommand(OpKind.FREEZE))
        assert result.ok and result.latency_ms <= 150.0
        assert fake_clock.now() - t0 <= 0.15
        assert ("stop", (None, "now", "operator_freeze", 30)) in rig.speech.calls
        await fake_clock.run_until_idle()
        assert rig.llm.closed_early == 1 and rig.llm.in_flight == 0  # the LLM was cancelled
        assert rig.brain.state == "paused" and rig.avatar.state == "paused"
        assert not rig.brain.intake.chat_intake
        # nothing starts until RESUME
        rig.say("ยังอยู่ไหม")
        rig.brain.on_chat(make_chat_message("ไพลิน!", user="x", clock=fake_clock))
        support = mk(
            fake_clock, StimulusKind.SUPPORT, "", speaker="rich", payload={"msg_kind": "sub"}
        )
        rig.brain.submit(support)
        await fake_clock.run_for(30)
        assert len(rig.llm.requests) == 1
        assert (await rig.brain.control(OpCommand(OpKind.SAY, {"text": "สวัสดี"}))).ok is False
        # RESUME clears the inbox except SUPPORT
        assert (await rig.brain.control(OpCommand(OpKind.RESUME))).ok
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=10)
        assert "rich" in rig.last_user(1) and "ยังอยู่ไหม" not in rig.last_user(1)
        assert rig.brain.intake.chat_intake
        await rig.idle_out()
        assert len(rig.llm.requests) == 2
    finally:
        await rig.stop()


async def test_skip_cuts_the_current_utterance(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        result = await rig.brain.control(OpCommand(OpKind.SKIP))
        assert result.ok
        await fake_clock.run_until(lambda: bool(rig.bus.of_type(UtteranceDone)), within=2)
        assert first_done(rig).cancelled
        assert (await rig.brain.control(OpCommand(OpKind.SKIP))).ok is False  # nothing plays
        await rig.idle_out()
        assert rig.brain.state == "idle" and len(rig.llm.requests) == 1  # the brain carries on
    finally:
        await rig.stop()


async def test_control_flags_and_unimplemented(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock)
    await rig.start()
    try:
        c = rig.brain.control
        assert (await c(OpCommand(OpKind.MUTE))).ok and rig.speech.muted and rig.brain.muted
        assert (await c(OpCommand(OpKind.UNMUTE))).ok and not rig.speech.muted
        assert (await c(OpCommand(OpKind.MIC_MODE, {"mode": "deafened"}))).ok
        assert rig.brain.intake.mic_mode == "deafened"
        assert not (await c(OpCommand(OpKind.MIC_MODE, {"mode": "loud"}))).ok
        assert (await c(OpCommand(OpKind.PTT, {"active": True}))).ok and rig.brain.intake.ptt_active
        assert (await c(OpCommand(OpKind.CHAT_INTAKE, {"on": False}))).ok
        assert not rig.brain.intake.chat_intake
        assert (await c(OpCommand(OpKind.STRICT, {"on": True}))).ok and rig.brain.strict()
        assert rig.gate.strict_mode("pailin")
        assert (await c(OpCommand(OpKind.STRICT, {"on": False}))).ok and not rig.brain.strict()
        assert (await c(OpCommand(OpKind.FILTER_RELOAD))).ok and rig.gate.reloads == 1
        assert (await c(OpCommand(OpKind.LLM_USE, {"name": "nope"}))).ok is False
        assert (await c(OpCommand(OpKind.LLM_ROLLBACK))).ok
        assert (await c(OpCommand(OpKind.MUTE_USER, {"user_id": "u-tom", "seconds": 60}))).ok
        res = await c(OpCommand(OpKind.TOOLS_MODE, {"mode": "off"}))
        assert not res.ok and "not implemented" in res.detail
    finally:
        await rig.stop()


# --- watchdog and failures (§2.8, §4.12) -------------------------------------------------------


class HangingRouter:
    """An LLMRouter whose stream never yields (a wedged provider)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.requests: list[ChatRequest] = []
        self.closed = 0

    async def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(req)
        try:
            await self.clock.sleep(1e9)
            yield TextDelta("never")
        finally:
            self.closed += 1

    async def prefill(self, req: ChatRequest) -> None:
        return None

    def promote(self, name: str) -> None: ...

    def rollback(self) -> None: ...

    def active(self) -> str:
        return "hang"

    def status(self) -> list[Any]:
        return []


async def test_watchdog_aborts_a_hung_decision_and_dumps(fake_clock: FakeClock) -> None:
    dumps: list[str] = []

    async def dump(why: str) -> None:
        dumps.append(why)

    router = HangingRouter(fake_clock)
    rig = BrainRig(
        fake_clock,
        router=router,
        reply_kwargs={"inter_token_timeout_s": 1000.0},
        dump_flight=dump,
    )
    await rig.start()
    try:
        rig.say("ตอบหน่อย")
        await fake_clock.run_until(lambda: router.requests != [], within=5)
        await fake_clock.run_for(29.0)
        assert rig.brain.slot.busy
        await fake_clock.run_for(1.5)
        assert not rig.brain.slot.busy
        assert [a.reason for a in rig.bus.of_type(DecisionAborted)] == ["watchdog"]
        assert dumps == ["watchdog"] and router.closed == 1
        assert any(a.level == "error" for a in rig.bus.of_type(Alert))
    finally:
        await rig.stop()


async def test_brain_freeze_line_once_per_outage(fake_clock: FakeClock) -> None:
    llm = FakeLLM([], fail="connect", clock=fake_clock)
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        rig.say("ฮัลโหล")
        await rig.idle_out()
        rig.say("ฮัลโหลอีกที")
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=10)
        await rig.idle_out()
        canned = [a for n, a in rig.speech.calls if n == "play_canned"]
        assert canned == [("brain_freeze", "pailin")]  # once per outage
        assert "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ" in rig.speech.heard_texts()
        assert [a.level for a in rig.bus.of_type(Alert)] == ["error"]
        assert rig.history.turns() == []  # nothing answered: no turn recorded
        # the LLM is back: the next outage plays the line again
        llm.fail = None
        llm.script.add(FakeReply("", "กลับมาแล้วค่ะ", repeat=True))
        rig.say("กลับมายัง")
        await rig.idle_out()
        llm.fail = "connect"
        rig.say("อีกรอบ")
        await rig.idle_out()
        canned = [a for n, a in rig.speech.calls if n == "play_canned"]
        assert len(canned) == 2
    finally:
        await rig.stop()


class DelayedDoneBus(FakeEventBus):
    """Delays ``UtteranceDone`` of cancelled utterances by ``delay`` clock seconds."""

    def __init__(self, clock: FakeClock, delay: float) -> None:
        super().__init__(clock)
        self.delay = delay
        self._later: set[asyncio.Task[None]] = set()

    def publish(self, event: Event) -> None:
        if isinstance(event, UtteranceDone) and event.cancelled and self.delay:
            task = asyncio.ensure_future(self._publish_later(event))
            self._later.add(task)
            task.add_done_callback(self._later.discard)
            return
        super().publish(event)

    async def _publish_later(self, event: Event) -> None:
        await self.clock.sleep(self.delay)
        super().publish(event)


@pytest.mark.parametrize(("delay", "exact"), [(0.1, True), (0.4, False)])
async def test_after_a_critical_cut_the_prompt_waits_for_utterance_done(
    fake_clock: FakeClock, delay: float, exact: bool
) -> None:
    bus = DelayedDoneBus(fake_clock, delay)
    llm = TimedLLM(
        [FakeReply("เล่ายาว", LONG), FakeReply("", "ค่ะ", repeat=True)], fake_clock, ttft_s=0.2
    )
    rig = BrainRig(fake_clock, llm=llm, bus=bus)
    await rig.start()
    try:
        rig.say("เล่ายาวๆ หน่อย")
        await fake_clock.run_until(
            lambda: rig.brain.state == "speaking" and not rig.brain.slot.busy, within=10
        )
        await fake_clock.run_for(1.0)
        bus.publish(BargeInConfirmed(text="เดี๋ยวก่อน", cut_local=True))
        t_cut = fake_clock.now()
        await fake_clock.run_for(0.05)
        rig.say("เดี๋ยวก่อน ขอถามหน่อย")
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=5)
        waited = llm.times[1] - t_cut
        assistant = [m for m in llm.requests[1].messages if m["role"] == "assistant"][-1]
        if exact:  # UtteranceDone arrived within 150 ms: exact heard text in the prompt
            done = first_done(rig)
            assert delay <= waited <= 0.05 + delay + 0.02
            assert assistant["content"].startswith(done.heard_text.strip())
        else:  # it did not: the prompt was built after at most 150 ms anyway
            assert rig.bus.of_type(UtteranceDone) == []
            assert waited <= 0.05 + 0.15 + 0.02
        assert assistant["content"].endswith("[ถูกขัดจังหวะ]")
        await rig.idle_out()
    finally:
        await rig.stop()


# --- prompt cache discipline (§4.8) -------------------------------------------------------------


async def test_prompt_prefix_is_byte_stable_across_decisions(fake_clock: FakeClock) -> None:
    llm = FakeLLM(
        [FakeReply("เรื่องที่ 3", LONG), FakeReply("", "ตอบสั้นๆ ค่ะ", repeat=True)],
        clock=fake_clock,
        ttft_s=0.2,
    )
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        for n in range(8):
            rig.say(f"เรื่องที่ {n} นะ")
            await fake_clock.run_until(lambda n=n: len(llm.requests) == n + 1, within=30)
            if n == 3:  # interrupt the long reply: HIGH while speaking
                await fake_clock.run_until(lambda: rig.brain.state == "speaking", within=10)
                continue
            await rig.idle_out()
        await rig.idle_out()
        dumps = [
            [json.dumps(m, ensure_ascii=False, sort_keys=True) for m in r.messages]
            for r in llm.requests
        ]
        for a, b in itertools.pairwise(dumps):
            prefix = a[:-1]  # everything but the volatile tail = epoch + history through N-1
            assert b[: len(prefix)] == prefix
            assert "".join(b).startswith("".join(prefix))
        assert all(r.slot == 0 for r in llm.requests)
        assert llm.requests[-1].messages[2]["content"] == "รับทราบค่ะ"
    finally:
        await rig.stop()


# --- the Filtered path (§4.10) -------------------------------------------------------------------


class DeltaRouter:
    """An LLMRouter replaying explicit ``(delta, sleep_after_s)`` scripts, one per request."""

    def __init__(self, clock: FakeClock, scripts: list[list[tuple[str, float]]]) -> None:
        self.clock = clock
        self.scripts = scripts
        self.requests: list[ChatRequest] = []
        self.closed_early = 0

    async def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(req)
        script = self.scripts.pop(0) if self.scripts else [("ค่ะ", 0.0)]
        finished = False
        try:
            await self.clock.sleep(0.2)
            for delta, pause in script:
                yield TextDelta(delta)
                if pause:
                    await self.clock.sleep(pause)
            text = "".join(d for d, _ in script)
            yield Done(
                "delta", "stop", 200.0, 10, 5, len(script), {"role": "assistant", "content": text}
            )
            finished = True
        finally:
            if not finished:
                self.closed_early += 1

    async def prefill(self, req: ChatRequest) -> None:
        return None

    def promote(self, name: str) -> None: ...

    def rollback(self) -> None: ...

    def active(self) -> str:
        return "delta"

    def status(self) -> list[Any]:
        return []


async def test_filtered_split_phrase(fake_clock: FakeClock) -> None:
    router = DeltaRouter(
        fake_clock,
        [
            [
                ("สวัสดีค่ะ ", 0.05),
                ("วันนี้จะพูดถึงคำต้อง ห้", 1.0),  # the LLM stalls: flushed at the last space
                ("ามนะ ซึ่งไม่ควรพูดออกอากาศเลยนะคะ ", 0.05),
                ("แล้วก็เรื่องลับอื่นๆ อีกมากมาย", 0.05),
                (" จบแล้วค่ะ", 0.0),
            ],
            [("เปลี่ยนเรื่องดีกว่าค่ะ", 0.0)],
        ],
    )
    rig = BrainRig(fake_clock, router=router, blocklist=["คำต้องห้าม"])
    await rig.start()
    try:
        rig.say("เล่าอะไรก็ได้")
        await fake_clock.run_until(lambda: rig.bus.of_type(Filtered), within=10)
        await rig.idle_out()
        # the split phrase was caught with the previous chunk's tail
        blocked = [o for o in rig.gate.outputs if o[0].startswith("ห้ามนะ")]
        assert blocked and blocked[0][2].endswith("คำต้อง ")
        assert router.closed_early == 1  # the LLM stream was closed
        names = rig.speech.call_names()
        stop_i = names.index("stop")
        assert rig.speech.calls[stop_i][1][1:3] == ("after_segment", "filtered")
        assert (
            names[stop_i + 1] == "play_canned" and rig.speech.calls[stop_i + 1][1][0] == "filtered"
        )
        assert "Filtered." in rig.speech.heard_texts()
        heard = "".join(rig.speech.heard_texts())
        assert "ห้ามนะ" not in heard and "เรื่องลับ" not in heard
        assert rig.avatar.emotions[-1] == "neutral"
        # history: heard text + [Filtered.]; the blocked text is never stored
        [assistant] = [t for t in rig.history.turns() if t.role == "assistant"]
        assert assistant.filtered
        stored = " ".join(f"{t.text} {t.heard_text or ''}" for _, t in rig.memory.state.turns)
        assert "ห้ามนะ" not in stored and "เรื่องลับ" not in stored
        # the next tail carries the note, and the history line ends with [Filtered.]
        rig.say("แล้วไงต่อ")
        await fake_clock.run_until(lambda: len(router.requests) == 2, within=10)
        tail = str(router.requests[1].messages[-1]["content"])
        assert "(ประโยคก่อนหน้าถูกกรอง)" in tail
        rendered = [m for m in router.requests[1].messages if m["role"] == "assistant"][-1]
        assert rendered["content"].endswith("[Filtered.]") and "ห้าม" not in rendered["content"]
        await rig.idle_out()
    finally:
        await rig.stop()


# --- tools (§4.9) -------------------------------------------------------------------------------


async def test_tool_only_reply_gets_one_follow_up(fake_clock: FakeClock) -> None:
    echo = FakeTool()
    call = tool_call("echo", {"text": "จำชื่อบอส"})
    llm = FakeLLM(
        [
            FakeReply("ช่วยจำ", "", tool_calls=[call]),
            FakeReply("ช่วยจำ", "จำไว้แล้วค่ะ"),
        ],
        clock=fake_clock,
        ttft_s=0.2,
    )
    rig = BrainRig(fake_clock, llm=llm, tools=[echo])
    await rig.start()
    try:
        rig.say("ช่วยจำชื่อบอสไว้หน่อย")
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=10)
        await rig.idle_out()
        assert len(llm.requests) == 2  # exactly one follow-up
        assert echo.calls == [{"text": "จำชื่อบอส"}]
        follow = llm.requests[1]
        assert [m["role"] for m in follow.messages[-3:]] == ["user", "assistant", "tool"]
        assert follow.messages[-2]["tool_calls"][0]["function"]["arguments"] == call.raw_arguments
        assert follow.messages[-1]["tool_call_id"] == call.id
        assert follow.tool_choice == "none"  # the last allowed round cannot call tools again
        assert follow.messages[:-2] == llm.requests[0].messages  # same prefix + tail
        assert "จำไว้แล้วค่ะ" in "".join(rig.speech.heard_texts())
        turns = rig.history.turns()
        assert roles(turns) == ["user", "assistant", "tool", "assistant"]
        assert turns[1].tool_calls and json.loads(turns[1].tool_calls)[0]["id"] == call.id
        # the next prompt renders the tool exchange in call order
        rig.say("ขอบคุณ")
        await fake_clock.run_until(lambda: len(llm.requests) == 3, within=10)
        roles3 = [m["role"] for m in llm.requests[2].messages]
        assert roles3[3:] == ["user", "assistant", "tool", "assistant", "user"]
    finally:
        await rig.stop()


async def test_voice_with_merged_chat_uses_a_chat_kind_tool_stimulus() -> None:
    from aivtube.brain.arbiter import MergedContext
    from aivtube.brain.loop import Brain
    from aivtube.contracts.chat import ChatSelection

    voice = stim(StimulusKind.VOICE, "จำไว้นะ")
    msg = make_chat_message("ผมชื่อต้น", user="ton")
    plain = Brain._tool_stimulus(MergedContext(voice))
    assert plain.kind is StimulusKind.VOICE
    merged = Brain._tool_stimulus(MergedContext(voice, chat=ChatSelection((), (msg,), ())))
    assert merged.kind is StimulusKind.CHAT and merged.payload["messages"] == (msg,)
    mention = stim(StimulusKind.MENTION, "ไพลิน", payload={"message": msg})
    via_mention = Brain._tool_stimulus(MergedContext(voice, merged=(mention,)))
    assert via_mention.kind is StimulusKind.CHAT


# --- idle (§4.13) -------------------------------------------------------------------------------


async def test_idle_turn_after_25s_with_recent_topics(fake_clock: FakeClock) -> None:
    llm = TimedLLM([FakeReply("", "ว่างจังเลย มาคุยเรื่องเกมกันไหม", repeat=True)], fake_clock)
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        t_live = fake_clock.now()
        await fake_clock.run_until(lambda: len(llm.requests) == 1, within=40)
        assert 20.0 <= llm.times[0] - t_live <= 30.0
        assert "ไม่มีใครคุยด้วย" in rig.last_user(0)
        await rig.idle_out()
        turns = rig.history.turns()
        assert roles(turns) == ["user", "assistant"] and turns[0].text == "[ไม่มีใครคุย]"
        # speech cancels the timer; the next idle turn lists the recent topics
        await fake_clock.run_for(10)
        rig.say("เงียบจัง")
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=10)
        await rig.idle_out()
        t_end = fake_clock.now()
        await fake_clock.run_until(lambda: len(llm.requests) == 3, within=40)
        assert 20.0 <= llm.times[2] - t_end <= 30.0
        assert "เรื่องที่เพิ่งคุยไป" in rig.last_user(2) and "ว่างจังเลย" in rig.last_user(2)
    finally:
        await rig.stop()


async def test_no_idle_while_the_streamer_speaks(fake_clock: FakeClock) -> None:
    rig = BrainRig(fake_clock, script=[FakeReply("", "ค่ะ", repeat=True)])
    await rig.start()
    try:
        rig.bus.publish(UserSpeechStarted(barge=False))
        await fake_clock.run_for(19.0)
        assert rig.llm.requests == [] and rig.brain.user_speaking()
        assert rig.avatar.state == "listening"
        rig.bus.publish(UserSpeechEnded(audio_s=19.0))
        await fake_clock.run_for(19.0)
        assert rig.llm.requests == []  # the idle timer restarted when the streamer stopped
    finally:
        await rig.stop()


# --- user_speaking (§4.3) -----------------------------------------------------------------------


async def test_no_decision_while_the_streamer_speaks_except_operator(fake_clock: FakeClock) -> None:
    llm = TimedLLM([FakeReply("", "ค่ะ", repeat=True)], fake_clock)
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        rig.bus.publish(UserSpeechStarted(barge=False))
        await fake_clock.run_until_idle()
        rig.brain.on_chat(make_chat_message("ไพลินเล่นเกมอะไรอยู่", user="amy", clock=fake_clock))
        rig.brain.on_chat(make_chat_message("สวัสดีครับ", user="bo", clock=fake_clock))
        await fake_clock.run_for(15.0)
        assert llm.requests == []  # chat waits while the streamer talks
        t_end = fake_clock.now()
        rig.bus.publish(UserSpeechEnded(audio_s=15.0))
        await fake_clock.run_until(lambda: len(llm.requests) == 1, within=20)
        assert "ไพลินเล่นเกมอะไรอยู่" in rig.last_user(0) and llm.times[0] >= t_end
        await rig.idle_out()
        # an operator instruction is decided even while the streamer talks
        rig.bus.publish(UserSpeechStarted(barge=False))
        await fake_clock.run_until_idle()
        await rig.brain.control(OpCommand(OpKind.DIRECT, {"text": "ทักทายคนดูใหม่"}))
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=2)
        assert "ทักทายคนดูใหม่" in rig.last_user(1) and rig.brain.user_speaking()
        assert "[คำสั่งผู้ควบคุม ห้ามอ่านออกเสียง]" in rig.last_user(1)
    finally:
        await rig.stop()


async def test_user_speaking_watchdog_clears_after_20s(fake_clock: FakeClock) -> None:
    llm = TimedLLM([FakeReply("", "ค่ะ", repeat=True)], fake_clock)
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        t0 = fake_clock.now()
        rig.bus.publish(UserSpeechStarted(barge=False))  # vad.end never arrives
        await fake_clock.run_until_idle()
        rig.brain.on_chat(make_chat_message("ไพลินอยู่ไหม", user="amy", clock=fake_clock))
        await fake_clock.run_until(lambda: len(llm.requests) == 1, within=30)
        assert 20.0 <= llm.times[0] - t0 <= 21.5
    finally:
        await rig.stop()


# --- read-aloud dedupe through the brain (§4.2) --------------------------------------------------


async def test_read_aloud_chat_is_answered_once(fake_clock: FakeClock) -> None:
    llm = TimedLLM([FakeReply("", "ได้เลยค่ะ", repeat=True)], fake_clock)
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        rig.bus.publish(UserSpeechStarted(barge=False))
        await fake_clock.run_until_idle()
        rig.brain.on_chat(make_chat_message("อยากฟังเรื่องผีครับ", user="tom", clock=fake_clock))
        await fake_clock.run_for(2.0)
        rig.bus.publish(UserSpeechEnded(audio_s=2.0))
        rig.say("ทอมบอกว่าอยากฟังเรื่องผีครับ")
        await fake_clock.run_until(lambda: len(llm.requests) == 1, within=5)
        assert "(สตรีมเมอร์อ่านแชทของ tom ให้ฟัง)" in rig.last_user(0)
        await rig.idle_out()
        await fake_clock.run_for(15.0)
        assert len(llm.requests) == 1  # no separate chat decision for the same message
    finally:
        await rig.stop()


# --- operator SAY (§4.2) and END_STREAM (§2.9) ---------------------------------------------------


async def test_say_bypasses_the_llm_and_cuts_current_speech(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        text = "ขออนุญาตพักเบรกห้านาทีนะคะ " * 3
        result = await rig.brain.control(OpCommand(OpKind.SAY, {"text": text}))
        assert result.ok
        await fake_clock.run_until_idle()
        assert [(m, r) for _, m, r, _ in stops(rig)] == [("now", "preempt_critical")]
        await rig.idle_out()
        assert len(rig.llm.requests) == 1  # no LLM for SAY
        segs = [a[0] for n, a in rig.speech.calls if n == "segment" and a[0].kind == "operator"]
        assert segs and segs[-1].last and all(len(s.text) <= 160 for s in segs)
        assert "".join(s.caption for s in segs).split() == text.split()
        turns = rig.history.turns()
        assert turns[-2].text.startswith("[ผู้ควบคุมให้พูด]")
        assert turns[-1].role == "assistant" and "พักเบรก" in (turns[-1].heard_text or "")
    finally:
        await rig.stop()


async def test_end_stream_queues_the_episode_and_goes_pre_show(fake_clock: FakeClock) -> None:
    llm = FakeLLM(
        [
            FakeReply("บทสนทนาในไลฟ์", '{"summary": "ไลฟ์นี้คุยเรื่องแมวกับเกม"}'),
            FakeReply("", "ค่ะ", repeat=True),
        ],
        clock=fake_clock,
        ttft_s=0.1,
    )
    rig = BrainRig(fake_clock, llm=llm)
    await rig.start()
    try:
        rig.say("วันนี้คุยเรื่องแมว")
        await rig.idle_out()
        result = await rig.brain.control(OpCommand(OpKind.END_STREAM))
        assert result.ok and not rig.brain.live and not rig.brain.intake.chat_intake
        await fake_clock.run_until(
            lambda: any(m.kind == "episode" for m in rig.memory.state.memories.values()),
            within=10,
        )
        assert rig.brain.state == "pre_show"
        await fake_clock.run_for(60)
        assert len(llm.requests) == 2  # no idle turns after the stream ended
    finally:
        await rig.stop()


# --- memory quarantine through a real decision (§6) ---------------------------------------------


async def test_remember_from_a_chat_merged_decision_is_quarantined(
    fake_clock: FakeClock, tmp_path: Any
) -> None:
    from aivtube.memory import SqliteMemory
    from aivtube.tools import RememberTool

    memory = SqliteMemory(tmp_path / "pailin.sqlite", "pailin", fake_clock, fts=False)
    remember = RememberTool(min_interval_s=0.0)
    llm = FakeLLM(
        [
            FakeReply(
                "ชอบแมว", "จำไว้แล้ว", tool_calls=[tool_call("remember", {"text": "สตรีมเมอร์ชอบแมว"})]
            ),
            FakeReply("ชื่อต้น", "โอเคต้น", tool_calls=[tool_call("remember", {"text": "มีคนดูชื่อต้น"})]),
            FakeReply("", "ค่ะ", repeat=True),
        ],
        clock=fake_clock,
        ttft_s=0.1,
    )
    rig = BrainRig(fake_clock, llm=llm, tools=[remember], memory=memory)
    await rig.start()
    try:
        rig.say("ไพลิน จำไว้นะว่าเราชอบแมว")  # pure voice: trusted
        await run_real(fake_clock, lambda: llm.requests and rig.brain.state == "idle")
        rig.bus.publish(UserSpeechStarted(barge=False))
        await fake_clock.run_until_idle()
        rig.brain.on_chat(make_chat_message("ไพลิน ผมชื่อต้นนะ", user="ton", clock=fake_clock))
        await run_real(fake_clock, lambda: rig.brain.state == "idle")
        rig.bus.publish(UserSpeechEnded(audio_s=1.0))
        rig.say("มีคนบอกชื่อต้นมา จำไว้หน่อย")  # voice, with the mention merged into it
        await run_real(fake_clock, lambda: len(llm.requests) == 2 and rig.brain.state == "idle")
        assert "ผมชื่อต้นนะ" in str(llm.requests[1].messages[-1]["content"])  # merged
        items = {m.text: m.status for m in await memory.list_memories(kind="core")}
        assert items == {"สตรีมเมอร์ชอบแมว": "active", "มีคนดูชื่อต้น": "quarantined"}
    finally:
        await rig.stop()
        await memory.aclose()


class HungBeginSpeech(FakeSpeechOutput):
    """``begin()`` never returns (the voice worker is wedged)."""

    async def begin(self, utt_id: str, character: str, **kw: Any) -> None:
        self.calls.append(("begin", (utt_id, character)))
        await self.clock.sleep(1e9)


async def test_a_hung_begin_fails_the_decision_and_stops_the_utterance(
    fake_clock: FakeClock,
) -> None:
    bus = FakeEventBus(fake_clock)
    speech = HungBeginSpeech(bus, fake_clock)
    rig = BrainRig(fake_clock, script=[FakeReply("", "ค่ะ", repeat=True)], bus=bus, speech=speech)
    await rig.start()
    try:
        rig.say("ได้ยินไหม")
        await fake_clock.run_until(lambda: bool(rig.bus.of_type(DecisionAborted)), within=5)
        assert rig.bus.of_type(DecisionAborted)[0].reason.startswith("error: DeadlineExceeded")
        await fake_clock.run_until_idle()
        [(utt, mode, reason, _)] = stops(rig)
        assert (mode, reason) == ("now", "begin_failed") and utt.startswith("u-")
        assert rig.brain.state == "idle" and not rig.brain.slot.busy
        assert rig.llm.requests == []  # nothing was generated for a turn that cannot speak
    finally:
        await rig.stop()


async def test_a_turn_whose_tools_ran_is_recorded_not_restarted(fake_clock: FakeClock) -> None:
    echo = FakeTool()
    call = tool_call("echo", {"text": "ส้ม"})
    llm = TimedLLM(
        [
            FakeReply("ช่วยจำ", "", tool_calls=[call]),
            FakeReply("ช่วยจำ", "จำแล้วค่ะ", ttft_s=3.0),
            FakeReply("", "ค่ะ", repeat=True),
        ],
        fake_clock,
        ttft_s=0.2,
    )
    rig = BrainRig(fake_clock, llm=llm, tools=[echo])
    await rig.start()
    try:
        rig.brain.submit(mk(fake_clock, StimulusKind.MENTION, "ไพลินช่วยจำคำว่าส้ม", speaker="amy"))
        await fake_clock.run_until(lambda: len(llm.requests) == 2, within=10)  # the /u2 round
        rig.say("ฟังทางนี้ก่อน")  # HIGH with a better rank, nothing audible yet
        await fake_clock.run_until(lambda: len(llm.requests) == 3, within=10)
        assert [a.reason for a in rig.bus.of_type(DecisionAborted)] == ["restart_merged"]
        assert echo.calls == [{"text": "ส้ม"}]  # the side effect ran once
        assert "ไพลินช่วยจำคำว่าส้ม" not in rig.last_user(2)  # recorded, not re-decided
        assert "ฟังทางนี้ก่อน" in rig.last_user(2)
        await rig.idle_out()
        assert roles(rig.history.turns())[:4] == ["user", "assistant", "tool", "assistant"]
        assert echo.calls == [{"text": "ส้ม"}]
    finally:
        await rig.stop()


async def test_voice_worker_crash_keeps_only_the_heard_text(fake_clock: FakeClock) -> None:
    rig = await speaking_rig(fake_clock)
    try:
        await fake_clock.run_until(
            lambda: any(u.heard for u in rig.brain._utts.values()), within=30
        )
        heard_before = "".join(next(iter(rig.brain._utts.values())).heard.values())
        rig.speech.set_ready(False)  # the link is down until the worker is back
        rig.bus.publish(HealthChanged(health=Health("voice", HealthState.DOWN, "exit 1")))
        await fake_clock.run_until_idle()
        assert rig.brain._utts == {} and rig.brain.state == "idle"
        assistant = next(t for t in rig.history.turns() if t.role == "assistant")
        assert assistant.interrupted and assistant.heard_text == heard_before
        # nothing is decided until the voice side is ready again
        rig.say("กลับมาแล้วเหรอ")
        await fake_clock.run_for(3.0)
        assert len(rig.llm.requests) == 1
        rig.speech.set_ready(True)
        await fake_clock.run_until(lambda: len(rig.llm.requests) == 2, within=5)
    finally:
        await rig.stop()
