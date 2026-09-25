"""Behaviour of the smaller fakes beyond their contract suites."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from aivtube.contracts.events import Alert, Filtered, ToolRejected
from aivtube.contracts.llm import ToolCall
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.safety import FilterContext, Verdict
from aivtube.contracts.tools import RateLimit, ToolPolicy
from aivtube.contracts.types import HealthState, MsgKind, Transcript, VoiceSpec
from aivtube.contracts.voice import AudioChunk, TTSUnavailable, VadStart, WordMark
from aivtube.testing.contracts import make_tool_context
from aivtube.testing.fakes import (
    ANON_NAME,
    FakeAudioIn,
    FakeAudioOut,
    FakeChatSource,
    FakeChatWindow,
    FakeClassifier,
    FakeClock,
    FakeEndpointer,
    FakeEventBus,
    FakeMemoryStore,
    FakePostProcessor,
    FakeRecognizer,
    FakeSafetyGate,
    FakeTaskSupervisor,
    FakeTextFilter,
    FakeTool,
    FakeToolRegistry,
    FakeTTS,
    FakeVAD,
    dominant_frequency,
    make_chat_message,
    marker_tone,
    resample_linear,
)

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural")


async def _synth(tts: FakeTTS, text: str, timeout: float = 2.0) -> list[AudioChunk | WordMark]:
    return [x async for x in tts.synth(text, VOICE, first_audio_timeout=timeout, idle_timeout=5.0)]


async def test_tts_duration_marks_and_tone() -> None:
    tts = FakeTTS(ttfa_s=0.0, chars_per_s=10.0, tone_hz=200.0, sample_rate=16000)
    items = await _synth(tts, "สวัสดี ค่ะ")  # 10 chars -> 1.0 s
    audio = np.concatenate([i.pcm for i in items if isinstance(i, AudioChunk)])
    assert audio.size == 16000 and audio.dtype == np.int16 and np.abs(audio).max() > 1000
    marks = [i for i in items if isinstance(i, WordMark)]
    assert [(m.text, m.offset_s) for m in marks] == [("สวัสดี", 0.0), ("ค่ะ", 0.7)]
    assert tts.requests == [("สวัสดี ค่ะ", VOICE)] and tts.completed == 1


async def test_tts_timeouts_failures_and_fake_clock() -> None:
    with pytest.raises(TTSUnavailable, match="first audio"):
        await _synth(FakeTTS(ttfa_s=3.0), "x", timeout=0.01)
    with pytest.raises(TTSUnavailable, match="no audio"):
        await _synth(FakeTTS(ttfa_s=0.0, no_audio=True), "x")
    flaky = FakeTTS(ttfa_s=0.0, fail_rate=0.5, seed=1)
    outcomes = []
    for _ in range(20):
        try:
            await _synth(flaky, "ข้อความ")
            outcomes.append(True)
        except TTSUnavailable:
            outcomes.append(False)
    assert 3 < outcomes.count(False) < 17 and flaky.failures == outcomes.count(False)
    clock = FakeClock(start=0.0)
    paced = FakeTTS(ttfa_s=0.4, chunk_s=0.1, pace=True, clock=clock)
    task = asyncio.ensure_future(_synth(paced, "x" * 25))  # 2.0 s of audio
    await clock.run_for(1.0)
    assert not task.done()
    await clock.run_for(2.0)
    await task
    assert clock.now() == 3.0 and clock.sleeps[0] == 0.4


def test_recognizer_tones_sequence_and_failures() -> None:
    rec = FakeRecognizer({"440": "สวัสดีค่ะ", "880hz": "ไพลิน"}, tolerance_hz=10.0)
    assert rec.transcribe(marker_tone(880.0, 0.5)).text == "ไพลิน"
    assert rec.transcribe(marker_tone(440.0, 0.5)).text == "สวัสดีค่ะ"
    assert rec.transcribe(marker_tone(600.0, 0.5)).text == ""
    assert rec.transcribe(np.zeros(8000, np.float32)).text == ""
    seq = FakeRecognizer(["หนึ่ง", "สอง"], delay_s=0.05, fail_times=1)
    with pytest.raises(RuntimeError):
        seq.transcribe(np.zeros(1600, np.float32))
    t0 = time.perf_counter()
    t = seq.transcribe(np.zeros(1600, np.float32), quick=True)
    assert t.text == "หนึ่ง" and time.perf_counter() - t0 >= 0.04 and t.latency_ms >= 40
    assert seq.calls == [(1600, False), (1600, True)]
    seq.close()
    with pytest.raises(RuntimeError):
        seq.transcribe(np.zeros(1600, np.float32))


def test_dominant_frequency_and_resample() -> None:
    assert dominant_frequency(marker_tone(523.0, 0.25)) == pytest.approx(523.0, abs=2.0)
    assert dominant_frequency(np.zeros(4000, np.float32)) is None
    y = resample_linear(marker_tone(100.0, 1.0, sample_rate=24000), 24000, 48000)
    assert y.size == 48000


def test_post_processor_aliases_and_echo() -> None:
    post = FakePostProcessor({"ไพลิน": ["ไทลิน", "ไทยลิน"]}, drop_if_in_tts=True, min_audio_s=0.3)
    t = Transcript("สวัสดีไทลิน", True, 1.0, 50.0, "fake")
    assert post(t, "").text == "สวัสดีไพลิน"  # type: ignore[union-attr]
    assert post(Transcript("ค่ะ", True, 0.1, 1.0, "fake"), "") is None
    assert post(Transcript("ยินดีต้อนรับ", True, 1.0, 1.0, "fake"), "ยินดีต้อนรับทุกคน") is None


def test_vad_script_then_energy_and_endpointer_script() -> None:
    vad = FakeVAD([0.9, 0.1])
    silence = np.zeros(512, np.float32)
    assert [vad.prob(silence), vad.prob(silence), vad.prob(silence)] == [0.9, 0.1, 0.0]
    assert vad.prob(marker_tone(440.0, 512 / 16000)) == 1.0
    with pytest.raises(ValueError):
        vad.prob(np.zeros(480, np.float32))
    ep = FakeEndpointer([[VadStart(1.0, False)], []])
    assert ep.push(silence, 1.0, False) == [VadStart(1.0, False)]
    assert ep.push(silence, 1.032, True) == [] and ep.push(silence, 1.064, False) == []
    assert ep.pushes[1] == (512, 1.032, True)


def test_audio_out_marks_gain_ramp_and_close() -> None:
    now = [10.0]
    out = FakeAudioOut(clock=lambda: now[0], output_latency_s=0.05)
    marks: list[tuple[bool, float]] = []
    out.play(np.full(960, 0.5, np.float32), 48000)
    out.mark(lambda h, t: marks.append((h, t)))
    out.play(np.full(480, 0.5, np.float32), 48000)
    out.mark(lambda h, t: marks.append((h, t)))
    y = out.pump(2)
    assert marks == [(True, pytest.approx(10.07))] and np.all(y == 0.5)
    out.set_gain(0.0, ramp_ms=10.0)
    y = out.pump(1)
    assert y[0] < 0.5 and y[-1] == pytest.approx(0.0, abs=1e-6)
    assert marks[-1] == (True, pytest.approx(10.06))
    out.play(np.ones(4800, np.float32), 48000)
    out.mark(lambda h, t: marks.append((h, t)))
    out.close()
    assert marks[-1][0] is False
    out.close()


def test_audio_in_blocks_and_capture_times() -> None:
    mic = FakeAudioIn(clock=lambda: 50.0)
    frames: list[tuple[int, float]] = []
    mic.start(lambda x, t: frames.append((x.size, t)))
    assert mic.push(np.zeros(1000, np.float32), t0=2.0) == 2
    assert mic.push(np.zeros(440, np.float32), t0=2.0 + 1000 / 48000) == 1
    assert frames == [(480, 2.0), (480, 2.01), (480, pytest.approx(2.02))]


async def test_bus_threadsafe_history_and_waiting() -> None:
    clock = FakeClock(start=7.0)
    bus = FakeEventBus(clock)
    sub = bus.subscribe(Alert, name="t", maxsize=10)
    threading.Thread(
        target=bus.publish_threadsafe, args=(Alert(level="info", message="x"),)
    ).start()
    ev = await sub.next(2.0)
    assert ev.ts == 7.0 and bus.names() == ["Alert"]
    waiter = asyncio.ensure_future(bus.wait_for(Alert, lambda a: a.message == "later"))
    await asyncio.sleep(0)
    bus.publish(Alert(level="warn", message="later"))
    assert (await waiter).message == "later"
    sub.close()
    assert sub not in bus.subscriptions


async def test_task_supervisor_crash_loop_breaker() -> None:
    sup = FakeTaskSupervisor()

    async def always_fails() -> None:
        raise RuntimeError("boom")

    sup.spawn("loop", always_fails, restart="on_error", backoff=(0.001, 0.002), breaker=(3, 60.0))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if sup.status()[0].state is HealthState.FAILED:
            break
    assert sup.status()[0].state is HealthState.FAILED and len(sup.errors) == 4

    async def fatal() -> None:
        raise ValueError("critical")

    sup.spawn("brain", fatal, critical=True)
    await asyncio.sleep(0.05)
    assert sup.critical_failures and sup.critical_failures[0][0] == "brain"
    await sup.aclose(1.0)


async def test_memory_store_specifics() -> None:
    store = FakeMemoryStore(core_slots=2, slot_max_chars=10)
    with pytest.raises(RuntimeError):
        await store.append_turn(object())  # type: ignore[arg-type]
    await store.start_session()
    with pytest.raises(ValueError):
        await store.remember(MemoryItem(None, "core", "ยาวเกินสิบตัวอักษรแน่นอน"))
    locked = await store.remember(MemoryItem(None, "core", "ล็อก", locked=True))
    with pytest.raises(PermissionError):
        await store.remember(MemoryItem(None, "core", "แทน"), replace_slot=locked.slot)
    with pytest.raises(PermissionError):
        await store.forget(locked.id or 0, by="t", reason="t")
    assert await store.search("ล็อ") and await store.search("") == []


def test_safety_filter_normalisation_and_classifier() -> None:
    f = FakeTextFilter(["badword"], replace={"สัส": "***"})
    ctx = FilterContext("out", "pailin")
    assert f.check("b a d w o r d", ctx).verdict is Verdict.BLOCK  # despaced
    assert f.check("bad​word", ctx).verdict is Verdict.BLOCK  # zero-width
    assert f.check("BADWORD", FilterContext("in", "pailin")).verdict is Verdict.DROP
    r = f.check("โอ้ย สัส", ctx)
    assert r.verdict is Verdict.REPLACE and r.text == "โอ้ย ***"

    async def score() -> list[float]:
        return await FakeClassifier({"พนัน": 0.95}).score(["เว็บพนัน", "สวัสดี"], "in")

    assert asyncio.run(score()) == [0.95, 0.0]


async def test_safety_gate_names_events_and_strict() -> None:
    bus = FakeEventBus()
    gate = FakeSafetyGate(["คำหยาบ"], bus=bus, strict={"twin"})
    msg = make_chat_message("สวัสดี", user="คำหยาบ123")
    res, name = gate.check_input(msg, character="pailin")
    assert res.verdict is Verdict.PASS and name == ANON_NAME
    out = await gate.check_output("บ", character="pailin", prev_tail="พูดคำหยา")
    assert out.verdict is Verdict.BLOCK and bus.of_type(Filtered)[0].direction == "out"
    assert gate.strict_mode("twin") and not gate.strict_mode("pailin")
    gate.set_strict("pailin")
    assert gate.strict_mode("pailin")


async def test_chat_source_failure_window_mentions() -> None:
    src = FakeChatSource([make_chat_message("a", id="1")])
    src.fail_with(ConnectionError("irc down"))
    it = src.messages()
    assert (await it.__anext__()).id == "1"
    with pytest.raises(ConnectionError):
        await it.__anext__()
    await src.aclose()
    assert src.health().state is HealthState.DOWN
    w = FakeChatWindow()
    w.add(make_chat_message("ไพลินจ๋า", id="m"))
    w.add(make_chat_message("ปกติ", id="n"))
    assert w.add(make_chat_message("บริจาค", id="d", kind=MsgKind.DONATION)) == "priority"
    assert w.pending() == (3, True)
    sel = w.select(time.perf_counter(), k=1)
    assert [m.id for m in sel.candidates] == ["m"] and [m.id for m in sel.must_ack] == ["d"]


async def test_tool_registry_policy_rate_limit_and_events() -> None:
    tool = FakeTool(policy=ToolPolicy(rate_limit=RateLimit(1, 60.0)))
    needs = FakeTool(
        spec=tool.spec.__class__("needs_timeout", "x", {"type": "object"}),
        policy=ToolPolicy(requires=frozenset({"timeout"})),
    )
    reg = FakeToolRegistry([tool, needs])
    ctx = make_tool_context()
    call = ToolCall("c1", "echo", {"text": "a"}, '{"text":"a"}', None)
    assert (await reg.execute(call, ctx)).ok
    limited = await reg.execute(call, ctx)
    assert not limited.ok and "rate limited" in limited.content
    missing = await reg.execute(ToolCall("c2", "needs_timeout", {}, "{}", None), ctx)
    assert not missing.ok and "timeout" in missing.content
    assert [e.reason for e in ctx.bus.of_type(ToolRejected)] == ["rate limited", "needs timeout"]  # type: ignore[attr-defined]
