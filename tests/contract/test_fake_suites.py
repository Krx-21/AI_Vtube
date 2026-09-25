"""Run every fake through its contract suite (aivtube.testing.contracts)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aivtube.contracts.types import MsgKind, Platform
from aivtube.testing import fakes as F
from aivtube.testing.contracts import (
    SpeechOutputHarness,
    audio_out_suite,
    avatar_sink_suite,
    case_id,
    channel_actions_suite,
    chat_source_suite,
    chat_window_suite,
    clock_suite,
    event_bus_suite,
    llm_provider_suite,
    llm_router_suite,
    local_server_manager_suite,
    memory_store_suite,
    phrase_cache_suite,
    safety_gate_suite,
    set_title_call,
    speech_output_suite,
    speech_recognizer_suite,
    task_supervisor_suite,
    text_filter_suite,
    tool_registry_suite,
    tts_backend_suite,
    vad_suite,
)


def cases(suite: list[Any]) -> Any:
    return pytest.mark.parametrize("case", suite, ids=case_id)


# --- infra -----------------------------------------------------------------------------------


async def _run_for(clock: Any, dt: float) -> None:
    await clock.run_for(dt)


@cases(clock_suite(F.FakeClock, advance=_run_for))
async def test_fake_clock(case: Callable[[], Any]) -> None:
    await case()


@cases(clock_suite(F.RealClock))
async def test_real_clock(case: Callable[[], Any]) -> None:
    await case()


@cases(event_bus_suite(F.FakeEventBus))
async def test_fake_event_bus(case: Callable[[], Any]) -> None:
    await case()


@cases(task_supervisor_suite(F.FakeTaskSupervisor))
async def test_fake_task_supervisor(case: Callable[[], Any]) -> None:
    await case()


# --- voice -----------------------------------------------------------------------------------


@cases(audio_out_suite(F.FakeAudioOut))
def test_fake_audio_out(case: Callable[[], None]) -> None:
    case()


@cases(vad_suite(F.FakeVAD, speech=F.marker_tone(440.0, 1.0)))
def test_fake_vad(case: Callable[[], None]) -> None:
    case()


@cases(
    speech_recognizer_suite(
        lambda: F.FakeRecognizer({"440": "สวัสดีค่ะ", "660hz": "ไพลิน"}),
        sample=F.marker_tone(660.0, 1.0),
        expected="ไพลิน",
    )
)
def test_fake_recognizer(case: Callable[[], None]) -> None:
    case()


@cases(
    tts_backend_suite(
        lambda: F.FakeTTS(ttfa_s=0.01),
        unavailable_factory=lambda: F.FakeTTS(ttfa_s=0.01, no_audio=True),
    )
)
async def test_fake_tts(case: Callable[[], Any]) -> None:
    await case()


@cases(phrase_cache_suite(F.FakePhraseCache))
def test_fake_phrase_cache(case: Callable[[], None]) -> None:
    case()


# --- speech output ---------------------------------------------------------------------------


def _speech_fake_time() -> SpeechOutputHarness:
    clock = F.FakeClock()
    bus = F.FakeEventBus(clock)
    out = F.FakeSpeechOutput(bus, clock)
    return SpeechOutputHarness(out, bus, advance=clock.run_for, aclose=out.aclose)


def _speech_real_time() -> SpeechOutputHarness:
    clock = F.RealClock()
    bus = F.FakeEventBus(clock)
    out = F.FakeSpeechOutput(bus, clock, chars_per_s=400.0)
    return SpeechOutputHarness(out, bus, aclose=out.aclose)


@cases(speech_output_suite(_speech_fake_time))
async def test_fake_speech_output_fake_time(case: Callable[[], Any]) -> None:
    await case()


@cases(speech_output_suite(_speech_real_time))
async def test_fake_speech_output_real_time(case: Callable[[], Any]) -> None:
    await case()


# --- LLM -------------------------------------------------------------------------------------


def _llm() -> F.FakeLLM:
    return F.FakeLLM(
        [
            F.FakeReply("สวัสดี", "สวัสดีค่ะ ไพลินเองน้า", repeat=True),
            F.FakeReply(
                "title",
                "ได้เลยค่ะ",
                tool_calls=[set_title_call("ไพลินเล่นเกม Minecraft")],
                repeat=True,
            ),
        ],
        ttft_s=0.0,
        tok_s=0.0,
    )


async def _closed_early(p: Any) -> bool:
    return bool(p.closed_early > 0)


@cases(
    llm_provider_suite(
        _llm,
        expect_text="สวัสดีค่ะ ไพลินเองน้า",
        tool_prompt="ช่วยเปลี่ยน title เป็นเล่นเกมหน่อย",
        closed_probe=_closed_early,
    )
)
async def test_fake_llm(case: Callable[[], Any]) -> None:
    await case()


@cases(llm_router_suite(lambda providers: F.FakeLLMRouter(providers)))
async def test_fake_llm_router(case: Callable[[], Any]) -> None:
    await case()


@cases(local_server_manager_suite(F.FakeLauncher))
async def test_fake_launcher(case: Callable[[], Any]) -> None:
    await case()


# --- avatar / chat ---------------------------------------------------------------------------


@cases(avatar_sink_suite(F.FakeAvatarSink))
async def test_fake_avatar_sink(case: Callable[[], Any]) -> None:
    await case()


def _chat_source() -> F.FakeChatSource:
    msgs = [
        F.make_chat_message("สวัสดีไพลิน", user="tom", id="a"),
        F.make_chat_message("สวัสดีไพลิน", user="tom", id="a"),  # duplicate id: must be deduped
        F.make_chat_message("เล่นเกมอะไร", user="mali", id="b"),
        F.make_chat_message("ขอบคุณค่ะ", user="santa", id="c", kind=MsgKind.DONATION, amount=100),
    ]
    return F.FakeChatSource(msgs)


@cases(chat_source_suite(_chat_source, expected_min=3))
async def test_fake_chat_source(case: Callable[[], Any]) -> None:
    await case()


@cases(channel_actions_suite(F.FakeChannelActions))
async def test_fake_channel_actions_full(case: Callable[[], Any]) -> None:
    await case()


@cases(channel_actions_suite(lambda: F.FakeChannelActions(Platform.YOUTUBE, capabilities={"send"})))
async def test_fake_channel_actions_limited(case: Callable[[], Any]) -> None:
    await case()


@cases(chat_window_suite(F.FakeChatWindow))
def test_fake_chat_window(case: Callable[[], None]) -> None:
    case()


# --- memory / safety / tools -----------------------------------------------------------------


@cases(memory_store_suite(F.FakeMemoryStore))
async def test_fake_memory_store(case: Callable[[], Any]) -> None:
    await case()


@cases(text_filter_suite(F.FakeTextFilter, blocked="คำต้องห้าม"))
def test_fake_text_filter(case: Callable[[], None]) -> None:
    case()


@cases(safety_gate_suite(lambda: F.FakeSafetyGate(["คำต้องห้าม"]), blocked="คำต้องห้าม"))
async def test_fake_safety_gate(case: Callable[[], Any]) -> None:
    await case()


@cases(tool_registry_suite(lambda tools, gate: F.FakeToolRegistry(tools, gate=gate)))
async def test_fake_tool_registry(case: Callable[[], Any]) -> None:
    await case()
