"""The real voice worker in-process: microphone → VAD → STT → core events, and barge-in."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from worker_kit import perf, stack, wait_for

from aivtube.contracts import ipc
from aivtube.contracts.events import (
    BargeInCandidate,
    BargeInConfirmed,
    SegmentStarted,
    UserSpeechEnded,
    UserSpeechStarted,
    UserTranscript,
    UtteranceDone,
)
from aivtube.contracts.speech import VoicePolicy


async def test_a_scripted_mic_utterance_becomes_a_user_transcript(tmp_path: Path) -> None:
    async with stack(tmp_path, stt_script=["สวัสดีครับ ไทลิน วันนี้เล่นเกมอะไร"]) as s:
        t0 = perf()
        s.say_into_mic(1.5)
        await wait_for(lambda: bool(s.of(UserTranscript)), 10.0)
        start = s.of(UserSpeechStarted)[0]
        end = s.of(UserSpeechEnded)[0]
        tr = s.of(UserTranscript)[0]
        assert not start.barge
        assert t0 < start.ts < end.ts <= tr.ts
        assert 1.0 < end.audio_s < 1.5 + 0.3 + 0.2 + 0.1  # speech (+ pre-roll and tail pad)
        assert tr.text == "สวัสดีครับ ไพลิน วันนี้เล่นเกมอะไร"  # the alias map fixed the name
        assert tr.engine == "fake_stt" and tr.parts == 1 and tr.audio_s > 1.0
        assert len(s.of(UserTranscript)) == 1


async def test_deafened_suppresses_vad_and_stt_messages(tmp_path: Path) -> None:
    async with stack(tmp_path, stt_script=["ได้ยินไหม", "ได้ยินแล้ว"]) as s:
        await s.out.set_policy(VoicePolicy(mic_mode="deafened"))
        await wait_for(lambda: s.worker is not None and s.worker.policy.mic_mode == "deafened")
        s.say_into_mic(1.2)
        await wait_for(lambda: s.device.mic_pending_s == 0.0, 10.0)
        await asyncio.sleep(1.0)  # long enough for a VadEnd and a decode, had it listened
        assert s.sent(ipc.VAD_START) == [] and s.sent(ipc.STT_FINAL) == []
        assert s.of(UserSpeechStarted, UserTranscript) == []
        await s.out.set_policy(VoicePolicy(mic_mode="open"))
        await wait_for(lambda: s.worker is not None and s.worker.policy.mic_mode == "open")
        s.say_into_mic(1.2, seed=3)
        await wait_for(lambda: bool(s.of(UserTranscript)), 10.0)
        assert [t.text for t in s.of(UserTranscript)] == ["ได้ยินไหม"]


async def _monologue(tmp_path: Path, speech_s: float, vad: dict[str, float]) -> None:
    script = ["ส่วนแรกของเรื่องยาว", "ส่วนที่สองจบแล้ว"]
    async with stack(tmp_path, stt_script=script, vad=vad) as s:
        s.say_into_mic(speech_s, seed=7)
        await wait_for(lambda: bool(s.of(UserTranscript)), speech_s + 10.0)
        await asyncio.sleep(0.5)
        finals = s.of(UserTranscript)
        assert len(finals) == 1 and len(s.of(UserSpeechStarted)) == 1  # no early final
        assert finals[0].parts >= 2
        assert finals[0].text == "ส่วนแรกของเรื่องยาว ส่วนที่สองจบแล้ว"
        assert finals[0].audio_s > vad["max_segment_s"]
        assert len(s.sent(ipc.VAD_END)) == 1 and len(s.sent(ipc.STT_FINAL)) == 1


async def test_a_forced_split_monologue_gives_one_final_with_two_parts(tmp_path: Path) -> None:
    """The forced split (VadPartial) path with the split point scaled down to 2 s."""
    await _monologue(tmp_path, 3.6, {"max_segment_s": 2.0})


@pytest.mark.timing
async def test_a_15_s_monologue_gives_one_final_with_two_parts(tmp_path: Path) -> None:
    """Acceptance as written: 17 s of speech at the default 15 s split (real time)."""
    await _monologue(tmp_path, 17.0, {"max_segment_s": 15.0})


async def test_barge_in_is_confirmed_and_cut_locally(tmp_path: Path) -> None:
    async with stack(tmp_path, stt_script=["หยุดก่อนนะ ฟังหน่อย", "หยุดก่อนนะ ฟังหน่อย"]) as s:
        await s.out.begin("u1", "pailin")
        await s.out.segment(s.speak("u1", ["พูดยาว ๆ ไปเรื่อย ๆ " * 8])[0])
        await wait_for(lambda: bool(s.of(SegmentStarted)), 10.0)
        s.say_into_mic(1.5, seed=11)  # the streamer talks over her
        await wait_for(lambda: bool(s.of(BargeInConfirmed)), 10.0)
        assert s.of(BargeInCandidate)
        confirmed = s.of(BargeInConfirmed)[0]
        assert confirmed.cut_local and "หยุดก่อน" in confirmed.text
        await wait_for(lambda: bool(s.of(UtteranceDone)), 5.0)
        done = s.of(UtteranceDone)[0]
        assert done.cancelled and done.reason in ("barge_in", "cut")
        assert any(utt == "u1" for utt, _ in s.cuts)  # the mouth closed
        await wait_for(lambda: bool(s.of(UserTranscript)), 10.0)  # the full turn still counts
        assert s.of(UserSpeechStarted)[0].barge
