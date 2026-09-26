"""Invariant I1 and link liveness with the real worker over a real localhost websocket."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from worker_kit import TOKEN, stack, wait_for

from aivtube.contracts import ipc
from aivtube.contracts.events import (
    HealthChanged,
    SegmentStarted,
    UserTranscript,
    UtteranceDone,
)
from aivtube.contracts.speech import VoicePolicy
from aivtube.infra import AsyncEventBus
from aivtube.ipc import IpcServer
from aivtube.speech import BusSpeechOutput


async def test_link_drop_finishes_the_segment_clears_the_queue_and_stops_listening(
    tmp_path: Path,
) -> None:
    async with stack(tmp_path, tts_cps=12.5, stt_script=["เล่นเกมต่อกันเลยไหม"]) as s:
        worker = s.worker
        assert worker is not None
        first = "หนึ่ง สอง สาม สี่ ห้า หก "  # 1.9 s at 12.5 chars/s
        await s.out.begin("u1", "pailin")
        for seg in s.speak("u1", [first, "ประโยคนี้ต้องไม่ถูกพูด"]):
            assert await s.out.segment(seg)
        await wait_for(lambda: bool(s.of(SegmentStarted)), 10.0)
        await asyncio.sleep(0.3)

        await s.stop_server()  # the core dies mid-utterance
        await wait_for(lambda: not worker.linked, 5.0)
        # the core side ended the utterance as voice_restart (keeping what was heard)
        core_done = [e for e in s.of(UtteranceDone) if e.utt_id == "u1"]
        assert core_done and core_done[0].cancelled and core_done[0].reason == "voice_restart"
        # the worker finishes the (already filtered) segment, then drops the rest
        await wait_for(lambda: bool(s.sent(ipc.UTT_DONE)), 10.0)
        seg_done = s.sent(ipc.SEG_DONE)
        assert [(d["seq"], d["heard"]) for d in seg_done] == [(0, True)]
        assert seg_done[0]["heard_text"] == first
        utt_done = s.sent(ipc.UTT_DONE)[0]
        assert utt_done["cancelled"] and utt_done["reason"] == "link_lost"
        assert [d["seq"] for d in s.sent(ipc.SEG_STARTED)] == [0]
        pipe = worker.pipeline
        assert pipe is not None
        await wait_for(lambda: pipe.queue.idle, 5.0)
        # ... and it stops listening while the link is down
        n_vad = len(s.sent(ipc.VAD_START))
        s.say_into_mic(1.2)
        await wait_for(lambda: s.device.mic_pending_s == 0.0, 10.0)
        await asyncio.sleep(1.0)
        assert len(s.sent(ipc.VAD_START)) == n_vad and not pipe.frontend.listening

        # a new core comes up on the same port: the worker reconnects and resumes
        bus2 = AsyncEventBus(s.clock)
        events2 = bus2.subscribe(name="core2")
        server2 = IpcServer("127.0.0.1", s.server.port, TOKEN, s.clock, bus2)
        out2 = BusSpeechOutput(
            server2,
            bus2,
            s.clock,
            lip_sink=lambda track: None,
            on_cut=lambda utt, t: None,
            configure=s.payload,
            policy=s.policy,
        )
        serving = asyncio.create_task(server2.serve())
        try:
            await wait_for(out2.ready, 15.0)
            assert worker.linked and pipe is worker.pipeline  # same payload: kept, not rebuilt
            await out2.begin("u2", "pailin")
            assert await out2.segment(s.speak("u2", ["กลับมาแล้วค่ะ"])[0])
            got: list[object] = []

            def seen(kind: type) -> bool:
                while (ev := events2.get_nowait()) is not None:  # type: ignore[attr-defined]
                    got.append(ev)
                return any(isinstance(e, kind) for e in got)

            await wait_for(lambda: seen(UtteranceDone), 15.0)
            done = next(e for e in got if isinstance(e, UtteranceDone))
            assert done.utt_id == "u2" and not done.cancelled
            s.say_into_mic(1.2, seed=5)  # listening again
            await wait_for(lambda: seen(UserTranscript), 15.0)
            tr = next(e for e in got if isinstance(e, UserTranscript))
            assert tr.text == "เล่นเกมต่อกันเลยไหม"
        finally:
            await out2.aclose()
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            events2.close()


async def test_an_idle_link_stays_up_on_heartbeats(tmp_path: Path) -> None:
    async with stack(tmp_path, heartbeat_s=0.2) as s:
        peer = s.server.peer("voice")
        assert peer is not None
        await asyncio.sleep(1.5)  # 7 heartbeat intervals without speech traffic
        assert s.server.peer("voice") is peer and peer.connected
        assert s.worker is not None and s.worker.linked
        assert peer.link.stats["rx"] >= 5 and s.out.ready()


async def test_worker_health_reports_starting_then_ok_with_stats(tmp_path: Path) -> None:
    async with stack(tmp_path) as s:
        states = [
            e.health.state.value for e in s.of(HealthChanged) if e.health.component == "voice"
        ]
        assert states[0] == "starting" and states[-1] == "ok"
        assert s.out.voice_stats.get("linked") is True
        assert "player" in s.out.voice_stats and "stt" in s.out.voice_stats


async def test_policy_echo_mode_switches_the_front_end_and_the_barge_threshold(
    tmp_path: Path,
) -> None:
    async with stack(tmp_path) as s:
        worker = s.worker
        assert worker is not None and worker.pipeline is not None
        pipe = worker.pipeline
        assert pipe.echo_mode == "none"  # headphones
        await s.out.set_policy(VoicePolicy(echo_mode="energy_dtd", barge_in="duck_only"))
        await wait_for(lambda: pipe.echo_mode == "energy_dtd", 10.0)
        assert pipe.endpointer.config.barge_threshold == 0.7  # §4.7: speakers need 0.65-0.7
        assert pipe.barge.policy == "duck_only"
        await s.out.set_policy(VoicePolicy(echo_mode="half_duplex"))
        await wait_for(lambda: pipe.echo_mode == "half_duplex", 10.0)
        assert pipe.endpointer.config.barge_threshold == 0.6
        await s.out.set_policy(VoicePolicy())  # auto: back to the config's choice
        await wait_for(lambda: pipe.echo_mode == "none", 10.0)
        assert worker.pipeline is pipe  # never rebuilt for a policy change


async def test_a_device_lost_for_long_reinitialises_portaudio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aivtube.voice import audio_io

    calls: list[object] = []
    monkeypatch.setattr(audio_io, "reinit_portaudio", lambda backend=None: calls.append(backend))
    async with stack(tmp_path) as s:
        worker = s.worker
        assert worker is not None and worker.pipeline is not None
        pipe = worker.pipeline
        pipe.player._device_ok = False  # the output device vanished and stays away
        t = s.clock.now()  # the worker's health loop checks on the same (real) timeline
        await worker._check_devices(t)
        await worker._check_devices(t + 5.0)
        assert calls == []  # the player retries by name every 2 s first
        await worker._check_devices(t + 6.5)
        assert calls == [worker._backend] and worker.stats["portaudio_reinit"] == 1
        await worker._check_devices(t + 8.0)
        assert len(calls) == 1  # at most once per 10 s
        state, detail = worker._compute_health()
        assert state == "degraded" and "output device lost" in detail
        pipe.player._device_ok = True
        await worker._check_devices(t + 9.0)
        assert worker._compute_health()[0] == "ok"
