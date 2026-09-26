"""``BusSpeechOutput`` against a scripted worker (a bare ``IpcClient``): the message mapping."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from ipc_kit import TOKEN, running_client, running_server, wait_for

from aivtube.contracts import ipc
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.events import (
    BargeInCandidate,
    BargeInConfirmed,
    BargeInRejected,
    Event,
    HealthChanged,
    ProviderSwitched,
    SegmentDone,
    SegmentStarted,
    UserSpeechEnded,
    UserSpeechStarted,
    UserTranscript,
    UtteranceDone,
    UtteranceStarted,
)
from aivtube.contracts.speech import SpeechOutput, TTSConstraints, VoicePolicy
from aivtube.contracts.types import HealthState, Segment
from aivtube.infra import AsyncEventBus, SystemClock
from aivtube.ipc import IpcClient, IpcServer
from aivtube.speech import BusSpeechOutput

CONFIGURE: dict[str, Any] = {
    "audio": {},
    "vad": {},
    "barge_in": {},
    "stt_chain": [],
    "tts": {"identities": {}, "backends": {}, "chunk": {}},
    "characters": {"pailin": {"identity_chain": ["premwadee"], "cached_phrases": ["Filtered."]}},
}


class SkewedClock(SystemClock):
    __slots__ = ()

    def now(self) -> float:
        return super().now() + 0.050  # the worker's clock runs 50 ms ahead


class Worker:
    """A scripted voice worker: records core messages, answers segments per ``answer``."""

    def __init__(self, url: str, clock: Any) -> None:
        self.client = IpcClient(url, TOKEN, "voice", clock, backoff=(0.05, 0.2))
        self.got: list[tuple[str, dict[str, Any]]] = []
        self.answer = "ok"
        for mtype in ipc.CORE_TO_VOICE:
            self.client.on(mtype, self._record)

    def _record(self, env: ipc.Envelope) -> None:
        self.got.append((env.type, dict(env.data)))
        if env.type == ipc.SPEAK_SEGMENT and self.answer != "silent":
            self.client.reply(
                env, self.answer, **({"detail": "x"} if self.answer == "error" else {})
            )

    def types(self) -> list[str]:
        return [t for t, _ in self.got]

    def send(self, mtype: str, data: dict[str, Any]) -> None:
        assert self.client.post(mtype, data), mtype


class Rig:
    def __init__(self, server: IpcServer, bus: AsyncEventBus, clock: SystemClock) -> None:
        self.events: list[Event] = []
        self.lips: list[LipTrack] = []
        self.cuts: list[tuple[str, float]] = []
        self.sub = bus.subscribe(name="rig")
        self.checked: list[Segment] = []
        self.out = BusSpeechOutput(
            server,
            bus,
            clock,
            lip_sink=self.lips.append,
            on_cut=lambda utt, t: self.cuts.append((utt, t)),
            configure=CONFIGURE,
            policy=VoicePolicy(mic_mode="ptt"),
            defaults={"pailin": TTSConstraints(8, 60, 160, "azure", "premwadee")},
            request_timeout_s=0.3,
            i7_check=self.checked.append,
        )

    def pull(self) -> list[Event]:
        self.events.extend(self.sub.drain())  # type: ignore[attr-defined]
        return self.events

    def of(self, *types: type[Event]) -> list[Any]:
        return [e for e in self.pull() if isinstance(e, types)]


@pytest.fixture
def clock() -> SystemClock:
    return SystemClock()


async def test_methods_become_messages_and_state_is_resent_after_a_reconnect(
    clock: SystemClock,
) -> None:
    bus = AsyncEventBus(clock)
    async with running_server(clock, bus) as server:
        rig = Rig(server, bus, clock)
        out: SpeechOutput = rig.out
        assert not out.ready()  # no worker yet
        assert out.constraints("pailin").min_chars == 60  # config defaults until reported
        assert out.constraints("other").max_chars == 160
        w = Worker(server.url, clock)
        async with running_client(w.client):
            await wait_for(lambda: ipc.VOICE_POLICY in w.types())
            assert w.types()[:2] == [ipc.VOICE_CONFIGURE, ipc.VOICE_POLICY]
            assert w.got[1][1]["mic_mode"] == "ptt"
            assert not out.ready()  # waits for the worker's health
            w.send(ipc.HEALTH, {"state": "starting", "detail": "loading", "stats": {}})
            w.send(ipc.HEALTH, {"state": "ok", "detail": "", "stats": {"linked": True}})
            await wait_for(out.ready)
            assert rig.out.voice_stats == {"linked": True}

            await out.begin("u1", "pailin", filler_after_s=1.2, gate_open=False)
            assert await out.segment(Segment("u1", 0, "สวัสดี", "สวัสดี!", emotion="happy"))
            w.answer = "busy"
            assert await out.segment(Segment("u1", 1, "ค่ะ", "ค่ะ")) is False
            w.answer = "error"
            assert await out.segment(Segment("u1", 1, "ค่ะ", "ค่ะ")) is True  # dropped
            w.answer = "silent"
            assert await out.segment(Segment("u1", 2, "นะ", "นะ", last=True)) is False  # timeout
            await out.open_gate("u1")
            await out.duck(1.7, ramp_ms=-5)  # clamped into the schema
            await out.mute(True)
            await out.set_voice_rate("pailin", 15)
            await out.set_policy(VoicePolicy(mic_mode="deafened"))
            await out.play_canned("filtered", "pailin")
            await out.stop(None, "after_segment", "filtered", fade_ms=40)
            await wait_for(lambda: ipc.SPEAK_STOP in w.types())
            sent = dict(w.got)
            assert sent[ipc.SPEAK_BEGIN] == {
                "utt": "u1",
                "character": "pailin",
                "filler_after_s": 1.2,
                "gate_open": False,
            }
            assert sent[ipc.SPEAK_DUCK] == {"gain": 1.0, "ramp_ms": 0}
            assert sent[ipc.VOICE_MUTE] == {"on": True}
            assert sent[ipc.VOICE_RATE] == {"character": "pailin", "percent": 15}
            assert sent[ipc.VOICE_POLICY]["mic_mode"] == "deafened"
            assert sent[ipc.SPEAK_CANNED] == {"key": "filtered", "character": "pailin"}
            assert sent[ipc.SPEAK_STOP] == {
                "utt": None,
                "mode": "after_segment",
                "reason": "filtered",
                "fade_ms": 40,
            }
            assert sent[ipc.SPEAK_GATE] == {"utt": "u1"}
            assert [s.seq for s in rig.checked] == [0, 1, 1, 2]  # the I7 hook saw every one
            assert rig.cuts == []  # after_segment: the mouth keeps moving

            n = len(w.got)
            w.client.disconnect("worker restart")
            await wait_for(lambda: len([t for t in w.types()[n:] if t == ipc.VOICE_RATE]) == 1)
            resent = w.types()[n:]
            assert resent[:3] == [ipc.VOICE_CONFIGURE, ipc.VOICE_POLICY, ipc.VOICE_MUTE]
            assert ipc.SPEAK_DUCK not in resent  # gain 1.0 is the default
            assert w.got[n + 1][1]["mic_mode"] == "deafened"
            # the utterance that was open on the old connection ended as voice_restart
            done = rig.of(UtteranceDone)
            assert [(d.utt_id, d.reason, d.cancelled, d.filtered) for d in done] == [
                ("u1", "voice_restart", True, True)
            ]
            assert rig.cuts and rig.cuts[0][0] == "u1"
            down = [e.health for e in rig.of(HealthChanged) if e.health.component == "voice"]
            assert any(h.state is HealthState.DOWN and "link lost" in h.detail for h in down)
        # the worker is gone: a new utterance ends at once, never hangs the brain
        await wait_for(lambda: server.peer("voice") is None)
        assert not out.ready()
        await out.begin("u2", "pailin")
        assert await out.segment(Segment("u2", 0, "x", "x", last=True))
        assert rig.of(UtteranceDone)[-1].reason == "voice_down"
        with pytest.raises(TypeError, match="I7"):
            await out.segment("not a segment")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            await out.begin("u2", "pailin")
        await rig.out.aclose()


async def test_worker_messages_become_events_on_the_core_clock(clock: SystemClock) -> None:
    bus = AsyncEventBus(clock)
    async with running_server(clock, bus) as server:
        rig = Rig(server, bus, clock)
        out = rig.out
        wclock = SkewedClock()
        w = Worker(server.url, wclock)
        async with running_client(w.client):
            await wait_for(lambda: ipc.VOICE_POLICY in w.types())
            peer = server.peer("voice")
            assert peer is not None and peer.clock_offset_s == pytest.approx(0.05, abs=0.002)
            w.send(ipc.HEALTH, {"state": "degraded", "detail": "tts edge out", "stats": {}})
            await wait_for(out.ready)
            bus.publish(UtteranceStarted(utt_id="u1", stimulus_id="s1", turn_id="t9"))
            await out.begin("u1", "pailin")
            assert await out.segment(Segment("u1", 0, "สวัสดีค่ะ", "สวัสดีค่ะ ", emotion="happy"))
            t = wclock.now()
            w.send(ipc.VAD_START, {"t": t, "barge": True})
            w.send(ipc.BARGE_CANDIDATE, {"t": t})
            w.send(ipc.BARGE_REJECTED, {"t": t})
            w.send(
                ipc.SEG_STARTED,
                {
                    "utt": "u1",
                    "seq": 0,
                    "t_audible": t,
                    "duration_s": 0.7,
                    "backend": "edge",
                    "silent": False,
                },
            )
            w.send(
                ipc.LIP_TRACK,
                {
                    "utt": "u1",
                    "seq": 0,
                    "t0": t,
                    "fps": 60,
                    "mouth": [0.1, 0.8],
                    "form": [0.5, 0.6],
                    "final": True,
                },
            )
            w.send(ipc.SEG_DONE, {"utt": "u1", "seq": 0, "heard": True, "heard_text": "สวัสดีค่ะ "})
            w.send(
                ipc.UTT_DONE,
                {"utt": "u1", "heard_text": "สวัสดีค่ะ ", "cancelled": False, "reason": None},
            )
            w.send(ipc.VAD_END, {"t": t + 1.0, "audio_s": 1.2})
            w.send(
                ipc.STT_FINAL,
                {
                    "text": "ไพลิน",
                    "engine": "typhoon_rt",
                    "latency_ms": 42.0,
                    "t_end": t + 1.0,
                    "audio_s": 1.2,
                    "parts": 2,
                },
            )
            w.send(ipc.BARGE_CONFIRMED, {"t": t + 0.5, "text": "หยุด", "cut_local": True})
            w.send(ipc.TTS_FALLBACK, {"from": "edge", "to": None, "reason": "timeout"})
            w.send(
                ipc.TTS_CONSTRAINTS,
                {
                    "character": "pailin",
                    "first_min_chars": 8,
                    "min_chars": 40,
                    "max_chars": 150,
                    "backend": "edge",
                    "identity": "premwadee",
                },
            )
            w.send(ipc.STT_SPECULATIVE, {"text": "x", "t": t, "silence_ms": 300.0})
            await wait_for(
                lambda: (
                    bool(rig.of(ProviderSwitched)) and out.constraints("pailin").max_chars == 150
                )
            )
            local = t - 0.05  # the same instant on the core clock
            started = rig.of(SegmentStarted)[0]
            assert started.t_audible == pytest.approx(local, abs=0.003)
            assert (started.character, started.turn_id, started.caption, started.emotion) == (
                "pailin",
                "t9",
                "สวัสดีค่ะ ",
                "happy",
            )
            seg_done = rig.of(SegmentDone)[0]
            assert seg_done.heard and seg_done.turn_id == "t9"
            done = rig.of(UtteranceDone)[0]
            assert (done.heard_text, done.cancelled, done.reason, done.filtered) == (
                "สวัสดีค่ะ ",
                False,
                None,
                False,
            )
            assert rig.of(UserSpeechStarted)[0].barge
            assert rig.of(UserSpeechStarted)[0].ts == pytest.approx(local, abs=0.003)
            assert rig.of(UserSpeechEnded)[0].audio_s == 1.2
            tr = rig.of(UserTranscript)[0]
            assert (tr.text, tr.engine, tr.parts, tr.latency_ms) == ("ไพลิน", "typhoon_rt", 2, 42.0)
            assert rig.of(BargeInCandidate) and rig.of(BargeInRejected)
            assert rig.of(BargeInConfirmed)[0].cut_local
            switched = rig.of(ProviderSwitched)[0]
            assert (switched.kind, switched.old, switched.new) == ("tts", "edge", "captions")
            assert out.constraints("pailin") == TTSConstraints(8, 40, 150, "edge", "premwadee")
            health = [e.health for e in rig.of(HealthChanged) if e.health.component == "voice"]
            assert health[-1].state is HealthState.DEGRADED
            # lip tracks go to the avatar driver only, on the core clock
            (track,) = rig.lips
            assert track.t0 == pytest.approx(local, abs=0.003) and track.mouth == (0.1, 0.8)
            assert not any(isinstance(e, LipTrack) for e in rig.pull())
            await asyncio.sleep(0.05)
            assert len(rig.of(UtteranceDone)) == 1  # nothing duplicated
        await rig.out.aclose()
