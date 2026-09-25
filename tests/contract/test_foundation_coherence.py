"""Cross-module coherence of the M0 foundation: contracts, IPC, config, infra and fakes.

These tests pin the seams that no single package owns: IPC messages map 1:1 onto the events
and ``SpeechOutput`` calls they stand for (so WP4 can translate mechanically), the fakes run on
the real infra, and the config defaults agree with what the fakes assume.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from aivtube.config import load_config
from aivtube.contracts import events as ev
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.ipc import MESSAGE_SCHEMAS
from aivtube.contracts.speech import SpeechOutput, TTSConstraints, VoicePolicy
from aivtube.contracts.types import Health, HealthState, Segment
from aivtube.infra import AsyncEventBus, FlightRecorder, SupervisedTasks, load_dump
from aivtube.testing.fakes import FakeClock, FakeSpeechOutput

ROOT = Path(__file__).resolve().parents[2]
_EVENT_BASE = {f.name for f in dataclasses.fields(ev.Event)}  # ts, character, turn_id
_RENAME = {"utt_id": "utt"}  # IPC uses the short name


def _props(mtype: str) -> set[str]:
    return set(MESSAGE_SCHEMAS[mtype]["properties"])


def _fields(cls: type, *, drop: set[str] = frozenset()) -> set[str]:  # type: ignore[assignment]
    names = {f.name for f in dataclasses.fields(cls)} - _EVENT_BASE - drop
    return {_RENAME.get(n, n) for n in names}


def _params(method: Any) -> set[str]:
    names = [p for p in inspect.signature(method).parameters if p != "self"]
    return {_RENAME.get(n, n) for n in names}


# voice -> core: message -> (target class, fields the core fills in itself, extra IPC fields)
VOICE_TO_CORE: dict[str, tuple[type, set[str], set[str]]] = {
    "vad.start": (ev.UserSpeechStarted, set(), {"t"}),
    "vad.end": (ev.UserSpeechEnded, set(), {"t"}),
    "stt.final": (ev.UserTranscript, set(), {"t_end"}),
    "barge.candidate": (ev.BargeInCandidate, set(), {"t"}),
    "barge.confirmed": (ev.BargeInConfirmed, set(), {"t"}),
    "barge.rejected": (ev.BargeInRejected, set(), {"t"}),
    # caption/emotion come from the core's own record of the queued segment
    "speech.segment_started": (ev.SegmentStarted, {"caption", "emotion"}, set()),
    "speech.segment_done": (ev.SegmentDone, set(), set()),
    # filtered: the core knows whether it sent a "filtered" segment
    "speech.utterance_done": (ev.UtteranceDone, {"filtered"}, set()),
    "lip.track": (LipTrack, set(), set()),
    "tts.constraints": (TTSConstraints, set(), {"character"}),
    "health": (Health, {"component", "since"}, {"stats"}),
}


@pytest.mark.parametrize("mtype", sorted(VOICE_TO_CORE))
def test_voice_messages_carry_every_field_of_their_event(mtype: str) -> None:
    cls, core_side, ipc_only = VOICE_TO_CORE[mtype]
    assert _props(mtype) == _fields(cls, drop=core_side) | ipc_only


# core -> voice: message -> SpeechOutput method it implements (BusSpeechOutput, WP4)
CORE_TO_VOICE = {
    "speak.begin": "begin",
    "speak.gate": "open_gate",
    "speak.stop": "stop",
    "speak.duck": "duck",
    "speak.canned": "play_canned",
    "voice.mute": "mute",
    "voice.rate": "set_voice_rate",
}


@pytest.mark.parametrize("mtype", sorted(CORE_TO_VOICE))
def test_core_messages_mirror_speech_output_methods(mtype: str) -> None:
    method = getattr(SpeechOutput, CORE_TO_VOICE[mtype])
    assert _props(mtype) == _params(method)


def test_dataclass_messages_mirror_their_dataclasses() -> None:
    assert _props("speak.segment") == _fields(Segment)
    assert _props("voice.policy") == _fields(VoicePolicy)


def test_health_states_and_config_chunker_agree_with_fakes() -> None:
    cfg = load_config(ROOT, profile="ci")
    chunk = cfg.tts.chunker
    clock = FakeClock()
    out = FakeSpeechOutput(AsyncEventBus(clock), clock)
    c = out.constraints("pailin")
    assert (c.first_min_chars, c.min_chars, c.max_chars) == (
        chunk.first_min_chars,
        chunk.min_chars,
        chunk.max_chars,
    )
    assert out.max_queued == cfg.tts.max_queued_segments == 8  # Appendix A backpressure
    assert {s.value for s in HealthState} == set(
        MESSAGE_SCHEMAS["health"]["properties"]["state"]["enum"]
    )


async def test_fake_speech_output_on_real_bus_and_flight_recorder(tmp_path: Path) -> None:
    clock = FakeClock()
    flight = FlightRecorder()
    bus: EventBus = AsyncEventBus(clock, flight=flight)
    sub = bus.subscribe(ev.SegmentStarted, ev.SegmentDone, ev.UtteranceDone, name="test")
    out = FakeSpeechOutput(bus, clock, chars_per_s=10.0)

    await out.begin("u1", "pailin")
    assert await out.segment(Segment("u1", 0, "สวัสดีค่ะ ", "สวัสดีค่ะ", emotion="happy"))
    assert await out.segment(Segment("u1", 1, "ไพลินเอง", "ไพลินเอง", last=True))
    await clock.run_until(lambda: out.idle, within=10.0)

    got = []
    while (item := sub.get_nowait()) is not None:  # type: ignore[attr-defined]
        got.append(item)
    kinds = [type(e).__name__ for e in got]
    assert kinds == [
        "SegmentStarted",
        "SegmentDone",
        "SegmentStarted",
        "SegmentDone",
        "UtteranceDone",
    ]
    assert all(e.character == "pailin" and e.ts > 0 for e in got)
    done = got[-1]
    assert isinstance(done, ev.UtteranceDone)
    assert done.heard_text == "สวัสดีค่ะ ไพลินเอง" and not done.cancelled  # lossless concat
    assert out.heard_texts() == ["สวัสดีค่ะ ", "ไพลินเอง"]

    events, _ = load_dump(flight.dump(tmp_path / "flight.json"))
    assert events == flight.events()
    await out.aclose()
    sub.close()


async def test_fake_clock_drives_the_real_supervisor_backoff() -> None:
    clock: Clock = FakeClock()
    assert isinstance(clock, FakeClock)
    bus = AsyncEventBus(clock)
    restarts = bus.subscribe(ev.ComponentRestarted, name="restarts")
    critical: list[str] = []
    sup: TaskSupervisor = SupervisedTasks(
        clock, bus, on_critical_failure=lambda name, exc: critical.append(name)
    )
    runs = 0

    async def flaky() -> None:
        nonlocal runs
        runs += 1
        raise RuntimeError("boom")

    sup.spawn("flaky", flaky, backoff=(0.5, 30.0), breaker=(6, 120.0))
    await clock.run_until_idle()
    assert runs == 1  # waiting 0.5 s of fake time before the first restart
    await clock.run_for(0.49)
    assert runs == 1
    await clock.run_for(0.02)
    assert runs == 2
    await clock.run_for(60.0)  # 1 + 2 + 4 + 8 s: breaker trips on the 6th failure
    assert runs == 6
    (status,) = sup.status()
    assert status.state is HealthState.FAILED and "crash loop" in status.detail
    counts = [e.count for e in restarts.drain()]  # type: ignore[attr-defined]
    assert counts == [1, 2, 3, 4, 5]  # exactly 5 restarts (§2.8)
    assert critical == []
    await sup.aclose()
    await asyncio.sleep(0)
