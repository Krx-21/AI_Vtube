"""``ConsoleSpeechOutput`` (text mode) and the core-side ``voice.configure`` payload."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from aivtube.config import load_characters, load_config
from aivtube.contracts import ipc
from aivtube.contracts.events import SegmentDone, SegmentStarted, UtteranceDone, UtteranceStarted
from aivtube.contracts.speech import SpeechOutput, VoicePolicy
from aivtube.contracts.types import Segment
from aivtube.infra import AsyncEventBus
from aivtube.speech import (
    ConsoleSpeechOutput,
    default_constraints,
    voice_configure,
    voice_policy,
)
from aivtube.testing.contracts import SpeechOutputHarness, case_id, speech_output_suite
from aivtube.testing.fakes import FakeClock

ROOT = Path(__file__).resolve().parents[3]


def _console_harness() -> SpeechOutputHarness:
    clock = FakeClock()
    bus = AsyncEventBus(clock)
    out = ConsoleSpeechOutput(io.StringIO(), bus, clock)
    return SpeechOutputHarness(output=out, bus=bus, advance=clock.run_for, aclose=out.aclose)


@pytest.mark.parametrize("case", speech_output_suite(_console_harness), ids=case_id)
async def test_console_speech_output_contract(case: object) -> None:
    await case()  # type: ignore[operator]


class Rig:
    def __init__(self, **kw: object) -> None:
        self.clock = FakeClock()
        self.bus = AsyncEventBus(self.clock)
        self.sub = self.bus.subscribe(SegmentStarted, SegmentDone, UtteranceDone, name="t")
        self.text = io.StringIO()
        self.out = ConsoleSpeechOutput(
            self.text,
            self.bus,
            self.clock,
            names={"pailin": "ไพลิน"},
            **kw,  # type: ignore[arg-type]
        )

    def events(self) -> list[object]:
        return self.sub.drain()  # type: ignore[attr-defined, no-any-return]


async def test_console_prints_captions_with_simulated_speaking_time() -> None:
    rig = Rig(chars_per_s=10.0)
    out: SpeechOutput = rig.out
    assert isinstance(out, SpeechOutput) and out.ready()
    rig.bus.publish(UtteranceStarted(utt_id="u1", stimulus_id="s1", turn_id="t-7"))
    await out.begin("u1", "pailin")
    assert await out.segment(Segment("u1", 0, "สวัสดีค่ะ ", "สวัสดีค่ะ ", emotion="happy"))
    assert await out.segment(Segment("u1", 1, "ไพลินเอง", "ไพลินเอง", last=True))
    t0 = rig.clock.now()
    await rig.clock.run_until(lambda: rig.out.idle, within=10.0)
    events = rig.events()
    assert [type(e).__name__ for e in events] == [
        "SegmentStarted",
        "SegmentDone",
        "SegmentStarted",
        "SegmentDone",
        "UtteranceDone",
    ]
    s0, _d0, s1, _d1, done = events
    assert isinstance(s0, SegmentStarted) and isinstance(s1, SegmentStarted)
    assert s0.t_audible == pytest.approx(t0) and s0.duration_s == pytest.approx(1.0)
    assert s1.t_audible == pytest.approx(t0 + 1.0)  # "สวัสดีค่ะ " is 10 chars at 10 cps
    assert s0.caption == "สวัสดีค่ะ " and s0.emotion == "happy" and s0.backend == "console"
    assert isinstance(done, UtteranceDone)
    assert done.heard_text == "สวัสดีค่ะ ไพลินเอง" and not done.cancelled and not done.filtered
    assert all(e.character == "pailin" and e.turn_id == "t-7" for e in events)  # type: ignore[attr-defined]
    assert rig.text.getvalue() == "ไพลิน: สวัสดีค่ะ \nไพลิน: ไพลินเอง\n"
    await rig.out.aclose()


async def test_console_cut_mute_filtered_and_filler() -> None:
    rig = Rig(chars_per_s=10.0)
    out = rig.out
    await out.begin("u1", "pailin", filler_after_s=0.5)
    await rig.clock.run_for(0.6)
    assert rig.text.getvalue() == "ไพลิน: อืม…\n"  # nothing to say yet: a filler
    await out.segment(Segment("u1", 0, "หนึ่ง สอง สาม สี่", "หนึ่ง สอง สาม สี่"))
    await rig.clock.run_for(0.95)  # "หนึ่ง สอง " heard
    await out.stop("u1", "now", "barge_in")
    await rig.clock.run_until(lambda: out.idle, within=5.0)
    *_, seg_done, done = rig.events()
    assert isinstance(seg_done, SegmentDone) and not seg_done.heard
    assert seg_done.heard_text == "หนึ่ง สอง"
    assert isinstance(done, UtteranceDone) and done.cancelled and done.reason == "barge_in"

    await out.begin("u2", "pailin")
    await out.segment(Segment("u2", 0, "ประโยคที่ดี ", "ประโยคที่ดี "))
    await out.segment(Segment("u2", 1, "ไม่ควรได้ยิน", "ไม่ควรได้ยิน", last=True))
    await rig.clock.run_for(0.1)
    await out.stop("u2", "after_segment", "filtered")
    await out.play_canned("filtered", "pailin")
    await rig.clock.run_until(lambda: out.idle, within=10.0)
    events = rig.events()
    assert [e.seq for e in events if isinstance(e, SegmentStarted)] == [0]  # type: ignore[attr-defined]
    done2 = events[-1]
    assert isinstance(done2, UtteranceDone) and done2.filtered and done2.cancelled
    assert rig.text.getvalue().endswith("ไพลิน: ประโยคที่ดี \nไพลิน: Filtered.\n")

    await out.mute(True)
    await out.begin("u3", "pailin")
    await out.segment(Segment("u3", 0, "เงียบ", "เงียบ", last=True))
    before = rig.text.getvalue()
    await rig.clock.run_until(lambda: out.idle, within=5.0)
    started, seg3, done3 = rig.events()
    assert isinstance(started, SegmentStarted) and started.silent
    assert isinstance(seg3, SegmentDone) and not seg3.heard and seg3.heard_text == ""
    assert isinstance(done3, UtteranceDone) and not done3.cancelled
    assert rig.text.getvalue() == before  # muted: nothing printed
    await out.aclose()


async def test_console_i7_accepts_only_segments_and_runs_the_hook() -> None:
    seen: list[Segment] = []

    def gate_hook(seg: Segment) -> None:
        assert seg.text == seg.caption, "only gate output reaches the voice (I7)"
        seen.append(seg)

    rig = Rig(i7_check=gate_hook)
    await rig.out.begin("u1", "pailin")
    with pytest.raises(TypeError, match="I7"):
        await rig.out.segment("raw LLM text")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        await rig.out.segment(Segment("u1", 0, "ไม่ผ่าน", "ผ่าน"))
    assert await rig.out.segment(Segment("u1", 0, "ผ่าน", "ผ่าน", last=True))
    assert [s.text for s in seen] == ["ผ่าน"]
    await rig.out.aclose()


async def test_console_rate_and_policy_and_duplicate_begin() -> None:
    rig = Rig(chars_per_s=10.0)
    out = rig.out
    await out.set_voice_rate("pailin", 100)  # twice as fast
    await out.set_policy(VoicePolicy(mic_mode="deafened"))
    await out.duck(0.3)
    assert out.policy.mic_mode == "deafened" and out.gain == 0.3
    await out.begin("u1", "pailin")
    with pytest.raises(ValueError):
        await out.begin("u1", "pailin")
    await out.segment(Segment("u1", 0, "abcdefghij", "abcdefghij", last=True))
    await rig.clock.run_until(lambda: out.idle, within=5.0)
    started = rig.events()[0]
    assert isinstance(started, SegmentStarted) and started.duration_s == pytest.approx(0.5)
    await out.aclose()


def test_voice_configure_payload_is_valid_ipc() -> None:
    cfg = load_config(ROOT, profile="stream", env={})
    chars = load_characters(cfg)
    data = voice_configure(cfg, chars)
    ipc.validate(ipc.Envelope(v=1, type=ipc.VOICE_CONFIGURE, id="c", ts=0.0, data=data))
    pailin = data["characters"]["pailin"]
    assert pailin["cached_phrases"][0] == "Filtered."
    assert pailin["identity_chain"] == ["premwadee"]
    assert pailin["stt_aliases"]["ไพลิน"]
    assert [e["name"] for e in data["stt_chain"]] == ["typhoon_rt", "pythaiasr"]  # no consent
    assert data["tts"]["chunk"]["max_chars"] == 160
    policy = voice_policy(cfg)
    assert policy.mic_mode == cfg.mic.mode and policy.barge_in == cfg.barge_in.policy
    c = default_constraints(cfg, chars["pailin"])
    assert (c.first_min_chars, c.min_chars, c.max_chars) == (8, 40, 160)
    assert (c.identity, c.backend) == ("premwadee", "edge")
