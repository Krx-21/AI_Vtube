"""FakeSpeechOutput: an exact audible timeline under FakeClock."""

from __future__ import annotations

import pytest

from aivtube.contracts.events import SegmentDone, SegmentStarted, UtteranceDone
from aivtube.contracts.types import Segment
from aivtube.testing.fakes import FakeClock, FakeEventBus, FakeSpeechOutput


def seg(utt: str, seq: int, text: str, last: bool = False, kind: str = "speech") -> Segment:
    return Segment(utt_id=utt, seq=seq, text=text, caption=text, last=last, kind=kind)  # type: ignore[arg-type]


@pytest.fixture
def rig() -> tuple[FakeClock, FakeEventBus, FakeSpeechOutput]:
    clock = FakeClock(start=100.0)
    bus = FakeEventBus(clock)
    return clock, bus, FakeSpeechOutput(bus, clock, chars_per_s=10.0)


async def test_timeline_is_exact(rig: tuple[FakeClock, FakeEventBus, FakeSpeechOutput]) -> None:
    clock, bus, out = rig
    await out.begin("u1", "pailin")
    await out.segment(seg("u1", 0, "abcdefghij"))  # 1.0 s
    await out.segment(seg("u1", 1, "klmno", last=True))  # 0.5 s
    await clock.run_for(2.0)
    assert out.audible == [(100.0, "abcdefghij"), (101.0, "klmno")]
    started = bus.of_type(SegmentStarted)
    assert [(e.seq, e.t_audible, e.duration_s) for e in started] == [
        (0, 100.0, 1.0),
        (1, 101.0, 0.5),
    ]
    done = bus.of_type(UtteranceDone)[0]
    assert done.ts == 101.5 and done.heard_text == "abcdefghijklmno" and done.character == "pailin"
    await out.aclose()


async def test_cut_truncates_heard_text_at_a_word(
    rig: tuple[FakeClock, FakeEventBus, FakeSpeechOutput],
) -> None:
    clock, bus, out = rig
    await out.begin("u1", "pailin")
    await out.segment(seg("u1", 0, "หนึ่ง สอง สาม สี่ ห้า", last=True))  # 21 chars, 2.1 s
    await clock.run_for(1.0)
    await out.stop("u1", "now", "barge_in")
    await clock.run_for(0.1)
    sd = bus.of_type(SegmentDone)[0]
    assert not sd.heard and sd.heard_text == "หนึ่ง สอง"
    done = bus.of_type(UtteranceDone)[0]
    assert done.cancelled and done.reason == "barge_in" and done.heard_text == "หนึ่ง สอง"
    await out.aclose()


async def test_utterances_queue_and_canned_filtered(
    rig: tuple[FakeClock, FakeEventBus, FakeSpeechOutput],
) -> None:
    clock, bus, out = rig
    await out.begin("u1", "pailin")
    await out.begin("u2", "pailin")
    await out.segment(seg("u2", 0, "second", last=True))
    await out.segment(seg("u1", 0, "first"))
    await clock.run_for(0.2)
    await out.stop("u1", "after_segment", "filtered")
    await out.play_canned("filtered", "pailin")
    await clock.run_for(5.0)
    assert out.heard_texts() == ["first", "second", "Filtered."]
    u1 = next(e for e in bus.of_type(UtteranceDone) if e.utt_id == "u1")
    assert u1.cancelled and u1.filtered
    assert out.call_names()[:3] == ["begin", "begin", "segment"]
    await out.aclose()


async def test_filler_mute_rate_and_backpressure(
    rig: tuple[FakeClock, FakeEventBus, FakeSpeechOutput],
) -> None:
    clock, bus, out = rig
    await out.begin("u1", "pailin", filler_after_s=1.2)
    await clock.run_for(1.5)
    assert out.audible == [(101.2, "อืม…")]
    await out.mute(True)
    await out.segment(seg("u1", 0, "muted words", last=True))
    await clock.run_for(3.0)
    started = bus.of_type(SegmentStarted)[0]
    assert started.silent and not bus.of_type(SegmentDone)[0].heard
    assert len(out.audible) == 1
    await out.mute(False)
    await out.set_voice_rate("pailin", 100)  # twice as fast
    await out.begin("u2", "pailin")
    for i in range(8):
        assert await out.segment(seg("u2", i, "x" * 10))
    assert await out.segment(seg("u2", 8, "x" * 10)) is False
    await clock.run_for(0.01)
    started_u2 = [e for e in bus.of_type(SegmentStarted) if e.utt_id == "u2"]
    assert started_u2[0].duration_s == pytest.approx(0.5)
    await out.stop(None, "now", "freeze")
    await clock.run_for(0.1)
    assert out.idle
    await out.aclose()


async def test_freeze_cuts_canned_phrases(
    rig: tuple[FakeClock, FakeEventBus, FakeSpeechOutput],
) -> None:
    clock, _, out = rig
    await out.play_canned("brain_freeze", "pailin")  # ~2.7 s at 10 chars/s
    await out.play_canned("thanks", "pailin")
    await clock.run_for(0.5)
    await out.stop(None, "now", "freeze")
    await clock.run_for(0.1)
    assert out.idle and out.heard_texts() == ["เอ๊ะ สมองไพลินค้างแป๊บนึงนะ"]
    await out.aclose()
