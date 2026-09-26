"""voice.endpointer: scripted-probability state machine and real Silero on synthetic signals."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.contracts.voice import (
    Endpointer,
    EndpointerConfig,
    VadEnd,
    VadEvent,
    VadPartial,
    VadStart,
)
from aivtube.testing.fakes import FakeVAD
from aivtube.voice.endpointer import SileroEndpointer

F = 512
DT = 0.032
T0 = 1000.0
CFG = EndpointerConfig()


def run_script(
    probs: Sequence[float],
    cfg: EndpointerConfig = CFG,
    *,
    ai: Sequence[bool] | bool = False,
    values: bool = False,
    ep: SileroEndpointer | None = None,
) -> tuple[list[tuple[int, VadEvent]], SileroEndpointer]:
    """Push one frame per probability; frame i starts at T0 + i*DT (value i if ``values``)."""
    ep = ep or SileroEndpointer(FakeVAD(probs), cfg)
    out: list[tuple[int, VadEvent]] = []
    for i in range(len(probs)):
        frame = np.full(F, float(i) if values else 0.0, np.float32)
        speaking = ai if isinstance(ai, bool) else ai[i]
        out.extend((i, ev) for ev in ep.push(frame, T0 + i * DT, speaking))
    return out, ep


def kinds(events: list[tuple[int, VadEvent]]) -> list[str]:
    return [type(ev).__name__ for _, ev in events]


class TestScripted:
    def test_is_an_endpointer(self) -> None:
        assert isinstance(SileroEndpointer(FakeVAD(), CFG), Endpointer)

    def test_short_blip_emits_nothing(self) -> None:
        events, _ = run_script([0.0] * 10 + [0.95] * 7 + [0.0] * 40)  # 224 ms < 250 ms
        assert events == []

    def test_min_speech_start_preroll_and_end_after_600ms(self) -> None:
        probs = [0.0] * 20 + [0.9] * 30 + [0.0] * 30
        events, _ = run_script(probs, values=True)
        assert kinds(events) == ["VadStart", "VadEnd"]
        (i_start, start), (i_end, end) = events
        assert i_start == 27  # the 8th speech frame (8 x 32 ms >= 250 ms)
        assert isinstance(start, VadStart) and start.barge is False
        assert start.t == pytest.approx(T0 + 20 * DT)  # the onset, not the detection time
        assert i_end == 49 + 19  # 19 silent frames (608 ms >= 600 ms)
        assert isinstance(end, VadEnd)
        assert end.t == pytest.approx(T0 + 50 * DT)  # end of the last speech frame
        audio = end.audio
        assert audio.dtype == np.float32
        assert audio.size == 4800 + 30 * F + 3200  # 300 ms pre-roll + speech + 200 ms tail
        # the pre-roll is exactly the 4800 samples before the onset (frames 10.625 .. 19)
        assert audio[0] == 10.0 and audio[4799] == 19.0 and audio[4800] == 20.0
        assert np.all(audio[4800 : 4800 + 30 * F] >= 20.0)

    def test_hysteresis_keeps_the_turn_open_between_thresholds(self) -> None:
        probs = [0.9] * 10 + [0.4] * 40 + [0.0] * 25
        events, _ = run_script(probs)
        assert kinds(events) == ["VadStart", "VadEnd"]
        assert events[1][1].t == pytest.approx(T0 + 50 * DT)

    def test_barge_threshold_while_she_speaks(self) -> None:
        events, _ = run_script([0.55] * 20, ai=True)  # threshold 0.5 < p < barge 0.6
        assert events == []
        events, _ = run_script([0.55] * 20, ai=False)
        assert kinds(events) == ["VadStart"] and events[0][1].barge is False
        events, _ = run_script([0.65] * 20, ai=True)
        assert kinds(events) == ["VadStart"]
        start = events[0][1]
        assert isinstance(start, VadStart) and start.barge is True

    def test_forced_split_is_partial_and_the_turn_continues(self) -> None:
        probs = [0.0] * 10 + [0.9] * int(40 / DT) + [0.0] * 30
        events, _ = run_script(probs)
        assert kinds(events) == ["VadStart", "VadPartial", "VadPartial", "VadEnd"]
        onset = T0 + 10 * DT
        (_, p1), (_, p2) = events[1], events[2]
        assert isinstance(p1, VadPartial) and isinstance(p2, VadPartial)
        assert p1.t - onset == pytest.approx(15.0, abs=DT)
        assert p2.t - p1.t == pytest.approx(15.0, abs=DT)
        n_speech = int(40 / DT)
        total = sum(ev.audio.size for _, ev in events if isinstance(ev, VadPartial | VadEnd))
        assert total == 4800 + n_speech * F + 3200  # contiguous, nothing lost or repeated

    def test_split_prefers_a_pause_within_the_last_second(self) -> None:
        dip_at = 10 + int(14.6 / DT)
        probs = [0.0] * 10 + [0.9] * int(20 / DT)
        probs[dip_at] = 0.36  # quieter, but still speech (>= neg_threshold)
        events, _ = run_script(probs)
        partial = next(ev for _, ev in events if isinstance(ev, VadPartial))
        assert partial.t == pytest.approx(T0 + (dip_at + 1) * DT)
        assert partial.audio.size == 4800 + (dip_at + 1 - 10) * F

    def test_split_waits_while_the_streamer_pauses(self) -> None:
        n = int(14.9 / DT)
        probs = [0.0] * 10 + [0.9] * n + [0.0] * 40
        events, _ = run_script(probs)
        assert kinds(events) == ["VadStart", "VadEnd"]  # the end wins; no silent partial

    def test_max_turn_ends_mid_speech(self) -> None:
        probs = [0.0] * 5 + [0.9] * int(70 / DT)
        events, _ = run_script(probs)
        assert kinds(events)[:5] == ["VadStart", "VadPartial", "VadPartial", "VadPartial", "VadEnd"]
        onset = T0 + 5 * DT
        _, end = events[4]
        assert isinstance(end, VadEnd)
        assert end.t - onset == pytest.approx(60.0, abs=DT)
        assert end.audio.size == pytest.approx(15.0 * 16000, abs=2 * F)  # untrimmed segment
        assert kinds(events)[5:] == ["VadStart"]  # still talking: a new turn starts

    def test_particle_endpointing_off_ignores_the_tail_hint(self) -> None:
        ep = SileroEndpointer(FakeVAD([0.9] * 10 + [0.0] * 40), CFG)
        ep.set_tail_hint("ไปกินข้าวกันครับ")
        assert ep.tail_hint == ""
        events, _ = run_script([0.9] * 10 + [0.0] * 40, ep=ep)
        assert [i for i, _ in events] == [7, 9 + 19]

    @pytest.mark.parametrize(
        ("hint", "silent_frames"),
        [("ไปกินข้าวกันครับ", 14), ("แล้วเราก็ไปเที่ยวกัน แล้วก็", 29), ("ฟังนะ", 14), ("", 19)],
    )
    def test_particle_endpointing_on(self, hint: str, silent_frames: int) -> None:
        cfg = dataclasses.replace(CFG, particle_endpointing=True)
        probs = [0.9] * 10 + [0.0] * 40
        ep = SileroEndpointer(FakeVAD(probs), cfg)
        events: list[tuple[int, VadEvent]] = []
        for i, _ in enumerate(probs):
            events.extend((i, e) for e in ep.push(np.zeros(F, np.float32), T0 + i * DT, False))
            if i == 12:
                ep.set_tail_hint(hint)
        assert [i for i, _ in events] == [7, 9 + silent_frames]

    def test_flush_ends_the_turn_at_once(self) -> None:
        probs = [0.9] * 12 + [0.0] * 3
        events, ep = run_script(probs)
        assert kinds(events) == ["VadStart"] and ep.in_speech
        (end,) = ep.flush(T0 + 15 * DT)
        assert isinstance(end, VadEnd) and end.t == pytest.approx(T0 + 12 * DT)
        assert end.audio.size == 12 * F + 3 * F  # tail pad capped by what was captured
        assert not ep.in_speech and ep.flush(T0 + 20 * DT) == []

    def test_quick_follow_up_gets_preroll_from_the_trailing_silence(self) -> None:
        probs = [0.9] * 10 + [0.0] * 19 + [0.0] * 3 + [0.9] * 10 + [0.0] * 19
        events, _ = run_script(probs)
        ends = [ev for _, ev in events if isinstance(ev, VadEnd)]
        assert len(ends) == 2
        assert ends[1].audio.size == 4800 + 10 * F + 3200

    def test_reset_forgets_the_turn_and_the_vad_state(self) -> None:
        vad = FakeVAD([0.9] * 12)
        ep = SileroEndpointer(vad, CFG)
        run_script([0.9] * 12, ep=ep)
        assert ep.in_speech
        ep.reset()
        assert not ep.in_speech and vad.resets == 1

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            SileroEndpointer(FakeVAD(), dataclasses.replace(CFG, neg_threshold=0.6))
        with pytest.raises(ValueError):
            SileroEndpointer(FakeVAD(), CFG, frame_ms=20)
        with pytest.raises(ValueError):
            SileroEndpointer(FakeVAD(), CFG).push(np.zeros(480, np.float32), T0, False)

    def test_set_config_changes_the_barge_threshold(self) -> None:
        ep = SileroEndpointer(FakeVAD([0.65] * 20), CFG)
        ep.set_config(dataclasses.replace(CFG, barge_threshold=0.7))
        events, _ = run_script([0.65] * 20, ai=True, ep=ep)
        assert events == [] and ep.config.barge_threshold == 0.7

    def test_push_prob_bypasses_the_vad(self) -> None:
        vad = FakeVAD()
        ep = SileroEndpointer(vad, CFG)
        for i in range(8):
            events = ep.push_prob(np.ones(F, np.float32), 0.9, T0 + i * DT, False)
        assert vad.calls == 0 and kinds([(0, e) for e in events]) == ["VadStart"]


# --- real Silero -----------------------------------------------------------------------------------


def run_audio(x16: np.ndarray, vad: Any, cfg: EndpointerConfig = CFG) -> list[VadEvent]:
    ep = SileroEndpointer(vad, cfg)
    events: list[VadEvent] = []
    for i in range(0, x16.size - F + 1, F):
        events.extend(ep.push(x16[i : i + F], T0 + i / 16000, False))
    return events


@pytest.fixture
def silero(silero_path: Path) -> Any:
    pytest.importorskip("onnxruntime")
    from aivtube.voice.vad import SileroOrtVAD

    return SileroOrtVAD(silero_path)


class TestRealSilero:
    def test_silence_and_noise_emit_nothing(self, silero: Any, synth: Any) -> None:
        x = np.concatenate([synth.silence(2.0), synth.noise(2.0, amp=0.003)])
        assert run_audio(x, silero) == []

    def test_tone_bursts_are_not_speech(self, silero: Any, synth: Any) -> None:
        bursts = [synth.tone(0.8, freq) for freq in (300.0, 440.0, 1000.0)]
        gaps = [synth.silence(0.7)] * 3
        x = np.concatenate([v for pair in zip(gaps, bursts, strict=True) for v in pair])
        assert run_audio(x, silero) == []

    def test_short_blip_emits_nothing(self, silero: Any, synth: Any) -> None:
        x = np.concatenate([synth.silence(1.0), synth.speech(0.12, seed=5), synth.silence(1.5)])
        assert run_audio(x, silero) == []

    def test_normal_utterance(self, silero: Any, synth: Any) -> None:
        x = np.concatenate([synth.silence(1.0), synth.speech(2.0, seed=1), synth.silence(1.5)])
        events = run_audio(x, silero)
        assert [type(e).__name__ for e in events] == ["VadStart", "VadEnd"]
        start, end = events
        assert isinstance(start, VadStart) and isinstance(end, VadEnd)
        assert start.t - T0 == pytest.approx(1.0, abs=0.1)
        assert end.t - T0 == pytest.approx(3.0, abs=0.15)  # Silero holds ~100 ms after speech
        # pre-roll (300 ms) + whole speech frames + 200 ms tail pad
        assert (end.audio.size - 4800 - 3200) % F == 0
        assert end.audio.size / 16000 == pytest.approx(0.3 + 2.0 + 0.2, abs=0.15)
        assert np.abs(end.audio[:4000]).max() < 1e-6  # the pre-roll is the silence before

    def test_two_utterances(self, silero: Any, synth: Any) -> None:
        x = np.concatenate(
            [
                synth.silence(0.5),
                synth.speech(1.0, seed=2),
                synth.silence(1.2),
                synth.speech(1.5, seed=3),
                synth.silence(1.0),
            ]
        )
        names = [type(e).__name__ for e in run_audio(x, silero)]
        assert names == ["VadStart", "VadEnd", "VadStart", "VadEnd"]

    def test_long_monologue_splits_at_15s_without_ending(self, silero: Any, synth: Any) -> None:
        chunk = synth.speech(8.0, seed=0)
        x = np.concatenate([synth.silence(1.0), *([chunk] * 4), synth.silence(1.0)])
        events = run_audio(x, silero)
        names = [type(e).__name__ for e in events]
        assert names == ["VadStart", "VadPartial", "VadPartial", "VadEnd"]
        start, p1, p2, _ = events
        assert p1.t - start.t == pytest.approx(15.0, abs=1.1)
        assert p2.t - p1.t == pytest.approx(15.0, abs=1.1)
