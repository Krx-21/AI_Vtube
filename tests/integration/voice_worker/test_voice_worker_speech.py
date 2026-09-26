"""The real voice worker in-process over a real localhost websocket: speaking (§10 layer 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from worker_kit import perf, stack, wait_for

from aivtube.contracts import ipc
from aivtube.contracts.events import (
    HealthChanged,
    ProviderSwitched,
    SegmentDone,
    SegmentStarted,
    UtteranceDone,
    UtteranceStarted,
)
from aivtube.contracts.types import HealthState


async def test_speaking_an_utterance_gives_ordered_events_lip_tracks_and_heard_text(
    tmp_path: Path,
) -> None:
    async with stack(tmp_path) as s:
        out = s.out
        health = [e.health for e in s.of(HealthChanged) if e.health.component == "voice"]
        assert health and health[-1].state is HealthState.OK, health
        c = out.constraints("pailin")
        assert (c.backend, c.identity, c.max_chars) == ("edge", "premwadee", 160)
        # "Filtered." and the other cached phrases were pre-synthesised before READY
        cached = list((tmp_path / "phrases" / "premwadee").glob("*.pcm"))
        assert len(cached) == len(s.payload["characters"]["pailin"]["cached_phrases"])

        s.bus.publish(UtteranceStarted(utt_id="u1", stimulus_id="s1", turn_id="turn-1"))
        t_begin = perf()
        await out.begin("u1", "pailin")
        for seg in s.speak("u1", ["สวัสดีค่ะ ทุกคน ", "วันนี้ไพลินมาแล้ว"]):
            assert await out.segment(seg)
        await wait_for(lambda: bool(s.of(UtteranceDone)), 15.0)
        mine = [e for e in s.of(SegmentStarted, SegmentDone, UtteranceDone) if e.utt_id == "u1"]
        assert [(type(e).__name__, getattr(e, "seq", None)) for e in mine] == [
            ("SegmentStarted", 0),
            ("SegmentDone", 0),
            ("SegmentStarted", 1),
            ("SegmentDone", 1),
            ("UtteranceDone", None),
        ]
        s0, d0, s1, d1, done = mine
        assert all(e.character == "pailin" and e.turn_id == "turn-1" for e in mine)
        assert s0.caption == "สวัสดีค่ะ ทุกคน " and s0.emotion == "happy"
        assert s0.backend == "edge" and not s0.silent
        assert t_begin < s0.t_audible < s1.t_audible  # DAC-timed, on the core clock
        assert s1.t_audible - s0.t_audible == pytest.approx(len("สวัสดีค่ะ ทุกคน") / 25.0, abs=0.12)
        assert d0.heard and d1.heard
        assert done.heard_text == "สวัสดีค่ะ ทุกคน วันนี้ไพลินมาแล้ว"
        assert not done.cancelled and done.reason is None and not done.filtered
        # lip tracks went to the avatar driver, not the bus, starting at the audible time
        tracks = [t for t in s.lips if t.utt_id == "u1" and t.seq == 1]
        assert tracks and tracks[0].t0 == pytest.approx(s1.t_audible, abs=1e-6)
        assert tracks[-1].final and max(max(t.mouth) for t in tracks) > 0.3
        # the speakers really played the tone
        assert float(np.abs(s.device.output_since(t_begin)).max()) > 0.1
        # every message the worker sent is valid Appendix A
        assert s.worker is not None
        for i, (t, d) in enumerate(list(s.worker.sent)):
            ipc.validate(ipc.Envelope(v=1, type=t, id=f"m{i}", ts=0.0, data=d))


async def test_stop_now_is_silent_within_one_block_and_cancelled(tmp_path: Path) -> None:
    async with stack(tmp_path, tts_cps=12.5) as s:
        out = s.out
        await out.begin("u1", "pailin")
        words = "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ "
        await out.segment(s.speak("u1", [words * 3])[0])
        await wait_for(lambda: bool(s.of(SegmentStarted)), 10.0)
        await wait_for(lambda: perf() > s.of(SegmentStarted)[0].t_audible + 1.0)
        before = s.device.output_between(perf() - 0.2, perf())
        assert float(np.abs(before).max()) > 0.1  # she is audible
        t_stop = perf()
        await out.stop("u1", "now", "skip", fade_ms=10)
        assert s.cuts and s.cuts[0][0] == "u1"  # the mouth closes at once
        await wait_for(lambda: bool(s.of(UtteranceDone)), 5.0)
        done = s.of(UtteranceDone)[0]
        assert done.cancelled and done.reason == "skip"
        seg_done = s.of(SegmentDone)[0]
        assert not seg_done.heard and seg_done.heard_text.startswith("หนึ่ง สอง")
        assert len(seg_done.heard_text) < len(words)  # cut at the last word heard
        # IPC hop + one 10 ms block of fade, then exact silence (tight bound: see below)
        await wait_for(lambda: perf() > t_stop + 0.5)
        after = s.device.output_since(t_stop + 0.25)
        assert after.size > 0 and float(np.abs(after).max()) == 0.0


@pytest.mark.timing
async def test_stop_now_reaches_silence_within_one_block_of_the_command(tmp_path: Path) -> None:
    async with stack(tmp_path, tts_cps=12.5) as s:
        await s.out.begin("u1", "pailin")
        await s.out.segment(s.speak("u1", ["ยาว " * 60])[0])
        await wait_for(lambda: bool(s.of(SegmentStarted)), 10.0)
        await wait_for(lambda: perf() > s.of(SegmentStarted)[0].t_audible + 0.5)
        t_stop = perf()
        await s.out.stop(None, "now", "freeze", fade_ms=10)
        await wait_for(lambda: perf() > t_stop + 0.3)
        audible = [
            ts for ts, blk in list(s.device.output) if ts >= t_stop and np.abs(blk).max() > 0
        ]
        # the command crosses the socket (~1 ms), the next 10 ms block fades out, then silence
        assert not audible or audible[-1] - t_stop < 0.05, [a - t_stop for a in audible]


async def test_mute_is_silent_and_a_failing_voice_falls_back_to_captions(tmp_path: Path) -> None:
    async with stack(tmp_path) as s:
        await s.out.mute(True)
        t0 = perf()
        await s.out.begin("u1", "pailin")
        await s.out.segment(s.speak("u1", ["เงียบ ๆ นะ"])[0])
        await wait_for(lambda: bool(s.of(UtteranceDone)), 10.0)
        started = s.of(SegmentStarted)[0]
        assert started.silent and started.backend == "muted"
        assert not s.of(SegmentDone)[0].heard  # MUTE: nobody heard it
        assert float(np.abs(s.device.output_since(t0)).max()) == 0.0
        await s.out.mute(False)

        s.tts.fail_rate = 1.0  # the only voice backend now fails every request
        await s.out.begin("u2", "pailin")
        await s.out.segment(s.speak("u2", ["ไม่มีเสียงแต่มีซับ"])[0])
        await wait_for(lambda: len(s.of(UtteranceDone)) == 2, 15.0)
        started2 = next(e for e in s.of(SegmentStarted) if e.utt_id == "u2")
        assert started2.silent and started2.backend == "captions"
        switched = s.of(ProviderSwitched)
        assert switched and switched[0].kind == "tts" and switched[0].new == "captions"
        done = s.of(UtteranceDone)[-1]
        assert not done.cancelled and done.heard_text == "ไม่มีเสียงแต่มีซับ"  # captions shown
