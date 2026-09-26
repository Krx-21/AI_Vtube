"""``SpeechQueue`` with FakeTTS, FakeAudioOut and FakeClock (voice.speech_queue acceptance).

A background task pumps the fake player one 10 ms block per 10 ms of fake time, like a device.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import numpy as np
import pytest

from aivtube.contracts import ipc
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.types import Segment, VoiceSpec
from aivtube.contracts.voice import AudioChunk, QuotaSpec, WordMark
from aivtube.testing.fakes import FakeAudioOut, FakeClock, FakePhraseCache, FakeTTS
from aivtube.voice.lipsync import LipSyncAnalyzer
from aivtube.voice.speech_queue import IpcSpeechCallbacks, SpeechQueue, truncate_heard
from aivtube.voice.tts import IdentityCfg, TTSRouter, phrase_key

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz")
CPS = 12.5


class TrackingTTS(FakeTTS):
    """FakeTTS that records when each synthesis starts and how many run at once."""

    def __init__(self, clock: FakeClock, **kw: Any) -> None:
        super().__init__(clock=clock, tone_hz=220.0, **kw)
        self._clk = clock
        self.starts: list[tuple[str, float]] = []
        self.active = 0
        self.max_active = 0

    async def synth(  # type: ignore[override]
        self, text: str, voice: VoiceSpec, *, first_audio_timeout: float, idle_timeout: float
    ) -> AsyncIterator[AudioChunk | WordMark]:
        self.starts.append((text, self._clk.now()))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            async for item in super().synth(
                text, voice, first_audio_timeout=first_audio_timeout, idle_timeout=idle_timeout
            ):
                yield item
        finally:
            self.active -= 1


class Recorder:
    """``SpeechCallbacks`` that records everything."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.log: list[tuple[str, Any]] = []
        self.started: list[dict[str, Any]] = []
        self.done: list[dict[str, Any]] = []
        self.utts: list[dict[str, Any]] = []
        self.lips: list[LipTrack] = []

    def segment_started(
        self,
        utt: str,
        seq: int,
        t_audible: float,
        duration_s: float | None,
        backend: str,
        silent: bool,
    ) -> None:
        d = dict(utt=utt, seq=seq, t_audible=t_audible, duration_s=duration_s, backend=backend,
                 silent=silent, at=self.clock.now())  # fmt: skip
        self.started.append(d)
        self.log.append(("started", (utt, seq)))

    def segment_done(self, utt: str, seq: int, heard: bool, heard_text: str) -> None:
        self.done.append(
            dict(utt=utt, seq=seq, heard=heard, heard_text=heard_text, at=self.clock.now())
        )
        self.log.append(("done", (utt, seq)))

    def utterance_done(
        self, utt: str, heard_text: str, cancelled: bool, reason: str | None
    ) -> None:
        self.utts.append(dict(utt=utt, heard_text=heard_text, cancelled=cancelled, reason=reason,
                              at=self.clock.now()))  # fmt: skip
        self.log.append(("utt_done", utt))

    def lip_track(self, track: LipTrack) -> None:
        self.lips.append(track)


class Rig:
    def __init__(self, *, identities: Mapping[str, IdentityCfg] | None = None, **qkw: Any) -> None:
        self.clock = FakeClock()
        self.player = FakeAudioOut(clock=self.clock.now, output_latency_s=0.04)
        self.edge = TrackingTTS(self.clock, name="edge", ttfa_s=0.25)
        self.azure = TrackingTTS(self.clock, name="azure", ttfa_s=0.3)
        self.cache = FakePhraseCache()
        self.router = TTSRouter(
            identities or {"premwadee": IdentityCfg("premwadee", VOICE, ("edge", "azure"))},
            {"edge": self.edge, "azure": self.azure},
            {"pailin": ["premwadee"]},
            clock=self.clock,
            cache=self.cache,
        )
        self.rec = Recorder(self.clock)
        self.q = SpeechQueue(
            player=self.player, tts=self.router, callbacks=self.rec, clock=self.clock, **qkw
        )
        self.pumped: list[tuple[float, np.ndarray]] = []
        self._pump: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Rig:
        self._pump = asyncio.create_task(self._pumper())
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.q.aclose()
        assert self._pump is not None
        self._pump.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self._pump

    async def _pumper(self) -> None:
        while True:
            await self.clock.sleep(0.01)
            self.pumped.append((self.clock.now(), self.player.pump(1)))

    def cache_phrase(self, text: str, seconds: float = 0.5) -> None:
        t = np.arange(int(seconds * 24000)) / 24000
        pcm = (6000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        self.cache.put(phrase_key(VOICE, text), AudioChunk(pcm, 24000))

    async def until(self, pred: Callable[[], bool], within: float = 30.0) -> None:
        await self.clock.run_until(pred, within=within, step=0.01)

    def utt_done(self, utt: str) -> bool:
        return any(u["utt"] == utt for u in self.rec.utts)

    def audible_at(self, t0: float, t1: float) -> float:
        """Peak output between two fake times."""
        peaks = [float(np.abs(y).max()) for t, y in self.pumped if t0 <= t <= t1 and y.size]
        return max(peaks, default=0.0)


def seg(
    utt: str, seq: int, text: str, *, last: bool = False, caption: str | None = None
) -> Segment:
    return Segment(
        utt_id=utt, seq=seq, text=text, caption=text if caption is None else caption, last=last
    )


TEXTS = ["สวัสดีค่ะ ทุกคน ", "วันนี้ ไพลิน จะมา เล่นเกม กันนะคะ ", "ขอบคุณ ที่มาดู นะคะ"]


async def test_plays_segments_in_order_and_reports_heard_text() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        for i, text in enumerate(TEXTS):
            assert await rig.q.segment(seg("u1", i, text, last=i == 2))
        await rig.until(lambda: rig.utt_done("u1"))
        assert [e for e in rig.rec.log if e[0] != "lip"] == [
            ("started", ("u1", 0)), ("done", ("u1", 0)),
            ("started", ("u1", 1)), ("done", ("u1", 1)),
            ("started", ("u1", 2)), ("done", ("u1", 2)),
            ("utt_done", "u1"),
        ]  # fmt: skip
        assert all(d["heard"] for d in rig.rec.done)
        assert rig.rec.utts[0] == dict(
            utt="u1",
            heard_text="".join(TEXTS),
            cancelled=False,
            reason=None,
            at=rig.rec.utts[0]["at"],
        )
        ts = [s["t_audible"] for s in rig.rec.started]
        assert ts == sorted(ts)
        # each segment lasts len/12.5 s and they follow each other closely
        for a, b, text in zip(rig.rec.started, rig.rec.started[1:], TEXTS, strict=False):
            gap = b["t_audible"] - a["t_audible"] - len(text) / CPS
            assert -0.01 <= gap < 0.1
        assert all(s["backend"] == "edge" and not s["silent"] for s in rig.rec.started)
        assert rig.q.idle


async def test_segment_k_plus_1_is_synthesised_before_segment_k_finishes() -> None:
    async with Rig() as rig:
        rig.edge.pace = True  # synthesis streams in real time: long overlaps
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        texts = [f"ประโยค ที่ {i} ยาว พอสมควร เลยนะ " for i in range(5)]
        for i, text in enumerate(texts):
            await rig.q.segment(seg("u1", i, text, last=i == 4))
        await rig.until(lambda: rig.utt_done("u1"), within=60)
        starts = [t for _, t in rig.edge.starts]
        done_at = [d["at"] for d in rig.rec.done]
        for k in range(4):
            assert starts[k + 1] < done_at[k]  # N+1 synthesised while N plays
        for k in range(2, 5):
            assert starts[k] >= done_at[k - 2]  # never more than one segment ahead
        assert rig.edge.max_active <= 2
        # streaming: the first segment started before its synthesis finished
        assert rig.rec.started[0]["duration_s"] is None


async def test_stop_now_fades_within_a_block_and_truncates_heard_text_at_the_last_word() -> None:
    async with Rig() as rig:
        words = "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด"
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u1", 0, TEXTS[0]))
        await rig.q.segment(seg("u1", 1, words))
        await rig.q.segment(seg("u1", 2, TEXTS[2], last=True))
        await rig.until(lambda: len(rig.rec.started) == 2)
        t_start = rig.rec.started[1]["t_audible"]
        # stop in the middle of "สี่" (starts at 15 chars / 12.5 cps = 1.2 s)
        await rig.until(lambda: rig.clock.now() >= t_start + 1.25 - 0.04)
        t_stop = rig.clock.now()
        await rig.q.stop("u1", "now", "barge_in", fade_ms=10)
        assert rig.rec.done[-1] == dict(
            utt="u1", seq=1, heard=False, heard_text="หนึ่ง สอง สาม", at=t_stop
        )
        assert rig.rec.utts[-1]["cancelled"] and rig.rec.utts[-1]["reason"] == "barge_in"
        assert rig.rec.utts[-1]["heard_text"] == TEXTS[0] + "หนึ่ง สอง สาม"
        await rig.clock.run_for(0.05)
        assert rig.audible_at(t_stop + 0.015, t_stop + 1.0) == 0.0  # silent after one block
        assert all(s["seq"] != 2 for s in rig.rec.started)
        assert await rig.q.segment(seg("u1", 3, "late")) is True  # accepted and dropped


async def test_stop_after_segment_finishes_the_current_segment_and_drops_the_rest() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        for i, text in enumerate(TEXTS):
            await rig.q.segment(seg("u1", i, text, last=i == 2))
        await rig.until(lambda: len(rig.rec.started) == 2)
        await rig.q.stop("u1", "after_segment", "filtered")
        await rig.q.play_canned("filtered", "pailin")  # not cached: synthesised, then queued
        await rig.until(lambda: rig.utt_done("u1"))
        assert [d["seq"] for d in rig.rec.done] == [0, 1] and all(d["heard"] for d in rig.rec.done)
        u = rig.rec.utts[-1]
        assert u["cancelled"] and u["reason"] == "filtered"
        assert u["heard_text"] == TEXTS[0] + TEXTS[1]
        done_at = u["at"]
        await rig.until(lambda: rig.q.idle, within=10)
        assert any(t == "Filtered." for t, _ in rig.edge.starts)
        assert rig.audible_at(done_at + 0.05, rig.clock.now()) > 0  # "Filtered." after it
        assert all(s["seq"] != 2 for s in rig.rec.started)


async def test_filler_plays_when_first_audio_is_late_and_at_most_once_per_30_s() -> None:
    async with Rig() as rig:
        rig.cache_phrase("อืม…", 0.4)
        rig.edge.ttfa_s = 1.9  # under the 2 s timeout, but later than the 1.2 s filler deadline
        await rig.q.begin("u1", "pailin", filler_after_s=1.2, gate_open=True)
        t_begin = rig.clock.now()
        await rig.q.segment(seg("u1", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.q.stats["fillers"] == 1)
        assert rig.clock.now() - t_begin == pytest.approx(1.2, abs=0.02)
        await rig.until(lambda: rig.utt_done("u1"))
        # the filler was audible before the segment and is never heard text
        assert rig.audible_at(t_begin + 1.25, t_begin + 1.5) > 0
        assert rig.rec.utts[0]["heard_text"] == TEXTS[0]
        assert "อืม…" in rig.q.recent_tts_text()
        assert any(tr.utt_id == "u1" and tr.final for tr in rig.rec.lips)
        # a second slow utterance within 30 s: no filler
        await rig.q.begin("u2", "pailin", filler_after_s=1.2, gate_open=True)
        await rig.q.segment(seg("u2", 0, TEXTS[1], last=True))
        await rig.until(lambda: rig.utt_done("u2"))
        assert rig.q.stats["fillers"] == 1
        # 30 s later it may play again
        await rig.clock.run_for(30.0)
        await rig.q.begin("u3", "pailin", filler_after_s=1.2, gate_open=True)
        await rig.q.segment(seg("u3", 0, TEXTS[2], last=True))
        await rig.until(lambda: rig.utt_done("u3"))
        assert rig.q.stats["fillers"] == 2


async def test_no_filler_when_audio_is_on_time_or_not_cached_or_not_requested() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=1.2, gate_open=True)
        rig.edge.ttfa_s = 1.9
        await rig.q.segment(seg("u1", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.utt_done("u1"))
        assert rig.q.stats["fillers"] == 0 and rig.q.stats["filler_missing"] == 1  # not cached
        rig.cache_phrase("อืม…")
        rig.edge.ttfa_s = 0.3
        await rig.q.begin("u2", "pailin", filler_after_s=1.2, gate_open=True)
        await rig.q.segment(seg("u2", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.utt_done("u2"))
        rig.edge.ttfa_s = 1.9
        await rig.q.begin("u3", "pailin", filler_after_s=None, gate_open=True)  # chat turn
        await rig.q.segment(seg("u3", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.utt_done("u3"))
        assert rig.q.stats["fillers"] == 0


async def test_gated_utterance_synthesises_but_waits_and_backpressure_after_8() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=False)
        for i in range(8):
            assert await rig.q.segment(seg("u1", i, f"ส่วนที่ {i} "))
        assert await rig.q.segment(seg("u1", 8, "เกิน")) is False
        await rig.clock.run_for(3.0)
        assert len(rig.edge.starts) == 2  # prefetch window, speculative
        assert rig.rec.started == []  # held until the gate opens
        await rig.q.open_gate("u1")
        await rig.until(lambda: len(rig.rec.done) == 1)
        assert await rig.q.segment(seg("u1", 8, "ส่วนสุดท้าย", last=True)) is True
        await rig.until(lambda: rig.utt_done("u1"), within=60)
        assert [d["seq"] for d in rig.rec.done] == list(range(9))


async def test_substitution_then_captions_fallback_through_the_queue() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        rig.edge.fail_rate = 1.0
        rig.edge.ttfa_s = 0.1
        await rig.q.segment(seg("u1", 0, TEXTS[0]))
        await rig.until(lambda: len(rig.rec.done) == 1)
        assert rig.rec.started[0]["backend"] == "azure"  # same voice, other backend
        rig.azure.fail_rate = 1.0
        await rig.q.segment(seg("u1", 1, TEXTS[1], caption="วันนี้ ไพลิน จะมาเล่นเกมกันนะคะ! "))
        await rig.q.segment(seg("u1", 2, TEXTS[2], last=True))
        await rig.until(lambda: rig.utt_done("u1"), within=60)
        silent = rig.rec.started[1:]
        assert all(s["silent"] and s["backend"] == "captions" for s in silent)
        assert silent[0]["duration_s"] == pytest.approx(len("วันนี้ ไพลิน จะมาเล่นเกมกันนะคะ! ") / CPS)
        # the captions overlay showed them: they count as said
        assert rig.rec.done[1]["heard"] and rig.rec.done[1]["heard_text"].startswith("วันนี้")
        assert not any(tr.seq in (1, 2) for tr in rig.rec.lips)  # no mouth for captions


async def test_canned_phrase_from_the_cache_needs_no_tts() -> None:
    async with Rig() as rig:
        rig.cache_phrase("Filtered.", 0.6)
        await rig.q.play_canned("filtered", "pailin")
        await rig.until(lambda: rig.q.idle and rig.q.stats["canned"] == 1)
        assert rig.edge.starts == [] and rig.azure.starts == []
        assert rig.rec.started == [] and rig.rec.utts == []  # canned phrases are not utterances
        tracks = [tr for tr in rig.rec.lips if tr.utt_id.startswith("canned-filtered")]
        assert len(tracks) == 1 and tracks[0].final and max(tracks[0].mouth) > 0.3
        assert "Filtered." in rig.q.recent_tts_text()


async def test_lip_tracks_start_at_the_audible_start_and_are_contiguous() -> None:
    async with Rig() as rig:
        rig.edge.pace = True
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u1", 0, "สวัสดีค่ะ ทุกคน วันนี้ อากาศ ดีมาก", last=True))
        await rig.until(lambda: rig.utt_done("u1"))
        tracks = [tr for tr in rig.rec.lips if tr.seq == 0]
        assert len(tracks) > 1  # emitted as the PCM was decoded
        assert tracks[0].t0 == pytest.approx(rig.rec.started[0]["t_audible"])
        n = 0
        for tr in tracks:
            assert tr.t0 == pytest.approx(tracks[0].t0 + n / tr.fps) and tr.fps == 60
            n += len(tr.mouth)
        assert [tr.final for tr in tracks] == [False] * (len(tracks) - 1) + [True]
        assert n == pytest.approx(len("สวัสดีค่ะ ทุกคน วันนี้ อากาศ ดีมาก") / CPS * 60, abs=2)
        assert max(max(tr.mouth) for tr in tracks) > 0.7  # a loud tone opens the mouth


async def test_mute_drops_segments_silently() -> None:
    async with Rig() as rig:
        rig.q.set_muted(True)
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u1", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.utt_done("u1"))
        assert rig.edge.starts == []
        assert rig.rec.started[0]["silent"] and rig.rec.started[0]["backend"] == "muted"
        assert rig.rec.done[0] == dict(
            utt="u1", seq=0, heard=False, heard_text="", at=rig.rec.done[0]["at"]
        )
        assert rig.q.recent_tts_text() == ""
        await rig.q.mute(False)
        await rig.q.duck(0.25)
        await rig.q.begin("u2", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u2", 0, TEXTS[0], last=True))
        await rig.until(lambda: rig.utt_done("u2"))
        assert rig.rec.done[-1]["heard"]
        assert 0.0 < rig.audible_at(0, rig.clock.now()) <= 8000 / 32768 * 0.25 + 0.01


async def test_external_cut_by_the_barge_in_reflex_ends_the_utterance() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u1", 0, "หนึ่ง สอง สาม สี่ ห้า หก"))
        await rig.q.segment(seg("u1", 1, TEXTS[1], last=True))
        await rig.until(lambda: len(rig.rec.started) == 1)
        t0 = rig.rec.started[0]["t_audible"]
        await rig.until(lambda: rig.clock.now() >= t0 + 1.0)
        rig.player.cancel(60.0)  # the BargeInController cuts locally
        await rig.until(lambda: rig.utt_done("u1"))
        assert rig.rec.utts[0]["cancelled"] and rig.rec.utts[0]["reason"] == "cut"
        assert rig.rec.done[0]["heard"] is False
        assert rig.rec.done[0]["heard_text"] in ("หนึ่ง สอง", "หนึ่ง สอง สาม")
        await rig.clock.run_for(1.0)
        assert [s["seq"] for s in rig.rec.started] == [0]


async def test_deferred_speculative_synthesis_runs_after_the_gate_opens() -> None:
    ident = {"az": IdentityCfg("az", VOICE, ("azure",))}
    async with Rig(identities=ident) as rig:
        rig.router.chains["pailin"] = ["az"]
        rig.azure.quota = QuotaSpec(18, 60.0)
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=False)
        await rig.q.segment(seg("u1", 0, TEXTS[0], last=True))
        await rig.clock.run_for(1.0)
        assert rig.azure.starts == []  # no speculative synthesis on a quota
        await rig.q.open_gate("u1")
        await rig.until(lambda: rig.utt_done("u1"))
        assert rig.rec.started[0]["backend"] == "azure" and rig.rec.done[0]["heard"]


async def test_drop_all_finishes_the_current_segment_then_goes_quiet() -> None:
    async with Rig() as rig:
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        for i, text in enumerate(TEXTS):
            await rig.q.segment(seg("u1", i, text, last=i == 2))
        await rig.until(lambda: len(rig.rec.started) == 1)
        task = asyncio.create_task(rig.q.drop_all())
        await rig.until(task.done)
        assert [d["seq"] for d in rig.rec.done] == [0] and rig.rec.done[0]["heard"]
        assert rig.rec.utts[0]["reason"] == "link_lost" and rig.q.idle


async def test_recent_tts_text_window_and_utterance_order() -> None:
    async with Rig() as rig:
        await rig.q.begin("a", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.begin("b", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("b", 0, "บี", last=True))
        await rig.q.segment(seg("a", 0, "เอ", last=True))
        await rig.until(lambda: rig.utt_done("b"))
        assert [u["utt"] for u in rig.rec.utts] == ["a", "b"]  # begin order
        assert "เอ" in rig.q.recent_tts_text() and "บี" in rig.q.recent_tts_text()
        await rig.clock.run_for(11.0)
        assert rig.q.recent_tts_text(10.0) == ""
        with pytest.raises(ValueError):
            await rig.q.begin("a", "pailin", filler_after_s=None, gate_open=True)
        assert await rig.q.segment(seg("nope", 0, "x")) is True  # unknown: dropped


async def test_stop_everything_including_queued_canned_and_shutdown() -> None:
    async with Rig() as rig:
        rig.cache_phrase("Filtered.")
        await rig.q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await rig.q.segment(seg("u1", 0, TEXTS[1]))
        await rig.q.play_canned("filtered", "pailin")
        await rig.q.begin("u2", "pailin", filler_after_s=None, gate_open=True)
        await rig.until(lambda: len(rig.rec.started) == 1)
        await rig.q.stop(None, "now", "freeze")
        assert {u["utt"] for u in rig.rec.utts} == {"u1", "u2"}
        assert all(u["cancelled"] and u["reason"] == "freeze" for u in rig.rec.utts)
        await rig.clock.run_for(2.0)
        assert rig.q.idle and not any(tr.utt_id.startswith("canned") for tr in rig.rec.lips)
        await rig.q.begin("u3", "pailin", filler_after_s=None, gate_open=True)
    assert rig.rec.utts[-1] == dict(
        utt="u3", heard_text="", cancelled=True, reason="shutdown", at=rig.rec.utts[-1]["at"]
    )


async def test_ipc_callbacks_produce_schema_valid_messages() -> None:
    sent: list[tuple[str, Mapping[str, Any]]] = []
    clock = FakeClock()
    player = FakeAudioOut(clock=clock.now)
    tts = TrackingTTS(clock, name="edge", ttfa_s=0.1)
    router = TTSRouter(
        {"premwadee": IdentityCfg("premwadee", VOICE, ("edge",))}, {"edge": tts}, {"pailin": ["premwadee"]},
        clock=clock,
    )  # fmt: skip
    q = SpeechQueue(player=player, tts=router, send=lambda t, d: sent.append((t, d)), clock=clock,
                    lipsync_factory=LipSyncAnalyzer)  # fmt: skip

    async def pump() -> None:
        while True:
            await clock.sleep(0.01)
            player.pump(1)

    pumper = asyncio.create_task(pump())
    try:
        await q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await q.segment(seg("u1", 0, "สวัสดีค่ะ ", last=False))
        await q.segment(seg("u1", 1, "ยินดีที่ได้รู้จัก", last=True))
        await clock.run_until(lambda: any(t == ipc.UTT_DONE for t, _ in sent), within=20)
    finally:
        await q.aclose()
        pumper.cancel()
    types = [t for t, _ in sent]
    assert {ipc.SEG_STARTED, ipc.SEG_DONE, ipc.UTT_DONE, ipc.LIP_TRACK} <= set(types)
    for i, (t, d) in enumerate(sent):
        ipc.validate(ipc.Envelope(v=ipc.IPC_VERSION, type=t, id=f"m{i}", ts=1.0, data=d))
    assert isinstance(IpcSpeechCallbacks(lambda t, d: None), IpcSpeechCallbacks)


def test_truncate_heard() -> None:
    marks = [WordMark("หนึ่ง", 0.0, 0.4), WordMark("สอง", 0.48, 0.24), WordMark("สาม", 0.8, 0.24)]
    text = "หนึ่ง สอง สาม"
    assert truncate_heard(text, marks, 2.0, 1.04) == text
    assert truncate_heard(text, marks, 0.9, 1.04) == "หนึ่ง สอง"
    assert truncate_heard(text, marks, 0.3, 1.04) == ""
    assert truncate_heard(text, marks, 0.0, 1.04) == ""
    # partial synthesis: the total is unknown, the marks still decide
    assert truncate_heard(text, marks, 0.75, float("inf")) == "หนึ่ง สอง"
    # no marks: proportional, cut at the last space
    assert truncate_heard("aaa bbb ccc", (), 0.7, 1.0) == "aaa bbb"
    # marks whose words are not in the text (normalisation) are skipped
    assert (
        truncate_heard("ฮ่าฮ่า ดี", [WordMark("555", 0, 0.1), WordMark("ฮ่าฮ่า", 0, 0.3)], 0.5, 1.0)
        == "ฮ่าฮ่า"
    )


async def test_mute_and_duck_route_through_an_injected_gain_setter() -> None:
    gains: list[tuple[float, float]] = []
    async with Rig(set_gain=lambda g, ramp: gains.append((g, ramp))) as rig:
        rig.q.set_muted(True)
        await rig.q.duck(0.5, ramp_ms=40)  # remembered while muted
        rig.q.set_muted(False)
        await rig.q.duck(1.0)
    assert gains == [(0.0, 20.0), (0.5, 20.0), (1.0, 30.0)]
