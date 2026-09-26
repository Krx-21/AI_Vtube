"""voice.barge: the barge-in reflex state machine (FakeClock) and its effect on real audio."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import numpy as np
import pytest

from aivtube.config.schema import BargeInConfig
from aivtube.contracts.ipc import BARGE_CANDIDATE, BARGE_CONFIRMED, BARGE_REJECTED
from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32
from aivtube.testing.fakes import FakeAudioOut, FakeClock, FakeSD
from aivtube.voice.audio_io import StreamingPlayer
from aivtube.voice.barge import (
    DEFAULT_BACKCHANNELS,
    BargeConfig,
    BargeInController,
    QuickTranscriber,
    RecentAudio,
    barge_threshold_for,
    is_backchannel,
)

DUCK = 10 ** (-12 / 20)


class SpyPlayer(FakeAudioOut):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock=clock.now)
        self.cancels: list[float] = []
        self.gains: list[tuple[float, float]] = []

    def cancel(self, fade_ms: float = 30.0) -> float:
        self.cancels.append(fade_ms)
        return super().cancel(fade_ms)

    def set_gain(self, gain: float, ramp_ms: float = 20.0) -> None:
        self.gains.append((gain, ramp_ms))
        super().set_gain(gain, ramp_ms)

    @property
    def target(self) -> float:
        return self._target_gain


class FakeQuickStt:
    """Scripted quick decodes: each call pops the next text (the last one repeats)."""

    def __init__(self, clock: FakeClock, texts: list[str], *, delay: float = 0.03) -> None:
        self.clock = clock
        self.texts = texts
        self.delay = delay
        self.calls: list[tuple[int, bool, str]] = []
        self.hang = False

    async def transcribe(
        self, pcm16k: F32, *, quick: bool = False, recent_tts_text: str = ""
    ) -> Transcript | None:
        self.calls.append((int(pcm16k.size), quick, recent_tts_text))
        if self.hang:
            await asyncio.Event().wait()
        await self.clock.sleep(self.delay)
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if not text:
            return None  # e.g. dropped by the post-processor (too short)
        return Transcript(text, True, pcm16k.size / 16000, self.delay * 1000, "fake")


class FakeFront:
    def recent_audio(self, seconds: float) -> F32:
        return np.zeros(round(seconds * 16000), np.float32)


class Harness:
    def __init__(self, texts: list[str], **kw: Any) -> None:
        self.clock = FakeClock()
        self.player = SpyPlayer(self.clock)
        self.stt = FakeQuickStt(self.clock, texts)
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.cuts = 0
        self.ctrl = BargeInController(
            player=self.player,
            stt=self.stt,
            frontend=FakeFront(),
            cfg=kw.pop("cfg", BargeConfig()),
            send=lambda kind, data: self.sent.append((kind, dict(data))),
            clock=self.clock,
            on_cut=self._on_cut,
            **kw,
        )
        self.task: asyncio.Task[None] | None = None

    def _on_cut(self) -> None:
        self.cuts += 1

    @property
    def kinds(self) -> list[str]:
        return [k for k, _ in self.sent]

    def she_speaks(self, seconds: float = 20.0) -> None:
        self.player.play(np.full(round(seconds * 48000), 0.1, np.float32), 48000)

    async def start(self) -> None:
        self.task = asyncio.create_task(self.ctrl.run())
        await self.clock.run_for(0.05)

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.task

    async def play_for(self, seconds: float) -> None:
        """Pump the fake device in lock-step with the fake clock (10 ms blocks)."""
        for _ in range(round(seconds / 0.01)):
            self.player.pump(1)
            await self.clock.run_for(0.01)

    def vad_start(self, onset_ago: float = 0.28) -> float:
        onset = self.clock.now() - onset_ago
        self.ctrl.on_vad_start(onset, True)
        return onset


@pytest.fixture
async def h() -> AsyncIterator[Harness]:
    harness = Harness(["เดี๋ยวก่อนนะ"])
    yield harness
    await harness.stop()


def test_protocols_and_threshold_table() -> None:
    assert isinstance(FakeQuickStt(FakeClock(), [""]), QuickTranscriber)
    assert isinstance(FakeFront(), RecentAudio)
    assert barge_threshold_for("none", 0.6) == 0.6
    assert barge_threshold_for("aec", 0.6) == 0.65
    assert barge_threshold_for("energy_dtd", 0.6) == 0.7
    assert barge_threshold_for("aec", 0.8) == 0.8


class TestBackchannels:
    @pytest.mark.parametrize(
        "text",
        ["อืม", " อืม ", "อืมมม", "อืม อืม", "ครับ", "ครับๆ", "ค่ะ!", "5555", "ฮ่าๆๆ", "ฮ่าฮ่า",
         "โอเค", "โอเคคค", "อ๋อ...", "เออ", "จ้า~"],
    )  # fmt: skip
    def test_backchannels(self, text: str) -> None:
        assert is_backchannel(text, DEFAULT_BACKCHANNELS)

    @pytest.mark.parametrize(
        "text", ["", "เดี๋ยวก่อน", "ไพลินหยุดก่อน", "ครับผม", "อืม เดี๋ยวนะ", "55 บาท", "ok"]
    )
    def test_not_backchannels(self, text: str) -> None:
        assert not is_backchannel(text, DEFAULT_BACKCHANNELS)

    def test_custom_list_and_config_defaults_match(self) -> None:
        assert is_backchannel("ok ok", frozenset({"OK"}))
        assert frozenset(BargeInConfig().backchannels) == DEFAULT_BACKCHANNELS

    def test_config_from_mapping(self) -> None:
        cfg = BargeConfig.from_mapping(BargeInConfig().model_dump())
        assert cfg.duck_db == -12.0 and cfg.quick_decode_s == 0.8
        assert cfg.backchannels == DEFAULT_BACKCHANNELS
        assert cfg.duck_gain == pytest.approx(DUCK)
        with pytest.raises(ValueError):
            BargeConfig(duck_db=3.0)


class TestReflex:
    async def test_real_speech_ducks_confirms_and_cuts(self, h: Harness) -> None:
        h.she_speaks()
        await h.start()
        await h.clock.run_for(1.5)  # well past her utterance start
        onset = h.vad_start()
        assert h.player.target == pytest.approx(DUCK)  # ducked at once
        assert h.player.gains[-1][1] <= 30.0  # ramp: + one block + device buffer ≈ 80 ms
        assert h.sent == [(BARGE_CANDIDATE, {"t": onset})]
        assert h.ctrl.state == "candidate"
        await h.clock.run_for(0.3)  # the decode runs 500 ms after the onset
        assert h.stt.calls[0] == (12800, True, "")  # the last 0.8 s, quick, no echo text
        assert h.kinds == [BARGE_CANDIDATE, BARGE_CONFIRMED]
        confirmed = h.sent[1][1]
        assert confirmed["cut_local"] is True and confirmed["text"] == "เดี๋ยวก่อนนะ"
        assert confirmed["t"] - onset == pytest.approx(0.53, abs=0.03)
        assert h.player.cancels == [60.0] and h.cuts == 1
        assert h.ctrl.state == "confirmed"
        await h.clock.run_for(0.5)
        assert h.player.target == 1.0  # the gain is restored once the fade has played
        h.ctrl.on_vad_end(h.clock.now())
        assert h.ctrl.state == "idle" and h.kinds == [BARGE_CANDIDATE, BARGE_CONFIRMED]

    async def test_backchannel_does_not_cut_and_times_out(self) -> None:
        h = Harness(["อืม"])
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(1.0)
            assert h.kinds == [BARGE_CANDIDATE] and len(h.stt.calls) >= 3  # keeps listening
            assert h.player.target == pytest.approx(DUCK)
            await h.clock.run_for(1.2)  # 2 s false-interruption timeout
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
            assert h.player.cancels == [] and h.player.target == 1.0
            assert h.ctrl.stats["rejected_timeout"] == 1
        finally:
            await h.stop()

    async def test_blip_ends_before_confirmation(self) -> None:
        h = Harness([""])
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start(onset_ago=0.25)
            await h.clock.run_for(0.1)
            h.ctrl.on_vad_end(h.clock.now())  # a 0.25 s blip: the VAD ends first
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
            assert h.player.target == 1.0 and h.player.cancels == []
            await h.clock.run_for(3.0)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED] and h.stt.calls == []
        finally:
            await h.stop()

    async def test_too_short_text_is_not_a_confirmation(self) -> None:
        h = Harness(["ไป", "ไป", "หยุดก่อน"])
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(0.3)
            assert h.kinds == [BARGE_CANDIDATE]
            await h.clock.run_for(0.6)  # re-decodes every 0.25 s
            assert h.kinds == [BARGE_CANDIDATE, BARGE_CONFIRMED]
            assert h.sent[1][1]["text"] == "หยุดก่อน"
        finally:
            await h.stop()

    async def test_backchannel_near_her_start_is_rejected_at_once(self) -> None:
        h = Harness(["อ๋อ"])
        try:
            await h.start()
            h.she_speaks()
            await h.clock.run_for(0.4)
            h.vad_start(onset_ago=0.3)  # onset 0.1 s after she started
            await h.clock.run_for(0.4)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
            assert h.ctrl.stats["rejected_backchannel"] == 1 and h.player.target == 1.0
        finally:
            await h.stop()

    async def test_duck_only_never_cuts(self) -> None:
        h = Harness(["หยุดก่อน"], policy="duck_only")
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(0.5)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_CONFIRMED]
            assert h.sent[1][1]["cut_local"] is False and h.player.cancels == []
            assert h.player.target == pytest.approx(DUCK)  # keeps playing ducked
            h.ctrl.on_vad_end(h.clock.now())
            assert h.player.target == 1.0 and h.cuts == 0
        finally:
            await h.stop()

    async def test_policy_off_ignores_everything(self) -> None:
        h = Harness(["หยุดก่อน"], policy="off")
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(1.0)
            assert h.sent == [] and h.player.gains == [] and h.stt.calls == []
        finally:
            await h.stop()

    async def test_switching_policy_off_drops_a_candidate(self, h: Harness) -> None:
        h.she_speaks()
        await h.start()
        await h.clock.run_for(1.5)
        h.vad_start()
        h.ctrl.set_policy("off")
        assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED] and h.player.target == 1.0

    async def test_no_candidate_when_she_is_silent(self, h: Harness) -> None:
        await h.start()
        h.ctrl.on_vad_start(h.clock.now(), True)  # stale barge flag; nothing is playing
        h.ctrl.on_vad_start(h.clock.now(), False)
        assert h.sent == [] and h.ctrl.stats["ignored_idle"] == 2

    async def test_aec_warmup_ignores_barge_in_for_3_s(self) -> None:
        h = Harness(["หยุดก่อน"])
        try:
            h.ctrl.set_echo_mode("aec")
            await h.start()
            h.she_speaks()
            await h.clock.run_for(1.0)
            h.vad_start()
            await h.clock.run_for(1.5)
            assert h.sent == [] and h.ctrl.stats["ignored_warmup"] == 1
            await h.clock.run_for(1.0)  # 3.5 s after she started: warm-up over
            h.vad_start()
            await h.clock.run_for(0.4)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_CONFIRMED]
        finally:
            await h.stop()

    async def test_echo_modes_pass_recent_tts_text_and_half_duplex_never_barges(
        self,
    ) -> None:
        h = Harness(["หยุดก่อน"], recent_tts_text=lambda: "วันนี้อากาศดีมาก")
        try:
            h.ctrl.set_echo_mode("energy_dtd")
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(0.4)
            assert h.stt.calls[0][2] == "วันนี้อากาศดีมาก"
            h.ctrl.on_vad_end(h.clock.now())
            h.ctrl.set_echo_mode("half_duplex")
            h.vad_start()
            assert h.ctrl.stats["ignored_policy"] == 1
        finally:
            await h.stop()

    async def test_her_playback_ending_closes_the_candidate(self) -> None:
        h = Harness(["อืม"])
        try:
            h.player.play(np.full(round(1.8 * 48000), 0.1, np.float32), 48000)
            await h.start()
            await h.play_for(1.5)
            h.vad_start()
            await h.play_for(0.7)  # she finishes at 1.8 s (+ 0.25 s tail)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
            assert h.ctrl.stats["rejected_playback_ended"] == 1 and h.player.target == 1.0
        finally:
            await h.stop()

    async def test_hung_decode_hits_its_deadline(self) -> None:
        h = Harness(["หยุดก่อน"], cfg=BargeConfig(decode_timeout_s=0.3))
        try:
            h.stt.hang = True
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(0.6)
            assert h.ctrl.stats["decode_timeout"] >= 1 and h.kinds == [BARGE_CANDIDATE]
            await h.clock.run_for(1.5)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
        finally:
            await h.stop()

    async def test_vad_end_during_a_decode_discards_its_result(self) -> None:
        h = Harness(["หยุดก่อน"])
        h.stt.delay = 0.2
        try:
            h.she_speaks()
            await h.start()
            await h.clock.run_for(1.5)
            h.vad_start()
            await h.clock.run_for(0.25)  # the decode is in flight
            assert len(h.stt.calls) == 1
            h.ctrl.on_vad_end(h.clock.now())
            await h.clock.run_for(0.3)
            assert h.kinds == [BARGE_CANDIDATE, BARGE_REJECTED]
            assert h.ctrl.stats["stale_decode"] == 1 and h.player.cancels == []
        finally:
            await h.stop()

    async def test_base_gain_is_what_an_unduck_restores(self, h: Harness) -> None:
        h.she_speaks()
        await h.start()
        await h.clock.run_for(1.5)
        h.ctrl.set_base_gain(0.0)  # muted by the operator
        h.vad_start()
        assert h.player.target == 0.0
        h.ctrl.on_vad_end(h.clock.now())
        assert h.player.target == 0.0
        h.ctrl.set_base_gain(1.0)
        assert h.player.target == 1.0

    async def test_send_failures_are_counted(self) -> None:
        clock = FakeClock()
        player = SpyPlayer(clock)

        def broken(kind: str, data: Mapping[str, Any]) -> None:
            raise ConnectionError("link down")

        ctrl = BargeInController(
            player=player,
            stt=FakeQuickStt(clock, ["x"]),
            frontend=FakeFront(),
            send=broken,
            clock=clock,
        )
        player.play(np.ones(48000, np.float32), 48000)
        ctrl.on_vad_start(clock.now(), True)
        assert ctrl.stats["send_error"] == 1 and ctrl.state == "candidate"
        ctrl.reset()
        assert ctrl.state == "idle" and player.target == 1.0

    async def test_suppress_backchannel_near_her_utterance(self, h: Harness) -> None:
        await h.start()
        t_before = h.clock.now()
        assert not h.ctrl.suppress_backchannel("ครับ", t_before)  # she has not spoken yet
        h.she_speaks(1.0)
        await h.play_for(0.5)
        assert h.ctrl.suppress_backchannel("อืม", h.clock.now())  # over her
        assert not h.ctrl.suppress_backchannel("หยุดก่อน", h.clock.now())
        await h.play_for(1.0)  # she ended ~1.3 s after she started
        now = h.clock.now()
        assert h.ctrl.suppress_backchannel("ค่ะ", now + 0.5)
        assert not h.ctrl.suppress_backchannel("ค่ะ", now + 1.6)


class TestAudio:
    """The reflex against the real player on FakeSD: duck and cut as heard at the device."""

    async def test_duck_within_80ms_and_cut_to_silence(self) -> None:
        clock = FakeClock()
        sd = FakeSD()
        player = StreamingPlayer(backend=sd, clock=clock.now, stall_timeout_s=3600.0)
        player.start()
        sent: list[str] = []
        try:
            ctrl = BargeInController(
                player=player,
                stt=FakeQuickStt(clock, ["หยุดก่อนนะ"]),
                frontend=FakeFront(),
                send=lambda kind, data: sent.append(kind),
                clock=clock,
            )
            player.play(np.full(48000 * 5, 0.5, np.float32), 48000)
            sd.pump(10)
            task = asyncio.create_task(ctrl.run())
            await clock.run_for(1.5)
            ctrl.on_vad_start(clock.now() - 0.28, True)
            out = sd.pump(4)[0]
            # 30 ms ramp: -12 dB from the 4th block on; + 40 ms device buffer = 80 ms audible
            assert out[3 * 480] == pytest.approx(0.5 * DUCK, rel=1e-3)
            assert player.output_latency_s + 0.03 + 0.01 <= 0.08 + 1e-9
            await clock.run_for(0.3)
            assert sent == [BARGE_CANDIDATE, BARGE_CONFIRMED]
            out = sd.pump(10)[0]
            assert np.abs(out[:480]).max() > 0  # the 60 ms fade ...
            assert np.all(out[6 * 480 :] == 0.0)  # ... then silence (≤ 100 ms with the buffer)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            player.close()
