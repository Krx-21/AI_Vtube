"""voice.frontend: timing, resampling, echo gating, listening controls, the recent-audio ring."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.contracts.speech import VoicePolicy
from aivtube.contracts.voice import F32, EndpointerConfig, VadEnd, VadEvent, VadStart
from aivtube.testing.fakes import FakeAudioOut, FakeEchoCanceller, FakeEndpointer, FakeVAD
from aivtube.voice.aec import EnergyDTD
from aivtube.voice.endpointer import SileroEndpointer
from aivtube.voice.frontend import VoiceFrontEnd, post_to_loop

T0 = 500.0


class Sink:
    def __init__(self) -> None:
        self.events: list[VadEvent] = []

    def __call__(self, ev: VadEvent) -> None:
        self.events.append(ev)

    @property
    def names(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


def feed_blocks(fe: VoiceFrontEnd, x: F32, *, rate: int, block: int, t0: float = T0) -> None:
    for i in range(0, x.size - block + 1, block):
        fe.feed(x[i : i + block], t0 + i / rate)


def scripted(probs: list[float], **kw: Any) -> tuple[VoiceFrontEnd, Sink, SileroEndpointer]:
    ep = SileroEndpointer(FakeVAD(probs), EndpointerConfig())
    sink = Sink()
    kw.setdefault("player", None)
    kw.setdefault("aec", None)
    kw.setdefault("dtd", None)
    fe = VoiceFrontEnd(ep, in_rate=16000, on_event=sink, **kw)
    return fe, sink, ep


class TestTiming:
    def test_frames_at_16k_carry_their_capture_time(self) -> None:
        ep = FakeEndpointer()
        fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, in_rate=16000, on_event=Sink())
        feed_blocks(fe, np.zeros(16000, np.float32), rate=16000, block=160)
        assert [n for n, _, _ in ep.pushes] == [512] * 31
        for m, (_, t, ai) in enumerate(ep.pushes):
            assert t == pytest.approx(T0 + m * 0.032, abs=1e-9) and ai is False

    def test_48k_input_is_resampled_and_time_aligned(self) -> None:
        pytest.importorskip("soxr")
        ep = FakeEndpointer()
        fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, on_event=Sink())
        feed_blocks(fe, np.zeros(48000, np.float32), rate=48000, block=480)
        assert len(ep.pushes) >= 29  # soxr holds back a few ms
        for m, (n, t, _) in enumerate(ep.pushes):
            assert n == 512 and t == pytest.approx(T0 + m * 0.032, abs=1e-9)

    def test_impulse_lands_in_the_right_frame(self) -> None:
        pytest.importorskip("soxr")

        class Rec(FakeEndpointer):
            def __init__(self) -> None:
                super().__init__()
                self.frames: list[tuple[float, F32]] = []

            def push(self, frame16k: F32, t: float, ai_speaking: bool) -> list[VadEvent]:
                self.frames.append((t, np.array(frame16k)))
                return []

        ep = Rec()
        fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, on_event=Sink())
        x = np.zeros(48000, np.float32)
        x[24000] = 1.0  # at 0.5 s
        feed_blocks(fe, x, rate=48000, block=480)
        t_peak, frame = max(ep.frames, key=lambda f: float(np.abs(f[1]).max()))
        t_hit = t_peak + int(np.argmax(np.abs(frame))) / 16000
        assert t_hit == pytest.approx(T0 + 0.5, abs=1e-4)


class TestRing:
    def test_recent_audio_returns_the_latest_samples(self) -> None:
        fe, _, _ = scripted([])
        ramp = np.arange(24000, dtype=np.float32)
        feed_blocks(fe, ramp, rate=16000, block=160)
        assert np.array_equal(fe.recent_audio(0.5), ramp[-8000:])
        assert np.array_equal(fe.recent_audio(5.0), ramp[-16000:])  # clamped to the 1 s ring
        assert fe.recent_audio(0.0).size == 0

    def test_ring_wraps_and_is_readable_from_another_thread(self) -> None:
        fe, _, _ = scripted([], ring_s=0.1)
        got: list[F32] = []
        feed_blocks(fe, np.arange(2000, dtype=np.float32), rate=16000, block=300)
        th = threading.Thread(target=lambda: got.append(fe.recent_audio(0.05)))
        th.start()
        th.join()
        assert np.array_equal(got[0], np.arange(1000, 1800, dtype=np.float32))


class TestListening:
    def test_disabled_emits_nothing_and_reenabling_starts_clean(self) -> None:
        fe, sink, _ = scripted([0.9] * 200)
        fe.set_enabled(False)
        feed_blocks(fe, np.zeros(16000, np.float32), rate=16000, block=512)
        assert sink.events == [] and not fe.listening
        fe.set_enabled(True)
        feed_blocks(fe, np.zeros(512 * 8, np.float32), rate=16000, block=512)
        assert sink.names == ["VadStart"]

    def test_deafened_policy_suppresses_events(self) -> None:
        fe, sink, _ = scripted([0.9] * 100 + [0.0] * 30)
        fe.apply_policy(VoicePolicy(mic_mode="deafened"))
        feed_blocks(fe, np.zeros(512 * 130, np.float32), rate=16000, block=512)
        assert sink.events == []

    def test_ptt_released_emits_nothing(self) -> None:
        fe, sink, _ = scripted([0.9] * 100)
        fe.apply_policy(VoicePolicy(mic_mode="ptt", ptt_active=False))
        feed_blocks(fe, np.zeros(512 * 50, np.float32), rate=16000, block=512)
        assert sink.events == []

    def test_ptt_release_ends_the_turn_at_once(self) -> None:
        fe, sink, _ = scripted([0.9] * 100)
        fe.set_mic_mode("ptt")
        fe.set_ptt(True)
        feed_blocks(fe, np.zeros(512 * 20, np.float32), rate=16000, block=512)
        assert sink.names == ["VadStart"]
        fe.set_ptt(False)
        fe.feed(np.zeros(512, np.float32), T0 + 20 * 0.032)
        assert sink.names == ["VadStart", "VadEnd"]
        end = sink.events[1]
        assert isinstance(end, VadEnd) and end.t == pytest.approx(T0 + 20 * 0.032)

    def test_deafen_mid_turn_drops_it(self) -> None:
        fe, sink, ep = scripted([0.9] * 100)
        feed_blocks(fe, np.zeros(512 * 20, np.float32), rate=16000, block=512)
        fe.set_enabled(False)
        fe.feed(np.zeros(512, np.float32), T0 + 1.0)
        assert sink.names == ["VadStart"] and not ep.in_speech

    def test_event_handler_errors_are_counted(self) -> None:
        def bad(ev: VadEvent) -> None:
            raise RuntimeError("boom")

        ep = SileroEndpointer(FakeVAD([0.9] * 20), EndpointerConfig())
        fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, in_rate=16000, on_event=bad)
        feed_blocks(fe, np.zeros(512 * 10, np.float32), rate=16000, block=512)
        assert fe.stats["errors"] == 1 and fe.stats["VadStart"] == 1


class TestEcho:
    def _player(self, clock: Any) -> FakeAudioOut:
        out = FakeAudioOut(sample_rate=16000, blocksize=160, clock=clock, output_latency_s=0.0)
        return out

    def test_barge_threshold_while_she_is_audible(self, manual_clock: Any) -> None:
        player = self._player(manual_clock)
        player.play(np.full(16000, 0.1, np.float32), 16000)  # queued: she is speaking
        fe, sink, _ = scripted([0.55] * 20 + [0.65] * 20, player=player)
        feed_blocks(fe, np.zeros(512 * 40, np.float32), rate=16000, block=512)
        assert sink.names == ["VadStart"]
        start = sink.events[0]
        assert isinstance(start, VadStart) and start.barge is True
        assert start.t == pytest.approx(T0 + 20 * 0.032)

    def test_reference_is_fed_once_and_before_the_mic_block(self, manual_clock: Any) -> None:
        calls: list[tuple[str, int]] = []

        class Rec(FakeEchoCanceller):
            def feed_reference(self, block: F32) -> None:
                calls.append(("ref", int(block[0] * 1000)))
                super().feed_reference(block)

            def process(self, mic_block: F32) -> F32:
                calls.append(("mic", 0))
                return super().process(mic_block)

        player = self._player(manual_clock)
        for i in range(6):
            player.play(np.full(160, (i + 1) / 1000, np.float32), 16000)
        fe, _, _ = scripted([], player=player, aec=Rec())
        player.pump(2)
        fe.feed(np.zeros(160, np.float32), T0)
        manual_clock.advance(0.02)
        player.pump(3)
        fe.feed(np.zeros(160, np.float32), T0 + 0.01)
        fe.feed(np.zeros(160, np.float32), T0 + 0.02)
        assert calls == [
            ("ref", 1),
            ("ref", 2),
            ("mic", 0),
            ("ref", 3),
            ("ref", 4),
            ("ref", 5),
            ("mic", 0),
            ("mic", 0),
        ]

    def test_reference_is_resampled_to_the_mic_rate(self, manual_clock: Any) -> None:
        pytest.importorskip("soxr")
        aec = FakeEchoCanceller()
        player = FakeAudioOut(sample_rate=48000, blocksize=480, clock=manual_clock)
        fe, _, _ = scripted([], player=player, aec=aec)
        player.play(np.zeros(48000, np.float32), 48000)
        for i in range(20):
            player.pump(1)  # the fake stamps every block of an unmoved clock alike
            fe.feed(np.zeros(160, np.float32), T0 + i * 0.01)
        fed = sum(r.size for r in aec.refs)
        assert 20 * 160 - 400 <= fed <= 20 * 160  # 16 kHz worth (soxr holds a little back)

    def test_aec_failure_falls_back_to_the_raw_mic(self) -> None:
        class Broken(FakeEchoCanceller):
            def process(self, mic_block: F32) -> F32:
                raise RuntimeError("ffi died")

        fe, sink, _ = scripted([0.9] * 20, aec=Broken())
        feed_blocks(fe, np.zeros(512 * 10, np.float32), rate=16000, block=512)
        assert fe.stats["aec_errors"] == 1 and sink.names == ["VadStart"]

    def test_half_duplex_ignores_the_mic_while_she_speaks(self, manual_clock: Any) -> None:
        player = self._player(manual_clock)
        player.play(np.full(16000, 0.1, np.float32), 16000)
        fe, sink, _ = scripted([0.99] * 40, player=player, half_duplex=True)
        feed_blocks(fe, np.zeros(512 * 20, np.float32), rate=16000, block=512)
        assert sink.events == [] and fe.stats["gated"] == 20
        player.cancel()
        player.pump(10)
        manual_clock.advance(1.0)  # she has stopped (past the 0.25 s tail)
        feed_blocks(fe, np.zeros(512 * 10, np.float32), rate=16000, block=512, t0=T0 + 1)
        assert sink.names == ["VadStart"]

    def test_energy_dtd_gates_echo_but_passes_the_streamer(self, manual_clock: Any) -> None:
        manual_clock.set(T0)  # the player's reference and the mic share one timeline
        player = self._player(manual_clock)
        ref_level = 0.2
        player.play(np.full(16000 * 3, ref_level, np.float32), 16000)
        # VAD: unsure while only her echo is heard (learns coupling), then fooled by the echo,
        # then the streamer talks over her
        probs = [0.1] * 10 + [0.9] * 15 + [0.9] * 15
        fe, sink, _ = scripted(probs, player=player, dtd=EnergyDTD(k=2.0))
        t = T0
        for i in range(len(probs)):
            player.pump(4)  # keep the reference ahead of the mic
            level = 0.3 if i >= 25 else 0.02  # echo at 0.1 x reference, then a loud streamer
            fe.feed(np.full(512, level, np.float32), t)
            t += 0.032
            manual_clock.advance(0.032)
        assert sink.names == ["VadStart"]
        start = sink.events[0]
        assert isinstance(start, VadStart) and start.barge is True
        assert start.t == pytest.approx(T0 + 25 * 0.032)
        assert fe.stats["gated"] >= 15

    def test_configure_echo_applies_at_the_next_feed(self, manual_clock: Any) -> None:
        player = self._player(manual_clock)
        player.play(np.full(16000, 0.1, np.float32), 16000)
        fe, sink, _ = scripted([0.99] * 40, player=player)
        fe.configure_echo(aec=None, dtd=None, half_duplex=True)
        feed_blocks(fe, np.zeros(512 * 20, np.float32), rate=16000, block=512)
        assert sink.events == []


async def test_post_to_loop_runs_on_the_loop_thread() -> None:
    loop = asyncio.get_running_loop()
    got: list[tuple[int, str]] = []
    done = asyncio.Event()

    def handler(x: int) -> None:
        got.append((x, threading.current_thread().name))
        done.set()

    post = post_to_loop(loop, handler)
    threading.Thread(target=post, args=(7,), name="mic").start()
    await asyncio.wait_for(done.wait(), 2.0)
    assert got == [(7, threading.current_thread().name)]


def test_post_to_loop_after_close_is_quiet() -> None:
    loop = asyncio.new_event_loop()
    post = post_to_loop(loop, lambda x: None)
    loop.close()
    post(1)  # no exception


class TestRealSilero:
    def test_48k_speech_gives_one_utterance(self, silero_path: Path, synth: Any) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("soxr")
        from aivtube.voice.vad import SileroOrtVAD

        sink = Sink()
        ep = SileroEndpointer(SileroOrtVAD(silero_path), EndpointerConfig())
        fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, on_event=sink)
        x16 = np.concatenate([synth.silence(1.0), synth.speech(2.0, seed=1), synth.silence(1.5)])
        feed_blocks(fe, synth.up48(x16), rate=48000, block=480)
        assert sink.names == ["VadStart", "VadEnd"]
        start, end = sink.events
        assert start.t == pytest.approx(T0 + 1.0, abs=0.1)
        assert end.t == pytest.approx(T0 + 3.0, abs=0.15)
        assert isinstance(end, VadEnd) and end.audio.size / 16000 == pytest.approx(2.5, abs=0.15)
        assert fe.recent_audio(0.8).size == 12800
