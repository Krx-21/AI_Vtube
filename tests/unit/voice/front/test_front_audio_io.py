"""voice.audio_io against FakeSD: device resolution, the always-open player and mic capture."""

from __future__ import annotations

import itertools
import threading
from typing import Any

import numpy as np
import pytest

from aivtube.testing.contracts import audio_out_suite, case_id
from aivtube.testing.contracts._base import wait_until
from aivtube.testing.fakes import WASAPI, FakeSD, FakeWasapiSettings
from aivtube.voice.audio_io import (
    AudioDeviceError,
    MicCapture,
    StreamingPlayer,
    list_devices,
    reinit_portaudio,
    resolve_device,
)

# FakeSD "windows" layout: 0 MME mic, 1 MME speakers, 2 MME cable (truncated),
# 3 WASAPI mic, 4 WASAPI speakers, 5 WASAPI cable.


def tone(seconds: float, sr: int = 48000, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_player(sd: FakeSD, **kw: Any) -> StreamingPlayer:
    kw.setdefault("stall_timeout_s", 3600.0)
    return StreamingPlayer(backend=sd, **kw)


# --- device resolution ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_default_is_wasapi_never_mme(self) -> None:
        sd = FakeSD()
        assert resolve_device(None, "output", sd) == 4
        assert resolve_device(None, "input", sd) == 3
        assert resolve_device("", "output", sd) == 4

    def test_name_substrings_resolve_inside_wasapi(self) -> None:
        sd = FakeSD()
        assert resolve_device("CABLE Input", "output", sd) == 5
        assert resolve_device("cable input", "output", sd) == 5
        assert resolve_device("Realtek", "output", sd) == 4
        assert resolve_device("usb mic", "input", sd) == 3

    def test_kind_filters_channels(self) -> None:
        sd = FakeSD()
        with pytest.raises(LookupError):
            resolve_device("USB Mic", "output", sd)

    def test_exact_name_wins_over_substring(self) -> None:
        sd = FakeSD()
        sd.add_device("Speakers", WASAPI, 0, 2)
        assert resolve_device("speakers", "output", sd) == 6
        assert resolve_device("Speakers (Realtek", "output", sd) == 4

    def test_falls_back_to_later_preferred_host_api(self) -> None:
        sd = FakeSD()
        idx = sd.add_device("Old Headset", "MME", 0, 2)
        assert resolve_device("Old Headset", "output", sd) == idx

    def test_non_windows_host_apis_are_searched_when_no_preferred_api_exists(self) -> None:
        sd = FakeSD("empty")
        sd.add_device("pulse", "ALSA", 32, 32, default=True)
        assert resolve_device(None, "output", sd) == 0
        assert resolve_device("puls", "input", sd) == 0

    def test_unknown_name_raises_lookup_error(self) -> None:
        with pytest.raises(LookupError):
            resolve_device("Nope", "output", FakeSD())

    def test_int_passes_through(self) -> None:
        assert resolve_device(1, "output", FakeSD()) == 1

    def test_list_devices_names_host_apis(self) -> None:
        devs = list_devices(FakeSD())
        assert [d.hostapi for d in devs].count(WASAPI) == 3
        cable = devs[5]
        assert cable.name.startswith("CABLE Input") and cable.max_out == 8 and cable.max_in == 0


# --- player ---------------------------------------------------------------------------------------


def _suite_factory() -> tuple[StreamingPlayer, Any]:
    sd = FakeSD()
    player = make_player(sd)
    return player, lambda n: sd.pump(n)[0]


@pytest.mark.parametrize("case", audio_out_suite(_suite_factory), ids=case_id)
def test_streaming_player_contract(case: Any) -> None:
    case()


class TestStreamingPlayer:
    def test_opens_wasapi_shared_with_auto_convert(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            (st,) = sd.output_streams
            assert st.device == 4 and st.samplerate == 48000 and st.channels == 2
            assert st.blocksize == 480 and st.dtype == "float32" and st.kw["latency"] == 0.04
            assert isinstance(st.extra_settings, FakeWasapiSettings)
            assert st.extra_settings.auto_convert and not st.extra_settings.exclusive
            assert player.output_latency_s == pytest.approx(0.04)
            assert player.device_ok and player.device_index == 4
        finally:
            player.close()

    def test_non_wasapi_device_gets_no_wasapi_settings(self) -> None:
        sd = FakeSD()
        player = make_player(sd, device=1)  # MME: WasapiSettings would fail to open
        player.start()
        try:
            assert sd.output_streams[0].extra_settings is None
        finally:
            player.close()

    def test_missing_device_raises_audio_device_error(self) -> None:
        player = make_player(FakeSD(), device="No Such Speaker")
        with pytest.raises(AudioDeviceError):
            player.start()
        player.close()

    def test_block_after_cancel_is_fade_then_zeros_and_marks_fail(self) -> None:
        sd = FakeSD()
        player = make_player(sd, fade_ms=8.0)
        player.start()
        marks: list[bool] = []
        try:
            player.play(tone(1.0, amp=0.5), 48000)
            player.mark(lambda heard, t: marks.append(heard))
            before = sd.pump(3)[0]
            assert np.abs(before).max() > 0.4
            dropped = player.cancel(fade_ms=8.0)
            assert dropped == pytest.approx(1.0 - 3 * 0.01, abs=1e-6)
            block = sd.pump(1)[0]
            # the fade continues exactly where playback stopped, ramping 1 -> 0 over 8 ms
            ramp = np.linspace(1.0, 0.0, 384, endpoint=False, dtype=np.float32)
            expected = tone(1.0, amp=0.5)[1440 : 1440 + 384] * ramp
            assert np.allclose(block[:384], expected, atol=1e-6)
            assert np.all(block[384:] == 0.0)
            assert np.all(sd.pump(5)[0] == 0.0)
            wait_until(lambda: marks == [False], 2.0, what="cancelled mark")
            assert player.queued_s() == 0.0
        finally:
            player.close()

    def test_longer_fade_spans_blocks_then_silence(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.play(tone(1.0), 48000)
            sd.pump(2)
            player.cancel(fade_ms=60.0)
            out = sd.pump(8)[0]
            assert np.abs(out[: 6 * 480]).max() > 0.0
            assert np.all(out[6 * 480 :] == 0.0)
        finally:
            player.close()

    def test_marks_fire_true_at_dac_time_with_injected_clock(self, manual_clock: Any) -> None:
        sd = FakeSD()
        player = make_player(sd, clock=manual_clock)
        player.start()
        fired: list[tuple[bool, float]] = []
        try:
            player.play(tone(0.1), 48000)
            player.mark(lambda heard, t: fired.append((heard, t)))
            for _ in range(12):
                sd.pump(1)
                manual_clock.advance(0.01)
            # audio starts audible at 100.00 + 0.04 (dac - current) and lasts 0.1 s; the 10th
            # block's callback ran at clock 100.09, so the end mark is audible at 100.14.
            assert fired == []  # not before the clock reaches the audible time
            manual_clock.set(100.139)
            threading.Event().wait(0.03)
            assert fired == []
            manual_clock.set(100.2)
            wait_until(lambda: bool(fired), 2.0, what="mark")
            heard, t = fired[0]
            assert heard is True
            assert t == pytest.approx(100.14, abs=1e-9)
        finally:
            player.close()

    def test_mark_order_is_preserved_across_cancel(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        order: list[tuple[str, bool]] = []
        try:
            player.play(tone(0.02), 48000)
            player.mark(lambda h, t: order.append(("a", h)))
            player.play(tone(0.5), 48000)
            player.mark(lambda h, t: order.append(("b", h)))
            sd.pump(3)  # mark "a" is consumed (audible 40 ms later)
            player.cancel()
            wait_until(lambda: len(order) == 2, 2.0, what="both marks")
            assert order == [("a", True), ("b", False)]
        finally:
            player.close()

    def test_gain_ramps_linearly_then_holds(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.play(np.full(48000, 0.5, np.float32), 48000)
            sd.pump(1)
            player.set_gain(0.25, ramp_ms=10.0)  # 480 samples
            ramped = sd.pump(1)[0] / 0.5
            assert ramped[0] == pytest.approx(1.0 - 0.75 / 480, abs=1e-5)
            assert ramped[239] == pytest.approx(1.0 - 0.75 * 240 / 480, abs=1e-5)
            assert ramped[-1] == pytest.approx(0.25, abs=1e-6)
            held = sd.pump(2)[0] / 0.5
            assert np.allclose(held, 0.25)
            player.set_gain(1.0, ramp_ms=0.0)
            assert np.allclose(sd.pump(1)[0][1:] / 0.5, 1.0)
            assert player.gain == 1.0
        finally:
            player.close()

    def test_duck_reaches_minus_12_db_within_three_blocks(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.play(np.full(48000, 0.5, np.float32), 48000)
            player.set_gain(10 ** (-12 / 20), ramp_ms=30.0)
            out = sd.pump(4)[0] / 0.5
            assert out[3 * 480] == pytest.approx(10 ** (-12 / 20), rel=1e-4)
        finally:
            player.close()

    def test_reference_is_post_gain_and_strictly_increasing(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.set_gain(0.5, ramp_ms=0.0)
            player.play(np.full(4800, 0.4, np.float32), 48000)
            sd.pump(5)
            ts = [t for t, _ in player.reference]
            assert len(ts) == 5 and all(b > a for a, b in itertools.pairwise(ts))
            assert np.allclose(player.reference[-1][1][10:], 0.2)
            assert player.ref_rms_max(ts[0] - 1.0, ts[-1] + 1.0) == pytest.approx(0.2, abs=1e-3)
        finally:
            player.close()

    def test_resampler_tail_is_flushed_at_marks(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.play(tone(0.1, sr=24000), 24000)
            assert player.queued_s() < 0.1  # soxr holds some samples back
            player.mark(lambda h, t: None)
            assert player.queued_s() == pytest.approx(0.1, abs=1e-4)
        finally:
            player.close()

    def test_2d_input_is_mixed_down_and_nan_is_sanitised(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            stereo = np.stack([np.full(480, 0.2), np.full(480, 0.4)], axis=1).astype(np.float32)
            player.play(stereo, 48000)
            bad = np.full(480, np.nan, np.float32)
            player.play(bad, 48000)
            out = sd.pump(2)[0]
            assert np.allclose(out[:480], 0.3) and np.all(out[480:] == 0.0)
        finally:
            player.close()

    def test_callback_error_zero_fills_and_is_counted(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        try:
            player.play(tone(0.1), 48000)
            out = np.full((480, 2), np.nan, np.float32)
            status = type("S", (), {"output_underflow": False})()
            player._callback(out, 960, None, status)  # wrong frame count: must not raise
            assert np.all(out == 0.0) and player.stats["cb_error"] == 1
            sd.output_streams[0].pump(1, underflow=True)
            assert player.stats["underflow"] == 1
        finally:
            player.close()

    def test_is_speaking_tail(self, manual_clock: Any) -> None:
        sd = FakeSD()
        player = make_player(sd, clock=manual_clock)
        player.start()
        try:
            assert not player.is_speaking()
            player.play(tone(0.02), 48000)
            assert player.is_speaking(tail_s=0.0)
            sd.pump(3)
            assert player.is_speaking(tail_s=0.0)  # audible until 100.06
            manual_clock.set(100.07)
            assert not player.is_speaking(tail_s=0.0)
            assert player.is_speaking(tail_s=0.25)
        finally:
            player.close()

    def test_levels_reach_on_level(self) -> None:
        sd = FakeSD()
        levels: list[float] = []
        player = make_player(sd, on_level=levels.append)
        player.start()
        try:
            player.play(np.full(4800, 0.5, np.float32), 48000)
            sd.pump(10)
            wait_until(lambda: bool(levels) and max(levels) > 0.49, 2.0, what="levels")
        finally:
            player.close()

    def test_close_is_idempotent_and_fails_pending_marks(self) -> None:
        sd = FakeSD()
        player = make_player(sd)
        player.start()
        marks: list[bool] = []
        player.play(tone(1.0), 48000)
        player.mark(lambda h, t: marks.append(h))
        player.close()
        player.close()
        assert marks == [False]
        player.mark(lambda h, t: marks.append(h))  # after close: fails at once
        player.play(tone(0.1), 48000)
        assert marks == [False, False] and player.stats["dropped"] == 1
        assert not sd.output_streams[0].active

    def test_device_loss_reopens_by_name_every_interval(self, manual_clock: Any) -> None:
        sd = FakeSD()
        lost: list[bool] = []
        player = make_player(
            sd, device="Realtek", clock=manual_clock, on_device_lost=lambda: lost.append(True)
        )
        player.start()
        try:
            sd.unplug("Realtek")  # the stream dies: finished_callback fires
            wait_until(lambda: player.stats["reopen_failed"] >= 1, 2.0, what="failed reopen")
            assert lost == [True] and not player.device_ok
            failures = player.stats["reopen_failed"]
            threading.Event().wait(0.05)
            assert player.stats["reopen_failed"] == failures  # retries wait for the interval
            sd.replug("Realtek")
            manual_clock.advance(2.0)
            wait_until(lambda: player.stats["reopened"] == 1, 2.0, what="reopen")
            assert player.device_ok and lost == [True]
            active = [s for s in sd.output_streams if s.active]
            assert len(active) == 1 and active[0].device == 4
            player.play(np.full(480, 0.3, np.float32), 48000)
            assert np.allclose(sd.pump(1)[0], 0.3)
        finally:
            player.close()

    def test_stall_reopens_and_resolves_the_name_again(self, manual_clock: Any) -> None:
        sd = FakeSD()
        player = StreamingPlayer("Realtek", backend=sd, clock=manual_clock)
        player.start()
        try:
            sd.unplug("USB Mic")
            reinit_portaudio(sd)  # device list shifts; every stream is closed (no callback)
            assert sd.reinit_count == 1
            manual_clock.advance(1.5)  # > 1 s without callbacks
            wait_until(lambda: player.stats["reopened"] == 1, 2.0, what="reopen after stall")
            active = [s for s in sd.output_streams if s.active]
            assert len(active) == 1
            assert list_devices(sd)[active[0].device].name == "Speakers (Realtek(R) Audio)"
            assert active[0].device == 2  # a new index, found by name
        finally:
            player.close()

    def test_mirror_plays_the_same_post_gain_blocks(self) -> None:
        sd = FakeSD()
        player = make_player(sd, mirror_device="CABLE Input")
        player.start()
        try:
            main, mirror = sd.output_streams
            assert main.device == 4 and mirror.device == 5
            player.set_gain(0.5, ramp_ms=0.0)
            player.play(tone(0.1, amp=0.8), 48000)
            outs = [sd.pump(1) for _ in range(12)]
            main_out = np.concatenate([o[0] for o in outs])
            mirror_out = np.concatenate([o[1] for o in outs])
            # the mirror primes two blocks, so it lags the main output by one block here
            assert np.allclose(mirror_out[480:], main_out[:-480], atol=1e-7)
            assert len(player.reference) == 12  # only the main sink is the AEC reference
        finally:
            player.close()

    def test_mirror_loss_does_not_touch_the_main_output(self, manual_clock: Any) -> None:
        sd = FakeSD()
        player = make_player(sd, mirror_device="CABLE Input", clock=manual_clock)
        player.start()
        try:
            sd.unplug("CABLE")
            wait_until(lambda: player.stats["mirror_lost"] == 1, 2.0, what="mirror lost")
            player.play(np.full(480, 0.3, np.float32), 48000)
            assert np.allclose(sd.output_streams[0].pump(1), 0.3)
            assert player.device_ok and player.stats["device_lost"] == 0
            sd.replug("CABLE")
            manual_clock.advance(2.0)
            wait_until(lambda: len([s for s in sd.output_streams if s.active]) == 2, 2.0)
        finally:
            player.close()


# --- mic --------------------------------------------------------------------------------------------


class Frames:
    def __init__(self) -> None:
        self.items: list[tuple[np.ndarray, float, str]] = []
        self.lock = threading.Lock()

    def __call__(self, x: np.ndarray, t: float) -> None:
        with self.lock:
            self.items.append((x, t, threading.current_thread().name))

    def __len__(self) -> int:
        with self.lock:
            return len(self.items)


class TestMicCapture:
    def test_opens_wasapi_mono_and_delivers_on_the_mic_thread(self, manual_clock: Any) -> None:
        sd = FakeSD()
        frames = Frames()
        mic = MicCapture(backend=sd, clock=manual_clock, stall_timeout_s=3600.0)
        mic.start(frames)
        try:
            (st,) = sd.input_streams
            assert st.device == 3 and st.channels == 1 and st.blocksize == 480
            assert isinstance(st.extra_settings, FakeWasapiSettings)
            assert st.extra_settings.auto_convert
            sd.push(np.arange(960, dtype=np.float32) / 1000.0)
            wait_until(lambda: len(frames) == 2, 2.0, what="two mic frames")
            (x0, t0, name0), (x1, _, _) = frames.items
            assert name0 == "mic" and x0.dtype == np.float32 and x0.shape == (480,)
            assert x1[0] == pytest.approx(0.48)
            assert t0 == pytest.approx(100.0 - 0.01)  # now - (currentTime - inputBufferAdcTime)
            assert mic.sample_rate == 48000 and mic.block_samples == 480
            assert mic.stats["frames"] == 2
        finally:
            mic.close()

    def test_stereo_fallback_mixes_down(self) -> None:
        sd = FakeSD()
        idx = sd.add_device("Picky Mic", "MME", 2, 0, refuse_mono=True)
        frames = Frames()
        mic = MicCapture(idx, backend=sd, stall_timeout_s=3600.0)
        mic.start(frames)
        try:
            (st,) = sd.input_streams
            assert st.channels == 2 and st.extra_settings is None
            st.push(np.stack([np.full(480, 0.2), np.full(480, 0.6)], axis=1))
            wait_until(lambda: len(frames) == 1, 2.0)
            assert np.allclose(frames.items[0][0], 0.4) and mic.channels == 2
        finally:
            mic.close()

    def test_full_queue_drops_and_counts(self) -> None:
        sd = FakeSD()
        gate = threading.Event()
        seen = Frames()

        def slow(x: np.ndarray, t: float) -> None:
            gate.wait(2.0)
            seen(x, t)

        mic = MicCapture(backend=sd, max_queue_s=0.05, stall_timeout_s=3600.0)  # 5 blocks
        mic.start(slow)
        try:
            sd.push(np.zeros(480 * 20, np.float32))
            assert mic.stats["dropped"] >= 10
            gate.set()
            wait_until(lambda: len(seen) >= 5, 2.0)
        finally:
            gate.set()
            mic.close()

    def test_frame_handler_errors_are_counted_not_fatal(self) -> None:
        sd = FakeSD()
        calls: list[int] = []

        def bad(x: np.ndarray, t: float) -> None:
            calls.append(1)
            raise RuntimeError("boom")

        mic = MicCapture(backend=sd, stall_timeout_s=3600.0)
        mic.start(bad)
        try:
            sd.push(np.zeros(480 * 3, np.float32))
            wait_until(lambda: mic.stats["frame_error"] == 3, 2.0, what="three errors")
            assert len(calls) == 3
        finally:
            mic.close()

    def test_device_loss_reopens_by_name(self, manual_clock: Any) -> None:
        sd = FakeSD()
        lost: list[int] = []
        frames = Frames()
        mic = MicCapture(
            "USB Mic", backend=sd, clock=manual_clock, on_device_lost=lambda: lost.append(1)
        )
        mic.start(frames)
        try:
            sd.unplug("USB Mic")
            wait_until(lambda: mic.stats["reopen_failed"] >= 1, 2.0, what="failed reopen")
            assert lost == [1] and not mic.device_ok
            sd.replug("USB Mic")
            manual_clock.advance(2.0)
            wait_until(lambda: mic.stats["reopened"] == 1, 2.0, what="reopen")
            sd.push(np.zeros(480, np.float32))
            wait_until(lambda: len(frames) == 1, 2.0)
            assert mic.device_ok and mic.device_index == 3
        finally:
            mic.close()

    def test_missing_device_raises_and_close_is_idempotent(self) -> None:
        mic = MicCapture("nothing", backend=FakeSD())
        with pytest.raises(AudioDeviceError):
            mic.start(lambda x, t: None)
        mic.close()
        mic.close()


def test_die_reopens_player_and_mic_on_the_same_devices(manual_clock: Any) -> None:
    sd = FakeSD()
    frames = Frames()
    player = StreamingPlayer(backend=sd, clock=manual_clock)
    mic = MicCapture(backend=sd, clock=manual_clock)
    player.start()
    mic.start(frames)
    try:
        sd.die()  # every active stream fires finished_callback
        wait_until(lambda: player.stats["reopened"] == 1, 2.0, what="player reopen")
        wait_until(lambda: mic.stats["reopened"] == 1, 2.0, what="mic reopen")
        assert player.stats["device_lost"] == 1 and mic.stats["device_lost"] == 1
        assert [s.device for s in sd.streams if s.active] == [4, 3]
        player.play(np.full(480, 0.25, np.float32), 48000)
        assert np.allclose(sd.pump(1)[0], 0.25)
        sd.push(np.zeros(480, np.float32))
        wait_until(lambda: len(frames) == 1, 2.0, what="mic frame after reopen")
    finally:
        mic.close()
        player.close()
