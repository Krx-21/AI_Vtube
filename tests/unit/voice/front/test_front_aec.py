"""voice.aec: canceller selection, livekit framing, ERLE, energy double-talk and half duplex."""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from aivtube.contracts.voice import EchoCanceller
from aivtube.voice.aec import (
    EnergyDTD,
    HalfDuplexGate,
    LiveKitAEC,
    NullAEC,
    WebRtcAEC,
    make_echo_canceller,
)


class _FakeFrame:
    def __init__(self, data: bytes, sample_rate: int, num_channels: int, samples: int) -> None:
        self._buf = bytearray(data)
        self.sample_rate, self.num_channels, self.samples_per_channel = (
            sample_rate,
            num_channels,
            samples,
        )

    @property
    def data(self) -> memoryview:
        return memoryview(self._buf).cast("h")


class _FakeAPM:
    instances: ClassVar[list[_FakeAPM]] = []

    def __init__(self, **kw: bool) -> None:
        self.kw = kw
        self.calls: list[tuple[str, Any]] = []
        _FakeAPM.instances.append(self)

    def process_reverse_stream(self, frame: _FakeFrame) -> None:
        self.calls.append(("reverse", frame))

    def set_stream_delay_ms(self, ms: int) -> None:
        self.calls.append(("delay", ms))

    def process_stream(self, frame: _FakeFrame) -> None:
        self.calls.append(("process", frame))
        pcm = np.frombuffer(frame._buf, np.int16).copy() // 2  # "cancel" half the signal
        frame._buf[:] = pcm.tobytes()


@pytest.fixture
def fake_livekit(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAPM]:
    _FakeAPM.instances.clear()
    rtc = types.SimpleNamespace(AudioProcessingModule=_FakeAPM, AudioFrame=_FakeFrame)
    pkg = types.ModuleType("livekit")
    pkg.rtc = rtc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "livekit", pkg)
    return _FakeAPM


@pytest.fixture
def no_livekit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "livekit", None)  # "import livekit" raises ImportError


class TestSelection:
    def test_auto_with_headphones_is_none(self, fake_livekit: Any) -> None:
        assert make_echo_canceller("auto", headphones=True) == (None, "none")

    def test_auto_with_speakers_is_aec(self, fake_livekit: Any) -> None:
        aec, mode = make_echo_canceller("auto", headphones=False)
        assert mode == "aec" and isinstance(aec, WebRtcAEC)
        assert fake_livekit.instances[0].kw == dict(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,
        )

    def test_import_error_falls_back_to_energy_dtd(self, no_livekit: None) -> None:
        assert make_echo_canceller("auto", headphones=False) == (None, "energy_dtd")
        assert make_echo_canceller("aec", headphones=True) == (None, "energy_dtd")

    def test_startup_failure_falls_back_to_energy_dtd(
        self, fake_livekit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**kw: Any) -> None:
            raise RuntimeError("ffi down")

        monkeypatch.setattr(sys.modules["livekit"].rtc, "AudioProcessingModule", boom)
        assert make_echo_canceller("aec", headphones=False) == (None, "energy_dtd")

    @pytest.mark.parametrize("mode", ["energy_dtd", "half_duplex", "none"])
    def test_gate_modes_need_no_canceller(self, mode: Any) -> None:
        assert make_echo_canceller(mode, headphones=False) == (None, mode)

    def test_bad_rate_is_rejected(self, fake_livekit: Any) -> None:
        with pytest.raises(ValueError):
            WebRtcAEC(44100)

    def test_protocol_conformance(self, fake_livekit: Any) -> None:
        assert isinstance(WebRtcAEC(), EchoCanceller)
        assert isinstance(NullAEC(), EchoCanceller)
        assert LiveKitAEC is WebRtcAEC


class TestWebRtcFraming:
    def test_frames_are_10ms_int16_and_delay_precedes_each_process(self, fake_livekit: Any) -> None:
        aec = WebRtcAEC(48000, delay_hint_ms=7)
        apm = fake_livekit.instances[0]
        aec.feed_reference(np.full(720, 0.5, np.float32))  # 1.5 frames: one frame + carry
        aec.feed_reference(np.full(240, 0.5, np.float32))  # completes the second frame
        out = aec.process(np.full(960, 0.5, np.float32))  # two mic frames
        kinds = [k for k, _ in apm.calls]
        assert kinds == ["reverse", "reverse", "delay", "process", "delay", "process"]
        for kind, arg in apm.calls:
            if kind == "delay":
                assert arg == 7
            else:
                assert arg.sample_rate == 48000 and arg.num_channels == 1
                assert arg.samples_per_channel == 480 and len(arg._buf) == 480 * 2
                assert np.frombuffer(arg._buf, np.int16).dtype == np.int16
        assert out.dtype == np.float32 and out.shape == (960,)
        assert np.allclose(out, 0.25, atol=1e-3)  # the fake APM halves the frame in place
        assert aec.frames_processed == 2 and aec.reverse_frames == 2

    def test_partial_mic_tail_passes_through(self, fake_livekit: Any) -> None:
        aec = WebRtcAEC(16000)
        out = aec.process(np.full(250, 0.5, np.float32))  # 1 frame (160) + 90 samples
        assert np.allclose(out[:160], 0.25, atol=1e-3) and np.allclose(out[160:], 0.5)
        assert aec.passthrough_samples == 90

    def test_clips_to_int16_range(self, fake_livekit: Any) -> None:
        aec = WebRtcAEC(16000)
        aec.feed_reference(np.full(160, 3.0, np.float32))
        frame = fake_livekit.instances[0].calls[0][1]
        assert np.frombuffer(frame._buf, np.int16).max() == 32767


def test_real_aec3_erle_at_120ms_delay(synth: Any) -> None:
    pytest.importorskip("livekit.rtc")
    pytest.importorskip("soxr")
    far = synth.up48(synth.speech(8.0, seed=3, f0=110.0, amp=0.4))
    rng = np.random.default_rng(1)
    rir = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 300.0)  # 50 ms room tail
    rir *= 0.5 / np.sqrt(np.sum(rir**2))
    delay = round(0.120 * 48000)
    echo = np.convolve(np.concatenate([np.zeros(delay), far]), rir)[: far.size]
    echo = echo.astype(np.float32)
    aec = WebRtcAEC(48000)
    out = np.zeros_like(echo)
    for i in range(0, far.size - 479, 480):
        aec.feed_reference(far[i : i + 480])  # reference first, as the front-end does
        out[i : i + 480] = aec.process(echo[i : i + 480])
    tail = slice(4 * 48000, 8 * 48000)
    erle = 10 * np.log10(np.sum(echo[tail] ** 2) / max(1e-12, float(np.sum(out[tail] ** 2))))
    assert erle >= 15.0, f"ERLE {erle:.1f} dB"


class TestNullAEC:
    def test_passes_through(self) -> None:
        x = np.linspace(-1, 1, 480, dtype=np.float32)
        aec = NullAEC()
        aec.feed_reference(x)
        assert np.array_equal(aec.process(x), x)


class TestEnergyDTD:
    def test_no_reference_passes_the_vad(self) -> None:
        assert EnergyDTD().gate(mic_rms=0.01, ref_rms=0.0, vad_p=0.9) == 0.9

    def test_learns_coupling_and_gates_echo(self) -> None:
        dtd = EnergyDTD(k=2.0)
        # she talks alone: echo at -20 dB of the reference, VAD unsure
        for _ in range(10):
            assert dtd.gate(mic_rms=0.02, ref_rms=0.2, vad_p=0.1) == 0.0
        assert dtd.coupling == pytest.approx(0.1)
        # echo that fools the VAD is still explained by the coupling
        assert dtd.gate(mic_rms=0.03, ref_rms=0.2, vad_p=0.9) == 0.0
        # the streamer is much louder than her echo: pass
        assert dtd.gate(mic_rms=0.2, ref_rms=0.2, vad_p=0.9) == 0.9
        assert dtd.gated == 11

    def test_coupling_decays(self) -> None:
        dtd = EnergyDTD(decay=0.5)
        dtd.gate(0.1, 0.2, 0.0)
        dtd.gate(0.0, 0.2, 0.0)
        assert dtd.coupling == pytest.approx(0.25)
        dtd.reset()
        assert dtd.coupling == 0.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            EnergyDTD(k=0)


def test_half_duplex_gate() -> None:
    g = HalfDuplexGate()
    assert g.gate(ai_speaking=True, vad_p=0.99) == 0.0
    assert g.gate(ai_speaking=False, vad_p=0.99) == 0.99
    assert g.gated == 1
