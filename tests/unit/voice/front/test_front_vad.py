"""voice.vad: Silero on onnxruntime and sherpa-onnx, the energy detector, make_vad."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.contracts.voice import VoiceActivityDetector
from aivtube.testing.contracts import case_id, vad_suite
from aivtube.voice.vad import (
    EnergyVAD,
    ModelChecksumError,
    SherpaSileroVAD,
    SileroOrtVAD,
    make_vad,
    sha256_file,
)

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "silero_vad.onnx"
PINNED = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
_SPEECH: Any = None


def _speech() -> np.ndarray:
    global _SPEECH
    if _SPEECH is None:
        from voicefront_signals import Synth

        _SPEECH = Synth().speech(2.0, seed=0)
    return np.asarray(_SPEECH)


def frames(x: np.ndarray) -> list[np.ndarray]:
    return [x[i : i + 512] for i in range(0, x.size - 511, 512)]


def _ort() -> SileroOrtVAD:
    pytest.importorskip("onnxruntime")
    return SileroOrtVAD(FIXTURE)


def _sherpa() -> SherpaSileroVAD:
    pytest.importorskip("sherpa_onnx")
    return SherpaSileroVAD(FIXTURE)


@pytest.mark.parametrize("case", vad_suite(_ort, speech=_speech()), ids=case_id)
def test_silero_ort_contract(case: Any) -> None:
    case()


@pytest.mark.parametrize("case", vad_suite(_sherpa, speech=_speech()), ids=case_id)
def test_silero_sherpa_contract(case: Any) -> None:
    case()


@pytest.mark.parametrize("case", vad_suite(EnergyVAD, speech=_speech()), ids=case_id)
def test_energy_vad_contract(case: Any) -> None:
    case()


class TestSileroOrt:
    def test_fixture_hash_is_pinned(self) -> None:
        assert sha256_file(FIXTURE) == PINNED

    def test_matches_reference_with_64_sample_context_and_state(self, synth: Any) -> None:
        ort = pytest.importorskip("onnxruntime")
        vad = SileroOrtVAD(FIXTURE)
        sess = ort.InferenceSession(str(FIXTURE), providers=["CPUExecutionProvider"])
        state = np.zeros((2, 1, 128), np.float32)
        ctx = np.zeros((1, 64), np.float32)
        x = np.concatenate([synth.silence(0.3), synth.speech(1.0, seed=1), synth.tone(0.3)])
        for f in frames(x):
            inp = np.concatenate([ctx, f.reshape(1, -1)], axis=1)
            assert inp.shape == (1, 576)
            out, state = sess.run(
                None, {"input": inp, "state": state, "sr": np.array(16000, np.int64)}
            )
            ctx = inp[:, -64:]
            assert vad.prob(f) == pytest.approx(float(out[0, 0]), abs=1e-5)
            assert vad._state.shape == (2, 1, 128)

    def test_speech_scores_high_and_tones_and_silence_low(self, synth: Any) -> None:
        vad = _ort()
        speech = [vad.prob(f) for f in frames(synth.speech(3.0, seed=2))]
        assert np.mean(np.array(speech[3:]) > 0.5) > 0.95
        vad.reset()
        assert max(vad.prob(f) for f in frames(synth.tone(1.0, 440.0))) < 0.1
        vad.reset()
        assert max(vad.prob(f) for f in frames(synth.noise(1.0))) < 0.2

    def test_reset_clears_the_recurrent_state(self, synth: Any) -> None:
        vad = _ort()
        x = frames(synth.speech(1.0, seed=4))
        first = [vad.prob(f) for f in x]
        vad.reset()
        again = [vad.prob(f) for f in x]
        assert np.allclose(first, again, atol=1e-6)

    def test_rejects_wrong_frame_size_and_sanitises_nan(self) -> None:
        vad = _ort()
        with pytest.raises(ValueError):
            vad.prob(np.zeros(480, np.float32))
        p = vad.prob(np.full(512, np.nan, np.float32))
        assert 0.0 <= p <= 1.0
        assert np.isfinite(vad._state).all()

    def test_rejects_a_model_that_is_not_silero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ort = pytest.importorskip("onnxruntime")

        class OtherModel:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def get_inputs(self) -> list[Any]:
                return [type("I", (), {"name": "pixel_values"})()]

        monkeypatch.setattr(ort, "InferenceSession", OtherModel)
        with pytest.raises(ValueError, match="not a Silero"):
            SileroOrtVAD(FIXTURE)

    def test_is_a_voice_activity_detector(self) -> None:
        assert isinstance(_ort(), VoiceActivityDetector)
        assert isinstance(EnergyVAD(), VoiceActivityDetector)


class TestSherpa:
    def test_agrees_with_onnxruntime_in_one_process(self, synth: Any) -> None:
        """ORT coexistence (spike S3): both runtimes load the same model in one process."""
        ort_vad = _ort()
        sherpa_vad = _sherpa()
        x = np.concatenate([synth.silence(1.0), synth.speech(2.0, seed=0), synth.silence(1.0)])
        a = np.array([ort_vad.prob(f) > 0.5 for f in frames(x)])
        b = np.array([sherpa_vad.prob(f) > 0.5 for f in frames(x)])
        assert b.sum() > 40 and np.mean(a == b) > 0.9
        onset_a, onset_b = int(np.argmax(a)), int(np.argmax(b))
        assert 0 <= onset_b - onset_a <= 3  # sherpa lags by a frame or two
        sherpa_vad.reset()
        assert sherpa_vad.prob(np.zeros(512, np.float32)) == 0.0


class TestEnergy:
    def test_maps_dbfs_around_the_threshold(self) -> None:
        vad = EnergyVAD(-42.0, width_db=12.0)
        amp_at = 10 ** (-42 / 20) * np.sqrt(2)  # a sine at -42 dBFS RMS
        t = np.arange(512) / 16000
        assert vad.prob((amp_at * np.sin(2 * np.pi * 300 * t)).astype(np.float32)) == (
            pytest.approx(0.5, abs=0.01)
        )
        assert vad.prob(np.zeros(512, np.float32)) == 0.0
        assert vad.prob(np.full(512, 0.5, np.float32)) == 1.0
        with pytest.raises(ValueError):
            EnergyVAD(width_db=0)


class TestMakeVad:
    def test_builds_each_backend(self) -> None:
        pytest.importorskip("onnxruntime")
        assert isinstance(make_vad("silero_ort", FIXTURE, PINNED), SileroOrtVAD)
        assert isinstance(make_vad("energy", Path("unused"), ""), EnergyVAD)
        if pytest.importorskip("sherpa_onnx"):
            assert isinstance(make_vad("silero_sherpa", FIXTURE, PINNED), SherpaSileroVAD)

    def test_raises_on_sha256_mismatch(self, tmp_path: Path) -> None:
        bad = tmp_path / "silero_vad.onnx"
        bad.write_bytes(FIXTURE.read_bytes()[:-1] + b"\0")
        with pytest.raises(ModelChecksumError):
            make_vad("silero_ort", bad, PINNED)
        with pytest.raises(ModelChecksumError):
            make_vad("silero_ort", FIXTURE, hashlib.sha256(b"other").hexdigest())

    def test_empty_hash_skips_the_check_and_missing_file_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        assert isinstance(make_vad("silero_ort", FIXTURE, ""), SileroOrtVAD)
        with pytest.raises(FileNotFoundError):
            make_vad("silero_ort", tmp_path / "nope.onnx", PINNED)

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError):
            make_vad("webrtc", FIXTURE, "")  # type: ignore[arg-type]
