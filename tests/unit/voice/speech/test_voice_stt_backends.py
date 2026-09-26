"""Real recognizers with injected fakes (sherpa, pythaiasr, HTTP), plus the STT factory.

The recognizers run through ``speech_recognizer_suite`` here; the real models run in the
nightly integration tests.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import numpy as np
import pytest

from aivtube.contracts.voice import SpeechRecognizer
from aivtube.testing.contracts import case_id, speech_recognizer_suite
from aivtube.testing.fakes import FakeRecognizer
from aivtube.voice.stt import (
    PyThaiAsrRecognizer,
    SherpaTyphoonRT,
    TyphoonApiRecognizer,
    build_stt_chain,
    find_transducer_files,
    wav_bytes,
)

# --- sherpa -----------------------------------------------------------------------------------


class _FakeStream:
    def __init__(self) -> None:
        self.result = SimpleNamespace(text="")
        self.samples = np.zeros(0, np.float32)
        self.sr = 0

    def accept_waveform(self, sr: int, x: np.ndarray) -> None:
        self.sr, self.samples = sr, x


class _FakeSherpa:
    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        self.decoded: list[int] = []

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def decode_stream(self, s: _FakeStream) -> None:
        assert s.sr == 16000 and s.samples.dtype == np.float32 and s.samples.flags.c_contiguous
        self.decoded.append(int(s.samples.size))
        s.result.text = "  สวัสดีครับ  "


def _model_dir(tmp_path: Path, int8: bool = True) -> Path:
    d = tmp_path / "typhoon-rt"
    d.mkdir(exist_ok=True)
    suffix = ".int8.onnx" if int8 else ".onnx"
    for part in ("encoder", "decoder", "joiner"):
        (d / f"{part}{suffix}").write_bytes(b"")
    (d / "tokens.txt").write_text("<blk> 0\n", encoding="utf-8")
    return d


def _sherpa(tmp_path: Path) -> tuple[SherpaTyphoonRT, list[_FakeSherpa]]:
    made: list[_FakeSherpa] = []

    def factory(**kw: Any) -> _FakeSherpa:
        made.append(_FakeSherpa(**kw))
        return made[-1]

    return SherpaTyphoonRT(_model_dir(tmp_path), num_threads=3, factory=factory), made


def test_sherpa_builds_the_nemo_transducer_with_the_documented_options(tmp_path: Path) -> None:
    rec, made = _sherpa(tmp_path)
    kw = made[0].kw
    assert kw["model_type"] == "nemo_transducer" and kw["feature_dim"] == 80
    assert kw["sample_rate"] == 16000 and kw["decoding_method"] == "greedy_search"
    assert kw["num_threads"] == 3 and kw["provider"] == "cpu"
    assert kw["encoder"].endswith("encoder.int8.onnx")
    assert isinstance(rec, SpeechRecognizer)


def test_sherpa_transcribes_pads_short_audio_and_trims_quick_decodes(tmp_path: Path) -> None:
    rec, made = _sherpa(tmp_path)
    t = rec.transcribe(np.zeros(32000, np.float32))
    assert t.text == "สวัสดีครับ" and t.audio_s == pytest.approx(2.0) and t.engine == "typhoon_rt"
    assert made[0].decoded[-1] == 32000
    t = rec.transcribe(np.zeros(1600, np.float32))  # 0.1 s is padded for the model
    assert t.audio_s == pytest.approx(0.1) and made[0].decoded[-1] == int(0.4 * 16000)
    rec.transcribe(np.zeros(5 * 16000, np.float32), quick=True)
    assert made[0].decoded[-1] == int(1.2 * 16000)
    t = rec.transcribe((np.ones(16000) * 16384).astype(np.int16))  # int16 accepted
    assert t.audio_s == pytest.approx(1.0)
    rec.warmup()
    assert rec.warmed
    rec.close()
    with pytest.raises(RuntimeError):
        rec.transcribe(np.zeros(16000, np.float32))


def test_find_transducer_files_prefers_int8_and_reports_missing(tmp_path: Path) -> None:
    d = _model_dir(tmp_path, int8=False)
    files = find_transducer_files(d)
    assert files["joiner"].name == "joiner.onnx"
    (d / "encoder.int8.onnx").write_bytes(b"")
    assert find_transducer_files(d)["encoder"].name == "encoder.int8.onnx"
    (d / "tokens.txt").unlink()
    with pytest.raises(FileNotFoundError):
        find_transducer_files(d)
    with pytest.raises(FileNotFoundError):
        find_transducer_files(tmp_path / "nope")


# --- pythaiasr --------------------------------------------------------------------------------


class _FakePyThaiModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls.append((int(audio.size), sample_rate))
        return " ทดสอบ "


def _pythai() -> tuple[PyThaiAsrRecognizer, list[Any]]:
    loads: list[Any] = []

    def loader(model_dir: Path | None, device: str) -> _FakePyThaiModel:
        loads.append((model_dir, device))
        return _FakePyThaiModel()

    return PyThaiAsrRecognizer(loader=loader), loads


def test_pythaiasr_loads_lazily_and_transcribes() -> None:
    rec, loads = _pythai()
    assert loads == []  # nothing is loaded (or downloaded) at construction
    t = rec.transcribe(np.zeros(8000, np.float32))
    assert t.text == "ทดสอบ" and t.engine == "pythaiasr" and t.audio_s == pytest.approx(0.5)
    assert loads == [(None, "cpu")]
    rec.warmup()
    assert len(loads) == 1
    rec.close()
    with pytest.raises(RuntimeError):
        rec.transcribe(np.zeros(8000, np.float32))


# --- Typhoon API ------------------------------------------------------------------------------


def _api(handler: Any) -> TyphoonApiRecognizer:
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return TyphoonApiRecognizer(
        "https://api.opentyphoon.ai/v1/", "typhoon-asr-realtime", "sk-test", client=client
    )


def test_typhoon_api_posts_wav_multipart_with_auth() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"text": " สวัสดี "})

    rec = _api(handler)
    t = rec.transcribe(np.zeros(16000, np.float32))
    assert t.text == "สวัสดี" and t.engine == "typhoon_api" and t.audio_s == pytest.approx(1.0)
    req = seen[0]
    assert req.url.path == "/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = req.read()
    assert b"typhoon-asr-realtime" in body and b"RIFF" in body and b"audio/wav" in body
    rec.close()


def test_typhoon_api_errors_raise_runtime_error_with_detail() -> None:
    rec = _api(lambda r: httpx2.Response(401, json={"detail": "invalid api key"}))
    with pytest.raises(RuntimeError, match="invalid api key"):
        rec.transcribe(np.zeros(16000, np.float32))

    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline", request=request)

    with pytest.raises(RuntimeError, match="ConnectError"):
        _api(boom).transcribe(np.zeros(16000, np.float32))
    with pytest.raises(ValueError):
        TyphoonApiRecognizer("https://x/v1", "m", "")


def test_wav_bytes_is_16k_mono_16bit() -> None:
    data = wav_bytes(np.full(1600, 2.0, np.float32))  # clipped
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (
            1,
            2,
            16000,
            1600,
        )
        frames = np.frombuffer(w.readframes(1600), "<i2")
    assert frames.max() == 32767


# --- contract suite against the real classes --------------------------------------------------


def _suite_factories(tmp_path: Path) -> list[Any]:
    return [
        lambda: _sherpa(tmp_path)[0],
        lambda: _pythai()[0],
        lambda: _api(lambda r: httpx2.Response(200, json={"text": "ok"})),
    ]


@pytest.mark.parametrize("which", [0, 1, 2], ids=["sherpa", "pythaiasr", "typhoon_api"])
def test_recognizers_pass_the_speech_recognizer_suite(tmp_path: Path, which: int) -> None:
    factory = _suite_factories(tmp_path)[which]
    cases = speech_recognizer_suite(factory)
    assert cases
    for case in cases:
        try:
            case()
        except Exception as exc:  # pragma: no cover - reported with the case name
            raise AssertionError(f"{case_id(case)} failed: {exc}") from exc


# --- factory ----------------------------------------------------------------------------------

BACKENDS: dict[str, dict[str, Any]] = {
    "typhoon_rt": {
        "kind": "sherpa_nemo_transducer",
        "model_dir": "models/stt/rt",
        "num_threads": 2,
    },
    "pythaiasr": {"kind": "pythaiasr"},
    "typhoon_api": {
        "kind": "openai_audio",
        "cloud": True,
        "base_url": "https://api.opentyphoon.ai/v1",
        "model": "typhoon-asr-realtime",
        "api_key_env": "TYPHOON_API_KEY",
    },
    "whisper_gpu": {"kind": "faster_whisper_worker", "enabled": False},
}


def test_factory_skips_cloud_without_consent_and_missing_models(tmp_path: Path) -> None:
    keys = {"TYPHOON_API_KEY": "sk-1"}
    chain = build_stt_chain(
        ["typhoon_rt", "pythaiasr", "typhoon_api", "whisper_gpu", "unknown"],
        BACKENDS,
        root=tmp_path,
        secrets=keys.get,
        cloud_consent=False,
    )
    # typhoon_rt: model files missing -> logged and skipped; cloud without consent -> skipped
    assert [r.name for r in chain] == ["pythaiasr"]
    with_consent = build_stt_chain(
        ["pythaiasr", "typhoon_api"], BACKENDS, root=tmp_path, secrets=keys.get, cloud_consent=True
    )
    assert [type(r).__name__ for r in with_consent] == [
        "PyThaiAsrRecognizer",
        "TyphoonApiRecognizer",
    ]
    for r in with_consent:
        r.close()
    no_key = build_stt_chain(
        ["typhoon_api"], BACKENDS, root=tmp_path, secrets=lambda _k: None, cloud_consent=True
    )
    assert no_key == []


def test_factory_builds_sherpa_from_a_relative_model_dir_and_inline_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model_dir(tmp_path)
    import aivtube.voice.stt.sherpa as sherpa_mod

    built: list[dict[str, Any]] = []

    def fake_init(self: SherpaTyphoonRT, model_dir: Path, **kw: Any) -> None:
        built.append({"model_dir": model_dir, **kw})
        self.name = kw["name"]

    monkeypatch.setattr(sherpa_mod.SherpaTyphoonRT, "__init__", fake_init)
    chain = build_stt_chain(
        [
            {"name": "rt", "kind": "sherpa_nemo_transducer", "model_dir": "typhoon-rt"},
            {"kind": "fake", "script": ["x"]},
        ],
        {},
        root=tmp_path,
        secrets=lambda _k: None,
    )
    assert built[0]["model_dir"] == tmp_path / "typhoon-rt" and built[0]["name"] == "rt"
    assert isinstance(chain[1], FakeRecognizer)
