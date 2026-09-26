"""Nightly: the real Typhoon ASR Realtime model (sherpa-onnx) on the golden Thai clips.

Needs ``$AIVTUBE_TEST_ASSETS`` with ``typhoon-rt/`` (encoder/decoder/joiner int8 + tokens) and
``audio/t1..t3.wav`` + ``audio/texts.txt``. PyThaiASR runs too when
``$AIVTUBE_PYTHAIASR_DIR`` holds its ONNX files. The clips are 16 kHz synthetic Thai speech;
t1 contains the name ไพลิน, which the model hears as ไทลิน (fixed by the alias map).
"""

from __future__ import annotations

import os
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.testing.contracts import case_id, speech_recognizer_suite
from aivtube.voice.stt import (
    NamePostProcessor,
    PyThaiAsrRecognizer,
    SherpaTyphoonRT,
    SttRunner,
)

pytestmark = pytest.mark.nightly

ALIASES = {"ไพลิน": ["ไทลิน", "ไทยลิน", "ไทลิล", "ไภลิน"]}
CER_MAX = 0.30  # per clip (English words come out in Thai script: "Minecraft" -> "เหมือนคราบ")
CER_MEAN_MAX = 0.15


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0


def cer(ref: str, hyp: str) -> float:
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[:], i
        for j, b in enumerate(h, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a != b))
    return d[len(h)] / max(1, len(r))


@pytest.fixture(scope="module")
def clips(test_assets: Path) -> list[tuple[np.ndarray, str]]:
    texts = (test_assets / "audio" / "texts.txt").read_text(encoding="utf-8").splitlines()
    return [(read_wav(test_assets / "audio" / f"t{i}.wav"), texts[i - 1]) for i in (1, 2, 3)]


@pytest.fixture(scope="module")
def typhoon_rt(test_assets: Path) -> Iterator[SherpaTyphoonRT]:
    pytest.importorskip("sherpa_onnx")
    rec = SherpaTyphoonRT(test_assets / "typhoon-rt", num_threads=2)
    rec.warmup()
    yield rec
    rec.close()


def test_typhoon_rt_cer_is_bounded_and_the_alias_map_recovers_the_name(
    typhoon_rt: SherpaTyphoonRT, clips: list[tuple[np.ndarray, str]]
) -> None:
    post = NamePostProcessor(ALIASES)
    scores = []
    for i, (audio, ref) in enumerate(clips, 1):
        raw = typhoon_rt.transcribe(audio)
        assert raw.engine == "typhoon_rt" and raw.audio_s == pytest.approx(audio.size / 16000)
        fixed = post(raw, "")
        assert fixed is not None
        score = cer(ref, fixed.text)
        scores.append(score)
        assert score <= CER_MAX, f"t{i}: CER {score:.3f} for {fixed.text!r} (ref {ref!r})"
        if i == 1:
            assert "ไพลิน" not in raw.text  # the model mishears the name ...
            assert "ไพลิน" in fixed.text  # ... and the alias map recovers it
    assert sum(scores) / len(scores) <= CER_MEAN_MAX


async def test_runner_with_the_real_model_and_a_quick_decode(
    typhoon_rt: SherpaTyphoonRT, clips: list[tuple[np.ndarray, str]]
) -> None:
    runner = SttRunner([_Shared(typhoon_rt)], NamePostProcessor(ALIASES))
    try:
        t = await runner.transcribe(clips[0][0])
        assert t is not None and "ไพลิน" in t.text
        quick = await runner.transcribe(clips[1][0][: int(1.5 * 16000)], quick=True)
        assert quick is not None and quick.text  # "ไพลินช่วย…" from the first 1.5 s
        assert quick.latency_ms < 2000
    finally:
        runner.close()


class _Shared:
    """Delegates to the module's recognizer but never closes it (the suite closes)."""

    def __init__(self, rec: SherpaTyphoonRT) -> None:
        self._rec = rec
        self.name = rec.name
        self.sample_rate = rec.sample_rate

    def warmup(self) -> None:
        self._rec.warmup()

    def transcribe(self, pcm16k: Any, *, quick: bool = False) -> Any:
        return self._rec.transcribe(pcm16k, quick=quick)

    def close(self) -> None:
        pass


def test_real_model_passes_the_speech_recognizer_suite(
    typhoon_rt: SherpaTyphoonRT, clips: list[tuple[np.ndarray, str]]
) -> None:
    cases = speech_recognizer_suite(
        lambda: _Shared(typhoon_rt), sample=clips[1][0], expected="คอมเมนต์"
    )
    for case in cases:
        try:
            case()
        except Exception as exc:  # pragma: no cover - reported with the case name
            raise AssertionError(f"{case_id(case)}: {exc}") from exc


def test_pythaiasr_fallback_on_the_same_clips(clips: list[tuple[np.ndarray, str]]) -> None:
    model_dir = os.environ.get("AIVTUBE_PYTHAIASR_DIR", "").strip()
    if not model_dir or not Path(model_dir).is_dir():
        pytest.skip("set AIVTUBE_PYTHAIASR_DIR to PyThaiASR's typhoon-asr-realtime directory")
    pytest.importorskip("pythaiasr")
    rec = PyThaiAsrRecognizer(model_dir=model_dir)
    try:
        rec.warmup()
        audio, ref = clips[1]
        t = rec.transcribe(audio)
        assert cer(ref, t.text) <= CER_MAX, t.text
    finally:
        rec.close()
