"""Nightly checks against real local models ($AIVTUBE_TEST_ASSETS); skipped by default."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.nightly
def test_committed_silero_is_the_real_model(test_assets: Path, silero_onnx: Path) -> None:
    asset = test_assets / "silero_vad.onnx"
    if not asset.is_file():
        pytest.skip("no silero_vad.onnx in the assets")
    assert asset.read_bytes() == silero_onnx.read_bytes()


@pytest.mark.nightly
def test_typhoon_rt_export_passes_the_verifier(test_assets: Path) -> None:
    pytest.importorskip("sherpa_onnx")
    model_dir = test_assets / "typhoon-rt"
    wav = test_assets / "audio" / "t2.wav"
    texts = test_assets / "audio" / "texts.txt"
    if not (model_dir.is_dir() and wav.is_file() and texts.is_file()):
        pytest.skip("Typhoon RT export or Thai clips missing from the assets")
    reference = texts.read_text(encoding="utf-8").splitlines()[1]
    spec = importlib.util.spec_from_file_location(
        "verify_rt", ROOT / "tools" / "verify_typhoon_rt_onnx.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = [str(model_dir), "--wav", str(wav), "--text", reference, "--max-cer", "0.2"]
    assert mod.main(args) == 0
