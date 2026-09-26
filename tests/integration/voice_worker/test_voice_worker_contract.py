"""``BusSpeechOutput`` + the real voice worker pass the ``SpeechOutput`` contract suite (§10)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from worker_kit import Stack

from aivtube.testing.contracts import SpeechOutputHarness, case_id, speech_output_suite

_TMP: list[Path] = []


async def _harness() -> SpeechOutputHarness:
    stack = await Stack(_TMP[-1], tts_cps=40.0).start()
    return SpeechOutputHarness(output=stack.out, bus=stack.bus, aclose=stack.aclose)


@pytest.mark.parametrize("case", speech_output_suite(_harness, within=20.0), ids=case_id)
async def test_bus_speech_output_contract(case: object, tmp_path: Path) -> None:
    _TMP.append(tmp_path)
    await case()  # type: ignore[operator]
