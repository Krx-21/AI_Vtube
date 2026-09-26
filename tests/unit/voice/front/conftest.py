"""Shared fixtures for the voice front-end tests (audio I/O, AEC, VAD, endpointer, barge-in).

Nothing here touches a real audio device: ``FakeSD`` stands in for PortAudio, and signals are
synthetic (see ``voicefront_signals``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from voicefront_signals import ManualClock, Synth


@pytest.fixture(scope="session")
def synth() -> Synth:
    return Synth()


@pytest.fixture
def manual_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture(scope="session")
def silero_path(silero_onnx: Path) -> Path:
    if not silero_onnx.is_file():
        pytest.skip("tests/fixtures/silero_vad.onnx is missing")
    return silero_onnx
