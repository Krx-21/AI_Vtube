"""Nightly: real Thai speech through the front-end and the committed Silero model.

Needs ``AIVTUBE_TEST_ASSETS`` with ``audio/t1.wav`` .. ``t3.wav`` (16 kHz mono Thai speech; never
committed, see the assets README).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.contracts.voice import EndpointerConfig, VadEnd, VadEvent, VadPartial, VadStart
from aivtube.voice.endpointer import SileroEndpointer
from aivtube.voice.frontend import VoiceFrontEnd

pytestmark = pytest.mark.nightly


def _clip(test_assets: Path, name: str) -> np.ndarray:
    sf = pytest.importorskip("soundfile")
    path = test_assets / "audio" / name
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == 16000
    return np.asarray(x, dtype=np.float32)


def _run(x16: np.ndarray, silero_path: Path) -> list[VadEvent]:
    pytest.importorskip("onnxruntime")
    soxr = pytest.importorskip("soxr")
    from aivtube.voice.vad import SileroOrtVAD

    events: list[VadEvent] = []
    ep = SileroEndpointer(SileroOrtVAD(silero_path), EndpointerConfig())
    fe = VoiceFrontEnd(ep, player=None, aec=None, dtd=None, on_event=events.append)
    x48 = np.asarray(soxr.resample(x16, 16000, 48000), dtype=np.float32)
    for i in range(0, x48.size - 479, 480):  # 10 ms mic blocks
        fe.feed(x48[i : i + 480], 100.0 + i / 48000)
    return events


@pytest.mark.parametrize("name", ["t1.wav", "t2.wav", "t3.wav"])
def test_real_thai_speech_is_exactly_one_utterance(
    name: str, test_assets: Path, silero_path: Path, synth: Any
) -> None:
    speech = _clip(test_assets, name)
    x = np.concatenate([synth.silence(1.0), speech, synth.silence(1.5)])
    events = _run(x, silero_path)
    assert [type(e).__name__ for e in events] == ["VadStart", "VadEnd"], events
    start, end = events
    assert isinstance(start, VadStart) and isinstance(end, VadEnd)
    assert not any(isinstance(e, VadPartial) for e in events)
    first, last = _voiced_span(speech)
    assert start.t - 100.0 == pytest.approx(1.0 + first, abs=0.15)
    assert end.t - 100.0 == pytest.approx(1.0 + last, abs=0.25)
    # pre-roll 0.3 s + the voiced span + tail pad 0.2 s (Silero's hangover adds a little)
    assert end.audio.size / 16000 == pytest.approx(0.3 + (last - first) + 0.2, abs=0.3)


def _voiced_span(x: np.ndarray) -> tuple[float, float]:
    """First and last 10 ms frame above -45 dBFS (the clips have silent lead-in and lead-out)."""
    frames = x[: x.size // 160 * 160].reshape(-1, 160)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    voiced = np.flatnonzero(db > -45.0)
    return voiced[0] * 0.01, (voiced[-1] + 1) * 0.01
