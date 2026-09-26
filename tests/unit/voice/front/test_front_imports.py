"""Import hygiene and cross-package seams of the voice front-end."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys

import pytest

import aivtube.voice as voice_pkg
from aivtube.voice.barge import QuickTranscriber

FRONT_MODULES = (
    "aivtube.voice",
    "aivtube.voice.audio_io",
    "aivtube.voice.aec",
    "aivtube.voice.vad",
    "aivtube.voice.endpointer",
    "aivtube.voice.frontend",
    "aivtube.voice.barge",
)
NATIVE = ("sounddevice", "soxr", "onnxruntime", "sherpa_onnx", "livekit", "av", "edge_tts", "torch")


def test_front_end_imports_no_native_audio_stack() -> None:
    """The worker must import (and CI must run) without PortAudio, soxr, ORT or livekit."""
    code = (
        "import importlib, sys\n"
        f"for m in {FRONT_MODULES!r}: importlib.import_module(m)\n"
        f"print(','.join(m for m in {NATIVE!r} if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"imported at module level: {out.stdout.strip()}"


def test_package_reexports_lazily() -> None:
    code = (
        "import sys, aivtube.voice as v\n"
        "assert 'aivtube.voice.audio_io' not in sys.modules\n"
        "v.StreamingPlayer\n"
        "assert 'aivtube.voice.audio_io' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    for name in voice_pkg.__all__:
        assert getattr(voice_pkg, name) is not None
    assert set(voice_pkg.__all__) <= set(dir(voice_pkg))
    with pytest.raises(AttributeError):
        _ = voice_pkg.NoSuchThing


def test_stt_runner_matches_the_quick_transcriber_seam() -> None:
    """``BargeInController`` calls ``SttRunner.transcribe(pcm, quick=True, recent_tts_text=...)``."""
    if importlib.util.find_spec("aivtube.voice.stt.runner") is None:
        pytest.skip("voice.stt is not built yet")
    from aivtube.voice.stt.runner import SttRunner

    ours = inspect.signature(QuickTranscriber.transcribe)
    theirs = inspect.signature(SttRunner.transcribe)
    assert inspect.iscoroutinefunction(SttRunner.transcribe)
    for name in ("pcm16k", "quick", "recent_tts_text"):
        assert name in theirs.parameters, f"SttRunner.transcribe lacks {name}"
        assert theirs.parameters[name].kind == ours.parameters[name].kind
    assert theirs.parameters["quick"].default is False
    assert theirs.parameters["recent_tts_text"].default == ""
