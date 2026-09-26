"""The IPC layer, the core's speech output and the worker entry import no native stacks."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("sounddevice", "onnxruntime", "sherpa_onnx", "av", "edge_tts", "livekit", "soxr")


def test_ipc_speech_and_worker_modules_import_without_native_audio_stacks() -> None:
    code = (
        "import sys\n"
        "import aivtube.ipc, aivtube.speech, aivtube.voice.worker\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "", out.stdout


def test_the_core_side_does_not_import_the_voice_worker() -> None:
    code = (
        "import sys\n"
        "import aivtube.speech\n"
        "print(','.join(m for m in sys.modules if m.startswith('aivtube.voice')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "", out.stdout
