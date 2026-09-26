"""``aivtube.memory`` and ``aivtube.tools`` import without native or heavy dependencies."""

from __future__ import annotations

import subprocess
import sys


def test_memory_and_tools_import_light() -> None:
    heavy = (
        "sounddevice",
        "sherpa_onnx",
        "onnxruntime",
        "av",
        "edge_tts",
        "azure",
        "livekit",
        "soxr",
        "numpy",
        "pythainlp",
    )
    code = (
        "import sys, aivtube.memory, aivtube.tools; "
        f"print(sorted(m for m in {heavy!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "[]"


def test_migrations_are_packaged() -> None:
    from importlib import resources

    names = {p.name for p in (resources.files("aivtube.memory") / "migrations").iterdir()}
    assert {"memory_0001_initial.sql", "memory_fts_trigram.sql", "ops_0001_initial.sql"} <= names
