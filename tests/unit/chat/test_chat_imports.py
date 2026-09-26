"""Importing ``aivtube.chat`` stays light: no native audio/ML stacks, httpx2 only when polling."""

from __future__ import annotations

import subprocess
import sys


def test_chat_import_pulls_no_heavy_or_lazy_deps() -> None:
    lazy = (
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
        "httpx2",
        "aiohttp",
    )
    code = (
        "import sys, aivtube.chat; "
        "from aivtube.chat import ScoredChatWindow, TwitchAnonIrc, YouTubeListPoller; "
        f"print(sorted(m for m in {lazy!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "[]"
