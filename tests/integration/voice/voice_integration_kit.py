"""Helpers for the voice integration tests: an offline edge-tts stand-in with real MP3 audio.

``ScriptedEdge`` replaces ``edge_tts.Communicate``: for each text it encodes a voiced
synthetic signal lasting ``len(text) / 12.5`` seconds to real MPEG Layer III frames with
PyAV's libmp3lame, and streams them in ~720-byte messages after WordBoundary metadata, like
the service does. No network, no voice data (license-clean).
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

SR = 24000
CPS = 12.5


def voiced(seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(max(1, int(seconds * sr))) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    x = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in (1, 2, 3, 5))
    return 0.4 * env * x / 1.6


@functools.lru_cache(maxsize=64)
def mp3_for(seconds: float) -> bytes:
    import av

    cc = av.CodecContext.create("libmp3lame", "w")
    cc.sample_rate = SR
    cc.layout = "mono"
    cc.format = "s32p"
    cc.bit_rate = 48000
    cc.open()
    x = voiced(seconds)
    fs = cc.frame_size
    out = b""
    for i in range(0, len(x), fs):
        blk = np.asarray(x[i : i + fs], np.float64)
        if blk.size < fs:
            blk = np.pad(blk, (0, fs - blk.size))
        arr = (np.clip(blk, -1, 1) * 2147483000).astype(np.int32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s32p", layout="mono")
        frame.sample_rate = SR
        frame.pts = i
        for pkt in cc.encode(frame):
            out += bytes(pkt)
    for pkt in cc.encode(None):
        out += bytes(pkt)
    return out


class ScriptedEdge:
    """``communicate_factory`` for ``EdgeTTSBackend``; records every request."""

    def __init__(self, sleep: Callable[[float], Any] = asyncio.sleep, ttfa_s: float = 0.2) -> None:
        self.sleep = sleep
        self.ttfa_s = ttfa_s
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, text: str, voice: str, **kwargs: Any) -> Any:
        self.requests.append((text, voice, kwargs))
        edge = self

        class _Communicate:
            def stream(self) -> AsyncIterator[dict[str, Any]]:
                return edge._stream(text)

        return _Communicate()

    async def _stream(self, text: str) -> AsyncIterator[dict[str, Any]]:
        mp3 = mp3_for(round(len(text) / CPS, 3))
        pos = 0
        for word in text.split(" "):
            if word:
                yield {
                    "type": "WordBoundary",
                    "offset": int(pos / CPS * 1e7),
                    "duration": int(len(word) / CPS * 1e7),
                    "text": word,
                }
            pos += len(word) + 1
        await self.sleep(self.ttfa_s)
        for i in range(0, len(mp3), 720):
            yield {"type": "audio", "data": mp3[i : i + 720]}
