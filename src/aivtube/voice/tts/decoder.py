"""Stateful streaming MP3 → int16 PCM decoder on PyAV (§4.6 "Decoding", TTS brief §2–3).

edge-tts delivers ``audio-24khz-48kbitrate-mono-mp3`` as ~720-byte WebSocket messages (five
MPEG frames). MP3 frames borrow bits from earlier frames (the bit reservoir), so decoding each
message on its own loses or corrupts samples. One ``Mp3StreamDecoder`` per synthesis keeps the
parser and decoder state across messages; the output is sample-exact against decoding the
whole buffer at once. ``av`` (PyAV, bundles FFmpeg) is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aivtube.contracts.voice import I16

__all__ = ["Mp3StreamDecoder"]


class Mp3StreamDecoder:
    """Feed MP3 bytes in any chunking; get mono int16 PCM as soon as frames complete."""

    def __init__(self, codec: str = "mp3") -> None:
        import av

        self._ctx: Any = av.CodecContext.create(codec, "r")
        self.sample_rate: int | None = None
        self.samples = 0
        self._closed = False

    def feed(self, data: bytes) -> I16:
        """Decode everything that ``data`` completes; returns possibly empty int16 PCM."""
        if self._closed:
            raise RuntimeError("decoder is closed")
        if not data:
            return np.zeros(0, np.int16)
        out: list[I16] = []
        for packet in self._ctx.parse(data):
            for frame in self._ctx.decode(packet):
                out.append(self._pcm(frame))
        return self._join(out)

    def flush(self) -> I16:
        """Drain the parser and the decoder at the end of the stream."""
        if self._closed:
            return np.zeros(0, np.int16)
        out: list[I16] = []
        for packet in self._ctx.parse(None):
            for frame in self._ctx.decode(packet):
                out.append(self._pcm(frame))
        for frame in self._ctx.decode(None):
            out.append(self._pcm(frame))
        self._closed = True
        return self._join(out)

    def _pcm(self, frame: Any) -> I16:
        self.sample_rate = int(frame.sample_rate)
        arr: np.ndarray = np.asarray(frame.to_ndarray())
        fmt = frame.format.name
        if arr.ndim == 2 and arr.shape[0] > 1:  # planar multi-channel: downmix
            arr = arr.mean(axis=0)
        arr = arr.reshape(-1)
        pcm: I16
        if fmt.startswith("s16"):
            pcm = arr.astype(np.int16, copy=False)
        elif fmt.startswith(("flt", "dbl")):
            pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype(np.int16)
        elif fmt.startswith("s32"):
            pcm = (arr.astype(np.int64) >> 16).astype(np.int16)
        else:  # pragma: no cover - FFmpeg's mp3 decoder emits s16p/fltp
            pcm = arr.astype(np.int16)
        return pcm

    def _join(self, parts: list[I16]) -> I16:
        if not parts:
            return np.zeros(0, np.int16)
        pcm = parts[0] if len(parts) == 1 else np.concatenate(parts)
        self.samples += int(pcm.size)
        return pcm
