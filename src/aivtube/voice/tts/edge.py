"""edge-tts 7.2.8 backend: ``th-TH-PremwadeeNeural`` streamed through a stateful MP3 decoder.

One ``edge_tts.Communicate`` per segment (``stream()`` works once per object) with
``boundary="WordBoundary"``; every WebSocket audio message goes into the same
``Mp3StreamDecoder`` (never decode the ~720-byte messages on their own). WordBoundary metadata
becomes ``WordMark`` (offsets are 100 ns ticks). Most marks arrive before the first audio.

Deadlines (I2): no decoded audio within ``first_audio_timeout`` seconds, or a gap longer than
``idle_timeout`` after that, raises ``TTSUnavailable``; so does every edge/network/decoder
error. The service fails slowly (``NoAudioReceived`` after ~4.6 s), so the router's 2 s / 4 s
timeouts are what keeps speech moving.

TLS: edge-tts builds its own SSL context from certifi (``edge_tts.communicate._SSL_CTX``) and
ignores the OS store. Behind a TLS-intercepting proxy pass ``ca_bundle`` (config
``tts.backends.edge.ca_bundle``, or env ``AIVTUBE_EDGE_CA_BUNDLE``) or ``ssl_ctx``; the
override is process-wide. On a normal PC leave both unset. ``HTTPS_PROXY`` is honoured by
aiohttp (``trust_env``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import ssl
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, QuotaSpec, TTSUnavailable, WordMark
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.voice.tts.decoder import Mp3StreamDecoder

__all__ = [
    "CA_BUNDLE_ENV",
    "EdgeTTS",
    "EdgeTTSBackend",
    "edge_ssl_context",
    "install_edge_ssl_context",
]

log = logging.getLogger("aivtube.voice.tts.edge")

CA_BUNDLE_ENV = "AIVTUBE_EDGE_CA_BUNDLE"
EDGE_SAMPLE_RATE = 24000
_TICKS = 1e7  # edge offsets/durations are 100 ns ticks


def edge_ssl_context(ca_bundle: str | Path) -> ssl.SSLContext:
    """A default-verifying SSL context that trusts ``ca_bundle`` (a PEM file)."""
    path = Path(ca_bundle).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CA bundle not found: {path}")
    return ssl.create_default_context(cafile=str(path))


def install_edge_ssl_context(ctx: ssl.SSLContext) -> None:
    """Make every later edge-tts connection in this process use ``ctx``."""
    import edge_tts.communicate as communicate

    communicate._SSL_CTX = ctx  # the documented override hook


class EdgeTTSBackend:
    """``TTSBackend`` over edge-tts (unofficial Edge Read Aloud endpoint; best effort).

    ``communicate_factory`` replaces ``edge_tts.Communicate`` (tests feed recorded messages);
    it is called like ``Communicate(text, voice, rate=, pitch=, volume=, boundary=, proxy=,
    connect_timeout=, receive_timeout=)`` and must return an object with ``stream()``.
    """

    normalizer: Literal["cloud", "local"] = "cloud"
    quota: QuotaSpec | None = None

    def __init__(
        self,
        *,
        name: str = "edge",
        ssl_ctx: ssl.SSLContext | None = None,
        ca_bundle: str | Path | None = None,
        proxy: str | None = None,
        connect_timeout_s: float = 5.0,
        clock: Clock | None = None,
        communicate_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.normalizer = "cloud"
        self.quota = None
        if ssl_ctx is None and not ca_bundle:
            ca_bundle = os.environ.get(CA_BUNDLE_ENV, "").strip() or None
        if ssl_ctx is None and ca_bundle:
            ssl_ctx = edge_ssl_context(ca_bundle)
        self._ssl_ctx = ssl_ctx
        self._ssl_installed = False
        self._proxy = proxy
        self._connect_timeout = max(1, math.ceil(connect_timeout_s))
        self._clock: Clock = clock or SystemClock()
        self._factory = communicate_factory
        self._imported = False
        self._closed = False
        self.requests = 0

    # --- TTSBackend -----------------------------------------------------------------------
    async def warmup(self) -> None:
        """Import edge-tts, aiohttp and PyAV off the event loop and apply the SSL override."""
        await asyncio.to_thread(self._prepare)

    def synth(
        self,
        text: str,
        voice: VoiceSpec,
        *,
        first_audio_timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[AudioChunk | WordMark]:
        return self._synth(text, voice, first_audio_timeout, idle_timeout)

    async def aclose(self) -> None:
        self._closed = True

    # --- internals ------------------------------------------------------------------------
    def _prepare(self) -> None:
        if not self._imported:
            import av  # noqa: F401  # load the FFmpeg libraries once

            self._imported = True
        if self._factory is None:
            import edge_tts

            self._factory = edge_tts.Communicate
        if self._ssl_ctx is not None and not self._ssl_installed:
            install_edge_ssl_context(self._ssl_ctx)
            self._ssl_installed = True

    async def _synth(
        self, text: str, voice: VoiceSpec, first_audio_timeout: float, idle_timeout: float
    ) -> AsyncIterator[AudioChunk | WordMark]:
        if self._closed:
            raise TTSUnavailable(f"{self.name} is closed")
        text = text.strip()
        if not text:
            return
        try:
            self._prepare()
            assert self._factory is not None
            communicate = self._factory(
                text,
                voice.voice,
                rate=voice.rate,
                pitch=voice.pitch,
                volume=voice.volume,
                boundary="WordBoundary",
                proxy=self._proxy,
                connect_timeout=self._connect_timeout,
                receive_timeout=max(1, math.ceil(idle_timeout)),
            )
            stream: Any = communicate.stream()
            decoder = Mp3StreamDecoder()
        except Exception as exc:
            raise TTSUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        self.requests += 1
        started = self._clock.now()
        got_audio = False
        try:
            while True:
                if got_audio:
                    budget = idle_timeout
                else:
                    budget = first_audio_timeout - (self._clock.now() - started)
                    if budget <= 0:
                        raise TTSUnavailable(
                            f"{self.name}: no audio within {first_audio_timeout:g} s"
                        )
                try:
                    async with deadline(budget, what=f"{self.name} audio", clock=self._clock):
                        msg = await anext(stream)
                except StopAsyncIteration:
                    break
                kind = msg.get("type")
                if kind == "audio":
                    pcm = decoder.feed(msg.get("data") or b"")
                    if pcm.size:
                        got_audio = True
                        yield AudioChunk(pcm, decoder.sample_rate or EDGE_SAMPLE_RATE)
                elif kind in ("WordBoundary", "SentenceBoundary"):
                    word = str(msg.get("text", "")).strip()
                    if word:
                        yield WordMark(
                            word,
                            float(msg.get("offset", 0)) / _TICKS,
                            float(msg.get("duration", 0)) / _TICKS,
                        )
            tail = decoder.flush()
            if tail.size:
                got_audio = True
                yield AudioChunk(tail, decoder.sample_rate or EDGE_SAMPLE_RATE)
            if not got_audio:
                raise TTSUnavailable(f"{self.name}: no audio received")
        except TTSUnavailable:
            raise
        except DeadlineExceeded as exc:
            what = "first audio" if not got_audio else "audio (stalled)"
            raise TTSUnavailable(f"{self.name}: no {what} within {exc.seconds:g} s") from exc
        except Exception as exc:  # edge, aiohttp, TLS, PyAV: all mean "unavailable"
            raise TTSUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        finally:
            await _aclose(stream)


async def _aclose(stream: Any) -> None:
    """Close the edge stream (its aiohttp session) with a short deadline."""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        async with asyncio.timeout(1.0):
            await close()


EdgeTTS = EdgeTTSBackend  # modules.json name
