"""Azure AI Speech backend: the same ``th-TH-PremwadeeNeural`` voice with raw PCM (TTS brief §7).

Output is ``Raw24Khz16BitMonoPcm`` streamed through the synthesizer's ``synthesizing`` events;
``synthesis_word_boundary`` events become ``WordMark``. Prosody comes from the ``VoiceSpec``
through SSML ``<prosody pitch rate volume>``, so edge and Azure sound the same and may
substitute for each other within one utterance.

SDK callbacks arrive on SDK threads and are handed to the event loop with
``call_soon_threadsafe``. Synthesizers are pooled (one request each at a time) and
pre-connected in ``warmup`` (``Connection.from_speech_synthesizer(syn).open(True)``); a
synthesizer whose request was aborted is discarded, so late events never leak into the next
request. The F0 tier allows 20 transactions per minute: ``quota`` advertises
``requests_per_min`` per 60 s and the router's token bucket enforces it.
``azure.cognitiveservices.speech`` is imported lazily (extra ``tts-azure``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, QuotaSpec, TTSUnavailable, WordMark
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline

__all__ = ["AzureTTS", "AzureTTSBackend", "azure_ssml"]

log = logging.getLogger("aivtube.voice.tts.azure")

AZURE_SAMPLE_RATE = 24000
_TICKS = 1e7


def azure_ssml(text: str, voice: VoiceSpec, lang: str = "th-TH") -> str:
    """SSML for one segment with the voice's prosody."""
    return (
        f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang={quoteattr(lang)}>"
        f"<voice name={quoteattr(voice.voice)}>"
        f"<prosody pitch={quoteattr(voice.pitch)} rate={quoteattr(voice.rate)} "
        f"volume={quoteattr(voice.volume)}>{escape(text)}</prosody></voice></speak>"
    )


class _Request:
    """Receives one request's events from SDK threads."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def push(self, kind: str, payload: Any = None) -> None:
        with contextlib.suppress(RuntimeError):  # loop closed during shutdown
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, payload))


class _Session:
    """One pooled ``SpeechSynthesizer`` with event handlers bound to its current request."""

    def __init__(self, sdk: Any, config: Any) -> None:
        self.sdk = sdk
        self.syn = sdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        self.target: _Request | None = None
        self.connection: Any = None
        self.syn.synthesizing.connect(self._on_audio)
        self.syn.synthesis_word_boundary.connect(self._on_word)
        self.syn.synthesis_completed.connect(self._on_done)
        self.syn.synthesis_canceled.connect(self._on_canceled)

    def preconnect(self) -> None:
        self.connection = self.sdk.Connection.from_speech_synthesizer(self.syn)
        self.connection.open(True)

    def _on_audio(self, evt: Any) -> None:
        req = self.target
        data = getattr(getattr(evt, "result", None), "audio_data", None)
        if req is not None and data:
            req.push("audio", bytes(data))

    def _on_word(self, evt: Any) -> None:
        req = self.target
        if req is None:
            return
        word_type = getattr(self.sdk.SpeechSynthesisBoundaryType, "Word", None)
        if word_type is not None and getattr(evt, "boundary_type", word_type) != word_type:
            return
        duration = getattr(evt, "duration", 0)
        seconds = (
            duration.total_seconds() if hasattr(duration, "total_seconds") else duration / _TICKS
        )
        req.push("word", (str(evt.text), float(evt.audio_offset) / _TICKS, float(seconds)))

    def _on_done(self, evt: Any) -> None:
        req = self.target
        if req is not None:
            req.push("done")

    def _on_canceled(self, evt: Any) -> None:
        req = self.target
        if req is None:
            return
        details = getattr(getattr(evt, "result", None), "cancellation_details", None)
        reason = (
            getattr(details, "error_details", None) or getattr(details, "reason", "") or "canceled"
        )
        req.push("canceled", str(reason))

    def stop(self) -> None:
        self.target = None
        with contextlib.suppress(Exception):
            self.syn.stop_speaking_async()

    def close(self) -> None:
        if self.target is not None:
            self.stop()
        if self.connection is not None:
            with contextlib.suppress(Exception):
                self.connection.close()
        self.connection = None


class AzureTTSBackend:
    """``TTSBackend`` over the Azure Speech SDK. ``sdk`` injects a fake SDK module (tests)."""

    normalizer: Literal["cloud", "local"] = "cloud"
    quota: QuotaSpec | None

    def __init__(
        self,
        key: str,
        region: str,
        *,
        tier: Literal["F0", "S0"] = "F0",
        requests_per_min: int = 18,
        name: str = "azure",
        pool_size: int = 2,
        sdk: Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not key or not region:
            raise ValueError("Azure TTS needs a key and a region")
        self.name = name
        self.normalizer = "cloud"
        self.tier = tier
        self.quota = QuotaSpec(requests_per_min, 60.0) if tier == "F0" else None
        self._key = key
        self._region = region
        self._sdk = sdk
        self._config: Any = None
        self._pool_size = max(1, pool_size)
        self._free: list[_Session] = []
        self._sessions = 0
        self._clock: Clock = clock or SystemClock()
        self._closed = False
        self.requests = 0

    # --- TTSBackend -----------------------------------------------------------------------
    async def warmup(self) -> None:
        """Load the SDK and pre-connect one synthesizer (in a thread, bounded by 10 s)."""
        async with deadline(10.0, what=f"{self.name} warm-up", clock=self._clock):
            session = await asyncio.to_thread(self._new_session, True)
        self._release(session)

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
        free, self._free = self._free, []
        for s in free:
            s.close()

    # --- internals ------------------------------------------------------------------------
    def _ensure_config(self) -> Any:
        if self._sdk is None:
            import azure.cognitiveservices.speech as speechsdk

            self._sdk = speechsdk
        if self._config is None:
            sdk = self._sdk
            cfg = sdk.SpeechConfig(subscription=self._key, region=self._region)
            cfg.set_speech_synthesis_output_format(
                sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
            )
            self._config = cfg
        return self._config

    def _new_session(self, preconnect: bool = False) -> _Session:
        config = self._ensure_config()
        session = _Session(self._sdk, config)
        self._sessions += 1
        if preconnect:
            try:
                session.preconnect()
            except Exception as exc:
                log.warning("%s: pre-connect failed: %s", self.name, exc)
        return session

    async def _acquire(self) -> _Session:
        if self._free:
            return self._free.pop()
        return await asyncio.to_thread(self._new_session)  # native constructor: off the loop

    def _release(self, session: _Session) -> None:
        session.target = None
        if self._closed or len(self._free) >= self._pool_size:
            session.close()
            self._sessions -= 1
        else:
            self._free.append(session)

    def _discard(self, session: _Session) -> None:
        session.close()
        self._sessions -= 1

    async def _synth(
        self, text: str, voice: VoiceSpec, first_audio_timeout: float, idle_timeout: float
    ) -> AsyncIterator[AudioChunk | WordMark]:
        if self._closed:
            raise TTSUnavailable(f"{self.name} is closed")
        text = text.strip()
        if not text:
            return
        loop = asyncio.get_running_loop()
        try:
            async with deadline(
                first_audio_timeout, what=f"{self.name} session", clock=self._clock
            ):
                session = await self._acquire()
        except Exception as exc:  # incl. DeadlineExceeded
            raise TTSUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        req = _Request(loop)
        session.target = req
        finished = False
        got_audio = False
        carry = b""
        started = self._clock.now()
        self.requests += 1
        try:
            session.syn.speak_ssml_async(azure_ssml(text, voice))
            while True:
                if got_audio:
                    budget = idle_timeout
                else:
                    budget = first_audio_timeout - (self._clock.now() - started)
                    if budget <= 0:
                        raise TTSUnavailable(
                            f"{self.name}: no audio within {first_audio_timeout:g} s"
                        )
                async with deadline(budget, what=f"{self.name} audio", clock=self._clock):
                    kind, payload = await req.queue.get()
                if kind == "audio":
                    data = carry + payload
                    even = len(data) - (len(data) % 2)
                    carry = data[even:]
                    if even:
                        got_audio = True
                        yield AudioChunk(
                            np.frombuffer(data[:even], dtype="<i2").astype(np.int16),
                            AZURE_SAMPLE_RATE,
                        )
                elif kind == "word":
                    word, offset, duration = payload
                    if word.strip():
                        yield WordMark(word.strip(), offset, duration)
                elif kind == "done":
                    finished = True
                    break
                elif kind == "canceled":
                    finished = True
                    raise TTSUnavailable(f"{self.name}: synthesis canceled: {payload}")
            if not got_audio:
                raise TTSUnavailable(f"{self.name}: no audio received")
        except TTSUnavailable:
            raise
        except DeadlineExceeded as exc:
            what = "first audio" if not got_audio else "audio (stalled)"
            raise TTSUnavailable(f"{self.name}: no {what} within {exc.seconds:g} s") from exc
        except Exception as exc:
            raise TTSUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        finally:
            if finished:
                self._release(session)
            else:  # aborted mid-request: stop it and never reuse this synthesizer
                self._discard(session)


AzureTTS = AzureTTSBackend  # modules.json name
