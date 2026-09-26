"""Typhoon hosted ASR (OpenAI-compatible ``/v1/audio/transcriptions``), cloud, opt-in only.

Audio leaves the PC, so this recognizer is built only with ``privacy.cloud_stt_consent`` (I8;
see ``voice.stt.factory``). It sends 16 kHz mono 16-bit WAV bytes as multipart
``file`` with ``model=typhoon-asr-realtime``. Errors come back as ``{"detail": …}``.
The call is blocking (STT thread) and bounded by ``timeout_s``.
"""

from __future__ import annotations

import io
import time
import wave
from typing import Any

import numpy as np

from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32
from aivtube.voice.stt.sherpa import to_mono_f32

__all__ = ["TyphoonApiRecognizer", "wav_bytes"]

SAMPLE_RATE = 16000


def wav_bytes(pcm16k: F32, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono 16-bit PCM WAV of float audio in [-1, 1]."""
    x = np.clip(to_mono_f32(pcm16k), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TyphoonApiRecognizer:
    """``SpeechRecognizer`` for ``https://api.opentyphoon.ai/v1`` (model ``typhoon-asr-realtime``).

    ``client`` is an ``httpx2.Client`` (tests pass one with a ``MockTransport``). Raises
    ``RuntimeError`` on HTTP or network errors so ``SttRunner`` moves down the chain.
    """

    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout_s: float = 3.0,
        language: str | None = "th",
        name: str = "typhoon_api",
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("TyphoonApiRecognizer needs an API key")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._own_client = client is None
        if client is None:
            import httpx2

            client = httpx2.Client(timeout=httpx2.Timeout(timeout_s, connect=min(2.0, timeout_s)))
        self._client: Any = client

    def warmup(self) -> None:
        """No-op: a network warm-up would send audio to the cloud for nothing."""

    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript:
        t0 = time.perf_counter()
        x = to_mono_f32(pcm16k)
        if self._client is None:
            raise RuntimeError(f"recognizer {self.name!r} is closed")
        data: dict[str, str] = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        try:
            resp = self._client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=data,
                files={"file": ("audio.wav", wav_bytes(x), "audio/wav")},
                timeout=self.timeout_s,
            )
        except Exception as exc:  # network errors of any flavour: next backend
            raise RuntimeError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            detail: Any = ""
            try:
                body = resp.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                detail = resp.text[:200]
            raise RuntimeError(f"{self.name}: HTTP {resp.status_code}: {detail}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.name}: invalid JSON response") from exc
        text = body.get("text", "") if isinstance(body, dict) else ""
        return Transcript(
            text=str(text or "").strip(),
            is_final=True,
            audio_s=x.size / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            engine=self.name,
        )

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._own_client:
            client.close()
