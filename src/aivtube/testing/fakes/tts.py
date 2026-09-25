"""TTS fakes (ARCHITECTURE.md §3.4, §10): ``FakeTTS`` and an in-memory ``PhraseCache``."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Literal

import numpy as np

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import I16, AudioChunk, QuotaSpec, TTSUnavailable, WordMark

__all__ = ["FakePhraseCache", "FakeTTS"]


class FakeTTS:
    """``TTSBackend`` producing ``len(text) / chars_per_s`` seconds of audio plus word marks.

    Knobs: ``ttfa_s`` (time to first audio), ``fail_rate`` (seeded), ``no_audio`` (edge-tts
    ``NoAudioReceived``: waits, then fails), ``tone_hz`` (a tone instead of silence, e.g. for
    lip-sync), ``chunk_s`` and ``pace`` (sleep in real time between chunks). Sleeps go through
    ``clock.sleep`` when a clock is given, so ``FakeClock`` controls them.
    """

    def __init__(
        self,
        *,
        ttfa_s: float = 0.25,
        fail_rate: float = 0.0,
        chars_per_s: float = 12.5,
        name: str = "fake",
        sample_rate: int = 24000,
        chunk_s: float = 0.1,
        no_audio: bool = False,
        tone_hz: float | None = None,
        pace: bool = False,
        normalizer: Literal["cloud", "local"] = "cloud",
        quota: QuotaSpec | None = None,
        seed: int = 0,
        clock: Clock | None = None,
    ) -> None:
        self.name = name
        self.normalizer: Literal["cloud", "local"] = normalizer
        self.quota = quota
        self.ttfa_s = ttfa_s
        self.fail_rate = fail_rate
        self.chars_per_s = chars_per_s
        self.sample_rate = sample_rate
        self.chunk_s = chunk_s
        self.no_audio = no_audio
        self.tone_hz = tone_hz
        self.pace = pace
        self._rng = random.Random(seed)
        self._sleep: Callable[[float], Awaitable[None]] = clock.sleep if clock else asyncio.sleep
        self.requests: list[tuple[str, VoiceSpec]] = []
        self.completed = 0
        self.failures = 0
        self.warmed = False
        self.closed = False

    async def warmup(self) -> None:
        self.warmed = True

    def duration_s(self, text: str) -> float:
        return len(text) / self.chars_per_s if self.chars_per_s > 0 else 0.0

    async def synth(
        self,
        text: str,
        voice: VoiceSpec,
        *,
        first_audio_timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[AudioChunk | WordMark]:
        if self.closed:
            raise TTSUnavailable(f"{self.name} is closed")
        self.requests.append((text, voice))
        if self._rng.random() < self.fail_rate:
            self.failures += 1
            await self._sleep(min(self.ttfa_s, first_audio_timeout))
            raise TTSUnavailable(f"{self.name}: scripted failure")
        if self.ttfa_s > first_audio_timeout:
            self.failures += 1
            await self._sleep(first_audio_timeout)
            raise TTSUnavailable(f"{self.name}: no first audio within {first_audio_timeout:.2f} s")
        await self._sleep(self.ttfa_s)
        if self.no_audio:
            self.failures += 1
            raise TTSUnavailable(f"{self.name}: no audio received")
        total = round(self.duration_s(text) * self.sample_rate)
        marks = self._word_marks(text)
        chunk = max(1, int(self.chunk_s * self.sample_rate))
        pos = 0
        mi = 0
        while pos < total:
            while mi < len(marks) and marks[mi].offset_s * self.sample_rate <= pos:
                yield marks[mi]
                mi += 1
            n = min(chunk, total - pos)
            yield AudioChunk(self._pcm(pos, n), self.sample_rate)
            pos += n
            if self.pace and pos < total:
                await self._sleep(n / self.sample_rate)
        for mark in marks[mi:]:
            yield mark
        self.completed += 1

    async def aclose(self) -> None:
        self.closed = True

    def _word_marks(self, text: str) -> list[WordMark]:
        if not text:
            return []
        cps = self.chars_per_s or 1.0
        marks: list[WordMark] = []
        start = 0
        for word in text.split(" "):
            if word:
                marks.append(WordMark(word, start / cps, len(word) / cps))
            start += len(word) + 1
        return marks

    def _pcm(self, pos: int, n: int) -> I16:
        if self.tone_hz is None:
            return np.zeros(n, np.int16)
        t = (np.arange(pos, pos + n, dtype=np.float64)) / self.sample_rate
        return (8000.0 * np.sin(2.0 * np.pi * self.tone_hz * t)).astype(np.int16)


class FakePhraseCache:
    """``PhraseCache`` held in a dict."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, ...], AudioChunk] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, ...]) -> AudioChunk | None:
        item = self.items.get(tuple(key))
        if item is None:
            self.misses += 1
        else:
            self.hits += 1
        return item

    def put(self, key: tuple[str, ...], audio: AudioChunk) -> None:
        self.items[tuple(key)] = audio
