"""Contract suites for the voice-worker Protocols (§3.4)."""

from __future__ import annotations

import collections
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from aivtube.contracts.types import Transcript, VoiceSpec
from aivtube.contracts.voice import (
    F32,
    AudioChunk,
    AudioOut,
    PhraseCache,
    QuotaSpec,
    SpeechRecognizer,
    TTSBackend,
    TTSUnavailable,
    VoiceActivityDetector,
    WordMark,
)
from aivtube.testing.contracts._base import (
    AsyncCase,
    SyncCase,
    _Cases,
    check,
    maybe_await,
    raises,
    wait_until,
)

__all__ = [
    "audio_out_suite",
    "phrase_cache_suite",
    "speech_recognizer_suite",
    "tts_backend_suite",
    "vad_suite",
]

Pump = Callable[[int], F32]


def _tone(seconds: float, rate: int, freq: float = 440.0, amp: float = 0.5) -> F32:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def audio_out_suite(
    factory: Callable[[], AudioOut | tuple[AudioOut, Pump]],
    *,
    blocksize: int = 480,
) -> list[SyncCase]:
    """``factory`` returns an ``AudioOut`` with a ``pump(n)`` method, or ``(out, pump)`` where
    ``pump(n)`` drives the device callback ``n`` blocks and returns the mono output (for the
    real ``StreamingPlayer`` that is ``lambda n: fake_sd.pump(n)[0]``)."""
    cases = _Cases("audio_out")

    def make() -> tuple[AudioOut, Pump]:
        made = factory()
        if isinstance(made, tuple):
            out, pump = made
        else:
            out, pump = made, made.pump  # type: ignore[attr-defined]
        out.start()
        return out, pump

    def blocks(out: AudioOut, seconds: float) -> int:
        return int(np.ceil(seconds * out.sample_rate / blocksize))

    @cases
    def has_contract_attributes() -> None:
        out, _ = make()
        try:
            check(isinstance(out.sample_rate, int) and out.sample_rate > 0, "sample_rate")
            check(out.output_latency_s >= 0.0, "output_latency_s")
            check(isinstance(out.reference, collections.deque), "reference must be a deque")
            check(isinstance(out.stats, Mapping), "stats must be a mapping")
        finally:
            out.close()

    @cases
    def plays_resampled_audio_losslessly() -> None:
        out, pump = make()
        try:
            out.play(_tone(0.2, 24000), 24000)
            y = pump(blocks(out, 0.35))
            nz = np.flatnonzero(np.abs(y) > 1e-3)
            check(nz.size > 0, "nothing was played")
            dur = (nz[-1] - nz[0] + 1) / out.sample_rate
            check(abs(dur - 0.2) < 0.01, f"played {dur:.3f} s of a 0.2 s tone")
        finally:
            out.close()

    @cases
    def mark_fires_true_when_audible() -> None:
        out, pump = make()
        marks: list[tuple[bool, float]] = []
        lock = threading.Lock()

        def cb(heard: bool, t: float) -> None:
            with lock:
                marks.append((heard, t))

        try:
            out.play(_tone(0.1, out.sample_rate), out.sample_rate)
            out.mark(cb)
            pump(blocks(out, 0.2))
            wait_until(lambda: bool(marks), 2.0, what="mark callback")
            heard, t = marks[0]
            check(heard is True and isinstance(t, float), f"mark fired {marks[0]!r}")
        finally:
            out.close()

    @cases
    def cancel_silences_next_block_and_fails_marks() -> None:
        out, pump = make()
        marks: list[bool] = []
        try:
            out.play(_tone(1.0, out.sample_rate, amp=0.5), out.sample_rate)
            out.mark(lambda heard, t: marks.append(heard))
            pump(5)
            dropped = out.cancel(fade_ms=8.0)
            check(dropped > 0.5, f"cancel() reported {dropped:.3f} s dropped")
            first = pump(1)
            rest = pump(3)
            check(float(np.abs(first).max()) <= 0.5 + 1e-3, "fade block louder than the audio")
            check(bool(np.all(rest == 0.0)), "audio continued after the fade block")
            wait_until(lambda: bool(marks), 2.0, what="cancelled mark")
            check(marks == [False], f"pending mark fired {marks!r} instead of [False]")
        finally:
            out.close()

    @cases
    def is_speaking_tracks_queue() -> None:
        out, pump = make()
        try:
            out.play(_tone(0.3, out.sample_rate), out.sample_rate)
            check(out.is_speaking(), "is_speaking() false with audio queued")
            pump(blocks(out, 0.5))
            wait_until(lambda: not out.is_speaking(tail_s=0.0), 2.0, what="is_speaking() false")
        finally:
            out.close()

    @cases
    def zero_gain_mutes() -> None:
        out, pump = make()
        try:
            out.set_gain(0.0, ramp_ms=10.0)
            out.play(_tone(0.3, out.sample_rate), out.sample_rate)
            y = pump(blocks(out, 0.3))
            tail = y[len(y) // 2 :]
            check(float(np.abs(tail).max()) < 1e-3, "set_gain(0) did not mute")
        finally:
            out.close()

    @cases
    def reference_records_played_blocks() -> None:
        out, pump = make()
        try:
            out.play(_tone(0.1, out.sample_rate), out.sample_rate)
            pump(blocks(out, 0.1))
            wait_until(lambda: len(out.reference) > 0, 1.0, what="reference blocks")
            t, blk = out.reference[-1]
            check(isinstance(t, float) and isinstance(blk, np.ndarray), "reference entries")
        finally:
            out.close()

    @cases
    def close_is_idempotent() -> None:
        out, _ = make()
        out.close()
        out.close()

    return cases.items


def vad_suite(
    factory: Callable[[], VoiceActivityDetector], *, speech: F32 | None = None
) -> list[SyncCase]:
    """``speech``: 16 kHz float32 audio that must score above 0.5 somewhere (optional)."""
    cases = _Cases("vad")

    @cases
    def frame_geometry() -> None:
        vad = factory()
        check(vad.sample_rate == 16000 and vad.frame_samples == 512, "16 kHz / 512-sample frames")

    @cases
    def silence_scores_low() -> None:
        vad = factory()
        vad.reset()
        probs = [vad.prob(np.zeros(vad.frame_samples, np.float32)) for _ in range(10)]
        check(max(probs) < 0.5, f"silence scored {max(probs):.2f}")

    @cases
    def probability_is_in_range() -> None:
        vad = factory()
        rng = np.random.default_rng(0)
        for _ in range(10):
            p = vad.prob((0.1 * rng.standard_normal(vad.frame_samples)).astype(np.float32))
            check(0.0 <= p <= 1.0, f"prob {p} outside [0, 1]")

    @cases
    def reset_is_repeatable() -> None:
        vad = factory()
        vad.reset()
        vad.reset()

    if speech is not None:

        @cases
        def speech_scores_high() -> None:
            vad = factory()
            n = vad.frame_samples
            probs = [vad.prob(speech[i : i + n]) for i in range(0, len(speech) - n + 1, n)]
            check(max(probs) > 0.5, f"speech never scored above 0.5 (max {max(probs):.2f})")

    return cases.items


def speech_recognizer_suite(
    factory: Callable[[], SpeechRecognizer],
    *,
    sample: F32 | None = None,
    expected: str | None = None,
) -> list[SyncCase]:
    """``sample``/``expected``: optional 16 kHz audio whose transcript must contain ``expected``."""
    cases = _Cases("speech_recognizer")

    def check_transcript(t: Transcript, rec: SpeechRecognizer, seconds: float) -> None:
        check(isinstance(t, Transcript), f"transcribe returned {type(t).__name__}")
        check(isinstance(t.text, str) and t.is_final, "final text transcript")
        check(t.engine == rec.name, f"engine {t.engine!r} != name {rec.name!r}")
        check(abs(t.audio_s - seconds) < 0.05, f"audio_s {t.audio_s} for {seconds} s")
        check(t.latency_ms >= 0.0, "latency_ms")

    @cases
    def attributes() -> None:
        rec = factory()
        try:
            check(isinstance(rec.name, str) and rec.name, "name")
            check(rec.sample_rate == 16000, "sample_rate must be 16000")
        finally:
            rec.close()

    @cases
    def transcribes_after_warmup() -> None:
        rec = factory()
        try:
            rec.warmup()
            check_transcript(rec.transcribe(np.zeros(16000, np.float32)), rec, 1.0)
        finally:
            rec.close()

    @cases
    def quick_decode() -> None:
        rec = factory()
        try:
            check_transcript(rec.transcribe(np.zeros(8000, np.float32), quick=True), rec, 0.5)
        finally:
            rec.close()

    if sample is not None and expected is not None:

        @cases
        def recognises_sample() -> None:
            rec = factory()
            try:
                t = rec.transcribe(sample)
                check(expected in t.text, f"{expected!r} not in {t.text!r}")
            finally:
                rec.close()

    return cases.items


def tts_backend_suite(
    factory: Callable[[], TTSBackend],
    *,
    voice: VoiceSpec | None = None,
    text: str = "สวัสดีค่ะ ยินดีที่ได้รู้จักนะคะ",
    unavailable_factory: Callable[[], TTSBackend] | None = None,
    first_audio_timeout: float = 5.0,
) -> list[AsyncCase]:
    """``unavailable_factory`` (optional) builds a backend that must fail with
    ``TTSUnavailable`` (never another exception type)."""
    cases = _Cases("tts_backend")
    spec = voice or VoiceSpec("premwadee", "th-TH-PremwadeeNeural", "+8%", "+20Hz")

    async def collect(tts: TTSBackend) -> list[AudioChunk | WordMark]:
        return [
            item
            async for item in tts.synth(
                text, spec, first_audio_timeout=first_audio_timeout, idle_timeout=5.0
            )
        ]

    @cases
    async def attributes() -> None:
        tts = await maybe_await(factory())
        try:
            check(isinstance(tts.name, str) and tts.name, "name")
            check(tts.normalizer in ("cloud", "local"), f"normalizer {tts.normalizer!r}")
            check(tts.quota is None or isinstance(tts.quota, QuotaSpec), "quota")
        finally:
            await tts.aclose()

    @cases
    async def synth_streams_int16_audio() -> None:
        tts = await maybe_await(factory())
        try:
            await tts.warmup()
            items = await collect(tts)
            chunks = [i for i in items if isinstance(i, AudioChunk)]
            check(chunks, "no AudioChunk produced")
            for c in chunks:
                check(c.pcm.dtype == np.int16 and c.pcm.ndim == 1, "pcm must be 1-D int16")
                check(c.sample_rate > 0, "sample_rate")
            seconds = sum(c.pcm.size / c.sample_rate for c in chunks)
            check(seconds > 0.1, f"only {seconds:.3f} s of audio")
            offsets = [i.offset_s for i in items if isinstance(i, WordMark)]
            check(all(o >= 0 for o in offsets), "negative word offsets")
            check(offsets == sorted(offsets), "word marks out of order")
        finally:
            await tts.aclose()

    @cases
    async def early_close_is_safe() -> None:
        tts = await maybe_await(factory())
        try:
            agen: Any = tts.synth(
                text, spec, first_audio_timeout=first_audio_timeout, idle_timeout=5.0
            )
            await agen.__anext__()
            await agen.aclose()
        finally:
            await tts.aclose()

    if unavailable_factory is not None:
        make_bad = unavailable_factory

        @cases
        async def failure_is_tts_unavailable() -> None:
            tts = await maybe_await(make_bad())
            try:
                with raises(TTSUnavailable, what="a failing backend"):
                    async for _ in tts.synth(text, spec, first_audio_timeout=0.5, idle_timeout=0.5):
                        pass
            finally:
                await tts.aclose()

    return cases.items


def phrase_cache_suite(factory: Callable[[], PhraseCache]) -> list[SyncCase]:
    cases = _Cases("phrase_cache")
    key = ("edge", "th-TH-PremwadeeNeural", "+8%", "+20Hz", "filtered.")
    audio = AudioChunk(np.arange(-100, 100, dtype=np.int16), 24000)

    @cases
    def miss_returns_none() -> None:
        check(factory().get(key) is None, "empty cache returned audio")

    @cases
    def put_then_get() -> None:
        cache = factory()
        cache.put(key, audio)
        got = cache.get(key)
        check(got is not None, "stored phrase not found")
        assert got is not None
        check(got.sample_rate == 24000 and np.array_equal(got.pcm, audio.pcm), "audio changed")
        check(cache.get((*key[:-1], "other")) is None, "a different key hit")

    @cases
    def overwrite_replaces() -> None:
        cache = factory()
        cache.put(key, audio)
        newer = AudioChunk(np.zeros(10, np.int16), 16000)
        cache.put(key, newer)
        got = cache.get(key)
        check(got is not None and got.sample_rate == 16000, "overwrite ignored")

    return cases.items
