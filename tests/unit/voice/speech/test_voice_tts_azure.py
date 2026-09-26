"""``AzureTTSBackend`` against a fake Speech SDK, plus ``TokenBucket`` (voice.tts)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from voice_speech_kit import FakeAzureSdk

from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, QuotaSpec, TTSBackend, TTSUnavailable, WordMark
from aivtube.testing.contracts import case_id, tts_backend_suite
from aivtube.testing.fakes import FakeClock
from aivtube.voice.tts import AzureTTSBackend, TokenBucket, azure_ssml

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz", volume="+0%")


def _azure(sdk: FakeAzureSdk, **kw: Any) -> AzureTTSBackend:
    return AzureTTSBackend("key", "southeastasia", sdk=sdk, **kw)


async def _collect(tts: TTSBackend, text: str = "สวัสดีค่ะ", **kw: float) -> list[Any]:
    return [
        i
        async for i in tts.synth(
            text,
            VOICE,
            first_audio_timeout=kw.get("first", 2.0),
            idle_timeout=kw.get("idle", 4.0),
        )
    ]


def test_ssml_carries_voice_prosody_and_escapes_text() -> None:
    ssml = azure_ssml("a < b & 'c'", VOICE)
    assert "name='th-TH-PremwadeeNeural'" in ssml or 'name="th-TH-PremwadeeNeural"' in ssml
    assert "+20Hz" in ssml and "+8%" in ssml and "+0%" in ssml
    assert "a &lt; b &amp; 'c'" in ssml and "xml:lang" in ssml


def test_quota_follows_the_tier_and_credentials_are_required() -> None:
    sdk = FakeAzureSdk()
    assert _azure(sdk).quota == QuotaSpec(18, 60.0)
    assert _azure(sdk, tier="F0", requests_per_min=12).quota == QuotaSpec(12, 60.0)
    assert _azure(sdk, tier="S0").quota is None
    with pytest.raises(ValueError):
        AzureTTSBackend("", "region", sdk=sdk)


async def test_streams_pcm_and_word_marks_and_reuses_the_synthesizer() -> None:
    sdk = FakeAzureSdk(seconds=0.5)
    tts = _azure(sdk)
    assert isinstance(tts, TTSBackend) and tts.normalizer == "cloud"
    items = await _collect(tts, "สวัสดีค่ะ")
    audio = [i for i in items if isinstance(i, AudioChunk)]
    marks = [i for i in items if isinstance(i, WordMark)]
    pcm = np.concatenate([a.pcm for a in audio])
    assert pcm.dtype == np.int16 and pcm.size == int(0.5 * 24000)  # odd splits carried
    assert all(a.sample_rate == 24000 for a in audio)
    assert [m.text for m in marks] == ["สวัสดี", "ค่ะ"]  # punctuation boundaries dropped
    assert marks[1].offset_s == pytest.approx(0.3) and marks[1].duration_s == pytest.approx(0.25)
    assert sdk.formats == ["raw24k"]
    assert "สวัสดีค่ะ" in sdk.ssml[0]
    await _collect(tts, "อีกครั้ง")
    assert len(sdk.synths) == 1  # the pooled synthesizer was reused


async def test_warmup_preconnects() -> None:
    sdk = FakeAzureSdk()
    tts = _azure(sdk)
    await tts.warmup()
    assert sdk.opened == 1 and len(sdk.synths) == 1
    await _collect(tts)
    assert len(sdk.synths) == 1


async def test_canceled_synthesis_is_unavailable() -> None:
    sdk = FakeAzureSdk()
    sdk.cancel = True
    with pytest.raises(TTSUnavailable, match="quota exceeded"):
        await _collect(_azure(sdk))


async def test_no_audio_within_the_first_audio_timeout() -> None:
    clock = FakeClock()
    sdk = FakeAzureSdk()
    sdk.silent = True
    tts = _azure(sdk, clock=clock)
    task = asyncio.create_task(_collect(tts, first=2.0))
    await clock.run_for(2.1)
    with pytest.raises(TTSUnavailable, match="first audio"):
        await task
    assert sdk.stopped == 1  # the aborted request was stopped and its synthesizer dropped
    await _collect(tts.__class__("k", "r", sdk=FakeAzureSdk()))


async def test_early_close_discards_the_synthesizer() -> None:
    sdk = FakeAzureSdk(seconds=1.0)
    tts = _azure(sdk)
    agen: Any = tts.synth("x", VOICE, first_audio_timeout=2.0, idle_timeout=4.0)
    await agen.__anext__()
    await agen.aclose()
    assert sdk.stopped == 1
    await _collect(tts)
    assert len(sdk.synths) == 2  # a fresh synthesizer; late events cannot leak across


def _suite() -> list[Any]:
    def bad() -> AzureTTSBackend:
        sdk = FakeAzureSdk()
        sdk.cancel = True
        return _azure(sdk)

    return tts_backend_suite(lambda: _azure(FakeAzureSdk(seconds=0.5)), unavailable_factory=bad)


@pytest.mark.parametrize("case", _suite(), ids=case_id)
async def test_azure_backend_passes_the_tts_backend_suite(case: Any) -> None:
    await case()


# --- token bucket -----------------------------------------------------------------------------


def test_token_bucket_takes_without_waiting_and_refills_continuously() -> None:
    clock = FakeClock()
    bucket = TokenBucket(3, 60.0, clock.now)
    assert [bucket.try_take() for _ in range(4)] == [True, True, True, False]
    clock.advance(19.0)
    assert not bucket.try_take()  # 0.95 tokens
    clock.advance(1.5)
    assert bucket.try_take()
    clock.advance(600.0)
    assert bucket.tokens == pytest.approx(3.0)  # never above capacity
    bucket.refund()
    assert bucket.tokens == pytest.approx(3.0)
    assert TokenBucket.from_quota(QuotaSpec(18, 60.0), clock.now).capacity == 18
    with pytest.raises(ValueError):
        TokenBucket(0, 60.0, clock.now)
