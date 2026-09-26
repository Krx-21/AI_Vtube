"""Stateful PyAV decoding and ``EdgeTTSBackend`` with a scripted edge-tts stream (voice.tts)."""

from __future__ import annotations

import asyncio
import shutil
import ssl
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("av")

from voice_speech_kit import (
    CommunicateFactory,
    decode_whole,
    edge_messages,
    encode_mp3,
    vowel,
)

from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, TTSBackend, TTSUnavailable, WordMark
from aivtube.testing.contracts import case_id, tts_backend_suite
from aivtube.testing.fakes import FakeClock
from aivtube.voice.tts import (
    CA_BUNDLE_ENV,
    EdgeTTSBackend,
    Mp3StreamDecoder,
    edge_ssl_context,
)

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz", volume="+0%")
TEXT = "สวัสดีค่ะ ทุกคน วันนี้ ไพลิน จะมา เล่นเกม กันนะคะ"


@pytest.fixture(scope="module")
def mp3() -> bytes:
    return encode_mp3(vowel(1.5))


# --- decoder ----------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [720, 37, 1, 100_000])
def test_stateful_decode_is_sample_exact_against_whole_buffer(mp3: bytes, chunk: int) -> None:
    ref = decode_whole(mp3)
    dec = Mp3StreamDecoder()
    parts = [dec.feed(mp3[i : i + chunk]) for i in range(0, len(mp3), chunk)]
    parts.append(dec.flush())
    got = np.concatenate(parts)
    assert got.dtype == np.int16 and dec.sample_rate == 24000
    assert np.array_equal(got, ref)
    assert dec.samples == ref.size
    # first PCM arrives long before the stream ends (streaming, not buffering)
    dec2 = Mp3StreamDecoder()
    first = next(i for i in range(0, len(mp3), 720) if dec2.feed(mp3[i : i + 720]).size)
    assert first <= 2 * 720


def test_decoder_rejects_use_after_flush_and_ignores_empty_input(mp3: bytes) -> None:
    dec = Mp3StreamDecoder()
    assert dec.feed(b"").size == 0
    dec.feed(mp3)
    dec.flush()
    assert dec.flush().size == 0
    with pytest.raises(RuntimeError):
        dec.feed(mp3[:720])


# --- edge backend -----------------------------------------------------------------------------


def _backend(factory: CommunicateFactory, clock: Any = None) -> EdgeTTSBackend:
    return EdgeTTSBackend(communicate_factory=factory, clock=clock)


async def _collect(tts: TTSBackend, text: str = TEXT, **kw: float) -> list[AudioChunk | WordMark]:
    kw.setdefault("first_audio_timeout", 2.0)
    kw.setdefault("idle_timeout", 4.0)
    return [
        item
        async for item in tts.synth(
            text,
            VOICE,
            first_audio_timeout=kw["first_audio_timeout"],
            idle_timeout=kw["idle_timeout"],
        )
    ]


async def test_edge_streams_int16_audio_and_word_marks(mp3: bytes) -> None:
    factory = CommunicateFactory(lambda text: edge_messages(text, mp3))
    tts = _backend(factory)
    assert isinstance(tts, TTSBackend) and tts.normalizer == "cloud" and tts.quota is None
    items = await _collect(tts)
    audio = [i for i in items if isinstance(i, AudioChunk)]
    marks = [i for i in items if isinstance(i, WordMark)]
    pcm = np.concatenate([a.pcm for a in audio])
    assert all(a.sample_rate == 24000 and a.pcm.dtype == np.int16 for a in audio)
    assert np.array_equal(pcm, decode_whole(mp3))  # the stateful decoder saw every message
    assert [m.text for m in marks] == TEXT.split(" ")
    assert marks[0].offset_s == 0.0 and marks[1].offset_s == pytest.approx(10 / 12.5)
    assert all(m.duration_s > 0 for m in marks)
    comm = factory.made[0]
    assert comm.voice == "th-TH-PremwadeeNeural" and comm.text == TEXT
    assert comm.kwargs["boundary"] == "WordBoundary"
    assert (comm.kwargs["rate"], comm.kwargs["pitch"], comm.kwargs["volume"]) == (
        "+8%",
        "+20Hz",
        "+0%",
    )
    assert isinstance(comm.kwargs["receive_timeout"], int) and comm.kwargs["receive_timeout"] >= 1
    assert comm.closed


async def test_first_audio_timeout_raises_tts_unavailable(mp3: bytes) -> None:
    clock = FakeClock()
    factory = CommunicateFactory(
        lambda text: edge_messages(text, mp3, marks_first=0), delays=[3.0], sleep=clock.sleep
    )
    tts = _backend(factory, clock)
    task = asyncio.create_task(_collect(tts, first_audio_timeout=2.0))
    await clock.run_for(1.9)
    assert not task.done()
    await clock.run_for(0.2)
    with pytest.raises(TTSUnavailable, match="first audio"):
        await task
    assert factory.made[0].closed


async def test_metadata_before_audio_counts_toward_the_first_audio_timeout(mp3: bytes) -> None:
    clock = FakeClock()
    words = [
        {"type": "WordBoundary", "offset": 0, "duration": 100, "text": f"w{i}"} for i in range(5)
    ]
    factory = CommunicateFactory(lambda text: words, delays=[0.6] * 5, sleep=clock.sleep)
    task = asyncio.create_task(_collect(_backend(factory, clock), first_audio_timeout=2.0))
    await clock.run_for(2.1)
    with pytest.raises(TTSUnavailable):
        await task


async def test_idle_timeout_after_audio_raises(mp3: bytes) -> None:
    clock = FakeClock()
    msgs = edge_messages(TEXT, mp3, marks_first=0)
    delays = [0.0] * len(msgs)
    delays[len(msgs) - 1] = 5.0  # the stream stalls before its last message
    factory = CommunicateFactory(lambda text: msgs, delays=delays, sleep=clock.sleep)
    got: list[Any] = []

    async def consume() -> None:
        async for item in _backend(factory, clock).synth(
            TEXT, VOICE, first_audio_timeout=2.0, idle_timeout=4.0
        ):
            got.append(item)

    task = asyncio.create_task(consume())
    await clock.run_for(4.1)
    with pytest.raises(TTSUnavailable, match="stalled"):
        await task
    assert any(isinstance(i, AudioChunk) for i in got)


@pytest.mark.parametrize("fail_after", [0, 4])
async def test_stream_errors_become_tts_unavailable(mp3: bytes, fail_after: int) -> None:
    class NoAudioReceived(Exception):
        pass

    factory = CommunicateFactory(
        lambda text: edge_messages(text, mp3), fail_after=fail_after, exc=NoAudioReceived("x")
    )
    with pytest.raises(TTSUnavailable, match="NoAudioReceived"):
        await _collect(_backend(factory))


async def test_no_audio_at_all_and_bad_messages_are_unavailable(mp3: bytes) -> None:
    only_words = CommunicateFactory(
        lambda text: [{"type": "WordBoundary", "offset": 0, "duration": 1, "text": "a"}]
    )
    with pytest.raises(TTSUnavailable, match="no audio"):
        await _collect(_backend(only_words))
    garbage = CommunicateFactory(lambda text: [{"type": "audio", "data": b"\x00" * 50}])
    with pytest.raises(TTSUnavailable):
        await _collect(_backend(garbage))

    def bad_factory(*a: Any, **k: Any) -> Any:
        raise ValueError("Invalid rate '+8'")

    with pytest.raises(TTSUnavailable, match="Invalid rate"):
        await _collect(EdgeTTSBackend(communicate_factory=bad_factory))


async def test_empty_text_yields_nothing_and_closed_backend_fails(mp3: bytes) -> None:
    factory = CommunicateFactory(lambda text: edge_messages(text, mp3))
    tts = _backend(factory)
    assert await _collect(tts, "   ") == []
    assert factory.made == []
    await tts.aclose()
    with pytest.raises(TTSUnavailable):
        await _collect(tts)


async def test_early_close_closes_the_edge_stream(mp3: bytes) -> None:
    factory = CommunicateFactory(lambda text: edge_messages(text, mp3))
    agen: Any = _backend(factory).synth(TEXT, VOICE, first_audio_timeout=2.0, idle_timeout=4.0)
    await agen.__anext__()
    await agen.aclose()
    assert factory.made[0].closed and factory.made[0].yielded < len(factory.made[0].messages)


def test_ssl_context_hook_from_ca_bundle_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import certifi
    import edge_tts.communicate as communicate

    bundle = tmp_path / "ca.pem"
    shutil.copy(certifi.where(), bundle)
    ctx = edge_ssl_context(bundle)
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED
    with pytest.raises(FileNotFoundError):
        edge_ssl_context(tmp_path / "missing.pem")

    monkeypatch.setattr(communicate, "_SSL_CTX", communicate._SSL_CTX)  # restored after the test
    tts = EdgeTTSBackend(ca_bundle=bundle)
    tts._prepare()
    assert communicate._SSL_CTX is tts._ssl_ctx and communicate._SSL_CTX is not None

    monkeypatch.setenv(CA_BUNDLE_ENV, str(bundle))
    from_env = EdgeTTSBackend()
    assert from_env._ssl_ctx is not None
    monkeypatch.delenv(CA_BUNDLE_ENV)
    assert EdgeTTSBackend()._ssl_ctx is None  # the user's PC: certifi default, untouched


async def test_warmup_imports_off_the_loop(mp3: bytes) -> None:
    tts = EdgeTTSBackend(communicate_factory=CommunicateFactory(lambda t: edge_messages(t, mp3)))
    await tts.warmup()
    assert tts._imported


def _suite_cases() -> list[Any]:
    mp3 = encode_mp3(vowel(1.0))
    good = lambda: _backend(CommunicateFactory(lambda text: edge_messages(text, mp3)))  # noqa: E731
    bad = lambda: _backend(  # noqa: E731
        CommunicateFactory(lambda text: edge_messages(text, mp3), fail_after=0)
    )
    return tts_backend_suite(good, unavailable_factory=bad)


@pytest.mark.parametrize("case", _suite_cases(), ids=case_id)
async def test_edge_backend_passes_the_tts_backend_suite(case: Any) -> None:
    await case()
