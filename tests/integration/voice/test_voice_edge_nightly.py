"""Nightly + network: real edge-tts (Premwadee) through the stateful PyAV decoder.

Allowed to be flaky (the endpoint is unofficial and slow from datacentres, §11 R1). Behind a
TLS-intercepting proxy set ``AIVTUBE_EDGE_CA_BUNDLE`` (or ``SSL_CERT_FILE``) to its CA bundle.
Prints the measured time to first audio.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, WordMark
from aivtube.voice.lipsync import LipSyncAnalyzer
from aivtube.voice.tts import CA_BUNDLE_ENV, EdgeTTSBackend

pytestmark = [pytest.mark.nightly, pytest.mark.network]

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz")
TEXT = "สวัสดีค่ะทุกคน วันนี้ไพลินจะมาเล่นเกมกันนะคะ"


def _ca_bundle() -> str | None:
    for name in (CA_BUNDLE_ENV, "SSL_CERT_FILE"):
        value = os.environ.get(name, "").strip()
        if value and os.path.isfile(value):
            return value
    return None


async def test_real_edge_tts_streams_thai_audio_and_word_marks() -> None:
    pytest.importorskip("edge_tts")
    pytest.importorskip("av")
    tts = EdgeTTSBackend(ca_bundle=_ca_bundle())
    await tts.warmup()
    t0 = time.perf_counter()
    ttfa: float | None = None
    audio: list[AudioChunk] = []
    marks: list[WordMark] = []
    async for item in tts.synth(TEXT, VOICE, first_audio_timeout=10.0, idle_timeout=10.0):
        if isinstance(item, AudioChunk):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            audio.append(item)
        else:
            marks.append(item)
    await tts.aclose()
    assert ttfa is not None
    pcm = np.concatenate([a.pcm for a in audio])
    seconds = pcm.size / audio[0].sample_rate
    print(f"\nedge TTFA {ttfa * 1000:.0f} ms, {seconds:.2f} s of audio, {len(marks)} word marks")
    assert pcm.dtype == np.int16 and audio[0].sample_rate == 24000
    assert 1.5 < seconds < 10.0 and int(np.abs(pcm).max()) > 1000
    assert marks and all(m.offset_s >= 0 for m in marks)
    assert [m.offset_s for m in marks] == sorted(m.offset_s for m in marks)
    assert "".join(m.text for m in marks).replace(" ", "")[:5] == TEXT[:5]
    mouth, _ = LipSyncAnalyzer().feed(pcm, 24000)
    assert float(mouth.max()) > 0.5  # real speech opens the mouth
