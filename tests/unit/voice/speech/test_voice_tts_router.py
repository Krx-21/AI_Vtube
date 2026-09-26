"""``TTSRouter``: substitution, breakers, quotas, captions fallback, cache and config (voice.tts)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.config import load_characters, load_config
from aivtube.contracts.speech import TTSConstraints
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, QuotaSpec, TTSUnavailable, WordMark
from aivtube.testing.fakes import FakeClock, FakePhraseCache, FakeTTS
from aivtube.voice.tts import (
    CAPTIONS,
    DiskPhraseCache,
    IdentityCfg,
    SynthStream,
    TTSRouter,
    build_tts_router,
    phrase_key,
    shift_rate,
)

ROOT = Path(__file__).resolve().parents[4]
VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz")
OFFLINE = VoiceSpec("offline", "models/tts/th_TH-tsync2-medium.onnx")
TEXT = "วันนี้ อากาศ ดีมาก เลย นะคะ"


class Rig:
    def __init__(self, **kw: Any) -> None:
        self.clock = FakeClock()
        self.edge = FakeTTS(name="edge", clock=self.clock, ttfa_s=0.2)
        self.azure = FakeTTS(name="azure", clock=self.clock, ttfa_s=0.3)
        self.piper = FakeTTS(name="piper", clock=self.clock, ttfa_s=0.1, normalizer="local")
        self.piper.whole_utterance = True  # type: ignore[attr-defined]
        self.fallbacks: list[tuple[str, str | None, str]] = []
        self.constraints: list[tuple[str, TTSConstraints]] = []
        identities = kw.pop(
            "identities",
            {
                "premwadee": IdentityCfg("premwadee", VOICE, ("edge", "azure")),
                "offline": IdentityCfg("offline", OFFLINE, ("piper",)),
            },
        )
        self.router = TTSRouter(
            identities,
            {"edge": self.edge, "azure": self.azure, "piper": self.piper},
            {"pailin": ["premwadee", "offline"]},
            clock=self.clock,
            on_fallback=lambda a, b, r: self.fallbacks.append((a, b, r)),
            on_constraints=lambda c, k: self.constraints.append((c, k)),
            min_chars_by_backend={"azure": 60},
            **kw,
        )

    async def run(self, stream: SynthStream, within: float = 60.0) -> list[AudioChunk | WordMark]:
        items: list[AudioChunk | WordMark] = []

        async def consume() -> None:
            async for item in stream:
                items.append(item)

        task = asyncio.create_task(consume())
        await self.clock.run_until(task.done, within=within, step=0.05)
        await task
        return items


def audio_of(items: list[AudioChunk | WordMark]) -> int:
    return sum(i.pcm.size for i in items if isinstance(i, AudioChunk))


async def test_first_audio_timeout_moves_to_the_next_backend_of_the_identity() -> None:
    rig = Rig()
    rig.edge.ttfa_s = 3.0  # slower than the 2 s first-segment timeout
    utt = rig.router.begin_utterance("pailin")
    assert utt.identity == "premwadee" and utt.voice == VOICE
    t0 = rig.clock.now()
    stream = utt.synth(TEXT, first=True)
    items = await rig.run(stream)
    assert stream.backend == "azure" and audio_of(items) > 0 and not stream.silent
    assert len(rig.edge.requests) == 1  # 0 retries on the first segment
    assert rig.clock.now() - t0 == pytest.approx(2.0 + 0.3, abs=0.06)
    assert rig.fallbacks and rig.fallbacks[0][:2] == ("edge", "azure")
    assert rig.azure.requests[0][1] == VOICE  # same voice: substitution within the identity


async def test_later_segments_get_4_s_and_one_retry() -> None:
    rig = Rig()
    rig.edge.ttfa_s = 5.0
    utt = rig.router.begin_utterance("pailin")
    t0 = rig.clock.now()
    stream = utt.synth(TEXT, first=False)
    await rig.run(stream)
    assert len(rig.edge.requests) == 2 and stream.backend == "azure"
    assert rig.clock.now() - t0 == pytest.approx(4.0 * 2 + 0.3, abs=0.1)


async def test_identity_exhausted_mid_utterance_goes_captions_and_next_utterance_switches() -> None:
    rig = Rig()
    utt = rig.router.begin_utterance("pailin")
    first = utt.synth("สวัสดีค่ะ", first=True)
    await rig.run(first)
    assert first.backend == "edge"
    rig.edge.fail_rate = rig.azure.fail_rate = 1.0
    second = utt.synth(TEXT, first=False)
    assert await rig.run(second) == []
    assert second.silent and second.captions and utt.captions_only and utt.degraded
    assert rig.fallbacks[-1][1] is None and "exhausted" in rig.fallbacks[-1][2]
    calls = (len(rig.edge.requests), len(rig.azure.requests))
    third = utt.synth("อีกประโยค", first=False)
    assert await rig.run(third) == [] and third.captions
    assert (len(rig.edge.requests), len(rig.azure.requests)) == calls  # no more tries
    assert rig.piper.requests == []  # never Piper (another identity) within the utterance
    # the next utterance uses the next identity in the chain
    nxt = rig.router.begin_utterance("pailin")
    assert nxt.identity == "offline"
    s = nxt.synth("สวัสดีค่ะ", first=True)
    await rig.run(s)
    assert s.backend == "piper"
    assert rig.router.constraints("pailin").identity == "offline"
    # after the demotion expires (and the backends recover) the preferred voice returns
    rig.edge.fail_rate = rig.azure.fail_rate = 0.0
    rig.clock.advance(121.0)
    assert rig.router.begin_utterance("pailin").identity == "premwadee"


async def test_whole_utterance_backend_never_substitutes_mid_utterance() -> None:
    rig = Rig(identities={"mixed": IdentityCfg("mixed", VOICE, ("edge", "piper"))})
    rig.router.chains["pailin"] = ["mixed"]
    utt = rig.router.begin_utterance("pailin")
    await rig.run(utt.synth("หนึ่ง", first=True))
    rig.edge.fail_rate = 1.0
    s = utt.synth("สอง", first=False)
    await rig.run(s)
    assert s.captions and rig.piper.requests == []


async def test_empty_bucket_skips_azure_for_the_segment_without_waiting() -> None:
    rig = Rig()
    rig.azure.quota = QuotaSpec(1, 60.0)
    rig.edge.fail_rate = 1.0
    rig.edge.ttfa_s = 0.1
    utt = rig.router.begin_utterance("pailin")
    s1 = utt.synth("หนึ่ง", first=True)
    await rig.run(s1)
    assert s1.backend == "azure"  # took the only token
    t0 = rig.clock.now()
    s2 = utt.synth("สอง", first=False)
    await rig.run(s2)
    assert len(rig.azure.requests) == 1  # empty bucket: skipped, not waited for
    assert rig.clock.now() - t0 < 1.0
    assert s2.captions


async def test_speculative_synthesis_never_uses_a_quota_backend() -> None:
    rig = Rig(identities={"az": IdentityCfg("az", VOICE, ("azure",))})
    rig.router.chains["pailin"] = ["az"]
    rig.azure.quota = QuotaSpec(18, 60.0)
    utt = rig.router.begin_utterance("pailin")
    s = utt.synth(TEXT, first=True, speculative=True)
    assert await rig.run(s) == []
    assert s.deferred and s.silent and not s.captions and not utt.captions_only
    assert rig.azure.requests == []
    again = utt.synth(TEXT, first=True)  # after the gate opens
    await rig.run(again)
    assert again.backend == "azure"


async def test_three_failures_in_60_s_open_the_breaker_for_120_s() -> None:
    rig = Rig()
    rig.edge.fail_rate = 1.0
    rig.edge.ttfa_s = 0.1
    utt = rig.router.begin_utterance("pailin")
    s1 = utt.synth("หนึ่ง", first=False)  # 2 attempts
    await rig.run(s1)
    assert not rig.router.breaker("edge").is_open()
    s2 = utt.synth("สอง", first=False)  # 3rd failure opens the breaker, no 4th attempt
    await rig.run(s2)
    assert rig.router.breaker("edge").is_open() and len(rig.edge.requests) == 3
    assert s2.backend == "azure"
    c = rig.router.constraints("pailin")
    assert (c.backend, c.identity, c.min_chars) == ("azure", "premwadee", 60)
    assert rig.constraints[-1][1].backend == "azure"  # on_constraints reported the change
    s3 = utt.synth("สาม", first=False)
    await rig.run(s3)
    assert len(rig.edge.requests) == 3 and s3.backend == "azure"
    rig.clock.advance(119.0)
    assert rig.router.breaker("edge").is_open()
    rig.clock.advance(2.0)
    # half-open: a single failed trial re-opens it at once
    s4 = utt.synth("สี่", first=False)
    await rig.run(s4)
    assert len(rig.edge.requests) == 4 and rig.router.breaker("edge").is_open()
    rig.edge.fail_rate = 0.0
    rig.clock.advance(121.0)
    s5 = utt.synth("ห้า", first=False)
    await rig.run(s5)
    assert s5.backend == "edge" and not rig.router.breaker("edge").is_open()
    assert rig.router.constraints("pailin").backend == "edge"


async def test_marks_of_a_failed_attempt_are_dropped_and_mid_stream_death_truncates() -> None:
    class MarksThenFail(FakeTTS):
        async def synth(self, text: str, voice: VoiceSpec, **kw: float) -> AsyncIterator[Any]:  # type: ignore[override]
            self.requests.append((text, voice))
            yield WordMark("ghost", 0.0, 0.1)
            raise TTSUnavailable("no audio after marks")

    class AudioThenFail(FakeTTS):
        async def synth(self, text: str, voice: VoiceSpec, **kw: float) -> AsyncIterator[Any]:  # type: ignore[override]
            self.requests.append((text, voice))
            yield WordMark("real", 0.0, 0.1)
            yield AudioChunk(np.zeros(2400, np.int16), 24000)
            raise TTSUnavailable("socket closed")

    rig = Rig()
    rig.router.backends["edge"] = MarksThenFail(name="edge")
    utt = rig.router.begin_utterance("pailin")
    s = utt.synth(TEXT, first=True)
    items = await rig.run(s)
    marks = [i.text for i in items if isinstance(i, WordMark)]
    assert "ghost" not in marks and marks == TEXT.split(" ") and s.backend == "azure"

    rig.router.backends["edge"] = AudioThenFail(name="edge")
    utt2 = rig.router.begin_utterance("pailin")
    s2 = utt2.synth(TEXT, first=True)
    items2 = await rig.run(s2)
    assert s2.truncated and s2.backend == "edge" and audio_of(items2) == 2400
    assert len(rig.azure.requests) == 1  # audio cannot be spliced from another backend


async def test_router_guard_catches_a_backend_that_ignores_its_timeout() -> None:
    class Hangs(FakeTTS):
        async def synth(self, text: str, voice: VoiceSpec, **kw: float) -> AsyncIterator[Any]:  # type: ignore[override]
            self.requests.append((text, voice))
            await self._sleep(100.0)
            yield AudioChunk(np.zeros(10, np.int16), 24000)

    rig = Rig()
    rig.router.backends["edge"] = Hangs(name="edge", clock=rig.clock)
    t0 = rig.clock.now()
    s = rig.router.begin_utterance("pailin").synth(TEXT, first=True)
    await rig.run(s)
    assert s.backend == "azure"
    assert rig.clock.now() - t0 == pytest.approx(2.0 + 0.5 + 0.3, abs=0.1)


async def test_phrase_cache_hit_needs_no_backend(tmp_path: Path) -> None:
    cache = DiskPhraseCache(tmp_path)
    rig = Rig(cache=cache)
    cache.put(
        phrase_key(VOICE, "Filtered."),
        AudioChunk(np.ones(4800, np.int16), 24000),
        [WordMark("Filtered.", 0.0, 0.2)],
    )
    utt = rig.router.begin_utterance("pailin")
    s = utt.synth("Filtered.", first=True)
    items = await rig.run(s)
    assert s.backend == "cache" and s.cached and audio_of(items) == 4800
    assert isinstance(items[0], WordMark)
    assert rig.edge.requests == [] and rig.azure.requests == []
    hit = rig.router.cached_phrase("pailin", "Filtered.")
    assert hit is not None and hit[1][0].text == "Filtered."
    assert rig.router.cached_phrase("pailin", "ไม่มี") is None


async def test_presynthesize_fills_the_cache_once() -> None:
    cache = FakePhraseCache()
    rig = Rig(cache=cache)
    phrases = ["Filtered.", "อืม…", "  "]
    task = asyncio.create_task(rig.router.presynthesize("pailin", phrases))
    await rig.clock.run_until(task.done, within=10.0)
    assert await task == {"Filtered.": True, "อืม…": True}
    assert len(rig.edge.requests) == 2
    task = asyncio.create_task(rig.router.presynthesize("pailin", phrases))
    await rig.clock.run_until(task.done, within=10.0)
    assert await task == {"Filtered.": True, "อืม…": True}
    assert len(rig.edge.requests) == 2  # cached now
    rig.edge.fail_rate = rig.azure.fail_rate = 1.0
    task = asyncio.create_task(rig.router.presynthesize("pailin", ["ใหม่"]))
    await rig.clock.run_until(task.done, within=30.0)
    assert await task == {"ใหม่": False}
    assert rig.router.begin_utterance("pailin").identity == "premwadee"  # not demoted


async def test_talking_speed_applies_from_the_next_utterance() -> None:
    rig = Rig()
    utt = rig.router.begin_utterance("pailin")
    rig.router.set_rate("pailin", 10)
    await rig.run(utt.synth("ก", first=True))
    assert rig.edge.requests[-1][1].rate == "+8%"
    nxt = rig.router.begin_utterance("pailin")
    await rig.run(nxt.synth("ข", first=True))
    assert rig.edge.requests[-1][1].rate == "+18%"
    assert shift_rate("+8%", -20) == "-12%" and shift_rate("bogus", 5) == "+5%"


async def test_no_identity_available_means_captions_only() -> None:
    rig = Rig()
    for name in ("edge", "azure", "piper"):
        rig.router.breaker(name).open_until = rig.clock.now() + 100.0
    utt = rig.router.begin_utterance("pailin")
    assert utt.captions_only and utt.identity == CAPTIONS
    s = utt.synth(TEXT, first=True)
    assert await rig.run(s) == [] and s.captions
    c = rig.router.constraints("pailin")
    assert (c.backend, c.identity) == (CAPTIONS, CAPTIONS)
    assert rig.router.status()[0]["open"] is True


async def test_warmup_and_close_reach_every_backend() -> None:
    rig = Rig()
    assert await rig.router.warmup() == {"edge": True, "azure": True, "piper": True}
    assert rig.edge.warmed
    await rig.router.aclose()
    assert rig.edge.closed and rig.azure.closed


def test_build_from_the_real_config(tmp_path: Path) -> None:
    cfg = load_config(ROOT, profile="stream", env={})
    chars = load_characters(cfg)
    chains = {cid: cfg.tts_chain_for(ch) for cid, ch in chars.items()}
    router = build_tts_router(cfg.tts.model_dump(), chains, root=tmp_path, secrets=lambda _k: None)
    assert set(router.identities) == {"premwadee", "offline"}
    assert set(router.backends) == {"edge"}  # no Azure key; Piper is not built in M1
    assert router.identities["premwadee"].voice == cfg.tts.identities["premwadee"].voice_spec(
        "premwadee"
    )
    c = router.constraints("pailin")
    assert (c.first_min_chars, c.min_chars, c.max_chars) == (8, 40, 160)
    assert (c.backend, c.identity) == ("edge", "premwadee")
    assert isinstance(router.cache, DiskPhraseCache)
    assert router.cache.root == tmp_path / "data" / "cache" / "tts"

    secrets = {"AZURE_SPEECH_KEY": "k", "AZURE_SPEECH_REGION": "southeastasia"}
    with_azure = build_tts_router(cfg.tts.model_dump(), chains, root=tmp_path, secrets=secrets.get)
    assert set(with_azure.backends) == {"edge", "azure"}
    assert with_azure.backends["azure"].quota == QuotaSpec(18, 60.0)
    with_azure.breaker("edge").open_until = with_azure.clock.now() + 60
    assert with_azure.constraints("pailin").min_chars == 60  # Azure F0 primary


def test_build_with_injected_backends() -> None:
    fake = FakeTTS(name="fake")
    router = build_tts_router(
        {
            "identities": {"v": {"voice": "x", "backends": ["fake"]}},
            "backends": {"fake": {"kind": "fake"}},
            "chunk": {"first_min_chars": 10, "min_chars": 50, "max_chars": 150},
        },
        {"pailin": ["v"]},
        root=Path("."),
        secrets=lambda _k: None,
        backends={"fake": fake},
    )
    assert router.backends["fake"] is fake
    c = router.constraints("pailin")
    assert (c.first_min_chars, c.min_chars, c.max_chars, c.backend) == (10, 50, 150, "fake")
