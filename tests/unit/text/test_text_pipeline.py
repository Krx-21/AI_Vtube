"""The §4.6 text path on real and scripted LLM streams: EmotionTagExtractor → ThaiSpeechChunker
→ is_speakable (merge forward) → normalize_cloud / normalize_local."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aivtube.config.schema import EMOTIONS
from aivtube.contracts.llm import ChatRequest, TextDelta
from aivtube.testing.fakes import FakeClock, FakeEventBus, FakeLLM, FakeReply, FakeSpeechOutput
from aivtube.text import (
    ChunkerConfig,
    EmotionTagExtractor,
    ThaiSpeechChunker,
    is_speakable,
    normalize_cloud,
    normalize_local,
    thai,
    warm_up_pythainlp,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warm_up_pythainlp()


def sse_deltas(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            text = (choice.get("delta") or {}).get("content")
            if text:
                out.append(text)
    return out


@dataclass
class TextPath:
    """A minimal ReplyPipeline text path (the brain owns the real one)."""

    chunker: ThaiSpeechChunker
    tags: EmotionTagExtractor = field(default_factory=lambda: EmotionTagExtractor(EMOTIONS))
    segments: list[tuple[str, str | None]] = field(default_factory=list)  # (chunk, emotion)
    carry: str = ""
    emotion: str | None = None

    def feed(self, delta: str) -> None:
        for text, emotion in self.tags.feed(delta):
            if emotion is not None:
                self.emotion = emotion
            self._take(self.chunker.feed(text))

    def poll(self, now: float) -> None:
        self._take(self.chunker.poll(now))

    def flush(self) -> None:
        for text, _ in self.tags.flush():
            self._take(self.chunker.feed(text))
        self._take(self.chunker.flush())

    def _take(self, chunks: list[str]) -> None:
        for chunk in chunks:
            chunk = self.carry + chunk
            if not is_speakable(chunk):  # merge forward
                self.carry = chunk
                continue
            self.carry = ""
            self.segments.append((chunk, self.emotion))


def check_chunks(stream_text: str, chunks: list[str], cfg: ChunkerConfig) -> None:
    assert "".join(chunks) == stream_text.strip()
    pos = len(stream_text) - len(stream_text.lstrip())
    for idx, chunk in enumerate(chunks):
        assert len(chunk) <= (cfg.first_max_chars if idx == 0 else cfg.max_chars)
        pos += len(chunk)
        if idx < len(chunks) - 1:
            assert thai.is_safe_cut(stream_text, pos), chunks
            assert chunk.rstrip()[-1] not in thai.LEADING_VOWELS


def test_recorded_llama_server_stream(fake_clock: FakeClock, fake_bus: FakeEventBus) -> None:
    """Typhoon 2.5 4B deltas from llama-server, combining marks in separate deltas."""
    deltas = sse_deltas(FIXTURES / "sse" / "llamacpp_text.sse")
    assert any(len(d) == 1 and d in thai.THAI_COMBINING for d in deltas)
    speech = FakeSpeechOutput(fake_bus, fake_clock)
    cfg = ChunkerConfig.from_constraints(speech.constraints("pailin"))
    assert cfg == ChunkerConfig()  # the fake worker reports the default 8 / 40 / 160
    path = TextPath(ThaiSpeechChunker(cfg, clock=fake_clock))
    for d in deltas:
        path.feed(d)
    path.flush()
    text = "".join(deltas)
    chunks = [c for c, _ in path.segments]
    check_chunks(text, chunks, cfg)
    assert chunks == ["เฮ้ยยย ไพลินดี๊ะ ", "เย้ยยย วันนี้สบายดีจ้า ", "สบาย"]
    assert [normalize_cloud(c) for c in chunks] == ["เฮ้ย ไพลินดี๊ะ", "เย้ย วันนี้สบายดีจ้า", "สบาย"]
    at_once = ThaiSpeechChunker(cfg)
    assert at_once.feed(text) + at_once.flush() == chunks  # the same cuts as the real deltas


REPLY = (
    "[happy] ว้าว มาแล้วเหรอคะ ขอบคุณคุณต้นกล้าที่โดเนทมา 100 บาทนะคะ ใจดีสุดๆ ไปเลย 😂 "
    "วันนี้ไพลินจะเล่น Minecraft Java Edition กันนะคะ [surprised] เดี๋ยวนะ ใครบอกว่าไพลิน"
    "เล่นเกมไม่เก่ง เมื่อวานชนะตั้งสามตาติดเลย 555"
)


async def test_fake_llm_stream_with_a_stall(fake_clock: FakeClock) -> None:
    """Tags split across FakeLLM deltas; the stall flush fires 500 ms after the last delta."""
    stall_after = 40
    llm = FakeLLM(
        [FakeReply("", REPLY, stall_after=stall_after, stall_s=1.5)],
        ttft_s=0.3,
        tok_s=40.0,
        clock=fake_clock,
    )
    path = TextPath(ThaiSpeechChunker(clock=fake_clock))
    fed: list[str] = []
    stall_emits: list[tuple[float, int]] = []
    done = asyncio.Event()

    async def consume() -> None:
        async for event in llm.stream(ChatRequest(messages=({"role": "user", "content": "hi"},))):
            if isinstance(event, TextDelta):
                fed.append(event.text)
                path.feed(event.text)
        path.flush()
        done.set()

    async def poll() -> None:
        while not done.is_set():
            await fake_clock.sleep(0.05)
            before = len(path.segments)
            path.poll(fake_clock.now())
            if len(path.segments) > before:
                stall_emits.append((fake_clock.now(), len(path.segments) - before))

    t0 = fake_clock.now()
    tasks = [asyncio.create_task(consume()), asyncio.create_task(poll())]
    await fake_clock.run_until(done.is_set, within=30.0)
    await fake_clock.run_for(0.1)  # let the poller see ``done``
    await asyncio.gather(*tasks)

    assert any("[" in d and "]" not in d for d in fed)  # a tag really was split
    stream_text = "".join(fed)
    chunks = [c for c, _ in path.segments]
    visible = "".join(t for t, _ in _extract(stream_text))
    check_chunks(visible, chunks, ChunkerConfig())
    assert all("[" not in c and "]" not in c for c in chunks)
    # Exactly one stall flush, and it happened during the 1.5 s stall.
    assert len(stall_emits) == 1
    t_last_delta = 0.3 + (stall_after - 1) / 40.0
    assert t_last_delta + 0.5 <= stall_emits[0][0] - t0 <= t_last_delta + 0.5 + 0.05 + 1e-6
    # Emotions: happy from the start, surprised from the segment after the tag.
    emotions = [e for _, e in path.segments]
    assert emotions[0] == "happy"
    assert emotions[-1] == "surprised"
    for chunk, _ in path.segments:
        assert is_speakable(chunk)
        assert normalize_cloud(chunk)
        assert normalize_local(chunk, {"Minecraft": "มายคราฟ"})


def _extract(text: str) -> list[tuple[str, str | None]]:
    x = EmotionTagExtractor(EMOTIONS)
    return x.feed(text) + x.flush()


def test_unspeakable_chunks_merge_forward() -> None:
    cfg = ChunkerConfig(first_min_chars=1, first_max_chars=60, min_chars=1, strong_min_chars=1)
    path = TextPath(ThaiSpeechChunker(cfg, word_tokenize=thai.newmm))
    path.feed("😂😂\nจริงเหรอ ไม่น่าเชื่อ")
    path.flush()
    first, _ = path.segments[0]
    assert first.startswith("😂😂\n")  # the emoji-only piece was carried into the next one
