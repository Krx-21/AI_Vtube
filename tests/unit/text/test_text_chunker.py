"""ThaiSpeechChunker unit tests (modules.json text.chunker; ARCHITECTURE.md §4.6, §10)."""

from __future__ import annotations

import asyncio
import random
import time

import pytest

from aivtube.contracts.speech import TTSConstraints
from aivtube.testing.fakes import FakeClock
from aivtube.text import thai
from aivtube.text.chunker import (
    LOOKAHEAD,
    ChunkerConfig,
    ThaiSpeechChunker,
    no_word_split,
    warm_up_pythainlp,
)

UNSPACED = (
    "ไพลินชอบกินข้าวมันไก่มากที่สุดในโลกเลยค่ะแต่วันนี้ไม่ได้กินเพราะร้านปิดก็เลยต้องกิน"
    "บะหมี่กึ่งสำเร็จรูปแทนซึ่งก็อร่อยดีเหมือนกันนะแต่ว่ามันไม่อิ่มเท่าไหร่เลยต้องสั่ง"
    "พิซซ่ามาเพิ่มอีกถาดหนึ่งแล้วก็นั่งดูหนังต่อจนถึงตีสอง"
)
SPACED_NO_PUNCT = (
    "วันนี้อากาศดีมากเลย ไพลินเลยอยากออกไปเดินเล่นที่สวนสาธารณะแถวบ้าน "
    "แต่ว่าแดดร้อนมากจนต้องกลับมานั่งเล่นเกมในห้องแอร์แทน "
    "ใครมีเกมสนุกๆ แนะนำไพลินบ้าง พิมพ์บอกในแชทได้เลยนะ"
)
MIXED = (
    "สวัสดีค่ะทุกคน วันนี้ไพลินจะมาเล่นเกม Minecraft Java Edition กันนะคะ "
    "ขอบคุณคุณ John_Doe สำหรับ 1,250.50 บาท ใจดีมากเลย!!! "
    "เมื่อวานไพลินนอนดึกมากเพราะดูหนังผี ตอนนั้นเวลา 02:30 น. แล้ว "
    "ใครเคยเป็นแบบนี้บ้างคะ พิมพ์บอกในแชทได้เลย 555\n"
    "ปี ค.ศ. 2026 นี้ไพลินอยากไปญี่ปุ่นมากที่สุดในโลกเลยค่ะ"
)
#: LLM-like replies in Pailin's style (tags already stripped by EmotionTagExtractor).
LLM_REPLIES = (
    "ว้าว มาแล้วเหรอ คิดถึงน้า วันนี้เล่นอะไรกันดีคะ",
    "ขอบคุณคุณต้นกล้าที่โดเนทมา 100 บาทนะคะ ใจดีสุดๆ ไปเลย ไพลินจะเอาไปซื้อขนมกินนะ ฮ่าๆ",
    "อ๋อ เกมนี้ไพลินเคยเล่นค่ะ ตอนแรกก็งงๆ อยู่เหมือนกัน แต่พอเล่นไปสักพักก็สนุกดีนะ "
    "โดยเฉพาะตอนที่ต้องหนีซอมบี้ตอนกลางคืน ตื่นเต้นมากเลย",
    "เดี๋ยวนะ ใครบอกว่าไพลินเล่นเกมไม่เก่ง! เมื่อวานไพลินชนะ ROV ตั้งสามตาติดเลยนะ แค่วันนี้มือไม่ค่อยดีเท่านั้นเอง 555",
    "Version 1.21.4 ออกแล้วนะคะ ราคา $29.99 เหมือนเดิม แต่ถ้าซื้อก่อนเวลา 23:59 "
    "จะลด 50% เหลือแค่ 14.99 ดอลลาร์เอง คุ้มมากๆ",
    "ไพลินว่านะ การนอนดึกไม่ดีต่อสุขภาพเลย ทุกคนพักผ่อนให้เพียงพอด้วยนะคะ\nพรุ่งนี้เจอกันใหม่ตอนสองทุ่มค่ะ บ๊ายบาย",
    "Hello everyone! Today we are going to play Minecraft together. "
    "Version 1.21.4 is out, and it costs $29.99. Isn't that great?",
)


@pytest.fixture(scope="module")
def warm() -> None:
    warm_up_pythainlp()


def run(
    text: str,
    *,
    seed: int | None = None,
    chunker: ThaiSpeechChunker | None = None,
    max_delta: int = 7,
) -> list[str]:
    """Feed ``text`` in random-sized deltas (or all at once when ``seed`` is None)."""
    c = chunker or ThaiSpeechChunker()
    out: list[str] = []
    if seed is None:
        out += c.feed(text)
    else:
        rnd = random.Random(seed)
        pos = 0
        while pos < len(text):
            n = rnd.randint(1, max_delta)
            out += c.feed(text[pos : pos + n])
            pos += n
    return out + c.flush()


def cut_offsets(text: str, chunks: list[str]) -> list[int]:
    """Positions in ``text`` where one chunk ends and the next begins."""
    pos = len(text) - len(text.lstrip())
    offsets = []
    for chunk in chunks[:-1]:
        pos += len(chunk)
        offsets.append(pos)
    return offsets


def assert_well_formed(text: str, chunks: list[str], cfg: ChunkerConfig | None = None) -> None:
    cfg = cfg or ChunkerConfig()
    assert "".join(chunks) == text.strip()
    for idx, chunk in enumerate(chunks):
        assert chunk.strip(), chunks
        limit = cfg.first_max_chars if idx == 0 else cfg.max_chars
        assert len(chunk) <= limit, (idx, len(chunk), chunk)
        if idx < len(chunks) - 1:  # a cut, not the end of the stream
            assert chunk.rstrip()[-1] not in thai.LEADING_VOWELS, chunk
    for off in cut_offsets(text, chunks):
        assert thai.is_safe_cut(text, off), (off, text[off - 5 : off + 5])


# --- basic behaviour -----------------------------------------------------------------------


def test_first_chunk_is_short_then_later_chunks_are_longer(warm: None) -> None:
    chunks = run(MIXED)
    assert_well_formed(MIXED, chunks)
    assert chunks[0] == "สวัสดีค่ะทุกคน "
    assert 8 <= len(chunks[0].strip()) <= 60
    for chunk in chunks[1:-1]:
        assert len(chunk.strip()) >= 20  # later chunks: punctuation/particle cuts from 20
        assert len(chunk) <= 160


def test_mixed_sample_golden(warm: None) -> None:
    assert run(MIXED) == [
        "สวัสดีค่ะทุกคน ",
        "วันนี้ไพลินจะมาเล่นเกม Minecraft Java Edition กันนะคะ ",
        "ขอบคุณคุณ John_Doe สำหรับ 1,250.50 บาท ใจดีมากเลย!!! ",
        "เมื่อวานไพลินนอนดึกมากเพราะดูหนังผี ตอนนั้นเวลา 02:30 น. ",
        "แล้ว ใครเคยเป็นแบบนี้บ้างคะ ",
        "พิมพ์บอกในแชทได้เลย 555\n",
        "ปี ค.ศ. 2026 นี้ไพลินอยากไปญี่ปุ่นมากที่สุดในโลกเลยค่ะ",
    ]


@pytest.mark.parametrize("seed", range(25))
def test_split_invariant_and_lossless_on_samples(seed: int, warm: None) -> None:
    for text in (MIXED, UNSPACED, SPACED_NO_PUNCT, *LLM_REPLIES):
        whole = run(text)
        assert run(text, seed=seed) == whole
        assert run(text, seed=seed, max_delta=2) == whole
        assert_well_formed(text, whole)


@pytest.mark.parametrize("text", LLM_REPLIES)
def test_llm_like_replies(text: str, warm: None) -> None:
    chunks = run(text, seed=1)
    assert_well_formed(text, chunks)
    assert len(chunks[0].strip()) >= 8


def test_llm_reply_chunks_golden(warm: None) -> None:
    assert run(LLM_REPLIES[1]) == [
        "ขอบคุณคุณต้นกล้าที่โดเนทมา 100 บาทนะคะ ",
        "ใจดีสุดๆ ไปเลย ไพลินจะเอาไปซื้อขนมกินนะ ฮ่าๆ",
    ]
    assert run(LLM_REPLIES[6]) == [
        "Hello everyone! ",
        "Today we are going to play Minecraft together. ",
        "Version 1.21.4 is out, and it costs $29.99. ",
        "Isn't that great?",
    ]


# --- never split ---------------------------------------------------------------------------

PROTECTED = ("100 บาท", "เวลา 02:30", "Minecraft Java Edition", "กันนะคะ", "3.5", "$29.99")


@pytest.mark.parametrize(
    "cfg",
    [
        ChunkerConfig(),
        ChunkerConfig(first_min_chars=4, first_max_chars=20, min_chars=8, max_chars=40,
                      strong_min_chars=4),
        ChunkerConfig(first_min_chars=1, first_max_chars=30, min_chars=1, max_chars=30,
                      strong_min_chars=1),
    ],
    ids=["default", "small", "eager"],
)  # fmt: skip
def test_protected_phrases_are_never_split(cfg: ChunkerConfig, warm: None) -> None:
    text = (
        "ของชิ้นนี้ราคา 100 บาท ตอนนี้เวลา 02:30 แล้ว ไพลินจะเล่น Minecraft Java Edition "
        "กันนะคะ คะแนน 3.5 ดาว ส่วนอีกเกมราคา $29.99 เองค่ะ ไปเล่นกันนะคะ"
    )
    for seed in range(15):
        chunks = run(text, seed=seed, chunker=ThaiSpeechChunker(cfg))
        assert_well_formed(text, chunks, cfg)
        offsets = cut_offsets(text, chunks)
        for phrase in PROTECTED:
            start = text.find(phrase)
            while start != -1:
                end = start + len(phrase)
                assert not any(start < off < end for off in offsets), (phrase, chunks)
                start = text.find(phrase, end)


def test_attach_left_particles_stay_with_the_clause(warm: None) -> None:
    cfg = ChunkerConfig(first_min_chars=3, first_max_chars=60, min_chars=3, strong_min_chars=3)
    chunks = run("ไปเล่นเกม Minecraft กันนะคะ แล้วก็ดูหนัง ด้วย ดีไหม", chunker=ThaiSpeechChunker(cfg))
    assert chunks[0] == "ไปเล่นเกม "
    assert "Minecraft กันนะคะ " in chunks
    assert all(not c.startswith(("กัน", "ด้วย", "ๆ")) for c in chunks)


def test_ambiguous_particle_prefixes_do_not_block_real_words(warm: None) -> None:
    cfg = ChunkerConfig(first_min_chars=3, min_chars=3, strong_min_chars=3)
    # คะแนน (score) and อะไร (what) start like the particles คะ / อะ but are words.
    chunks = run("ได้เต็มสิบเลย คะแนนดีมาก อะไรก็ได้ค่ะ", chunker=ThaiSpeechChunker(cfg))
    assert chunks == ["ได้เต็มสิบเลย ", "คะแนนดีมาก ", "อะไรก็ได้ค่ะ"]


def test_never_cuts_before_combining_mark_or_after_leading_vowel() -> None:
    cfg = ChunkerConfig(first_min_chars=1, first_max_chars=7, min_chars=1, max_chars=7,
                        strong_min_chars=1)  # fmt: skip
    text = "เที่ยวแม่น้ำโขงไม่ได้ไปไหนเลยเพราะว่าแดดแรงเกินไป"
    for seed in range(10):
        chunks = run(text, seed=seed, chunker=ThaiSpeechChunker(cfg, word_tokenize=no_word_split))
        assert_well_formed(text, chunks, cfg)


def test_newline_and_punctuation_are_preferred(warm: None) -> None:
    cfg = ChunkerConfig(first_min_chars=5)
    chunks = run("อืม! ไพลินคิดว่า\nน่าจะใช่นะ... แต่ไม่แน่ใจ", chunker=ThaiSpeechChunker(cfg))
    assert chunks == ["อืม! ไพลินคิดว่า\n", "น่าจะใช่นะ... แต่ไม่แน่ใจ"]


def test_short_hard_punct_before_first_min_does_not_cut() -> None:
    assert run("อ๋อ! เข้าใจแล้วค่ะ ขอบคุณนะ") == ["อ๋อ! เข้าใจแล้วค่ะ ", "ขอบคุณนะ"]


def test_decimal_abbreviation_and_title_dots_do_not_cut() -> None:
    cfg = ChunkerConfig(first_min_chars=1, min_chars=1, strong_min_chars=1)
    text = "ค่า pi คือ 3.14 นะ ดร. สมชาย บอกว่า e.g. แบบนี้ ค.ศ. 2026 จบ. ต่อไป"
    offsets = cut_offsets(text, run(text, chunker=ThaiSpeechChunker(cfg)))
    for phrase in ("3.14", "ดร. สมชาย", "e.g.", "ค.ศ. 2026"):
        start = text.index(phrase)
        assert not any(start < off < start + len(phrase) for off in offsets), phrase
    assert text.index("ต่อไป") in offsets  # "จบ. " is a real sentence end


def test_laughter_attaches_to_the_previous_clause() -> None:
    cfg = ChunkerConfig(first_min_chars=3, min_chars=3, strong_min_chars=3)
    chunks = run("ตลกมาก 5555 แล้วก็ ฮ่าๆ ไปต่อเลย", chunker=ThaiSpeechChunker(cfg))
    assert chunks == ["ตลกมาก 5555 ", "แล้วก็ ฮ่าๆ ", "ไปต่อเลย"]


def test_emoji_stays_with_the_clause_before_it() -> None:
    cfg = ChunkerConfig(first_min_chars=3)
    chunks = run("ดีใจมากเลย 😂 ขอบคุณทุกคนนะ", chunker=ThaiSpeechChunker(cfg))
    assert chunks == ["ดีใจมากเลย 😂 ", "ขอบคุณทุกคนนะ"]


# --- long runs -----------------------------------------------------------------------------


def test_150_char_unpunctuated_thai_stream_gives_at_least_3_chunks(warm: None) -> None:
    text = UNSPACED[:150]
    for seed in (None, 0, 1, 2):
        chunks = run(text, seed=seed)
        assert len(chunks) >= 3, chunks
        assert_well_formed(text, chunks)
    spaced = SPACED_NO_PUNCT[:150]
    assert len(run(spaced)) >= 3
    assert len(run(spaced, chunker=ThaiSpeechChunker(word_tokenize=no_word_split))) >= 3


def test_unspaced_run_is_cut_at_word_boundaries(warm: None) -> None:
    chunks = run(UNSPACED)
    assert_well_formed(UNSPACED, chunks)
    words = set(thai.newmm(UNSPACED))
    cuts = cut_offsets(UNSPACED, chunks)
    bounds = set()
    pos = 0
    for w in thai.newmm(UNSPACED):
        pos += len(w)
        bounds.add(pos)
    assert set(cuts) <= bounds, (chunks, words)
    assert all(len(c) <= ChunkerConfig().word_split_chars for c in chunks[1:])


def test_without_tokenizer_forced_cut_is_syllable_safe() -> None:
    chunks = run(UNSPACED, chunker=ThaiSpeechChunker(word_tokenize=no_word_split))
    assert_well_formed(UNSPACED, chunks)
    assert len(chunks[0]) <= 60
    assert all(len(c) <= 160 for c in chunks)


def test_long_english_without_punctuation_cuts_between_words() -> None:
    text = " ".join(["streaming", "minecraft", "together", "tonight"] * 12)
    chunks = run(text, chunker=ThaiSpeechChunker(word_tokenize=no_word_split))
    assert_well_formed(text, chunks)
    for chunk in chunks[:-1]:
        assert chunk.endswith(" "), chunk  # only ever between two words


def test_tokenizer_failure_falls_back_to_safe_cuts() -> None:
    def broken(text: str) -> list[str]:
        raise RuntimeError("boom")

    chunks = run(UNSPACED, chunker=ThaiSpeechChunker(word_tokenize=broken))
    assert_well_formed(UNSPACED, chunks)


def test_non_lossless_tokenizer_is_ignored() -> None:
    chunks = run(UNSPACED, chunker=ThaiSpeechChunker(word_tokenize=lambda t: t.split()))
    assert_well_formed(UNSPACED, chunks)


def test_whitespace_edges_and_empty_input() -> None:
    c = ThaiSpeechChunker()
    assert c.feed("") == []
    assert c.feed("   \n ") == []
    assert c.flush() == []
    assert run("   สวัสดีค่ะ   ") == ["สวัสดีค่ะ"]
    huge_gap = "สวัสดีค่ะทุกคน" + " " * 400 + "วันนี้ไพลินมาแล้วนะ"
    chunks = run(huge_gap, seed=3)
    assert " ".join("".join(chunks).split()) == " ".join(huge_gap.split())
    assert all(len(ch) <= 160 for ch in chunks)


def test_flush_and_reset_start_a_new_stream() -> None:
    c = ThaiSpeechChunker()
    assert c.feed("สวัสดีค่ะทุกคน วันนี้") == ["สวัสดีค่ะทุกคน "]
    assert c.emitted == 1
    assert c.pending == "วันนี้"
    c.reset()
    assert c.pending == "" and c.emitted == 0
    assert c.feed("ขอบคุณมากค่ะ ทุกคน") == ["ขอบคุณมากค่ะ "]  # a "first" chunk again
    assert c.flush() == ["ทุกคน"]
    assert c.emitted == 0
    assert c.feed("ใช่แล้วค่ะ ต่อ") == ["ใช่แล้วค่ะ "]


def test_waits_for_right_context_before_deciding() -> None:
    c = ThaiSpeechChunker()
    # "... Edition " might be followed by an attach-left particle: no decision yet.
    assert c.feed("วันนี้ไพลินจะเล่นเกม Minecraft ") == ["วันนี้ไพลินจะเล่นเกม "]
    assert c.feed("กั") == []
    assert c.feed("นนะคะ ต") == []
    assert c.flush() == ["Minecraft กันนะคะ ต"]


# --- stall flush ---------------------------------------------------------------------------


def test_stall_flush_cuts_at_the_last_space_never_mid_word() -> None:
    c = ThaiSpeechChunker()
    assert c.feed("สวัสดีค่ะ") == []
    assert c.feed("ทุกคน") == []
    assert c.feed("ๆ") == []
    # After the first chunk, clause spaces below min_chars (40) do not cut on their own.
    assert c.feed(" วันนี้อากาศดีมาก ไพลินอยากออกไป ข้างนอกแต่ฝนต") == ["สวัสดีค่ะทุกคนๆ "]
    assert c.stall_flush() == ["วันนี้อากาศดีมาก ไพลินอยากออกไป "]
    assert c.pending == "ข้างนอกแต่ฝนต"
    assert c.stall_flush() == []  # no boundary left: never cuts inside ข้างนอกแต่ฝนต
    assert c.flush() == ["ข้างนอกแต่ฝนต"]


def test_stall_flush_respects_protected_spaces_and_min_size() -> None:
    c = ThaiSpeechChunker()
    assert c.feed("ของชิ้นนี้ราคา 100 บา") == []
    assert c.stall_flush() == []  # the only spaces are inside "ราคา 100 บาท"
    c.reset()
    assert c.feed("อืม ใช่") == []
    assert c.stall_flush() == []  # "อืม " is shorter than first_min_chars
    c.reset()
    assert c.feed("ไพลินชอบเล่นเกม Minecraft ") == ["ไพลินชอบเล่นเกม "]
    assert c.feed("มากเลยค่ะ ") == []
    assert c.stall_flush() == ["Minecraft มากเลยค่ะ "]  # trailing space after a particle


def test_stall_flush_takes_trailing_punctuation() -> None:
    c = ThaiSpeechChunker(ChunkerConfig(first_min_chars=12))
    assert c.feed("จริงเหรอ!") == []
    assert c.stall_flush() == []  # 9 chars < first_min_chars
    assert c.feed(" ไม่น่าเชื่อเลยนะเนี่ย!") == []
    assert c.stall_flush() == ["จริงเหรอ! ไม่น่าเชื่อเลยนะเนี่ย!"]


def test_poll_fires_once_per_stall_with_fake_clock() -> None:
    clock = FakeClock()
    c = ThaiSpeechChunker(clock=clock)
    c.feed("สวัสดีค่ะทุกคน วันนี้ไพลินอยากเล่า ")
    assert c.poll(clock.now()) == []  # the first chunk is already out; nothing stalled yet
    c.feed("เรื่องตลก")
    deadline = c.stall_deadline()
    assert deadline == pytest.approx(clock.now() + 0.5)
    clock.advance(0.3)
    assert c.poll(clock.now()) == []
    clock.advance(0.2)
    assert c.poll(clock.now()) == ["วันนี้ไพลินอยากเล่า "]
    clock.advance(1.0)
    assert c.poll(clock.now()) == []  # once per stall
    assert c.stall_deadline() is None
    c.feed("ให้ฟังค่ะ ")  # re-arms
    clock.advance(0.4)
    assert c.poll() == []  # the chunker's own clock is used when ``now`` is omitted
    clock.advance(0.1)
    assert c.poll() == ["เรื่องตลกให้ฟังค่ะ "]  # a trailing space after a particle is safe
    assert c.flush() == []


async def test_stall_flush_in_a_streaming_loop() -> None:
    clock = FakeClock()
    chunker = ThaiSpeechChunker(clock=clock)
    emitted: list[tuple[float, str]] = []
    done = asyncio.Event()
    t0 = clock.now()
    deltas = [
        (0.05, "สวัสดีค่ะ"), (0.05, "ทุกคน "), (0.05, "วันนี้"), (0.05, "อากาศ"),
        (0.05, "ดีมาก ไพ"), (0.05, "ลินอยาก"), (0.05, "ออกไป ข้าง"),
        (1.0, "นอกแต่ฝนตก"),  # the LLM stalls for 1 s before this delta
        (0.05, "หนักมากเลย"),
    ]  # fmt: skip

    async def producer() -> None:
        for delay, delta in deltas:
            await clock.sleep(delay)
            emitted.extend((clock.now() - t0, c) for c in chunker.feed(delta))
        emitted.extend((clock.now() - t0, c) for c in chunker.flush())
        done.set()

    async def poller() -> None:
        while not done.is_set():
            await clock.sleep(0.1)
            emitted.extend((clock.now() - t0, c) for c in chunker.poll(clock.now()))

    tasks = [asyncio.create_task(producer()), asyncio.create_task(poller())]
    await clock.run_for(3.0)
    await asyncio.gather(*tasks)
    texts = [c for _, c in emitted]
    assert texts == [
        "สวัสดีค่ะทุกคน ",
        "วันนี้อากาศดีมาก ไพลินอยากออกไป ",
        "ข้างนอกแต่ฝนตกหนักมากเลย",
    ]
    stall_t = emitted[1][0]
    assert 0.35 + 0.5 <= stall_t <= 0.35 + 0.5 + 0.1 + 1e-9  # last delta at 0.35 s


# --- config --------------------------------------------------------------------------------


def test_config_from_constraints_and_settings() -> None:
    f0 = TTSConstraints(first_min_chars=8, min_chars=60, max_chars=160, backend="azure",
                        identity="premwadee")  # fmt: skip
    cfg = ChunkerConfig.from_constraints(f0)
    assert (cfg.first_min_chars, cfg.first_max_chars, cfg.min_chars, cfg.max_chars) == (
        8, 60, 60, 160,
    )  # fmt: skip
    assert cfg.word_split_chars == 120
    assert cfg.strong_min_chars == 40  # raised with min_chars to protect the F0 quota
    default = ChunkerConfig.from_constraints(
        TTSConstraints(
            first_min_chars=8, min_chars=40, max_chars=160, backend="edge", identity="premwadee"
        )
    )
    assert default == ChunkerConfig()
    small = ChunkerConfig.from_constraints(
        TTSConstraints(first_min_chars=4, min_chars=10, max_chars=30, backend="x", identity="y")
    )
    assert small.strong_min_chars == 10 and small.first_max_chars == 30
    tiny = ChunkerConfig.from_constraints(
        TTSConstraints(first_min_chars=50, min_chars=90, max_chars=40, backend="x", identity="y")
    )
    assert tiny.first_min_chars <= tiny.first_max_chars <= tiny.max_chars == 40
    assert tiny.min_chars <= tiny.max_chars and tiny.strong_min_chars <= tiny.min_chars

    from aivtube.config import load_config

    from_toml = ChunkerConfig.from_settings(load_config(ROOT, profile="ci").tts.chunker)
    assert from_toml == ChunkerConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_min_chars": 0},
        {"first_min_chars": 70},
        {"first_max_chars": 200},
        {"min_chars": 200},
        {"stall_flush_s": 0},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)  # type: ignore[arg-type]


def test_lookahead_bounds_the_wait() -> None:
    c = ThaiSpeechChunker(ChunkerConfig(first_min_chars=3), word_tokenize=no_word_split)
    # A space followed by a possible particle prefix waits, but at most LOOKAHEAD characters.
    out = c.feed("ใช่แล้ว " + "ก" * LOOKAHEAD)
    assert out == ["ใช่แล้ว "]


@pytest.mark.timing
def test_feed_is_fast(warm: None) -> None:
    text = (MIXED + " " + SPACED_NO_PUNCT + " " + UNSPACED) * 3
    c = ThaiSpeechChunker()
    worst = 0.0
    t_all = time.perf_counter()
    for i in range(0, len(text), 3):
        t = time.perf_counter()
        c.feed(text[i : i + 3])
        worst = max(worst, time.perf_counter() - t)
    c.flush()
    total = time.perf_counter() - t_all
    assert worst < 0.005, worst  # never block the loop for 5 ms
    assert total / (len(text) / 100) < 0.002  # < 2 ms per 100 chars


ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
