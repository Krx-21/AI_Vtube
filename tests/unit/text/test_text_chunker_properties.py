"""Hypothesis properties of ThaiSpeechChunker (ARCHITECTURE.md §10: split-invariant, lossless,
never cuts before a combining mark, every chunk ≤ max)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aivtube.text import thai
from aivtube.text.chunker import ChunkerConfig, ThaiSpeechChunker, no_word_split

THAI_WORDS = (
    "ไพลิน", "สวัสดี", "วันนี้", "เล่นเกม", "ข้าวมันไก่", "อร่อย", "มาก", "ที่สุด", "ใน", "โลก",
    "เพราะ", "ร้าน", "ปิด", "น้ำ", "เที่ยว", "ทะเล", "ใจดี", "เหมือน", "เรียน", "แปลก",
    "โรงเรียน", "ใคร", "ไป", "ไหน", "ก็", "ได้", "ที่", "ผี", "หนัง", "สนุก", "ความรัก", "กำลัง",
    "ปั่น", "ซิ่ง", "กิ๊ก", "คะแนน", "อะไร", "ค่าใช้จ่าย", "กันยายน", "สิบ", "เก่ง", "เดี๋ยว",
    "แล้ว", "ไม่", "ใช่", "ญี่ปุ่น", "ฤดู", "กรุงเทพฯ", "เเปลก", "สำเร็จรูป",
)  # fmt: skip
PARTICLES = (
    "ค่ะ", "คะ", "ครับ", "นะ", "นะคะ", "กัน", "กันนะคะ", "เลย", "ด้วย", "จ้า", "จ้าาา", "ๆ",
    "555", "5555+", "ฮ่าๆ", "น้า", "ค่า",
)  # fmt: skip
NUMBER_UNITS = ("100 บาท", "เวลา 02:30", "3.5 ดาว", "1,250.50 บาท", "$29.99", "ปี 2026", "50%",
                "๑๒๓ คน", "โทร 0812345678", "เวอร์ชัน 1.21.4")  # fmt: skip
LATIN = ("Minecraft Java Edition", "John_Doe", "OK", "e.g.", "VTube Studio", "don't know",
         "Mr. Bean", "a?b")  # fmt: skip
PUNCT = ("!", "?", "!!!", "...", "…", ".", ",", "😂", "👍🏽", "~", "。", "！")
CLAUSE_END = (" ", " ", " ", "\n", "  ", "! ", "? ", "... ", ", ", "😂 ", ". ", "\n\n")

CONFIGS = {
    "default": ChunkerConfig(),
    "small": ChunkerConfig(
        first_min_chars=4, first_max_chars=20, min_chars=10, max_chars=40, strong_min_chars=5
    ),
    "azure_f0": ChunkerConfig(min_chars=60),
}
TOKENIZERS = {"newmm": None, "none": no_word_split}


@dataclass
class Sample:
    text: str
    protected: list[tuple[int, int]]


@st.composite
def clause(draw: st.DrawFn) -> tuple[str, str | None]:
    words = draw(st.lists(st.sampled_from(THAI_WORDS), min_size=1, max_size=4))
    seps = draw(st.lists(st.sampled_from(("", "", " ")), min_size=len(words), max_size=len(words)))
    body = "".join(w + s for w, s in zip(words, seps, strict=True)).rstrip()
    item = draw(st.none() | st.sampled_from(NUMBER_UNITS + LATIN))
    if draw(st.booleans()):
        body += draw(st.sampled_from(PARTICLES))
    return body, item


@st.composite
def thai_texts(draw: st.DrawFn) -> Sample:
    """Clauses of Thai words with numbers, Latin phrases, particles and punctuation. Protected
    spans are the number/Latin items: the chunker must never cut inside them."""
    text = ""
    protected: list[tuple[int, int]] = []
    for _ in range(draw(st.integers(min_value=1, max_value=12))):
        body, item = draw(clause())
        text += body
        if item is not None:
            text += " "
            protected.append((len(text), len(text) + len(item)))
            text += item + " " + draw(st.sampled_from(THAI_WORDS))
        text += draw(st.sampled_from(CLAUSE_END))
    return Sample(text, protected)


RAW_ALPHABET = st.one_of(
    st.characters(min_codepoint=0x0E00, max_codepoint=0x0E7F),
    st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    st.sampled_from(" \n\t😂🏽‍️…！"),
)


def chunk_all(text: str, sizes: list[int], cfg: ChunkerConfig, tok: object) -> list[str]:
    c = ThaiSpeechChunker(cfg, word_tokenize=tok)  # type: ignore[arg-type]
    out: list[str] = []
    pos = i = 0
    while pos < len(text):
        n = sizes[i % len(sizes)]
        out += c.feed(text[pos : pos + n])
        pos += n
        i += 1
    return out + c.flush()


def cut_offsets(text: str, chunks: list[str]) -> list[int]:
    pos = len(text) - len(text.lstrip())
    offsets = []
    for chunk in chunks[:-1]:
        pos += len(chunk)
        offsets.append(pos)
    return offsets


def check_common(text: str, whole: list[str], split: list[str], cfg: ChunkerConfig) -> None:
    assert split == whole  # split-invariant
    assert "".join(whole) == text.strip()  # lossless (exact slices; only edge whitespace goes)
    for idx, chunk in enumerate(whole):
        assert chunk.strip()
        assert len(chunk) <= (cfg.first_max_chars if idx == 0 else cfg.max_chars)


SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    thai.warm_up_pythainlp()


@pytest.mark.parametrize("tok_name", TOKENIZERS)
@pytest.mark.parametrize("cfg_name", CONFIGS)
@SETTINGS
@given(
    sample=thai_texts(),
    sizes=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=20),
)
def test_thai_text_properties(
    cfg_name: str, tok_name: str, sample: Sample, sizes: list[int]
) -> None:
    cfg, tok = CONFIGS[cfg_name], TOKENIZERS[tok_name]
    text = sample.text
    whole = chunk_all(text, [len(text) or 1], cfg, tok)
    split = chunk_all(text, sizes, cfg, tok)
    check_common(text, whole, split, cfg)
    offsets = cut_offsets(text, whole)
    for off in offsets:
        # never before a combining mark / following vowel / ๆ, never after a leading vowel
        assert thai.is_safe_cut(text, off), (off, text[max(0, off - 8) : off + 8], whole)
    for idx, chunk in enumerate(whole):
        assert not thai.is_combining(chunk[0]), whole
        if idx < len(whole) - 1:
            assert chunk.rstrip()[-1] not in thai.LEADING_VOWELS, whole
    if cfg_name != "small":
        for start, end in sample.protected:
            assert not any(start < off < end for off in offsets), (text[start:end], whole)


@pytest.mark.parametrize("tok_name", TOKENIZERS)
@SETTINGS
@given(
    text=st.text(alphabet=RAW_ALPHABET, max_size=600),
    sizes=st.lists(st.integers(min_value=1, max_value=30), min_size=1, max_size=20),
)
def test_arbitrary_text_is_lossless_split_invariant_and_bounded(
    tok_name: str, text: str, sizes: list[int]
) -> None:
    cfg = ChunkerConfig()
    whole = chunk_all(text, [len(text) or 1], cfg, TOKENIZERS[tok_name])
    split = chunk_all(text, sizes, cfg, TOKENIZERS[tok_name])
    check_common(text, whole, split, cfg)


@SETTINGS
@given(
    text=st.text(alphabet=RAW_ALPHABET, max_size=300),
    sizes=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=10),
)
def test_tiny_limits_still_bounded_and_invariant(text: str, sizes: list[int]) -> None:
    cfg = ChunkerConfig(first_min_chars=1, first_max_chars=3, min_chars=1, max_chars=5,
                        strong_min_chars=1)  # fmt: skip
    whole = chunk_all(text, [len(text) or 1], cfg, None)
    split = chunk_all(text, sizes, cfg, None)
    check_common(text, whole, split, cfg)
