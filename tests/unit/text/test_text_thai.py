"""Thai helpers: character classes, safe cuts, particles, matching utilities, the lazy newmm
tokenizer (modules.json text.normalize; §4.6, §7)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aivtube.text import thai
from aivtube.text.thai import (
    NameMatcher,
    ThaiWordTokenizer,
    attach_left,
    collapse_repeats,
    ends_with_final_particle,
    estimate_tokens,
    is_question,
    is_safe_cut,
    nfkc_casefold,
    strip_zero_width,
    thai_digits_to_arabic,
    warm_up_pythainlp,
)

REPO = Path(__file__).resolve().parents[3]


# --- character classes and safe cuts -------------------------------------------------------


def test_combining_marks_are_exactly_the_specified_ranges() -> None:
    expected = {0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)}
    assert {ord(c) for c in thai.THAI_COMBINING} == expected
    assert set("เแโใไ") == thai.LEADING_VOWELS
    for ch in thai.THAI_COMBINING:
        assert thai.is_combining(ch)
    assert thai.is_combining("́")  # any Unicode mark
    assert not thai.is_combining("ก")


@pytest.mark.parametrize(
    ("text", "k", "safe"),
    [
        ("น้ำ", 1, False),  # before MAI THO
        ("น้ำ", 2, False),  # before SARA AM
        ("กิน", 1, False),  # before SARA I
        ("เที่ยว", 1, False),  # after a leading vowel
        ("ไป", 1, False),
        ("ดีๆ", 2, False),  # before MAI YAMOK
        ("กรุงเทพฯ", 7, False),  # before PAIYANNOI
        ("ไปเที่ยว", 2, True),
        ("กินข้าว", 3, True),
        ("👍🏽", 1, False),  # before a skin-tone modifier
        ("❤️", 1, False),  # before VS16
        ("ab", 0, True),
        ("ab", 2, True),
    ],
)
def test_is_safe_cut(text: str, k: int, safe: bool) -> None:
    assert is_safe_cut(text, k) is safe


def test_is_thai_and_letters() -> None:
    assert thai.is_thai("ก") and thai.is_thai("๙") and thai.is_thai("่")
    assert not thai.is_thai("a")
    assert thai.is_thai_letter("ก") and thai.is_thai_letter("เ") and thai.is_thai_letter("า")
    assert not thai.is_thai_letter("่") and not thai.is_thai_letter("ๆ")
    assert not thai.is_thai_letter("๑")


# --- particles ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rest", "complete", "expected"),
    [
        ("กันนะคะ ต่อ", False, True),
        ("ค่ะ ", False, True),
        ("ครับผม", False, True),  # unambiguous particle
        ("ๆ ไป", False, True),
        ("555 ตลก", False, True),
        ("เลยนะ", False, True),
        ("กันยายน", False, False),  # a word starting like กัน
        ("คะแนน", False, False),
        ("อะไร", False, False),
        ("สิบ", False, False),
        ("ไปกัน", False, False),
        ("ก", False, None),  # may grow into กัน
        ("คะ", False, None),  # lone ambiguous particle without right context
        ("คะ", True, True),
        ("คะ ", False, True),
        ("ก", True, False),
    ],
)
def test_attach_left(rest: str, complete: bool, expected: bool | None) -> None:
    assert attach_left(rest, complete) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("ไปกันนะคะ", True), ("ดีจ้าาา", True), ("ขอบคุณครับ  ", True), ("สวัสดี", False), ("", False)],
)
def test_ends_with_final_particle(text: str, expected: bool) -> None:
    assert ends_with_final_particle(text) is expected


# --- matching normalisation -----------------------------------------------------------------


def test_strip_zero_width() -> None:
    assert strip_zero_width("ไพ​ลิน‌‍﻿⁠­‮") == "ไพลิน"


def test_nfkc_casefold_keeps_sara_am_and_is_idempotent() -> None:
    assert nfkc_casefold("ＡＢＣ ทำ น้ำ") == "abc ทำ น้ำ"
    assert "ำ" in nfkc_casefold("ทำ")


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=40))
def test_nfkc_casefold_idempotent(text: str) -> None:
    once = nfkc_casefold(text)
    assert nfkc_casefold(once) == once


def test_collapse_repeats() -> None:
    assert collapse_repeats("ควายยยยย") == "ควายย"
    assert collapse_repeats("ว้าวววว", 1) == "ว้าว"
    assert collapse_repeats("aaa bbb", 3) == "aaa bbb"
    with pytest.raises(ValueError):
        collapse_repeats("x", 0)


def test_thai_digits_to_arabic() -> None:
    assert thai_digits_to_arabic("๐๑๒๓๔๕๖๗๘๙ บาท") == "0123456789 บาท"


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("สวัสดี") == 3  # 6 Thai chars / 2
    assert estimate_tokens("hello") == 2  # ceil(5 / 4)
    assert estimate_tokens("สวัสดี hello") == 5  # ceil(3 + 1.25); spaces are free


@pytest.mark.parametrize(
    "text",
    ["กินข้าวยัง", "ทำไมล่ะ", "ไปไหนมาคะ", "ไปกันไหมครับ", "ใช่ป่ะ", "ok?", "how are you",
     "เป็นยังไงบ้าง", "ใครชนะ", "เท่าไหร่คะ", "จริงเหรอ", "Is it live", "เล่นเกมอะไรอยู่"],
)  # fmt: skip
def test_is_question_true(text: str) -> None:
    assert is_question(text)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "อะไรก็ได้", "ไม่มีใครรู้", "ไปด้วยนะคะ", "บ้านไหม้", "จริงหรอก", "สวัสดีค่ะ",
     "hello there", "ไม่ว่าใครก็ทำได้"],
)  # fmt: skip
def test_is_question_false(text: str) -> None:
    assert not is_question(text)


def test_name_matcher() -> None:
    m = NameMatcher(["ไพลิน", "Pailin", "ai", "  "])
    assert m("ไพ​ลิน สวัสดี")
    assert m("hi PAILIN!")
    assert m("ＰＡＩＬＩＮ")
    assert m("AI ตอบหน่อย")
    assert m("p a i l i n") and m("ไพ ลิน")
    assert m("น้องไพลินน่ารัก")  # Thai matches inside words (no spaces)
    assert not m("he said")  # Latin aliases need Latin word boundaries
    assert not m("pailin2")
    assert not m("สวัสดีทุกคน")
    assert not NameMatcher([])("ไพลิน")


# --- the newmm tokenizer ----------------------------------------------------------------------


def test_newmm_tokens_join_back_to_the_input() -> None:
    warm_up_pythainlp()
    text = "ไพลินชอบกินข้าวมันไก่  มากที่สุด Minecraft!"
    tokens = thai.newmm(text)
    assert "".join(tokens) == text
    assert len(tokens) > 3
    assert thai.newmm("") == []
    assert thai.newmm.warmed and thai.newmm.available
    assert thai.newmm.warm() == 0.0  # already warm


def test_warm_up_returns_seconds() -> None:
    assert warm_up_pythainlp() >= 0.0


async def test_lazy_warm_on_the_event_loop_warns(caplog: pytest.LogCaptureFixture) -> None:
    tok = ThaiWordTokenizer()
    with caplog.at_level(logging.WARNING, logger="aivtube.text.thai"):
        assert "".join(tok("สวัสดีค่ะ")) == "สวัสดีค่ะ"
    assert "event loop" in caplog.text
    assert tok.warm_s is not None


def test_missing_pythainlp_degrades_to_one_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", None)  # import raises ImportError
    tok = ThaiWordTokenizer()
    assert tok.warm() >= 0.0
    assert tok.warmed and not tok.available
    assert tok("ไพลินชอบกินข้าว") == ["ไพลินชอบกินข้าว"]


def test_importing_the_text_package_does_not_import_pythainlp() -> None:
    code = (
        "import sys, aivtube.text; "
        "from aivtube.text import ThaiSpeechChunker, normalize_cloud; "
        "normalize_cloud('สวัสดี 555'); ThaiSpeechChunker(); "
        "print(sorted(m for m in ('pythainlp', 'numpy') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True
    )
    assert out.stdout.strip() == "[]"
