"""chat._text: normalisation, near-duplicate keys, alias matching and question shape."""

from __future__ import annotations

import pytest

from aivtube.chat import AliasMatcher, dedupe_key, is_question, normalize
from aivtube.chat._text import collapse_runs, strip_zero_width

ALIASES = ["ไพลิน", "pailin", "ไพ่ลิน", "น้องไพลิน"]


def test_normalize_strips_zero_width_nfkc_and_case() -> None:
    assert normalize("  ไพ​ลิน­  HELLO\t\nＷｏｒｌｄ ") == "ไพลิน hello world"
    assert strip_zero_width("a​b‌c‍d⁠e﻿") == "abcde"
    assert collapse_runs("5555 ไพลินนนน") == "5 ไพลิน"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("555555", "5555 !!!"),
        ("ไพลินนนนน", "ไพลิน"),
        ("HELLO  chat", "hello chat"),
        ("ไพ​ลิน น่ารัก", "ไพลินน่ารัก!!"),
        ("😂😂😂", "😂"),
    ],
)
def test_near_duplicates_share_a_key(a: str, b: str) -> None:
    assert dedupe_key(a) == dedupe_key(b)


def test_different_messages_have_different_keys() -> None:
    assert dedupe_key("ไพลินกินข้าวยัง") != dedupe_key("ไพลินเล่นเกมอะไร")
    assert dedupe_key("😂") != dedupe_key("🔥")


@pytest.mark.parametrize(
    "text",
    [
        "สวัสดีไพลิน",
        "ไพ​ลินนนน",
        "@PaiLin hi",
        "PAILIN!!",
        "ไพ่ลินจ๋า",
        "ไพ ลิน ทำอะไรอยู่",
        "น้องไพลินน่ารัก",
        "ไพลิ้น",  # tone-mark variant
    ],
)
def test_alias_matcher_finds_mentions(text: str) -> None:
    assert AliasMatcher(ALIASES)(text)


@pytest.mark.parametrize("text", ["สวัสดีทุกคน", "hello chat", "", "ลิน", "ไพ"])
def test_alias_matcher_ignores_other_text(text: str) -> None:
    assert not AliasMatcher(ALIASES)(text)


def test_alias_matcher_accepts_a_single_string_and_skips_tiny_aliases() -> None:
    assert AliasMatcher("pailin")("hi pailin")
    assert not AliasMatcher(["a"])("a banana")  # one-character aliases would match everything
    assert "pailin" in repr(AliasMatcher(["pailin"]))


@pytest.mark.parametrize(
    "text",
    [
        "เพลงนี้ชื่ออะไรครับ",
        "กินข้าวยังคะ",
        "ไปไหม",
        "จริงเหรอ 555",
        "เท่าไหร่คะ",
        "ทำอะไรอยู่อ่ะ 555",
        "ทำไมวันนี้ไลฟ์ช้า",
        "เล่นเกมนี้ได้ไหมคะ",
        "ok?",
        "จริงดิ？",
        "what game is this",
        "Do you like cats",
    ],
)
def test_question_shapes(text: str) -> None:
    assert is_question(text)


@pytest.mark.parametrize(
    "text",
    ["555555", "ไพลินน่ารัก", "ไม่เป็นไร", "ok", "ขอบคุณค่ะ", "", "คะ", "ยังไงก็ได้นะ"],
)
def test_not_questions(text: str) -> None:
    assert not is_question(text)
