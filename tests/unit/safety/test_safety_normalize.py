"""Matching forms (§7 normalisation): norm, despaced and leet."""

from __future__ import annotations

import pytest

from aivtube.safety.normalize import clean, despace, has_thai, leet, match_forms, norm

ZW = chr(0x200B)
ZWJ = chr(0x200D)
SHY = chr(0x00AD)
BOM = chr(0xFEFF)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"เห{ZW}ี้ย", "เหี้ย"),  # zero-width stripped
        (f"ใน{BOM}หลวง{SHY}", "ในหลวง"),
        ("ＦＵＣＫ", "fuck"),  # NFKC + casefold
        ("เเตด", "แตด"),  # pythainlp: two SARA E become SARA AE
        ("เหี้้้ย", "เหี้ย"),  # duplicate tone marks
        ("ควยยยยยย", "ควยย"),  # runs longer than 2 shortened to 2
        ("555555", "55"),
        ("ม.๑๑๒", "ม.112"),  # Thai digits
        ("  สวัสดี \t ค่ะ  ", "สวัสดี ค่ะ"),  # whitespace runs
    ],
)
def test_norm(raw: str, expected: str) -> None:
    assert norm(raw) == expected


def test_norm_keeps_sara_am_composed() -> None:
    assert norm("ส้มตำ") == "ส้มตำ"
    assert norm(norm("ส้มตำ")) == norm("ส้มตำ")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("f u c k you", "fuck you"),
        ("f.u.c.k", "fuck"),
        ("fu-ck", "fuck"),
        ("ค ว ย", "ควย"),
        ("ค.ว.ย", "ควย"),
        ("ค_ว_ย", "ควย"),
        ("ค*ว*ย", "ควย"),
        ("เ หี้ ย", "เหี้ย"),
        ("hello world", "hello world"),  # multi-letter words are not joined
        ("a big cat", "a big cat"),  # a lone single letter stays
    ],
)
def test_despace(raw: str, expected: str) -> None:
    assert despace(norm(raw)) == expected


def test_leet_map() -> None:
    assert leet("n1gg3r") == "nigger"
    assert leet("$h1t") == "shit"
    assert leet("4ss") == "ass"
    assert leet("@ss") == "ass"
    plain = "no leet here"
    assert leet(plain) is plain


def test_clean_keeps_case_and_drops_invisible() -> None:
    assert clean(f"Hello{ZW} ไพ{ZWJ}ลิน") == "Hello ไพลิน"
    assert clean("ＡＢＣ") == "ABC"


def test_match_forms_tokens_and_forms() -> None:
    forms = match_forms("ส่งหีบห่อมาให้หน่อย")
    assert "หีบห่อ" in forms.tokens
    assert "หี" not in forms.tokens
    assert forms.norm == forms.despaced == forms.leet
    spaced = match_forms("n 1 g g 3 r")
    assert spaced.despaced == "n1gg3r"
    assert spaced.leet == "nigger"


def test_has_thai() -> None:
    assert has_thai("abc ไทย")
    assert not has_thai("abc 123")
