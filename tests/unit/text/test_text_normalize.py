"""TTS normalisers and the speakability check (modules.json text.normalize; §4.6, §4.10)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aivtube.config import load_lexicon
from aivtube.text import normalize, thai
from aivtube.text.normalize import (
    FILTERED,
    apply_lexicon,
    is_speakable,
    normalize_cloud,
    normalize_local,
)

ROOT = Path(__file__).resolve().parents[3]
PAILIN_LEXICON = load_lexicon(ROOT / "characters" / "pailin" / "lexicon.toml")

ALPHABET = st.one_of(
    st.characters(min_codepoint=0x0E00, max_codepoint=0x0E7F),
    st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    st.sampled_from(" \n\t😂🏽‍️…！​๑๒"),
)
SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    thai.warm_up_pythainlp()


# --- normalize_cloud -----------------------------------------------------------------------


def test_cloud_acceptance_emoji_markdown_urls_laughter_elongation() -> None:
    text = "ว้าวววว!!! 😂 **สุดยอด** ดูที่ https://example.com/x?y=1 นะ 5555+ `code` # หัวข้อ"
    assert normalize_cloud(text) == "ว้าว! สุดยอด ดูที่ ลิงก์ นะ ฮ่าฮ่าฮ่า code หัวข้อ"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ว้าวววว", "ว้าว"),
        ("555", "ฮ่าฮ่าฮ่า"),
        ("555555 ตลกมาก", "ฮ่าฮ่าฮ่า ตลกมาก"),
        ("ตลก5555+ค่ะ", "ตลก ฮ่าฮ่าฮ่า ค่ะ"),
        ("ดีใจ 👍🏽❤️✨ มาก", "ดีใจ มาก"),
        ("👨‍👩‍👧 ครอบครัว", "ครอบครัว"),
        ("*หัวเราะ* โอเคค่ะ", "โอเคค่ะ"),
        ("__ตัวหนา__ และ `โค้ด`", "ตัวหนา และ โค้ด"),
        ("[ลิงก์นี้](http://a.b/c) ![รูป](x.png)", "ลิงก์นี้ รูป"),
        ("- ข้อแรก\n- ข้อสอง\n3. ข้อสาม", "ข้อแรก ข้อสอง ข้อสาม"),
        ("> อ้างอิง ~ ไพลิน", "อ้างอิง ไพลิน"),
        ("เข้า www.twitch.tv/pailin ได้เลย", "เข้า ลิงก์ ได้เลย"),
        ("ไปที่ pailin.gg สิ", "ไปที่ ลิงก์ สิ"),
        ("จริงเหรอ??? ไม่นะ!!! ?!", "จริงเหรอ? ไม่นะ! ?"),
        ("รอแป๊บ....... …… ต่อ", "รอแป๊บ... … ต่อ"),
        ("ดี ๆ นะ", "ดีๆ นะ"),
        ("ไพ​ลิน‍", "ไพลิน"),
        ("Wooow coool", "Wow col"),
        ("Final Fantasy XIII กับ AAA", "Final Fantasy XIII กับ AAA"),
        ("  หลาย\n\nบรรทัด\t ", "หลาย บรรทัด"),
        ("- - ข้อแรก", "ข้อแรก"),
        ("-\n-\nไปกัน", "ไปกัน"),
        ("😂 - ไปกัน", "ไปกัน"),
        ("-555", "ฮ่าฮ่าฮ่า"),
        ("ดีๆ ๆ ๆ", "ดีๆ"),
        ("ดู pailin๑.com", "ดู ลิงก์"),
    ],
)
def test_cloud_rules(raw: str, expected: str) -> None:
    assert normalize_cloud(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ราคา 555 บาท",
        "มีคน 555 คน",
        "ได้ 5555 คะแนน",
        "ขอบคุณ John สำหรับ 1,250.50 บาท",
        "เวลา 02:30 น.",
        "ราคา $29.99 ลด 50% โทร 0812345678",
        "ปี 2026 เวอร์ชัน 1.21.4",
        "Minecraft กับ VTube Studio",
        "3.14",
    ],
)
def test_cloud_keeps_numbers_and_english_for_the_voice(text: str) -> None:
    assert normalize_cloud(text) == text  # edge/Azure read these natively


def test_cloud_maps_thai_digits() -> None:
    assert normalize_cloud("ปี ๒๕๖๙ มี ๑๒๓ คน") == "ปี 2569 มี 123 คน"


@pytest.mark.parametrize(
    "text",
    [
        FILTERED,
        f"**{FILTERED}**",
        f"ขอโทษนะ {FILTERED} ต่อเลย",
        f"*แอบ {FILTERED} บอก*",
        "Filtered.!!!",
    ],
)
def test_cloud_keeps_filtered_intact(text: str) -> None:
    out = normalize_cloud(text, lexicon={"Filtered.": "ฟิลเทอร์ด", "Filtered": "x"})
    assert FILTERED in out


def test_cloud_optional_lexicon() -> None:
    assert normalize_cloud("เล่น ROV กัน", lexicon={"ROV": "อาร์โอวี"}) == "เล่น อาร์โอวี กัน"
    assert normalize_cloud("เล่น ROV กัน") == "เล่น ROV กัน"


def test_cloud_private_use_input_cannot_forge_the_placeholder() -> None:
    assert normalize_cloud("ab") == "ab"


@SETTINGS
@given(text=st.text(alphabet=ALPHABET, max_size=160))
def test_cloud_is_single_line_trimmed_and_idempotent(text: str) -> None:
    out = normalize_cloud(text)
    assert out == out.strip()
    assert "\n" not in out and "\t" not in out and "  " not in out
    assert normalize_cloud(out) == out


@SETTINGS
@given(a=st.text(alphabet=ALPHABET, max_size=40), b=st.text(alphabet=ALPHABET, max_size=40))
def test_cloud_never_loses_filtered(a: str, b: str) -> None:
    assert FILTERED in normalize_cloud(f"{a} {FILTERED} {b}")


# --- normalize_local -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("21", "ยี่สิบเอ็ด"),
        ("101 คน", "หนึ่งร้อยเอ็ด คน"),
        ("ปี 2026", "ปี สองพันยี่สิบหก"),
        ("1,000,000 วิว", "หนึ่งล้าน วิว"),
        ("3.14", "สามจุดหนึ่งสี่"),
        ("เวอร์ชัน 1.21.4", "เวอร์ชัน หนึ่งจุดยี่สิบเอ็ดจุดสี่"),
        ("๑๒๓ คน", "หนึ่งร้อยยี่สิบสาม คน"),
        ("-5 องศา", "ลบห้า องศา"),
        ("ขอ 10-20 คน", "ขอ สิบ ถึง ยี่สิบ คน"),
        ("30°C", "สามสิบองศาเซลเซียส"),
        ("ลด 50%", "ลด ห้าสิบเปอร์เซ็นต์"),
        ("$29.99", "ยี่สิบเก้าจุดเก้าเก้าดอลลาร์"),
        ("฿100", "หนึ่งร้อยบาท"),
        ("1,250.50 บาท", "หนึ่งพันสองร้อยห้าสิบบาทห้าสิบสตางค์"),
        ("เวลา 02:30 น.", "เวลา สองนาฬิกาสามสิบนาที"),
        ("เจอกัน 10.30 น.", "เจอกัน สิบนาฬิกาสามสิบนาที"),
        ("โทร 081-234-5678", "โทร ศูนย์แปดหนึ่ง สองสามสี่ ห้าหกเจ็ดแปด"),
        ("รหัส 007", "รหัส ศูนย์ศูนย์เจ็ด"),
        ("ดีๆ ทั้งนั้น", "ดีดี ทั้งนั้น"),
        ("555 ตลก", "ฮ่าฮ่าฮ่า ตลก"),
        ("NPC ตัวนี้", "เอ็นพีซี ตัวนี้"),
        ("hello ไพลิน", "ไพลิน"),
        ("Hello everyone!", ""),  # Latin words go, and so does the punctuation they leave
        ("we play Minecraft now.", "มายคราฟ."),
        ("it costs $29.99, ok?", "ยี่สิบเก้าจุดเก้าเก้าดอลลาร์,?"),
        ("ไม่เก่ง! ok", "ไม่เก่ง!"),
    ],
)
def test_local_reads_numbers_in_thai(raw: str, expected: str) -> None:
    assert normalize_local(raw, {"Minecraft": "มายคราฟ"}) == expected


def test_local_acceptance_numbers_and_lexicon() -> None:
    lexicon = {"Minecraft": "มายคราฟต์", "VTuber": "วีทูบเบอร์"}
    out = normalize_local("VTuber ชื่อไพลินเล่น Minecraft มา 21 วัน", lexicon)
    assert out == "วีทูบเบอร์ ชื่อไพลินเล่น มายคราฟต์ มา ยี่สิบเอ็ด วัน"


def test_local_with_the_pailin_lexicon() -> None:
    text = "Pailin เล่น Minecraft บน VTube Studio แล้ว live บน Twitch ROV GG OK ไหม"
    assert normalize_local(text, PAILIN_LEXICON) == (
        "ไพลิน เล่น มายคราฟ บน วีทูบสตูดิโอ แล้ว ไลฟ์ บน ทวิช อาร์โอวี จีจี โอเค ไหม"
    )


def test_local_reads_filtered_via_the_lexicon() -> None:
    assert normalize_local(FILTERED) == "ฟิลเทอร์ด"  # built-in reading
    assert normalize_local(FILTERED, {FILTERED: "ถูกกรอง"}) == "ถูกกรอง"


def test_local_maiyamok_without_the_tokenizer_is_dropped_not_doubled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", None)  # import raises ImportError
    broken = thai.ThaiWordTokenizer()
    broken.warm()
    monkeypatch.setattr(normalize, "newmm", broken)
    assert normalize_local("วันนี้อากาศดีๆ นะ", {}) == "วันนี้อากาศดี นะ"


@SETTINGS
@given(text=st.text(alphabet=ALPHABET, max_size=160))
def test_local_output_is_thai_script_only(text: str) -> None:
    out = normalize_local(text, PAILIN_LEXICON)
    assert not re.search(r"[A-Za-z0-9]", out), out
    assert out == out.strip() and "\n" not in out


# --- lexicon -------------------------------------------------------------------------------


def test_lexicon_whole_words_case_insensitive_longest_first_single_pass() -> None:
    lex = {"live": "ไลฟ์", "VTube": "วีทูบ", "VTube Studio": "วีทูบสตูดิโอ", "a": "live"}
    assert apply_lexicon("LIVE livestream Live!", lex) == "ไลฟ์ livestream ไลฟ์!"
    assert apply_lexicon("เปิด vtube   studio กับ VTube", lex) == "เปิด วีทูบสตูดิโอ กับ วีทูบ"
    assert apply_lexicon("a b", lex) == "live b"  # replacements are not re-replaced


def test_lexicon_thai_keys_match_inside_words() -> None:
    assert apply_lexicon("ไพลินน่ารัก", {"ไพลิน": "ไพ-ลิน"}) == "ไพ-ลินน่ารัก"
    assert apply_lexicon("ข้อความ", {}) == "ข้อความ"
    assert apply_lexicon("ข้อความ", {"": "x", " ": "y"}) == "ข้อความ"


# --- is_speakable --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "   \n", "3.", "12)", "(3)", "...", "!!!", "…", "😂😂", "👍🏽", "**", "~", "*หัวเราะ*",
     "​", "ๆ", "่", "๓.", "1. 2."],
)  # fmt: skip
def test_not_speakable(text: str) -> None:
    assert is_speakable(text) is False


@pytest.mark.parametrize(
    "text", ["ก", "ค่ะ", "ok", "555", "42", "5", "123.", FILTERED, "ว้าว!", "1,250 บาท", "😂 ฮ่าๆ"]
)
def test_speakable(text: str) -> None:
    assert is_speakable(text) is True
