"""Built-in PII patterns: masking spans on the display text."""

from __future__ import annotations

import re
from itertools import pairwise

import pytest

from aivtube.safety.pii import find_pii, mask_pii, mask_spans, thai_id_valid

VALID_ID = "1101700230708"  # checksum-valid, not a real person


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("โทร 0812345678", "phone"),
        ("โทร 081-234-5678", "phone"),
        ("โทร 081 234 5678", "phone"),
        ("โทร 08.1234.5678", "phone"),
        ("โทร ๐๘๑๒๓๔๕๖๗๘", "phone"),
        ("โทร +66812345678", "phone"),
        (f"บัตร {VALID_ID}", "national_id"),
        ("บัตร 1-1017-00230-70-8", "national_id"),
        ("ไปที่ https://example.com/a?b=1", "url"),
        ("ไปที่ www.example.com", "url"),
        ("ไปที่ example.com", "url"),
        ("ไปที่ bit.ly/3abc", "url"),
        ("เว็บbadsite.com", "url"),
        ("ส่งมาที่ someone@example.co.th", "email"),
        ("ตามไปที่ @spam_account", "handle"),
    ],
)
def test_detects(text: str, kind: str) -> None:
    spans = find_pii(text)
    assert [s.kind for s in spans] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "ราคา 1,299 บาท",
        "ได้คะแนน 9999999 แต้ม",
        "เวลา 20.30 น.",
        "เวอร์ชัน 1.2.3",
        "ไฟล์ save.dat",
        "Node.js ใช้ยากไหม",
        "1000000000000",  # 13 digits, bad checksum
        "0.8123456789",  # a decimal, not a phone number
        "e.g. ตัวอย่าง",
        "a@b",  # no domain
    ],
)
def test_ignores_non_pii(text: str) -> None:
    assert find_pii(text) == []


def test_thai_id_checksum() -> None:
    assert thai_id_valid(VALID_ID)
    assert thai_id_valid("๑๑๐๑๗๐๐๒๓๐๗๐๘")
    assert not thai_id_valid("1101700230709")
    assert not thai_id_valid("123")


def test_email_wins_over_url_and_handle() -> None:
    spans = find_pii("mail me at a.b@example.com now")
    assert [s.kind for s in spans] == ["email"]


def test_mask_spans_uses_placeholders() -> None:
    text = "โทร 0812345678 หรือ https://x.com"
    masked = mask_spans(text, find_pii(text), mask_text="[ลิงก์]")
    assert masked == "โทร [เบอร์โทร] หรือ [ลิงก์]"


def test_mask_pii_cleans_first() -> None:
    zw = chr(0x200B)
    assert mask_pii(f"โทร 081{zw}2345678") == "โทร [เบอร์โทร]"
    assert mask_pii("ไม่มีอะไร") == "ไม่มีอะไร"


def test_allow_ranges_and_handle_aliases() -> None:
    text = "ดูที่ youtube.com/@pailin_th และ @pailin_th"
    assert [s.kind for s in find_pii(text, handle_aliases=("pailin",))] == ["url"]
    start = text.index("youtube")
    allowed = [(start, start + len("youtube.com/@pailin_th"))]
    assert find_pii(text, allow_ranges=allowed, handle_aliases=("pailin",)) == []


def test_extra_patterns_are_links() -> None:
    extra = [("link", re.compile(r"discord\s*gg\s*/\s*\w+", re.IGNORECASE))]
    spans = find_pii("มา discord gg / abc", extra)
    assert [s.kind for s in spans] == ["link"]


def test_overlaps_resolve_by_priority_then_position() -> None:
    text = "x@a.com y.com @zz 0812345678"
    assert [s.kind for s in find_pii(text)] == ["email", "url", "handle", "phone"]
    spans = find_pii("@ab " * 300)
    assert len(spans) == 300 and all(a.end <= b.start for a, b in pairwise(spans))
