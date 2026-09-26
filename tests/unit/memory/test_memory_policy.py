"""The quarantine rule and the PII guard (§6)."""

from __future__ import annotations

import pytest

from aivtube.memory.policy import find_pii, initial_status, may_forget, origin_kind, stricter


@pytest.mark.parametrize(
    ("kind", "source", "origin", "chat_sourced", "expected"),
    [
        ("core", "model", "voice", "quarantine", "active"),
        ("core", "model", "operator", "quarantine", "active"),
        ("core", "model", "chat:twitch:123", "quarantine", "quarantined"),
        ("core", "model", "support:youtube:x", "quarantine", "quarantined"),
        ("core", "model", "mention", "quarantine", "quarantined"),
        ("core", "model", "chat:twitch:123", "allow", "active"),
        ("core", "model", "game_force", "allow", "quarantined"),
        ("core", "model", "idle", "quarantine", "quarantined"),
        ("core", "model", "", "quarantine", "quarantined"),
        ("core", "model", " VOICE ", "quarantine", "active"),
        ("viewer", "model", "voice", "allow", "quarantined"),
        ("viewer", "operator", "panel", "quarantine", "active"),
        ("fact", "import", "", "quarantine", "active"),
        ("episode", "consolidation", "", "quarantine", "active"),
    ],
)
def test_initial_status(
    kind: str, source: str, origin: str, chat_sourced: str, expected: str
) -> None:
    assert initial_status(kind, source, origin, chat_sourced=chat_sourced) == expected  # type: ignore[arg-type]


def test_requested_status_can_only_be_stricter() -> None:
    assert initial_status("core", "model", "voice", requested="quarantined") == "quarantined"
    assert initial_status("core", "operator", "", requested="quarantined") == "quarantined"
    with pytest.raises(ValueError):
        initial_status("core", "operator", "", requested="deleted")
    assert stricter("active", "quarantined") == "quarantined"
    assert stricter("active", "active") == "active"


def test_may_forget_and_origin_kind() -> None:
    assert origin_kind("chat:twitch:1") == "chat"
    assert may_forget("voice") and may_forget("operator:panel")
    assert not may_forget("chat:twitch:1") and not may_forget("")
    assert may_forget("chat", chat_sourced="allow")
    assert not may_forget("game_force", chat_sourced="allow")


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("อีเมลคือ pailin.fan+1@example.co.th นะ", "email"),
        ("บัตรประชาชน 1-1037-00123-45-6", "national_id"),
        ("เลขบัตร 1103700123456", "national_id"),
        ("โทร 081-234-5678 ได้เลย", "phone"),
        ("เบอร์ 0812345678", "phone"),
        ("บ้าน 02 123 4567", "phone"),
        ("+66 81 234 5678", "phone"),
        ("call +1 415 555 0100", "phone"),
        ("เบอร์ ๐๘๑๒๓๔๕๖๗๘", "phone"),
    ],
)
def test_pii_is_found(text: str, category: str) -> None:
    assert find_pii(text) == category


@pytest.mark.parametrize(
    "text",
    [
        "ไพลินชอบกินข้าวมันไก่",
        "โดเนท 1000 บาท",
        "สตรีมวันที่ 2026-09-26 เวลา 20:30",
        "เลเวล 99 แล้ว",
        "ราคา 1,250,000 บาท",
        "แพทช์ 14.18.2",
    ],
)
def test_ordinary_text_is_not_pii(text: str) -> None:
    assert find_pii(text) is None
