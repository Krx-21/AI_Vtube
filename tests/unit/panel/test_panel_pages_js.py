"""The panel's inline JavaScript, run by Node against a fake DOM with virtual timers.

Covers the client-side acceptance criteria that no server test can: FREEZE falls back to the
launcher's ``/hardkill`` when the core does not answer within 300 ms, the S/M/F shortcuts use
physical keys (they work with the Thai keyboard layout), and the captions overlay shows a
segment at its audible time and "Filtered." after an output block. Skipped without ``node``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
STATIC = HERE.parents[2] / "src" / "aivtube" / "panel" / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(page: str, scenario: str) -> dict[str, Any]:
    assert NODE is not None
    out = subprocess.run(
        [NODE, str(HERE / "panel_js_harness.js"), str(STATIC / page), scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    result: dict[str, Any] = json.loads(out.stdout)
    return result


def test_freeze_falls_back_to_launcher_hardkill_after_300_ms() -> None:
    r = _run("index.html", "freeze_hang")
    assert r["freeze_posted"] == [{"kind": "freeze"}]
    assert r["hardkill_before_300ms"] == 0
    assert r["hardkills"] == [
        {
            "url": "http://127.0.0.1:8779/hardkill?token=EMERG",
            "method": "POST",
            "mode": "no-cors",
            "after_ms": 300,
        }
    ]
    assert r["hardkills_total"] == 1  # the later request timeout does not kill twice
    assert "HARD KILL" in r["alarm"]
    assert r["ws_url"] == "ws://127.0.0.1:8770/ws?token=PANEL"


def test_freeze_answered_in_time_does_not_hardkill() -> None:
    r = _run("index.html", "freeze_fast")
    assert r["freeze_posted"] == [{"kind": "freeze"}]
    assert r["hardkills_total"] == 0 and r["alarm"] == ""


def test_shortcuts_use_physical_keys_and_ignore_typing() -> None:
    assert _run("index.html", "keys")["commands"] == ["skip", "mute"]


def test_captions_overlay_shows_segments_at_audible_time_and_filtered() -> None:
    r = _run("overlay_captions.html", "captions")
    assert "character=pailin" in r["ws_url"] and "token=PANEL" in r["ws_url"]
    assert r["before_audible"] == {"text": "", "shown": False}
    assert r["at_audible"] == {"text": "สวัสดีค่ะ", "shown": True}
    # the segment already playing was clean: it stays until it has been heard (§4.10) ...
    assert r["during_playing"] == {"text": "สวัสดีค่ะ", "red": False}
    assert r["after_filtered_segment"] == "สวัสดีค่ะ"  # segments after the block never show
    # ... then "Filtered." follows, also while the canned clip plays
    assert r["filtered"] == {"text": "Filtered.", "shown": True, "red": True}
    assert r["canned"] == {"text": "Filtered.", "red": True}
    assert r["hidden_later"] is True


def test_hardkill_works_after_a_reload_while_the_core_is_wedged() -> None:
    r = _run("index.html", "reload_wedged")
    assert r["config_requests"] >= 1 and "ติดต่อ core ไม่ได้" in r["boot_alarm"]
    assert r["alarm_hardkill_button"] is True
    assert "HARD KILL" in r["alarm"]
    assert r["hardkills"] == [
        {"url": "http://127.0.0.1:8779/hardkill?token=EMERG", "after_ms": 300}
    ]


def test_tool_approvals_and_tool_toggles() -> None:
    r = _run("index.html", "approvals")
    assert r["visible"] is True and "timeout_user" in r["text"] and "troll" in r["text"]
    assert r["buttons"] == ["อนุมัติ", "ปฏิเสธ"]
    assert r["tools"] == ["timeout_user", "play_sound"]
    assert "live" in r["resume_note"]
    assert r["commands"] == [
        {"kind": "approve", "args": {"request": "ap1", "approved": True}, "character": None},
        {"kind": "tool_enable", "args": {"name": "play_sound", "enabled": True}, "character": None},
    ]
    assert r["hidden_after"] is True


def test_moderation_feed_polls_on_filtered_and_mutes_the_chatter() -> None:
    r = _run("index.html", "moderation")
    assert "คำต้องห้าม" not in r["before_poll"]  # Filtered only triggers a poll (300 ms)
    assert "u-9: คำต้องห้าม นะ" in r["after_poll"] and "ทิ้ง · slur" in r["after_poll"]
    assert r["buttons"] == ["ปิดเสียงผู้ใช้ 10 นาที", "เพิ่มในบัญชีดำ", "กรองผิด (false positive)"]
    assert r["items"] == 1  # the duplicate ChatDropped is noise; the re-polled item is deduped
    assert "since_ts=1000.499" in r["last_poll"]
    assert r["mute"] == [
        {
            "action": "mute_user",
            "platform": "twitch",
            "user_id": "u-9",
            "name": "u-9",
            "minutes": 10,
        }
    ]


def test_memory_buttons_follow_what_the_store_can_edit() -> None:
    r = _run("index.html", "memory")
    assert r["rows"] == 2
    assert r["quarantined"] == ["อนุมัติ", "ปฏิเสธ", "แก้ไข", "ล็อก", "ลบ"]
    assert r["core"] == ["กักกัน", "แก้ไข", "ปลดล็อก", "ลบ"]  # no pin: the store cannot
    assert r["patch"] == [["PATCH", {"status": "active", "character": "pailin"}]]
