"""Pure panel helpers: command parsing, hotkeys, alert ingest, JSON and latency stats."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any

import pytest

from aivtube.contracts.control import OpKind
from aivtube.contracts.types import MsgKind, Platform
from aivtube.panel._json import dumps, to_jsonable
from aivtube.panel.commands import (
    HOTKEYS,
    hotkey_command,
    hotkey_names,
    parse_op_command,
    validate_args,
)
from aivtube.panel.ingest import MAX_TEXT_CHARS, alert_message, chat_message
from aivtube.panel.stats import Budget, percentile, trace_summary, waterfall_row
from aivtube.testing.fakes import FakeClock

# --- commands -------------------------------------------------------------------------------


def test_parse_op_command_normalises_kind_and_args() -> None:
    cmd = parse_op_command(
        {"kind": "SAY", "args": {"text": "  สวัสดีค่ะ  "}, "character": "pailin"}, operator="panel"
    )
    assert cmd.kind is OpKind.SAY
    assert cmd.args == {"text": "สวัสดีค่ะ"}
    assert cmd.character == "pailin"
    assert cmd.operator == "panel"
    assert cmd.id


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"kind": "explode"},
        {"kind": "say"},
        {"kind": "say", "args": {"text": "   "}},
        {"kind": "say", "args": {"text": "x" * 601}},
        {"kind": "mic_mode", "args": {"mode": "loud"}},
        {"kind": "ptt", "args": {"active": "maybe"}},
        {"kind": "tools_mode", "args": {"mode": "yolo"}},
        {"kind": "memory_status", "args": {"id": 1, "status": "gone"}},
        {"kind": "memory_edit", "args": {"id": "abc"}},
        {"kind": "restart", "args": {}},
        {"kind": "freeze", "args": ["not", "a", "map"]},
        {"kind": "freeze", "character": ""},
    ],
)
def test_parse_op_command_rejects_bad_bodies(body: object) -> None:
    with pytest.raises(ValueError):
        parse_op_command(body, operator="panel")


def test_validate_args_coerces_booleans_and_ids() -> None:
    assert validate_args(OpKind.PTT, {"active": "true"}) == {"active": True}
    assert validate_args(OpKind.CHAT_INTAKE, {}) == {"on": True}
    assert validate_args(OpKind.TOOL_ENABLE, {"name": "remember", "enabled": 0}) == {
        "name": "remember",
        "enabled": False,
    }
    assert validate_args(OpKind.MEMORY_STATUS, {"id": "7", "status": "active"})["id"] == 7
    # kinds the panel does not model pass through untouched
    assert validate_args(OpKind.APPROVE, {"x": 1}) == {"x": 1}


def test_validate_args_for_memory_edits_and_tool_approvals() -> None:
    edit = validate_args(
        OpKind.MEMORY_EDIT, {"id": 3, "importance": "4", "locked": "off", "text": "ชอบแมว"}
    )
    assert edit == {"id": 3, "importance": 4, "locked": False, "text": "ชอบแมว"}
    for bad in ({"importance": 9}, {"importance": True}, {"text": 5}, {"pinned": "maybe"}):
        with pytest.raises(ValueError):
            validate_args(OpKind.MEMORY_EDIT, {"id": 3, **bad})
    assert validate_args(OpKind.APPROVE, {"request": "ap1"}) == {
        "request": "ap1",
        "approved": True,
    }
    assert validate_args(OpKind.APPROVE, {"request": "ap1", "approved": "no"})["approved"] is False
    with pytest.raises(ValueError):
        validate_args(OpKind.APPROVE, {"request": ""})


def test_parse_op_command_takes_a_short_operator_label() -> None:
    cmd = parse_op_command({"kind": "freeze", "operator": "launcher"}, operator="panel")
    assert cmd.operator == "launcher"
    assert parse_op_command({"kind": "freeze"}, operator="panel").operator == "panel"
    for label in ("Launcher", "", "x" * 40, 5, "a b"):
        with pytest.raises(ValueError):
            parse_op_command({"kind": "freeze", "operator": label}, operator="panel")


def test_hotkeys_map_to_commands_and_toggles_read_the_snapshot() -> None:
    assert hotkey_command("freeze", {}).kind is OpKind.FREEZE
    assert hotkey_command("ptt-down", {}).args == {"active": True}
    assert hotkey_command("mute_toggle", {"muted": False}).kind is OpKind.MUTE
    assert hotkey_command("mute_toggle", {"muted": True}).kind is OpKind.UNMUTE
    deafen = hotkey_command("deafen_toggle", {"mic_mode": "ptt"})
    assert (deafen.kind, deafen.args) == (OpKind.MIC_MODE, {"mode": "deafened"})
    undeafen = hotkey_command(
        "deafen_toggle", {"mic_mode": "deafened", "mic_mode_before_deafen": "ptt"}
    )
    assert undeafen.args == {"mode": "ptt"}
    assert hotkey_command("skip", {}, character="pailin").character == "pailin"
    with pytest.raises(KeyError):
        hotkey_command("self_destruct", {})
    assert set(HOTKEYS) <= set(hotkey_names())
    assert "mute_toggle" in hotkey_names()


# --- ingest ---------------------------------------------------------------------------------


def test_alert_message_builds_a_donation(fake_clock: FakeClock) -> None:
    msg = alert_message(
        {"user": "ต้นกล้า", "amount": 100, "currency": "THB", "text": "เป็นกำลังใจให้นะ", "id": 42},
        clock=fake_clock,
    )
    assert msg.kind is MsgKind.DONATION
    assert msg.platform is Platform.ALERT
    assert msg.user.name == "ต้นกล้า"
    assert (msg.amount, msg.currency, msg.value_usd) == (100.0, "THB", 0.0)
    assert msg.id == "alert:42"
    assert msg.ts == msg.received == fake_clock.now()


def test_alert_message_kinds_and_value(fake_clock: FakeClock) -> None:
    sub = alert_message({"user": "bob", "kind": "resub", "months": 7}, clock=fake_clock)
    assert sub.kind is MsgKind.SUB
    assert sub.user.is_sub and sub.user.sub_months == 7
    bits = alert_message({"user": "amy", "kind": "cheer", "amount": 250, "currency": "bits"},
                         clock=fake_clock)  # fmt: skip
    assert bits.kind is MsgKind.DONATION and bits.value_usd == 2.5
    usd = alert_message({"user": "amy", "amount": 5, "currency": "USD"}, clock=fake_clock)
    assert usd.value_usd == 5.0
    raid = alert_message({"user": "crew", "kind": "raid", "platform": "twitch"}, clock=fake_clock)
    assert raid.kind is MsgKind.RAID and raid.platform is Platform.TWITCH
    text = alert_message({"user": "x", "text": "hi"}, clock=fake_clock)
    assert text.kind is MsgKind.TEXT
    long = alert_message({"user": "x", "amount": 1, "text": "ก" * 2000}, clock=fake_clock)
    assert len(long.text) == MAX_TEXT_CHARS


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user": ""},
        {"user": "a", "amount": -1},
        {"user": "a", "amount": "lots"},
        {"user": "a", "amount": float("nan")},
        {"user": "a", "amount": True},
        {"user": "a", "kind": "bribe"},
        {"user": "a", "platform": "myspace"},
        {"user": "a", "months": -2},
        {"user": "a", "months": "many"},
        {"user": "a", "text": 5},
        {"user": "a", "id": {"x": 1}},
    ],
)
def test_alert_message_rejects_bad_payloads(
    payload: dict[str, object], fake_clock: FakeClock
) -> None:
    with pytest.raises(ValueError):
        alert_message(payload, clock=fake_clock)


def test_chat_message_defaults_to_console(fake_clock: FakeClock) -> None:
    msg = chat_message("tom", " hello   there ", clock=fake_clock)
    assert msg.platform is Platform.CONSOLE and msg.kind is MsgKind.TEXT
    assert msg.text == "hello there"
    assert msg.user.id == "console:tom"
    other = chat_message("tom", "again", clock=fake_clock)
    assert other.id != msg.id
    with pytest.raises(ValueError):
        chat_message("tom", "  ", clock=fake_clock)


# --- json -----------------------------------------------------------------------------------


def test_to_jsonable_handles_dataclasses_enums_and_non_finite(fake_clock: FakeClock) -> None:
    msg = chat_message("tom", "hi", clock=fake_clock, raw={"secret": "tags"})
    data = to_jsonable({"m": msg, "k": OpKind.FREEZE, "x": math.inf, "t": (1, 2)})
    assert data["k"] == "freeze" and data["x"] is None and data["t"] == [1, 2]
    assert "raw" not in data["m"] and data["m"]["platform"] == "console"
    text = dumps({"nan": math.nan, "thai": "ไพลิน"})
    assert json.loads(text) == {"nan": None, "thai": "ไพลิน"}
    assert "ไพลิน" in text  # not \u-escaped


# --- stats ----------------------------------------------------------------------------------


def test_percentile_interpolates() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert percentile([1, math.nan, 3], 50) == 2.0


def _trace(
    i: int, ttfa: float, *, kind: str = "voice", opener: str = "", **kw: object
) -> dict[str, Any]:
    return {
        "turn_id": f"t{i}",
        "kind": kind,
        "character": "pailin",
        "stages_ms": {"vad_end": 0.0, "llm_first_token": ttfa / 2, "first_audible": ttfa},
        "ttfa_ms": ttfa,
        "opener": opener,
        **kw,
    }


def test_trace_summary_badges_against_budget() -> None:
    traces = [_trace(i, 1000.0 + i) for i in range(10)] + [_trace(99, 500.0, kind="chat")]
    summary = trace_summary(traces)
    assert len(summary["rows"]) == 11
    voice = summary["badges"]["voice"]
    assert voice["status"] == "ok" and voice["n"] == 10
    assert voice["p50_ms"] == pytest.approx(1004.5)
    assert summary["badges"]["chat"]["status"] == "ok"
    slow = trace_summary([_trace(i, 2500.0) for i in range(5)])
    assert slow["badges"]["voice"]["status"] == "over"
    custom = trace_summary([_trace(0, 1500.0)], budgets={"voice": Budget(1300.0, 2200.0)})
    assert custom["badges"]["voice"]["status"] == "over"
    assert trace_summary([])["badges"]["voice"]["status"] == "none"


def test_trace_summary_keeps_the_last_n() -> None:
    summary = trace_summary([_trace(i, 1000.0) for i in range(30)], n=20)
    assert summary["rows"][0]["turn_id"] == "t10"


def test_cache_and_opener_alarms() -> None:
    # llama.cpp: prompt_n = tokens processed now, cache_n = tokens reused from the KV cache
    cold = [_trace(i, 900.0, prompt_n=300, cache_n=100) for i in range(5)]
    summary = trace_summary(cold)
    assert {a["kind"] for a in summary["alarms"]} == {"cache_ratio"}
    assert summary["alarms"][0]["value"] == 0.25
    assert summary["cache"] == {"ratio": 0.25, "turns": 5, "min": 0.85}
    warm = [_trace(i, 900.0, prompt_n=40, cache_n=1960) for i in range(5)]
    assert trace_summary(warm)["alarms"] == []
    assert trace_summary(warm)["cache"]["ratio"] == 0.98
    # cache_n / prompt_n would read 0.9 / 0.1 = 9.0 here; the ratio must stay a share
    assert (
        trace_summary([_trace(0, 900.0, prompt_n=10, cache_n=90)])["rows"][0]["cache_ratio"] == 0.9
    )
    short = trace_summary(cold[:4])
    assert short["alarms"] == [] and short["cache"]["turns"] == 4
    assert trace_summary([_trace(0, 900.0)])["cache"] is None
    same = [_trace(i, 900.0, opener="ว้าวววว") for i in range(4)] + [
        _trace(10 + i, 900.0, opener=f"o{i}") for i in range(6)
    ]
    alarm = trace_summary(same)["alarms"]
    assert [a["kind"] for a in alarm] == ["opener"] and alarm[0]["opener"] == "ว้าวววว"
    few = [_trace(i, 900.0, opener="x") for i in range(3)]
    assert trace_summary(few)["alarms"] == []


def test_waterfall_row_reads_ops_rows_and_orders_stages() -> None:
    row = waterfall_row(
        {
            "turn_id": "t1",
            "kind": "chat",
            "stages": {"done": 2000, "decision_start": 0, "first_audible": 900.5, "bad": "x"},
            "ttfa_ms": None,
            "prompt_n": 10,
            "cache_n": 90,
        }
    )
    assert list(row["stages_ms"]) == ["decision_start", "first_audible", "done"]
    assert row["ttfa_ms"] == 900.5
    assert row["cache_ratio"] == 0.9


def test_panel_helpers_import_without_aiohttp() -> None:
    code = (
        "import sys, aivtube.panel.commands, aivtube.panel.ingest, aivtube.panel.stats, "
        "aivtube.panel; print('aiohttp' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "False"
