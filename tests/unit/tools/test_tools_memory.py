"""The M1 memory tools ``remember``/``forget`` on the real store and registry (§4.9, §6)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tools_testkit import ListAudit, call, context, stimulus

from aivtube.config import load_characters, load_config
from aivtube.config.schema import MemoryConfig
from aivtube.contracts.chat import ChatSelection
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.tools import Tool, ToolContext, ToolResult
from aivtube.contracts.types import Platform, StimulusKind
from aivtube.memory import OpsDb, SqliteMemory
from aivtube.testing.fakes import (
    FakeClock,
    FakeEventBus,
    FakeMemoryStore,
    FakeSafetyGate,
    make_chat_message,
)
from aivtube.tools import (
    FORGET_SPEC,
    REMEMBER_SPEC,
    ForgetTool,
    PolicyToolRegistry,
    RememberTool,
    memory_tools,
)
from aivtube.tools.builtin.memory import chat_authors, stimulus_origin

CHAT = stimulus(StimulusKind.CHAT, source="twitch")
VOICE = stimulus(StimulusKind.VOICE)


def body(res: ToolResult) -> dict[str, Any]:
    out = json.loads(res.content)
    assert isinstance(out, dict)
    return out


@pytest.fixture
def remember() -> RememberTool:
    return RememberTool()


def test_specs_match_the_design() -> None:
    props = REMEMBER_SPEC.parameters["properties"]
    assert REMEMBER_SPEC.name == "remember" and REMEMBER_SPEC.parameters["required"] == ["text"]
    assert props["text"]["maxLength"] == 120
    assert (props["importance"]["minimum"], props["importance"]["maximum"]) == (1, 5)
    assert (props["replace_slot"]["minimum"], props["replace_slot"]["maximum"]) == (1, 16)
    assert set(props) == {"text", "about", "importance", "replace_slot"}
    assert FORGET_SPEC.parameters["required"] == ["slot"]
    tool: Tool = RememberTool()
    assert isinstance(tool, Tool) and isinstance(ForgetTool(), Tool)
    cfg = MemoryConfig(core_slots=8, slot_max_chars=80, chat_sourced="allow")
    rem, fgt = memory_tools(cfg)
    assert isinstance(rem, RememberTool) and isinstance(fgt, ForgetTool)
    assert rem.spec.parameters["properties"]["text"]["maxLength"] == 80
    assert fgt.spec.parameters["properties"]["slot"]["maximum"] == 8
    assert rem.chat_sourced == "allow" and fgt.chat_sourced == "allow"
    assert [t.spec.name for t in memory_tools()] == ["remember", "forget"]


async def test_voice_write_is_active_and_chat_write_quarantined(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
    clock: FakeClock,
) -> None:
    res = await remember({"text": "สตรีมเมอร์ชอบกาแฟดำ", "importance": 4}, make_ctx(VOICE))
    assert res.ok and body(res) == {"ok": True, "saved": "core", "slot": 1, "status": "active"}
    clock.advance(121)
    res = await remember({"text": "แชทบอกว่าไพลินเกลียดแมว"}, make_ctx(CHAT))
    assert res.ok and body(res)["status"] == "pending_review"
    items = {m.text: m for m in await memory.list_memories(kind="core")}
    voice, chat = items["สตรีมเมอร์ชอบกาแฟดำ"], items["แชทบอกว่าไพลินเกลียดแมว"]
    assert (voice.status, voice.origin, voice.importance, voice.source) == (
        "active",
        "voice",
        4,
        "model",
    )
    assert (chat.status, chat.origin) == ("quarantined", "chat:twitch")
    assert [m.text for m in (await memory.prefix_block()).core] == ["สตรีมเมอร์ชอบกาแฟดำ"]


async def test_chat_sourced_allow(
    memory: SqliteMemory, make_ctx: Callable[..., ToolContext], tmp_path: Any, clock: FakeClock
) -> None:
    permissive = SqliteMemory(tmp_path / "allow.sqlite", "pailin", clock, chat_sourced="allow")
    await permissive.start_session()
    try:
        tool = RememberTool(chat_sourced="allow")
        ctx = dataclasses.replace(make_ctx(CHAT), memory=permissive)
        res = await tool({"text": "แชทเล่าเรื่องจริง"}, ctx)
        assert body(res)["status"] == "active"
    finally:
        await permissive.aclose()


@pytest.mark.parametrize(
    "text",
    ["เบอร์ต้นกล้าคือ 081-234-5678", "อีเมล a.b@example.com", "เลขบัตร 1103700123456"],
)
async def test_pii_is_refused(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
    text: str,
) -> None:
    res = await remember({"text": text}, make_ctx(VOICE))
    assert not res.ok and body(res)["error"] == "pii"
    assert text not in res.content
    assert await memory.list_memories() == []
    # a refused write costs no budget: the next valid write goes through at once
    assert (await remember({"text": "ข้อมูลปกติ"}, make_ctx(VOICE))).ok


async def test_rate_limits(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
    clock: FakeClock,
) -> None:
    assert (await remember({"text": "หนึ่ง"}, make_ctx(VOICE))).ok
    clock.advance(60)
    res = await remember({"text": "สอง"}, make_ctx(VOICE))
    assert not res.ok and body(res) == {
        "ok": False,
        "error": "rate_limited",
        "detail": "remembering too often",
        "retry_in_s": 60,
    }
    for i in range(7):
        clock.advance(120)
        assert (await remember({"text": f"ความจำ {i}"}, make_ctx(VOICE))).ok
    clock.advance(120)
    res = await remember({"text": "เกินโควตา"}, make_ctx(VOICE))
    assert not res.ok and "limit" in body(res)["detail"]
    assert len(await memory.list_memories()) == 8
    # a new session resets the per-session budget
    await memory.end_session()
    await memory.start_session("next")
    assert (await remember({"text": "สตรีมใหม่"}, make_ctx(VOICE))).ok


async def test_slots_full_returns_the_slot_list(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
) -> None:
    for i in range(16):
        text = f"ความจำหมายเลข {i} " + "ยาว" * 20 if i == 0 else f"ความจำหมายเลข {i}"
        await memory.remember(MemoryItem(id=None, kind="core", text=text, source="operator"))
    res = await remember({"text": "ความจำใหม่"}, make_ctx(VOICE))
    assert not res.ok
    data = body(res)
    assert data["error"] == "slots_full" and "replace_slot" in data["hint"]
    assert [s["slot"] for s in data["slots"]] == list(range(1, 17))
    assert data["slots"][0]["text"].endswith("…") and len(data["slots"][0]["text"]) == 41
    # the model follows up with replace_slot straight away (SlotsFull cost no budget)
    res = await remember({"text": "ความจำใหม่", "replace_slot": 3}, make_ctx(VOICE))
    assert res.ok and body(res)["slot"] == 3
    active = {m.slot: m.text for m in await memory.list_memories(kind="core", status="active")}
    assert active[3] == "ความจำใหม่"


async def test_locked_slot_replacement_is_refused(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
) -> None:
    await memory.remember(
        MemoryItem(id=None, kind="core", text="ชื่อไพลิน", source="operator", locked=True)
    )
    res = await remember({"text": "ทับ", "replace_slot": 1}, make_ctx(VOICE))
    assert body(res) == {"ok": False, "error": "slot_locked"}


async def test_about_a_viewer_in_the_chat_block(
    remember: RememberTool,
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
    clock: FakeClock,
) -> None:
    msg = make_chat_message("ผมชอบเกมผี", user="ต้นกล้า", user_id="u-42")
    other = make_chat_message("สวัสดี", user="Mali", platform=Platform.YOUTUBE, user_id="yt-1")
    selection = ChatSelection(must_ack=(), candidates=(msg,), ambient=(other,))
    stim = dataclasses.replace(VOICE, payload={"selection": selection})
    res = await remember({"text": "ต้นกล้าชอบเกมผี", "about": "@ต้นกล้า"}, make_ctx(stim))
    assert res.ok and body(res)["saved"] == "viewer" and body(res)["status"] == "pending_review"
    (fact,) = await memory.list_memories(kind="viewer")
    assert (fact.platform, fact.user_id, fact.subject, fact.status) == (
        "twitch",
        "u-42",
        "ต้นกล้า",
        "quarantined",
    )
    clock.advance(121)
    res = await remember({"text": "Mali มาจากเชียงใหม่", "about": "mali"}, make_ctx(stim))
    assert body(res)["about"] == "Mali"
    # about= a name not in the chat block: a core memory with a subject
    clock.advance(121)
    res = await remember({"text": "แม่ของสตรีมเมอร์ชื่อสมศรี", "about": "แม่"}, make_ctx(VOICE))
    assert res.ok and body(res)["saved"] == "core"
    core = (await memory.list_memories(kind="core"))[0]
    assert core.subject == "แม่" and core.status == "active"
    # opted-out viewers are refused
    await memory.set_viewer_opt_out("twitch", "u-42")
    clock.advance(121)
    res = await remember({"text": "อีกเรื่อง", "about": "ต้นกล้า"}, make_ctx(stim))
    assert body(res) == {"ok": False, "error": "viewer_opted_out"}


def test_chat_authors_and_origin() -> None:
    a = make_chat_message("x", user="a", user_id="1")
    b = make_chat_message("y", user="b", user_id="2")
    stim = stimulus(StimulusKind.MENTION, source="twitch", messages=[a, b, a])
    assert [u.id for u in chat_authors(stim)] == ["1", "2"]
    stim = dataclasses.replace(stim, payload={"m": a, "list": [b, {"nested": [a]}]})
    assert [u.id for u in chat_authors(stim)] == ["1", "2"]
    assert stimulus_origin(stim) == "mention:twitch"
    assert stimulus_origin(VOICE) == "voice"


async def test_forget(
    memory: SqliteMemory, make_ctx: Callable[..., ToolContext], clock: FakeClock
) -> None:
    forget = ForgetTool()
    a = await memory.remember(MemoryItem(id=None, kind="core", text="ลืมได้", source="operator"))
    await memory.remember(
        MemoryItem(id=None, kind="core", text="ห้ามลืม", source="operator", locked=True)
    )
    res = await forget({"slot": a.slot}, make_ctx(CHAT))
    assert body(res)["error"] == "needs_streamer"
    assert body(await forget({"slot": 2}, make_ctx(VOICE))) == {
        "ok": False,
        "error": "slot_locked",
        "slot": 2,
    }
    assert body(await forget({"slot": 9}, make_ctx(VOICE)))["error"] == "slot_empty"
    res = await forget({"slot": a.slot}, make_ctx(VOICE))
    assert body(res) == {"ok": True, "forgot_slot": 1}
    assert [m.text for m in (await memory.prefix_block()).core] == ["ห้ามลืม"]
    res = await ForgetTool(chat_sourced="allow")({"slot": 5}, make_ctx(CHAT))
    assert body(res)["error"] == "slot_empty"  # allowed to try; the slot is just empty


async def test_through_the_registry_with_audit(
    memory: SqliteMemory,
    ops: OpsDb,
    make_ctx: Callable[..., ToolContext],
    events: FakeEventBus,
    clock: FakeClock,
) -> None:
    gate = FakeSafetyGate(["คำหยาบ"])
    reg = PolicyToolRegistry(
        memory_tools(),
        gate=gate,
        ops=ops,
        bus=events,
        clock=clock,
        enabled_by_character={"pailin": ["remember", "forget"]},
    )
    assert [s.name for s in reg.specs("pailin")] == ["remember", "forget"]
    ctx = make_ctx(CHAT)
    res = await reg.execute(call("remember", {"text": "แชทสอนคำหยาบ"}), ctx)
    assert not res.ok and "blocked" in res.content
    assert gate.arg_checks[-1][0] == "memory"
    res = await reg.execute(call("remember", {"text": "ไพลินชอบฝนตก", "importance": 9}), ctx)
    assert not res.ok and "importance: maximum is 5" in json.loads(res.content)["details"]
    res = await reg.execute(call("remember", {"text": "ไพลินชอบฝนตก"}), ctx)
    assert res.ok and json.loads(res.content)["status"] == "pending_review"
    res = await reg.execute(call("forget", {"slot": 1}), make_ctx(VOICE))
    assert not res.ok  # nothing active in slot 1 (the chat write is quarantined)
    rows = await ops.audit_rows("tool_audit")
    assert [(r["tool"], r["verdict"]) for r in rows] == [
        ("remember", "blocked"),
        ("remember", "invalid"),
        ("remember", "ok"),
        ("forget", "error"),
    ]
    assert "คำหยาบ" not in rows[0]["args"]
    reg.set_mode("dry_run")
    res = await reg.execute(call("remember", {"text": "ลองเฉยๆ"}), make_ctx(VOICE))
    assert res.ok and len(await memory.list_memories()) == 1


async def test_works_with_the_fake_store(clock: FakeClock, events: FakeEventBus) -> None:
    fake = FakeMemoryStore("pailin", clock)
    await fake.start_session()
    res = await RememberTool()({"text": "ใช้กับ fake ได้"}, context(fake, clock, events, VOICE))
    assert res.ok and body(res)["slot"] == 1


async def test_list_audit_sink_collects_rows(
    memory: SqliteMemory,
    make_ctx: Callable[..., ToolContext],
    events: FakeEventBus,
    clock: FakeClock,
) -> None:
    audit = ListAudit()
    reg = PolicyToolRegistry(
        memory_tools(),
        gate=FakeSafetyGate(),
        ops=audit,
        bus=events,
        clock=clock,
        enabled_by_character={"pailin": ["remember"]},
    )
    await reg.execute(call("remember", {"text": "จดไว้"}), make_ctx(VOICE))
    await reg.execute(call("forget", {"slot": 1}), make_ctx(VOICE))
    assert audit.verdicts == ["ok", "unknown"]


def test_registry_from_config(clock: FakeClock, events: FakeEventBus) -> None:
    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root, profile="ci")
    chars = load_characters(cfg)
    reg = PolicyToolRegistry.from_config(
        memory_tools(cfg.memory),
        cfg.tools,
        list(chars.values()),
        gate=FakeSafetyGate(),
        ops=ListAudit(),
        bus=events,
        clock=clock,
    )
    assert [s.name for s in reg.specs("pailin")] == ["remember", "forget"]
    assert reg.mode == cfg.tools.mode and reg.approval_timeout_s == 20.0
