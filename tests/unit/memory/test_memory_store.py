"""``SqliteMemory`` acceptance tests (modules.json ``memory``; ARCHITECTURE.md §6)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts.events import MemoryWritten
from aivtube.contracts.memory import MemoryItem, SlotsFull, Turn
from aivtube.contracts.types import HealthState
from aivtube.memory import (
    MemoryLocked,
    SqliteMemory,
    ViewerOptedOut,
    fts5_trigram_available,
    render_context,
)
from aivtube.testing.fakes import FakeClock, FakeEventBus

needs_trigram = pytest.mark.skipif(
    not fts5_trigram_available(), reason="SQLite without FTS5 trigram"
)


def core(text: str, **kw: Any) -> MemoryItem:
    kw.setdefault("source", "operator")
    return MemoryItem(id=None, kind="core", text=text, **kw)


def model(text: str, origin: str, kind: str = "core", **kw: Any) -> MemoryItem:
    return MemoryItem(id=None, kind=kind, text=text, source="model", origin=origin, **kw)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def make(tmp_path: Path, clock: FakeClock) -> AsyncIterator[Callable[..., SqliteMemory]]:
    stores: list[SqliteMemory] = []

    def factory(name: str = "pailin.sqlite", **kw: Any) -> SqliteMemory:
        kw.setdefault("tokenizer", lambda text: [text])  # keep pythainlp out of most tests
        store = SqliteMemory(tmp_path / name, "pailin", clock, **kw)
        stores.append(store)
        return store

    yield factory
    for s in stores:
        await s.aclose()


@pytest.fixture
async def store(make: Callable[..., SqliteMemory]) -> SqliteMemory:
    s = make()
    await s.start_session("test")
    return s


# --- Thai FTS -----------------------------------------------------------------------------------
@needs_trigram
async def test_thai_trigram_search_finds_substring(store: SqliteMemory) -> None:
    assert store.fts
    await store.remember(
        MemoryItem(id=None, kind="fact", text="ไพลินชอบกินข้าวมันไก่มาก", source="operator")
    )
    await store.remember(MemoryItem(id=None, kind="fact", text="ไพลินกลัวผี", source="operator"))
    hits = await store.search("มันไก่")
    assert [h.text for h in hits] == ["ไพลินชอบกินข้าวมันไก่มาก"]
    assert await store.search("ข้าวผัด") == []


@needs_trigram
async def test_thai_search_uses_words_for_sentences(make: Callable[..., SqliteMemory]) -> None:
    store = make(tokenizer=None)  # real pythainlp newmm word split
    await store.start_session()
    await store.remember(
        MemoryItem(id=None, kind="fact", text="ไพลินชอบกินข้าวมันไก่มาก", source="operator")
    )
    await store.remember(
        MemoryItem(id=None, kind="fact", text="สตรีมเมอร์เลี้ยงแมวชื่อส้ม", source="operator")
    )
    hits = await store.search("วันนี้กินข้าวมันไก่กันไหม", k=1)
    assert [h.text for h in hits] == ["ไพลินชอบกินข้าวมันไก่มาก"]
    hits = await store.search("แมว ของสตรีมเมอร์ ชื่ออะไร", k=1)
    assert [h.text for h in hits] == ["สตรีมเมอร์เลี้ยงแมวชื่อส้ม"]


async def test_like_fallback_when_trigram_is_unavailable(make: Callable[..., SqliteMemory]) -> None:
    store = make(fts=False)
    await store.start_session()
    assert not store.fts
    await store.remember(
        MemoryItem(id=None, kind="fact", text="ไพลินชอบกินข้าวมันไก่มาก", source="operator")
    )
    await store.remember(MemoryItem(id=None, kind="fact", text="100% แมวส้ม_จริง", source="operator"))
    assert [h.text for h in await store.search("มันไก่")] == ["ไพลินชอบกินข้าวมันไก่มาก"]
    # LIKE wildcards in the query are literal characters
    assert [h.text for h in await store.search("0% แ")] == ["100% แมวส้ม_จริง"]
    assert await store.search("_") != []
    assert await store.search("%%%") == []
    tables = await _objects(store)
    assert "memory_fts" not in tables


@needs_trigram
async def test_fts_rebuilds_rows_written_while_trigram_was_missing(
    make: Callable[..., SqliteMemory],
) -> None:
    first = make()
    await first.start_session()
    await first.remember(MemoryItem(id=None, kind="fact", text="ก่อนปิดดัชนี มันไก่", source="operator"))
    await first.aclose()
    no_fts = make(fts=False)  # e.g. an old SQLite opened the file: triggers dropped, marked stale
    await no_fts.resume_session(3600)
    await no_fts.remember(
        MemoryItem(id=None, kind="fact", text="ระหว่างนั้น ข้าวมันไก่", source="operator")
    )
    await no_fts.aclose()
    again = make()
    await again.resume_session(3600)
    texts = sorted(h.text for h in await again.search("มันไก่", k=5))
    assert texts == ["ก่อนปิดดัชนี มันไก่", "ระหว่างนั้น ข้าวมันไก่"]


async def test_short_queries_use_like_and_search_filters(store: SqliteMemory) -> None:
    await store.remember(MemoryItem(id=None, kind="fact", text="ชอบสีฟ้า", source="operator"))
    await store.remember(core("ชอบสีแดง"))
    await store.remember(model("สีเขียวจากแชท", "chat:twitch:u1", kind="fact"))  # quarantined
    assert {h.text for h in await store.search("สี", k=5)} == {"ชอบสีฟ้า", "ชอบสีแดง"}
    assert [h.text for h in await store.search("สี", k=5, kinds=["fact"])] == ["ชอบสีฟ้า"]
    assert await store.search("สี", kinds=[]) == []
    assert await store.search("   ") == []
    assert await store.search("สี", k=0) == []
    assert len(await store.search("ชอบสี", k=1)) == 1


# --- quarantine ---------------------------------------------------------------------------------
async def test_quarantine_rule(store: SqliteMemory) -> None:
    chat = await store.remember(model("แชทบอกว่าไพลินชอบแมว", "chat:twitch:u1"))
    support = await store.remember(model("ขอบคุณคนโดเนท", "support:youtube:x"))
    idle = await store.remember(model("คิดเองตอนว่าง", "idle"))
    unknown = await store.remember(model("ไม่รู้ที่มา", ""))
    voice = await store.remember(model("สตรีมเมอร์ชอบกาแฟ", "voice"))
    op = await store.remember(core("ชื่อช่องคือ Pailin Ch", origin="panel"))
    await store.upsert_viewer("twitch", "u-9", "ต้นกล้า")
    viewer = await store.remember(
        model("ต้นกล้าชอบเกมผี", "voice", kind="viewer", platform="twitch", user_id="u-9")
    )
    assert [chat.status, support.status, idle.status, unknown.status] == ["quarantined"] * 4
    assert (voice.status, op.status, viewer.status) == ("active", "active", "quarantined")
    assert chat.slot is None and voice.slot is not None
    prefix = await store.prefix_block()
    assert {m.text for m in prefix.core} == {voice.text, op.text}
    pending = {m.id for m in await store.pending_since_epoch()}
    assert pending == {voice.id, op.id}
    assert await store.viewer_facts([("twitch", "u-9")]) == []
    assert await store.search("แมว") == []
    # the operator approves: the item takes a slot and becomes pending for <new_memories>
    assert chat.id is not None
    await store.set_status(chat.id, "active", by="operator")
    approved = {m.id: m for m in (await store.prefix_block()).core}[chat.id]
    assert approved.slot is not None and approved.status == "active"
    assert chat.id in {m.id for m in await store.pending_since_epoch()}


async def test_chat_sourced_allow(make: Callable[..., SqliteMemory]) -> None:
    store = make(chat_sourced="allow")
    await store.start_session()
    chat = await store.remember(model("ข้อมูลจากแชท", "chat:twitch:u1"))
    game = await store.remember(model("ข้อมูลจากเกม", "game_context"))
    viewer = await store.remember(
        model("ข้อมูลผู้ชม", "chat", kind="viewer", platform="twitch", user_id="u1")
    )
    assert (chat.status, game.status, viewer.status) == ("active", "quarantined", "quarantined")


async def test_caller_can_only_make_status_stricter(store: SqliteMemory) -> None:
    item = await store.remember(model("จากเสียง แต่ขอกักไว้", "voice", status="quarantined"))
    assert item.status == "quarantined"
    with pytest.raises(ValueError):
        await store.remember(core("ลบแล้ว", status="deleted"))


async def test_quarantined_replace_slot_waits_for_approval(store: SqliteMemory) -> None:
    old = await store.remember(core("ความจำเดิม"))
    q = await store.remember(model("ความจำใหม่จากแชท", "chat"), replace_slot=old.slot)
    assert q.status == "quarantined" and q.slot == old.slot
    assert [m.id for m in (await store.prefix_block()).core] == [old.id]
    assert q.id is not None
    await store.set_status(q.id, "active", by="operator")
    core_now = (await store.prefix_block()).core
    assert [(m.id, m.slot) for m in core_now] == [(q.id, old.slot)]
    deleted = await store.list_memories(status="deleted")
    assert [m.id for m in deleted] == [old.id]


# --- slots ----------------------------------------------------------------------------------------
async def test_slots_full_flow(store: SqliteMemory) -> None:
    items = [await store.remember(core(f"ความจำ {i}")) for i in range(16)]
    assert sorted(m.slot or 0 for m in items) == list(range(1, 17))
    assert all(m.pinned for m in items)
    with pytest.raises(SlotsFull) as full:
        await store.remember(core("เกิน"))
    assert [m.slot for m in full.value.slots] == list(range(1, 17))
    with pytest.raises(SlotsFull):  # a quarantined write learns about it too
        await store.remember(model("เกินจากแชท", "chat"))
    # replace_slot
    new = await store.remember(core("แทนที่ช่อง 5"), replace_slot=5)
    assert new.slot == 5
    active = {m.slot: m.text for m in await store.list_memories(kind="core", status="active")}
    assert active[5] == "แทนที่ช่อง 5" and len(active) == 16
    # forget frees a slot for the next write
    assert items[2].id is not None
    await store.forget(items[2].id, by="model", reason="test")
    again = await store.remember(core("ช่องว่างแล้ว"))
    assert again.slot == items[2].slot
    # approving a quarantined core item needs a free slot
    q = await store.remember(model("รออนุมัติ", "chat"), replace_slot=7)
    assert q.id is not None
    await store.forget(new.id or 0, by="op", reason="make room")  # slot 5 is free now
    await store.set_status(q.id, "active", by="operator")  # replaces slot 7 as requested
    assert {m.slot: m.text for m in (await store.prefix_block()).core}[7] == "รออนุมัติ"


async def test_slot_validation(store: SqliteMemory) -> None:
    with pytest.raises(ValueError):
        await store.remember(core("ก" * 121))
    await store.remember(MemoryItem(id=None, kind="fact", text="ก" * 500, source="operator"))
    with pytest.raises(ValueError):
        await store.remember(core("x"), replace_slot=17)
    with pytest.raises(ValueError):
        await store.remember(
            MemoryItem(id=None, kind="fact", text="x", source="operator"), replace_slot=1
        )
    with pytest.raises(ValueError):
        await store.remember(core("   "))
    with pytest.raises(ValueError):
        await store.remember(core("x", importance=6))
    with pytest.raises(ValueError):
        await store.remember(MemoryItem(id=None, kind="viewer", text="no user", source="operator"))


async def test_duplicate_writes_reuse_the_item(store: SqliteMemory) -> None:
    a = await store.remember(core("ไพลินชอบแมว"))
    b = await store.remember(core("  ไพลินชอบแมว "))
    assert a.id == b.id and len(await store.list_memories(kind="core")) == 1


async def test_locked_items(store: SqliteMemory) -> None:
    locked = await store.remember(core("ชื่อไพลิน", locked=True))
    assert locked.id is not None
    with pytest.raises(MemoryLocked):
        await store.forget(locked.id, by="model", reason="x")
    with pytest.raises(MemoryLocked):
        await store.set_status(locked.id, "deleted", by="operator")
    with pytest.raises(MemoryLocked):
        await store.remember(core("แทนที่"), replace_slot=locked.slot)
    with pytest.raises(KeyError):
        await store.forget(9999, by="x", reason="y")
    edited = await store.edit(locked.id, by="operator", locked=False, text="ชื่อไพลินค่ะ")
    assert not edited.locked and edited.text == "ชื่อไพลินค่ะ"
    await store.forget(locked.id, by="operator", reason="unlocked")
    await store.forget(locked.id, by="operator", reason="twice is a no-op")


# --- prefix and epochs ----------------------------------------------------------------------------
async def test_prefix_digest_stable_across_unrelated_appends(store: SqliteMemory) -> None:
    await store.remember(core("ความจำหลัก"))
    base = await store.prefix_block()
    assert (
        base.digest
        == __import__("hashlib")
        .sha256(render_context(base.core, base.episodes, base.rolling_summary).encode())
        .hexdigest()
    )
    t = await store.append_turn(Turn(role="user", text="สวัสดี", source="voice"))
    await store.append_turn(Turn(role="assistant", text="สวัสดีค่ะ", source="speak"))
    await store.remember(MemoryItem(id=None, kind="fact", text="ข้อเท็จจริง", source="operator"))
    await store.upsert_viewer("twitch", "u1", "มะลิ")
    await store.remember(
        MemoryItem(
            id=None,
            kind="viewer",
            text="มะลิชอบแมว",
            platform="twitch",
            user_id="u1",
            source="operator",
        )
    )
    await store.remember(model("กักไว้", "chat"))
    await store.new_epoch("", t, "hash-1")  # same (empty) summary: new epoch, same content
    same = await store.prefix_block()
    assert same.digest == base.digest and same.epoch > base.epoch
    # each of core slots, episodes and the rolling summary changes the digest
    await store.remember(
        MemoryItem(id=None, kind="episode", text="สตรีมที่แล้วเล่นเกมผี", source="consolidation")
    )
    ep = await store.prefix_block()
    assert ep.episodes == ("สตรีมที่แล้วเล่นเกมผี",) and ep.digest != same.digest
    await store.new_epoch("สรุปใหม่", t, "hash-2")
    summ = await store.prefix_block()
    assert summ.rolling_summary == "สรุปใหม่" and summ.digest != ep.digest
    extra = await store.remember(core("ช่องใหม่"))
    changed = await store.prefix_block()
    assert changed.digest != summ.digest
    assert extra.id is not None
    await store.forget(extra.id, by="op", reason="undo")
    assert (await store.prefix_block()).digest == summ.digest


async def test_only_last_two_episodes(store: SqliteMemory) -> None:
    for i in range(4):
        await store.remember(
            MemoryItem(id=None, kind="episode", text=f"ตอนที่ {i}", source="consolidation")
        )
    assert (await store.prefix_block()).episodes == ("ตอนที่ 2", "ตอนที่ 3")


async def test_recent_turns_follow_the_epoch(store: SqliteMemory, clock: FakeClock) -> None:
    ids = []
    for i in range(4):
        clock.advance(1.0)
        ids.append(
            await store.append_turn(Turn(role="user", text=f"t{i}", source="chat", speaker="a"))
        )
    first = (await store.prefix_block()).epoch
    epoch = await store.new_epoch("สรุป t0-t1", ids[1], "h")
    assert [t.text for t in await store.session_turns()] == ["t0", "t1", "t2", "t3"]
    assert await store.session_turns(9999) == []
    turns = await store.recent_turns(epoch)
    assert [t.text for t in turns] == ["t2", "t3"]
    assert [t.text for t in await store.recent_turns(first)] == ["t0", "t1", "t2", "t3"]
    assert await store.recent_turns(9999) == []
    # ts round-trips (perf -> wall -> perf) through the clock
    assert turns[-1].ts == pytest.approx(clock.now(), abs=1e-6)
    t = Turn(
        role="assistant",
        text="x",
        source="speak",
        ts=clock.now() - 2.5,
        heard_text="x",
        interrupted=True,
        filtered=True,
        provider="local-30b",
        turn_ref="r1",
        tool_calls='[{"name":"remember"}]',
        provider_extra="{}",
    )
    tid = await store.append_turn(t)
    back = (await store.recent_turns(epoch))[-1]
    assert back.id == tid and back.ts == pytest.approx(t.ts, abs=1e-6)
    assert (back.interrupted, back.filtered, back.turn_ref, back.tool_calls) == (
        True,
        True,
        "r1",
        '[{"name":"remember"}]',
    )


async def test_concurrent_appends_keep_order(store: SqliteMemory) -> None:
    ids = await asyncio.gather(
        *(store.append_turn(Turn(role="user", text=str(i), source="chat")) for i in range(50))
    )
    assert len(set(ids)) == 50
    texts = [t.text for t in await store.recent_turns(store.epoch)]
    assert sorted(texts, key=int) == [str(i) for i in range(50)]


# --- sessions --------------------------------------------------------------------------------------
async def test_session_resume_within_max_age(
    make: Callable[..., SqliteMemory], clock: FakeClock
) -> None:
    a = make()
    sid = await a.start_session("stream")
    await a.append_turn(Turn(role="user", text="x", source="voice"))
    await a.aclose()
    clock.advance(5 * 3600)  # the core restarts after 5 h: resumed
    b = make()
    assert await b.resume_session(6 * 3600) == sid
    assert b.session_id == sid and b.epoch > 0
    clock.advance(3 * 3600)  # 8 h after the last turn, but the session is only checked once
    await b.append_turn(Turn(role="user", text="y", source="voice"))  # activity keeps it young
    await b.aclose()
    clock.advance(5 * 3600)
    c = make()
    assert await c.resume_session(6 * 3600) == sid
    await c.aclose()
    clock.advance(7 * 3600)
    d = make()
    assert await d.resume_session(6 * 3600) is None
    new = await d.start_session("next")  # closes the stale session
    assert new != sid
    rows = await _query(d, "SELECT id, ended_at IS NOT NULL FROM session ORDER BY id")
    assert rows == [(sid, 1), (new, 0)]


async def test_session_errors(make: Callable[..., SqliteMemory]) -> None:
    store = make()
    assert await store.resume_session(3600) is None
    with pytest.raises(RuntimeError):
        await store.append_turn(Turn(role="user", text="x", source="voice"))
    with pytest.raises(RuntimeError):
        await store.new_epoch("", 0, "")
    await store.end_session()  # no session: a no-op
    await store.start_session()
    await store.end_session("จบ")
    with pytest.raises(RuntimeError):
        await store.append_turn(Turn(role="user", text="x", source="voice"))
    await store.aclose()
    with pytest.raises(RuntimeError):
        await store.start_session()


# --- viewers ---------------------------------------------------------------------------------------
async def test_viewer_opt_out(store: SqliteMemory) -> None:
    await store.upsert_viewer("twitch", "u1", "มะลิ")
    await store.upsert_viewer("twitch", "u1", "มะลิ2")
    row = await store.viewer("twitch", "u1")
    assert row is not None and row["name"] == "มะลิ2" and row["messages"] == 2
    fact = await store.remember(
        MemoryItem(
            id=None,
            kind="viewer",
            text="มะลิชอบแมว",
            platform="twitch",
            user_id="u1",
            source="operator",
        )
    )
    q = await store.remember(
        model("มะลิอยู่เชียงใหม่", "chat", kind="viewer", platform="twitch", user_id="u1")
    )
    assert [m.id for m in await store.viewer_facts([("twitch", "u1")])] == [fact.id]
    assert await store.set_viewer_opt_out("twitch", "u1") == 2
    assert await store.viewer_facts([("twitch", "u1")]) == []
    with pytest.raises(ViewerOptedOut):
        await store.remember(
            MemoryItem(
                id=None,
                kind="viewer",
                text="อีก",
                platform="twitch",
                user_id="u1",
                source="operator",
            )
        )
    await store.set_viewer_opt_out("twitch", "u1", False)
    assert q.id is not None
    await store.set_status(q.id, "active", by="operator")
    assert [m.id for m in await store.viewer_facts([("twitch", "u1")])] == [q.id]
    await store.set_viewer_opt_out("twitch", "u1")
    with pytest.raises(ViewerOptedOut):
        await store.set_status(fact.id or 0, "active", by="operator")


async def test_viewer_facts_limit_and_order(store: SqliteMemory) -> None:
    for i in range(5):
        await store.remember(
            MemoryItem(
                id=None,
                kind="viewer",
                text=f"a{i}",
                platform="twitch",
                user_id="a",
                source="operator",
                importance=1 + i % 5,
            )
        )
    await store.remember(
        MemoryItem(
            id=None,
            kind="viewer",
            text="b",
            platform="youtube",
            user_id="b",
            source="operator",
            importance=5,
        )
    )
    facts = await store.viewer_facts([("twitch", "a"), ("youtube", "b"), ("twitch", "a")], limit=3)
    assert [m.text for m in facts] == ["b", "a4", "a3"]
    assert await store.viewer_facts([], limit=3) == []


# --- events, health, executor ----------------------------------------------------------------------
async def test_memory_written_events(make: Callable[..., SqliteMemory], clock: FakeClock) -> None:
    bus = FakeEventBus(clock)
    store = make(bus=bus)
    await store.start_session()
    item = await store.remember(model("จากแชท", "chat"))
    assert item.id is not None
    await store.set_status(item.id, "active", by="operator")
    await store.forget(item.id, by="operator", reason="x")
    events = bus.of_type(MemoryWritten)
    assert [(e.memory_id, e.status) for e in events] == [
        (item.id, "quarantined"),
        (item.id, "active"),
        (item.id, "deleted"),
    ]
    assert all(e.character == "pailin" for e in events)


async def test_all_io_runs_on_the_given_executor(make: Callable[..., SqliteMemory]) -> None:
    threads: set[str] = set()
    pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="mem-test")

    class Spy(concurrent.futures.Executor):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
            def wrapped() -> Any:
                threads.add(threading.current_thread().name)
                return fn(*args, **kwargs)

            return pool.submit(wrapped)

    store = make(executor=Spy())
    await store.start_session()
    await store.remember(core("x"))
    await store.prefix_block()
    await store.aclose()
    pool.shutdown()
    assert threads and all(name.startswith("mem-test") for name in threads)
    assert threading.current_thread().name not in threads


async def test_health_ok_after_open(make: Callable[..., SqliteMemory]) -> None:
    store = make()
    assert store.health().state is HealthState.STARTING
    await store.start()
    assert store.health().state is HealthState.OK and not store.degraded


# --- helpers ----------------------------------------------------------------------------------------
async def _query(store: SqliteMemory, sql: str) -> list[tuple[Any, ...]]:
    return await store._db.run(lambda conn: conn.execute(sql).fetchall(), mode="read")


async def _objects(store: SqliteMemory) -> set[str]:
    rows = await _query(store, "SELECT name FROM sqlite_master")
    return {r[0] for r in rows}


def test_sqlite_module_is_new_enough() -> None:
    assert sqlite3.sqlite_version_info >= (3, 34)


async def test_from_config(tmp_path: Path, clock: FakeClock) -> None:
    from aivtube.config import load_characters, load_config
    from aivtube.memory import OpsDb

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root, profile="ci")
    char = load_characters(cfg)["pailin"]
    store = SqliteMemory.from_config(cfg, char, clock)
    ops = OpsDb.from_config(cfg, clock=clock)
    try:
        assert store._db.path is not None and store._db.path.parts[-3:] == (
            "data",
            "memory",
            "pailin.sqlite",
        )
        assert ops._db.path == root / "data" / "ops.db"
        assert store.core_slots == cfg.memory.core_slots and store.character == "pailin"
    finally:  # never opened: nothing is written under the repo
        await store.aclose()
        await ops.aclose()
