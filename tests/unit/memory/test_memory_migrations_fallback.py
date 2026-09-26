"""Migrations (``user_version``) and the busy/corrupt fallback to RAM (§6, §2.8)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aivtube.contracts.events import HealthChanged
from aivtube.contracts.memory import MemoryItem, Turn
from aivtube.contracts.types import HealthState
from aivtube.memory import SqliteMemory, fts5_trigram_available
from aivtube.memory._db import (
    DatabaseCorrupt,
    Migration,
    SchemaTooNew,
    apply_migrations,
    classify_error,
    load_migrations,
)
from aivtube.testing.fakes import FakeClock, FakeEventBus

EXPECTED_TABLES = {"meta", "session", "epoch", "turn", "memory", "viewer", "runtime_state"}


def _schema(path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(path)
    try:
        return sorted(conn.execute("SELECT type, name FROM sqlite_master").fetchall())
    finally:
        conn.close()


def _pragma(path: Path, name: str) -> object:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


async def test_migrations_are_numbered_and_idempotent(tmp_path: Path) -> None:
    mem = load_migrations("memory")
    assert [m.version for m in mem] == list(range(1, len(mem) + 1))
    assert [m.version for m in load_migrations("ops")] == [1]
    path = tmp_path / "pailin.sqlite"
    for _ in range(3):
        store = SqliteMemory(path, "pailin", FakeClock())
        await store.start_session()
        await store.remember(MemoryItem(id=None, kind="core", text="x", source="operator"))
        await store.aclose()
    assert _pragma(path, "user_version") == len(mem)
    assert _pragma(path, "journal_mode") == "wal"
    names = {n for _, n in _schema(path)}
    assert names >= EXPECTED_TABLES
    assert {"turn_epoch", "memory_core_slot", "memory_viewer"} <= names
    if fts5_trigram_available():
        assert {"memory_fts", "memory_ai", "memory_ad", "memory_au"} <= names
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        before = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        assert apply_migrations(conn, mem) == len(mem)
        assert apply_migrations(conn, mem) == len(mem)
        assert conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == before
    finally:
        conn.close()


def test_the_memory_schema_matches_architecture() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        apply_migrations(conn, load_migrations("memory"))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memory)")]
        assert cols == [
            "id",
            "kind",
            "slot",
            "subject",
            "platform",
            "user_id",
            "text",
            "importance",
            "source",
            "origin",
            "origin_turn_id",
            "status",
            "pinned",
            "locked",
            "epoch_seen",
            "created_at",
            "updated_at",
            "last_used_at",
            "uses",
        ]
        turn_cols = [r[1] for r in conn.execute("PRAGMA table_info(turn)")]
        assert turn_cols[-3:] == ["tool_calls", "provider_extra", "tokens"]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO memory(kind, text, source, created_at, updated_at) "
                "VALUES('bogus', 'x', 'model', 0, 0)"
            )
    finally:
        conn.close()


def test_a_failing_migration_rolls_back() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        good = Migration(1, "t_0001_a.sql", "CREATE TABLE a(x);")
        bad = Migration(2, "t_0002_b.sql", "CREATE TABLE b(x); INSERT INTO nope VALUES(1);")
        with pytest.raises(sqlite3.OperationalError):
            apply_migrations(conn, [good, bad])
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert names == {"a"}
        assert not conn.in_transaction
        with pytest.raises(SchemaTooNew):
            apply_migrations(conn, [])
    finally:
        conn.close()


def test_error_classification() -> None:
    busy = sqlite3.OperationalError("database is locked")
    busy.sqlite_errorcode = 5  # type: ignore[attr-defined]
    notadb = sqlite3.DatabaseError("file is not a database")
    notadb.sqlite_errorcode = 26  # type: ignore[attr-defined]
    missing = sqlite3.OperationalError("no such table: x")
    missing.sqlite_errorcode = 1  # type: ignore[attr-defined]
    assert classify_error(busy) == "busy"
    assert classify_error(notadb) == "corrupt"
    assert classify_error(DatabaseCorrupt("quick_check")) == "corrupt"
    assert classify_error(missing) is None
    assert classify_error(sqlite3.IntegrityError("UNIQUE")) is None
    assert classify_error(ValueError("x")) is None
    assert classify_error(sqlite3.OperationalError("database is locked")) == "busy"


async def test_corrupt_file_falls_back_to_ram(tmp_path: Path) -> None:
    path = tmp_path / "pailin.sqlite"
    path.write_bytes(b"this is not a database at all" * 200)
    clock = FakeClock()
    bus = FakeEventBus(clock)
    store = SqliteMemory(path, "pailin", clock, bus=bus)
    try:
        sid = await store.start_session()
        assert store.degraded
        health = store.health()
        assert health.state is HealthState.DEGRADED and "not persisted" in health.detail
        events = bus.of_type(HealthChanged)
        assert len(events) == 1 and events[0].health.state is HealthState.DEGRADED
        await store.append_turn(Turn(role="user", text="ยังคุยได้", source="voice"))
        await store.remember(MemoryItem(id=None, kind="core", text="ยังจำได้", source="operator"))
        assert [m.text for m in (await store.prefix_block()).core] == ["ยังจำได้"]
        assert sid == store.session_id
        backup = await store.backup(tmp_path / "bk")
        assert backup.exists()
    finally:
        await store.aclose()
    assert path.read_bytes().startswith(b"this is not a database")  # the file is left alone


async def test_newer_schema_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "pailin.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    store = SqliteMemory(path, "pailin", FakeClock())
    try:
        await store.start_session()
        assert store.degraded and "newer" in store.health().detail
    finally:
        await store.aclose()
    assert _pragma(path, "user_version") == 99


async def test_busy_at_open_retries_then_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "pailin.sqlite"
    first = SqliteMemory(path, "pailin", FakeClock())
    await first.start_session()
    await first.aclose()
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("PRAGMA journal_mode = DELETE")  # a rollback journal lets a lock block readers
    blocker.execute("BEGIN EXCLUSIVE")
    store = SqliteMemory(path, "pailin", FakeClock(), busy_timeout_ms=50)
    try:
        await store.start_session()
        assert store.degraded and "busy" in store.health().detail
    finally:
        await store.aclose()
        blocker.execute("ROLLBACK")
        blocker.close()


async def test_busy_at_runtime_seeds_the_fallback(tmp_path: Path) -> None:
    path = tmp_path / "pailin.sqlite"
    clock = FakeClock()
    store = SqliteMemory(path, "pailin", clock, busy_timeout_ms=50)
    blocker = sqlite3.connect(path, isolation_level=None, timeout=5)
    try:
        sid = await store.start_session("live")
        await store.remember(MemoryItem(id=None, kind="core", text="ความจำสำคัญ", source="operator"))
        t = await store.append_turn(Turn(role="user", text="ก่อนล็อก", source="voice"))
        await store.new_epoch("สรุปล่าสุด", t, "h")
        before = await store.prefix_block()
        blocker.execute("BEGIN IMMEDIATE")  # another writer holds the write lock
        await store.append_turn(Turn(role="user", text="ระหว่างล็อก", source="voice"))
        assert store.degraded
        after = await store.prefix_block()
        assert after.digest == before.digest and after.epoch == before.epoch
        assert store.session_id == sid
        assert [t.text for t in await store.recent_turns(after.epoch)] == ["ระหว่างล็อก"]
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        await store.aclose()


async def test_programming_errors_do_not_trigger_the_fallback(tmp_path: Path) -> None:
    store = SqliteMemory(tmp_path / "pailin.sqlite", "pailin", FakeClock())
    try:
        with pytest.raises(sqlite3.OperationalError):
            await store._db.run(lambda conn: conn.execute("SELECT * FROM nope"), mode="read")
        assert not store.degraded
    finally:
        await store.aclose()
