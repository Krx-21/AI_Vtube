"""Backups through ``sqlite3.backup`` with rotation (§6: keep 14)."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

from aivtube.contracts.memory import MemoryItem
from aivtube.memory import SqliteMemory, backup_database_file, backup_files, list_backups
from aivtube.memory.backup import rotate_backups, safe_stem
from aivtube.testing.fakes import FakeClock


async def test_backup_keeps_fourteen_and_is_restorable(tmp_path: Path) -> None:
    clock = FakeClock()
    store = SqliteMemory(tmp_path / "pailin.sqlite", "pailin", clock)
    dest = tmp_path / "backups"
    dest.mkdir()
    other = dest / "mali-20200101-000000.sqlite"  # another character's backup
    other.write_bytes(b"x")
    unrelated = dest / "pailin-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    try:
        await store.start_session()
        await store.remember(
            MemoryItem(id=None, kind="core", text="ความจำสำรอง", source="operator")
        )
        made = []
        for i in range(16):
            if i % 2:
                clock.advance(3600)  # every other backup lands in the same second
            made.append(await store.backup(dest))
    finally:
        await store.aclose()
    assert len(set(made)) == 16
    kept = list_backups(dest, "pailin")
    assert len(kept) == 14 and kept == made[-14:]
    assert other.exists() and unrelated.exists()
    assert not list(dest.glob("*.tmp"))
    conn = sqlite3.connect(kept[-1])
    try:
        rows = conn.execute("SELECT text FROM memory WHERE kind = 'core'").fetchall()
    finally:
        conn.close()
    assert rows == [("ความจำสำรอง",)]


def test_backup_database_file_and_files(tmp_path: Path) -> None:
    src = tmp_path / "ops.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()
    conn.close()
    out = backup_database_file(src, tmp_path / "bk", wall=1_760_000_000.0, keep=2)
    check = sqlite3.connect(out)
    assert check.execute("SELECT x FROM t").fetchall() == [(42,)]
    check.close()

    tokens = tmp_path / "data" / "tokens"
    tokens.mkdir(parents=True)
    (tokens / "vts_pailin.txt").write_text("secret-token", encoding="utf-8")
    user = tmp_path / "config" / "user.toml"
    user.parent.mkdir()
    user.write_text("[app]\n", encoding="utf-8")
    zips = [
        backup_files(
            [user, tokens, tmp_path / "missing.toml"],
            tmp_path / "bk",
            wall=1_760_000_000.0 + i,
            keep=3,
        )
        for i in range(5)
    ]
    assert all(z is not None for z in zips)
    kept = list_backups(tmp_path / "bk", "config", ".zip")
    assert kept == zips[-3:]
    with zipfile.ZipFile(kept[-1]) as zf:
        assert sorted(zf.namelist()) == ["tokens/vts_pailin.txt", "user.toml"]
    assert backup_files([tmp_path / "nothing"], tmp_path / "bk", wall=0.0) is None


def test_rotation_helpers(tmp_path: Path) -> None:
    assert safe_stem("pai lin/../x") == "pai_lin_.._x"
    assert list_backups(tmp_path / "missing", "a") == []
    for name in (
        "a-20260101-000000.sqlite",
        "a-20260101-000000-2.sqlite",
        "a-20250101-000000.sqlite",
    ):
        (tmp_path / name).write_bytes(b"")
    removed = rotate_backups(tmp_path, "a", keep=1)
    assert sorted(p.name for p in removed) == [
        "a-20250101-000000.sqlite",
        "a-20260101-000000.sqlite",
    ]
    assert [p.name for p in list_backups(tmp_path, "a")] == ["a-20260101-000000-2.sqlite"]
