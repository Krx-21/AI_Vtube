"""``aivtube db {backup,restore}``: SQLite backups of the memory and ops databases (§6).

Backups use SQLite's online backup API (safe while the database is open elsewhere), are named
``<stem>-YYYYmmdd-HHMMSS[-N].sqlite`` in ``memory.backup_dir`` (the naming the memory package's
own rotation uses) and are rotated to ``memory.backup_keep`` per database.

A restore refuses to run while aivtube is running (its panel or emergency port answers),
checks the backup with ``PRAGMA integrity_check``, backs up the current database first, and
then copies the backup over it.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

__all__ = ["DbError", "db_backup", "db_restore", "list_backups", "target_databases"]

log = logging.getLogger("aivtube.ops.db")

_NAME = re.compile(r"^(?P<stem>.+)-(?P<ts>\d{8}-\d{6})(?:-(?P<n>\d+))?\.sqlite$")


class DbError(RuntimeError):
    """A backup or restore could not be done (the message says why, in English and Thai)."""


def target_databases(root: Path) -> dict[str, Path]:
    """``{stem: path}`` of every database aivtube keeps: one per character, plus ops."""
    from aivtube.config import load_character, load_config

    cfg = load_config(root)
    out: dict[str, Path] = {}
    for cid in cfg.characters:
        char = load_character(cfg.root, cid, app=cfg)
        path = char.resolve_path(char.memory.db)
        out[path.stem] = path
    ops = cfg.resolve_path(cfg.memory.ops_db)
    out[ops.stem] = ops
    return out


def _backup_dir(root: Path) -> tuple[Path, int]:
    from aivtube.config import load_config

    cfg = load_config(root)
    return cfg.resolve_path(cfg.memory.backup_dir), cfg.memory.backup_keep


def list_backups(folder: Path, stem: str) -> list[Path]:
    """Backups of ``stem`` in ``folder``, oldest first."""
    if not folder.is_dir():
        return []
    found: list[tuple[str, int, Path]] = []
    for path in folder.iterdir():
        m = _NAME.match(path.name)
        if m and m["stem"] == stem and path.is_file():
            found.append((m["ts"], int(m["n"] or 1), path))
    return [p for _, _, p in sorted(found)]


def _ro_uri(path: Path) -> str:
    """A read-only SQLite URI (percent-encoded, so ``#``/``?``/spaces in folders work)."""
    return path.resolve().as_uri() + "?mode=ro"


def _copy_db(src: Path, dst: Path) -> None:
    """Online-backup ``src`` into ``dst`` (overwritten)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(_ro_uri(src), uri=True, timeout=10.0)
    try:
        target = sqlite3.connect(dst, timeout=10.0)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _new_name(folder: Path, stem: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stem}-{ts}.sqlite"
    n = 2
    while path.exists():
        path = folder / f"{stem}-{ts}-{n}.sqlite"
        n += 1
    return path


def db_backup(root: Path) -> list[Path]:
    """Back up every database that exists; returns the new backup files. Blocking."""
    root = Path(root)
    folder, keep = _backup_dir(root)
    made: list[Path] = []
    for stem, path in target_databases(root).items():
        if not path.is_file():
            continue
        dest = _new_name(folder, stem)
        _copy_db(path, dest)
        made.append(dest)
        for old in list_backups(folder, stem)[: -max(1, keep)]:
            with contextlib.suppress(OSError):
                old.unlink()
    return made


def _running(root: Path) -> bool:
    from aivtube.config.layers import collect_layers
    from aivtube.launcher.heartbeat import port_open

    ports = collect_layers(root).merged.get("ports", {})
    return any(
        isinstance(ports.get(k), int) and port_open("127.0.0.1", int(ports[k]), 0.3)
        for k in ("panel", "emergency")
    )


def db_restore(
    root: Path, backup: Path, *, is_running: Callable[[Path], bool] | None = None
) -> None:
    """Restore ``backup`` over the database it was taken from. Raises ``DbError``."""
    root = Path(root)
    backup = Path(backup)
    m = _NAME.match(backup.name)
    if m is None or not backup.is_file():
        raise DbError(f"{backup} is not an aivtube backup / ไม่ใช่ไฟล์สำรองของ aivtube")
    targets = target_databases(root)
    target = targets.get(m["stem"])
    if target is None:
        raise DbError(f"no database named {m['stem']!r} / ไม่มีฐานข้อมูลชื่อ {m['stem']!r}")
    if (is_running or _running)(root):
        raise DbError("aivtube is running: stop it first / ปิด aivtube ก่อนแล้วค่อยกู้ข้อมูล")
    con = sqlite3.connect(_ro_uri(backup), uri=True)
    try:
        verdict = con.execute("PRAGMA integrity_check").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise DbError(f"{backup.name} is damaged: {exc} / ไฟล์สำรองเสียหาย") from exc
    finally:
        con.close()
    if verdict != "ok":
        raise DbError(f"{backup.name} failed the integrity check: {verdict} / ไฟล์สำรองเสียหาย")
    if target.is_file():
        folder, _ = _backup_dir(root)
        safety = _new_name(folder, m["stem"])
        _copy_db(target, safety)
        log.info("current %s saved to %s before the restore", target.name, safety)
    _copy_db(backup, target)
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            target.with_name(target.name + suffix).unlink()


def main(argv: list[str] | None = None) -> int:
    """``aivtube db {backup,restore <file>,list}``."""
    import argparse

    from aivtube.config import find_root

    p = argparse.ArgumentParser(prog="aivtube db")
    p.add_argument("--root", type=Path, default=None, help="the AI_Vtube folder")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("backup", help="back up every database now")
    restore = sub.add_parser("restore", help="restore one backup (aivtube must be stopped)")
    restore.add_argument("backup", type=Path)
    sub.add_parser("list", help="list the backups")
    args = p.parse_args(argv)
    root = Path(args.root) if args.root else find_root()
    try:
        if args.action == "backup":
            made = db_backup(root)
            for path in made:
                print(f"✔ {path}")
            if not made:
                print("no database yet / ยังไม่มีฐานข้อมูล")
        elif args.action == "restore":
            db_restore(root, args.backup)
            print(f"✔ restored {args.backup.name} / กู้ข้อมูลแล้ว")
        else:
            folder, _ = _backup_dir(root)
            for stem in target_databases(root):
                for path in list_backups(folder, stem):
                    print(path)
    except DbError as exc:
        print(f"✖ {exc}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
