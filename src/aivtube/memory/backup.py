"""Backups through ``sqlite3.backup`` with rotation (ARCHITECTURE.md §6).

Daily backups plus one at every session end, keeping the newest ``keep`` (14) per stem. Tokens
and ``user.toml`` are archived with ``backup_files``. Everything here blocks: run it in the
database worker (``SqliteMemory.backup``/``OpsDb.backup``) or in ``asyncio.to_thread``.

File names are ``<stem>-YYYYmmdd-HHMMSS[-N].sqlite`` in local time; ``-N`` separates backups
taken within the same second. Rotation only ever touches files matching that exact pattern
for the given stem, so other characters' backups in the same folder are safe.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sqlite3
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "backup_connection",
    "backup_database_file",
    "backup_files",
    "list_backups",
    "rotate_backups",
    "safe_stem",
]

log = logging.getLogger("aivtube.memory.backup")


def safe_stem(name: str) -> str:
    """A file-name-safe stem (``[A-Za-z0-9_.-]``; anything else becomes ``_``)."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name).strip(".") or "db"


def _pattern(stem: str, suffix: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(stem)}-(?P<ts>\d{{8}}-\d{{6}})(?:-(?P<n>\d+))?{re.escape(suffix)}$"
    )


def list_backups(dest_dir: Path, stem: str, suffix: str = ".sqlite") -> list[Path]:
    """Backups of ``stem`` in ``dest_dir``, oldest first."""
    if not dest_dir.is_dir():
        return []
    pattern = _pattern(stem, suffix)
    found: list[tuple[str, int, Path]] = []
    for path in dest_dir.iterdir():
        m = pattern.match(path.name)
        if m is not None and path.is_file():
            found.append((m["ts"], int(m["n"] or 1), path))
    found.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in found]


def rotate_backups(dest_dir: Path, stem: str, keep: int, suffix: str = ".sqlite") -> list[Path]:
    """Delete all but the newest ``keep`` backups of ``stem``; returns the deleted paths."""
    backups = list_backups(dest_dir, stem, suffix)
    doomed = backups[: max(0, len(backups) - max(keep, 1))]
    removed: list[Path] = []
    for path in doomed:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:  # e.g. held open on Windows; try again at the next rotation
            log.warning("could not remove old backup %s", path, exc_info=True)
            continue
        removed.append(path)
    return removed


def _target(dest_dir: Path, stem: str, wall: float, suffix: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(wall))
    path = dest_dir / f"{stem}-{ts}{suffix}"
    n = 2
    while path.exists():
        path = dest_dir / f"{stem}-{ts}-{n}{suffix}"
        n += 1
    return path


def backup_connection(
    conn: sqlite3.Connection, dest_dir: Path, *, stem: str, wall: float, keep: int = 14
) -> Path:
    """Copy the database behind ``conn`` with ``Connection.backup`` into ``dest_dir``.

    The copy is written to a ``.tmp`` file first and renamed into place, so a crash never
    leaves a half-written file under a backup name. Then old backups are rotated.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(stem)
    final = _target(dest_dir, stem, wall, ".sqlite")
    tmp = final.with_name(final.name + ".tmp")
    target = sqlite3.connect(str(tmp))
    try:
        conn.backup(target)
    except BaseException:
        target.close()
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    target.close()
    os.replace(tmp, final)
    rotate_backups(dest_dir, stem, keep)
    return final


def backup_database_file(
    src: Path, dest_dir: Path, *, stem: str | None = None, wall: float, keep: int = 14
) -> Path:
    """Back up a database file that this process does not hold open (e.g. from the CLI)."""
    conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return backup_connection(conn, dest_dir, stem=stem or src.stem, wall=wall, keep=keep)
    finally:
        conn.close()


def backup_files(
    files: Sequence[Path], dest_dir: Path, *, stem: str = "config", wall: float, keep: int = 14
) -> Path | None:
    """Zip the existing ``files`` (tokens, ``user.toml``) into ``dest_dir`` and rotate.

    Directories are added recursively. Returns ``None`` when none of the files exist.
    """
    present: list[tuple[Path, str]] = []
    for f in files:
        if f.is_file():
            present.append((f, f.name))
        elif f.is_dir():
            present.extend(
                (p, (Path(f.name) / p.relative_to(f)).as_posix())
                for p in sorted(f.rglob("*"))
                if p.is_file()
            )
    if not present:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(stem)
    final = _target(dest_dir, stem, wall, ".zip")
    tmp = final.with_name(final.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, arcname in present:
                zf.write(path, arcname)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    os.replace(tmp, final)
    rotate_backups(dest_dir, stem, keep, ".zip")
    return final
