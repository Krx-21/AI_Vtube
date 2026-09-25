"""``config/user.toml`` maintenance: schema migration and override writing (tomli-w).

Both write a ``.bak`` copy of the previous file first and replace the file atomically.
tomli-w does not keep comments, which is why the backup exists.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

import tomli_w

from aivtube.config import layers as _layers
from aivtube.config.errors import ConfigError
from aivtube.config.layers import USER_FILE, collect_layers, expand_dotted, read_toml

__all__ = ["MIGRATIONS", "dumps_toml", "migrate", "write_user_overrides"]

Migration = Callable[[dict[str, Any]], list[str]]
MIGRATIONS: dict[int, Migration] = {}
"""``MIGRATIONS[n]`` upgrades a user.toml dict from schema ``n`` to ``n + 1`` in place and
returns human-readable change notes. Empty while the schema is at version 1."""

_HEADER = (
    "# config/user.toml: your overrides of config/defaults.toml (gitignored).\n"
    "# Written by aivtube (setup, bench, panel). Comments are not kept; the previous version is\n"
    "# saved as user.toml.bak.\n\n"
)


def migrate(root: Path, *, dry_run: bool = False) -> list[str]:
    """Upgrade ``config/user.toml`` to the current schema; returns what changed.

    A missing user.toml needs nothing. A file without ``schema_version`` is stamped with the
    current version. A file from a newer aivtube raises ConfigError.
    """
    path = Path(root) / USER_FILE
    if not path.is_file():
        return []
    data = read_toml(path)
    current = _layers.CURRENT_SCHEMA_VERSION
    notes: list[str] = []
    version = data.get("schema_version")
    if version is None:
        version = current
        notes.append(f"set schema_version = {current}")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ConfigError(
            "schema_version",
            f"schema_version must be a positive integer, got {version!r}.",
            f"schema_version ต้องเป็นจำนวนเต็มบวก แต่ได้ {version!r}",
            f"Set schema_version = {current}.",
            hint_th=f"ตั้ง schema_version = {current}",
            source=str(USER_FILE),
        )
    if version > current:
        raise ConfigError(
            "schema_version",
            f"user.toml has schema {version}, newer than this aivtube ({current}).",
            f"user.toml ใช้ schema {version} ซึ่งใหม่กว่า aivtube เวอร์ชันนี้ ({current})",
            "Update aivtube, or restore user.toml.bak.",
            hint_th="อัปเดต aivtube หรือกู้ไฟล์ user.toml.bak คืน",
            source=str(USER_FILE),
        )
    while version < current:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ConfigError(
                "schema_version",
                f"No migration from schema {version} to {version + 1}.",
                f"ไม่มีวิธีย้ายจาก schema {version} ไป {version + 1}",
                "Restore defaults by renaming user.toml and run aivtube setup again.",
                hint_th="เปลี่ยนชื่อไฟล์ user.toml แล้วรัน aivtube setup ใหม่",
                source=str(USER_FILE),
            )
        notes += [f"v{version}→v{version + 1}: {n}" for n in step(data)]
        version += 1
        notes.append(f"set schema_version = {version}")
    if notes and not dry_run:
        data["schema_version"] = version
        _backup(path)
        _atomic_write(path, _HEADER + dumps_toml(data))
    return notes


def write_user_overrides(root: Path, updates: Mapping[str, Any]) -> Path:
    """Deep-merge ``updates`` into ``config/user.toml`` and save it.

    Keys may be nested tables or dotted (``{"llm.servers.local30b.placement": "pinned"}``);
    a ``None`` value deletes the key. The result is validated first (when defaults.toml
    exists), so an invalid update raises ConfigError and leaves the file untouched.
    Blocking file I/O: call it through ``asyncio.to_thread`` from the event loop.
    """
    root = Path(root)
    path = root / USER_FILE
    data = read_toml(path, required=False) or {"schema_version": _layers.CURRENT_SCHEMA_VERSION}
    merged = _plain(_apply(data, expand_dotted(updates)))
    if (root / _layers.DEFAULTS_FILE).is_file():
        from aivtube.config.load import validate_layers

        validate_layers(root, collect_layers(root, env={}, user_data=merged))
    text = _HEADER + tomli_w.dumps(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        _backup(path)
    _atomic_write(path, text)
    return path


def dumps_toml(data: Mapping[str, Any]) -> str:
    """TOML text for plain config data (``None`` dropped, tuples/Paths/enums converted)."""
    return tomli_w.dumps(_plain(data))


def _apply(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _apply(out[key], value)
        elif isinstance(value, Mapping):
            out[key] = _apply({}, value)
        else:
            out[key] = value
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k.value if isinstance(k, Enum) else k): _plain(v)
            for k, v in value.items()
            if v is not None
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value if v is not None]
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool | int | float | str | date | datetime | time):
        return value
    raise ConfigError(
        "user.toml",
        f"Cannot store a {type(value).__name__} in TOML.",
        f"บันทึกค่าชนิด {type(value).__name__} ลงไฟล์ TOML ไม่ได้",
    )


def _backup(path: Path) -> Path:
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return bak


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
