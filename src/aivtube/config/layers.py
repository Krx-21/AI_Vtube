"""Raw config layering (ARCHITECTURE.md §8), standard library only.

Layers, later wins: ``config/defaults.toml`` → the active ``[profiles.<name>]`` (deep-merged)
→ ``config/user.toml`` → environment ``AIVTUBE__SECTION__KEY`` → CLI overrides.
Character files (layer 3) are character-scoped and handled by ``load_character``; user,
env and CLI layers reach them through ``character_overrides.<id>``.

The stdlib-only launcher can call :func:`collect_layers` without importing pydantic.
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivtube.config.errors import ConfigError

__all__ = [
    "CHARACTER_PLACEHOLDER",
    "CURRENT_SCHEMA_VERSION",
    "DEFAULTS_FILE",
    "ENV_PREFIX",
    "USER_FILE",
    "Layers",
    "collect_layers",
    "deep_merge",
    "env_overrides",
    "expand_character",
    "expand_dotted",
    "find_root",
    "parse_env_value",
    "read_toml",
]

CURRENT_SCHEMA_VERSION = 1
DEFAULTS_FILE = Path("config") / "defaults.toml"
USER_FILE = Path("config") / "user.toml"
ENV_PREFIX = "AIVTUBE__"
CHARACTER_PLACEHOLDER = "{character}"
DEFAULT_PROFILE = "stream"

_ENV_PART = re.compile(r"^[a-z0-9_-]+$")


def read_toml(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Parse a TOML file. A missing optional file gives ``{}``; errors become ConfigError."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        if not required:
            return {}
        raise ConfigError(
            str(path),
            "Config file not found.",
            "ไม่พบไฟล์ตั้งค่า",
            "Run the command from the AI_Vtube folder, or restore the file from git.",
            hint_th="รันคำสั่งจากโฟลเดอร์ AI_Vtube หรือกู้ไฟล์คืนจาก git",
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            str(path),
            f"TOML syntax error: {exc}",
            f"ไฟล์ TOML เขียนผิดรูปแบบ: {exc}",
            "Fix the line mentioned above (strings need quotes, tables need [brackets]).",
            hint_th="แก้บรรทัดที่ระบุ (ข้อความต้องอยู่ในเครื่องหมายคำพูด ชื่อตารางต้องอยู่ใน [วงเล็บ])",
        ) from None
    except OSError as exc:
        raise ConfigError(
            str(path),
            f"Cannot read the config file: {exc}",
            f"อ่านไฟล์ตั้งค่าไม่ได้: {exc}",
            "Check the file permissions.",
            hint_th="ตรวจสอบสิทธิ์การเข้าถึงไฟล์",
        ) from None


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``. Tables merge; anything else
    (including lists) is replaced. Neither input is modified."""
    out: dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in over.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def expand_dotted(data: Mapping[str, Any]) -> dict[str, Any]:
    """Turn ``{"llm.chain": x}`` into ``{"llm": {"chain": x}}``, recursively."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        parts = str(key).split(".")
        if any(not p for p in parts):
            raise ConfigError(
                str(key),
                f"Invalid override key {key!r}.",
                f"คีย์สำหรับแก้ค่าไม่ถูกต้อง {key!r}",
                "Use dotted keys such as 'llm.chain'.",
                hint_th="ใช้คีย์แบบมีจุด เช่น 'llm.chain'",
            )
        leaf = expand_dotted(value) if isinstance(value, Mapping) else value
        node: dict[str, Any] = {}
        cursor = node
        for part in parts[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = leaf
        out = deep_merge(out, node)
    return out


def parse_env_value(raw: str) -> Any:
    """Parse an env value as a TOML literal (``8080``, ``true``, ``["a"]``); else a string."""
    text = raw.strip()
    if not text:
        return ""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def env_overrides(env: Mapping[str, str], *, prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Collect ``AIVTUBE__SECTION__KEY=value`` variables into a nested dict.

    Names are case-insensitive (Windows upper-cases them). Single-underscore variables such as
    ``AIVTUBE_BUS_TOKEN`` are not config and are ignored.
    """
    out: dict[str, Any] = {}
    for name in sorted(env):
        if not name.upper().startswith(prefix):
            continue
        parts = [p.lower() for p in name[len(prefix) :].split("__")]
        if not parts or any(not _ENV_PART.match(p) for p in parts):
            raise ConfigError(
                name,
                f"Malformed config variable {name!r}.",
                f"ตัวแปรสภาพแวดล้อม {name!r} รูปแบบไม่ถูกต้อง",
                "Use AIVTUBE__SECTION__KEY, e.g. AIVTUBE__MIC__MODE=ptt.",
                hint_th="ใช้รูปแบบ AIVTUBE__SECTION__KEY เช่น AIVTUBE__MIC__MODE=ptt",
                source="environment",
            )
        node: dict[str, Any] = {}
        cursor = node
        for part in parts[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parse_env_value(env[name])
        out = deep_merge(out, node)
    return out


def expand_character(value: Any, character: str) -> Any:
    """Replace every ``{character}`` in strings nested inside ``value`` with ``character``."""
    if isinstance(value, str):
        return value.replace(CHARACTER_PLACEHOLDER, character)
    if isinstance(value, Mapping):
        return {k: expand_character(v, character) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [expand_character(v, character) for v in value]
    return value


def find_root(start: Path | None = None) -> Path:
    """Find the project root: the nearest folder (upwards) holding ``config/defaults.toml``."""
    candidates: list[Path] = []
    here = (start or Path.cwd()).resolve()
    candidates.extend([here, *here.parents])
    pkg = Path(__file__).resolve()
    candidates.extend(pkg.parents)
    for folder in candidates:
        if (folder / DEFAULTS_FILE).is_file():
            return folder
    raise ConfigError(
        str(DEFAULTS_FILE),
        "Cannot find the AI_Vtube folder (config/defaults.toml).",
        "หาโฟลเดอร์ AI_Vtube (config/defaults.toml) ไม่เจอ",
        "Run aivtube from inside the AI_Vtube folder.",
        hint_th="รัน aivtube จากในโฟลเดอร์ AI_Vtube",
    )


@dataclass(frozen=True, slots=True)
class Layers:
    """The raw layers (name, data) in precedence order, and their merge."""

    profile: str
    named: tuple[tuple[str, dict[str, Any]], ...]
    merged: dict[str, Any]

    def rank_of(self, path: tuple[str, ...]) -> int:
        """Index of the highest layer that sets ``path`` (or its nearest parent); -1 if none."""
        for depth in range(len(path), 0, -1):
            prefix = path[:depth]
            for index in range(len(self.named) - 1, -1, -1):
                if _has_path(self.named[index][1], prefix):
                    return index
        return -1

    def source_of(self, path: tuple[str, ...]) -> str | None:
        """The name of the highest layer that sets ``path`` (or its nearest parent table)."""
        index = self.rank_of(path)
        return self.named[index][0] if index >= 0 else None


def _has_path(data: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    node: Any = data
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def collect_layers(
    root: Path,
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    user_data: Mapping[str, Any] | None = None,
) -> Layers:
    """Read and merge every layer. ``user_data`` replaces ``config/user.toml`` (for writers
    that validate before saving). ``env=None`` reads ``os.environ``."""
    root = Path(root)
    defaults = read_toml(root / DEFAULTS_FILE)
    user = dict(user_data) if user_data is not None else read_toml(root / USER_FILE, required=False)
    _check_schema_version(user, str(USER_FILE))
    env_layer = env_overrides(os.environ if env is None else env)
    cli = expand_dotted(cli_overrides or {})

    name = profile or _pick_profile(defaults, user, env_layer, cli)
    profiles = deep_merge(_table(defaults, "profiles"), _table(user, "profiles"))
    if name not in profiles and name != DEFAULT_PROFILE:
        known = ", ".join(sorted(set(profiles) | {DEFAULT_PROFILE}))
        raise ConfigError(
            "active_profile",
            f"Unknown profile {name!r}. Known profiles: {known}.",
            f"ไม่รู้จักโปรไฟล์ {name!r} โปรไฟล์ที่มี: {known}",
            f"Pick one of: {known}.",
            hint_th=f"เลือกหนึ่งใน: {known}",
        )
    overlay = _table(profiles, name)
    named: list[tuple[str, dict[str, Any]]] = [
        (str(DEFAULTS_FILE), defaults),
        (f"profile {name}", overlay),
        (str(USER_FILE), user),
        ("environment", env_layer),
        ("command line", cli),
    ]
    merged: dict[str, Any] = {}
    for _, data in named:
        merged = deep_merge(merged, data)
    merged["active_profile"] = name
    return Layers(profile=name, named=tuple(named), merged=merged)


def _table(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(
            key,
            f"{key!r} must be a table.",
            f"{key!r} ต้องเป็นตาราง [..]",
            f"Write it as [{key}.<name>] sections.",
            hint_th=f"เขียนเป็นหัวข้อ [{key}.<ชื่อ>]",
        )
    return dict(value)


def _pick_profile(*layers: Mapping[str, Any]) -> str:
    name: Any = DEFAULT_PROFILE
    for layer in layers:
        if "active_profile" in layer:
            name = layer["active_profile"]
    if not isinstance(name, str) or not name:
        raise ConfigError(
            "active_profile",
            "active_profile must be a profile name.",
            "active_profile ต้องเป็นชื่อโปรไฟล์",
            'For example: active_profile = "light".',
            hint_th='ตัวอย่าง: active_profile = "light"',
        )
    return name


def _check_schema_version(user: Mapping[str, Any], where: str) -> None:
    version = user.get("schema_version", CURRENT_SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ConfigError(
            "schema_version",
            f"schema_version must be a positive integer, got {version!r}.",
            f"schema_version ต้องเป็นจำนวนเต็มบวก แต่ได้ {version!r}",
            f"Set schema_version = {CURRENT_SCHEMA_VERSION}.",
            hint_th=f"ตั้ง schema_version = {CURRENT_SCHEMA_VERSION}",
            source=where,
        )
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            "schema_version",
            f"{where} was written by a newer aivtube (schema {version}, this build reads "
            f"{CURRENT_SCHEMA_VERSION}).",
            f"{where} ถูกเขียนโดย aivtube เวอร์ชันที่ใหม่กว่า (schema {version} แต่เวอร์ชันนี้อ่านได้ถึง "
            f"{CURRENT_SCHEMA_VERSION})",
            "Update aivtube, or restore user.toml.bak.",
            hint_th="อัปเดต aivtube หรือกู้ไฟล์ user.toml.bak คืน",
            source=where,
        )
    if version < CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            "schema_version",
            f"{where} uses the old schema {version}.",
            f"{where} ใช้ schema เก่า ({version})",
            "Run: aivtube config migrate",
            hint_th="รันคำสั่ง: aivtube config migrate",
            source=where,
        )
