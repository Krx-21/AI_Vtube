"""Filter list files: parsing and validation (ARCHITECTURE.md §7, "Lists and overlays").

Every list is a TOML file. Two shapes can be mixed in one file:

Compact category file (the committed base lists)::

    category = "slur"            # one of CATEGORIES
    token = ["..."]              # matched on newmm token boundaries (Thai-safe)
    substring = ["..."]          # unambiguous: anywhere, also in despaced and leet forms
    regex = ['...']              # Python regex over the normalised (casefolded) form
    allow = ["..."]              # words that must never be matched as a hit

Tables (overlays: ``config/filters/private/*.toml``, platform files, ``filters.toml``)::

    [[allow]]
    text = "หีบ"
    [[deny]]
    category = "slur"            # optional when the file sets ``category``
    match = "token"              # token (default) | substring | regex
    text = "..."
    [[replace]]                  # a hard-coded speech patch
    match = "regex"              # regex (default) | substring (case-insensitive literal)
    text = "(?i)as an ai language model"
    with = ""
    directions = ["out"]         # default ["out"]

``pii`` and ``injection`` rules run on the display text (case kept), so they only accept
``substring`` or ``regex``. ``allow`` never loosens a fail-closed category (monarchy_112).
Parsing is pure; compiling happens in :mod:`aivtube.safety.keyword`.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

__all__ = [
    "CATEGORIES",
    "MATCH_CATEGORIES",
    "SPAN_CATEGORIES",
    "DenyRule",
    "FilterListError",
    "MatchMode",
    "ReplaceRule",
    "RuleSet",
    "list_files",
    "load_rule_dir",
    "load_rule_file",
    "parse_rules",
]

MatchMode: TypeAlias = Literal["token", "substring", "regex"]

#: Categories matched on the normalised forms, in reporting priority order.
MATCH_CATEGORIES: Final = (
    "monarchy_112",
    "self_harm",
    "violent_extreme",
    "doxx",
    "sexual",
    "slur",
    "gambling_scam",
    "politics",
)
#: Categories matched on the display text: masked (pii) or stripped (injection).
SPAN_CATEGORIES: Final = ("pii", "injection")
CATEGORIES: Final = MATCH_CATEGORIES + SPAN_CATEGORIES

_MODES: Final = ("token", "substring", "regex")
_TOP_KEYS: Final = frozenset(
    {"category", "token", "substring", "regex", "allow", "deny", "replace", "version", "note"}
)
_DIRECTIONS: Final = frozenset({"in", "out", "tool", "memory", "name", "game"})


class FilterListError(ValueError):
    """A filter list file is missing or invalid (message in English and Thai)."""

    def __init__(self, path: Path | str, message: str, message_th: str = "") -> None:
        self.path = str(path)
        self.message = message
        self.message_th = message_th or message
        super().__init__(f"{self.path}: {message} / {self.message_th}")


@dataclass(frozen=True, slots=True)
class DenyRule:
    text: str
    category: str
    match: MatchMode
    source: str  # "<file name>" for diagnostics

    @property
    def rule_id(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class ReplaceRule:
    pattern: str
    replacement: str
    match: Literal["regex", "substring"]
    directions: frozenset[str]
    source: str


@dataclass(frozen=True, slots=True)
class RuleSet:
    deny: tuple[DenyRule, ...] = ()
    allow: tuple[str, ...] = ()
    replace: tuple[ReplaceRule, ...] = ()
    files: tuple[Path, ...] = field(default=())

    def __add__(self, other: RuleSet) -> RuleSet:
        return RuleSet(
            self.deny + other.deny,
            self.allow + other.allow,
            self.replace + other.replace,
            self.files + other.files,
        )

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(CATEGORIES, 0)
        for rule in self.deny:
            out[rule.category] += 1
        return out


def _strings(path: Path, key: str, value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise FilterListError(
            path, f"{key!r} must be a list of strings", f"{key!r} ต้องเป็นรายการข้อความ"
        )
    return [v for v in value if v.strip()]


def _category(path: Path, value: Any, where: str) -> str:
    if value not in CATEGORIES:
        raise FilterListError(
            path,
            f"{where}: unknown category {value!r}; use one of {', '.join(CATEGORIES)}",
            f"{where}: ไม่รู้จักหมวด {value!r}",
        )
    return cast(str, value)


def _check_regex(path: Path, pattern: str, where: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise FilterListError(
            path, f"{where}: bad regex {pattern!r}: {exc}", f"{where}: regex ผิด {pattern!r}"
        ) from exc


def _deny(path: Path, text: str, category: str, match: str, where: str) -> DenyRule:
    if match not in _MODES:
        raise FilterListError(
            path, f"{where}: match must be token, substring or regex", f"{where}: match ไม่ถูกต้อง"
        )
    if category in SPAN_CATEGORIES and match == "token":
        raise FilterListError(
            path,
            f"{where}: {category} rules must use substring or regex",
            f"{where}: หมวด {category} ใช้ได้แค่ substring หรือ regex",
        )
    if match == "regex":
        _check_regex(path, text, where)
    return DenyRule(text, category, cast(MatchMode, match), path.name)


def _tables(path: Path, data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = data.get(key, [])
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw) and key == "allow":
        return [{"text": t} for t in raw]
    if not isinstance(raw, list) or not all(isinstance(t, Mapping) for t in raw):
        raise FilterListError(path, f"[[{key}]] must be an array of tables", f"[[{key}]] ผิดรูปแบบ")
    return cast(list[Mapping[str, Any]], raw)


def parse_rules(path: Path, data: Mapping[str, Any]) -> RuleSet:
    """Validate one parsed TOML document (``path`` is only used in messages)."""
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise FilterListError(path, f"unknown keys {unknown}", f"คีย์ที่ไม่รู้จัก {unknown}")
    file_cat = data.get("category")
    if file_cat is not None:
        file_cat = _category(path, file_cat, "category")
    deny: list[DenyRule] = []
    for mode in _MODES:
        if mode not in data:
            continue
        if file_cat is None:
            raise FilterListError(
                path,
                f"{mode!r} needs a file-level category",
                f"{mode!r} ต้องกำหนด category ของไฟล์",
            )
        for i, text in enumerate(_strings(path, mode, data[mode])):
            deny.append(_deny(path, text, file_cat, mode, f"{mode}[{i}]"))
    for i, table in enumerate(_tables(path, data, "deny")):
        where = f"deny[{i}]"
        deny_text = table.get("text")
        if not isinstance(deny_text, str) or not deny_text.strip():
            raise FilterListError(path, f"{where}: text is required", f"{where}: ต้องมี text")
        cat = _category(path, table.get("category", file_cat), where)
        deny.append(_deny(path, deny_text, cat, str(table.get("match", "token")), where))
    allow: list[str] = []
    for i, table in enumerate(_tables(path, data, "allow")):
        allow_text = table.get("text")
        if not isinstance(allow_text, str):
            raise FilterListError(path, f"allow[{i}]: text is required", f"allow[{i}]: ต้องมี text")
        if allow_text.strip():
            allow.append(allow_text)
    replace: list[ReplaceRule] = []
    for i, table in enumerate(_tables(path, data, "replace")):
        where = f"replace[{i}]"
        pattern = table.get("text")
        repl = table.get("with", "")
        match = table.get("match", "regex")
        dirs = table.get("directions", ["out"])
        if not isinstance(pattern, str) or not pattern:
            raise FilterListError(path, f"{where}: text is required", f"{where}: ต้องมี text")
        if not isinstance(repl, str):
            raise FilterListError(
                path, f"{where}: with must be a string", f"{where}: with ต้องเป็นข้อความ"
            )
        if match not in ("regex", "substring"):
            raise FilterListError(
                path, f"{where}: match must be regex or substring", f"{where}: match ไม่ถูกต้อง"
            )
        directions = frozenset(_strings(path, f"{where}.directions", dirs))
        if not directions or not directions <= _DIRECTIONS:
            raise FilterListError(
                path,
                f"{where}: directions must be a subset of {sorted(_DIRECTIONS)}",
                f"{where}: directions ไม่ถูกต้อง",
            )
        if match == "regex":
            _check_regex(path, pattern, where)
        replace.append(ReplaceRule(pattern, repl, match, directions, path.name))
    return RuleSet(tuple(deny), tuple(allow), tuple(replace), (path,))


def load_rule_file(path: Path) -> RuleSet:
    """Read and validate one list file (blocking)."""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise FilterListError(path, "file not found", "ไม่พบไฟล์") from exc
    except tomllib.TOMLDecodeError as exc:
        raise FilterListError(path, f"invalid TOML: {exc}", f"TOML ผิดรูปแบบ: {exc}") from exc
    except OSError as exc:
        raise FilterListError(path, f"cannot read: {exc}", f"อ่านไฟล์ไม่ได้: {exc}") from exc
    return parse_rules(path, data)


def list_files(directory: Path) -> list[Path]:
    """The ``*.toml`` files of a list directory, sorted by name (stable precedence)."""
    return sorted(p for p in directory.glob("*.toml") if p.is_file())


def load_rule_dir(directory: Path, *, required: bool) -> RuleSet:
    """Every ``*.toml`` file in ``directory``. A missing optional directory is empty."""
    if not directory.is_dir():
        if required:
            raise FilterListError(
                directory, "filter list directory not found", "ไม่พบโฟลเดอร์รายการคำกรอง"
            )
        return RuleSet()
    total = RuleSet()
    for path in list_files(directory):
        total = total + load_rule_file(path)
    return total
