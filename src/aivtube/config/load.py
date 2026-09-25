"""Load and validate configuration; translate pydantic errors into bilingual ``ConfigError``."""

from __future__ import annotations

import difflib
import os
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from aivtube.config.errors import ConfigError
from aivtube.config.layers import (
    USER_FILE,
    Layers,
    collect_layers,
    deep_merge,
    expand_character,
    read_toml,
)
from aivtube.config.schema import (
    VTS_PORT_RANGE,
    AppConfig,
    CharacterConfig,
    Secrets,
    is_secret_name,
    is_valid_character_id,
)

__all__ = [
    "config_error_from_validation",
    "load_character",
    "load_characters",
    "load_config",
    "load_secrets",
    "validate_stage",
]

CHARACTERS_DIR = Path("characters")


def load_config(
    root: Path,
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load ``root``'s config: defaults → profile → user.toml → env → CLI (§8).

    ``env=None`` reads ``os.environ``; pass ``{}`` for a hermetic load. Raises ConfigError.
    """
    root = Path(root)
    layers = collect_layers(root, profile=profile, cli_overrides=cli_overrides, env=env)
    return validate_layers(root, layers)


def validate_layers(root: Path, layers: Layers) -> AppConfig:
    try:
        cfg = AppConfig.model_validate(layers.merged)
    except ValidationError as exc:
        raise config_error_from_validation(exc, model=AppConfig, layers=layers) from None
    cfg._root = Path(root)
    return cfg


def load_character(
    root: Path,
    char_id: str,
    *,
    app: AppConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CharacterConfig:
    """Load ``characters/<char_id>/character.toml``.

    Overrides come from ``overrides``, else ``app.character_overrides[char_id]`` (which already
    merges user.toml, env and CLI), else ``[character_overrides.<id>]`` in user.toml.
    ``{character}`` placeholders expand to ``char_id``.
    """
    root = Path(root)
    char_dir = root / CHARACTERS_DIR / char_id
    rel = CHARACTERS_DIR / char_id / "character.toml"
    if not is_valid_character_id(char_id):
        raise ConfigError(
            "characters",
            f"Invalid character id {char_id!r}.",
            f"รหัสตัวละคร {char_id!r} ไม่ถูกต้อง",
            "Use lowercase letters, digits and _ (the folder name under characters/).",
            hint_th="ใช้ตัวพิมพ์เล็ก ตัวเลข และ _ (ตรงกับชื่อโฟลเดอร์ใน characters/)",
        )
    raw = read_toml(char_dir / "character.toml")
    if overrides is None:
        if app is not None:
            overrides = app.character_overrides.get(char_id, {})
        else:
            user = read_toml(root / USER_FILE, required=False)
            overrides = _mapping(_mapping(user.get("character_overrides")).get(char_id))
    named = ((str(rel), raw), ("character_overrides", dict(overrides)))
    data = expand_character(deep_merge(raw, overrides), char_id)
    try:
        cfg = CharacterConfig.model_validate(data)
    except ValidationError as exc:
        layers = Layers(profile="", named=named, merged=data)
        raise config_error_from_validation(exc, model=CharacterConfig, layers=layers) from None
    if cfg.id != char_id:
        raise ConfigError(
            "id",
            f"character.toml says id = {cfg.id!r} but the folder is {char_id!r}.",
            f"ใน character.toml ตั้ง id = {cfg.id!r} แต่ชื่อโฟลเดอร์คือ {char_id!r}",
            f'Set id = "{char_id}" or rename the folder.',
            hint_th=f'ตั้ง id = "{char_id}" หรือเปลี่ยนชื่อโฟลเดอร์',
            source=str(rel),
        )
    persona = char_dir / cfg.persona
    if not persona.is_file():
        raise ConfigError(
            "persona",
            f"Persona file {persona.name!r} is missing.",
            f"ไม่พบไฟล์บุคลิก {persona.name!r}",
            "Copy persona.th.md from characters/_template/ and edit it.",
            hint_th="คัดลอก persona.th.md จาก characters/_template/ แล้วแก้ไข",
            source=str(rel),
        )
    cfg._root = root
    cfg._dir = char_dir
    return cfg


def load_characters(app: AppConfig) -> dict[str, CharacterConfig]:
    """Load every character in ``app.characters`` and check them together."""
    chars = {cid: load_character(app.root, cid, app=app) for cid in app.characters}
    validate_stage(app, chars)
    return chars


def validate_stage(app: AppConfig, chars: Mapping[str, CharacterConfig]) -> None:
    """Cross-checks between the app config and its characters. Raises ConfigError."""
    shared_ports = {port: path for path, port in app.port_items()}
    shared_ports.pop(app.ports.neuro_sdk, None)  # the default SDK hub port a character may take
    game_ports: dict[int, str] = {}
    for cid, char in chars.items():
        where = str(CHARACTERS_DIR / cid / "character.toml")
        for ident in app.tts_chain_for(char):
            if ident not in app.tts.identities:
                raise ConfigError(
                    "tts_identity_chain",
                    f"TTS identity {ident!r} is not defined in [tts.identities].",
                    f"ไม่มีเสียง {ident!r} ใน [tts.identities]",
                    f"Known identities: {', '.join(sorted(app.tts.identities))}.",
                    hint_th="เลือกเสียงที่มีอยู่ใน [tts.identities] หรือเพิ่มหัวข้อใหม่",
                    source=where,
                )
        port = char.games.port
        clash = game_ports.get(port) or shared_ports.get(port)
        if clash is not None or port in VTS_PORT_RANGE:
            raise ConfigError(
                "games.port",
                f"Game SDK port {port} of {cid!r} clashes with {clash or 'VTube Studio'}.",
                f"พอร์ต SDK เกม {port} ของ {cid!r} ซ้ำกับ {clash or 'VTube Studio'}",
                "Pailin uses 8000 and the twin 8010.",
                hint_th="ไพลินใช้ 8000 ส่วนตัวละครที่สองใช้ 8010",
                source=where,
            )
        game_ports[port] = f"{cid} games.port"
        if app.safety.filtered_text not in char.cached_phrases:
            raise ConfigError(
                "cached_phrases",
                f"cached_phrases must include {app.safety.filtered_text!r} (played when output is "
                "blocked).",
                f"cached_phrases ต้องมี {app.safety.filtered_text!r} (ใช้เล่นเมื่อคำพูดถูกกรอง)",
                f'Add "{app.safety.filtered_text}" to cached_phrases.',
                hint_th=f'เพิ่ม "{app.safety.filtered_text}" ใน cached_phrases',
                source=where,
            )


def load_secrets(root: Path | None = None, *, env: Mapping[str, str] | None = None) -> Secrets:
    """Read secrets from ``<root>/.env`` and the environment (the environment wins).

    ``env=None`` uses ``os.environ``; pass a mapping for a hermetic load.
    """
    fields = Secrets.model_fields
    values: dict[str, str] = {}
    if root is not None:
        path = Path(root) / ".env"
        if path.is_file():
            values.update({k.upper(): v for k, v in dotenv_values(path).items() if v})
    source = os.environ if env is None else env
    values.update(
        {
            k.upper(): v
            for k, v in source.items()
            if v and (k.upper() in fields or is_secret_name(k))
        }
    )
    kwargs: dict[str, Any] = {name: values.get(name) for name in fields}
    secrets = Secrets.model_validate(kwargs)  # explicit values only; no implicit sources
    secrets._extra = {k: v for k, v in values.items() if k not in fields}
    return secrets


# --- ValidationError → ConfigError ------------------------------------------------------

_TH_BY_TYPE: dict[str, str] = {
    "missing": "ต้องกำหนดค่านี้",
    "extra_forbidden": "ไม่รู้จักคีย์นี้ (อาจพิมพ์ผิด)",
    "literal_error": "ค่านี้ใช้ไม่ได้ ต้องเป็นหนึ่งใน: {expected}",
    "enum": "ค่านี้ใช้ไม่ได้ ต้องเป็นหนึ่งใน: {expected}",
    "string_type": "ต้องเป็นข้อความ (ใส่ในเครื่องหมายคำพูด)",
    "int_type": "ต้องเป็นจำนวนเต็ม",
    "int_parsing": "ต้องเป็นจำนวนเต็ม",
    "int_from_float": "ต้องเป็นจำนวนเต็ม ไม่มีทศนิยม",
    "float_type": "ต้องเป็นตัวเลข",
    "float_parsing": "ต้องเป็นตัวเลข",
    "bool_type": "ต้องเป็น true หรือ false",
    "bool_parsing": "ต้องเป็น true หรือ false",
    "list_type": 'ต้องเป็นรายการ เช่น ["a", "b"]',
    "tuple_type": "ต้องเป็นรายการ เช่น [1, 2]",
    "dict_type": "ต้องเป็นตาราง [..]",
    "model_type": "ต้องเป็นตาราง [..]",
    "model_attributes_type": "ต้องเป็นตาราง [..]",
    "greater_than": "ค่าต้องมากกว่า {gt}",
    "greater_than_equal": "ค่าต้องไม่น้อยกว่า {ge}",
    "less_than": "ค่าต้องน้อยกว่า {lt}",
    "less_than_equal": "ค่าต้องไม่เกิน {le}",
    "too_short": "มีจำนวนรายการน้อยเกินไป (อย่างน้อย {min_length})",
    "too_long": "มีจำนวนรายการมากเกินไป (ไม่เกิน {max_length})",
    "string_too_short": "ข้อความสั้นเกินไป (อย่างน้อย {min_length} ตัวอักษร)",
    "string_too_long": "ข้อความยาวเกินไป (ไม่เกิน {max_length} ตัวอักษร)",
    "string_pattern_mismatch": "รูปแบบข้อความไม่ถูกต้อง",
}


_GENERIC_HINT = ("Compare with config/defaults.toml.", "เทียบกับค่าใน config/defaults.toml")


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "?"


def config_error_from_validation(
    exc: ValidationError,
    *,
    model: type[BaseModel],
    layers: Layers | None = None,
) -> ConfigError:
    """Turn every pydantic error into a bilingual ConfigError (the first one leads)."""
    issues = [_issue(err, model, layers) for err in exc.errors(include_url=False)]
    first = issues[0]
    return ConfigError(
        first.path,
        first.message_en,
        first.message_th,
        first.hint,
        hint_th=first.hint_th,
        source=first.source,
        issues=issues,
    )


def _issue(err: Mapping[str, Any], model: type[BaseModel], layers: Layers | None) -> ConfigError:
    ctx: dict[str, Any] = dict(err.get("ctx") or {})
    loc = tuple(str(p) for p in err.get("loc", ()) if p != "[key]")
    if "path" in ctx:
        loc = tuple(str(ctx["path"]).split("."))
    elif "field" in ctx:
        loc = (*loc, str(ctx["field"]))
    if "also" in ctx and layers is not None:
        other = tuple(str(ctx["also"]).split("."))
        if layers.rank_of(other) > layers.rank_of(loc):
            loc = other
    path = ".".join(loc) or "config"
    source = layers.source_of(loc) if layers is not None else None
    kind = str(err.get("type", ""))
    hint, hint_th = _GENERIC_HINT
    if "th" in ctx:  # our own errors carry their Thai text and hints
        return ConfigError(
            path,
            str(ctx["en"]),
            str(ctx["th"]),
            str(ctx.get("hint") or hint),
            hint_th=str(ctx.get("hint_th") or hint_th),
            source=source,
        )
    shown = _short(err.get("input"))
    en = str(err.get("msg", "invalid value"))
    if kind not in ("missing", "extra_forbidden"):
        en += f" (got {shown})"
    th = _TH_BY_TYPE.get(kind, "ค่าไม่ถูกต้อง").format_map(_SafeDict(ctx))
    if kind not in ("missing", "extra_forbidden"):
        th += f" (ได้ {shown})"
    if kind == "extra_forbidden":
        known = _known_keys(model, loc[:-1])
        close = difflib.get_close_matches(loc[-1] if loc else "", known, n=1)
        if close:
            hint, hint_th = f"Did you mean {close[0]!r}?", f"หมายถึง {close[0]!r} หรือเปล่า"
        else:
            hint = "Remove it; see config/defaults.toml for valid keys."
            hint_th = "ลบคีย์นี้ออก (ดูคีย์ที่ถูกต้องใน config/defaults.toml)"
    elif kind == "missing":
        hint, hint_th = "Add this key.", "เพิ่มคีย์นี้"
    elif kind in ("literal_error", "enum"):
        hint = f"Use one of: {ctx.get('expected', '?')}."
        hint_th = f"ใช้หนึ่งใน: {ctx.get('expected', '?')}"
    return ConfigError(path, en, th, hint, hint_th=hint_th, source=source)


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _known_keys(model: type[BaseModel], loc: tuple[str, ...]) -> list[str]:
    """Valid keys of the table at ``loc`` (for typo suggestions)."""
    current: Any = model
    for part in loc:
        current = _step(current, part)
        if current is None:
            return []
    current = _unwrap(current)
    if isinstance(current, type) and issubclass(current, BaseModel):
        return list(current.model_fields)
    return []


def _unwrap(tp: Any) -> Any:
    while True:
        origin = typing.get_origin(tp)
        if origin is typing.Annotated:
            tp = typing.get_args(tp)[0]
        elif origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            if len(args) != 1:
                return tp
            tp = args[0]
        else:
            return tp


def _step(tp: Any, part: str) -> Any:
    tp = _unwrap(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        info = tp.model_fields.get(part)
        return info.annotation if info is not None else None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (dict, Mapping) and len(args) == 2:
        return args[1]
    if origin in (list, tuple) and args:
        return args[0]
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
