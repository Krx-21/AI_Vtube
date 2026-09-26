"""Operator command parsing shared by the panel, hotkeys and the text console (§2.11, §3.12).

Standard library only. Parsers raise ``ValueError`` with a short English reason.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from typing import Any, Final

from aivtube.contracts.control import OpCommand, OpKind

__all__ = [
    "FAST_KINDS",
    "HOTKEYS",
    "MIC_MODES",
    "TEXT_KINDS",
    "TOOLS_MODES",
    "hotkey_command",
    "hotkey_names",
    "new_command_id",
    "parse_op_command",
    "validate_args",
]

#: Commands with a synchronous fast path (I4): they never wait behind other commands.
FAST_KINDS: Final = frozenset({OpKind.FREEZE, OpKind.SKIP, OpKind.MUTE, OpKind.UNMUTE, OpKind.PTT})
#: Commands whose ``args["text"]`` must be a non-empty string.
TEXT_KINDS: Final = frozenset({OpKind.SAY, OpKind.DIRECT})
MIC_MODES: Final = ("open", "ptt", "deafened")
TOOLS_MODES: Final = ("live", "dry_run", "off")
MAX_SAY_CHARS: Final = 600
_MAX_ARGS: Final = 32
_OPERATOR_RE: Final = re.compile(r"[a-z][a-z0-9_.:-]{0,31}")

#: Hotkey name -> (kind, args). ``*_toggle`` hotkeys are resolved against the snapshot.
HOTKEYS: Final[Mapping[str, tuple[OpKind, Mapping[str, Any]]]] = {
    "freeze": (OpKind.FREEZE, {}),
    "resume": (OpKind.RESUME, {}),
    "skip": (OpKind.SKIP, {}),
    "mute": (OpKind.MUTE, {}),
    "unmute": (OpKind.UNMUTE, {}),
    "go_live": (OpKind.GO_LIVE, {}),
    "chat_on": (OpKind.CHAT_INTAKE, {"on": True}),
    "chat_off": (OpKind.CHAT_INTAKE, {"on": False}),
    "ptt_down": (OpKind.PTT, {"active": True}),
    "ptt_up": (OpKind.PTT, {"active": False}),
    "mic_open": (OpKind.MIC_MODE, {"mode": "open"}),
    "mic_ptt": (OpKind.MIC_MODE, {"mode": "ptt"}),
    "mic_deafened": (OpKind.MIC_MODE, {"mode": "deafened"}),
    "deafen": (OpKind.MIC_MODE, {"mode": "deafened"}),
}
_TOGGLES: Final = frozenset({"mute_toggle", "deafen_toggle"})


def new_command_id() -> str:
    return uuid.uuid4().hex[:12]


def _kind(value: Any) -> OpKind:
    if isinstance(value, OpKind):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("kind must be a non-empty string")
    try:
        return OpKind(value.strip().casefold())
    except ValueError:
        raise ValueError(f"unknown command kind {value!r}") from None


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in ("1", "true", "on", "yes"):
        return True
    if isinstance(value, str) and value.strip().casefold() in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{field} must be a boolean")


def validate_args(kind: OpKind, args: Mapping[str, Any]) -> dict[str, Any]:
    """Check and normalise the arguments of commands whose shape the panel knows.

    Unknown kinds keep their arguments unchanged (the control surface validates them).
    """
    out = dict(args)
    if kind in TEXT_KINDS:
        text = out.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{kind.value} needs a non-empty 'text'")
        if len(text) > MAX_SAY_CHARS:
            raise ValueError(f"'text' is longer than {MAX_SAY_CHARS} characters")
        out["text"] = text.strip()
    elif kind is OpKind.MIC_MODE:
        mode = out.get("mode")
        if mode not in MIC_MODES:
            raise ValueError(f"mode must be one of {', '.join(MIC_MODES)}")
    elif kind is OpKind.PTT:
        out["active"] = _bool(out.get("active"), field="active")
    elif kind in (OpKind.CHAT_INTAKE, OpKind.STRICT):
        out["on"] = _bool(out.get("on", True), field="on")
    elif kind is OpKind.TOOLS_MODE:
        if out.get("mode") not in TOOLS_MODES:
            raise ValueError(f"mode must be one of {', '.join(TOOLS_MODES)}")
    elif kind is OpKind.TOOL_ENABLE:
        if not isinstance(out.get("name"), str) or not out["name"]:
            raise ValueError("tool_enable needs a tool 'name'")
        out["enabled"] = _bool(out.get("enabled", True), field="enabled")
    elif kind is OpKind.LLM_USE:
        if not isinstance(out.get("name"), str) or not out["name"]:
            raise ValueError("llm_use needs a provider 'name'")
    elif kind in (OpKind.MEMORY_EDIT, OpKind.MEMORY_STATUS):
        memory_id = out.get("id")
        if isinstance(memory_id, bool) or not isinstance(memory_id, int | str):
            raise ValueError("memory commands need an integer 'id'")
        try:
            out["id"] = int(memory_id)
        except ValueError:
            raise ValueError("memory commands need an integer 'id'") from None
        if kind is OpKind.MEMORY_STATUS and out.get("status") not in (
            "active",
            "quarantined",
            "deleted",
        ):
            raise ValueError("status must be active, quarantined or deleted")
        if kind is OpKind.MEMORY_EDIT:
            _memory_edit_args(out)
    elif kind is OpKind.RESTART:
        if not isinstance(out.get("component"), str) or not out["component"]:
            raise ValueError("restart needs a 'component'")
    elif kind is OpKind.APPROVE and "request" in out:  # a pending tool call (panel queue)
        if not isinstance(out["request"], str) or not out["request"]:
            raise ValueError("approve needs a 'request' id")
        out["approved"] = _bool(out.get("approved", True), field="approved")
    return out


def _memory_edit_args(out: dict[str, Any]) -> None:
    for name in ("text", "subject"):
        if out.get(name) is not None and not isinstance(out[name], str):
            raise ValueError(f"'{name}' must be a string")
    importance = out.get("importance")
    if importance is not None:
        if isinstance(importance, bool) or not isinstance(importance, int | str):
            raise ValueError("importance must be an integer 1-5")
        try:
            out["importance"] = int(importance)
        except ValueError:
            raise ValueError("importance must be an integer 1-5") from None
        if not 1 <= out["importance"] <= 5:
            raise ValueError("importance must be an integer 1-5")
    for name in ("locked", "pinned"):
        if out.get(name) is not None:
            out[name] = _bool(out[name], field=name)


def parse_op_command(body: Any, *, operator: str) -> OpCommand:
    """Build an ``OpCommand`` from a ``POST /api/cmd`` body
    ``{kind, args?, character?, id?, operator?}``.

    ``operator`` is the default audit label; a body may name itself with a short lowercase
    label (the launcher's ``notify_freeze`` sends ``"launcher"``).
    """
    if not isinstance(body, Mapping):
        raise ValueError("body must be a JSON object")
    label = body.get("operator")
    if label is not None:
        if not isinstance(label, str) or not _OPERATOR_RE.fullmatch(label):
            raise ValueError("operator must be a short lowercase label")
        operator = label
    kind = _kind(body.get("kind"))
    args = body.get("args") or {}
    if not isinstance(args, Mapping):
        raise ValueError("args must be an object")
    if len(args) > _MAX_ARGS:
        raise ValueError("too many args")
    character = body.get("character")
    if character is not None and (not isinstance(character, str) or not character):
        raise ValueError("character must be a non-empty string")
    cmd_id = body.get("id")
    if cmd_id is not None and not isinstance(cmd_id, str):
        raise ValueError("id must be a string")
    return OpCommand(
        kind=kind,
        args=validate_args(kind, {str(k): v for k, v in args.items()}),
        character=character,
        operator=operator,
        id=(cmd_id or new_command_id())[:64],
    )


def hotkey_command(
    name: str,
    snapshot: Mapping[str, Any],
    *,
    operator: str = "hotkey",
    character: str | None = None,
) -> OpCommand:
    """The command for ``GET /hotkey/<name>``; toggles read ``muted`` / ``mic_mode`` from the
    control snapshot. Raises ``KeyError`` for an unknown hotkey name."""
    key = name.strip().casefold().replace("-", "_")
    kind: OpKind
    args: Mapping[str, Any]
    if key == "mute_toggle":
        kind, args = (OpKind.UNMUTE, {}) if snapshot.get("muted") else (OpKind.MUTE, {})
    elif key == "deafen_toggle":
        deafened = snapshot.get("mic_mode") == "deafened"
        previous = snapshot.get("mic_mode_before_deafen") or "open"
        kind, args = OpKind.MIC_MODE, {"mode": previous if deafened else "deafened"}
    elif key in HOTKEYS:
        kind, args = HOTKEYS[key]
    else:
        raise KeyError(name)
    return OpCommand(
        kind=kind,
        args=dict(args),
        character=character,
        operator=operator,
        id=new_command_id(),
    )


def hotkey_names() -> tuple[str, ...]:
    return tuple(sorted({*HOTKEYS, *_TOGGLES}))
