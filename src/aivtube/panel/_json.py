"""JSON helpers for the panel: dataclasses, enums and mappings to strict (NaN-free) JSON."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from aivtube.contracts.types import ChatMessage

__all__ = ["dumps", "message_json", "to_jsonable"]


def to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Convert ``value`` to plain JSON types. Non-finite floats become ``None``."""
    if _depth > 32:
        return None
    if value is None or isinstance(value, bool | str | int):
        return value.value if isinstance(value, Enum) else value
    if isinstance(value, Enum):
        return to_jsonable(value.value, _depth=_depth + 1)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, ChatMessage):
        return message_json(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name), _depth=_depth + 1)
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k.value if isinstance(k, Enum) else k): to_jsonable(v, _depth=_depth + 1)
                for k, v in value.items()}  # fmt: skip
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v, _depth=_depth + 1) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist(), _depth=_depth + 1)
    if isinstance(value, int | float):  # numbers.Real subclasses (numpy scalars)
        return to_jsonable(float(value), _depth=_depth + 1)
    return str(value)


def message_json(m: ChatMessage) -> dict[str, Any]:
    """A chat message for the panel: every field except ``raw`` (platform payloads)."""
    return {
        f.name: to_jsonable(getattr(m, f.name)) for f in dataclasses.fields(m) if f.name != "raw"
    }


def dumps(value: Any) -> str:
    """Compact UTF-8 JSON; values that are not plain JSON go through :func:`to_jsonable`."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(
            to_jsonable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
