"""Build ``ChatMessage`` objects for generic alert ingest (``POST /api/event``) and the console.

Standard library only (the text console imports this without aiohttp). Every builder validates
its input and raises ``ValueError`` with a short English reason the HTTP layer returns as 400.
Input text is **not** filtered here: messages go through the normal intake path (tier-0 input
check, then SUPPORT or the chat window), exactly like platform chat.
"""

from __future__ import annotations

import itertools
import math
import uuid
from collections.abc import Mapping
from typing import Any, Final

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import ChatMessage, ChatUser, MsgKind, Platform

__all__ = [
    "MAX_NAME_CHARS",
    "MAX_TEXT_CHARS",
    "alert_message",
    "chat_message",
]

MAX_TEXT_CHARS: Final = 500
MAX_NAME_CHARS: Final = 64
_MAX_CURRENCY_CHARS: Final = 8
_MAX_AMOUNT: Final = 1e9
_ids = itertools.count(1)

#: ``/api/event`` kind aliases (alert tools name things differently).
_KIND_ALIASES: Final[Mapping[str, MsgKind]] = {
    "donation": MsgKind.DONATION,
    "tip": MsgKind.DONATION,
    "superchat": MsgKind.DONATION,
    "super_chat": MsgKind.DONATION,
    "bits": MsgKind.DONATION,
    "cheer": MsgKind.DONATION,
    "sub": MsgKind.SUB,
    "subscription": MsgKind.SUB,
    "resub": MsgKind.SUB,
    "member": MsgKind.SUB,
    "membership": MsgKind.SUB,
    "gift_sub": MsgKind.GIFT_SUB,
    "giftsub": MsgKind.GIFT_SUB,
    "gift": MsgKind.GIFT_SUB,
    "raid": MsgKind.RAID,
    "host": MsgKind.RAID,
    "redeem": MsgKind.REDEEM,
    "redemption": MsgKind.REDEEM,
    "text": MsgKind.TEXT,
    "chat": MsgKind.TEXT,
    "system": MsgKind.SYSTEM,
}


def _clean(text: Any, *, field: str, limit: int, required: bool = False) -> str:
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValueError(f"{field} must be a string")
    cleaned = " ".join(text.replace("\x00", "").split())
    if required and not cleaned:
        raise ValueError(f"{field} is required")
    return cleaned[:limit]


def _number(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if not math.isfinite(number) or number < 0 or number > _MAX_AMOUNT:
        raise ValueError(f"{field} must be between 0 and {_MAX_AMOUNT:g}")
    return number


def _user_id(name: str, platform: Platform) -> str:
    return f"{platform.value}:{name.casefold()}"


def chat_message(
    name: str,
    text: str,
    *,
    clock: Clock,
    platform: Platform = Platform.CONSOLE,
    kind: MsgKind = MsgKind.TEXT,
    amount: float = 0.0,
    currency: str = "",
    value_usd: float = 0.0,
    months: int = 0,
    user_id: str | None = None,
    msg_id: str | None = None,
    raw: Mapping[str, Any] | None = None,
) -> ChatMessage:
    """A chat line, donation or sub from the console or the panel's fake-chat box."""
    name = _clean(name, field="name", limit=MAX_NAME_CHARS, required=True)
    text = _clean(text, field="text", limit=MAX_TEXT_CHARS, required=kind is MsgKind.TEXT)
    if months < 0:
        raise ValueError("months must be >= 0")
    now = clock.now()
    is_sub = kind in (MsgKind.SUB, MsgKind.GIFT_SUB) or months > 0
    user = ChatUser(
        platform=platform,
        id=user_id or _user_id(name, platform),
        name=name,
        is_sub=is_sub,
        sub_months=months,
    )
    return ChatMessage(
        platform=platform,
        id=msg_id or f"{platform.value}-{next(_ids)}-{uuid.uuid4().hex[:8]}",
        user=user,
        text=text,
        ts=now,
        received=now,
        kind=kind,
        amount=amount,
        currency=currency,
        value_usd=value_usd,
        raw=dict(raw or {}),
    )


def _kind(value: Any, amount: float) -> MsgKind:
    if value is None or value == "":
        return MsgKind.DONATION if amount > 0 else MsgKind.TEXT
    if not isinstance(value, str):
        raise ValueError("kind must be a string")
    kind = _KIND_ALIASES.get(value.strip().casefold().replace("-", "_"))
    if kind is None:
        raise ValueError(f"unknown kind {value!r}")
    return kind


def _platform(value: Any) -> Platform:
    if value is None or value == "":
        return Platform.ALERT
    if not isinstance(value, str):
        raise ValueError("platform must be a string")
    try:
        return Platform(value.strip().casefold())
    except ValueError:
        raise ValueError(f"unknown platform {value!r}") from None


def _value_usd(payload: Mapping[str, Any], amount: float, currency: str) -> float:
    if payload.get("value_usd") not in (None, ""):
        return _number(payload.get("value_usd"), field="value_usd")
    unit = currency.casefold()
    if unit == "usd":
        return amount
    if unit == "bits":
        return round(amount / 100.0, 2)
    return 0.0  # no FX table: the alert tool may send value_usd itself


def alert_message(payload: Mapping[str, Any], *, clock: Clock) -> ChatMessage:
    """Validate an ``/api/event`` body and build the matching ``ChatMessage``.

    Body: ``{"user": str, "kind"?: donation|sub|gift_sub|raid|redeem|text|…, "amount"?: n,
    "currency"?: str, "value_usd"?: n, "text"?: str, "months"?: int, "platform"?: str,
    "user_id"?: str, "id"?: str}``. ``kind`` defaults to ``donation`` when ``amount > 0``.
    The message id is ``alert:<id>`` when the tool sends one (used for de-duplication).
    """
    if not isinstance(payload, Mapping):
        raise ValueError("body must be a JSON object")
    name = payload.get("user", payload.get("name"))
    amount = _number(payload.get("amount"), field="amount")
    currency = _clean(payload.get("currency"), field="currency", limit=_MAX_CURRENCY_CHARS)
    kind = _kind(payload.get("kind", payload.get("type")), amount)
    platform = _platform(payload.get("platform"))
    months_raw = payload.get("months", 0)
    if isinstance(months_raw, bool) or not isinstance(months_raw, int | float | str):
        raise ValueError("months must be an integer")
    try:
        months = int(months_raw or 0)
    except ValueError:
        raise ValueError("months must be an integer") from None
    if not 0 <= months <= 1200:
        raise ValueError("months must be between 0 and 1200")
    ext_id = payload.get("id")
    if ext_id is not None and not isinstance(ext_id, str | int):
        raise ValueError("id must be a string")
    user_id = payload.get("user_id")
    if user_id is not None and not isinstance(user_id, str):
        raise ValueError("user_id must be a string")
    raw = {k: payload[k] for k in ("id", "kind", "type", "count", "source") if k in payload}
    return chat_message(
        name if isinstance(name, str) else "",
        payload.get("text", payload.get("message", "")),
        clock=clock,
        platform=platform,
        kind=kind,
        amount=amount,
        currency=currency,
        value_usd=_value_usd(payload, amount, currency),
        months=months,
        user_id=(user_id[:MAX_NAME_CHARS] if user_id else None),
        msg_id=None if ext_id is None else f"alert:{str(ext_id)[:128]}",
        raw=raw,
    )
