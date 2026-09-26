"""Memory write policy: the quarantine rule and the PII guard (ARCHITECTURE.md §6).

Quarantine rule (``initial_status``), applied by ``SqliteMemory.remember`` and the memory tools:

1. Writes whose ``source`` is not ``"model"`` (``operator``, ``import``, ``consolidation``)
   keep the status the caller asked for (default ``active``).
2. Model writes of ``kind="viewer"`` are always ``quarantined``.
3. Other model writes look at ``origin``, formatted ``"<stimulus kind>[:<detail>]"`` (for
   example ``"voice"``, ``"chat:twitch:12345"``, ``"support:youtube:abc"``). The stimulus kind
   is the kind of the stimulus that triggered the decision (``StimulusKind`` values):

   - ``voice`` and ``operator`` are trusted, so the write is ``active``;
   - ``chat``, ``support`` and ``mention`` are ``quarantined``, or ``active`` when
     ``chat_sourced="allow"`` (``[memory] chat_sourced`` in the config);
   - anything else (games, the twin, vision, idle, an empty or unknown origin) is
     ``quarantined``. Unknown means untrusted.

4. A caller may always ask for a stricter status: the result is the stricter of the caller's
   status and the rule's.

Quarantined items stay out of the prompt (prefix, recall, viewer facts and ``<new_memories>``)
until the operator approves them in the panel. ``may_forget`` applies the same trust rule to
deletions requested by the model.
"""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

from aivtube.contracts.memory import MemKind, MemSource, MemStatus
from aivtube.text.thai import thai_digits_to_arabic

__all__ = [
    "CHAT_ORIGINS",
    "TRUSTED_ORIGINS",
    "ChatSourced",
    "find_pii",
    "initial_status",
    "may_forget",
    "origin_kind",
    "stricter",
]

ChatSourced: TypeAlias = Literal["quarantine", "allow"]

TRUSTED_ORIGINS = frozenset({"voice", "operator"})
CHAT_ORIGINS = frozenset({"chat", "support", "mention"})


def origin_kind(origin: str) -> str:
    """The stimulus kind part of an origin string (``"chat:twitch:1"`` -> ``"chat"``)."""
    return origin.split(":", 1)[0].strip().lower()


def _trusted(origin: str, chat_sourced: ChatSourced) -> bool:
    kind = origin_kind(origin)
    return kind in TRUSTED_ORIGINS or (kind in CHAT_ORIGINS and chat_sourced == "allow")


def initial_status(
    kind: MemKind,
    source: MemSource,
    origin: str,
    *,
    requested: MemStatus = "active",
    chat_sourced: ChatSourced = "quarantine",
) -> MemStatus:
    """The status a new item is stored with (see the module docstring for the rule)."""
    if requested == "deleted":
        raise ValueError("a new memory cannot be stored as deleted")
    if source != "model":
        return requested
    if kind == "viewer" or not _trusted(origin, chat_sourced):
        return "quarantined"
    return requested


def stricter(a: MemStatus, b: MemStatus) -> MemStatus:
    """``quarantined`` if either status is, else ``active`` (``deleted`` is not a write status)."""
    return "quarantined" if "quarantined" in (a, b) else "active"


def may_forget(origin: str, *, chat_sourced: ChatSourced = "quarantine") -> bool:
    """Whether a model-requested ``forget`` may run for a decision with this ``origin``.

    A deletion cannot be quarantined, so untrusted decisions (chat-triggered unless
    ``chat_sourced="allow"``, and every unknown origin) are refused instead.
    """
    return _trusted(origin, chat_sourced)


# --- PII (phone numbers, Thai national ID, e-mail) -----------------------------------------------
_SEP = r"[\s\-.]?"
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# 13 digits in the 1-4-5-2-1 grouping, separators optional.
_NATIONAL_ID = re.compile(rf"(?<!\d)\d{_SEP}\d{{4}}{_SEP}\d{{5}}{_SEP}\d{{2}}{_SEP}\d(?!\d)")
# Thai numbers: 0 + 8 or 9 digits (landline/mobile) or +66 + 8 or 9 digits; any other
# international number written with a leading "+" and 8-14 digits.
_PHONE_TH = re.compile(rf"(?<![\d+])(?:\+?66{_SEP}|0)(?:\d{_SEP}){{7,8}}\d(?!\d)")
_PHONE_INTL = re.compile(rf"(?<![\d+])\+\d(?:{_SEP}\d){{7,13}}(?!\d)")


def find_pii(text: str) -> str | None:
    """The PII category found in ``text`` (``email``, ``national_id``, ``phone``) or ``None``.

    Thai digits are read as Arabic digits first. This is the memory tool's own guard; the
    safety gate's ``memory`` direction applies its tier-0 rules on top.
    """
    probe = thai_digits_to_arabic(text)
    if _EMAIL.search(probe):
        return "email"
    if _NATIONAL_ID.search(probe):
        return "national_id"
    if _PHONE_TH.search(probe) or _PHONE_INTL.search(probe):
        return "phone"
    return None
