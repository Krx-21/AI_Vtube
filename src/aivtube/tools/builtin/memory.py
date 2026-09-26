"""The M1 memory tools: ``remember`` and ``forget`` (ARCHITECTURE.md §4.9, §6).

``remember(text, about?, importance?, replace_slot?)``:

- ``text`` (at most 120 characters) goes into a core slot. With ``about`` naming an author
  of a message in the current chat block, it becomes a viewer fact keyed by
  ``(platform, user_id)`` instead (always quarantined; refused for viewers who opted out).
  An ``about`` that matches no chat author is kept as the core item's ``subject``.
- The origin is the decision's stimulus (``"<kind>[:<source>]"``) and the status follows
  ``aivtube.memory.policy.initial_status`` (chat/support-triggered writes are quarantined).
- Phone numbers, national IDs and e-mail addresses are refused (``find_pii``); the registry
  has already run the safety gate with direction ``"memory"`` over the free text.
- Rate limits count stored writes only: at most one per ``min_interval_s`` (120 s) and
  ``max_writes_per_session`` (8) per session, per character. A refused write (slots full,
  PII, ...) costs nothing, so the model can fix the call right away.
- When every slot is taken the result is ``{"ok": false, "error": "slots_full", "slots":
  [...]}`` with each slot's number and the start of its text, so the model can ``forget`` a
  slot or retry with ``replace_slot``.

``forget(slot)`` deletes the active core item in ``slot``. A deletion cannot be quarantined,
so it is refused for untrusted decisions (``policy.may_forget``); locked slots refuse too.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from aivtube.contracts.llm import ToolSpec
from aivtube.contracts.memory import MemoryItem, SlotsFull
from aivtube.contracts.tools import RateLimit, ToolContext, ToolPolicy, ToolResult
from aivtube.contracts.types import ChatMessage, ChatUser, Stimulus
from aivtube.memory.policy import ChatSourced, find_pii, initial_status, may_forget
from aivtube.memory.store import ViewerOptedOut

__all__ = [
    "FORGET_SPEC",
    "REMEMBER_SPEC",
    "ForgetTool",
    "RememberTool",
    "chat_authors",
    "forget_spec",
    "memory_tools",
    "remember_spec",
    "stimulus_origin",
]

_SLOT_PREVIEW = 40


def remember_spec(core_slots: int = 16, max_chars: int = 120) -> ToolSpec:
    return ToolSpec(
        name="remember",
        description=(
            "Save one short fact worth keeping across streams in long-term memory. "
            "Set about to a viewer's name from the chat block when the fact is about them. "
            "If memory is full, forget a slot or pass replace_slot."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_chars,
                    "description": "The fact, one short sentence.",
                },
                "about": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 60,
                    "description": "Name of the viewer in the chat block this fact is about.",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "1 trivial to 5 essential (default 3).",
                },
                "replace_slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": core_slots,
                    "description": "Overwrite this memory slot.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    )


def forget_spec(core_slots: int = 16) -> ToolSpec:
    return ToolSpec(
        name="forget",
        description="Delete one long-term memory slot.",
        parameters={
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": core_slots,
                    "description": "The slot number to forget.",
                }
            },
            "required": ["slot"],
            "additionalProperties": False,
        },
    )


REMEMBER_SPEC = remember_spec()
FORGET_SPEC = forget_spec()


def _json(**body: Any) -> str:
    return json.dumps(body, ensure_ascii=False)


def _refused(error: str, **extra: Any) -> ToolResult:
    return ToolResult(False, _json(ok=False, error=error, **extra), note=error)


def stimulus_origin(stimulus: Stimulus) -> str:
    """The memory ``origin`` of a decision: ``"<stimulus kind>[:<source>]"``."""
    kind = str(stimulus.kind.value)
    return f"{kind}:{stimulus.source}" if stimulus.source else kind


def _messages(value: Any, depth: int = 0) -> Iterator[ChatMessage]:
    if depth > 3:
        return
    if isinstance(value, ChatMessage):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _messages(v, depth + 1)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _messages(v, depth + 1)
    else:  # ChatSelection or any other container of messages
        for attr in ("must_ack", "candidates", "ambient"):
            yield from _messages(getattr(value, attr, ()), depth + 1)


def chat_authors(stimulus: Stimulus) -> list[ChatUser]:
    """Authors of the chat messages carried by ``stimulus.payload`` (the chat block).

    Any payload value may hold a ``ChatMessage``, a ``ChatSelection`` or lists/mappings of
    them; the brain decides the keys.
    """
    seen: dict[tuple[str, str], ChatUser] = {}
    for msg in _messages(stimulus.payload):
        seen.setdefault((msg.user.platform.value, msg.user.id), msg.user)
    return list(seen.values())


def _norm_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().lstrip("@").casefold()


class RememberTool:
    """``remember``: store a long-term memory (see the module docstring)."""

    arg_direction: Literal["tool", "memory", "game"] = "memory"

    def __init__(
        self,
        *,
        max_writes_per_session: int = 8,
        min_interval_s: float = 120.0,
        chat_sourced: ChatSourced = "quarantine",
        core_slots: int = 16,
        slot_max_chars: int = 120,
    ) -> None:
        self.spec = remember_spec(core_slots, slot_max_chars)
        self.policy = ToolPolicy(risk="safe", side_effect=True, timeout_s=5.0)
        self.max_writes_per_session = max_writes_per_session
        self.min_interval_s = min_interval_s
        self.chat_sourced: ChatSourced = chat_sourced
        self._writes: dict[tuple[str, object], int] = {}
        self._last_write: dict[str, float] = {}

    def reset(self) -> None:
        """Forget the rate-limit state (e.g. at a new session without a session id)."""
        self._writes.clear()
        self._last_write.clear()

    async def __call__(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        text = str(args.get("text", "")).strip()
        about = str(args["about"]).strip() if args.get("about") else None
        importance = int(args.get("importance", 3))
        replace_slot = args.get("replace_slot")
        if not text:
            return _refused("empty", detail="nothing to remember")
        pii = find_pii(text) or (find_pii(about) if about else None)
        if pii is not None:
            return _refused("pii", detail=f"{pii} details are never remembered")
        session = getattr(ctx.memory, "session_id", None)
        key = (ctx.character, session)
        now = ctx.clock.now()
        if self._writes.get(key, 0) >= self.max_writes_per_session:
            return _refused("rate_limited", detail="memory write limit for this stream reached")
        last = self._last_write.get(ctx.character)
        if last is not None and now - last < self.min_interval_s:
            wait = round(self.min_interval_s - (now - last))
            return _refused("rate_limited", detail="remembering too often", retry_in_s=wait)

        origin = stimulus_origin(ctx.stimulus)
        viewer = self._viewer(ctx.stimulus, about) if about else None
        if viewer is not None:
            item = MemoryItem(
                id=None,
                kind="viewer",
                text=text,
                subject=viewer.name,
                platform=viewer.platform.value,
                user_id=viewer.id,
                importance=importance,
                source="model",
                origin=origin,
                status="quarantined",
            )
            replace_slot = None
        else:
            item = MemoryItem(
                id=None,
                kind="core",
                text=text,
                subject=about,
                importance=importance,
                source="model",
                origin=origin,
                status=initial_status("core", "model", origin, chat_sourced=self.chat_sourced),
            )
        try:
            stored = await ctx.memory.remember(
                item, replace_slot=int(replace_slot) if replace_slot is not None else None
            )
        except SlotsFull as full:
            slots = [
                {
                    "slot": m.slot,
                    "text": m.text
                    if len(m.text) <= _SLOT_PREVIEW
                    else m.text[:_SLOT_PREVIEW] + "…",
                    "locked": m.locked,
                }
                for m in sorted(full.slots, key=lambda m: m.slot or 0)
            ]
            return _refused(
                "slots_full",
                slots=slots,
                hint="call forget(slot) first, or remember again with replace_slot",
            )
        except ViewerOptedOut:
            return _refused("viewer_opted_out")
        except PermissionError:  # MemoryLocked (or a store's plain PermissionError)
            return _refused("slot_locked")
        except ValueError as exc:
            return _refused("invalid", detail=str(exc)[:200])
        self._writes[key] = self._writes.get(key, 0) + 1
        self._last_write[ctx.character] = now
        body: dict[str, Any] = {"ok": True, "saved": stored.kind}
        if stored.kind == "viewer":
            body["about"] = stored.subject
        elif stored.slot is not None:
            body["slot"] = stored.slot
        if stored.status == "quarantined":
            body["status"] = "pending_review"
            body["note"] = "saved; the streamer will review it before it is used"
        else:
            body["status"] = "active"
        return ToolResult(True, json.dumps(body, ensure_ascii=False))

    @staticmethod
    def _viewer(stimulus: Stimulus, about: str) -> ChatUser | None:
        target = _norm_name(about)
        for user in chat_authors(stimulus):
            if target in (_norm_name(user.name), _norm_name(user.id)):
                return user
        return None


class ForgetTool:
    """``forget``: delete one core slot (see the module docstring)."""

    arg_direction: Literal["tool", "memory", "game"] = "memory"

    def __init__(
        self,
        *,
        chat_sourced: ChatSourced = "quarantine",
        core_slots: int = 16,
        rate_limit: RateLimit | None = None,
    ) -> None:
        self.spec = forget_spec(core_slots)
        self.policy = ToolPolicy(
            risk="moderate",
            side_effect=True,
            timeout_s=5.0,
            rate_limit=rate_limit if rate_limit is not None else RateLimit(2, 120.0),
        )
        self.chat_sourced: ChatSourced = chat_sourced

    async def __call__(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        slot = int(args["slot"])
        origin = stimulus_origin(ctx.stimulus)
        if not may_forget(origin, chat_sourced=self.chat_sourced):
            return _refused("needs_streamer", detail="only the streamer can make me forget")
        items = await ctx.memory.list_memories(kind="core", status="active")
        target = next((m for m in items if m.slot == slot), None)
        if target is None or target.id is None:
            return _refused("slot_empty", slot=slot)
        if target.locked:
            return _refused("slot_locked", slot=slot)
        try:
            await ctx.memory.forget(
                target.id, by=f"model:{origin}", reason=f"forget tool, turn {ctx.turn_id}"
            )
        except PermissionError:
            return _refused("slot_locked", slot=slot)
        return ToolResult(True, _json(ok=True, forgot_slot=slot))


def memory_tools(memory_cfg: Any | None = None) -> list[RememberTool | ForgetTool]:
    """``[RememberTool, ForgetTool]`` configured from ``AppConfig.memory`` (or defaults)."""
    if memory_cfg is None:
        return [RememberTool(), ForgetTool()]
    return [
        RememberTool(
            max_writes_per_session=memory_cfg.max_writes_per_session,
            min_interval_s=memory_cfg.min_write_interval_s,
            chat_sourced=memory_cfg.chat_sourced,
            core_slots=memory_cfg.core_slots,
            slot_max_chars=memory_cfg.slot_max_chars,
        ),
        ForgetTool(chat_sourced=memory_cfg.chat_sourced, core_slots=memory_cfg.core_slots),
    ]
