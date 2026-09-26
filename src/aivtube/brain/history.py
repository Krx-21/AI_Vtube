"""In-memory conversation history of the current epoch (ARCHITECTURE.md §4.1, §4.8, §6).

Every turn is persisted through ``MemoryStore.append_turn`` (I5) and kept here in render
order. History is append-only, so llama.cpp's cached prefix stays byte-stable.

**Frozen at first render.** An assistant line renders as what was *heard*: ``heard_text`` if
known, else the emitted text; ``… [ถูกขัดจังหวะ]`` marks a cut, `` [Filtered.]`` a filtered
reply. The brain calls :meth:`History.mark_heard` (the synchronous part of :meth:`set_heard`)
with a tentative heard text before it builds a prompt and :meth:`History.freeze_all` right
after, so the text is fixed from its first render on. A later final heard text that disagrees
with the frozen text does not change it; it queues a correction note for the next tail
(:meth:`History.pending_correction`).

**Audit.** ``MemoryStore`` has no update: the final heard text of each utterance is persisted
as a hidden ``note`` row (``source="heard"``, ``turn_ref`` = the utterance id), which
:meth:`History.load` applies to its assistant line. Hidden notes never render.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from aivtube.contracts.infra import Clock
from aivtube.contracts.memory import MemoryStore, Turn
from aivtube.contracts.types import Stimulus
from aivtube.infra.clock import SystemClock, deadline

__all__ = [
    "FILTERED_MARK",
    "HIDDEN_NOTE_SOURCES",
    "INTERRUPTED_MARK",
    "History",
    "render_turn_text",
    "visible",
]

log = logging.getLogger("aivtube.brain.history")

INTERRUPTED_MARK: Final = " [ถูกขัดจังหวะ]"
FILTERED_MARK: Final = " [Filtered.]"
#: Note sources that are stored for audit but never rendered into a prompt.
HIDDEN_NOTE_SOURCES: Final = frozenset({"heard", "correction"})
_WRITE_TIMEOUT_S: Final = 5.0


def visible(turn: Turn) -> bool:
    """Whether a stored turn renders into the prompt."""
    return not (turn.role == "note" and turn.source in HIDDEN_NOTE_SOURCES)


def render_turn_text(turn: Turn) -> str:
    """The text a turn renders as (assistant lines: heard text plus cut/filter markers)."""
    if turn.role != "assistant":
        return turn.text
    base = (turn.heard_text if turn.heard_text is not None else turn.text).rstrip()
    if turn.filtered:
        return (base + FILTERED_MARK).lstrip()
    if turn.interrupted:
        return base + "…" + INTERRUPTED_MARK
    return base


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if unicodedata.category(ch)[0] in "LMN")


@dataclass(slots=True, eq=False)
class _Entry:
    turn: Turn
    frozen: bool = False


class History:
    """See the module docstring. Writes are serialised, so persisted order = render order."""

    def __init__(
        self,
        memory: MemoryStore,
        *,
        budget_tokens: int,
        estimate: Callable[[str], int],
        clock: Clock | None = None,
    ) -> None:
        self._memory = memory
        self.budget_tokens = budget_tokens
        self._estimate = estimate
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._entries: list[_Entry] = []
        self._corrections: list[str] = []
        self._lock = asyncio.Lock()
        self._local_ids = itertools.count(-1, -1)  # ids for turns the store refused
        self._max_id = 0  # the highest id the store gave us (hidden notes included)
        self.epoch = 0

    # --- loading ----------------------------------------------------------------------------
    async def load(self, epoch: int) -> None:
        """Rebuild from the store (core restart, I5): every loaded turn counts as rendered."""
        async with deadline(_WRITE_TIMEOUT_S, what="history load", clock=self._clock):
            turns = await self._memory.recent_turns(epoch)
        heard: dict[str, Turn] = {}
        for t in turns:
            if t.role == "note" and t.source == "heard" and t.turn_ref:
                heard[t.turn_ref] = t
        entries: list[_Entry] = []
        for t in turns:
            if t.id is not None and t.id > self._max_id:
                self._max_id = t.id
            if not visible(t):
                continue
            audit = heard.get(t.turn_ref) if t.role == "assistant" else None
            if audit is not None:
                t = dataclasses.replace(
                    t,
                    heard_text=audit.heard_text,
                    interrupted=audit.interrupted,
                    filtered=audit.filtered,
                )
            entries.append(_Entry(t, frozen=True))
        self._entries = entries
        self._corrections.clear()
        self.epoch = epoch

    # --- reading ----------------------------------------------------------------------------
    def turns(self) -> list[Turn]:
        return [e.turn for e in self._entries]

    def get(self, turn_id: int) -> Turn | None:
        entry = self._find(turn_id)
        return entry.turn if entry is not None else None

    def tokens(self) -> int:
        return sum(self._estimate(render_turn_text(e.turn)) for e in self._entries)

    def needs_compaction(self) -> bool:
        return self.tokens() > self.budget_tokens

    def base_id(self) -> int:
        """The turn id just before this history: a new epoch with ``upto_turn_id = base_id()``
        keeps exactly these turns (``MemoryStore.recent_turns`` returns ids above it)."""
        ids = [e.turn.id for e in self._entries if e.turn.id is not None and e.turn.id > 0]
        return min(ids) - 1 if ids else self._max_id

    # --- appending --------------------------------------------------------------------------
    async def append_user(
        self, compact_text: str, stimulus: Stimulus, *, turn_ref: str = ""
    ) -> int:
        return await self._append(
            Turn(
                role="user",
                text=compact_text,
                source=stimulus.kind.value,
                speaker=stimulus.speaker,
                turn_ref=turn_ref or stimulus.id,
            )
        )

    async def append_assistant(
        self,
        *,
        turn_ref: str,
        emitted: str,
        heard: str | None,
        interrupted: bool,
        filtered: bool,
        provider: str | None,
        tool_calls: str | None,
        provider_extra: str | None,
    ) -> int:
        return await self._append(
            Turn(
                role="assistant",
                text=emitted,
                source="llm",
                heard_text=heard,
                interrupted=interrupted,
                filtered=filtered,
                provider=provider,
                turn_ref=turn_ref,
                tool_calls=tool_calls,
                provider_extra=provider_extra,
            )
        )

    async def append_tool(self, content: str, turn_ref: str) -> int:
        return await self._append(Turn(role="tool", text=content, source="tool", turn_ref=turn_ref))

    async def append_note(self, text: str, *, source: str = "note", turn_ref: str = "") -> int:
        return await self._append(Turn(role="note", text=text, source=source, turn_ref=turn_ref))

    # --- heard text and freezing ------------------------------------------------------------
    def freeze(self, turn_id: int) -> None:
        entry = self._find(turn_id)
        if entry is not None:
            entry.frozen = True

    def freeze_all(self) -> None:
        for entry in self._entries:
            entry.frozen = True

    def is_frozen(self, turn_id: int) -> bool:
        entry = self._find(turn_id)
        return entry is not None and entry.frozen

    def mark_heard(
        self,
        turn_id: int,
        heard: str,
        *,
        interrupted: bool,
        filtered: bool,
        final: bool = True,
    ) -> Turn | None:
        """Synchronous part of :meth:`set_heard`: update the line (or queue a correction if it
        is frozen). Returns the hidden audit row to persist when ``final``, else ``None``."""
        entry = self._find(turn_id)
        if entry is None or entry.turn.role != "assistant":
            return None
        before = entry.turn
        updated = dataclasses.replace(
            before, heard_text=heard, interrupted=interrupted, filtered=filtered
        )
        if not entry.frozen:
            entry.turn = updated
        elif final and _differs(before, updated):
            self._corrections.append(_correction(updated))
        if not final:
            return None
        return Turn(
            role="note",
            text="",
            source="heard",
            heard_text=heard,
            interrupted=interrupted,
            filtered=filtered,
            turn_ref=before.turn_ref,
            ts=self._clock.now(),
        )

    async def persist_audit(self, audit: Turn) -> None:
        """Store a hidden audit row from :meth:`mark_heard` (never raises)."""
        await self._persist(audit)

    async def set_heard(
        self,
        turn_id: int,
        heard: str,
        *,
        interrupted: bool,
        filtered: bool,
        final: bool = True,
    ) -> None:
        """Record what was heard of an assistant line (see the module docstring)."""
        audit = self.mark_heard(
            turn_id, heard, interrupted=interrupted, filtered=filtered, final=final
        )
        if audit is not None:
            await self._persist(audit)

    def pending_correction(self) -> str | None:
        """Correction notes for the next tail (consumed by the call)."""
        if not self._corrections:
            return None
        text = " ".join(self._corrections)
        self._corrections.clear()
        return text

    async def rebase(self, epoch: int, upto_turn_id: int) -> None:
        """Epoch flip: turns up to ``upto_turn_id`` now live in the rolling summary."""
        self.rebase_now(epoch, upto_turn_id)

    def rebase_now(self, epoch: int, upto_turn_id: int) -> None:
        """Synchronous :meth:`rebase`, so an epoch flip is atomic on the event loop. An append
        still being persisted lands after the cut, which is where it belongs."""
        self._entries = [
            e
            for e in self._entries
            if e.turn.id is None or e.turn.id < 0 or e.turn.id > upto_turn_id
        ]
        self.epoch = epoch

    # --- internals --------------------------------------------------------------------------
    async def _append(self, turn: Turn) -> int:
        async with self._lock:
            stamped = dataclasses.replace(turn, ts=turn.ts or self._clock.now())
            turn_id = await self._persist(stamped)
            self._entries.append(_Entry(dataclasses.replace(stamped, id=turn_id)))
            return turn_id

    async def _persist(self, turn: Turn) -> int:
        try:
            async with deadline(_WRITE_TIMEOUT_S, what="history write", clock=self._clock):
                turn_id = await self._memory.append_turn(turn)
            self._max_id = max(self._max_id, turn_id)
            return turn_id
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("history: could not persist a %s turn; kept in memory only", turn.role)
            return next(self._local_ids)

    def _find(self, turn_id: int) -> _Entry | None:
        for entry in reversed(self._entries):
            if entry.turn.id == turn_id:
                return entry
        return None


def _differs(a: Turn, b: Turn) -> bool:
    if a.filtered != b.filtered:
        return True
    return _letters(render_turn_text(a)) != _letters(render_turn_text(b))


def _correction(t: Turn) -> str:
    heard = (t.heard_text or "").strip()
    if t.filtered:
        return "(ประโยคก่อนหน้าของเธอถูกกรอง)"
    if not heard:
        return "(แก้ไข: คนดูไม่ได้ยินประโยคก่อนหน้าของเธอเลย)"
    return f"(แก้ไข: ประโยคก่อนหน้าของเธอถูกตัด คนดูได้ยินแค่ “{heard}”)"
