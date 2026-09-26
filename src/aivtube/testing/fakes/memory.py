"""``FakeMemoryStore``: an in-memory ``MemoryStore`` with the §6 semantics (§3.9)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aivtube.contracts.infra import Clock
from aivtube.contracts.memory import (
    MemKind,
    MemoryItem,
    MemStatus,
    PrefixMemory,
    SlotsFull,
    Turn,
)
from aivtube.testing.fakes.clock import RealClock

__all__ = ["FakeMemoryStore"]


@dataclass
class _Session:
    id: int
    started: float
    title: str | None
    ended: float | None = None
    summary: str | None = None


@dataclass
class _Epoch:
    id: int
    session_id: int
    rolling_summary: str = ""
    upto_turn_id: int = 0
    prefix_hash: str = ""
    first_memory_id: int = 0  # memories with a larger id were written during this epoch


@dataclass
class _Viewer:
    platform: str
    user_id: str
    name: str
    messages: int = 0


@dataclass
class _State:
    sessions: list[_Session] = field(default_factory=list)
    epochs: list[_Epoch] = field(default_factory=list)
    turns: list[tuple[int, Turn]] = field(default_factory=list)  # (epoch id, turn)
    memories: dict[int, MemoryItem] = field(default_factory=dict)
    viewers: dict[tuple[str, str], _Viewer] = field(default_factory=dict)
    audit: list[tuple[str, int, str, str]] = field(default_factory=list)  # (action, id, by, why)


class FakeMemoryStore:
    """In-memory ``MemoryStore``.

    - ``kind="core"`` items take one of ``core_slots`` slots (1-based) while not deleted;
      ``remember`` raises ``SlotsFull`` when none is free, unless ``replace_slot`` is given.
      Text longer than ``slot_max_chars`` raises ``ValueError``.
    - Locked items refuse ``forget`` and replacement (``PermissionError``).
    - ``prefix_block`` shows active core slots, the last 2 active episodes and the rolling summary;
      its digest changes only when those change.
    - ``search`` is a case-insensitive substring search (the real store uses FTS5 trigram).
    """

    def __init__(
        self,
        character: str = "pailin",
        clock: Clock | None = None,
        *,
        core_slots: int = 16,
        slot_max_chars: int = 120,
    ) -> None:
        self.character = character
        self.clock: Clock = clock or RealClock()
        self.core_slots = core_slots
        self.slot_max_chars = slot_max_chars
        self.state = _State()
        self._next_turn = 1
        self._next_memory = 1
        self._session: _Session | None = None
        self.backups: list[Path] = []

    # --- sessions and epochs ----------------------------------------------------------------
    async def start_session(self, title: str | None = None) -> int:
        s = _Session(len(self.state.sessions) + 1, self.clock.now(), title)
        self.state.sessions.append(s)
        self._session = s
        self._new_epoch_row(s.id, "", 0, "")
        return s.id

    async def resume_session(self, max_age_s: float) -> int | None:
        if not self.state.sessions:
            return None
        s = self.state.sessions[-1]  # only the latest session can be resumed
        if s.ended is not None or self.clock.now() - s.started > max_age_s:
            return None
        self._session = s
        return s.id

    async def end_session(self, summary: str | None = None) -> None:
        s = self._require_session()
        s.ended = self.clock.now()
        s.summary = summary
        self._session = None

    async def new_epoch(self, rolling_summary: str, upto_turn_id: int, prefix_hash: str) -> int:
        s = self._require_session()
        return self._new_epoch_row(s.id, rolling_summary, upto_turn_id, prefix_hash).id

    def _new_epoch_row(self, session_id: int, summary: str, upto: int, prefix_hash: str) -> _Epoch:
        e = _Epoch(len(self.state.epochs) + 1, session_id, summary, upto, prefix_hash)
        e.first_memory_id = self._next_memory - 1
        self.state.epochs.append(e)
        return e

    @property
    def epoch(self) -> int:
        return self.state.epochs[-1].id if self.state.epochs else 0

    # --- turns ------------------------------------------------------------------------------
    async def append_turn(self, turn: Turn) -> int:
        self._require_session()
        tid = self._next_turn
        self._next_turn += 1
        stored = dataclasses.replace(turn, id=tid, ts=turn.ts or self.clock.now())
        self.state.turns.append((self.epoch, stored))
        return tid

    async def recent_turns(self, epoch: int) -> list[Turn]:
        """Turns of the session that owns ``epoch`` after that epoch's ``upto_turn_id``."""
        e = next((x for x in self.state.epochs if x.id == epoch), None)
        if e is None:
            return []
        session_epochs = {x.id for x in self.state.epochs if x.session_id == e.session_id}
        return [
            t for ep, t in self.state.turns if ep in session_epochs and (t.id or 0) > e.upto_turn_id
        ]

    # --- prefix -----------------------------------------------------------------------------
    async def prefix_block(self) -> PrefixMemory:
        core = tuple(
            sorted(
                (
                    m
                    for m in self.state.memories.values()
                    if m.kind == "core" and m.status == "active"
                ),
                key=lambda m: m.slot or 0,
            )
        )
        episodes = tuple(
            m.text
            for m in sorted(
                (
                    m
                    for m in self.state.memories.values()
                    if m.kind == "episode" and m.status == "active"
                ),
                key=lambda m: m.id or 0,
            )[-2:]
        )
        summary = self.state.epochs[-1].rolling_summary if self.state.epochs else ""
        rendered = json.dumps(
            {
                "core": [(m.slot, m.text, m.subject) for m in core],
                "episodes": list(episodes),
                "summary": summary,
            },
            ensure_ascii=False,
        )
        digest = hashlib.sha256(rendered.encode()).hexdigest()
        return PrefixMemory(core, episodes, summary, self.epoch, digest)

    # --- long-term items --------------------------------------------------------------------
    async def remember(self, item: MemoryItem, *, replace_slot: int | None = None) -> MemoryItem:
        if not item.text.strip():
            raise ValueError("empty memory text")
        if item.kind == "core" and len(item.text) > self.slot_max_chars:
            raise ValueError(f"core memory longer than {self.slot_max_chars} characters")
        slot: int | None = None
        if item.kind == "core":
            used = {m.slot: m for m in self._core_items()}
            if replace_slot is not None:
                if not 1 <= replace_slot <= self.core_slots:
                    raise ValueError(f"replace_slot must be 1..{self.core_slots}")
                old = used.get(replace_slot)
                if old is not None:
                    if old.locked:
                        raise PermissionError(f"slot {replace_slot} is locked")
                    self._set(old, status="deleted", slot=None)
                    self.state.audit.append(("replace", old.id or 0, item.source, "replaced"))
                slot = replace_slot
            else:
                free = [s for s in range(1, self.core_slots + 1) if s not in used]
                if not free:
                    raise SlotsFull(sorted(used.values(), key=lambda m: m.slot or 0))
                slot = free[0]
        mid = self._next_memory
        self._next_memory += 1
        stored = dataclasses.replace(item, id=mid, slot=slot)
        self.state.memories[mid] = stored
        return stored

    async def forget(self, memory_id: int, *, by: str, reason: str) -> None:
        m = self._get(memory_id)
        if m.locked:
            raise PermissionError(f"memory {memory_id} is locked")
        self._set(m, status="deleted", slot=None)
        self.state.audit.append(("forget", memory_id, by, reason))

    async def set_status(self, memory_id: int, status: MemStatus, *, by: str) -> None:
        m = self._get(memory_id)
        if status == "deleted" and m.locked:
            raise PermissionError(f"memory {memory_id} is locked")
        slot = m.slot
        if m.kind == "core" and status != "deleted" and slot is None:
            used = {x.slot for x in self._core_items()}
            free = [s for s in range(1, self.core_slots + 1) if s not in used]
            if not free:
                raise SlotsFull(self._core_items())
            slot = free[0]
        self._set(m, status=status, slot=None if status == "deleted" else slot)
        self.state.audit.append(("status", memory_id, by, status))

    async def list_memories(
        self, *, kind: MemKind | None = None, status: MemStatus | None = None
    ) -> list[MemoryItem]:
        return [
            m
            for m in sorted(self.state.memories.values(), key=lambda m: m.id or 0)
            if (kind is None or m.kind == kind) and (status is None or m.status == status)
        ]

    async def pending_since_epoch(self) -> list[MemoryItem]:
        first = self.state.epochs[-1].first_memory_id if self.state.epochs else 0
        return [
            m
            for m in sorted(self.state.memories.values(), key=lambda m: m.id or 0)
            if (m.id or 0) > first and m.status != "deleted"
        ]

    async def search(self, query: str, k: int = 3) -> list[MemoryItem]:
        q = query.strip().casefold()
        if not q:
            return []
        hits = [
            m
            for m in sorted(self.state.memories.values(), key=lambda m: m.id or 0)
            if m.status == "active"
            and (q in m.text.casefold() or (m.subject is not None and q in m.subject.casefold()))
        ]
        return hits[:k]

    async def viewer_facts(
        self, users: Sequence[tuple[str, str]], limit: int = 6
    ) -> list[MemoryItem]:
        wanted = {(p, u) for p, u in users}
        out = [
            m
            for m in sorted(self.state.memories.values(), key=lambda m: m.id or 0)
            if m.kind == "viewer"
            and m.status == "active"
            and (m.platform or "", m.user_id or "") in wanted
        ]
        return out[:limit]

    async def upsert_viewer(self, platform: str, user_id: str, name: str) -> None:
        v = self.state.viewers.get((platform, user_id))
        if v is None:
            self.state.viewers[(platform, user_id)] = _Viewer(platform, user_id, name, 1)
        else:
            v.name = name
            v.messages += 1

    async def backup(self, dest_dir: Path, keep: int = 14) -> Path:
        n = len(self.backups) + 1
        payload = {
            "character": self.character,
            "memories": [dataclasses.asdict(m) for m in self.state.memories.values()],
            "turns": [dataclasses.asdict(t) for _, t in self.state.turns],
        }
        path = await asyncio.to_thread(self._write_backup, dest_dir, n, payload, keep)
        self.backups.append(path)
        return path

    def _write_backup(self, dest_dir: Path, n: int, payload: object, keep: int) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{self.character}-{n:04d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        mine = sorted(dest_dir.glob(f"{self.character}-*.json"))
        for old in mine[:-keep] if keep > 0 else []:
            old.unlink(missing_ok=True)
        return path

    # --- helpers ----------------------------------------------------------------------------
    def _require_session(self) -> _Session:
        if self._session is None:
            raise RuntimeError("no active session; call start_session() first")
        return self._session

    def _core_items(self) -> list[MemoryItem]:
        return [
            m
            for m in self.state.memories.values()
            if m.kind == "core" and m.status != "deleted" and m.slot is not None
        ]

    def _get(self, memory_id: int) -> MemoryItem:
        try:
            return self.state.memories[memory_id]
        except KeyError:
            raise KeyError(f"no memory {memory_id}") from None

    def _set(self, m: MemoryItem, **changes: object) -> None:
        assert m.id is not None
        self.state.memories[m.id] = dataclasses.replace(m, **changes)  # type: ignore[arg-type]
