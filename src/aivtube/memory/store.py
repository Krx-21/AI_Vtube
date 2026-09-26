"""``SqliteMemory``: the per-character memory store (ARCHITECTURE.md §3.9, §6).

One SQLite file per character (``data/memory/<char>.sqlite``) holding sessions, epochs, every
history turn, long-term items (core slots, facts, viewer facts, episodes) and viewers. All I/O
runs on one worker thread (``SqliteWorker``); the event loop only awaits it.

Semantics beyond the Protocol docstrings:

- **Time.** Rows store wall-clock seconds (``Clock.wall()``). ``Turn.ts`` is perf_counter
  seconds (§3.1): it is converted to wall time on write and back on read.
- **Sessions.** ``start_session`` closes any session left open by a crash. ``resume_session``
  resumes the latest session if it was not ended and its last activity (last turn, else its
  start) is at most ``max_age_s`` old.
- **Core slots.** ``kind="core"`` items are pinned and hold one of ``core_slots`` slots
  (1-based). An active write takes the lowest free slot or raises ``SlotsFull``;
  ``replace_slot`` deletes the slot's occupant (locked occupants raise ``MemoryLocked``). A
  quarantined core item holds no slot until approved: ``replace_slot`` is kept as the slot it
  asks for and the occupant is replaced only on approval (``set_status(..., "active")``). A
  quarantined write still raises ``SlotsFull`` when every slot is taken and no
  ``replace_slot`` was given, so the model learns about it at write time.
- **Quarantine.** See ``aivtube.memory.policy``. Quarantined items never appear in
  ``prefix_block``, ``search``, ``viewer_facts`` or ``pending_since_epoch``.
- **Pending.** ``pending_since_epoch`` lists items that became active during the current epoch
  (``memory.epoch_seen``), i.e. what belongs in ``<new_memories>`` until the next flip.
- **Digest.** ``PrefixMemory.digest`` is the sha256 of ``render_context(...)``: it changes only
  when active core slots, the last two episodes or the rolling summary change.
- **Search.** FTS5 trigram when available (terms of 3+ characters: the whole query plus its
  words, Thai split with newmm), ranked by bm25; otherwise, or for shorter queries, ``LIKE``.
- **Failures.** Busy or corrupt files fall back to an in-memory database seeded with the
  current session, epoch and core slots; ``health()`` turns DEGRADED (§2.8).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import sqlite3
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from aivtube.contracts.events import HealthChanged, MemoryWritten
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.memory import (
    MemKind,
    MemoryItem,
    MemStatus,
    PrefixMemory,
    SlotsFull,
    Turn,
)
from aivtube.contracts.types import Health, HealthState
from aivtube.memory._db import (
    SqliteWorker,
    TxnMode,
    apply_migrations,
    fts5_trigram_available,
    load_migrations,
    read_sql,
)
from aivtube.memory.backup import backup_connection
from aivtube.memory.policy import ChatSourced, initial_status

__all__ = [
    "MemoryLocked",
    "SqliteMemory",
    "ViewerOptedOut",
    "render_context",
]

log = logging.getLogger("aivtube.memory")
_T = TypeVar("_T")

_ITEM_COLS = (
    "id, kind, text, slot, subject, platform, user_id, importance, source, origin, status, "
    "pinned, locked"
)
_TURN_COLS = (
    "id, ts, role, source, speaker, text, heard_text, interrupted, filtered, provider, "
    "turn_ref, tool_calls, provider_extra"
)
_FTS_TRIGGERS = ("memory_ai", "memory_ad", "memory_au")
# Thai particles and question words that carry no recall signal.
_STOPWORDS = frozenset(
    {"ครับ", "ค่ะ", "นะคะ", "จ้า", "ไหม", "มั้ย", "เหรอ", "อะไร", "ยังไง", "ทำไม", "แล้ว", "เป็น"}
)
_TERM_STRIP = " \t\r\n\"'`.,!?;:()[]{}<>«»“”‘’…~*_/\\|#@"
_MAX_TERMS = 16


class MemoryLocked(PermissionError):
    """The item (or the slot's occupant) is locked by the operator."""


class ViewerOptedOut(PermissionError):
    """The viewer opted out of being remembered."""


def render_context(
    core: Sequence[MemoryItem], episodes: Sequence[str], rolling_summary: str
) -> str:
    """The canonical text of the memory context block; ``PrefixMemory.digest`` hashes it."""
    lines: list[str] = []
    if core:
        lines.append("[ความจำ]")
        for m in core:
            about = f" (เกี่ยวกับ {m.subject})" if m.subject else ""
            lines.append(f"{m.slot}. {m.text}{about}")
    if episodes:
        lines.append("[สตรีมก่อน ๆ]")
        lines.extend(f"- {e}" for e in episodes)
    if rolling_summary:
        lines.append("[สรุปก่อนหน้า]")
        lines.append(rolling_summary)
    return "\n".join(lines)


def _item(row: Sequence[Any]) -> MemoryItem:
    return MemoryItem(
        id=int(row[0]),
        kind=row[1],
        text=row[2],
        slot=row[3],
        subject=row[4],
        platform=row[5],
        user_id=row[6],
        importance=int(row[7]),
        source=row[8],
        origin=row[9],
        status=row[10],
        pinned=bool(row[11]),
        locked=bool(row[12]),
    )


def _turn(row: Sequence[Any], now: float, wall: float) -> Turn:
    """A ``turn`` row as a ``Turn``; the stored wall time is mapped back to perf_counter."""
    return Turn(
        id=int(row[0]),
        ts=now - (wall - float(row[1])),
        role=row[2],
        source=row[3],
        speaker=row[4],
        text=row[5],
        heard_text=row[6],
        interrupted=bool(row[7]),
        filtered=bool(row[8]),
        provider=row[9],
        turn_ref=row[10] or "",
        tool_calls=row[11],
        provider_extra=row[12],
    )


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _has_thai(text: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


class SqliteMemory:
    """``MemoryStore`` on SQLite (see the module docstring).

    Extra keyword arguments beyond the modules.json API: ``bus`` (publishes ``MemoryWritten``
    and ``HealthChanged``), ``chat_sourced`` (the §6 quarantine switch), ``fts`` (``False``
    forces the ``LIKE`` fallback), ``busy_timeout_ms`` and ``tokenizer`` (Thai words for
    search; defaults to pythainlp newmm). ``path=None`` keeps everything in memory.
    """

    def __init__(
        self,
        path: Path | None,
        character: str,
        clock: Clock,
        *,
        core_slots: int = 16,
        slot_max_chars: int = 120,
        executor: concurrent.futures.Executor | None = None,
        bus: EventBus | None = None,
        chat_sourced: ChatSourced = "quarantine",
        fts: bool | None = None,
        busy_timeout_ms: int = 5000,
        tokenizer: Callable[[str], list[str]] | None = None,
        max_text_chars: int = 2000,
    ) -> None:
        if core_slots < 1:
            raise ValueError("core_slots must be >= 1")
        self.character = character
        self.name = f"memory.{character}"
        self.core_slots = core_slots
        self.slot_max_chars = slot_max_chars
        self.max_text_chars = max_text_chars
        self.chat_sourced: ChatSourced = chat_sourced
        self._clock = clock
        self._bus = bus
        available = fts5_trigram_available()
        if fts and not available:
            log.warning("FTS5 trigram is not available in SQLite; memory search uses LIKE")
        self.fts: bool = available if fts is None else (fts and available)
        self._tokenizer = tokenizer
        self._db = SqliteWorker(
            path,
            name=self.name,
            migrate=self._migrate,
            busy_timeout_ms=busy_timeout_ms,
            executor=executor,
        )
        self._db.on_fallback = self._seed_fallback
        self._db.on_degraded = self._on_degraded
        # State below is only written on the worker thread (inside a job).
        self._session_id: int | None = None
        self._epoch_id = 0
        self._summary = ""
        self._core_cache: tuple[MemoryItem, ...] = ()
        self._health = Health(self.name, HealthState.STARTING, "", clock.now())

    @classmethod
    def from_config(
        cls,
        app_cfg: Any,
        char_cfg: Any,
        clock: Clock,
        *,
        bus: EventBus | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> SqliteMemory:
        """Build from ``AppConfig.memory`` and a ``CharacterConfig`` (``[memory] db``)."""
        mem = app_cfg.memory
        return cls(
            char_cfg.resolve_path(char_cfg.memory.db),
            char_cfg.id,
            clock,
            core_slots=mem.core_slots,
            slot_max_chars=mem.slot_max_chars,
            executor=executor,
            bus=bus,
            chat_sourced=mem.chat_sourced,
        )

    # --- Component-style lifecycle ------------------------------------------------------------
    async def start(self) -> None:
        """Open the database now (it otherwise opens on first use)."""
        await self._run(lambda conn: None, "raw")

    async def aclose(self) -> None:
        await self._db.aclose()

    def health(self) -> Health:
        return self._health

    @property
    def session_id(self) -> int | None:
        """The current session (``None`` before ``start_session``/after ``end_session``)."""
        return self._session_id

    @property
    def epoch(self) -> int:
        return self._epoch_id

    @property
    def degraded(self) -> bool:
        return self._db.degraded is not None

    # --- sessions and epochs ------------------------------------------------------------------
    async def start_session(self, title: str | None = None) -> int:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> int:
            conn.execute(
                "UPDATE session SET ended_at = COALESCE("
                "(SELECT MAX(ts) FROM turn WHERE turn.session_id = session.id), started_at) "
                "WHERE ended_at IS NULL"
            )
            sid = _lastrowid(
                conn.execute("INSERT INTO session(started_at, title) VALUES(?, ?)", (wall, title))
            )
            eid = _lastrowid(
                conn.execute("INSERT INTO epoch(session_id, started_at) VALUES(?, ?)", (sid, wall))
            )
            self._session_id, self._epoch_id, self._summary = sid, eid, ""
            return sid

        return await self._run(op, "write")

    async def resume_session(self, max_age_s: float) -> int | None:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> int | None:
            row = conn.execute(
                "SELECT id, started_at, ended_at FROM session ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None or row[2] is not None:
                return None
            sid, started = int(row[0]), float(row[1])
            last = conn.execute("SELECT MAX(ts) FROM turn WHERE session_id = ?", (sid,)).fetchone()
            activity = max(started, float(last[0])) if last[0] is not None else started
            if wall - activity > max_age_s:
                return None
            ep = conn.execute(
                "SELECT id, rolling_summary FROM epoch WHERE session_id = ? ORDER BY id DESC "
                "LIMIT 1",
                (sid,),
            ).fetchone()
            if ep is None:
                eid = _lastrowid(
                    conn.execute(
                        "INSERT INTO epoch(session_id, started_at) VALUES(?, ?)", (sid, wall)
                    )
                )
                summary = ""
            else:
                eid, summary = int(ep[0]), str(ep[1])
            self._session_id, self._epoch_id, self._summary = sid, eid, summary
            return sid

        return await self._run(op, "write")

    async def end_session(self, summary: str | None = None) -> None:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> None:
            sid = self._session_id
            if sid is None:
                log.debug("end_session() without an active session")
                return
            conn.execute(
                "UPDATE session SET ended_at = ?, summary = COALESCE(?, summary) WHERE id = ?",
                (wall, summary, sid),
            )
            self._session_id = None

        await self._run(op, "write")

    async def new_epoch(self, rolling_summary: str, upto_turn_id: int, prefix_hash: str) -> int:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> int:
            sid = self._require_session()
            eid = _lastrowid(
                conn.execute(
                    "INSERT INTO epoch(session_id, started_at, rolling_summary, upto_turn_id, "
                    "prefix_hash) VALUES(?, ?, ?, ?, ?)",
                    (sid, wall, rolling_summary, upto_turn_id, prefix_hash),
                )
            )
            self._epoch_id, self._summary = eid, rolling_summary
            return eid

        return await self._run(op, "write")

    # --- turns --------------------------------------------------------------------------------
    async def append_turn(self, turn: Turn) -> int:
        now, wall = self._clock.now(), self._clock.wall()
        ts = wall - (now - turn.ts) if turn.ts else wall

        def op(conn: sqlite3.Connection) -> int:
            sid = self._require_session()
            return _lastrowid(
                conn.execute(
                    "INSERT INTO turn(session_id, epoch_id, ts, role, source, speaker, text, "
                    "heard_text, interrupted, filtered, provider, turn_ref, tool_calls, "
                    "provider_extra) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        self._epoch_id,
                        ts,
                        turn.role,
                        turn.source,
                        turn.speaker,
                        turn.text,
                        turn.heard_text,
                        int(turn.interrupted),
                        int(turn.filtered),
                        turn.provider,
                        turn.turn_ref,
                        turn.tool_calls,
                        turn.provider_extra,
                    ),
                )
            )

        return await self._run(op, "write")

    async def recent_turns(self, epoch: int) -> list[Turn]:
        """The turns of ``epoch``'s session after the epoch's ``upto_turn_id``, oldest first."""
        now, wall = self._clock.now(), self._clock.wall()

        def op(conn: sqlite3.Connection) -> list[Turn]:
            ep = conn.execute(
                "SELECT session_id, COALESCE(upto_turn_id, 0) FROM epoch WHERE id = ?", (epoch,)
            ).fetchone()
            if ep is None:
                return []
            rows = conn.execute(
                f"SELECT {_TURN_COLS} FROM turn WHERE session_id = ? AND id > ? ORDER BY id",
                (ep[0], ep[1]),
            ).fetchall()
            return [_turn(r, now, wall) for r in rows]

        return await self._run(op, "read")

    async def session_turns(self, session_id: int | None = None) -> list[Turn]:
        """Every turn of a session (default: the current one), oldest first; for the episode
        summary job, which needs the whole session regardless of epochs."""
        now, wall = self._clock.now(), self._clock.wall()

        def op(conn: sqlite3.Connection) -> list[Turn]:
            sid = self._session_id if session_id is None else session_id
            if sid is None:
                return []
            rows = conn.execute(
                f"SELECT {_TURN_COLS} FROM turn WHERE session_id = ? ORDER BY id", (sid,)
            ).fetchall()
            return [_turn(r, now, wall) for r in rows]

        return await self._run(op, "read")

    # --- prefix -------------------------------------------------------------------------------
    async def prefix_block(self) -> PrefixMemory:
        def op(conn: sqlite3.Connection) -> PrefixMemory:
            core = self._refresh_core(conn)
            episodes = tuple(
                str(r[0])
                for r in reversed(
                    conn.execute(
                        "SELECT text FROM memory WHERE kind = 'episode' AND status = 'active' "
                        "ORDER BY id DESC LIMIT 2"
                    ).fetchall()
                )
            )
            summary = ""
            if self._epoch_id:
                row = conn.execute(
                    "SELECT rolling_summary FROM epoch WHERE id = ?", (self._epoch_id,)
                ).fetchone()
                summary = str(row[0]) if row is not None else self._summary
            digest = hashlib.sha256(
                render_context(core, episodes, summary).encode("utf-8")
            ).hexdigest()
            return PrefixMemory(core, episodes, summary, self._epoch_id, digest)

        return await self._run(op, "read")

    # --- long-term items ----------------------------------------------------------------------
    async def remember(self, item: MemoryItem, *, replace_slot: int | None = None) -> MemoryItem:
        text = item.text.strip()
        self._validate(item, text, replace_slot)
        status = initial_status(
            item.kind,
            item.source,
            item.origin,
            requested=item.status,
            chat_sourced=self.chat_sourced,
        )
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> tuple[MemoryItem, bool]:
            if item.kind == "viewer":
                opt = conn.execute(
                    "SELECT opt_out FROM viewer WHERE platform = ? AND user_id = ?",
                    (item.platform, item.user_id),
                ).fetchone()
                if opt is not None and opt[0]:
                    raise ViewerOptedOut(f"viewer {item.platform}:{item.user_id} opted out")
            dup = conn.execute(
                f"SELECT {_ITEM_COLS} FROM memory WHERE kind = ? AND text = ? AND status = ? "
                "AND COALESCE(platform, '') = ? AND COALESCE(user_id, '') = ? LIMIT 1",
                (item.kind, text, status, item.platform or "", item.user_id or ""),
            ).fetchone()
            if dup is not None and (replace_slot is None or dup[3] == replace_slot):
                return _item(dup), False
            slot: int | None = None
            if item.kind == "core":
                used = self._active_slots(conn)
                if replace_slot is not None:
                    occupant = used.get(replace_slot)
                    if occupant is not None and occupant.locked:
                        raise MemoryLocked(f"slot {replace_slot} is locked")
                    if occupant is not None and status == "active":
                        self._delete_row(conn, occupant.id, wall)
                    slot = replace_slot
                else:
                    free = self._free_slot(used)
                    if free is None:
                        raise SlotsFull(sorted(used.values(), key=lambda m: m.slot or 0))
                    slot = free if status == "active" else None
            pinned = item.kind == "core" or item.pinned
            mid = _lastrowid(
                conn.execute(
                    "INSERT INTO memory(kind, slot, subject, platform, user_id, text, importance, "
                    "source, origin, status, pinned, locked, epoch_seen, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.kind,
                        slot,
                        item.subject,
                        item.platform,
                        item.user_id,
                        text,
                        item.importance,
                        item.source,
                        item.origin,
                        status,
                        int(pinned),
                        int(item.locked),
                        self._epoch_id if status == "active" else None,
                        wall,
                        wall,
                    ),
                )
            )
            if item.kind == "core":
                self._refresh_core(conn)
            stored = MemoryItem(
                id=mid,
                kind=item.kind,
                text=text,
                slot=slot,
                subject=item.subject,
                platform=item.platform,
                user_id=item.user_id,
                importance=item.importance,
                source=item.source,
                origin=item.origin,
                status=status,
                pinned=pinned,
                locked=item.locked,
            )
            return stored, True

        stored, created = await self._run(op, "write")
        if created:
            self._written(stored)
        return stored

    async def forget(self, memory_id: int, *, by: str, reason: str) -> None:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> MemoryItem | None:
            row = self._get(conn, memory_id)
            if row.locked:
                raise MemoryLocked(f"memory {memory_id} is locked")
            if row.status == "deleted":
                return None
            self._delete_row(conn, memory_id, wall)
            if row.kind == "core":
                self._refresh_core(conn)
            return row

        row = await self._run(op, "write")
        if row is not None:
            log.info("memory %d (%s) forgotten by %s: %s", memory_id, row.kind, by, reason)
            self._written(MemoryItem(id=memory_id, kind=row.kind, text=row.text, status="deleted"))

    async def set_status(self, memory_id: int, status: MemStatus, *, by: str) -> None:
        """Approve (``active``), quarantine or delete an item. Approving a core item takes its
        requested slot (replacing an unlocked occupant) or the lowest free one, else raises
        ``SlotsFull``; approving a fact about an opted-out viewer raises ``ViewerOptedOut``."""
        if status not in ("active", "quarantined", "deleted"):
            raise ValueError(f"unknown status {status!r}")
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> MemoryItem | None:
            row = self._get(conn, memory_id)
            if row.status == status:
                return None
            if status == "deleted":
                if row.locked:
                    raise MemoryLocked(f"memory {memory_id} is locked")
                self._delete_row(conn, memory_id, wall)
            elif status == "quarantined":
                conn.execute(
                    "UPDATE memory SET status = 'quarantined', epoch_seen = NULL, updated_at = ? "
                    "WHERE id = ?",
                    (wall, memory_id),
                )
            else:
                slot = row.slot
                if row.kind == "core":
                    used = self._active_slots(conn)
                    if slot is not None and slot in used and used[slot].id != memory_id:
                        occupant = used[slot]
                        if occupant.locked:
                            raise MemoryLocked(f"slot {slot} is locked")
                        assert occupant.id is not None
                        self._delete_row(conn, occupant.id, wall)
                    elif slot is None or not 1 <= slot <= self.core_slots:
                        slot = self._free_slot(used)
                        if slot is None:
                            raise SlotsFull(sorted(used.values(), key=lambda m: m.slot or 0))
                if row.kind == "viewer" and self._opted_out(conn, row.platform, row.user_id):
                    raise ViewerOptedOut(f"viewer {row.platform}:{row.user_id} opted out")
                conn.execute(
                    "UPDATE memory SET status = 'active', slot = ?, epoch_seen = ?, "
                    "updated_at = ? WHERE id = ?",
                    (slot, self._epoch_id, wall, memory_id),
                )
            if row.kind == "core":
                self._refresh_core(conn)
            return row

        row = await self._run(op, "write")
        if row is not None:
            log.info("memory %d (%s) set %s by %s", memory_id, row.kind, status, by)
            self._written(MemoryItem(id=memory_id, kind=row.kind, text=row.text, status=status))

    async def edit(
        self,
        memory_id: int,
        *,
        by: str,
        text: str | None = None,
        subject: str | None = None,
        importance: int | None = None,
        locked: bool | None = None,
    ) -> MemoryItem:
        """Operator edit (panel ``MEMORY_EDIT``): change text, subject, importance or lock."""
        if text is not None:
            text = text.strip()
            if not text:
                raise ValueError("empty memory text")
        if importance is not None and not 1 <= importance <= 5:
            raise ValueError("importance must be 1..5")
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> MemoryItem:
            row = self._get(conn, memory_id)
            new_text = row.text if text is None else text
            limit = self.slot_max_chars if row.kind == "core" else self.max_text_chars
            if len(new_text) > limit:
                raise ValueError(f"memory text longer than {limit} characters")
            conn.execute(
                "UPDATE memory SET text = ?, subject = COALESCE(?, subject), "
                "importance = COALESCE(?, importance), locked = COALESCE(?, locked), "
                "updated_at = ? WHERE id = ?",
                (
                    new_text,
                    subject,
                    importance,
                    None if locked is None else int(locked),
                    wall,
                    memory_id,
                ),
            )
            if row.kind == "core":
                self._refresh_core(conn)
            return self._get(conn, memory_id)

        item = await self._run(op, "write")
        log.info("memory %d edited by %s", memory_id, by)
        self._written(item)
        return item

    async def list_memories(
        self, *, kind: MemKind | None = None, status: MemStatus | None = None
    ) -> list[MemoryItem]:
        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            rows = conn.execute(
                f"SELECT {_ITEM_COLS} FROM memory WHERE (? IS NULL OR kind = ?) "
                "AND (? IS NULL OR status = ?) ORDER BY id",
                (kind, kind, status, status),
            ).fetchall()
            return [_item(r) for r in rows]

        return await self._run(op, "read")

    async def pending_since_epoch(self) -> list[MemoryItem]:
        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            rows = conn.execute(
                f"SELECT {_ITEM_COLS} FROM memory WHERE status = 'active' AND epoch_seen = ? "
                "ORDER BY id",
                (self._epoch_id,),
            ).fetchall()
            return [_item(r) for r in rows]

        return await self._run(op, "read")

    async def search(
        self, query: str, k: int = 3, *, kinds: Collection[MemKind] | None = None
    ) -> list[MemoryItem]:
        """Active items matching ``query``, best first (``kinds`` narrows the item kinds)."""
        q = " ".join(query.split())
        if not q or k <= 0:
            return []
        kind_list = tuple(kinds) if kinds is not None else None
        if kind_list is not None and not kind_list:
            return []

        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            terms = self._terms(q)
            if self.fts and terms:
                return self._search_fts(conn, terms, k, kind_list)
            return self._search_like(conn, terms or [q], k, kind_list)

        return await self._run(op, "read")

    async def viewer_facts(
        self, users: Sequence[tuple[str, str]], limit: int = 6
    ) -> list[MemoryItem]:
        pairs = list(dict.fromkeys((str(p), str(u)) for p, u in users))
        if not pairs or limit <= 0:
            return []

        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            out: list[MemoryItem] = []
            for start in range(0, len(pairs), 200):  # stay far below the bound-parameter limit
                chunk = pairs[start : start + 200]
                values = ", ".join("(?, ?)" for _ in chunk)
                params: list[Any] = [x for pair in chunk for x in pair]
                rows = conn.execute(
                    f"SELECT {_ITEM_COLS} FROM memory m WHERE kind = 'viewer' "
                    f"AND status = 'active' AND (platform, user_id) IN (VALUES {values}) "
                    "AND NOT EXISTS (SELECT 1 FROM viewer v WHERE v.platform = m.platform "
                    "AND v.user_id = m.user_id AND v.opt_out = 1) "
                    "ORDER BY importance DESC, id DESC LIMIT ?",
                    (*params, limit),
                ).fetchall()
                out.extend(_item(r) for r in rows)
            out.sort(key=lambda m: (-m.importance, -(m.id or 0)))
            return out[:limit]

        return await self._run(op, "read")

    async def upsert_viewer(self, platform: str, user_id: str, name: str) -> None:
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO viewer(platform, user_id, name, first_seen, last_seen, messages) "
                "VALUES(?, ?, ?, ?, ?, 1) ON CONFLICT(platform, user_id) DO UPDATE SET "
                "name = excluded.name, last_seen = excluded.last_seen, "
                "messages = viewer.messages + 1",
                (platform, user_id, name, wall, wall),
            )

        await self._run(op, "write")

    async def set_viewer_opt_out(
        self, platform: str, user_id: str, opt_out: bool = True, *, by: str = "operator"
    ) -> int:
        """Set a viewer's opt-out. Opting out also deletes their stored facts (returns how many)."""
        wall = self._clock.wall()

        def op(conn: sqlite3.Connection) -> int:
            conn.execute(
                "INSERT INTO viewer(platform, user_id, first_seen, last_seen, opt_out) "
                "VALUES(?, ?, ?, ?, ?) ON CONFLICT(platform, user_id) DO UPDATE SET "
                "opt_out = excluded.opt_out",
                (platform, user_id, wall, wall, int(opt_out)),
            )
            if not opt_out:
                return 0
            cur = conn.execute(
                "UPDATE memory SET status = 'deleted', updated_at = ? WHERE kind = 'viewer' "
                "AND platform = ? AND user_id = ? AND status != 'deleted' AND locked = 0",
                (wall, platform, user_id),
            )
            return int(cur.rowcount)

        removed = await self._run(op, "write")
        log.info(
            "viewer %s:%s opt_out=%s by %s (%d facts removed)",
            platform,
            user_id,
            opt_out,
            by,
            removed,
        )
        return removed

    async def viewer(self, platform: str, user_id: str) -> Mapping[str, Any] | None:
        """The viewer row (name, counters, opt_out, …) or ``None``."""

        def op(conn: sqlite3.Connection) -> Mapping[str, Any] | None:
            cur = conn.execute(
                "SELECT * FROM viewer WHERE platform = ? AND user_id = ?", (platform, user_id)
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {d[0]: v for d, v in zip(cur.description, row, strict=True)}

        return await self._run(op, "read")

    async def backup(self, dest_dir: Path, keep: int = 14) -> Path:
        wall = self._clock.wall()
        return await self._run(
            lambda conn: backup_connection(
                conn, dest_dir, stem=self.character, wall=wall, keep=keep
            ),
            "raw",
        )

    # --- internals (worker thread) ------------------------------------------------------------
    async def _run(self, fn: Callable[[sqlite3.Connection], _T], mode: TxnMode) -> _T:
        result = await self._db.run(fn, mode=mode)
        if self._health.state is HealthState.STARTING:
            self._health = Health(self.name, HealthState.OK, "", self._clock.now())
        return result

    def _migrate(self, conn: sqlite3.Connection, is_file: bool) -> None:
        apply_migrations(conn, load_migrations("memory"))
        conn.execute("BEGIN IMMEDIATE")
        try:
            has_fts = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
                ).fetchone()
                is not None
            )
            stale = (
                conn.execute("SELECT value FROM meta WHERE key = 'fts_stale'").fetchone()
                is not None
            )
            if self.fts:
                for stmt in _split_sql(read_sql("memory_fts_trigram.sql")):
                    conn.execute(stmt)
                if not has_fts or stale:
                    conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
                    conn.execute("DELETE FROM meta WHERE key = 'fts_stale'")
            else:
                for trigger in _FTS_TRIGGERS:
                    conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                if has_fts:
                    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('fts_stale', '1')")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _seed_fallback(self, conn: sqlite3.Connection) -> None:
        """Carry the current session, epoch and active core slots into the fallback DB."""
        wall = self._clock.wall()
        sid = self._session_id
        if sid is not None:
            conn.execute(
                "INSERT INTO session(id, started_at, title) VALUES(?, ?, 'fallback')", (sid, wall)
            )
            if self._epoch_id:
                conn.execute(
                    "INSERT INTO epoch(id, session_id, started_at, rolling_summary) "
                    "VALUES(?, ?, ?, ?)",
                    (self._epoch_id, sid, wall, self._summary),
                )
        for m in self._core_cache:
            conn.execute(
                "INSERT INTO memory(id, kind, slot, subject, platform, user_id, text, importance, "
                "source, origin, status, pinned, locked, created_at, updated_at) "
                "VALUES(?, 'core', ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, ?)",
                (
                    m.id,
                    m.slot,
                    m.subject,
                    m.platform,
                    m.user_id,
                    m.text,
                    m.importance,
                    m.source,
                    m.origin,
                    int(m.locked),
                    wall,
                    wall,
                ),
            )

    def _on_degraded(self, detail: str) -> None:
        self._health = Health(
            self.name,
            HealthState.DEGRADED,
            f"memory is in RAM only (not persisted): {detail}"[:300],
            self._clock.now(),
        )
        if self._bus is not None:
            self._bus.publish(HealthChanged(health=self._health, character=self.character))

    def _written(self, item: MemoryItem) -> None:
        if self._bus is not None and item.id is not None:
            self._bus.publish(
                MemoryWritten(
                    character=self.character,
                    memory_id=item.id,
                    kind=item.kind,
                    status=item.status,
                )
            )

    def _validate(self, item: MemoryItem, text: str, replace_slot: int | None) -> None:
        if not text:
            raise ValueError("empty memory text")
        limit = self.slot_max_chars if item.kind == "core" else self.max_text_chars
        if len(text) > limit:
            raise ValueError(f"{item.kind} memory longer than {limit} characters")
        if not 1 <= item.importance <= 5:
            raise ValueError("importance must be 1..5")
        if item.status == "deleted":
            raise ValueError("a new memory cannot be stored as deleted")
        if replace_slot is not None:
            if item.kind != "core":
                raise ValueError("replace_slot only applies to core memories")
            if not 1 <= replace_slot <= self.core_slots:
                raise ValueError(f"replace_slot must be 1..{self.core_slots}")
        if item.kind == "viewer" and not (item.platform and item.user_id):
            raise ValueError("a viewer memory needs platform and user_id")

    def _require_session(self) -> int:
        if self._session_id is None:
            raise RuntimeError("no active session; call start_session() or resume_session()")
        return self._session_id

    @staticmethod
    def _get(conn: sqlite3.Connection, memory_id: int) -> MemoryItem:
        row = conn.execute(f"SELECT {_ITEM_COLS} FROM memory WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(f"no memory {memory_id}")
        return _item(row)

    @staticmethod
    def _active_slots(conn: sqlite3.Connection) -> dict[int, MemoryItem]:
        rows = conn.execute(
            f"SELECT {_ITEM_COLS} FROM memory WHERE kind = 'core' AND status = 'active' "
            "AND slot IS NOT NULL"
        ).fetchall()
        return {int(r[3]): _item(r) for r in rows}

    def _free_slot(self, used: Mapping[int, MemoryItem]) -> int | None:
        return next((s for s in range(1, self.core_slots + 1) if s not in used), None)

    @staticmethod
    def _delete_row(conn: sqlite3.Connection, memory_id: int | None, wall: float) -> None:
        conn.execute(
            "UPDATE memory SET status = 'deleted', slot = NULL, epoch_seen = NULL, "
            "updated_at = ? WHERE id = ?",
            (wall, memory_id),
        )

    @staticmethod
    def _opted_out(conn: sqlite3.Connection, platform: str | None, user_id: str | None) -> bool:
        row = conn.execute(
            "SELECT opt_out FROM viewer WHERE platform = ? AND user_id = ?", (platform, user_id)
        ).fetchone()
        return bool(row is not None and row[0])

    def _refresh_core(self, conn: sqlite3.Connection) -> tuple[MemoryItem, ...]:
        rows = conn.execute(
            f"SELECT {_ITEM_COLS} FROM memory WHERE kind = 'core' AND status = 'active' "
            "ORDER BY slot, id"
        ).fetchall()
        self._core_cache = tuple(_item(r) for r in rows)
        return self._core_cache

    def _terms(self, q: str) -> list[str]:
        """Search terms of 3+ characters: the whole query (if short) and its words."""
        terms: list[str] = []

        def add(term: str) -> None:
            t = term.strip(_TERM_STRIP)
            if len(t) >= 3 and t.casefold() not in _STOPWORDS and t not in terms:
                terms.append(t)

        if len(q) <= 48:
            add(q)
        for chunk in q.split(" "):
            if len(chunk) > 3 and _has_thai(chunk):
                for word in self._words(chunk):
                    add(word)
            else:
                add(chunk)
        return terms[:_MAX_TERMS]

    def _words(self, text: str) -> Iterable[str]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            from aivtube.text.thai import newmm

            tokenizer = newmm
        try:
            return tokenizer(text)
        except Exception:
            log.debug("Thai word split failed; searching the whole chunk", exc_info=True)
            return [text]

    @staticmethod
    def _kind_filter(kinds: Sequence[str] | None) -> tuple[str, list[Any]]:
        if kinds is None:
            return "", []
        return f" AND m.kind IN ({', '.join('?' for _ in kinds)})", list(kinds)

    def _search_fts(
        self, conn: sqlite3.Connection, terms: Sequence[str], k: int, kinds: Sequence[str] | None
    ) -> list[MemoryItem]:
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        where, params = self._kind_filter(kinds)
        cols = ", ".join(f"m.{c.strip()}" for c in _ITEM_COLS.split(","))
        rows = conn.execute(
            f"SELECT {cols} FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
            f"WHERE memory_fts MATCH ? AND m.status = 'active'{where} "
            "ORDER BY bm25(memory_fts), m.importance DESC, m.id DESC LIMIT ?",
            (match, *params, k),
        ).fetchall()
        return [_item(r) for r in rows]

    def _search_like(
        self, conn: sqlite3.Connection, terms: Sequence[str], k: int, kinds: Sequence[str] | None
    ) -> list[MemoryItem]:
        where, params = self._kind_filter(kinds)
        score = " + ".join(
            "(CASE WHEN m.text LIKE ? ESCAPE '\\' OR COALESCE(m.subject, '') LIKE ? ESCAPE '\\' "
            "THEN 1 ELSE 0 END)"
            for _ in terms
        )
        like_params = [p for t in terms for p in (_like(t), _like(t))]
        cols = ", ".join(f"m.{c.strip()}" for c in _ITEM_COLS.split(","))
        rows = conn.execute(
            f"SELECT * FROM (SELECT {cols}, ({score}) AS hits FROM memory m "
            f"WHERE m.status = 'active'{where}) WHERE hits > 0 "
            "ORDER BY hits DESC, importance DESC, id DESC LIMIT ?",
            (*like_params, *params, k),
        ).fetchall()
        return [_item(r) for r in rows]


def _lastrowid(cur: sqlite3.Cursor) -> int:
    rowid = cur.lastrowid
    if rowid is None:  # pragma: no cover - INSERT always sets it
        raise RuntimeError("INSERT did not return a row id")
    return int(rowid)


def _split_sql(script: str) -> list[str]:
    """Split a trusted script into complete statements (``sqlite3.complete_statement``)."""
    out: list[str] = []
    buf = ""
    for line in script.splitlines(keepends=True):
        if not buf and line.lstrip().startswith("--"):
            continue
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out
