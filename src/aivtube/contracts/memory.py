"""Memory protocols: turns, long-term items and the store (ARCHITECTURE.md §3.9, §6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

__all__ = [
    "MemKind",
    "MemSource",
    "MemStatus",
    "MemoryItem",
    "MemoryStore",
    "PrefixMemory",
    "SlotsFull",
    "Turn",
]

MemKind: TypeAlias = Literal["core", "fact", "viewer", "episode"]
MemStatus: TypeAlias = Literal["active", "quarantined", "deleted"]
MemSource: TypeAlias = Literal["model", "operator", "consolidation", "import"]


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: int | None
    kind: MemKind
    text: str
    slot: int | None = None
    subject: str | None = None
    platform: str | None = None
    user_id: str | None = None
    importance: int = 3
    source: MemSource = "model"
    origin: str = ""
    status: MemStatus = "active"
    pinned: bool = False
    locked: bool = False


@dataclass(frozen=True, slots=True)
class Turn:
    """One history line (a row of the ``turn`` table)."""

    role: Literal["user", "assistant", "tool", "note"]
    text: str
    source: str
    speaker: str | None = None
    heard_text: str | None = None
    interrupted: bool = False
    filtered: bool = False
    provider: str | None = None
    turn_ref: str = ""
    tool_calls: str | None = None
    provider_extra: str | None = None
    ts: float = 0.0
    id: int | None = None


@dataclass(frozen=True, slots=True)
class PrefixMemory:
    """Everything rendered into the cached prompt prefix for one epoch."""

    core: tuple[MemoryItem, ...]
    episodes: tuple[str, ...]
    rolling_summary: str
    epoch: int
    digest: str


class SlotsFull(Exception):
    """``remember`` found every long-term slot taken; ``slots`` lists the current items."""

    slots: list[MemoryItem]

    def __init__(
        self, slots: Sequence[MemoryItem], msg: str = "long-term memory slots are full"
    ) -> None:
        super().__init__(msg)
        self.slots = list(slots)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.slots, str(self)))


@runtime_checkable
class MemoryStore(Protocol):
    async def start_session(self, title: str | None = None) -> int: ...

    async def resume_session(self, max_age_s: float) -> int | None: ...

    async def end_session(self, summary: str | None = None) -> None: ...

    async def append_turn(self, turn: Turn) -> int: ...

    async def recent_turns(self, epoch: int) -> list[Turn]: ...

    async def prefix_block(self) -> PrefixMemory: ...

    async def new_epoch(self, rolling_summary: str, upto_turn_id: int, prefix_hash: str) -> int: ...

    async def remember(self, item: MemoryItem, *, replace_slot: int | None = None) -> MemoryItem:
        """Store ``item``; may raise ``SlotsFull``."""
        ...

    async def forget(self, memory_id: int, *, by: str, reason: str) -> None:
        """Delete an item; locked items refuse."""
        ...

    async def set_status(self, memory_id: int, status: MemStatus, *, by: str) -> None: ...

    async def list_memories(
        self, *, kind: MemKind | None = None, status: MemStatus | None = None
    ) -> list[MemoryItem]: ...

    async def pending_since_epoch(self) -> list[MemoryItem]: ...

    async def search(self, query: str, k: int = 3) -> list[MemoryItem]:
        """FTS5 trigram search for queries of 3+ characters, otherwise LIKE."""
        ...

    async def viewer_facts(
        self, users: Sequence[tuple[str, str]], limit: int = 6
    ) -> list[MemoryItem]: ...

    async def upsert_viewer(self, platform: str, user_id: str, name: str) -> None: ...

    async def backup(self, dest_dir: Path, keep: int = 14) -> Path: ...
