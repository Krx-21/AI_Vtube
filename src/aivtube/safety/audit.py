"""Moderation audit: every DROP, BLOCK and REVIEW becomes a ``moderation_log`` row (§7).

Rows hold PII-masked text plus the sha256 of the original, never the raw text. The gate runs
synchronously on the event loop, so :meth:`ModerationAudit.record` only queues the row; one
drain task (started through ``TaskSupervisor.track``) writes queued rows in order through the
injected sink, each write under a deadline. The ops database belongs to the memory package:
wire it with ``ops_sink(ops_db)``, or pass any ``async (ModerationRecord) -> None`` callable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from aivtube.contracts.infra import Clock, TaskSupervisor
from aivtube.contracts.safety import FilterResult
from aivtube.infra.clock import deadline
from aivtube.safety.pii import mask_pii

__all__ = [
    "ModerationAudit",
    "ModerationRecord",
    "ModerationSink",
    "SupportsLogModeration",
    "ops_sink",
    "sha256_text",
]

log = logging.getLogger("aivtube.safety.audit")


@dataclass(frozen=True, slots=True)
class ModerationRecord:
    """One ``moderation_log`` row (ARCHITECTURE.md §6 schema, without ``id``)."""

    ts: float  # wall-clock seconds (storage only)
    character: str
    direction: str
    source: str  # chat platform, "llm" for speech, or tool | memory | game
    tier: str
    category: str | None
    rule: str | None
    verdict: str
    text_masked: str
    text_sha256: str
    author: str | None = None
    turn_id: str | None = None

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


ModerationSink: TypeAlias = Callable[[ModerationRecord], Awaitable[None]]


class SupportsLogModeration(Protocol):
    """The part of ``OpsDb`` (memory package) the audit needs."""

    async def log_moderation(self, **row: Any) -> None: ...


def ops_sink(ops: SupportsLogModeration) -> ModerationSink:
    """Adapt an ``OpsDb``-like object to a :data:`ModerationSink`."""

    async def sink(record: ModerationRecord) -> None:
        await ops.log_moderation(**record.as_row())

    return sink


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ModerationAudit:
    """Queues moderation rows and writes them through ``sink`` in the background.

    ``recent`` keeps the last ``keep`` records for the panel and tests. Without a running
    event loop rows stay queued until the next ``record()`` on the loop or :meth:`flush`.
    When more than ``max_pending`` rows are waiting, the oldest are dropped (``dropped``).
    """

    def __init__(
        self,
        sink: ModerationSink | None,
        *,
        clock: Clock,
        tasks: TaskSupervisor | None = None,
        mask_text: str = "[ลิงก์]",
        timeout_s: float = 2.0,
        max_pending: int = 256,
        keep: int = 200,
        max_chars: int = 500,
    ) -> None:
        if sink is not None and tasks is None:
            raise ValueError("ModerationAudit needs a TaskSupervisor to write through a sink")
        self._sink = sink
        self._clock = clock
        self._tasks = tasks
        self._mask_text = mask_text
        self._timeout_s = timeout_s
        self._max_pending = max(1, max_pending)
        self._max_chars = max_chars
        self._pending: deque[ModerationRecord] = deque()
        self._drainer: asyncio.Task[None] | None = None
        self.recent: deque[ModerationRecord] = deque(maxlen=keep)
        self.written = 0
        self.errors = 0
        self.dropped = 0

    @property
    def pending(self) -> int:
        return len(self._pending)

    def record(
        self,
        *,
        character: str,
        direction: str,
        source: str,
        result: FilterResult,
        text: str,
        author: str | None = None,
        turn_id: str | None = None,
    ) -> ModerationRecord:
        """Build a masked row for ``text`` and queue it (never blocks, never raises)."""
        rec = ModerationRecord(
            ts=self._clock.wall(),
            character=character,
            direction=direction,
            source=source,
            tier=result.tier,
            category=result.category,
            rule=result.rule,
            verdict=result.verdict.value,
            text_masked=mask_pii(text, mask_text=self._mask_text)[: self._max_chars],
            text_sha256=sha256_text(text),
            author=author,
            turn_id=turn_id,
        )
        self.recent.append(rec)
        if self._sink is not None:
            if len(self._pending) >= self._max_pending:
                self._pending.popleft()
                self.dropped += 1
            self._pending.append(rec)
            self._kick()
        return rec

    def _kick(self) -> None:
        if self._drainer is not None and not self._drainer.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet: written by the next record() on the loop or flush()
        assert self._tasks is not None
        self._drainer = self._tasks.track(self._drain(), name="safety.audit")

    async def _drain(self) -> None:
        sink = self._sink
        if sink is None:
            return
        while self._pending:
            rec = self._pending[0]
            try:
                async with deadline(
                    self._timeout_s, what="moderation_log write", clock=self._clock
                ):
                    await sink(rec)
            except Exception:
                self.errors += 1
                log.warning("moderation_log write failed; row skipped", exc_info=True)
            else:
                self.written += 1
            if self._pending and self._pending[0] is rec:
                self._pending.popleft()

    async def flush(self) -> None:
        """Wait until every queued row has been written (or failed)."""
        if self._sink is None:
            return
        while self._pending:
            if self._drainer is None or self._drainer.done():
                assert self._tasks is not None
                self._drainer = self._tasks.track(self._drain(), name="safety.audit")
            drainer = self._drainer
            try:
                await asyncio.shield(drainer)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if not drainer.cancelled() or (task is not None and task.cancelling()):
                    raise  # we are being cancelled ourselves
                # the drainer was cancelled by someone else (shutdown): start a new one

    async def aclose(self, timeout_s: float = 2.0) -> None:
        """Flush for at most ``timeout_s`` at shutdown; rows still queued are logged as lost."""
        try:
            async with deadline(timeout_s, what="moderation_log flush", clock=self._clock):
                await self.flush()
        except TimeoutError:
            log.warning("moderation_log: %d rows not written at shutdown", len(self._pending))
