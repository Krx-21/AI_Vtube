"""``OpsDb``: ``data/ops.db``, shared by every character (ARCHITECTURE.md §6, §2.10).

The sink for turn traces, the moderation log (PII-masked text + sha256, 30-day retention),
the tool audit, the operator audit and the background job queue. Same single-worker SQLite
plumbing as the memory store, with the same busy/corrupt fallback to RAM (health DEGRADED).

Audit writes (``add_trace``, ``log_*``) never raise: an audit failure is logged and the turn
goes on. Row keyword arguments are matched to the table's columns; unknown keys are ignored
with a debug log, mappings and sequences are stored as JSON, and ``ts`` defaults to
``Clock.wall()``. Job times (``run_at``, ``now``) are wall-clock seconds because jobs outlive
the process (an episode job waits for the next start when the LLM is down).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from aivtube.contracts.events import HealthChanged
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.types import Health, HealthState
from aivtube.infra.clock import SystemClock
from aivtube.memory._db import SqliteWorker, TxnMode, apply_migrations, load_migrations
from aivtube.memory.backup import backup_connection

__all__ = ["AuditTable", "OpsDb"]

log = logging.getLogger("aivtube.memory.ops")
_T = TypeVar("_T")

AuditTable = Literal["tool_audit", "op_audit", "moderation_log"]

_COLUMNS: dict[str, tuple[str, ...]] = {
    "turn_trace": (
        "turn_id",
        "character",
        "session_id",
        "kind",
        "provider",
        "tts_backend",
        "tts_identity",
        "stages",
        "ttfa_ms",
        "tokens_out",
        "tok_s",
        "prompt_n",
        "cache_n",
        "speculative",
        "opener",
        "outcome",
    ),
    "moderation_log": (
        "ts",
        "character",
        "direction",
        "source",
        "tier",
        "category",
        "rule",
        "verdict",
        "text_masked",
        "text_sha256",
        "author",
        "turn_id",
    ),
    "tool_audit": (
        "ts",
        "character",
        "turn_id",
        "tool",
        "args",
        "verdict",
        "result",
        "approved_by",
        "dry_run",
    ),
    "op_audit": ("ts", "operator", "command", "args", "result", "latency_ms"),
}
MODERATION_RETENTION_S = 30 * 86400.0


def _cell(value: Any) -> Any:
    """A value SQLite can store: JSON for containers, int for bools, str for enums/others."""
    if value is None or isinstance(value, int | float | str | bytes):
        return int(value) if isinstance(value, bool) else value
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        items = dict(value) if isinstance(value, Mapping) else list(value)
        return json.dumps(items, ensure_ascii=False, default=str, sort_keys=True)
    return str(value)


def _row_dict(cur: sqlite3.Cursor, row: Sequence[Any]) -> dict[str, Any]:
    return {d[0]: v for d, v in zip(cur.description, row, strict=True)}


class OpsDb:
    """``data/ops.db`` (see the module docstring). ``path=None`` keeps it in RAM (tests)."""

    def __init__(
        self,
        path: Path | None,
        *,
        executor: concurrent.futures.Executor | None = None,
        clock: Clock | None = None,
        bus: EventBus | None = None,
        busy_timeout_ms: int = 5000,
        max_job_attempts: int = 5,
        job_backoff_s: tuple[float, float] = (60.0, 3600.0),
    ) -> None:
        self.name = "ops_db"
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._bus = bus
        self.max_job_attempts = max(1, max_job_attempts)
        self.job_backoff_s = job_backoff_s
        self._db = SqliteWorker(
            path,
            name=self.name,
            migrate=self._migrate,
            busy_timeout_ms=busy_timeout_ms,
            executor=executor,
        )
        self._db.on_degraded = self._on_degraded
        self._health = Health(self.name, HealthState.STARTING, "", self._clock.now())

    @classmethod
    def from_config(
        cls, app_cfg: Any, *, clock: Clock | None = None, bus: EventBus | None = None
    ) -> OpsDb:
        """``data/ops.db`` (``[memory] ops_db``) resolved against the project root."""
        return cls(app_cfg.resolve_path(app_cfg.memory.ops_db), clock=clock, bus=bus)

    # --- lifecycle ----------------------------------------------------------------------------
    async def start(self) -> None:
        await self._run(lambda conn: None, "raw")

    async def aclose(self) -> None:
        await self._db.aclose()

    def health(self) -> Health:
        return self._health

    @property
    def degraded(self) -> bool:
        return self._db.degraded is not None

    # --- audit sinks (never raise) --------------------------------------------------------------
    async def add_trace(self, trace: Mapping[str, Any]) -> None:
        """Store a finished ``TurnTraceRecorder`` trace (``stages`` keeps the ms offsets)."""
        row = {k: trace.get(k) for k in _COLUMNS["turn_trace"] if k in trace}
        if not row.get("turn_id"):
            log.warning("trace without turn_id ignored")
            return
        stages = trace.get("stages_ms", trace.get("stages", {}))
        row["stages"] = stages if isinstance(stages, Mapping) else {}
        await self._insert("turn_trace", row, replace=True)

    def trace_sink(self, tasks: TaskSupervisor) -> Callable[[Mapping[str, Any]], None]:
        """A ``TurnTraceRecorder(on_finish=...)`` callback: stores each finished trace in the
        background through ``tasks.track`` (call it on the loop thread)."""

        def sink(trace: Mapping[str, Any]) -> None:
            tasks.track(self.add_trace(dict(trace)), name="ops:trace")

        return sink

    async def log_moderation(self, **row: Any) -> None:
        """One moderation verdict. Pass ``text_masked`` (PII-masked) and ``text_sha256``;
        blocked text itself must never be stored."""
        await self._insert("moderation_log", row)

    async def log_tool(self, **row: Any) -> None:
        await self._insert("tool_audit", row)

    async def log_op(self, **row: Any) -> None:
        await self._insert("op_audit", row)

    # --- reads for the panel and reports ------------------------------------------------------
    async def recent_traces(self, n: int = 20) -> list[Mapping[str, Any]]:
        """The last ``n`` traces, oldest first, with ``stages`` decoded."""

        def op(conn: sqlite3.Connection) -> list[Mapping[str, Any]]:
            cur = conn.execute("SELECT * FROM turn_trace ORDER BY rowid DESC LIMIT ?", (max(0, n),))
            out: list[Mapping[str, Any]] = []
            for row in reversed(cur.fetchall()):
                d = _row_dict(cur, row)
                try:
                    d["stages"] = json.loads(d["stages"])
                except (TypeError, ValueError):
                    d["stages"] = {}
                out.append(d)
            return out

        return await self._run(op, "read")

    async def audit_rows(
        self, table: AuditTable, *, since_id: int = 0, limit: int = 100
    ) -> list[Mapping[str, Any]]:
        """Rows of an audit table with ``id > since_id``, oldest first (panel ``/api/audit``)."""
        if table not in ("tool_audit", "op_audit", "moderation_log"):
            raise ValueError(f"unknown audit table {table!r}")

        def op(conn: sqlite3.Connection) -> list[Mapping[str, Any]]:
            cur = conn.execute(
                f"SELECT * FROM {table} WHERE id > ? ORDER BY id LIMIT ?", (since_id, limit)
            )
            return [_row_dict(cur, r) for r in cur.fetchall()]

        return await self._run(op, "read")

    async def purge_moderation(self, max_age_s: float = MODERATION_RETENTION_S) -> int:
        """Delete moderation rows older than ``max_age_s`` (30 days by default)."""
        cutoff = self._clock.wall() - max_age_s
        return await self._run(
            lambda conn: int(
                conn.execute("DELETE FROM moderation_log WHERE ts < ?", (cutoff,)).rowcount
            ),
            "write",
        )

    # --- jobs ---------------------------------------------------------------------------------
    async def enqueue_job(
        self, kind: str, payload: Mapping[str, Any], *, run_at: float | None = None
    ) -> int:
        """Queue a job to run at wall time ``run_at`` (default: now); returns its id."""
        when = self._clock.wall() if run_at is None else run_at
        body = json.dumps(dict(payload), ensure_ascii=False, default=str)

        def op(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO job(kind, state, payload, attempts, next_run_at) "
                "VALUES(?, 'pending', ?, 0, ?)",
                (kind, body, when),
            )
            return int(cur.lastrowid or 0)

        return await self._run(op, "write")

    async def due_jobs(self, now: float) -> list[Mapping[str, Any]]:
        """Claim every pending job due at wall time ``now`` (they become ``running``).

        Each item has ``id``, ``kind``, ``payload`` (decoded) and ``attempts``. Report the
        outcome with ``finish_job``; jobs still ``running`` at the next start are re-queued.
        """

        def op(conn: sqlite3.Connection) -> list[Mapping[str, Any]]:
            rows = conn.execute(
                "SELECT id, kind, payload, attempts FROM job WHERE state = 'pending' "
                "AND (next_run_at IS NULL OR next_run_at <= ?) ORDER BY next_run_at, id",
                (now,),
            ).fetchall()
            out: list[Mapping[str, Any]] = []
            for job_id, kind, payload, attempts in rows:
                conn.execute("UPDATE job SET state = 'running' WHERE id = ?", (job_id,))
                try:
                    body = json.loads(payload) if payload else {}
                except ValueError:
                    body = {}
                out.append({"id": job_id, "kind": kind, "payload": body, "attempts": attempts})
            return out

        return await self._run(op, "write")

    async def finish_job(self, job_id: int, *, ok: bool, error: str | None = None) -> None:
        """``ok`` marks the job done. A failure is retried with exponential backoff until
        ``max_job_attempts``, then the job is marked ``failed``."""
        wall = self._clock.wall()
        low, high = self.job_backoff_s

        def op(conn: sqlite3.Connection) -> None:
            row = conn.execute("SELECT attempts FROM job WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"no job {job_id}")
            if ok:
                conn.execute(
                    "UPDATE job SET state = 'done', last_error = NULL WHERE id = ?", (job_id,)
                )
                return
            attempts = int(row[0]) + 1
            state = "failed" if attempts >= self.max_job_attempts else "pending"
            delay = min(high, low * (2 ** (attempts - 1)))
            conn.execute(
                "UPDATE job SET state = ?, attempts = ?, next_run_at = ?, last_error = ? "
                "WHERE id = ?",
                (state, attempts, wall + delay, (error or "failed")[:500], job_id),
            )

        await self._run(op, "write")

    async def jobs(self, *, state: str | None = None) -> list[Mapping[str, Any]]:
        """Every job (optionally in one ``state``), oldest first, payload decoded."""

        def op(conn: sqlite3.Connection) -> list[Mapping[str, Any]]:
            cur = conn.execute(
                "SELECT * FROM job WHERE (? IS NULL OR state = ?) ORDER BY id", (state, state)
            )
            out: list[Mapping[str, Any]] = []
            for row in cur.fetchall():
                d = _row_dict(cur, row)
                try:
                    d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
                except ValueError:
                    d["payload"] = {}
                out.append(d)
            return out

        return await self._run(op, "read")

    async def backup(self, dest_dir: Path, keep: int = 14) -> Path:
        wall = self._clock.wall()
        return await self._run(
            lambda conn: backup_connection(conn, dest_dir, stem="ops", wall=wall, keep=keep),
            "raw",
        )

    # --- internals ------------------------------------------------------------------------------
    async def _run(self, fn: Callable[[sqlite3.Connection], _T], mode: TxnMode) -> _T:
        result = await self._db.run(fn, mode=mode)
        if self._health.state is HealthState.STARTING:
            self._health = Health(self.name, HealthState.OK, "", self._clock.now())
        return result

    async def _insert(self, table: str, row: Mapping[str, Any], *, replace: bool = False) -> None:
        columns = _COLUMNS[table]
        unknown = sorted(set(row) - set(columns))
        if unknown:
            log.debug("%s: ignoring unknown columns %s", table, unknown)
        values = {k: _cell(v) for k, v in row.items() if k in columns}
        if "ts" in columns and values.get("ts") is None:
            values["ts"] = self._clock.wall()
        names = list(values)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        sql = f"{verb} INTO {table}({', '.join(names)}) VALUES({', '.join('?' for _ in names)})"
        params = tuple(values[n] for n in names)
        try:
            await self._run(lambda conn: conn.execute(sql, params), "write")
        except RuntimeError:
            log.warning("%s write dropped: ops database is closed", table)
        except Exception:
            log.exception("%s write failed", table)

    def _migrate(self, conn: sqlite3.Connection, is_file: bool) -> None:
        apply_migrations(conn, load_migrations("ops"))
        # Jobs claimed by a process that died are re-queued.
        conn.execute("UPDATE job SET state = 'pending' WHERE state = 'running'")

    def _on_degraded(self, detail: str) -> None:
        self._health = Health(
            self.name,
            HealthState.DEGRADED,
            f"ops.db is in RAM only (not persisted): {detail}"[:300],
            self._clock.now(),
        )
        if self._bus is not None:
            self._bus.publish(HealthChanged(health=self._health))
