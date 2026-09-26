"""Single-writer SQLite plumbing shared by the memory store and the ops database (§6, §2.8).

One connection per database, used only from one worker (a 1-thread executor), so the event
loop never touches SQLite. Files run in WAL mode with ``synchronous=NORMAL``,
``busy_timeout=5000`` and foreign keys on. Numbered migrations (``migrations/<db>_NNNN_*.sql``)
are tracked by ``PRAGMA user_version`` and each one is applied atomically.

Failure policy (§2.8 "SQLite busy or corrupt"): a busy/locked/I-O error is retried once, and a
corrupt file is not retried. If the database still cannot be used, the worker switches to a
fresh in-memory database with the same schema, sets ``degraded`` and reports it once through
``on_degraded``. The owner seeds that fallback (``on_fallback``) so the stream can go on.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal, TypeVar

__all__ = [
    "DatabaseCorrupt",
    "Migration",
    "SchemaTooNew",
    "SqliteWorker",
    "TxnMode",
    "apply_migrations",
    "classify_error",
    "fts5_trigram_available",
    "load_migrations",
    "read_sql",
]

T = TypeVar("T")
TxnMode = Literal["read", "write", "raw"]
log = logging.getLogger("aivtube.memory.db")

_MIGRATION_RE = re.compile(r"^(?P<db>[a-z]+)_(?P<num>\d{4})_[a-z0-9_]+\.sql$")

# Primary SQLite result codes (https://sqlite.org/rescode.html).
_BUSY_CODES = frozenset(
    {5, 6, 8, 10, 13, 14, 15}
)  # BUSY LOCKED READONLY IOERR FULL CANTOPEN PROTOCOL
_CORRUPT_CODES = frozenset({11, 26})  # CORRUPT NOTADB


class DatabaseCorrupt(sqlite3.DatabaseError):
    """``PRAGMA quick_check`` failed."""


class SchemaTooNew(sqlite3.DatabaseError):
    """The file was written by a newer version of the app (``user_version`` is ahead)."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


def read_sql(name: str) -> str:
    """The text of ``migrations/<name>`` (packaged with the module)."""
    return (resources.files("aivtube.memory") / "migrations" / name).read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def load_migrations(db: str) -> tuple[Migration, ...]:
    """The numbered migrations for ``db`` (``memory`` or ``ops``), in order, without gaps."""
    root = resources.files("aivtube.memory") / "migrations"
    found: list[Migration] = []
    for entry in root.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m is not None and m["db"] == db:
            found.append(Migration(int(m["num"]), entry.name, entry.read_text(encoding="utf-8")))
    found.sort(key=lambda mig: mig.version)
    for expected, mig in enumerate(found, start=1):
        if mig.version != expected:
            raise RuntimeError(f"{db} migrations are not numbered 1..n: {mig.name}")
    return tuple(found)


def apply_migrations(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> int:
    """Apply every migration newer than ``PRAGMA user_version``; returns the final version.

    Each migration runs in its own ``BEGIN IMMEDIATE`` transaction together with the
    ``user_version`` bump, so a failing script leaves the previous version intact. Running it
    again is a no-op. ``conn`` must be in autocommit mode (``isolation_level=None``).
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    latest = migrations[-1].version if migrations else 0
    if current > latest:
        raise SchemaTooNew(f"schema version {current} is newer than this app ({latest})")
    for mig in migrations:
        if mig.version <= current:
            continue
        try:
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{mig.sql}\n;PRAGMA user_version = {mig.version:d};\nCOMMIT;"
            )
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            raise
        log.info("applied migration %s", mig.name)
        current = mig.version
    return current


@lru_cache(maxsize=1)
def fts5_trigram_available() -> bool:
    """True when this SQLite has FTS5 with the ``trigram`` tokenizer (SQLite >= 3.34)."""
    try:
        conn = sqlite3.connect(":memory:")
    except sqlite3.Error:  # pragma: no cover - sqlite3 always opens :memory:
        return False
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x, tokenize='trigram')")
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return True


def classify_error(exc: BaseException) -> Literal["busy", "corrupt"] | None:
    """``busy`` (worth one retry), ``corrupt`` (switch now), or ``None`` (a bug: re-raise)."""
    if isinstance(exc, DatabaseCorrupt | SchemaTooNew):
        return "corrupt"
    if not isinstance(exc, sqlite3.DatabaseError) or isinstance(
        exc, sqlite3.IntegrityError | sqlite3.ProgrammingError | sqlite3.DataError
    ):
        return None
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        primary = code & 0xFF
        if primary in _CORRUPT_CODES:
            return "corrupt"
        if primary in _BUSY_CODES:
            return "busy"
        return None
    text = str(exc).lower()
    if "malformed" in text or "not a database" in text:
        return "corrupt"
    if any(w in text for w in ("locked", "busy", "disk i/o", "readonly", "unable to open")):
        return "busy"
    return None


class SqliteWorker:
    """One SQLite connection driven from one worker thread.

    ``path=None`` opens an in-memory database (not degraded). ``migrate(conn, is_file)`` runs
    in the worker thread whenever a connection is opened (file or fallback). ``on_fallback``
    seeds a fresh in-memory fallback; ``on_degraded(detail)`` is called once, on the loop
    thread, after the switch. A caller-supplied ``executor`` must not run two jobs at once
    for this worker (a lock serialises them anyway).
    """

    def __init__(
        self,
        path: Path | None,
        *,
        name: str,
        migrate: Callable[[sqlite3.Connection, bool], None],
        busy_timeout_ms: int = 5000,
        quick_check: bool = True,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self._migrate = migrate
        self._busy_timeout_ms = max(0, int(busy_timeout_ms))
        self._quick_check = quick_check
        self._own_executor = executor is None
        self._executor: concurrent.futures.Executor = executor or (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"db-{name}")
        )
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._closing = False  # loop side: refuse new jobs
        self._closed = False  # worker side: the connection is gone
        self._report_pending = False
        self.degraded: str | None = None
        self.on_fallback: Callable[[sqlite3.Connection], None] | None = None
        self.on_degraded: Callable[[str], None] | None = None

    @property
    def closed(self) -> bool:
        return self._closing

    # --- public (loop thread) -----------------------------------------------------------------
    async def run(self, fn: Callable[[sqlite3.Connection], T], *, mode: TxnMode) -> T:
        """Run ``fn(conn)`` on the worker thread: ``read`` in a deferred transaction, ``write``
        in ``BEGIN IMMEDIATE``, ``raw`` without one (for backups). Cancelling the caller does
        not interrupt a job that already started; its transaction still completes."""
        if self._closing:
            raise RuntimeError(f"{self.name} database is closed")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._run, fn, mode)
        finally:
            self._report()

    async def open(self) -> None:
        await self.run(lambda conn: None, mode="raw")

    async def aclose(self) -> None:
        """Close the connection after every queued job, then release the executor."""
        if self._closing:
            return
        self._closing = True
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._close)
        finally:
            if self._own_executor:
                self._executor.shutdown(wait=False)

    # --- worker thread ------------------------------------------------------------------------
    def _run(self, fn: Callable[[sqlite3.Connection], T], mode: TxnMode) -> T:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name} database is closed")
            conn = self._ensure_open()
            try:
                return self._transaction(conn, fn, mode)
            except Exception as exc:
                kind = classify_error(exc)
                if kind is None or self.degraded is not None:
                    raise
                if kind == "busy":
                    log.warning("%s database busy (%s); retrying once", self.name, exc)
                    try:
                        return self._transaction(conn, fn, mode)
                    except Exception as again:
                        if classify_error(again) is None:
                            raise
                        exc = again
                conn = self._switch_to_fallback(f"{kind}: {exc}")
                return self._transaction(conn, fn, mode)

    @staticmethod
    def _transaction(
        conn: sqlite3.Connection, fn: Callable[[sqlite3.Connection], T], mode: TxnMode
    ) -> T:
        if mode == "raw":
            return fn(conn)
        conn.execute("BEGIN IMMEDIATE" if mode == "write" else "BEGIN")
        try:
            result = fn(conn)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            raise
        return result

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self.path is None:
            self._conn = self._open_memory()
            return self._conn
        try:
            self._conn = self._open_file(self.path)
        except Exception as exc:
            kind = classify_error(exc)
            if kind is None:
                raise
            if kind == "busy":
                log.warning("%s database busy at open (%s); retrying once", self.name, exc)
                try:
                    self._conn = self._open_file(self.path)
                    return self._conn
                except Exception as again:
                    if classify_error(again) is None:
                        raise
                    exc = again
            self._conn = self._switch_to_fallback(f"open failed ({kind}): {exc}")
        return self._conn

    def _open_file(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path),
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms:d}")
            mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode != "wal":
                log.warning("%s database: journal_mode is %s, not WAL", self.name, mode)
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            if self._quick_check:
                verdict = str(conn.execute("PRAGMA quick_check").fetchone()[0])
                if verdict != "ok":
                    raise DatabaseCorrupt(f"quick_check: {verdict[:200]}")
            self._migrate(conn, True)
        except BaseException:
            conn.close()
            raise
        return conn

    def _open_memory(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            self._migrate(conn, False)
        except BaseException:
            conn.close()
            raise
        return conn

    def _switch_to_fallback(self, detail: str) -> sqlite3.Connection:
        log.error(
            "%s database unusable (%s); switching to an in-memory store (not persisted)",
            self.name,
            detail,
        )
        if self._conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None
        conn = self._open_memory()
        seed = self.on_fallback
        if seed is not None:
            try:
                self._transaction(conn, seed, "write")
            except Exception:
                log.exception("%s: seeding the in-memory fallback failed", self.name)
        self._conn = conn
        self.degraded = detail
        self._report_pending = True
        return conn

    def _close(self) -> None:
        with self._lock:
            self._closed = True
            if self._conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.close()
                self._conn = None

    def _report(self) -> None:
        if not self._report_pending:
            return
        self._report_pending = False
        callback = self.on_degraded
        if callback is not None and self.degraded is not None:
            try:
                callback(self.degraded)
            except Exception:
                log.exception("%s: on_degraded callback failed", self.name)
