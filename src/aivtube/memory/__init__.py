"""Memory: the per-character SQLite store, the shared ops database and backups (§6).

``SqliteMemory`` implements ``MemoryStore``; ``OpsDb`` is ``data/ops.db`` (turn traces,
moderation log, tool and operator audit, jobs). Standard library only.
"""

from aivtube.memory._db import fts5_trigram_available
from aivtube.memory.backup import backup_database_file, backup_files, list_backups
from aivtube.memory.ops_db import OpsDb
from aivtube.memory.policy import find_pii, initial_status, may_forget, origin_kind
from aivtube.memory.store import MemoryLocked, SqliteMemory, ViewerOptedOut, render_context

__all__ = [
    "MemoryLocked",
    "OpsDb",
    "SqliteMemory",
    "ViewerOptedOut",
    "backup_database_file",
    "backup_files",
    "find_pii",
    "fts5_trigram_available",
    "initial_status",
    "list_backups",
    "may_forget",
    "origin_kind",
    "render_context",
]
