"""The MemoryStore contract suite against the real ``SqliteMemory`` (§3.9, §10)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts.memory import MemoryStore
from aivtube.infra import SystemClock
from aivtube.memory import SqliteMemory, fts5_trigram_available
from aivtube.testing.contracts import case_id, memory_store_suite

_base: list[Path] = []
_n = itertools.count()


@pytest.fixture(autouse=True)
def _db_dir(tmp_path: Path) -> Iterator[None]:
    _base.append(tmp_path)
    yield
    _base.pop()


def _file_store(**kw: Any) -> Callable[[], SqliteMemory]:
    def factory() -> SqliteMemory:
        path = _base[-1] / f"pailin-{next(_n)}.sqlite"
        return SqliteMemory(path, "pailin", SystemClock(), **kw)

    return factory


def _memory_store() -> SqliteMemory:
    return SqliteMemory(None, "pailin", SystemClock())


def cases(suite: list[Any]) -> Any:
    return pytest.mark.parametrize("case", suite, ids=case_id)


@pytest.mark.skipif(not fts5_trigram_available(), reason="SQLite without FTS5 trigram")
@cases(memory_store_suite(_file_store()))
async def test_sqlite_memory_contract_fts(case: Callable[[], Any]) -> None:
    await case()


@cases(memory_store_suite(_file_store(fts=False)))
async def test_sqlite_memory_contract_like_fallback(case: Callable[[], Any]) -> None:
    await case()


@cases(memory_store_suite(_memory_store))
async def test_sqlite_memory_contract_in_memory(case: Callable[[], Any]) -> None:
    await case()


def test_sqlite_memory_is_a_memory_store() -> None:
    store: MemoryStore = SqliteMemory(None, "pailin", SystemClock())
    assert isinstance(store, MemoryStore)
