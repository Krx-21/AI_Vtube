"""Shared fixtures for the tool registry and memory tool tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from tools_testkit import context

from aivtube.contracts.tools import ToolContext
from aivtube.contracts.types import Stimulus
from aivtube.memory import OpsDb, SqliteMemory
from aivtube.testing.fakes import FakeClock, FakeEventBus


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def events(clock: FakeClock) -> FakeEventBus:
    return FakeEventBus(clock)


@pytest.fixture
async def memory(tmp_path: Path, clock: FakeClock) -> AsyncIterator[SqliteMemory]:
    store = SqliteMemory(tmp_path / "pailin.sqlite", "pailin", clock, tokenizer=lambda t: [t])
    await store.start_session("tools")
    yield store
    await store.aclose()


@pytest.fixture
async def ops(tmp_path: Path, clock: FakeClock) -> AsyncIterator[OpsDb]:
    db = OpsDb(tmp_path / "ops.db", clock=clock)
    yield db
    await db.aclose()


@pytest.fixture
def make_ctx(
    memory: SqliteMemory, clock: FakeClock, events: FakeEventBus
) -> Callable[..., ToolContext]:
    def factory(stim: Stimulus | None = None, **kw: Any) -> ToolContext:
        return context(memory, clock, events, stim, **kw)

    return factory
