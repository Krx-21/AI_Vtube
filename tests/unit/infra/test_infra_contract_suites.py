"""The real infra implementations pass the shared contract suites (§10)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aivtube.infra import AsyncEventBus, SupervisedTasks, SystemClock
from aivtube.testing.contracts import (
    case_id,
    clock_suite,
    event_bus_suite,
    task_supervisor_suite,
)


def cases(suite: list[Any]) -> Any:
    return pytest.mark.parametrize("case", suite, ids=case_id)


def _supervisor() -> SupervisedTasks:
    clock = SystemClock()
    return SupervisedTasks(clock, AsyncEventBus(clock), on_critical_failure=lambda n, e: None)


@cases(clock_suite(SystemClock))
async def test_system_clock_contract(case: Callable[[], Any]) -> None:
    await case()


@cases(event_bus_suite(lambda: AsyncEventBus(SystemClock())))
async def test_event_bus_contract(case: Callable[[], Any]) -> None:
    await case()


@cases(task_supervisor_suite(_supervisor))
async def test_task_supervisor_contract(case: Callable[[], Any]) -> None:
    await case()
