"""The real infra implementations against the same contract suites as the fakes (§10 layer 2).

Skipped while ``aivtube.infra`` is not importable (it is written in parallel with the fakes).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aivtube.testing.contracts import case_id, clock_suite, event_bus_suite, task_supervisor_suite

infra = pytest.importorskip("aivtube.infra")


def _bus() -> Any:
    return infra.AsyncEventBus(infra.SystemClock())


def _supervisor() -> Any:
    clock = infra.SystemClock()
    failures: list[tuple[str, BaseException]] = []
    return infra.SupervisedTasks(
        clock, infra.AsyncEventBus(clock), on_critical_failure=lambda n, e: failures.append((n, e))
    )


@pytest.mark.parametrize("case", clock_suite(lambda: infra.SystemClock()), ids=case_id)
async def test_system_clock(case: Callable[[], Any]) -> None:
    await case()


@pytest.mark.parametrize("case", event_bus_suite(_bus), ids=case_id)
async def test_async_event_bus(case: Callable[[], Any]) -> None:
    await case()


@pytest.mark.parametrize("case", task_supervisor_suite(_supervisor), ids=case_id)
async def test_supervised_tasks(case: Callable[[], Any]) -> None:
    await case()
