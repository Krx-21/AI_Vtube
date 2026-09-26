"""The ToolRegistry contract suite against the real ``PolicyToolRegistry`` (§3.11, §10)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from tools_testkit import ListAudit

from aivtube.contracts.safety import SafetyGate
from aivtube.contracts.tools import Tool, ToolRegistry
from aivtube.infra import SystemClock
from aivtube.testing.contracts import case_id, tool_registry_suite
from aivtube.testing.fakes import FakeEventBus
from aivtube.tools import PolicyToolRegistry


def _factory(tools: Sequence[Tool], gate: SafetyGate) -> PolicyToolRegistry:
    clock = SystemClock()
    return PolicyToolRegistry(
        tools,
        gate=gate,
        ops=ListAudit(),
        bus=FakeEventBus(clock),
        clock=clock,
        enabled_by_character={"pailin": [t.spec.name for t in tools]},
    )


@pytest.mark.parametrize("case", tool_registry_suite(_factory), ids=case_id)
async def test_policy_registry_contract(case: Callable[[], Any]) -> None:
    await case()


def test_policy_registry_is_a_tool_registry() -> None:
    reg: ToolRegistry = _factory([], gate=None)  # type: ignore[arg-type]
    assert isinstance(reg, ToolRegistry)
