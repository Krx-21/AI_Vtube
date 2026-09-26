"""``CoreControl`` against the real safety gate, tool registry and SQLite memory store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from panel_testkit import FakeOps, RecordingBrain

from aivtube.contracts.control import OpCommand, OpKind
from aivtube.contracts.memory import MemoryItem
from aivtube.memory import SqliteMemory
from aivtube.panel.control import CoreControl
from aivtube.safety import FilterListError, LayeredSafetyGate
from aivtube.testing.fakes import (
    ECHO_SPEC,
    FakeClock,
    FakeEventBus,
    FakeLLM,
    FakeLLMRouter,
    FakeSpeechOutput,
    FakeTaskSupervisor,
    FakeTextFilter,
    FakeTool,
    make_chat_message,
)
from aivtube.tools import PolicyToolRegistry


class BrokenListsFilter(FakeTextFilter):
    """A tier-0 filter whose list files are invalid when ``broken`` is set."""

    broken = False

    def reload(self) -> None:
        if self.broken:
            raise FilterListError(Path("config/filters/base/slurs.toml"), "bad TOML", "ไฟล์เสีย")
        super().reload()


@dataclass
class Real:
    clock: FakeClock
    bus: FakeEventBus
    tier0: BrokenListsFilter
    gate: LayeredSafetyGate
    registry: PolicyToolRegistry
    memory: SqliteMemory
    brains: dict[str, RecordingBrain]
    ops: FakeOps
    control: CoreControl

    async def op(
        self, kind: OpKind, character: str | None = None, **args: object
    ) -> tuple[bool, str]:
        result = await self.control.execute(OpCommand(kind, dict(args), character, "panel", "x"))
        return result.ok, result.detail


def _mode(real: Real) -> str:
    """The registry's tools mode (a call, so mypy does not narrow it between commands)."""
    return str(real.registry.mode)


@pytest.fixture
async def real() -> AsyncIterator[Real]:
    clock = FakeClock()
    bus = FakeEventBus(clock)
    tier0 = BrokenListsFilter(["คำต้องห้าม"])
    gate = LayeredSafetyGate(tier0, bus=bus, clock=clock)
    registry = PolicyToolRegistry(
        [FakeTool(ECHO_SPEC)],
        gate=gate,
        ops=None,
        bus=bus,
        clock=clock,
        enabled_by_character={"pailin": ["echo"], "twin": []},
    )
    memory = SqliteMemory(None, "pailin", clock, tokenizer=str.split)
    await memory.start_session("test")
    brains = {"pailin": RecordingBrain(), "twin": RecordingBrain()}
    ops = FakeOps()
    tasks = FakeTaskSupervisor(clock)

    async def restart(name: str) -> None:
        return None

    control = CoreControl(
        brains,
        router=FakeLLMRouter([FakeLLM([], name="local-30b", clock=clock)], clock=clock),
        registry=registry,
        speech=FakeSpeechOutput(bus, clock),
        memory_by_char={"pailin": memory},
        safety=gate,
        ops=ops,
        bus=bus,
        clock=clock,
        restart=restart,
        tasks=tasks,
    )
    try:
        yield Real(clock, bus, tier0, gate, registry, memory, brains, ops, control)
    finally:
        await tasks.aclose()
        await memory.aclose()


async def test_strict_uses_the_gate_for_one_or_every_character(real: Real) -> None:
    assert await real.op(OpKind.STRICT, "pailin", on=True) == (True, "strict on: pailin")
    assert real.gate.strict_mode("pailin") and not real.gate.strict_mode("twin")
    assert real.gate.status()["strict"] == {"pailin": "operator"}
    ok, _ = await real.op(OpKind.STRICT, on=True)
    assert ok and real.gate.strict_mode("twin")
    ok, _ = await real.op(OpKind.STRICT, on=False)
    assert ok and not real.gate.strict_mode("pailin") and not real.gate.strict_mode("twin")
    assert real.control.snapshot()["strict"] == {"pailin": False, "twin": False}
    assert real.control.snapshot()["safety"]["strict"] == {}
    # the brains hear about it (flags) but the gate is the source of truth
    assert real.brains["pailin"].kinds() == [OpKind.STRICT] * 3
    ok, detail = await real.op(OpKind.STRICT, "nobody", on=True)
    assert not ok and "unknown character" in detail


async def test_filter_reload_reports_bad_lists_and_keeps_the_old_ones(real: Real) -> None:
    assert await real.op(OpKind.FILTER_RELOAD) == (True, "filters reloaded")
    assert real.tier0.reloads == 1
    real.tier0.broken = True
    ok, detail = await real.op(OpKind.FILTER_RELOAD)
    assert not ok and detail.startswith("filter lists not reloaded:") and "bad TOML" in detail
    result, _name = real.gate.check_input(make_chat_message("คำต้องห้าม"), character="pailin")
    assert result.verdict.value == "drop"  # the previous lists still work
    await real.clock.run_until_idle()
    assert [r["result"].startswith("error:") for r in real.ops.rows] == [False, True]


async def test_freeze_switches_tools_off_until_resume(real: Real) -> None:
    assert _mode(real) == "live"
    assert (await real.op(OpKind.FREEZE))[0]
    assert _mode(real) == "off"
    assert real.control.snapshot()["tools"]["resume_mode"] == "live"
    ok, detail = await real.op(OpKind.TOOLS_MODE, mode="dry_run")
    assert ok and "after RESUME" in detail and _mode(real) == "off"
    assert (await real.op(OpKind.RESUME))[0]
    assert _mode(real) == "dry_run"
    tools = real.control.snapshot()["tools"]
    assert tools["mode"] == "dry_run" and "resume_mode" not in tools
    assert tools["tools"] == {"echo": {"enabled": True}}
    # a character-scoped FREEZE leaves the shared registry alone
    assert (await real.op(OpKind.FREEZE, "twin"))[0]
    assert _mode(real) == "dry_run"


async def test_memory_round_trip_on_the_real_store(real: Real) -> None:
    item = await real.memory.remember(
        MemoryItem(id=None, kind="fact", text="ชอบแมว", source="operator", status="quarantined")
    )
    assert item.id is not None and item.status == "quarantined"
    mid = item.id
    ok, detail = await real.op(OpKind.MEMORY_STATUS, id=mid, status="active")
    assert ok, detail
    ok, detail = await real.op(OpKind.MEMORY_EDIT, id=mid, text="ชอบแมวส้ม", locked=True)
    assert ok, detail
    [stored] = await real.memory.list_memories()
    assert (stored.status, stored.text, stored.locked) == ("active", "ชอบแมวส้ม", True)
    ok, detail = await real.op(OpKind.MEMORY_STATUS, id=mid, status="deleted")
    assert not ok and detail.startswith("refused:") and "locked" in detail
    ok, detail = await real.op(OpKind.MEMORY_EDIT, id=mid, pinned=True)
    assert not ok and "cannot edit pinned" in detail
    ok, detail = await real.op(OpKind.MEMORY_EDIT, id=mid, text="   ")
    assert not ok and "empty" in detail
    assert await real.op(OpKind.MEMORY_STATUS, id=999, status="active") == (
        False,
        "unknown memory 999",
    )
    await real.clock.run_until_idle()
    assert [r["command"] for r in real.ops.rows] == [
        "memory_status",
        "memory_edit",
        "memory_status",
        "memory_edit",
        "memory_edit",
        "memory_status",
    ]
