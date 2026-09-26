"""Shared doubles for the panel and console tests (imported as ``panel_testkit``)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.tools import ToolContext
from aivtube.contracts.types import Stimulus, StimulusKind
from aivtube.testing.fakes import FakeMemoryStore, FakeSpeechOutput

__all__ = ["EditableMemory", "FakeOps", "RecordingBrain", "tool_context"]


class RecordingBrain:
    """A ``BrainControl`` that records commands; can hang or raise on demand."""

    def __init__(self, *, hang: frozenset[OpKind] = frozenset(), fail: bool = False) -> None:
        self.commands: list[OpCommand] = []
        self.hang = hang
        self.fail = fail
        self.state = "idle"

    async def control(self, cmd: OpCommand) -> OpResult:
        self.commands.append(cmd)
        if cmd.kind in self.hang:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("brain exploded")
        if cmd.kind is OpKind.FREEZE:
            self.state = "paused"
        elif cmd.kind is OpKind.RESUME:
            self.state = "idle"
        return OpResult(True)

    def snapshot(self) -> Mapping[str, Any]:
        return {"state": self.state}

    def kinds(self) -> list[OpKind]:
        return [c.kind for c in self.commands]


class EditableMemory(FakeMemoryStore):
    """``FakeMemoryStore`` plus the operator ``edit`` that the real ``SqliteMemory`` offers."""

    async def edit(
        self,
        memory_id: int,
        *,
        by: str,
        text: str | None = None,
        subject: str | None = None,
        importance: int | None = None,
        locked: bool | None = None,
    ) -> MemoryItem:
        item = self._get(memory_id)
        fields = {"text": text, "subject": subject, "importance": importance, "locked": locked}
        self._set(item, **{k: v for k, v in fields.items() if v is not None})
        return self._get(memory_id)


class FakeOps:
    """``OpAuditSink``: collects ``op_audit`` rows."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def log_op(self, **row: Any) -> None:
        self.rows.append(row)


def tool_context(bus: EventBus, clock: Clock, *, character: str = "pailin") -> ToolContext:
    """A minimal ``ToolContext`` for running tools through the real registry."""
    stimulus = Stimulus(
        id="s-1", kind=StimulusKind.CHAT, character=character, text="ทดสอบ", created=0.0
    )
    return ToolContext(
        character=character,
        turn_id="t-1",
        stimulus=stimulus,
        memory=FakeMemoryStore(),
        speech=FakeSpeechOutput(bus, clock),
        avatar=None,
        channels={},
        bus=bus,
        clock=clock,
    )
