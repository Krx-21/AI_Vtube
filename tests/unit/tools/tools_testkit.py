"""Helpers shared by the tool tests (imported as ``tools_testkit``)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from aivtube.contracts.chat import ChannelActions
from aivtube.contracts.llm import ToolCall
from aivtube.contracts.memory import MemoryStore
from aivtube.contracts.tools import ToolContext
from aivtube.contracts.types import ChatMessage, Platform, Stimulus, StimulusKind
from aivtube.testing.fakes import FakeClock, FakeEventBus, FakeSpeechOutput

__all__ = ["ListAudit", "call", "context", "stimulus"]


class ListAudit:
    """An in-memory ``ToolAuditSink``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def log_tool(self, **row: Any) -> None:
        self.rows.append(row)

    @property
    def verdicts(self) -> list[str]:
        return [r["verdict"] for r in self.rows]


def call(name: str, args: Mapping[str, Any] | None = None, *, raw: str | None = None) -> ToolCall:
    raw_args = raw if raw is not None else json.dumps(dict(args or {}), ensure_ascii=False)
    return ToolCall(
        id=f"call-{name}",
        name=name,
        arguments=None if args is None else dict(args),
        raw_arguments=raw_args,
        extra=None,
    )


def stimulus(
    kind: StimulusKind = StimulusKind.VOICE,
    *,
    source: str = "",
    messages: Sequence[ChatMessage] = (),
    character: str = "pailin",
    created: float = 0.0,
) -> Stimulus:
    payload: dict[str, Any] = {"messages": list(messages)} if messages else {}
    return Stimulus(
        id="s-1",
        kind=kind,
        character=character,
        text="ทดสอบ",
        created=created,
        source=source,
        payload=payload,
    )


def context(
    memory: MemoryStore,
    clock: FakeClock,
    bus: FakeEventBus,
    stim: Stimulus | None = None,
    *,
    channels: Mapping[Platform, ChannelActions] | None = None,
    turn_id: str = "t-1",
) -> ToolContext:
    return ToolContext(
        character="pailin",
        turn_id=turn_id,
        stimulus=stim or stimulus(),
        memory=memory,
        speech=FakeSpeechOutput(bus, clock),
        avatar=None,
        channels=channels or {},
        bus=bus,
        clock=clock,
    )
