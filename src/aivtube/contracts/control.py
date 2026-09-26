"""Operator control: commands, results and the control surface (ARCHITECTURE.md §3.12, §2.11)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = ["ControlSurface", "OpCommand", "OpKind", "OpResult"]


class OpKind(StrEnum):
    SKIP = "skip"
    MUTE = "mute"
    UNMUTE = "unmute"
    FREEZE = "freeze"
    RESUME = "resume"
    GO_LIVE = "go_live"
    CHAT_INTAKE = "chat_intake"
    MIC_MODE = "mic_mode"
    PTT = "ptt"
    SAY = "say"
    DIRECT = "direct"
    FAKE_CHAT = "fake_chat"
    INJECT_EVENT = "inject_event"
    LLM_USE = "llm_use"
    LLM_ROLLBACK = "llm_rollback"
    TTS_IDENTITY = "tts_identity"
    TOOLS_MODE = "tools_mode"
    TOOL_ENABLE = "tool_enable"
    APPROVE = "approve"
    MEMORY_EDIT = "memory_edit"
    MEMORY_STATUS = "memory_status"
    MUTE_USER = "mute_user"
    STRICT = "strict"
    FILTER_RELOAD = "filter_reload"
    RESTART = "restart"
    END_STREAM = "end_stream"


@dataclass(frozen=True, slots=True)
class OpCommand:
    kind: OpKind
    args: Mapping[str, Any] = field(default_factory=dict)
    character: str | None = None
    operator: str = "local"
    id: str = ""


@dataclass(frozen=True, slots=True)
class OpResult:
    ok: bool
    detail: str = ""
    latency_ms: float = 0.0


@runtime_checkable
class ControlSurface(Protocol):
    async def execute(self, cmd: OpCommand) -> OpResult:
        """Run an operator command (I4). FREEZE, SKIP and MUTE take a synchronous fast path."""
        ...

    def snapshot(self) -> Mapping[str, Any]: ...
