"""Tool protocols: policy, context, tools and the registry (ARCHITECTURE.md §3.11, §4.9)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.avatar import AvatarSink
from aivtube.contracts.chat import ChannelActions
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.llm import ToolCall, ToolSpec
from aivtube.contracts.memory import MemoryStore
from aivtube.contracts.speech import SpeechOutput
from aivtube.contracts.types import Platform, Stimulus

__all__ = [
    "RateLimit",
    "Risk",
    "Tool",
    "ToolContext",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
]

Risk: TypeAlias = Literal["safe", "moderate", "dangerous"]


@dataclass(frozen=True, slots=True)
class RateLimit:
    max_calls: int
    per_s: float


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Enforced in code by the registry, never by the model. ``requires`` lists channel
    capabilities (for example ``"timeout"``) that must be available."""

    risk: Risk = "safe"
    side_effect: bool = False
    follow_up: bool = False
    requires_approval: bool = False
    rate_limit: RateLimit | None = None
    timeout_s: float = 5.0
    requires: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """``content`` is fed back to the model; ``note`` is for logs and the panel only."""

    ok: bool
    content: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    character: str
    turn_id: str
    stimulus: Stimulus
    memory: MemoryStore
    speech: SpeechOutput
    avatar: AvatarSink | None
    channels: Mapping[Platform, ChannelActions]
    bus: EventBus
    clock: Clock


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec
    policy: ToolPolicy

    async def __call__(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult: ...


@runtime_checkable
class ToolRegistry(Protocol):
    def specs(self, character: str) -> tuple[ToolSpec, ...]:
        """Static per session, in a cache-stable order (disabled tools stay listed)."""
        ...

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Validate the arguments, filter them, apply the policy, then run the tool."""
        ...

    def set_enabled(self, name: str, enabled: bool) -> None: ...

    def set_mode(self, mode: Literal["live", "dry_run", "off"]) -> None: ...
