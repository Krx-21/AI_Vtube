"""LLM protocols: providers, router and local server manager (ARCHITECTURE.md §3.6)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.types import Health

__all__ = [
    "ChatRequest",
    "Done",
    "LLMEvent",
    "LLMProvider",
    "LLMRouter",
    "LocalServerManager",
    "ProviderCaps",
    "ProviderFailed",
    "ProviderStatus",
    "SlotRole",
    "TextDelta",
    "ToolCall",
    "ToolSpec",
]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]  # JSON Schema with "type": "object"


SlotRole: TypeAlias = Literal["speak", "game", "background"]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One chat completion. Message ``content`` is always a plain string (never parts)."""

    messages: tuple[Mapping[str, Any], ...]
    purpose: SlotRole = "speak"
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: Literal["auto", "required", "none"] = "auto"
    response_schema: Mapping[str, Any] | None = None
    max_tokens: int = 256
    temperature: float = 0.6
    character: str = ""
    turn_id: str = ""
    slot: int | None = None
    first_token_timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed tool call. ``arguments`` is ``None`` when ``raw_arguments`` is not valid JSON;
    ``extra`` carries provider data to echo back verbatim (Gemini ``thought_signature``)."""

    id: str
    name: str
    arguments: Mapping[str, Any] | None
    raw_arguments: str
    extra: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class Done:
    provider: str
    finish_reason: str | None
    ttft_ms: float
    prompt_n: int | None
    cache_n: int | None
    completion_tokens: int | None
    assistant_message: Mapping[str, Any]


LLMEvent: TypeAlias = TextDelta | ToolCall | Done
"""Stream items. ``ToolCall``s are yielded after the text stream ends, then ``Done``."""


@dataclass(frozen=True, slots=True)
class ProviderCaps:
    tools: bool
    parallel_tools: bool
    json_schema: bool
    prompt_cache: bool
    cloud: bool
    keeps_thought_signatures: bool
    reasoning: Literal["none", "effort", "template_kwarg"]


class ProviderFailed(Exception):
    """A provider failed. ``emitted`` tells whether any event had already been yielded; the
    router falls back to the next provider only when it is ``False``."""

    emitted: bool

    def __init__(self, msg: str, *, emitted: bool) -> None:
        super().__init__(msg)
        self.emitted = emitted

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild_provider_failed, (str(self), self.emitted))


def _rebuild_provider_failed(msg: str, emitted: bool) -> ProviderFailed:
    return ProviderFailed(msg, emitted=emitted)


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    caps: ProviderCaps

    async def probe(self) -> Health: ...

    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        """Stream one completion. Closing the iterator (``aclose()``) closes the HTTP stream,
        and the server must cancel generation within 0.2 s."""
        ...

    async def prefill(self, req: ChatRequest) -> float:
        """Warm the KV cache for ``req``'s prefix; returns seconds taken (cloud: no-op, 0.0)."""
        ...


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    name: str
    healthy: bool
    active: bool
    enabled: bool
    cloud: bool
    down_until: float
    fails: int
    ttft_p50_ms: float | None


@runtime_checkable
class LLMRouter(Protocol):
    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        """Like ``LLMProvider.stream``; falls back to the next provider only before the first
        event, otherwise raises ``ProviderFailed(emitted=True)``."""
        ...

    async def prefill(self, req: ChatRequest) -> None: ...

    def promote(self, name: str) -> None:
        """Make ``name`` active at the next decision boundary; the previous one is kept."""
        ...

    def rollback(self) -> None: ...

    def active(self) -> str: ...

    def status(self) -> list[ProviderStatus]: ...


@runtime_checkable
class LocalServerManager(Protocol):
    """Starts/stops llama-server instances (through the launcher) and manages slot files."""

    async def ensure_running(self, server: str, timeout_s: float) -> bool: ...

    async def stop(self, server: str) -> None: ...

    async def props(self, server: str) -> Mapping[str, Any]: ...

    async def save_slot(self, server: str, slot: int, filename: str) -> bool: ...

    async def restore_slot(self, server: str, slot: int, filename: str) -> bool: ...
