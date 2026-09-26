"""Chat protocols: sources, channel actions and the chat window (ARCHITECTURE.md §3.8)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from aivtube.contracts.types import ChatMessage, ChatUser, Health, Platform

__all__ = ["ChannelActions", "ChatSelection", "ChatSource", "ChatWindow"]


@runtime_checkable
class ChatSource(Protocol):
    """``TwitchAnonIrc`` | ``YouTubeListPoller`` | (M3) EventSub, streamList, alerts | fakes."""

    platform: Platform

    def messages(self) -> AsyncIterator[ChatMessage]:
        """Yield messages forever; reconnects, dedupes and skips the backlog internally."""
        ...

    def health(self) -> Health: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class ChannelActions(Protocol):
    """Actions on the streamer's channel (M3)."""

    platform: Platform
    capabilities: frozenset[str]  # subset of {"send", "timeout", "untimeout", "poll", "title"}

    async def send(self, text: str, reply_to: str | None = None) -> None: ...

    async def timeout(self, user: ChatUser, seconds: int, reason: str) -> None: ...

    async def untimeout(self, user: ChatUser) -> None: ...

    async def create_poll(self, title: str, choices: Sequence[str], seconds: int) -> str: ...

    async def set_title(self, title: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatSelection:
    must_ack: tuple[ChatMessage, ...]
    candidates: tuple[ChatMessage, ...]
    ambient: tuple[ChatMessage, ...]


@runtime_checkable
class ChatWindow(Protocol):
    def add(self, m: ChatMessage) -> Literal["window", "priority", "dropped"]: ...

    def select(self, now: float, k: int = 3) -> ChatSelection:
        """Pick up to ``k`` candidates; consumes the window."""
        ...

    def pending(self) -> tuple[int, bool]:
        """``(count, has_mention)``."""
        ...

    def consume(self, message_id: str) -> ChatMessage | None:
        """Remove a message that the streamer read aloud (read-aloud dedupe)."""
        ...

    def recent(self, seconds: float) -> tuple[ChatMessage, ...]: ...

    def snapshot(self) -> list[tuple[ChatMessage, float]]:
        """Window contents with scores, for the panel."""
        ...
