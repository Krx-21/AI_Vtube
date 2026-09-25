"""Chat fakes: ``FakeChatSource``, ``FakeChannelActions``, ``FakeChatWindow`` (§3.8)."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal

from aivtube.contracts.chat import ChatSelection
from aivtube.contracts.infra import Clock
from aivtube.contracts.types import ChatMessage, ChatUser, Health, HealthState, MsgKind, Platform

__all__ = [
    "ALL_CAPABILITIES",
    "FakeChannelActions",
    "FakeChatSource",
    "FakeChatWindow",
    "make_chat_message",
]

ALL_CAPABILITIES = frozenset({"send", "timeout", "untimeout", "poll", "title"})
_ids = itertools.count(1)


def make_chat_message(
    text: str,
    *,
    user: str = "viewer",
    platform: Platform = Platform.TWITCH,
    kind: MsgKind = MsgKind.TEXT,
    id: str | None = None,
    ts: float | None = None,
    user_id: str | None = None,
    amount: float = 0.0,
    currency: str = "",
    value_usd: float = 0.0,
    first_msg: bool = False,
    is_mod: bool = False,
    is_sub: bool = False,
    is_vip: bool = False,
    is_broadcaster: bool = False,
    clock: Clock | None = None,
) -> ChatMessage:
    """Build a ``ChatMessage`` with sensible defaults for tests."""
    now = clock.now() if clock else time.perf_counter()
    n = next(_ids)
    return ChatMessage(
        platform=platform,
        id=id or f"msg-{n}",
        user=ChatUser(
            platform=platform,
            id=user_id or f"u-{user}",
            name=user,
            is_broadcaster=is_broadcaster,
            is_mod=is_mod,
            is_vip=is_vip,
            is_sub=is_sub,
        ),
        text=text,
        ts=now if ts is None else ts,
        received=now,
        kind=kind,
        amount=amount,
        currency=currency,
        value_usd=value_usd,
        first_msg=first_msg,
    )


_CLOSE = object()


class FakeChatSource:
    """``ChatSource`` yielding scripted messages, then anything ``push``-ed, deduped by id.

    ``fail_with(exc)`` makes the iterator raise ``exc`` once (to exercise supervisor restarts).
    """

    def __init__(
        self,
        messages: Iterable[ChatMessage] = (),
        *,
        platform: Platform = Platform.TWITCH,
        dedupe: bool = True,
    ) -> None:
        self.platform = platform
        self.dedupe = dedupe
        self._q: asyncio.Queue[Any] = asyncio.Queue()
        for m in messages:
            self._q.put_nowait(m)
        self._seen: set[str] = set()
        self._state = HealthState.OK
        self._detail = ""
        self.closed = False
        self.yielded: list[ChatMessage] = []
        self.duplicates = 0

    def push(self, message: ChatMessage) -> None:
        self._q.put_nowait(message)

    def push_text(self, user: str, text: str, **kw: Any) -> ChatMessage:
        m = make_chat_message(text, user=user, platform=self.platform, **kw)
        self.push(m)
        return m

    def fail_with(self, exc: BaseException) -> None:
        self._q.put_nowait(exc)

    def set_health(self, state: HealthState, detail: str = "") -> None:
        self._state, self._detail = state, detail

    async def messages(self) -> AsyncIterator[ChatMessage]:
        while not self.closed:
            item = await self._q.get()
            if item is _CLOSE:
                break
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, ChatMessage)
            if self.dedupe:
                if item.id in self._seen:
                    self.duplicates += 1
                    continue
                self._seen.add(item.id)
            self.yielded.append(item)
            yield item

    def health(self) -> Health:
        state = HealthState.DOWN if self.closed else self._state
        return Health(f"chat:{self.platform.value}", state, self._detail)

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self._q.put_nowait(_CLOSE)


class FakeChannelActions:
    """``ChannelActions`` that records every call; actions outside ``capabilities`` raise
    ``NotImplementedError`` and a non-positive timeout raises ``ValueError``."""

    def __init__(
        self,
        platform: Platform = Platform.TWITCH,
        capabilities: Iterable[str] = ALL_CAPABILITIES,
    ) -> None:
        self.platform = platform
        self.capabilities = frozenset(capabilities)
        self.sent: list[tuple[str, str | None]] = []
        self.timeouts: list[tuple[ChatUser, int, str]] = []
        self.untimeouts: list[ChatUser] = []
        self.polls: list[tuple[str, tuple[str, ...], int]] = []
        self.titles: list[str] = []

    def _need(self, cap: str) -> None:
        if cap not in self.capabilities:
            raise NotImplementedError(f"{self.platform.value} channel cannot {cap!r}")

    async def send(self, text: str, reply_to: str | None = None) -> None:
        self._need("send")
        self.sent.append((text, reply_to))

    async def timeout(self, user: ChatUser, seconds: int, reason: str) -> None:
        self._need("timeout")
        if seconds <= 0:
            raise ValueError("timeout seconds must be positive")
        self.timeouts.append((user, seconds, reason))

    async def untimeout(self, user: ChatUser) -> None:
        self._need("untimeout")
        self.untimeouts.append(user)

    async def create_poll(self, title: str, choices: Sequence[str], seconds: int) -> str:
        self._need("poll")
        if len(choices) < 2:
            raise ValueError("a poll needs at least two choices")
        self.polls.append((title, tuple(choices), seconds))
        return f"poll-{len(self.polls)}"

    async def set_title(self, title: str) -> None:
        self._need("title")
        self.titles.append(title)


class FakeChatWindow:
    """A simple ``ChatWindow``: support/mentions go to must-ack, the newest ``k`` texts are
    candidates, the rest is ambient. Duplicate ids and overflow beyond ``maxlen`` are dropped."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        mention_names: Sequence[str] = ("ไพลิน", "pailin"),
        maxlen: int = 50,
    ) -> None:
        self._clock = clock
        self.mention_names = tuple(n.casefold() for n in mention_names)
        self.maxlen = maxlen
        self._window: list[ChatMessage] = []
        self._priority: list[ChatMessage] = []
        self._seen: set[str] = set()
        self._history: list[ChatMessage] = []

    def _now(self) -> float:
        return self._clock.now() if self._clock else time.perf_counter()

    def _is_mention(self, m: ChatMessage) -> bool:
        t = m.text.casefold()
        return any(n in t for n in self.mention_names)

    def add(self, m: ChatMessage) -> Literal["window", "priority", "dropped"]:
        if m.id in self._seen:
            return "dropped"
        self._seen.add(m.id)
        self._history.append(m)
        if m.kind is not MsgKind.TEXT:
            self._priority.append(m)
            return "priority"
        if len(self._window) >= self.maxlen:
            return "dropped"
        self._window.append(m)
        return "window"

    def select(self, now: float, k: int = 3) -> ChatSelection:
        mentions = [m for m in self._window if self._is_mention(m)]
        rest = [m for m in self._window if not self._is_mention(m)]
        ordered = mentions + sorted(rest, key=lambda m: m.received, reverse=True)
        candidates = tuple(ordered[:k])
        ambient = tuple(m for m in self._window if m not in candidates)
        sel = ChatSelection(tuple(self._priority), candidates, ambient)
        self._window.clear()
        self._priority.clear()
        return sel

    def pending(self) -> tuple[int, bool]:
        count = len(self._window) + len(self._priority)
        return count, any(self._is_mention(m) for m in self._window)

    def consume(self, message_id: str) -> ChatMessage | None:
        for bucket in (self._window, self._priority):
            for i, m in enumerate(bucket):
                if m.id == message_id:
                    return bucket.pop(i)
        return None

    def recent(self, seconds: float) -> tuple[ChatMessage, ...]:
        cutoff = self._now() - seconds
        return tuple(m for m in self._history if m.received >= cutoff)

    def snapshot(self) -> list[tuple[ChatMessage, float]]:
        now = self._now()
        return [
            (m, (2.0 if self._is_mention(m) else 1.0) / (1.0 + max(0.0, now - m.received)))
            for m in self._window
        ]
