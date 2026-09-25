"""Contract suites for ``ChatSource``, ``ChannelActions`` and ``ChatWindow`` (§3.8)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from aivtube.contracts.chat import ChannelActions, ChatSelection, ChatSource, ChatWindow
from aivtube.contracts.types import ChatMessage, ChatUser, Health, MsgKind, Platform
from aivtube.testing.contracts._base import (
    AsyncCase,
    SyncCase,
    _Cases,
    check,
    close_quietly,
    maybe_await,
    raises,
)

__all__ = ["channel_actions_suite", "chat_source_suite", "chat_window_suite"]

KNOWN_CAPABILITIES = frozenset({"send", "timeout", "untimeout", "poll", "title"})


def chat_source_suite(
    factory: Callable[[], ChatSource | Awaitable[ChatSource]],
    *,
    expected_min: int = 1,
    within: float = 5.0,
) -> list[AsyncCase]:
    """``factory`` returns a source that will deliver at least ``expected_min`` distinct
    messages (for example ``TwitchAnonIrc`` pointed at a ``FakeIrcServer``)."""
    cases = _Cases("chat_source")

    @cases
    async def attributes() -> None:
        src = await maybe_await(factory())
        try:
            check(isinstance(src.platform, Platform), "platform must be a Platform")
            check(isinstance(src.health(), Health), "health() must return Health")
        finally:
            await src.aclose()

    @cases
    async def yields_distinct_chat_messages() -> None:
        src = await maybe_await(factory())
        got: list[ChatMessage] = []

        async def read() -> None:
            async for m in src.messages():
                got.append(m)
                if len(got) >= expected_min:
                    return

        try:
            await asyncio.wait_for(read(), within)
            for m in got:
                check(isinstance(m, ChatMessage), f"yielded {type(m).__name__}")
                check(m.platform == src.platform, "message platform differs from the source")
                check(isinstance(m.user, ChatUser) and m.id, "user and id")
            ids = [m.id for m in got]
            check(len(ids) == len(set(ids)), f"duplicate ids delivered: {ids}")
        finally:
            await src.aclose()

    @cases
    async def aclose_is_idempotent() -> None:
        src = await maybe_await(factory())
        await src.aclose()
        await src.aclose()
        check(isinstance(src.health(), Health), "health() after aclose()")

    return cases.items


def channel_actions_suite(
    factory: Callable[[], ChannelActions | Awaitable[ChannelActions]],
) -> list[AsyncCase]:
    """Supported actions must succeed; actions outside ``capabilities`` must raise."""
    cases = _Cases("channel_actions")
    viewer = ChatUser(Platform.TWITCH, "u-1", "viewer")

    @cases
    async def capabilities_are_known() -> None:
        ch = await maybe_await(factory())
        check(isinstance(ch.platform, Platform), "platform")
        check(isinstance(ch.capabilities, frozenset), "capabilities must be a frozenset")
        check(
            ch.capabilities <= KNOWN_CAPABILITIES,
            f"unknown: {ch.capabilities - KNOWN_CAPABILITIES}",
        )

    @cases
    async def supported_actions_work_unsupported_raise() -> None:
        ch = await maybe_await(factory())
        user = ChatUser(ch.platform, viewer.id, viewer.name)
        actions: dict[str, Callable[[], Awaitable[object]]] = {
            "send": lambda: ch.send("สวัสดีค่ะ"),
            "timeout": lambda: ch.timeout(user, 60, "สแปม"),
            "untimeout": lambda: ch.untimeout(user),
            "poll": lambda: ch.create_poll("เล่นเกมอะไรดี", ["มายคราฟ", "โอเวอร์คุก"], 60),
            "title": lambda: ch.set_title("ไพลินเล่นเกม"),
        }
        for cap, call in actions.items():
            if cap in ch.capabilities:
                result = await call()
                if cap == "poll":
                    check(isinstance(result, str) and result, "create_poll must return an id")
            else:
                with raises(Exception, what=f"unsupported {cap!r}"):
                    await call()
        await close_quietly(ch)

    return cases.items


def _msg(i: int, text: str, *, kind: MsgKind = MsgKind.TEXT, now: float) -> ChatMessage:
    user = ChatUser(Platform.TWITCH, f"u-{i}", f"viewer{i}")
    return ChatMessage(Platform.TWITCH, f"m-{i}", user, text, now, now, kind=kind)


def chat_window_suite(
    factory: Callable[[], ChatWindow],
    *,
    now: Callable[[], float] = time.perf_counter,
) -> list[SyncCase]:
    """``now`` must be the clock the window uses (``FakeClock.now`` when it is injected)."""
    cases = _Cases("chat_window")
    literals = ("window", "priority", "dropped")

    @cases
    def add_classifies_and_drops_duplicates() -> None:
        w = factory()
        m = _msg(1, "เล่นเกมอะไรอยู่คะ", now=now())
        check(w.add(m) in literals, "add() must return window|priority|dropped")
        check(w.add(m) == "dropped", "a duplicate id must be dropped")

    @cases
    def select_consumes_the_window() -> None:
        w = factory()
        t = now()
        added = [_msg(i, f"ข้อความที่ {i}", now=t) for i in range(6)]
        for m in added:
            w.add(m)
        check(w.pending()[0] >= 1, "pending() did not count the messages")
        sel = w.select(now(), k=3)
        check(isinstance(sel, ChatSelection), "select() must return ChatSelection")
        check(len(sel.candidates) <= 3, f"{len(sel.candidates)} candidates for k=3")
        ids = {m.id for m in added}
        for m in (*sel.must_ack, *sel.candidates, *sel.ambient):
            check(m.id in ids, "select() returned a message that was never added")
        check(w.pending()[0] == 0, "select() must consume the window")

    @cases
    def consume_removes_a_message() -> None:
        w = factory()
        t = now()
        a, b = _msg(1, "ข้อความแรก", now=t), _msg(2, "ข้อความสอง", now=t)
        w.add(a)
        w.add(b)
        got = w.consume(a.id)
        check(got is not None and got.id == a.id, "consume() did not return the message")
        check(w.consume("never-added") is None, "consume() of an unknown id must be None")
        sel = w.select(now(), k=3)
        check(
            a.id not in {m.id for m in (*sel.must_ack, *sel.candidates, *sel.ambient)},
            "a consumed message was selected",
        )

    @cases
    def recent_and_snapshot() -> None:
        w = factory()
        m = _msg(1, "ทักทายค่ะ", now=now())
        w.add(m)
        check(m.id in {x.id for x in w.recent(60.0)}, "recent() misses a new message")
        snap = w.snapshot()
        check(isinstance(snap, list), "snapshot() must be a list")
        for item, score in snap:
            check(isinstance(item, ChatMessage) and isinstance(score, float), "snapshot entries")

    return cases.items
