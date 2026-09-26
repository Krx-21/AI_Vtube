"""TwitchAnonIrc against FakeIrcServer (real localhost WebSocket) and a scripted socket:
login, PING/PONG, keepalive, RECONNECT, disconnect + backoff, dedupe, backlog skip, health."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError

from aivtube.chat import TwitchAnonIrc
from aivtube.contracts.chat import ChatSource
from aivtube.contracts.types import ChatMessage, HealthState, MsgKind
from aivtube.infra import SystemClock
from aivtube.testing.fakes import SAMPLE_IRC_LINES, FakeClock, FakeIrcServer

CHANNEL = "pailin_th"
UNIQUE_IDS = [
    f"8c0f4d2e-00{n:02d}-4000-8000-0000000000{n:02d}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13)
]


async def until(predicate: Callable[[], bool], timeout: float = 3.0, what: str = "") -> None:
    """Wait in real time (sockets are real even when the clock is fake)."""
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > end:
            raise AssertionError(f"timed out waiting for {what or predicate}")
        await asyncio.sleep(0.005)


class Reader:
    """Consumes ``messages()`` in a background task."""

    def __init__(self, src: TwitchAnonIrc) -> None:
        self.src = src
        self.got: list[ChatMessage] = []
        self.task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        async for m in self.src.messages():
            self.got.append(m)

    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.got]

    async def stop(self) -> None:
        await self.src.aclose()
        await asyncio.wait_for(self.task, 3.0)


# --- scripted socket -------------------------------------------------------------------------
_EOF = object()


class ScriptedWs:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._q: asyncio.Queue[Any] = asyncio.Queue()

    def feed(self, *lines: str) -> None:
        self._q.put_nowait("".join(f"{line}\r\n" for line in lines))

    def hang_up(self) -> None:
        self._q.put_nowait(_EOF)

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionClosedError(None, None)
        self.sent.append(message)

    async def recv(self) -> str:
        item = await self._q.get()
        if item is _EOF or self.closed:
            self.closed = True
            raise ConnectionClosedError(None, None)
        assert isinstance(item, str)
        return item

    async def close(self) -> None:
        self.closed = True
        self._q.put_nowait(_EOF)


class Connector:
    """``connect`` stand-in: hands out scripted sockets (or raises scripted errors)."""

    def __init__(self, *script: ScriptedWs | BaseException) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sockets: list[ScriptedWs] = []

    async def __call__(self, url: str, **kwargs: Any) -> ScriptedWs:
        self.calls.append((url, kwargs))
        item = self.script.pop(0) if self.script else ScriptedWs()
        if isinstance(item, BaseException):
            raise item
        self.sockets.append(item)
        return item


def joined(src: TwitchAnonIrc) -> str:
    return f":{src.nick}!{src.nick}@{src.nick}.tmi.twitch.tv JOIN #{CHANNEL}"


def privmsg(msg_id: str, text: str, *, sent_ms: int, user: str = "viewer") -> str:
    return (
        f"@badges=;display-name={user};id={msg_id};room-id=12345;tmi-sent-ts={sent_ms};"
        f"user-id=u-{user} :{user}!{user}@{user}.tmi.twitch.tv PRIVMSG #{CHANNEL} :{text}"
    )


# --- tests: real localhost socket ------------------------------------------------------------
async def test_reads_the_recorded_lines_over_a_real_socket(fake_clock: FakeClock) -> None:
    async with FakeIrcServer() as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url)
        assert isinstance(src, ChatSource)
        assert src.health().state is HealthState.STARTING
        reader = Reader(src)
        await until(lambda: len(reader.got) >= len(UNIQUE_IDS), what="all sample messages")
        await asyncio.sleep(0.05)
        assert reader.ids == UNIQUE_IDS  # duplicate id dropped, shared chat dropped
        assert src.duplicates == 1 and src.dropped_shared == 1
        kinds = {m.id[-2:]: m.kind for m in reader.got}
        assert kinds["05"] is MsgKind.DONATION and kinds["10"] is MsgKind.RAID
        assert kinds["07"] is kinds["08"] is MsgKind.SUB and kinds["09"] is MsgKind.GIFT_SUB
        first = reader.got[0]
        assert first.ts == pytest.approx(fake_clock.now() - 0.0, abs=1.0)  # wall matches ts
        assert re.fullmatch(r"NICK justinfan\d{5}", server.received[2])
        assert server.received[:4] == [
            "CAP REQ :twitch.tv/tags twitch.tv/commands",
            "PASS SCHMOOPIIE",
            server.received[2],
            f"JOIN #{CHANNEL}",
        ]
        health = src.health()
        assert health.state is HealthState.OK and health.component == "chat:twitch"
        assert src.room_id == "12345"
        await reader.stop()
        assert src.health().state is HealthState.DOWN


async def test_server_ping_gets_pong(fake_clock: FakeClock) -> None:
    async with FakeIrcServer(lines=[]) as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url)
        reader = Reader(src)
        await until(lambda: src.health().state is HealthState.OK, what="joined")
        await server.ping()
        await until(lambda: server.pongs == 1, what="PONG")
        assert "PONG :tmi.twitch.tv" in server.received and src.pings_answered == 1
        await reader.stop()


async def test_reconnect_command_reconnects_immediately(fake_clock: FakeClock) -> None:
    async with FakeIrcServer(lines=SAMPLE_IRC_LINES[:3]) as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url)
        reader = Reader(src)
        await until(lambda: len(reader.got) == 3, what="first batch")
        await server.send_reconnect()
        await until(lambda: server.connections_total == 2, what="reconnect")
        assert src.last_backoff == 0.0 and fake_clock.now() == 1000.0  # no backoff sleep
        await until(lambda: src.health().state is HealthState.OK)
        await asyncio.sleep(0.05)
        assert len(reader.got) == 3 and src.duplicates == 3  # replayed lines deduped by id
        await reader.stop()


async def test_disconnect_reconnects_with_backoff_and_dedupes(fake_clock: FakeClock) -> None:
    async with FakeIrcServer(disconnect_after=4) as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url)
        reader = Reader(src)
        await until(lambda: src.last_backoff is not None, what="backoff after the drop")
        assert len(reader.got) == 4
        assert src.last_backoff is not None and 0.8 <= src.last_backoff <= 1.2
        assert src.health().state is HealthState.DEGRADED
        assert "reconnecting" in src.health().detail
        assert server.connections_total == 1
        fake_clock.advance(src.last_backoff)
        await until(lambda: len(reader.got) >= len(UNIQUE_IDS), what="the rest after reconnect")
        await asyncio.sleep(0.05)
        assert reader.ids == UNIQUE_IDS and server.connections_total == 2
        assert src.health().state is HealthState.OK
        await reader.stop()


async def test_backlog_is_skipped_after_a_reconnect(fake_clock: FakeClock) -> None:
    base = 1_760_000_000_000
    lines = [privmsg(f"b-{i}", f"ข้อความ {i}", sent_ms=base + i * 1000) for i in range(3)]
    async with FakeIrcServer(lines=lines, disconnect_after=3) as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url)
        reader = Reader(src)
        await until(lambda: src.last_backoff is not None)
        fake_clock.advance(src.last_backoff or 0.0)
        await until(lambda: server.connections_total == 2 and src.duplicates == 3)
        await server.push(privmsg("missed-old", "เก่า", sent_ms=base + 1500))
        await server.push(privmsg("new-1", "ใหม่", sent_ms=base + 5000))
        await until(lambda: "new-1" in reader.ids)
        assert "missed-old" not in reader.ids and src.skipped_backlog == 1
        assert reader.ids == ["b-0", "b-1", "b-2", "new-1"]
        await reader.stop()


async def test_include_shared_keeps_shared_chat(fake_clock: FakeClock) -> None:
    async with FakeIrcServer(lines=SAMPLE_IRC_LINES[10:11]) as server:
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=server.url, include_shared=True)
        reader = Reader(src)
        await until(lambda: len(reader.got) == 1)
        assert reader.got[0].source_channel == "99999"
        await reader.stop()


async def test_other_channels_are_ignored(fake_clock: FakeClock) -> None:
    async with FakeIrcServer(lines=[]) as server:
        src = TwitchAnonIrc(f"#{CHANNEL.upper()}", clock=fake_clock, url=server.url)
        assert src.channel == CHANNEL
        reader = Reader(src)
        await until(lambda: src.health().state is HealthState.OK)
        await server.push(privmsg("x-1", "hi", sent_ms=1).replace(f"#{CHANNEL}", "#other"))
        await server.push(privmsg("x-2", "hi", sent_ms=2))
        await until(lambda: len(reader.got) == 1)
        assert reader.ids == ["x-2"]
        await reader.stop()


# --- tests: scripted socket (fully deterministic) --------------------------------------------
async def test_client_pings_when_idle_and_reconnects_without_pong(fake_clock: FakeClock) -> None:
    ws1, ws2 = ScriptedWs(), ScriptedWs()
    conn = Connector(ws1, ws2)
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=conn)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    ws1.feed(joined(src))
    await fake_clock.run_until_idle()
    assert src.health().state is HealthState.OK
    await fake_clock.run_for(60.0)
    assert ws1.sent[-1] == "PING :tmi.twitch.tv"
    ws1.feed("PONG :tmi.twitch.tv")  # any traffic counts as proof of life
    await fake_clock.run_for(59.0)
    assert ws1.sent.count("PING :tmi.twitch.tv") == 1 and not ws1.closed
    await fake_clock.run_for(1.0)
    assert ws1.sent.count("PING :tmi.twitch.tv") == 2
    await fake_clock.run_for(10.0)  # no reply within pong_timeout_s
    assert ws1.closed and src.health().state is HealthState.DEGRADED
    assert "PING" in src.health().detail
    await fake_clock.run_for(1.3)
    assert len(conn.calls) == 2
    await reader.stop()


async def test_login_failure_backs_off(fake_clock: FakeClock) -> None:
    ws = ScriptedWs()
    conn = Connector(ws)
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=conn)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    ws.feed(":tmi.twitch.tv NOTICE * :Login authentication failed")
    await fake_clock.run_until_idle()
    assert ws.closed and "login refused" in src.health().detail
    assert src.last_backoff is not None and len(conn.calls) == 1
    await reader.stop()


async def test_connect_failures_back_off_exponentially(fake_clock: FakeClock) -> None:
    conn = Connector(*(OSError(f"refused {i}") for i in range(4)))
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=conn, jitter=0.0)
    reader = Reader(src)
    delays: list[float] = []
    for expected in (1.0, 2.0, 4.0):
        await fake_clock.run_until_idle()
        assert src.last_backoff == expected
        delays.append(expected)
        await fake_clock.run_for(expected)
    health = src.health()
    assert health.state is HealthState.DOWN and "refused" in health.detail
    assert len(conn.calls) == 4
    await reader.stop()
    assert reader.task.done() and not reader.got


async def test_scripted_messages_ping_and_reconnect(fake_clock: FakeClock) -> None:
    ws1, ws2 = ScriptedWs(), ScriptedWs()
    conn = Connector(ws1, ws2)
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=conn, nick="justinfan12345")
    reader = Reader(src)
    await fake_clock.run_until_idle()
    assert ws1.sent == [
        "CAP REQ :twitch.tv/tags twitch.tv/commands",
        "PASS SCHMOOPIIE",
        "NICK justinfan12345",
        f"JOIN #{CHANNEL}",
    ]
    ws1.feed(joined(src), "PING :tmi.twitch.tv", privmsg("s-1", "สวัสดี", sent_ms=1))
    await fake_clock.run_until_idle()
    assert "PONG :tmi.twitch.tv" in ws1.sent and reader.ids == ["s-1"]
    ws1.feed(":tmi.twitch.tv RECONNECT")
    await fake_clock.run_until_idle()
    assert ws1.closed and len(conn.calls) == 2 and src.last_backoff == 0.0
    ws2.feed(joined(src), privmsg("s-1", "สวัสดี", sent_ms=1), privmsg("s-2", "ใหม่", sent_ms=2))
    await fake_clock.run_until_idle()
    assert reader.ids == ["s-1", "s-2"] and src.connects == 2
    await reader.stop()


async def test_health_degrades_after_five_minutes_without_traffic(fake_clock: FakeClock) -> None:
    ws = ScriptedWs()
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=Connector(ws), idle_ping_s=10_000)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    ws.feed(joined(src))
    await fake_clock.run_until_idle()
    fake_clock.advance(299.0)
    assert src.health().state is HealthState.OK
    fake_clock.advance(2.0)
    health = src.health()
    assert health.state is HealthState.DEGRADED and "no PING" in health.detail
    ws.feed("PING :tmi.twitch.tv")
    await fake_clock.run_until_idle()
    assert src.health().state is HealthState.OK
    await reader.stop()


async def test_loopback_urls_disable_the_proxy(fake_clock: FakeClock) -> None:
    for url, expect_proxy_kw in (
        ("ws://127.0.0.1:1234", True),
        ("ws://localhost:1234", True),
        ("ws://[::1]:1234", True),
        ("wss://irc-ws.chat.twitch.tv:443", False),
    ):
        conn = Connector()
        src = TwitchAnonIrc(CHANNEL, clock=fake_clock, url=url, connect=conn)
        reader = Reader(src)
        await fake_clock.run_until_idle()
        assert conn.calls[0][0] == url
        if expect_proxy_kw:
            assert conn.calls[0][1] == {"proxy": None}
        else:
            assert "proxy" not in conn.calls[0][1]  # websockets' default proxy discovery
        await reader.stop()


async def test_aclose_is_idempotent_and_ends_a_backoff_sleep(fake_clock: FakeClock) -> None:
    conn = Connector(OSError("down"))
    src = TwitchAnonIrc(CHANNEL, clock=fake_clock, connect=conn)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    assert src.last_backoff is not None
    await src.aclose()
    await src.aclose()
    await asyncio.wait_for(reader.task, 1.0)
    assert src.health().state is HealthState.DOWN
    assert [m async for m in src.messages()] == []


async def test_constructor_and_from_config() -> None:
    with pytest.raises(ValueError):
        TwitchAnonIrc("  #  ", clock=SystemClock())
    from aivtube.config.schema import TwitchIrcConfig

    src = TwitchAnonIrc.from_config(TwitchIrcConfig(channel="Pailin_TH"), clock=SystemClock())
    assert src.channel == "pailin_th" and src.url == "wss://irc-ws.chat.twitch.tv:443"
    assert re.fullmatch(r"justinfan\d{5}", src.nick)
    await src.aclose()


@pytest.mark.network
async def test_live_anonymous_login_to_twitch() -> None:
    """Joins a real channel anonymously (AIVTUBE_NETWORK=1); does not wait for chat."""
    src = TwitchAnonIrc("twitch", clock=SystemClock())
    reader = Reader(src)
    try:
        await until(lambda: src.health().state is HealthState.OK, timeout=20.0, what="JOIN")
    finally:
        await reader.stop()
