"""FakeIrcServer: anonymous Twitch login, replay, PING/PONG, scripted disconnects, RECONNECT."""

from __future__ import annotations

import asyncio
from pathlib import Path

from websockets.asyncio.client import ClientConnection, connect

from aivtube.testing.fakes import SAMPLE_IRC_LINES, FakeIrcServer


async def _login(url: str, channel: str = "pailin_th") -> ClientConnection:
    ws = await connect(url, proxy=None)
    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
    await ws.send("PASS SCHMOOPIIE")
    await ws.send("NICK justinfan12345")
    await ws.send(f"JOIN #{channel}")
    return ws


async def _lines(ws: ClientConnection, until: str, limit: int = 200) -> list[str]:
    out: list[str] = []
    while len(out) < limit:
        frame = await asyncio.wait_for(ws.recv(), 2.0)
        assert isinstance(frame, str) and frame.endswith("\r\n")
        for line in frame.split("\r\n"):
            if line:
                out.append(line)
                if until in line:
                    return out
    raise AssertionError(f"{until!r} not seen")


async def test_login_and_replay() -> None:
    async with FakeIrcServer(lines=SAMPLE_IRC_LINES) as server:
        ws = await _login(server.url)
        lines = await _lines(ws, "8c0f4d2e-0013")
        assert ":tmi.twitch.tv CAP * ACK :twitch.tv/tags twitch.tv/commands" in lines
        assert any(" 001 justinfan12345 " in x for x in lines)
        assert any("ROOMSTATE #pailin_th" in x for x in lines)
        privmsgs = [x for x in lines if " PRIVMSG #pailin_th " in x]
        assert len(privmsgs) == 9 and not any("{channel}" in x for x in lines)
        assert server.joined == ["pailin_th"] and "PASS SCHMOOPIIE" in server.received
        await ws.close()


async def test_ping_pong_both_ways() -> None:
    async with FakeIrcServer(lines=[]) as server:
        ws = await _login(server.url)
        await _lines(ws, "ROOMSTATE")
        await ws.send("PING :tmi.twitch.tv")
        assert await _lines(ws, "PONG") == ["PONG :tmi.twitch.tv"]
        await server.ping()
        assert await _lines(ws, "PING") == ["PING :tmi.twitch.tv"]
        await ws.send("PONG :tmi.twitch.tv")
        await asyncio.sleep(0.05)
        assert server.pongs == 1
        await ws.close()


async def test_disconnect_after_then_full_replay() -> None:
    async with FakeIrcServer(lines=SAMPLE_IRC_LINES, disconnect_after=3) as server:
        ws = await _login(server.url)
        first = await _lines(ws, "8c0f4d2e-0003")
        assert sum(" PRIVMSG " in x for x in first) == 3
        await asyncio.wait_for(ws.wait_closed(), 2.0)
        # the reconnect replays everything (dedupe is the client's job)
        ws2 = await _login(server.url)
        again = await _lines(ws2, "8c0f4d2e-0013")
        assert any("8c0f4d2e-0001" in x for x in again)
        assert server.connections_total == 2 and server.disconnects_done == 1
        await server.send_reconnect()
        assert await _lines(ws2, "RECONNECT") == [":tmi.twitch.tv RECONNECT"]
        await ws2.close()


async def test_live_push_and_drop() -> None:
    async with FakeIrcServer(lines=[]) as server:
        ws = await _login(server.url, "somechan")
        await _lines(ws, "ROOMSTATE")
        await server.push("@id=x :a!a@a.tmi.twitch.tv PRIVMSG #{channel} :สด ๆ")
        assert await _lines(ws, "PRIVMSG") == ["@id=x :a!a@a.tmi.twitch.tv PRIVMSG #somechan :สด ๆ"]
        await server.drop()
        await asyncio.wait_for(ws.wait_closed(), 2.0)


def test_fixture_file_matches_sample_lines(fixtures_dir: Path) -> None:
    text = (fixtures_dir / "irc" / "twitch_sample.txt").read_text(encoding="utf-8")
    assert text.splitlines() == list(SAMPLE_IRC_LINES)
