"""``FakeIrcServer``: Twitch chat IRC over a local WebSocket (§3.8, §10).

It speaks enough of ``irc-ws.chat.twitch.tv`` for the anonymous reader: CAP ACK, the 001-376
welcome burst, JOIN/NAMES/ROOMSTATE, then replays recorded lines (``{channel}`` is replaced by
the joined channel). It answers client PINGs, sends its own PINGs, can drop the connection
after N lines (the next connection replays everything, so the client must dedupe by id) and
can send RECONNECT. Twitch packs several ``\\r\\n``-terminated lines into one frame; so does this.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

__all__ = ["SAMPLE_IRC_LINES", "FakeIrcServer"]

_TS = 1_760_000_000_000

SAMPLE_IRC_LINES: tuple[str, ...] = (
    # 0: a first-time chatter
    "@badge-info=;badges=;color=#1E90FF;display-name=Tom123;emotes=;first-msg=1;flags=;"
    f"id=8c0f4d2e-0001-4000-8000-000000000001;mod=0;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS};turbo=0;user-id=1001;user-type= "
    ":tom123!tom123@tom123.tmi.twitch.tv PRIVMSG #{channel} :สวัสดีครับไพลิน วันนี้เล่นเกมอะไร",
    # 1: a 14-month subscriber
    "@badge-info=subscriber/14;badges=subscriber/12,premium/1;color=#FF69B4;display-name=มะลิ;"
    "emotes=;first-msg=0;flags=;id=8c0f4d2e-0002-4000-8000-000000000002;mod=0;"
    f"returning-chatter=0;room-id=12345;subscriber=1;tmi-sent-ts={_TS + 1500};turbo=0;"
    "user-id=1002;user-type= :mali_th!mali_th@mali_th.tmi.twitch.tv PRIVMSG #{channel} "
    ":ไพลินน่ารักมาก 555",
    # 2: a moderator
    "@badge-info=;badges=moderator/1;color=#00FF7F;display-name=ModBoss;emotes=;first-msg=0;"
    "flags=;id=8c0f4d2e-0003-4000-8000-000000000003;mod=1;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS + 3000};turbo=0;user-id=1003;user-type=mod "
    ":modboss!modboss@modboss.tmi.twitch.tv PRIVMSG #{channel} :ทุกคนอย่าลืมกดติดตามนะ",
    # 3: a VIP
    "@badge-info=;badges=vip/1;color=;display-name=VipFan;emotes=;first-msg=0;flags=;"
    "id=8c0f4d2e-0004-4000-8000-000000000004;mod=0;returning-chatter=1;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS + 4500};turbo=0;user-id=1004;user-type=;vip=1 "
    ":vipfan!vipfan@vipfan.tmi.twitch.tv PRIVMSG #{channel} :เพลงนี้ชื่ออะไรครับ?",
    # 4: a cheer (bits=100 -> DONATION, 1.00 USD)
    "@badge-info=;badges=bits/100;bits=100;color=;display-name=Cheerer;emotes=;first-msg=0;"
    "flags=;id=8c0f4d2e-0005-4000-8000-000000000005;mod=0;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS + 6000};turbo=0;user-id=1005;user-type= "
    ":cheerer!cheerer@cheerer.tmi.twitch.tv PRIVMSG #{channel} :Cheer100 ให้กำลังใจไพลินค่ะ",
    # 5: a mention
    "@badge-info=;badges=;color=;display-name=Asker;emotes=;first-msg=0;flags=;"
    "id=8c0f4d2e-0006-4000-8000-000000000006;mod=0;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS + 7500};turbo=0;user-id=1006;user-type= "
    ":asker!asker@asker.tmi.twitch.tv PRIVMSG #{channel} :@pailin ชอบกินอะไรที่สุด",
    # 6: a new sub (USERNOTICE sub)
    "@badge-info=subscriber/1;badges=subscriber/0;color=;display-name=NewSub;emotes=;flags=;"
    "id=8c0f4d2e-0007-4000-8000-000000000007;login=newsub;mod=0;msg-id=sub;"
    "msg-param-cumulative-months=1;msg-param-months=0;msg-param-multimonth-duration=1;"
    "msg-param-multimonth-tenure=0;msg-param-should-share-streak=0;"
    "msg-param-sub-plan-name=Channel\\sSubscription;msg-param-sub-plan=1000;"
    "msg-param-was-gifted=false;room-id=12345;subscriber=1;"
    f"system-msg=NewSub\\ssubscribed\\sat\\sTier\\s1.;tmi-sent-ts={_TS + 9000};user-id=2001;"
    "user-type= :tmi.twitch.tv USERNOTICE #{channel}",
    # 7: a resub with a message
    "@badge-info=subscriber/6;badges=subscriber/6;color=;display-name=Loyal;emotes=;flags=;"
    "id=8c0f4d2e-0008-4000-8000-000000000008;login=loyal;mod=0;msg-id=resub;"
    "msg-param-cumulative-months=6;msg-param-months=0;msg-param-multimonth-duration=0;"
    "msg-param-multimonth-tenure=0;msg-param-should-share-streak=0;msg-param-sub-plan=1000;"
    "msg-param-sub-plan-name=Channel\\sSubscription;msg-param-was-gifted=false;room-id=12345;"
    f"subscriber=1;system-msg=Loyal\\ssubscribed\\sfor\\s6\\smonths!;tmi-sent-ts={_TS + 10500};"
    "user-id=2002;user-type= :tmi.twitch.tv USERNOTICE #{channel} :ครบหกเดือนแล้วนะไพลิน",
    # 8: a gifted sub
    "@badge-info=;badges=sub-gifter/5;color=;display-name=Santa;emotes=;flags=;"
    "id=8c0f4d2e-0009-4000-8000-000000000009;login=santa;mod=0;msg-id=subgift;"
    "msg-param-gift-months=1;msg-param-months=1;msg-param-origin-id=abc;"
    "msg-param-recipient-display-name=LuckyViewer;msg-param-recipient-id=3003;"
    "msg-param-recipient-user-name=luckyviewer;msg-param-sender-count=5;"
    "msg-param-sub-plan-name=Channel\\sSubscription;msg-param-sub-plan=1000;room-id=12345;"
    "subscriber=0;system-msg=Santa\\sgifted\\sa\\sTier\\s1\\ssub\\sto\\sLuckyViewer!;"
    f"tmi-sent-ts={_TS + 12000};user-id=2003;user-type= :tmi.twitch.tv USERNOTICE #{{channel}}",
    # 9: a raid
    "@badge-info=;badges=;color=;display-name=RaiderCh;emotes=;flags=;"
    "id=8c0f4d2e-0010-4000-8000-000000000010;login=raiderch;mod=0;msg-id=raid;"
    "msg-param-displayName=RaiderCh;msg-param-login=raiderch;"
    "msg-param-profileImageURL=https://example.invalid/raider.png;msg-param-viewerCount=42;"
    "room-id=12345;subscriber=0;system-msg=42\\sraiders\\sfrom\\sRaiderCh\\shave\\sjoined!;"
    f"tmi-sent-ts={_TS + 13500};user-id=2004;user-type= :tmi.twitch.tv USERNOTICE #{{channel}}",
    # 10: shared chat from another channel (source-room-id differs -> must be dropped)
    "@badge-info=;badges=;color=;display-name=Elsewhere;emotes=;first-msg=0;flags=;"
    "id=8c0f4d2e-0011-4000-8000-000000000011;mod=0;returning-chatter=0;room-id=12345;"
    "source-badge-info=;source-badges=;source-id=8c0f4d2e-0011-4000-8000-0000000000ff;"
    f"source-room-id=99999;subscriber=0;tmi-sent-ts={_TS + 15000};turbo=0;user-id=1011;"
    "user-type= :elsewhere!elsewhere@elsewhere.tmi.twitch.tv PRIVMSG #{channel} "
    ":ข้อความจากอีกห้อง",
    # 11: a duplicate of line 0 (same id; dedupe)
    "@badge-info=;badges=;color=#1E90FF;display-name=Tom123;emotes=;first-msg=1;flags=;"
    f"id=8c0f4d2e-0001-4000-8000-000000000001;mod=0;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS};turbo=0;user-id=1001;user-type= "
    ":tom123!tom123@tom123.tmi.twitch.tv PRIVMSG #{channel} :สวัสดีครับไพลิน วันนี้เล่นเกมอะไร",
    # 12: the broadcaster
    "@badge-info=;badges=broadcaster/1;color=;display-name=Streamer;emotes=;first-msg=0;flags=;"
    "id=8c0f4d2e-0013-4000-8000-000000000013;mod=0;returning-chatter=0;room-id=12345;"
    f"subscriber=0;tmi-sent-ts={_TS + 16500};turbo=0;user-id=12345;user-type= "
    ":streamer!streamer@streamer.tmi.twitch.tv PRIVMSG #{channel} :ไพลิน ทักทายทุกคนหน่อย",
)
"""Synthetic but realistically tagged Twitch IRC lines (``{channel}`` placeholder, room 12345)."""


@dataclass(eq=False)
class _Client:
    ws: ServerConnection
    nick: str = ""
    channels: list[str] = field(default_factory=list)
    replaying: asyncio.Task[None] | None = None


class FakeIrcServer:
    """Twitch IRC stand-in. ``start()`` returns ``ws://127.0.0.1:<port>``.

    ``lines`` are replayed after each JOIN (``line_delay_s`` apart). ``disconnect_after`` closes
    the first ``max_disconnects`` connections after that many lines. ``ping_interval_s``
    enables server PINGs. Instrumentation: ``received`` (client lines), ``joined``, ``pongs``,
    ``connections_total``.
    """

    def __init__(
        self,
        port: int = 0,
        lines: Sequence[str] = SAMPLE_IRC_LINES,
        *,
        host: str = "127.0.0.1",
        line_delay_s: float = 0.0,
        ping_interval_s: float | None = None,
        disconnect_after: int | None = None,
        max_disconnects: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.lines = list(lines)
        self.line_delay_s = line_delay_s
        self.ping_interval_s = ping_interval_s
        self.disconnect_after = disconnect_after
        self.max_disconnects = max_disconnects
        self.received: list[str] = []
        self.joined: list[str] = []
        self.pongs = 0
        self.pings_sent = 0
        self.connections_total = 0
        self.disconnects_done = 0
        self._clients: set[_Client] = set()
        self._server: Server | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def start(self) -> str:
        if self._server is None:
            self._server = await serve(
                self._handler, self.host, self.port, compression=None, ping_interval=None
            )
            sock = next(iter(self._server.sockets))
            self.port = int(sock.getsockname()[1])
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            server.close()
            await server.wait_closed()

    async def __aenter__(self) -> FakeIrcServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    async def push(self, line: str) -> None:
        """Send one live line to every joined client (``{channel}`` substituted)."""
        for c in list(self._clients):
            for ch in c.channels:
                await self._send(c, [line.replace("{channel}", ch)])

    async def send_reconnect(self) -> None:
        for c in list(self._clients):
            await self._send(c, [":tmi.twitch.tv RECONNECT"])

    async def ping(self) -> None:
        for c in list(self._clients):
            self.pings_sent += 1
            await self._send(c, ["PING :tmi.twitch.tv"])

    async def drop(self) -> None:
        """Close every client connection abruptly-ish (code 1006 is not sendable; use 1001)."""
        for c in list(self._clients):
            await c.ws.close(1001, "server going away")

    async def _send(self, c: _Client, lines: Sequence[str]) -> None:
        with contextlib.suppress(ConnectionClosed):
            await c.ws.send("".join(f"{line}\r\n" for line in lines))

    async def _handler(self, ws: ServerConnection) -> None:
        client = _Client(ws)
        self._clients.add(client)
        self.connections_total += 1
        pinger = (
            asyncio.get_running_loop().create_task(self._pinger(client))
            if self.ping_interval_s
            else None
        )
        try:
            async for raw in ws:
                text = raw.decode() if isinstance(raw, bytes) else raw
                for line in text.replace("\r\n", "\n").split("\n"):
                    if line:
                        await self._on_line(client, line)
        except ConnectionClosed:
            pass
        finally:
            for task in (pinger, client.replaying):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self._clients.discard(client)

    async def _pinger(self, client: _Client) -> None:
        assert self.ping_interval_s
        while True:
            await asyncio.sleep(self.ping_interval_s)
            self.pings_sent += 1
            await self._send(client, ["PING :tmi.twitch.tv"])

    async def _on_line(self, c: _Client, line: str) -> None:
        self.received.append(line)
        cmd, _, rest = line.partition(" ")
        cmd = cmd.upper()
        if cmd == "CAP":
            caps = rest.partition(":")[2]
            await self._send(c, [f":tmi.twitch.tv CAP * ACK :{caps}"])
        elif cmd == "NICK":
            c.nick = rest.strip()
            n = c.nick
            await self._send(
                c,
                [
                    f":tmi.twitch.tv 001 {n} :Welcome, GLHF!",
                    f":tmi.twitch.tv 002 {n} :Your host is tmi.twitch.tv",
                    f":tmi.twitch.tv 003 {n} :This server is rather new",
                    f":tmi.twitch.tv 004 {n} :-",
                    f":tmi.twitch.tv 375 {n} :-",
                    f":tmi.twitch.tv 372 {n} :You are in a maze of twisty passages, all alike.",
                    f":tmi.twitch.tv 376 {n} :>",
                ],
            )
        elif cmd == "JOIN":
            for ch in (x.strip().lstrip("#") for x in rest.split(",")):
                if not ch:
                    continue
                c.channels.append(ch)
                self.joined.append(ch)
                n = c.nick or "justinfan12345"
                await self._send(
                    c,
                    [
                        f":{n}!{n}@{n}.tmi.twitch.tv JOIN #{ch}",
                        f":{n}.tmi.twitch.tv 353 {n} = #{ch} :{n}",
                        f":{n}.tmi.twitch.tv 366 {n} #{ch} :End of /NAMES list",
                        "@emote-only=0;followers-only=-1;r9k=0;room-id=12345;slow=0;subs-only=0 "
                        f":tmi.twitch.tv ROOMSTATE #{ch}",
                    ],
                )
                if c.replaying is None:
                    c.replaying = asyncio.get_running_loop().create_task(self._replay(c, ch))
        elif cmd == "PING":
            await self._send(c, [f"PONG {rest}" if rest else "PONG :tmi.twitch.tv"])
        elif cmd == "PONG":
            self.pongs += 1

    async def _replay(self, c: _Client, channel: str) -> None:
        cut = (
            self.disconnect_after
            if self.disconnect_after is not None and self.disconnects_done < self.max_disconnects
            else None
        )
        for i, line in enumerate(self.lines):
            if cut is not None and i >= cut:
                self.disconnects_done += 1
                await c.ws.close(1001, "scripted disconnect")
                return
            if self.line_delay_s:
                await asyncio.sleep(self.line_delay_s)
            await self._send(c, [line.replace("{channel}", channel)])
        if cut is not None and cut >= len(self.lines):
            self.disconnects_done += 1
            await c.ws.close(1001, "scripted disconnect")
