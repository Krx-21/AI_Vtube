"""``TwitchAnonIrc``: zero-config, read-only anonymous Twitch chat (ARCHITECTURE.md §3.8).

Connects to ``wss://irc-ws.chat.twitch.tv:443`` as ``justinfan<5 digits>`` (verified live on
2026-09-25; IRC is not deprecated, only the non-TLS WebSocket was retired), requests the
``twitch.tv/tags`` and ``twitch.tv/commands`` capabilities and joins one channel.

- ``PRIVMSG`` with IRCv3 tags becomes a ``ChatMessage``: badges map to broadcaster/mod/VIP/sub
  (with sub months from ``badge-info``), ``bits`` to a DONATION (0.01 USD per bit),
  ``first-msg``, ``reply-parent-msg-id`` and ``custom-reward-id`` (a REDEEM).
- ``USERNOTICE`` sub/resub/subgift/submysterygift/raid (and paid upgrades) become
  SUB/GIFT_SUB/RAID.
- Shared-chat messages from another channel (``source-room-id`` ≠ ``room-id``) are dropped.
- The server's ``PING`` gets a ``PONG``; after 60 s of silence we ``PING`` the server and
  reconnect if nothing comes back within 10 s. ``RECONNECT`` reconnects at once; any other
  disconnect reconnects with exponential backoff and jitter.
- Messages are deduped by id (the server may resend), and after a reconnect anything sent
  before the newest message we had already delivered is skipped as backlog.

Every network await has a deadline (I2) measured on the injected ``Clock``. For loopback test
servers the WebSocket proxy is disabled; real hosts use websockets' default proxy discovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import random
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from aivtube.chat._util import sleep_unless_set
from aivtube.contracts.infra import Clock
from aivtube.contracts.types import ChatMessage, ChatUser, Health, HealthState, MsgKind, Platform
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.infra.tasks import backoff_delay

__all__ = [
    "DEFAULT_URL",
    "SUB_PLAN_USD",
    "USD_PER_BIT",
    "IrcLine",
    "TwitchAnonIrc",
    "map_platform_ts",
    "message_from_irc",
    "parse_irc",
    "parse_irc_line",
    "unescape_tag",
]

log = logging.getLogger("aivtube.chat.twitch_irc")

DEFAULT_URL = "wss://irc-ws.chat.twitch.tv:443"
USD_PER_BIT = 0.01
SUB_PLAN_USD: Mapping[str, float] = {"Prime": 4.99, "1000": 4.99, "2000": 9.99, "3000": 24.99}
"""Nominal sub prices, used only to rank must-acknowledge messages."""

MAX_MAPPED_DELAY_S = 30.0
"""Platform timestamps further than this from our wall clock are treated as skew, not delay."""

_ESCAPES = {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}
_GIFT_KINDS = frozenset({"subgift", "anonsubgift", "submysterygift", "anonsubmysterygift"})
_SUB_KINDS = frozenset(
    {"sub", "resub", "giftpaidupgrade", "anongiftpaidupgrade", "primepaidupgrade"}
)


# --- parsing ---------------------------------------------------------------------------------
def unescape_tag(value: str) -> str:
    """Unescape an IRCv3 tag value (``\\s`` → space, ``\\:`` → ``;`` …)."""
    if "\\" not in value:
        return value
    out: list[str] = []
    i, n = 0, len(value)
    while i < n:
        ch = value[i]
        if ch == "\\":
            if i + 1 < n:
                nxt = value[i + 1]
                out.append(_ESCAPES.get(nxt, nxt))
            i += 2  # a trailing lone backslash is dropped
        else:
            out.append(ch)
            i += 1
    return "".join(out)


@dataclass(frozen=True, slots=True)
class IrcLine:
    """One parsed IRC line: IRCv3 ``tags``, ``prefix``, upper-case ``command`` and ``params``
    (the trailing parameter, if any, is the last element)."""

    tags: Mapping[str, str] = field(default_factory=dict)
    prefix: str = ""
    command: str = ""
    params: tuple[str, ...] = ()

    @property
    def trailing(self) -> str:
        return self.params[-1] if self.params else ""

    @property
    def nick(self) -> str:
        return self.prefix.split("!", 1)[0]


def parse_irc(line: str) -> IrcLine | None:
    """Parse one raw IRC line; ``None`` for blank or malformed input (never raises)."""
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    tags: dict[str, str] = {}
    if line.startswith("@"):
        raw_tags, _, line = line[1:].partition(" ")
        for item in raw_tags.split(";"):
            if item:
                k, _, v = item.partition("=")
                tags[k] = unescape_tag(v)
    line = line.lstrip(" ")
    prefix = ""
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
        line = line.lstrip(" ")
    head, sep, trailing = line.partition(" :")
    if not sep and head.startswith(":"):  # command with only a trailing parameter
        head, trailing, sep = "", head[1:], " :"
    parts = head.split()
    if not parts:
        return None
    params = tuple(parts[1:]) + ((trailing,) if sep else ())
    return IrcLine(tags=tags, prefix=prefix, command=parts[0].upper(), params=params)


def map_platform_ts(
    sent_epoch_s: float | None, *, received: float, wall_now: float | None
) -> float:
    """Map a platform wall-clock timestamp to our ``perf_counter`` timebase.

    ``ts = received - (wall_now - sent)`` when that delay is plausible (0..30 s); otherwise the
    platform clock and ours disagree (skew), and ``received`` is used.
    """
    if sent_epoch_s is None or wall_now is None:
        return received
    delay = wall_now - sent_epoch_s
    return received - delay if 0.0 <= delay <= MAX_MAPPED_DELAY_S else received


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _badges(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in value.split(","):
        name, sep, version = item.partition("/")
        if name and sep:
            out[name] = version
    return out


def _sent_s(tags: Mapping[str, str]) -> float | None:
    ms = _int(tags.get("tmi-sent-ts"), -1)
    return ms / 1000.0 if ms >= 0 else None


def _user(tags: Mapping[str, str], login: str) -> ChatUser:
    badges = _badges(tags.get("badges", ""))
    info = _badges(tags.get("badge-info", ""))
    months = _int(info.get("subscriber") or info.get("founder"))
    return ChatUser(
        platform=Platform.TWITCH,
        id=tags.get("user-id") or login,
        name=tags.get("display-name") or login,
        is_broadcaster="broadcaster" in badges,
        is_mod="moderator" in badges or tags.get("mod") == "1",
        is_vip="vip" in badges or "vip" in tags,
        is_sub="subscriber" in badges or "founder" in badges or tags.get("subscriber") == "1",
        sub_months=months,
        is_verified="partner" in badges,
    )


def _shared_from(tags: Mapping[str, str]) -> str | None:
    src = tags.get("source-room-id")
    return src if src and src != tags.get("room-id") else None


def _text(trailing: str) -> str:
    if trailing.startswith("\x01ACTION ") and trailing.endswith("\x01"):
        return trailing[8:-1]
    return trailing


def message_from_irc(
    msg: IrcLine, *, received: float, wall_now: float | None = None
) -> ChatMessage | None:
    """``PRIVMSG``/``USERNOTICE`` → ``ChatMessage``; anything else (or an unknown notice
    type) → ``None``."""
    tags = msg.tags
    ts = map_platform_ts(_sent_s(tags), received=received, wall_now=wall_now)
    if msg.command == "PRIVMSG" and len(msg.params) >= 2:
        login = msg.nick
        user = _user(tags, login)
        bits = _int(tags.get("bits"))
        kind = MsgKind.TEXT
        if bits > 0:
            kind = MsgKind.DONATION
        elif tags.get("custom-reward-id"):
            kind = MsgKind.REDEEM
        return ChatMessage(
            platform=Platform.TWITCH,
            id=tags.get("id") or f"{tags.get('room-id', '')}:{user.id}:{tags.get('tmi-sent-ts')}",
            user=user,
            text=_text(msg.trailing),
            ts=ts,
            received=received,
            kind=kind,
            amount=float(bits),
            currency="BITS" if bits else "",
            value_usd=bits * USD_PER_BIT,
            first_msg=tags.get("first-msg") == "1",
            reply_to=tags.get("reply-parent-msg-id") or None,
            source_channel=_shared_from(tags),
            raw=dict(tags),
        )
    if msg.command == "USERNOTICE" and msg.params:
        notice = tags.get("msg-id", "")
        if notice in _SUB_KINDS:
            kind, amount = MsgKind.SUB, 1.0
        elif notice in _GIFT_KINDS:
            if notice in ("subgift", "anonsubgift") and tags.get("msg-param-community-gift-id"):
                return None  # one of a mystery-gift batch; the submysterygift notice counts it
            kind = MsgKind.GIFT_SUB
            amount = float(_int(tags.get("msg-param-mass-gift-count"), 1))
        elif notice == "raid":
            kind, amount = MsgKind.RAID, float(_int(tags.get("msg-param-viewerCount")))
        else:
            return None
        login = tags.get("login") or msg.nick
        user = _user(tags, login)
        months = _int(tags.get("msg-param-cumulative-months"))
        if months and kind is MsgKind.SUB:
            user = ChatUser(
                platform=user.platform,
                id=user.id,
                name=user.name,
                is_broadcaster=user.is_broadcaster,
                is_mod=user.is_mod,
                is_vip=user.is_vip,
                is_sub=True,
                sub_months=months,
                is_verified=user.is_verified,
            )
        plan_usd = SUB_PLAN_USD.get(tags.get("msg-param-sub-plan", ""), 0.0)
        value = plan_usd * amount if kind is not MsgKind.RAID else 0.0
        return ChatMessage(
            platform=Platform.TWITCH,
            id=tags.get("id") or f"{tags.get('room-id', '')}:{notice}:{tags.get('tmi-sent-ts')}",
            user=user,
            text=_text(msg.params[1]) if len(msg.params) >= 2 else "",
            ts=ts,
            received=received,
            kind=kind,
            amount=amount,
            value_usd=value,
            source_channel=_shared_from(tags),
            raw=dict(tags),
        )
    return None


def parse_irc_line(
    line: str, *, received: float, wall_now: float | None = None
) -> ChatMessage | None:
    """Parse one raw Twitch IRC line into a ``ChatMessage`` (``None`` if it is not chat)."""
    parsed = parse_irc(line)
    return (
        None if parsed is None else message_from_irc(parsed, received=received, wall_now=wall_now)
    )


# --- the source ------------------------------------------------------------------------------
class _WebSocket(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


class _Reconnect(Exception):
    """The server asked us to reconnect (``RECONNECT``)."""


class _Dead(Exception):
    """The connection is unusable (no PONG, login refused)."""


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class TwitchAnonIrc:
    """``ChatSource`` for one Twitch channel over anonymous IRC (read-only).

    ``connect`` is ``websockets.asyncio.client.connect`` or a stand-in with the same call
    shape (``await connect(url, **kwargs)`` returning an object with ``send``/``recv``/
    ``close``). Instrumentation counters: ``connects``, ``duplicates``, ``skipped_backlog``,
    ``dropped_shared``, ``pings_answered``; ``last_backoff`` is the latest reconnect delay.
    """

    def __init__(
        self,
        channel: str,
        *,
        clock: Clock,
        url: str = DEFAULT_URL,
        connect: Callable[..., Any] = ws_connect,
        nick: str | None = None,
        backoff: tuple[float, float] = (1.0, 30.0),
        jitter: float = 0.2,
        connect_timeout_s: float = 15.0,
        idle_ping_s: float = 60.0,
        pong_timeout_s: float = 10.0,
        stale_after_s: float = 300.0,
        include_shared: bool = False,
        dedupe_lru: int = 5000,
        rng: random.Random | None = None,
    ) -> None:
        name = channel.strip().lstrip("#").lower()
        if not name:
            raise ValueError("a Twitch channel name is required")
        self.platform: Platform = Platform.TWITCH
        self.channel = name
        self.url = url
        self._clock = clock
        self._connect = connect
        self._rng = rng or random.Random()
        self.nick = nick or f"justinfan{self._rng.randint(10000, 99999)}"
        self.backoff = backoff
        self.jitter = jitter
        self.connect_timeout_s = connect_timeout_s
        self.idle_ping_s = idle_ping_s
        self.pong_timeout_s = pong_timeout_s
        self.stale_after_s = stale_after_s
        self.include_shared = include_shared
        self._dedupe_lru = dedupe_lru
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._newest_sent_ms = -1
        self._resume_after_ms: int | None = None
        self._ws: _WebSocket | None = None
        self._closed = False
        self._closed_event = asyncio.Event()
        self._state = HealthState.STARTING
        self._detail = f"connecting to #{name}"
        self._since = clock.now()
        self._last_traffic: float | None = None
        self.room_id: str | None = None
        self.connects = 0
        self.duplicates = 0
        self.skipped_backlog = 0
        self.dropped_shared = 0
        self.pings_answered = 0
        self.last_backoff: float | None = None

    @classmethod
    def from_config(cls, cfg: Any, *, clock: Clock, **kwargs: Any) -> TwitchAnonIrc:
        """Build from ``config.schema.TwitchIrcConfig`` (duck-typed ``channel`` and ``url``)."""
        return cls(cfg.channel, clock=clock, url=cfg.url, **kwargs)

    # --- ChatSource ------------------------------------------------------------------------
    def health(self) -> Health:
        component = f"chat:{self.platform.value}"
        if self._closed:
            return Health(component, HealthState.DOWN, "closed", self._since)
        if (
            self._state is HealthState.OK
            and self._last_traffic is not None
            and self._clock.now() - self._last_traffic > self.stale_after_s
        ):
            silent = self._clock.now() - self._last_traffic
            return Health(
                component,
                HealthState.DEGRADED,
                f"no PING or traffic from Twitch for {silent:.0f} s",
                self._last_traffic + self.stale_after_s,
            )
        return Health(component, self._state, self._detail, self._since)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_event.set()
        self._set_state(HealthState.DOWN, "closed")
        ws, self._ws = self._ws, None
        if ws is not None:
            await self._close_ws(ws)

    async def messages(self) -> AsyncIterator[ChatMessage]:
        failures = 0
        while not self._closed:
            ws: _WebSocket | None = None
            immediate = False
            joined = False
            try:
                ws = await self._open()
                if self._closed:
                    break
                awaiting_pong = False
                while True:
                    try:
                        timeout = self.pong_timeout_s if awaiting_pong else self.idle_ping_s
                        async with deadline(timeout, what="twitch irc recv", clock=self._clock):
                            frame = await ws.recv()
                    except DeadlineExceeded:
                        if awaiting_pong:
                            raise _Dead(
                                f"no reply to PING within {self.pong_timeout_s:g} s"
                            ) from None
                        await self._send(ws, "PING :tmi.twitch.tv")
                        awaiting_pong = True
                        continue
                    awaiting_pong = False
                    self._last_traffic = self._clock.now()
                    text = frame.decode("utf-8", "replace") if isinstance(frame, bytes) else frame
                    for raw in text.split("\r\n"):
                        line = parse_irc(raw)
                        if line is None:
                            continue
                        if await self._control(ws, line):
                            if not joined and self._state is HealthState.OK:
                                joined = True
                                failures = 0
                            continue
                        msg = self._accept(line)
                        if msg is not None:
                            yield msg
            except _Reconnect:
                immediate = True
                self._set_state(HealthState.DEGRADED, "server asked to reconnect")
            except (_Dead, OSError, TimeoutError, WebSocketException) as exc:
                if not self._closed:
                    self._on_failure(exc, failures + 1)
            finally:
                self._ws = None
                if ws is not None:
                    await self._close_ws(ws)
                if self._newest_sent_ms >= 0:
                    self._resume_after_ms = self._newest_sent_ms
            if self._closed:
                break
            if immediate:
                delay = 0.0
            else:
                failures += 1
                delay = backoff_delay(failures, self.backoff)
                delay *= 1.0 + self.jitter * (2.0 * self._rng.random() - 1.0)
            self.last_backoff = delay
            if not await self._sleep(delay):
                break

    # --- connection ------------------------------------------------------------------------
    async def _open(self) -> _WebSocket:
        kwargs: dict[str, Any] = {}
        if _is_loopback(self.url):
            kwargs["proxy"] = None
        async with deadline(self.connect_timeout_s, what="twitch irc connect", clock=self._clock):
            ws: _WebSocket = await self._connect(self.url, **kwargs)
            self._ws = ws
            self.connects += 1
            try:
                for out in (
                    "CAP REQ :twitch.tv/tags twitch.tv/commands",
                    "PASS SCHMOOPIIE",
                    f"NICK {self.nick}",
                    f"JOIN #{self.channel}",
                ):
                    await ws.send(out)
            except BaseException:
                self._ws = None
                await self._close_ws(ws)
                raise
        if self._closed:
            await self._close_ws(ws)
        self._last_traffic = self._clock.now()
        return ws

    async def _send(self, ws: _WebSocket, line: str) -> None:
        async with deadline(5.0, what="twitch irc send", clock=self._clock):
            await ws.send(line)

    async def _close_ws(self, ws: _WebSocket) -> None:
        with contextlib.suppress(Exception):
            async with deadline(2.0, what="twitch irc close", clock=self._clock):
                await ws.close()

    async def _sleep(self, seconds: float) -> bool:
        """Sleep on the clock unless ``aclose()`` happens first; False once closed."""
        if self._closed:
            return False
        return await sleep_unless_set(self._clock, seconds, self._closed_event)

    async def _control(self, ws: _WebSocket, line: IrcLine) -> bool:
        """Handle non-chat commands; True when ``line`` was consumed here."""
        cmd = line.command
        if cmd in ("PRIVMSG", "USERNOTICE"):
            return False
        if cmd == "PING":
            await self._send(ws, f"PONG :{line.trailing or 'tmi.twitch.tv'}")
            self.pings_answered += 1
        elif cmd == "RECONNECT":
            raise _Reconnect
        elif cmd == "ROOMSTATE" and line.tags.get("room-id"):
            self.room_id = line.tags["room-id"]
        elif cmd == "JOIN" and line.nick == self.nick:
            self._set_state(HealthState.OK, f"joined #{self.channel}")
        elif cmd == "NOTICE":
            text = line.trailing
            notice = line.tags.get("msg-id", "")
            if "login" in text.lower() and ("fail" in text.lower() or "improperly" in text.lower()):
                raise _Dead(f"login refused: {text}")
            if notice.startswith("msg_channel_suspended") or notice == "msg_banned":
                self._set_state(HealthState.DEGRADED, f"#{self.channel}: {text}")
            log.info("twitch notice for #%s: %s (%s)", self.channel, text, notice or "-")
        return True

    def _accept(self, line: IrcLine) -> ChatMessage | None:
        target = line.params[0].lstrip("#").lower() if line.params else ""
        if target and target != self.channel:
            return None
        msg = message_from_irc(line, received=self._clock.now(), wall_now=self._clock.wall())
        if msg is None:
            return None
        if msg.id in self._seen:
            self.duplicates += 1
            self._seen.move_to_end(msg.id)
            return None
        self._seen[msg.id] = None
        while len(self._seen) > self._dedupe_lru:
            self._seen.popitem(last=False)
        sent_ms = _int(line.tags.get("tmi-sent-ts"), -1)
        if self._resume_after_ms is not None and 0 <= sent_ms <= self._resume_after_ms:
            self.skipped_backlog += 1
            return None
        if sent_ms > self._newest_sent_ms:
            self._newest_sent_ms = sent_ms
        if msg.source_channel and not self.include_shared:
            self.dropped_shared += 1
            return None
        return msg

    # --- health ----------------------------------------------------------------------------
    def _set_state(self, state: HealthState, detail: str) -> None:
        if state is not self._state:
            self._since = self._clock.now()
        self._state = state
        self._detail = detail

    def _on_failure(self, exc: BaseException, streak: int) -> None:
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        state = HealthState.DOWN if streak >= 3 else HealthState.DEGRADED
        self._set_state(state, f"reconnecting to #{self.channel} ({reason})")
        log.warning("twitch irc #%s disconnected (%s); attempt %d", self.channel, reason, streak)
