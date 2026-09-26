"""The worker's IPC client: connect, ``hello``, run the link, reconnect (Appendix A).

``run()`` loops until cancelled: connect to the core (``proxy=None``, loopback only), send
``hello{role, pid, version, token, caps}``, wait for ``reply{status: "ok"}``, then run the
``Link`` until it ends and reconnect with backoff (``backoff`` min/max seconds, reset after a
successful hello). A rejected hello (bad token 4001, version 4002) is logged as an error and
retried at the maximum backoff: the launcher may be restarting the core with a new token.

``on_link_up`` fires after every successful hello and ``on_link_lost`` when that link ends;
the voice worker uses them for invariant I1 (finish the current segment, clear the queue,
stop listening, wait).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Final
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from aivtube.contracts import ipc
from aivtube.contracts.infra import Clock
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.infra.logging import add_secrets
from aivtube.infra.tasks import backoff_delay
from aivtube.ipc.link import (
    CLOSE_BAD_TOKEN,
    CLOSE_SHUTDOWN,
    CLOSE_VERSION,
    MAX_FRAME_BYTES,
    Handler,
    IpcLinkError,
    Link,
    run_handlers,
)
from aivtube.ipc.server import is_loopback_host

__all__ = ["HandshakeRejected", "IpcClient"]

log = logging.getLogger("aivtube.ipc.client")

_FATAL_CODES: Final = frozenset({CLOSE_BAD_TOKEN, CLOSE_VERSION})


class HandshakeRejected(ConnectionError):
    """The core refused our hello (``code`` is the websocket close code)."""

    def __init__(self, code: int | None, reason: str) -> None:
        super().__init__(f"hello rejected ({code}): {reason}")
        self.code = code
        self.reason = reason


def _noop() -> None:
    return None


class IpcClient:
    """The worker end of the IPC bus (see the module docstring)."""

    def __init__(
        self,
        url: str,
        token: str,
        role: str,
        clock: Clock,
        *,
        caps: Mapping[str, Any] | None = None,
        version: int | str = ipc.IPC_VERSION,
        pid: int | None = None,
        heartbeat_s: float = 1.0,
        misses: int = 3,
        validate: bool = True,
        connect_timeout_s: float = 3.0,
        hello_timeout_s: float = 5.0,
        backoff: tuple[float, float] = (0.2, 2.0),
        max_size: int = MAX_FRAME_BYTES,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme != "ws" or not parts.hostname or not is_loopback_host(parts.hostname):
            raise ValueError(f"the IPC client connects to ws://<loopback> only, not {url!r}")
        if not token:
            raise ValueError("the IPC client needs a token")
        add_secrets(token)  # never in the logs
        self.url = url
        self.role = role
        self._token = token
        self._clock = clock
        self.caps: dict[str, Any] = dict(caps or {})
        self.version = version
        self.pid = os.getpid() if pid is None else pid
        self.heartbeat_s = heartbeat_s
        self.misses = misses
        self._validate = validate
        self.connect_timeout_s = connect_timeout_s
        self.hello_timeout_s = hello_timeout_s
        self.backoff = backoff
        self._max_size = max_size
        self._handlers: dict[str, list[Handler]] = collections.defaultdict(list)
        self._link: Link | None = None
        self._up = asyncio.Event()
        self._closing = False
        self.on_link_lost: Callable[[], None] = _noop
        self.on_link_up: Callable[[], None] = _noop
        self.last_reason: str | None = None
        self.last_reject: HandshakeRejected | None = None
        self.core_info: dict[str, Any] = {}
        self.stats: collections.Counter[str] = collections.Counter()

    # --- API ------------------------------------------------------------------------------
    def on(self, msg_type: str, handler: Handler) -> None:
        """Call ``handler(envelope)`` for every valid ``msg_type`` message, in order."""
        self._handlers[msg_type].append(handler)

    @property
    def connected(self) -> bool:
        link = self._link
        return link is not None and link.connected

    @property
    def link(self) -> Link | None:
        return self._link

    async def wait_connected(self) -> None:
        """Wait until a hello has been accepted (use a deadline around it)."""
        await self._up.wait()

    def post(self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None) -> bool:
        """Queue a message; ``False`` if it was dropped (not connected, or invalid)."""
        link = self._link
        if link is None or not link.connected:
            self.stats["dropped_down"] += 1
            return False
        return link.post(msg_type, data, corr=corr) is not None

    async def send(
        self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None
    ) -> None:
        """Queue a message; raises ``IpcLinkError`` when not connected."""
        link = self._link
        if link is None:
            raise IpcLinkError("ipc client is not connected")
        await link.send(msg_type, data, corr=corr)

    async def request(
        self,
        msg_type: str,
        data: Mapping[str, Any],
        timeout: float = 1.0,  # noqa: ASYNC109 - the reply deadline is part of the API
    ) -> ipc.Envelope:
        link = self._link
        if link is None:
            raise IpcLinkError("ipc client is not connected")
        return await link.request(msg_type, data, timeout=timeout)

    def reply(self, to: ipc.Envelope, status: str, **extra: Any) -> bool:
        """Answer a core request (``speak.segment``) with ``reply{status}``."""
        link = self._link
        if link is None or not link.connected:
            self.stats["dropped_down"] += 1
            return False
        return link.reply(to, status, **extra) is not None

    def disconnect(self, reason: str = "disconnect requested") -> None:
        """Drop the current link (``run`` reconnects); for tests and operator restarts."""
        link = self._link
        if link is not None:
            link.close(CLOSE_SHUTDOWN, reason)

    async def aclose(self) -> None:
        """Stop reconnecting and close the current link (``run`` then returns)."""
        self._closing = True
        link = self._link
        if link is not None:
            link.close(CLOSE_SHUTDOWN, "worker shutting down")
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(3.0):
                    await link.wait_closed()

    # --- the loop -------------------------------------------------------------------------
    async def run(self) -> None:
        """Connect and reconnect until cancelled or ``aclose()``."""
        self._closing = False
        failures = 0
        while not self._closing:
            delay: float
            try:
                accepted = await self._session()
            except HandshakeRejected as exc:
                self.last_reject = exc
                self.stats["rejected"] += 1
                log.error("IPC: the core rejected us: %s", exc)
                delay = self.backoff[1]
            except (
                OSError,
                InvalidHandshake,
                InvalidURI,
                ConnectionClosed,
                DeadlineExceeded,
                ipc.IpcError,
            ) as exc:
                failures += 1
                self.stats["connect_failed"] += 1
                if failures <= 3 or failures % 30 == 0:
                    log.info("IPC: cannot reach the core at %s: %r", self.url, exc)
                delay = backoff_delay(failures, self.backoff)
            else:
                if accepted:
                    failures = 0
                delay = backoff_delay(max(1, failures), self.backoff)
            if self._closing:
                break
            await self._clock.sleep(delay)

    async def _session(self) -> bool:
        """One connection; ``True`` if the hello was accepted (the link then ran and ended)."""
        async with deadline(self.connect_timeout_s, what="ipc connect", clock=self._clock):
            ws = await connect(
                self.url,
                proxy=None,
                compression=None,
                max_size=self._max_size,
                open_timeout=self.connect_timeout_s,
                ping_interval=None,
                close_timeout=1.0,
                user_agent_header=None,
            )
        try:
            link = await self._hello(ws)
        except BaseException:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2.0):
                    await ws.close()
            raise
        self._link = link
        if self._closing:  # aclose() ran during the handshake
            link.close(CLOSE_SHUTDOWN, "worker shutting down")
        runner = asyncio.create_task(link.run(), name=f"ipc-link:{self.role}")
        self._up.set()
        self.stats["connected"] += 1
        log.info("IPC: connected to the core at %s", self.url)
        self._call(self.on_link_up, "on_link_up")
        try:
            self.last_reason = await runner
        finally:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
            self._up.clear()
            self._link = None
            if runner.done() and not runner.cancelled() and runner.exception() is None:
                self.last_reason = runner.result()
            if not self._closing:
                log.warning("IPC: link to the core lost: %s", self.last_reason)
                self._call(self.on_link_lost, "on_link_lost")
        return True

    async def _hello(self, ws: ClientConnection) -> Link:
        link = Link(
            ws,
            clock=self._clock,
            local_role=self.role,
            peer_role="core",
            dispatch=self._dispatch,
            heartbeat_s=self.heartbeat_s,
            misses=self.misses,
            validate=self._validate,
        )
        hello = ipc.Envelope(
            v=ipc.IPC_VERSION,
            type=ipc.HELLO,
            id=link.new_id(),
            ts=self._clock.now(),
            data={
                "role": self.role,
                "pid": self.pid,
                "version": self.version,
                "token": self._token,
                "caps": self.caps,
            },
        )
        if self._validate:
            ipc.validate(hello)
        try:
            async with deadline(self.hello_timeout_s, what="ipc hello", clock=self._clock):
                await ws.send(ipc.encode(hello))
                raw = await ws.recv()
        except ConnectionClosed as exc:
            rcvd = exc.rcvd
            code = rcvd.code if rcvd is not None else None
            reason = rcvd.reason if rcvd is not None else "connection closed"
            if code is not None and (code in _FATAL_CODES or code >= 4000):
                raise HandshakeRejected(code, reason) from exc
            raise
        env = ipc.decode(raw)
        if env.type != ipc.REPLY or env.corr != hello.id or env.data.get("status") != "ok":
            raise HandshakeRejected(None, f"unexpected answer to hello: {env.type}")
        self.core_info = {k: v for k, v in env.data.items() if k != "status"}
        return link

    async def _dispatch(self, env: ipc.Envelope) -> None:
        handlers = self._handlers.get(env.type)
        if handlers:
            await run_handlers(handlers, env)
        else:
            self.stats[f"unhandled:{env.type}"] += 1

    def _call(self, cb: Callable[[], None], what: str) -> None:
        try:
            cb()
        except Exception:
            log.exception("IPC %s callback failed", what)
