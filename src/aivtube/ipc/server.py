"""The core's IPC server: ``ws://127.0.0.1:<ports.bus>/bus`` (Appendix A, §2.2).

Loopback only. A client must send ``hello{role, pid, version, token, caps}`` within
``hello_timeout_s``; a wrong token closes with 4001, a different major version with 4002 and
anything else malformed with 4000. The core answers ``reply{status: "ok"}`` (``corr`` = the
hello's id), runs the 5-ping clock check and then reports the peer ready. One peer per role: a
new hello for a role replaces (closes, 4003) the old connection, which then counts as lost.
Requests with an ``Origin`` header (browsers) are refused, so a web page cannot open the bus.

Handlers registered with ``on`` run for every valid message of their type, in arrival order,
before anything reaches the event bus (they are what publishes the events). The server
itself publishes ``HealthChanged`` for ``ipc.<role>`` when a peer comes up or goes down.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import hmac
import ipaddress
import logging
import os
from collections.abc import Callable, Mapping
from http import HTTPStatus
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from aivtube.contracts import ipc
from aivtube.contracts.events import HealthChanged
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.types import Health, HealthState
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.infra.logging import add_secrets
from aivtube.ipc.link import (
    CLOSE_BAD_TOKEN,
    CLOSE_PROTOCOL,
    CLOSE_REPLACED,
    CLOSE_SHUTDOWN,
    CLOSE_VERSION,
    MAX_FRAME_BYTES,
    Handler,
    IpcLinkError,
    Link,
    close_reason,
    run_handlers,
)

__all__ = ["IpcPeer", "IpcServer", "is_loopback_host", "major_version"]

log = logging.getLogger("aivtube.ipc.server")


def is_loopback_host(host: str) -> bool:
    """``127.0.0.1``, ``::1`` or ``localhost`` (our sockets never bind anything else)."""
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def major_version(version: object) -> int | None:
    """The major number of a hello ``version`` (``1`` or ``"1.2"``); ``None`` if malformed."""
    if isinstance(version, bool):
        return None
    if isinstance(version, int):
        return version
    if isinstance(version, str):
        head = version.strip().split(".", 1)[0]
        return int(head) if head.isdigit() else None
    return None


class IpcPeer:
    """A connected worker as seen by the core."""

    def __init__(
        self,
        role: str,
        link: Link,
        *,
        pid: int,
        version: int | str,
        caps: Mapping[str, Any],
    ) -> None:
        self.role = role
        self.link = link
        self.pid = pid
        self.version = version
        self.caps = dict(caps)

    @property
    def connected(self) -> bool:
        return self.link.connected

    @property
    def rtt_s(self) -> float:
        return self.link.rtt_s

    @property
    def clock_offset_s(self) -> float:
        """The correction in effect (peer - core); 0 unless the offset exceeded 2 ms."""
        return self.link.clock_offset_s

    @property
    def measured_offset_s(self) -> float:
        return self.link.measured_offset_s

    def to_local(self, t_peer: float) -> float:
        return self.link.to_local(t_peer)

    async def send(
        self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None
    ) -> None:
        await self.link.send(msg_type, data, corr=corr)

    def post(self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None) -> bool:
        return self.link.post(msg_type, data, corr=corr) is not None

    async def request(
        self,
        msg_type: str,
        data: Mapping[str, Any],
        timeout: float = 1.0,  # noqa: ASYNC109 - the reply deadline is part of the API
    ) -> ipc.Envelope:
        return await self.link.request(msg_type, data, timeout=timeout)

    def reply(self, to: ipc.Envelope, status: str, **extra: Any) -> bool:
        return self.link.reply(to, status, **extra) is not None

    def close(self, code: int = CLOSE_SHUTDOWN, reason: str = "") -> None:
        self.link.close(code, reason)

    def __repr__(self) -> str:
        return f"IpcPeer(role={self.role!r}, pid={self.pid}, connected={self.connected})"


class IpcServer:
    """The core end of the IPC bus (see the module docstring).

    Run ``serve()`` as a supervised critical task; it listens until cancelled (or ``aclose``).
    ``port=0`` picks a free port (tests); read it from ``port`` after ``wait_started()``. A later
    ``serve()`` (a supervised restart) binds the same port again, so workers find it.
    """

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        clock: Clock,
        bus: EventBus,
        *,
        path: str = "/bus",
        heartbeat_s: float = 1.0,
        misses: int = 3,
        hello_timeout_s: float = 5.0,
        validate: bool = True,
        clock_sync_rounds: int = 5,
        offset_warn_s: float = 0.002,
        max_size: int = MAX_FRAME_BYTES,
    ) -> None:
        if not is_loopback_host(host):
            raise ValueError(f"the IPC server binds loopback only, not {host!r}")
        if not token:
            raise ValueError("the IPC server needs a token")
        add_secrets(token)  # never in the logs, even inside an error message
        self.host = host
        self._port = int(port)
        self._token = token.encode("utf-8")
        self._clock = clock
        self._bus = bus
        self.path = path
        self.heartbeat_s = heartbeat_s
        self.misses = misses
        self.hello_timeout_s = hello_timeout_s
        self._validate = validate
        self._sync_rounds = clock_sync_rounds
        self._offset_warn_s = offset_warn_s
        self._max_size = max_size
        self._handlers: dict[str, list[Handler]] = collections.defaultdict(list)
        self._lost_cbs: list[Callable[[str, str], None]] = []
        self._ready_cbs: list[Callable[[IpcPeer], None]] = []
        self._peers: dict[str, IpcPeer] = {}
        self._server: Server | None = None
        self._started = asyncio.Event()
        self._stop = asyncio.Event()
        self.stats: collections.Counter[str] = collections.Counter()

    # --- registration -------------------------------------------------------------------
    def on(self, msg_type: str, handler: Handler) -> None:
        """Call ``handler(envelope)`` for every valid ``msg_type`` message from any peer."""
        self._handlers[msg_type].append(handler)

    def on_peer_lost(self, cb: Callable[[str, str], None]) -> None:
        """``cb(role, reason)`` when a peer's connection ends (not when it is replaced)."""
        self._lost_cbs.append(cb)

    def on_peer_ready(self, cb: Callable[[IpcPeer], None]) -> None:
        """``cb(peer)`` after a peer's handshake and clock check."""
        self._ready_cbs.append(cb)

    def peer(self, role: str) -> IpcPeer | None:
        """The connected peer for ``role``, if any."""
        p = self._peers.get(role)
        return p if p is not None and p.connected else None

    @property
    def peers(self) -> list[IpcPeer]:
        return [p for p in self._peers.values() if p.connected]

    @property
    def port(self) -> int:
        """The listening port (the real one once bound when constructed with ``port=0``)."""
        return self._port

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"ws://{host}:{self.port}{self.path}"

    async def wait_started(self) -> None:
        await self._started.wait()

    # --- serving ------------------------------------------------------------------------
    async def serve(self) -> None:
        """Listen until cancelled or ``aclose()``; closes every peer on the way out."""
        self._stop.clear()
        async with serve(
            self._handle,
            self.host,
            self._port,
            origins=[None],
            process_request=self._process_request,
            compression=None,
            max_size=self._max_size,
            ping_interval=None,
            close_timeout=1.0,
            open_timeout=self.hello_timeout_s,
            server_header=None,
        ) as server:
            self._server = server
            for sock in server.sockets:  # port=0: keep the real port for restarts
                self._port = int(sock.getsockname()[1])
                break
            self._started.set()
            log.info("IPC server listening on %s", self.url)
            try:
                await self._stop.wait()
            finally:
                for peer in list(self._peers.values()):
                    peer.close(CLOSE_SHUTDOWN, "core shutting down")
                self._started.clear()
                self._server = None

    async def aclose(self) -> None:
        """Stop ``serve()`` (for owners that do not cancel it)."""
        self._stop.set()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path.split("?", 1)[0] != self.path:
            self.stats["bad_path"] += 1
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    # --- one connection -----------------------------------------------------------------
    async def _handle(self, ws: ServerConnection) -> None:
        hello = await self._handshake(ws)
        if hello is None:
            return
        data = hello.data
        role = str(data["role"])
        link = Link(
            ws,
            clock=self._clock,
            local_role="core",
            peer_role=role,
            dispatch=self._dispatch,
            heartbeat_s=self.heartbeat_s,
            misses=self.misses,
            validate=self._validate,
            offset_warn_s=self._offset_warn_s,
        )
        peer = IpcPeer(
            role, link, pid=int(data["pid"]), version=data["version"], caps=data.get("caps") or {}
        )
        old = self._peers.get(role)
        if old is not None:
            log.warning("IPC: a new %s connection replaces the old one", role)
            old.close(CLOSE_REPLACED, "replaced by a new connection")
        self._peers[role] = peer
        link.reply(hello, "ok", role="core", pid=os.getpid(), version=ipc.IPC_VERSION)
        self.stats["connected"] += 1
        runner = asyncio.create_task(link.run(), name=f"ipc-link:{role}")
        reason = "closed"
        try:
            with contextlib.suppress(IpcLinkError):  # dropped mid-check: the runner says why
                await link.sync_clock(self._sync_rounds)
            if link.connected:
                self._publish(role, HealthState.OK, f"connected (pid {peer.pid})")
                for cb in list(self._ready_cbs):
                    try:
                        cb(peer)
                    except Exception:
                        log.exception("on_peer_ready callback failed")
            reason = await runner
        finally:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
            if runner.done() and not runner.cancelled() and runner.exception() is None:
                reason = runner.result()
            if self._peers.get(role) is peer:
                del self._peers[role]
                log.warning("IPC: %s link lost: %s", role, reason)
                self._publish(role, HealthState.DOWN, reason)
                for lost_cb in list(self._lost_cbs):
                    try:
                        lost_cb(role, reason)
                    except Exception:
                        log.exception("on_peer_lost callback failed")
            else:
                log.info("IPC: replaced %s link closed: %s", role, reason)

    async def _handshake(self, ws: ServerConnection) -> ipc.Envelope | None:
        try:
            async with deadline(self.hello_timeout_s, what="ipc hello", clock=self._clock):
                raw = await ws.recv()
        except DeadlineExceeded:
            await self._reject(ws, CLOSE_PROTOCOL, "no hello")
            return None
        except ConnectionClosed:
            return None
        try:
            env = ipc.decode(raw)
        except ipc.IpcError:
            await self._reject(ws, CLOSE_PROTOCOL, "malformed hello")
            return None
        if env.type != ipc.HELLO:
            await self._reject(ws, CLOSE_PROTOCOL, "expected hello")
            return None
        major = major_version(env.data.get("version"))
        if env.v != ipc.IPC_VERSION or major != ipc.IPC_VERSION:
            await self._reject(
                ws,
                CLOSE_VERSION,
                f"IPC version {env.data.get('version')!r} not supported "
                f"(core speaks {ipc.IPC_VERSION})",
            )
            return None
        try:
            ipc.validate(env)
        except ipc.IpcError as exc:
            await self._reject(ws, CLOSE_PROTOCOL, f"invalid hello: {exc}")
            return None
        token = str(env.data["token"]).encode("utf-8")
        if not hmac.compare_digest(token, self._token):
            await self._reject(ws, CLOSE_BAD_TOKEN, "bad token")
            return None
        return env

    async def _reject(self, ws: ServerConnection, code: int, reason: str) -> None:
        self.stats[f"rejected_{code}"] += 1
        log.warning("IPC: rejected a connection from %s: %s", ws.remote_address, reason)
        try:
            async with asyncio.timeout(2.0):
                await ws.close(code, close_reason(reason))
        except (TimeoutError, ConnectionClosed, OSError):
            pass

    async def _dispatch(self, env: ipc.Envelope) -> None:
        handlers = self._handlers.get(env.type)
        if handlers:
            await run_handlers(handlers, env)

    def _publish(self, role: str, state: HealthState, detail: str) -> None:
        health = Health(f"ipc.{role}", state, detail, self._clock.now())
        try:
            self._bus.publish(HealthChanged(health=health))
        except Exception:
            log.exception("publishing IPC health failed")
