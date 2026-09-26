"""One authenticated core <-> voice connection, used by both ends (Appendix A).

After the handshake (``hello`` → ``reply``), ``Link`` owns the websocket and runs four tasks:

- **reader**: decodes every frame, drops malformed or schema-invalid ones (logged, counted),
  ignores unknown message types, answers ``ping`` with ``pong`` at once, resolves ``reply`` /
  ``pong`` by ``corr`` and queues everything else for the dispatcher. Any frame counts as proof
  of life.
- **dispatcher**: runs the endpoint's handlers one message at a time, in arrival order. A slow
  handler (model loading on ``voice.configure``) delays later messages but never the reader,
  so heartbeats and pongs keep flowing.
- **writer**: sends queued frames in order. The outgoing queue is bounded; on overflow the
  oldest droppable frame (``lip.track``, ``heartbeat``, ``health``) goes first.
- **heartbeat**: sends ``heartbeat`` every ``heartbeat_s`` (1 Hz) and declares the peer dead when
  nothing arrived for ``misses`` (3) intervals.

Every outgoing message is validated against ``MESSAGE_SCHEMAS`` before it is queued, and every
incoming one after it is decoded. ``request`` correlates a ``reply``/``pong`` by ``corr`` and has
a deadline. ``sync_clock`` measures RTT and the peer's clock offset with 5 ``ping``s: offsets
above ``offset_warn_s`` (2 ms) are logged and applied by ``to_local``; smaller ones are noise.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import logging
import math
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final

from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed

from aivtube.contracts import ipc
from aivtube.contracts.infra import Clock
from aivtube.infra.clock import DeadlineExceeded, deadline

__all__ = [
    "CLOSE_BAD_TOKEN",
    "CLOSE_DEAD",
    "CLOSE_PROTOCOL",
    "CLOSE_REPLACED",
    "CLOSE_SHUTDOWN",
    "CLOSE_VERSION",
    "DROPPABLE",
    "MAX_FRAME_BYTES",
    "Handler",
    "IpcLinkError",
    "Link",
    "close_reason",
    "run_handlers",
]

log = logging.getLogger("aivtube.ipc")

Handler = Callable[[ipc.Envelope], Awaitable[None] | None]
"""A message handler: called with the envelope, may be a coroutine function."""

CLOSE_SHUTDOWN: Final = 1001
CLOSE_PROTOCOL: Final = 4000
CLOSE_BAD_TOKEN: Final = 4001
CLOSE_VERSION: Final = 4002
CLOSE_REPLACED: Final = 4003
CLOSE_DEAD: Final = 4004

MAX_FRAME_BYTES: Final = 4 * 1024 * 1024
DROPPABLE: Final[frozenset[str]] = frozenset({ipc.LIP_TRACK, ipc.HEARTBEAT, ipc.HEALTH})
_CLOSE_TIMEOUT_S: Final = 2.0
_DRAIN_S: Final = 0.5


class IpcLinkError(ConnectionError):
    """The link is not connected, or it was lost while a request waited for its reply."""


def close_reason(text: str, limit: int = 120) -> str:
    """A websocket close reason: at most ``limit`` UTF-8 bytes (the protocol allows 123)."""
    raw = text.encode("utf-8")[:limit]
    return raw.decode("utf-8", errors="ignore")


async def run_handlers(handlers: Sequence[Handler], env: ipc.Envelope) -> None:
    """Run every handler registered for ``env.type`` in order; failures are logged."""
    for handler in handlers:
        try:
            result = handler(env)
            if result is not None:
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ipc handler %r for %s failed", handler, env.type)


class _LogLimiter:
    """Logs the first few occurrences of a problem, then every ``every``-th."""

    def __init__(self, first: int = 5, every: int = 100) -> None:
        self._first = first
        self._every = every
        self._counts: collections.Counter[str] = collections.Counter()

    def __call__(self, key: str) -> bool:
        self._counts[key] += 1
        n = self._counts[key]
        return n <= self._first or n % self._every == 0


class Link:
    """One live connection after a successful handshake (see the module docstring).

    Create it on the event loop that will run it, then await ``run()`` (it returns the reason
    the link ended). ``post``/``send``/``request`` may be used before ``run`` starts: frames
    wait in the outgoing queue.
    """

    def __init__(
        self,
        ws: Connection,
        *,
        clock: Clock,
        local_role: str,
        peer_role: str,
        dispatch: Callable[[ipc.Envelope], Awaitable[None]],
        heartbeat_s: float = 1.0,
        misses: int = 3,
        validate: bool = True,
        max_queue: int = 2048,
        send_timeout_s: float = 5.0,
        offset_warn_s: float = 0.002,
    ) -> None:
        if heartbeat_s <= 0 or misses < 1:
            raise ValueError("heartbeat_s must be positive and misses >= 1")
        self._ws = ws
        self._clock = clock
        self.local_role = local_role
        self.peer_role = peer_role
        self._dispatch = dispatch
        self.heartbeat_s = float(heartbeat_s)
        self.misses = int(misses)
        self._validate = validate
        self._max_queue = max(16, int(max_queue))
        self._send_timeout_s = send_timeout_s
        self.offset_warn_s = offset_warn_s
        self._ids = itertools.count(1)
        self._prefix = f"{local_role[:1] or 'x'}{secrets.token_hex(3)}-"
        self._out: collections.deque[tuple[str, str]] = collections.deque()
        self._out_evt = asyncio.Event()
        self._inbox: collections.deque[ipc.Envelope] = collections.deque()
        self._inbox_evt = asyncio.Event()
        self._inbox_closed = False
        self._pending: dict[str, asyncio.Future[ipc.Envelope]] = {}
        self._close_evt = asyncio.Event()
        self._close_code = CLOSE_SHUTDOWN
        self._close_reason = ""
        self._done = asyncio.Event()
        self._reason: str | None = None
        self._limit = _LogLimiter()
        self.connected = True
        self.last_rx = clock.now()
        self.rtt_s = math.nan
        self.measured_offset_s = 0.0
        self.clock_offset_s = 0.0  # applied correction: peer time - local time
        self.stats: collections.Counter[str] = collections.Counter()

    # --- sending ----------------------------------------------------------------------------
    def new_id(self) -> str:
        return f"{self._prefix}{next(self._ids)}"

    def _encode(self, msg_type: str, data: Mapping[str, Any], corr: str | None) -> tuple[str, str]:
        env = ipc.Envelope(
            v=ipc.IPC_VERSION,
            type=msg_type,
            id=self.new_id(),
            ts=self._clock.now(),
            data=data,
            corr=corr,
        )
        if self._validate:
            ipc.validate(env)
        return env.id, ipc.encode(env)

    def _enqueue(self, msg_type: str, raw: str) -> None:
        if len(self._out) >= self._max_queue:
            victim = next((i for i, (t, _) in enumerate(self._out) if t in DROPPABLE), None)
            if victim is None and msg_type in DROPPABLE:
                self.stats["dropped_overflow"] += 1
                return
            if victim is not None:
                del self._out[victim]
            else:
                self._out.popleft()
            self.stats["dropped_overflow"] += 1
            if self._limit("overflow"):
                log.warning("ipc %s: outgoing queue full; dropped a frame", self.peer_role)
        self._out.append((msg_type, raw))
        self._out_evt.set()

    def post(
        self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None
    ) -> str | None:
        """Queue a message without waiting; returns its id, or ``None`` if it was dropped
        (link down, or invalid: logged)."""
        if not self.connected:
            self.stats["dropped_closed"] += 1
            return None
        try:
            mid, raw = self._encode(msg_type, data, corr)
        except ipc.IpcError as exc:
            self.stats["invalid_out"] += 1
            if self._limit(f"out:{msg_type}"):
                log.error("ipc: not sending invalid %s: %s", msg_type, exc)
            return None
        self._enqueue(msg_type, raw)
        return mid

    async def send(self, msg_type: str, data: Mapping[str, Any], *, corr: str | None = None) -> str:
        """Queue a message; raises ``IpcLinkError`` when down and ``IpcError`` when invalid."""
        if not self.connected:
            raise IpcLinkError(f"ipc link to {self.peer_role} is down")
        mid, raw = self._encode(msg_type, data, corr)
        self._enqueue(msg_type, raw)
        return mid

    async def request(
        self,
        msg_type: str,
        data: Mapping[str, Any],
        *,
        timeout: float = 1.0,  # noqa: ASYNC109 - the reply deadline is part of the API
    ) -> ipc.Envelope:
        """Send and wait for the ``reply``/``pong`` whose ``corr`` is this message's id.

        Raises ``DeadlineExceeded`` (a ``TimeoutError``) after ``timeout`` seconds and
        ``IpcLinkError`` when the link is or goes down.
        """
        if not self.connected:
            raise IpcLinkError(f"ipc link to {self.peer_role} is down")
        mid, raw = self._encode(msg_type, data, None)
        fut: asyncio.Future[ipc.Envelope] = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        try:
            self._enqueue(msg_type, raw)
            async with deadline(timeout, what=f"ipc {msg_type} reply", clock=self._clock):
                return await fut
        finally:
            self._pending.pop(mid, None)
            if not fut.done():
                fut.cancel()

    def reply(self, to: ipc.Envelope, status: str, **extra: Any) -> str | None:
        """Answer a request with ``reply{status, ...}`` (``corr`` = its id)."""
        return self.post(ipc.REPLY, {"status": status, **extra}, corr=to.id)

    # --- clock ------------------------------------------------------------------------------
    def to_local(self, t_peer: float) -> float:
        """A peer ``perf_counter`` time on the local clock."""
        return t_peer - self.clock_offset_s

    async def sync_clock(self, rounds: int = 5, timeout: float = 1.0) -> None:  # noqa: ASYNC109
        """Measure RTT and clock offset with ``rounds`` pings (the best RTT sample wins)."""
        best: tuple[float, float] | None = None
        for seq in range(rounds):
            t1 = self._clock.now()
            try:
                pong = await self.request(ipc.PING, {"seq": seq}, timeout=timeout)
            except DeadlineExceeded:
                log.warning("ipc %s: ping %d timed out", self.peer_role, seq)
                continue
            t3 = self._clock.now()
            rtt = max(0.0, t3 - t1)
            offset = pong.ts - (t1 + t3) / 2.0
            if best is None or rtt < best[0]:
                best = (rtt, offset)
        if best is None:
            log.warning("ipc %s: clock check failed (no pong)", self.peer_role)
            return
        self.rtt_s, self.measured_offset_s = best
        if abs(self.measured_offset_s) > self.offset_warn_s:
            self.clock_offset_s = self.measured_offset_s
            log.warning(
                "ipc %s: clock offset %.2f ms exceeds %.1f ms; correcting peer times",
                self.peer_role,
                self.measured_offset_s * 1e3,
                self.offset_warn_s * 1e3,
            )
        else:
            self.clock_offset_s = 0.0
        log.info(
            "ipc %s: rtt %.2f ms, clock offset %.3f ms",
            self.peer_role,
            self.rtt_s * 1e3,
            self.measured_offset_s * 1e3,
        )

    # --- lifecycle --------------------------------------------------------------------------
    @property
    def reason(self) -> str | None:
        """Why the link ended (``None`` while it runs)."""
        return self._reason

    def close(self, code: int = CLOSE_SHUTDOWN, reason: str = "") -> None:
        """Ask ``run()`` to close the socket with ``code`` and return."""
        if self._close_evt.is_set():
            return
        self._close_code = code
        self._close_reason = reason
        self._set_reason(f"closed locally ({code} {reason})".rstrip())
        self._close_evt.set()

    async def wait_closed(self) -> None:
        await self._done.wait()

    async def run(self) -> str:
        """Run the link until the peer closes, dies or ``close()`` is called."""
        self.last_rx = self._clock.now()
        name = f"ipc:{self.peer_role}"
        tasks = [
            asyncio.create_task(self._reader(), name=f"{name}:reader"),
            asyncio.create_task(self._writer(), name=f"{name}:writer"),
            asyncio.create_task(self._heartbeat(), name=f"{name}:heartbeat"),
            asyncio.create_task(self._close_evt.wait(), name=f"{name}:close"),
        ]
        dispatcher = asyncio.create_task(self._dispatcher(), name=f"{name}:dispatch")
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    exc = task.exception()
                    log.error("ipc %s: %s failed", self.peer_role, task.get_name(), exc_info=exc)
                    self._set_reason(f"internal error: {exc!r}")
        finally:
            self.connected = False
            self._set_reason("closed")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown(dispatcher)
        return self._reason or "closed"

    async def _shutdown(self, dispatcher: asyncio.Task[None]) -> None:
        code = self._close_code if self._close_evt.is_set() else CLOSE_DEAD
        if self._reason and self._reason.startswith("closed by peer"):
            code = CLOSE_SHUTDOWN
        with contextlib.suppress(Exception):
            async with asyncio.timeout(_CLOSE_TIMEOUT_S):
                await self._ws.close(code, close_reason(self._close_reason or self._reason or ""))
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(IpcLinkError(f"ipc link lost: {self._reason}"))
        self._pending.clear()
        self._out.clear()
        # messages that arrived before the drop are still handled, in order (bounded)
        self._inbox_closed = True
        self._inbox_evt.set()
        try:
            await asyncio.wait({dispatcher}, timeout=_DRAIN_S)
        finally:
            if not dispatcher.done():
                self.stats["dropped_inbox"] += len(self._inbox)
                dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)
            self._done.set()

    def _set_reason(self, reason: str) -> None:
        if self._reason is None:
            self._reason = reason

    # --- tasks ------------------------------------------------------------------------------
    async def _reader(self) -> None:
        try:
            async for raw in self._ws:
                self._on_frame(raw)
        except ConnectionClosed as exc:
            rcvd = exc.rcvd
            detail = f"{rcvd.code} {rcvd.reason}".strip() if rcvd is not None else "no close frame"
            self._set_reason(f"closed by peer ({detail})")
            return
        self._set_reason("closed by peer")

    def _on_frame(self, raw: str | bytes) -> None:
        self.last_rx = self._clock.now()
        self.stats["rx"] += 1
        try:
            env = ipc.decode(raw)
        except ipc.IpcError as exc:
            self.stats["invalid_in"] += 1
            if self._limit("decode"):
                log.warning("ipc %s: dropped a malformed frame: %s", self.peer_role, exc)
            return
        if env.type not in ipc.MESSAGE_SCHEMAS:
            self.stats["unknown_in"] += 1
            if self._limit(f"unknown:{env.type}"):
                log.info("ipc %s: ignoring unknown message type %r", self.peer_role, env.type)
            return
        if self._validate:
            try:
                ipc.validate(env)
            except ipc.IpcError as exc:
                self.stats["invalid_in"] += 1
                if self._limit(f"in:{env.type}"):
                    log.warning("ipc %s: dropped invalid %s: %s", self.peer_role, env.type, exc)
                return
        mtype = env.type
        if mtype in (ipc.REPLY, ipc.PONG):
            fut = self._pending.get(env.corr or "")
            if fut is not None and not fut.done():
                fut.set_result(env)
            else:
                self.stats["late_reply"] += 1
            return
        if mtype == ipc.PING:
            self.post(ipc.PONG, {"ping_ts": env.ts, "seq": env.data.get("seq", 0)}, corr=env.id)
            return
        if mtype == ipc.HEARTBEAT:
            return
        if mtype == ipc.HELLO:
            self.stats["unexpected_hello"] += 1
            return
        self._inbox.append(env)
        self._inbox_evt.set()

    async def _dispatcher(self) -> None:
        while True:
            while not self._inbox:
                if self._inbox_closed:
                    return
                self._inbox_evt.clear()
                await self._inbox_evt.wait()
            env = self._inbox.popleft()
            try:
                await self._dispatch(env)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats["handler_error"] += 1
                log.exception("ipc %s: handler for %s failed", self.peer_role, env.type)

    async def _writer(self) -> None:
        while True:
            while not self._out:
                self._out_evt.clear()
                await self._out_evt.wait()
            _mtype, raw = self._out.popleft()
            try:
                async with deadline(self._send_timeout_s, what="ipc send", clock=self._clock):
                    await self._ws.send(raw)
            except DeadlineExceeded:
                self._set_reason(f"send stalled for {self._send_timeout_s:g} s")
                return
            except ConnectionClosed as exc:
                rcvd = exc.rcvd
                detail = f"{rcvd.code} {rcvd.reason}".strip() if rcvd is not None else "lost"
                self._set_reason(f"closed by peer ({detail})")
                return
            self.stats["tx"] += 1

    async def _heartbeat(self) -> None:
        for seq in itertools.count():
            self.post(ipc.HEARTBEAT, {"seq": seq})
            await self._clock.sleep(self.heartbeat_s)
            silent = self._clock.now() - self.last_rx
            if silent >= self.misses * self.heartbeat_s:
                self.stats["heartbeat_dead"] += 1
                self._set_reason(f"{self.misses} heartbeats missed ({silent:.1f} s silent)")
                log.warning("ipc %s: peer is dead (%s)", self.peer_role, self._reason)
                return
