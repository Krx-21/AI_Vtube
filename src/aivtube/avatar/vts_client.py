"""Minimal VTube Studio Public API client on websockets' asyncio client (avatar brief).

- Connects to ``127.0.0.1`` with ``proxy=None`` (websockets >= 15 would otherwise pick up the
  environment's proxy), ``compression=None`` and ``ping_interval=None`` (it is unverified that
  VTS answers pings; the 60 Hz injection stream and an ``APIStateRequest`` heartbeat keep the
  link alive).
- Awaited requests match on ``requestID`` **and** the expected ``<X>Response``/``APIError``
  messageType: VTS events can carry the requestID of the subscription that created them.
- ``inject()`` is fire-and-forget: at most ``max_inflight`` (8) unanswered frames, extra frames
  are dropped, never queued (VTS answers once per render frame; stale mouth values are worse
  than skipped ones). Errors surface in ``last_ff_error`` and ``on_ff_error``.
- Auth: stored token → ``AuthenticationRequest``; if VTS says the token is invalid,
  ``AuthenticationTokenRequest`` (the streamer clicks Allow) → persist → authenticate again.

The reconnect loop lives in ``VTSSink.run``; a client object is reused across connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from enum import IntEnum
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from aivtube.contracts.infra import Clock, TaskSupervisor
from aivtube.infra.clock import SystemClock, deadline

__all__ = [
    "API_NAME",
    "API_VERSION",
    "VTSAPIError",
    "VTSClient",
    "VTSDisconnected",
    "VTSErrorID",
]

log = logging.getLogger("aivtube.avatar.vts")

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"
MAX_MESSAGE_BYTES = 16 * 2**20
_FF = "ff-"
_INJECT = "InjectParameterDataRequest"
_INJECT_RESPONSE = "InjectParameterDataResponse"


class VTSErrorID(IntEnum):
    """The VTS ``APIError`` ids this package reacts to (``Files/ErrorID.cs``)."""

    REQUEST_REQUIRES_AUTH = 8
    TOKEN_REQUEST_DENIED = 50
    TOKEN_REQUEST_ONGOING = 51
    HOTKEY_QUEUE_FULL = 200
    HOTKEY_NO_MODEL = 201
    HOTKEY_NOT_FOUND = 202
    HOTKEY_COOLDOWN = 203
    MOVE_NO_MODEL = 300
    CUSTOM_PARAM_NAME_INVALID = 350
    CUSTOM_PARAM_TAKEN = 352
    CUSTOM_PARAM_DEFAULT_NAME = 354
    INJECT_VALUE_INVALID = 451
    INJECT_PARAM_NOT_FOUND = 453
    INJECT_PARAM_HELD = 454
    EXPRESSION_BAD_FILENAME = 650
    EXPRESSION_NOT_FOUND = 651
    EXPRESSION_NO_MODEL = 652


class VTSAPIError(RuntimeError):
    """VTS answered a request with ``APIError``."""

    def __init__(self, request_type: str, error_id: int, message: str) -> None:
        super().__init__(f"{request_type} failed: errorID={error_id} {message}")
        self.request_type = request_type
        self.error_id = error_id
        self.message = message


class VTSDisconnected(ConnectionError):
    """No connection, or the connection closed while a request was waiting."""


def response_type(request_type: str) -> str:
    """``"HotkeyTriggerRequest"`` → ``"HotkeyTriggerResponse"``."""
    return request_type.removesuffix("Request") + "Response"


def _finite(value: float) -> float | None:
    v = float(value)
    if not math.isfinite(v):
        return None
    return min(1e6, max(-1e6, v))


class VTSClient:
    """One VTS plugin session at a time; ``connect()`` again to reconnect."""

    def __init__(
        self,
        url: str,
        plugin_name: str,
        plugin_developer: str,
        token_path: Path,
        *,
        connect: Callable[..., Any] = ws_connect,
        clock: Clock | None = None,
        request_timeout: float = 2.0,
        token_timeout: float = 120.0,
        open_timeout: float = 3.0,
        max_inflight: int = 8,
        ff_expire_s: float = 2.0,
        plugin_icon_b64: str | None = None,
        events_maxsize: int = 512,
        tasks: TaskSupervisor | None = None,
    ) -> None:
        self.url = url
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.token_path = Path(token_path)
        self.plugin_icon_b64 = plugin_icon_b64
        self.request_timeout = request_timeout
        self.token_timeout = token_timeout
        self.open_timeout = open_timeout
        self.max_inflight = max(1, max_inflight)
        self.ff_expire_s = ff_expire_s
        self._connect = connect
        self._clock: Clock = clock or SystemClock()
        self._tasks = tasks
        self.events: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue(maxsize=events_maxsize)
        self.on_ff_error: Callable[[int, str], None] | None = None
        self.last_ff_error: int | None = None
        self.last_ff_error_message = ""
        self.auth_error = ""
        self.last_rx = 0.0
        self.sent_frames = 0
        self.acked_frames = 0
        self.dropped_frames = 0
        self.lost_frames = 0
        self.ff_errors = 0
        self._ws: Any = None
        self._io_task: asyncio.Task[None] | None = None
        self._outbox: asyncio.Queue[str] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._closed.set()
        self._pending: dict[str, tuple[str, asyncio.Future[Mapping[str, Any]]]] = {}
        self._ff: dict[str, float] = {}  # requestID -> send time
        self._ids = itertools.count()
        self._authenticated = False

    # --- connection ---------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._ws is not None and self._io_task is not None and not self._io_task.done()

    @property
    def authenticated(self) -> bool:
        return self._authenticated and self.connected

    @property
    def in_flight(self) -> int:
        return len(self._ff)

    async def connect(self, url: str | None = None) -> None:
        """Open a new connection (closing any previous one) and start the reader/writer."""
        if url:
            self.url = url
        await self.aclose()
        async with deadline(self.open_timeout + 1.0, what="VTS connect", clock=self._clock):
            ws = await self._connect(
                self.url,
                proxy=None,
                compression=None,
                ping_interval=None,
                open_timeout=self.open_timeout,
                close_timeout=1.0,
                max_size=MAX_MESSAGE_BYTES,
            )
        self._ws = ws
        self._authenticated = False
        self._outbox = asyncio.Queue()
        self._closed = asyncio.Event()
        self.last_rx = self._clock.now()
        coro = self._io(ws)
        if self._tasks is not None:
            self._io_task = self._tasks.track(coro, name="vts-io")
        else:
            self._io_task = asyncio.get_running_loop().create_task(coro, name="vts-io")

    async def aclose(self) -> None:
        """Close the connection (if any) and wait briefly for the I/O task to end."""
        ws, task = self._ws, self._io_task
        self._io_task = None
        if ws is not None:
            try:
                async with deadline(2.0, what="VTS close"):
                    await ws.close()
            except Exception as exc:  # already broken; nothing else to do
                log.debug("VTS close: %r", exc)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait([task], timeout=1.0)
        if ws is not None:
            self._teardown(ws)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def _io(self, ws: Any) -> None:
        loop = asyncio.get_running_loop()
        writer = loop.create_task(self._write_loop(ws), name="vts-writer")
        try:
            async for raw in ws:
                self.last_rx = self._clock.now()
                self._dispatch(raw)
        except ConnectionClosed as exc:
            log.info("VTS connection closed: %s", exc)
        except Exception:
            log.exception("VTS reader failed")
        finally:
            self._teardown(ws)
            writer.cancel()
            await asyncio.wait([writer])
            if not writer.cancelled() and writer.exception() is not None:
                log.warning("VTS writer failed: %r", writer.exception())

    async def _write_loop(self, ws: Any) -> None:
        outbox = self._outbox
        limit = max(1.0, self.request_timeout)
        try:
            while True:
                message = await outbox.get()
                # send() only suspends when the socket buffer is full (VTS stopped reading)
                async with deadline(limit, what="VTS send"):
                    await ws.send(message)
        except ConnectionClosed:
            return
        except Exception:
            with contextlib.suppress(Exception):
                async with deadline(2.0, what="VTS close"):
                    await ws.close()
            raise

    def _teardown(self, ws: Any) -> None:
        if self._ws is not None and self._ws is not ws:
            return  # a newer connection owns the state
        self._ws = None
        self._authenticated = False
        pending, self._pending = self._pending, {}
        for _, fut in pending.values():
            if not fut.done():
                fut.set_exception(VTSDisconnected("VTS connection closed"))
        self._ff.clear()
        self._closed.set()

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            log.debug("VTS sent invalid JSON: %.80r", raw)
            return
        if not isinstance(msg, dict):
            return
        rid = str(msg.get("requestID") or "")
        mtype = str(msg.get("messageType") or "")
        entry = self._pending.get(rid)
        if entry is not None and mtype in (entry[0], "APIError"):
            del self._pending[rid]
            if not entry[1].done():
                entry[1].set_result(msg)
        elif rid.startswith(_FF) and mtype in (_INJECT_RESPONSE, "APIError"):
            self._ff.pop(rid, None)
            if mtype == "APIError":
                self._ff_error(msg)
            else:
                self.acked_frames += 1
        elif mtype.endswith("Event"):
            if self.events.full():
                self.events.get_nowait()  # drop the oldest event
            self.events.put_nowait(msg)
        else:
            log.debug("VTS unmatched message %s (%s)", mtype, rid)

    def _ff_error(self, msg: Mapping[str, Any]) -> None:
        data = msg.get("data") or {}
        eid = int(data.get("errorID", -1))
        message = str(data.get("message", ""))
        self.last_ff_error = eid
        self.last_ff_error_message = message
        self.ff_errors += 1
        if self.ff_errors <= 5 or self.ff_errors % 600 == 0:
            log.warning("VTS injection error %d (#%d): %s", eid, self.ff_errors, message)
        cb = self.on_ff_error
        if cb is not None:
            try:
                cb(eid, message)
            except Exception:
                log.exception("on_ff_error callback failed")

    def _send(self, request_id: str, message_type: str, data: Mapping[str, Any] | None) -> None:
        payload = {
            "apiName": API_NAME,
            "apiVersion": API_VERSION,
            "requestID": request_id,
            "messageType": message_type,
            "data": dict(data or {}),
        }
        self._outbox.put_nowait(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    # --- requests -----------------------------------------------------------------------------
    async def request(
        self,
        message_type: str,
        data: Mapping[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - public API from modules.json
    ) -> Mapping[str, Any]:
        """Send one request and await its response ``data`` (2 s deadline by default).

        Raises ``VTSAPIError`` (APIError), ``VTSDisconnected`` or ``DeadlineExceeded``.
        """
        if not self.connected:
            raise VTSDisconnected(f"not connected to VTS ({message_type})")
        rid = uuid.uuid4().hex
        fut: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = (response_type(message_type), fut)
        self._send(rid, message_type, data)
        limit = self.request_timeout if timeout is None else timeout
        try:
            async with deadline(limit, what=f"VTS {message_type}", clock=self._clock):
                msg = await fut
        finally:
            self._pending.pop(rid, None)
        body = msg.get("data") or {}
        if msg.get("messageType") == "APIError":
            raise VTSAPIError(
                message_type, int(body.get("errorID", -1)), str(body.get("message", ""))
            )
        return body if isinstance(body, Mapping) else {}

    def inject(
        self,
        values: Mapping[str, float],
        *,
        mode: str = "set",
        face_found: bool = True,
        weights: Mapping[str, float] | None = None,
    ) -> bool:
        """Fire-and-forget ``InjectParameterDataRequest``; False if dropped or not connected."""
        if not self.connected or not values:
            return False
        now = self._clock.now()
        if self._ff:
            expired = [rid for rid, t in self._ff.items() if now - t > self.ff_expire_s]
            for rid in expired:
                del self._ff[rid]
            self.lost_frames += len(expired)
        if len(self._ff) >= self.max_inflight:
            self.dropped_frames += 1
            return False
        params: list[dict[str, Any]] = []
        for name, raw in values.items():
            value = _finite(raw)
            if value is None:
                continue
            item: dict[str, Any] = {"id": name, "value": value}
            if weights and name in weights and mode == "set":
                item["weight"] = min(1.0, max(0.0, float(weights[name])))
            params.append(item)
        if not params:
            return False
        rid = f"{_FF}{next(self._ids)}"
        self._ff[rid] = now
        self._send(rid, _INJECT, {"faceFound": face_found, "mode": mode, "parameterValues": params})
        self.sent_frames += 1
        return True

    async def subscribe(
        self, event_name: str, config: Mapping[str, Any] | None = None, *, subscribe: bool = True
    ) -> Mapping[str, Any]:
        """``EventSubscriptionRequest``; events then arrive on ``events``."""
        return await self.request(
            "EventSubscriptionRequest",
            {"eventName": event_name, "subscribe": subscribe, "config": dict(config or {})},
        )

    # --- authentication -----------------------------------------------------------------------
    async def authenticate(self) -> bool:
        """Authenticate this session, requesting (and persisting) a new token if needed."""
        self._authenticated = False
        ident = {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer}
        token = await asyncio.to_thread(self._read_token)
        if token:
            data = await self.request(
                "AuthenticationRequest", {**ident, "authenticationToken": token}
            )
            if data.get("authenticated"):
                self._authenticated = True
                self.auth_error = ""
                return True
            log.info("VTS rejected the stored token (%s); requesting a new one", data.get("reason"))
        req: dict[str, Any] = dict(ident)
        if self.plugin_icon_b64:
            req["pluginIcon"] = self.plugin_icon_b64
        log.info("VTS: requesting plugin access; click Allow in VTube Studio")
        try:
            data = await self.request("AuthenticationTokenRequest", req, timeout=self.token_timeout)
        except VTSAPIError as exc:
            if exc.error_id in (VTSErrorID.TOKEN_REQUEST_DENIED, VTSErrorID.TOKEN_REQUEST_ONGOING):
                self.auth_error = f"plugin access denied in VTS (error {exc.error_id})"
                log.warning("VTS: %s", self.auth_error)
                return False
            raise
        token = str(data.get("authenticationToken") or "")
        if not token:
            self.auth_error = "VTS returned an empty token"
            return False
        await asyncio.to_thread(self._write_token, token)
        data = await self.request("AuthenticationRequest", {**ident, "authenticationToken": token})
        self._authenticated = bool(data.get("authenticated"))
        self.auth_error = "" if self._authenticated else str(data.get("reason") or "rejected")
        return self._authenticated

    def _read_token(self) -> str:
        try:
            return self.token_path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return ""
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("cannot read VTS token %s: %s", self.token_path, exc)
            return ""

    def _write_token(self, token: str) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_name(self.token_path.name + ".tmp")
        tmp.write_text(token, encoding="ascii")
        if sys.platform != "win32":
            with contextlib.suppress(OSError):
                os.chmod(tmp, 0o600)
        os.replace(tmp, self.token_path)
