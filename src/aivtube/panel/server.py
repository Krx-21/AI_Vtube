"""``PanelServer``: operator panel, OBS overlays, hotkeys and alert ingest (§2.10, §2.11, §4.2).

An aiohttp app bound to loopback only (``127.0.0.1:<ports.panel>``, 8770):

=========================  ====================================================================
``GET /``                  the operator SPA (``static/index.html``, vanilla JS, no CDN)
``GET /overlay/captions``  OBS browser source: captions at their audible time and "Filtered."
``GET /overlay/status``    OBS browser source: state badge and alarms
``GET /healthz``           launcher heartbeat (no token; answers only if the loop is alive)
``GET /api/ping``          SPA liveness check (the SPA hard-kills via the launcher on timeout)
``GET /api/config``        emergency endpoint URL + token, characters, hotkeys, budgets
``GET /api/state``         snapshot (alias ``/api/snapshot``): control + health + recent feeds
                           (health: ``health_source`` merged with ``HealthChanged`` events)
``POST /api/cmd``          ``{kind, args?, character?, id?}`` → ``ControlSurface.execute``
``POST /api/mic``          ``{mode}`` or ``{ptt: bool}``
``GET|POST /hotkey/<n>``   AutoHotkey / Stream Deck (``?token=``)
``POST /api/event``        generic donation/alert ingest → ``ChatMessage`` → ``ingest``
``GET /api/traces?n=``     latency waterfall rows, p50/p95 badges, cache/opener alarms
``GET /api/audit``         ``?table=op_audit|tool_audit|moderation_log&since=&limit=``
``GET /api/chat``          chat window with scores (``?character=``)
``GET /api/memory``        memories (``?character=&kind=&status=``)
``PATCH|DELETE /api/memory/{id}``  status / edit / delete (through the control surface)
``GET /api/moderation``    moderation feed: BLOCK/DROP/REVIEW records (``?since_ts=&since=``)
``POST /api/moderation``   ``{action: mute_user|blocklist|false_positive, ...}``
``GET /ws``                live bus events (``?types=A,B&character=``)
=========================  ====================================================================

Security: loopback bind and loopback peers only; the ``Host`` header must name a loopback
host (DNS rebinding) and a browser ``Origin`` must be this panel; ``/api/*``, ``/ws`` and
``/hotkey/*`` need the panel token (``X-Aivtube-Token``, ``Authorization: Bearer`` or
``?token=``). Static pages hold no data and no secrets.

The WebSocket mirrors every bus event as ``event_to_json`` (chat ``raw`` payloads stripped).
Each client has a bounded drop-oldest queue; frames are batched and sent at most
``ws_max_hz`` times per second, so no event type exceeds that rate. A client that cannot take
a frame within ``ws_send_timeout_s`` is disconnected.

Operator commands are audited by the control surface (``CoreControl`` writes ``op_audit``);
actions the panel performs itself (blocklist, false positive) are written through ``ops``.

The moderation feed reads the live records of the safety gate (``moderation_log``, e.g.
``lambda: gate.audit.recent``: ``Filtered`` events carry no verdict, so REVIEW items exist only
there) or, without it, the ``moderation_log`` table through ``ops``. Items carry ``ref`` (the
first 16 hex digits of the text's sha256, as in ``Filtered.ref``) and, for chat input, the
``platform`` and ``user_id`` the mute action needs.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hmac
import inspect
import ipaddress
import json
import logging
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from aiohttp import WSCloseCode, WSMsgType, web
from yarl import URL

from aivtube.contracts.chat import ChatWindow
from aivtube.contracts.control import ControlSurface, OpCommand, OpKind, OpResult
from aivtube.contracts.events import (
    EVENT_TYPES,
    Alert,
    ChatDropped,
    ChatReceived,
    Event,
    Filtered,
    HealthChanged,
    OperatorAction,
    StateChanged,
    SupportReceived,
    TurnTraceReady,
    event_to_json,
)
from aivtube.contracts.infra import Clock, EventBus, Overflow, Subscription
from aivtube.contracts.memory import MemKind, MemoryItem, MemoryStore, MemStatus
from aivtube.contracts.types import ChatMessage, Health, HealthState, Platform
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.panel._json import dumps, message_json, to_jsonable
from aivtube.panel.commands import hotkey_command, hotkey_names, new_command_id, parse_op_command
from aivtube.panel.ingest import alert_message
from aivtube.panel.stats import DEFAULT_BUDGETS, Budget, trace_summary

__all__ = [
    "LOOPBACK_HOSTS",
    "AuditTable",
    "ModerationActions",
    "PanelOps",
    "PanelServer",
]

log = logging.getLogger("aivtube.panel")

LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})
AuditTable = Literal["tool_audit", "op_audit", "moderation_log"]
_AUDIT_TABLES: Final[tuple[AuditTable, ...]] = ("op_audit", "tool_audit", "moderation_log")
_TOKEN_HEADER: Final = "X-Aivtube-Token"
_STATIC_DIR: Final = Path(__file__).resolve().parent / "static"
_PAGES: Final[Mapping[str, str]] = {
    "/": "index.html",
    "/index.html": "index.html",
    "/overlay/captions": "overlay_captions.html",
    "/overlay/status": "overlay_status.html",
}
_MAX_BODY: Final = 256 * 1024
_MAX_BATCH: Final = 256
_RECENT: Final = 100
_MEM_KINDS: Final = ("core", "fact", "viewer", "episode")
_MEM_STATUSES: Final = ("active", "quarantined", "deleted")
_MEM_EDIT_FIELDS: Final = ("text", "subject", "importance", "locked", "pinned")
_PLATFORMS: Final = frozenset(p.value for p in Platform)
_MOD_FIELDS: Final = ("character", "direction", "source", "tier", "category", "rule", "verdict")


class PanelOps(Protocol):
    """The parts of ``memory.OpsDb`` the panel uses (all never raise on write)."""

    async def recent_traces(self, n: int = 20) -> list[Mapping[str, Any]]: ...

    async def audit_rows(
        self, table: AuditTable, *, since_id: int = 0, limit: int = 100
    ) -> list[Mapping[str, Any]]: ...

    async def log_op(self, **row: Any) -> None: ...


class ModerationActions(Protocol):
    """Moderation-feed actions that have no ``OpKind`` (mute-user uses ``MUTE_USER``)."""

    async def add_to_blocklist(
        self, term: str, *, category: str, character: str | None
    ) -> OpResult: ...

    async def mark_false_positive(self, ref: str, *, note: str) -> OpResult: ...


class _BadRequest(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass(eq=False)
class _Client:
    """One WebSocket subscriber with a bounded drop-oldest queue of serialised events."""

    ws: web.WebSocketResponse
    types: frozenset[str] | None
    character: str | None
    maxlen: int
    queue: collections.deque[str] = field(default_factory=collections.deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    dropped: int = 0
    dropped_total: int = 0
    sent: int = 0

    def offer(self, type_name: str, character: str | None, payload: str) -> None:
        if self.types is not None and type_name not in self.types:
            return
        if self.character is not None and character not in (None, self.character):
            return
        if len(self.queue) >= self.maxlen:
            self.queue.popleft()
            self.dropped += 1
            self.dropped_total += 1
        self.queue.append(payload)
        self.wake.set()


def _host_name(host: str) -> str:
    """The host part of a ``Host`` header (``127.0.0.1:8770``, ``[::1]:8770``, ``localhost``)."""
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end].lower() if end > 0 else host.lower()
    if host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host.lower()


def _is_loopback(name: str) -> bool:
    """True for ``localhost`` and loopback IP literals (127.0.0.0/8, ::1)."""
    name = name.strip().lower().rstrip(".")
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _json(data: Any, status: int = 200) -> web.Response:
    return web.Response(
        text=dumps(data), status=status, content_type="application/json", charset="utf-8"
    )


def _error(message: str, status: int = 400, **extra: Any) -> web.Response:
    return _json({"ok": False, "error": message, **extra}, status=status)


def _origin_of(url: str) -> str | None:
    try:
        parsed = URL(url)
    except (TypeError, ValueError):
        return None
    if parsed.scheme not in ("http", "https") or not parsed.host:
        return None
    return str(parsed.origin())


class PanelServer:
    """The panel web server (see the module docstring). Also a ``Component`` named "panel"."""

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        control: ControlSurface,
        bus: EventBus,
        static_dir: Path | None = None,
        *,
        emergency_url: str,
        emergency_token: str,
        ingest: Callable[[ChatMessage], None],
        clock: Clock | None = None,
        memory: Mapping[str, MemoryStore] | None = None,
        windows: Mapping[str, ChatWindow] | None = None,
        ops: PanelOps | None = None,
        moderation: ModerationActions | None = None,
        moderation_log: Callable[[], Iterable[Any]] | None = None,
        health_source: Callable[[], Iterable[Health]] | None = None,
        characters: Sequence[str] = (),
        default_character: str | None = None,
        ws_max_hz: float = 20.0,
        ws_queue: int = 500,
        ws_send_timeout_s: float = 2.0,
        cmd_timeout_s: float = 15.0,
        budgets: Mapping[str, Budget] | None = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError(f"the panel binds loopback only, not {host!r}")
        if not token:
            raise ValueError("the panel needs a non-empty token")
        if ws_max_hz <= 0 or ws_queue < 1:
            raise ValueError("ws_max_hz must be > 0 and ws_queue >= 1")
        self.name = "panel"
        self._host = host
        self._port = port
        self._token = token
        self._control = control
        self._bus = bus
        self._static = Path(static_dir) if static_dir is not None else _STATIC_DIR
        self._emergency_url = emergency_url.rstrip("/")
        self._emergency_token = emergency_token
        self._ingest = ingest
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._memory = dict(memory or {})
        self._windows = dict(windows or {})
        self._ops = ops
        self._moderation = moderation
        self._moderation_log = moderation_log
        self._health_source = health_source
        self._characters = list(characters) or list(self._memory) or list(self._windows)
        self._default = default_character or (self._characters[0] if self._characters else None)
        self._ws_max_hz = float(ws_max_hz)
        self._ws_queue = int(ws_queue)
        self._ws_send_timeout = ws_send_timeout_s
        self._cmd_timeout = cmd_timeout_s
        self._budgets = dict(budgets or DEFAULT_BUDGETS)
        self._runner: web.AppRunner | None = None
        self._sub: Subscription | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._clients: set[_Client] = set()
        self._health_state = Health(self.name, HealthState.STARTING, "", self._clock.now())
        # state mirrored from the bus for the snapshot and new clients
        self._health: dict[str, Any] = {}
        self._states: dict[str, str] = {}
        self._alerts: collections.deque[Any] = collections.deque(maxlen=50)
        self._traces: collections.deque[Mapping[str, Any]] = collections.deque(maxlen=20)
        self._chat: collections.deque[Any] = collections.deque(maxlen=_RECENT)
        self._mod_feed: collections.deque[Any] = collections.deque(maxlen=_RECENT)
        self._op_feed: collections.deque[Any] = collections.deque(maxlen=_RECENT)
        self._seen_alerts: collections.OrderedDict[str, None] = collections.OrderedDict()
        self.events_seen = 0
        self.serialize_errors = 0

    # --- lifecycle (Component) --------------------------------------------------------------
    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._port}"

    def health(self) -> Health:
        return self._health_state

    async def start(self) -> None:
        """Bind and serve. Raises ``OSError`` when the port is taken."""
        if self._runner is not None:
            return
        runner = web.AppRunner(
            self.build_app(), handler_cancellation=True, access_log=None, shutdown_timeout=1.0
        )
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        try:
            await site.start()
        except BaseException as exc:
            await runner.cleanup()
            self._set_health(HealthState.DOWN, f"cannot bind {self._host}:{self._port}")
            if isinstance(exc, OSError):
                log.error(
                    "panel cannot listen on %s:%d (%s); is another aivtube running? "
                    "/ แผงควบคุมเปิดพอร์ตไม่ได้ อาจมี aivtube อีกตัวทำงานอยู่",
                    self._host,
                    self._port,
                    exc,
                )
            raise
        addresses: list[Any] = list(runner.addresses)
        if addresses:
            self._port = int(addresses[0][1])
        self._runner = runner
        self._set_health(HealthState.OK, self.url)
        log.info("panel on %s", self.url)

    async def aclose(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()  # on_shutdown closes clients, on_cleanup stops the pump
        self._set_health(HealthState.DOWN, "closed")

    async def run(self) -> None:
        """Serve until cancelled (a supervised task). A crashed event pump ends the run."""
        await self.start()
        try:
            pump = self._pump_task
            if pump is None:  # pragma: no cover - start() always creates it
                raise RuntimeError("panel event pump did not start")
            await asyncio.wait({pump})
            if not pump.cancelled() and (exc := pump.exception()) is not None:
                raise exc
        finally:
            await self.aclose()

    def build_app(self) -> web.Application:
        """The aiohttp application (tests may serve it with ``aiohttp.test_utils``)."""
        app = web.Application(middlewares=[self._guard_mw], client_max_size=_MAX_BODY)
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        app.on_cleanup.append(self._on_cleanup)
        app.on_response_prepare.append(self._on_prepare)
        r = app.router
        for path in _PAGES:
            r.add_get(path, self._page)
        r.add_get("/healthz", self._healthz)
        r.add_get("/api/ping", self._ping)
        r.add_get("/api/config", self._config)
        r.add_get("/api/state", self._state)
        r.add_get("/api/snapshot", self._state)
        r.add_post("/api/cmd", self._cmd)
        r.add_post("/api/mic", self._mic)
        r.add_route("GET", "/hotkey/{name}", self._hotkey)
        r.add_route("POST", "/hotkey/{name}", self._hotkey)
        r.add_post("/api/event", self._event)
        r.add_get("/api/traces", self._traces_handler)
        r.add_get("/api/audit", self._audit_handler)
        r.add_get("/api/chat", self._chat_handler)
        r.add_get("/api/memory", self._memory_list)
        r.add_patch("/api/memory/{id}", self._memory_patch)
        r.add_delete("/api/memory/{id}", self._memory_delete)
        r.add_get("/api/moderation", self._moderation_feed)
        r.add_post("/api/moderation", self._moderation_handler)
        r.add_get("/ws", self._ws)
        return app

    def _set_health(self, state: HealthState, detail: str) -> None:
        if state is not self._health_state.state or detail != self._health_state.detail:
            self._health_state = Health(self.name, state, detail, self._clock.now())

    async def _on_startup(self, app: web.Application) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._sub = self._bus.subscribe(
                name="panel", maxsize=4096, overflow=Overflow.DROP_OLDEST
            )
            self._pump_task = asyncio.get_running_loop().create_task(
                self._pump(self._sub), name="panel-pump"
            )

    async def _on_shutdown(self, app: web.Application) -> None:
        clients = list(self._clients)
        if clients:
            await asyncio.gather(
                *(self._close_ws(c.ws, WSCloseCode.GOING_AWAY, b"panel shutdown") for c in clients)
            )

    async def _on_cleanup(self, app: web.Application) -> None:
        sub, self._sub = self._sub, None
        if sub is not None:
            sub.close()
        pump, self._pump_task = self._pump_task, None
        if pump is not None and not pump.done():
            pump.cancel()
            await asyncio.wait({pump})

    async def _close_ws(self, ws: web.WebSocketResponse, code: int, message: bytes) -> None:
        if ws.closed:
            return
        try:
            async with deadline(2.0, what="panel ws close", clock=self._clock):
                await ws.close(code=code, message=message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # DeadlineExceeded, ConnectionResetError (RST on Windows), ...
            log.debug("closing a panel websocket failed: %r", exc)

    # --- security ---------------------------------------------------------------------------
    def _authorized(self, request: web.Request) -> bool:
        given = request.headers.get(_TOKEN_HEADER, "")
        if not given:
            auth = request.headers.get("Authorization", "")
            if auth[:7].lower() == "bearer ":
                given = auth[7:].strip()
        if not given:
            given = request.query.get("token", "")
        return bool(given) and hmac.compare_digest(given.encode(), self._token.encode())

    def _origin_ok(self, request: web.Request) -> bool:
        origin = request.headers.get("Origin")
        if origin is None:
            return True  # not a browser cross-origin request (curl, AutoHotkey, Stream Deck)
        try:
            parsed = URL(origin)
        except (TypeError, ValueError):
            return False
        if parsed.scheme != "http" or not _is_loopback(parsed.host or ""):
            return False
        return parsed.port == request.url.port

    @web.middleware
    async def _guard_mw(
        self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        remote = request.remote
        if remote and not _is_loopback(remote):
            return _error("loopback only", 403)
        host = request.headers.get("Host")
        if host is not None and not _is_loopback(_host_name(host)):
            return _error("bad Host header", 403)
        if not self._origin_ok(request):
            return _error("forbidden origin", 403)
        path = request.path
        protected = path.startswith(("/api/", "/hotkey/")) or path == "/ws"
        if protected and not self._authorized(request):
            return _error("unauthorized", 401)
        try:
            return await handler(request)
        except _BadRequest as exc:
            return _error(str(exc), exc.status)
        except web.HTTPException:
            raise
        except DeadlineExceeded as exc:
            log.warning("panel request %s %s timed out: %s", request.method, path, exc.what)
            return _error(f"timeout: {exc.what}", 504)
        except Exception:
            log.exception("panel request %s %s failed", request.method, path)
            return _error("internal error", 500)

    async def _on_prepare(self, request: web.Request, response: web.StreamResponse) -> None:
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "no-referrer")
        if response.content_type == "text/html":
            headers["Cache-Control"] = "no-cache"
            headers["Content-Security-Policy"] = self._csp()
        else:
            headers.setdefault("Cache-Control", "no-store")

    def _csp(self) -> str:
        connect = ["'self'", "ws://127.0.0.1:*", "ws://localhost:*"]
        emergency = _origin_of(self._emergency_url)
        if emergency:
            connect.append(emergency)
        return (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            f"img-src 'self' data:; connect-src {' '.join(connect)}; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'"
        )

    # --- pages and health -------------------------------------------------------------------
    async def _page(self, request: web.Request) -> web.StreamResponse:
        path = self._static / _PAGES[request.path]
        if not path.is_file():
            return _error("page not found", 404)
        return web.FileResponse(path, headers={"Content-Type": "text/html; charset=utf-8"})

    async def _healthz(self, request: web.Request) -> web.Response:
        return _json({"ok": True, "now": self._clock.now(), "clients": len(self._clients)})

    async def _ping(self, request: web.Request) -> web.Response:
        return _json({"ok": True, "now": self._clock.now()})

    async def _config(self, request: web.Request) -> web.Response:
        return _json(
            {
                "emergency_url": self._emergency_url,
                "emergency_token": self._emergency_token,
                "characters": self._characters,
                "default_character": self._default,
                "ws_max_hz": self._ws_max_hz,
                "hotkeys": list(hotkey_names()),
                "budgets": to_jsonable(self._budgets),
                "memory": bool(self._memory),
                "memory_edit_fields": self._memory_edit_fields(),
                "ops": self._ops is not None,
                "moderation": self._moderation is not None,
                "moderation_log": self._moderation_log is not None or self._ops is not None,
                "event_types": sorted(EVENT_TYPES),
            }
        )

    def _control_snapshot(self) -> Any:
        try:
            return to_jsonable(self._control.snapshot())
        except Exception:
            log.exception("control snapshot failed")
            return {"error": "snapshot failed"}

    async def _state(self, request: web.Request) -> web.Response:
        return _json(
            {
                "ok": True,
                "now": self._clock.now(),
                "wall": self._clock.wall(),
                "characters": self._characters,
                "default_character": self._default,
                "control": self._control_snapshot(),
                "health": self._health_list(),
                "states": dict(self._states),
                "recent": {
                    "alerts": list(self._alerts),
                    "chat": list(self._chat),
                    "moderation": list(self._mod_feed),
                    "operator": list(self._op_feed),
                },
                "ws": {
                    "clients": len(self._clients),
                    "dropped": sum(c.dropped_total for c in self._clients),
                    "bus_dropped": self._sub.dropped if self._sub is not None else 0,
                },
            }
        )

    def _health_list(self) -> list[dict[str, Any]]:
        """Component health: ``health_source`` (current state, e.g. supervisor status) merged
        with ``HealthChanged`` events seen on the bus; the newer ``since`` wins."""
        merged: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = [_health_json(self._health_state)]
        if self._health_source is not None:
            try:
                entries += [_health_json(h) for h in self._health_source()]
            except Exception:
                log.exception("panel health source failed")
        entries += list(self._health.values())
        for entry in entries:
            name = str(entry.get("component"))
            old = merged.get(name)
            if old is None or float(entry.get("since") or 0.0) >= float(old.get("since") or 0.0):
                merged[name] = entry
        return [merged[k] for k in sorted(merged)]

    # --- commands ---------------------------------------------------------------------------
    async def _body(self, request: web.Request) -> Any:
        if not request.can_read_body:
            return {}
        try:
            return await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _BadRequest("body is not valid JSON") from None

    async def _execute(self, cmd: OpCommand) -> web.Response:
        try:
            async with deadline(self._cmd_timeout, what=f"op {cmd.kind.value}", clock=self._clock):
                result = await self._control.execute(cmd)
        except DeadlineExceeded:
            return _json(
                {"ok": False, "detail": "timeout", "kind": cmd.kind.value, "id": cmd.id}, 504
            )
        body = {
            "ok": result.ok,
            "detail": result.detail,
            "latency_ms": result.latency_ms,
            "kind": cmd.kind.value,
            "id": cmd.id,
        }
        return _json(body, 200 if result.ok else 409)

    async def _cmd(self, request: web.Request) -> web.Response:
        try:
            cmd = parse_op_command(await self._body(request), operator="panel")
        except ValueError as exc:
            return _error(str(exc))
        return await self._execute(cmd)

    async def _mic(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        if not isinstance(body, Mapping) or ("mode" in body) == ("ptt" in body):
            return _error("send exactly one of {mode} or {ptt}")
        character = body.get("character")
        payload = (
            {"kind": OpKind.MIC_MODE.value, "args": {"mode": body["mode"]}}
            if "mode" in body
            else {"kind": OpKind.PTT.value, "args": {"active": body["ptt"]}}
        )
        if character is not None:
            payload["character"] = character
        try:
            cmd = parse_op_command(payload, operator="panel")
        except ValueError as exc:
            return _error(str(exc))
        return await self._execute(cmd)

    async def _hotkey(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        try:
            snapshot = self._control.snapshot()
        except Exception:
            log.exception("control snapshot failed")
            snapshot = {}
        try:
            cmd = hotkey_command(
                name, snapshot, operator="hotkey", character=request.query.get("character") or None
            )
        except KeyError:
            return _error(f"unknown hotkey {name!r}", 404, hotkeys=list(hotkey_names()))
        return await self._execute(cmd)

    # --- alert ingest -----------------------------------------------------------------------
    async def _event(self, request: web.Request) -> web.Response:
        try:
            msg = alert_message(await self._body(request), clock=self._clock)
        except ValueError as exc:
            return _error(str(exc))
        if msg.id.startswith("alert:"):
            if msg.id in self._seen_alerts:
                return _json({"ok": True, "id": msg.id, "duplicate": True})
            self._seen_alerts[msg.id] = None
            while len(self._seen_alerts) > 512:
                self._seen_alerts.popitem(last=False)
        try:
            self._ingest(msg)
        except Exception:
            log.exception("alert ingest failed")
            self._seen_alerts.pop(msg.id, None)
            return _error("ingest failed", 503)
        log.info(
            "alert %s from %s (%s %s)", msg.kind.value, msg.user.name, msg.amount, msg.currency
        )
        return _json({"ok": True, "id": msg.id, "kind": msg.kind.value})

    # --- traces and audit -------------------------------------------------------------------
    @staticmethod
    def _int_query(request: web.Request, name: str, default: int, lo: int, hi: int) -> int:
        raw = request.query.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            raise _BadRequest(f"{name} must be an integer") from None
        return min(max(value, lo), hi)

    async def _traces_handler(self, request: web.Request) -> web.Response:
        n = self._int_query(request, "n", 20, 1, 200)
        traces: list[Mapping[str, Any]] = list(self._traces)
        source = "bus"
        if self._ops is not None:
            try:
                async with deadline(5.0, what="ops recent_traces", clock=self._clock):
                    stored = list(await self._ops.recent_traces(n))
                if len(stored) >= min(n, len(traces)):  # ops.db also has earlier sessions
                    traces, source = stored, "ops"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("reading traces from ops.db failed: %r", exc)
        summary = trace_summary(traces, budgets=self._budgets, n=n)
        return _json({"ok": True, "source": source, **summary})

    async def _audit_handler(self, request: web.Request) -> web.Response:
        if self._ops is None:
            return _error("ops database not available", 503)
        table = request.query.get("table", "op_audit")
        if table not in _AUDIT_TABLES:
            return _error(f"table must be one of {', '.join(_AUDIT_TABLES)}")
        since = self._int_query(request, "since", 0, 0, 2**62)
        limit = self._int_query(request, "limit", 100, 1, 500)
        async with deadline(5.0, what="ops audit_rows", clock=self._clock):
            rows = await self._ops.audit_rows(_audit_table(table), since_id=since, limit=limit)
        return _json({"ok": True, "table": table, "rows": to_jsonable(rows)})

    async def _audit(self, command: str, args: Mapping[str, Any], result: OpResult) -> None:
        """Audit an action the panel performs itself (not through the control surface)."""
        clean = to_jsonable(dict(args))
        self._bus.publish(
            OperatorAction(kind=command, args=clean, ok=result.ok, latency_ms=result.latency_ms)
        )
        if self._ops is None:
            return
        try:
            async with deadline(2.0, what="op_audit write", clock=self._clock):
                await self._ops.log_op(
                    ts=self._clock.wall(),
                    operator="panel",
                    command=command,
                    args=clean,
                    result="ok" if result.ok else f"error: {result.detail}",
                    latency_ms=result.latency_ms,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("op_audit write failed for %s", command)

    # --- chat window ------------------------------------------------------------------------
    def _character(self, request: web.Request, body: Mapping[str, Any] | None = None) -> str:
        raw = request.query.get("character") or (body or {}).get("character") or self._default
        if not isinstance(raw, str) or not raw:
            raise _BadRequest("no character")
        return raw

    async def _chat_handler(self, request: web.Request) -> web.Response:
        character = self._character(request)
        window = self._windows.get(character)
        if window is None:
            return _error(f"no chat window for {character!r}", 404)
        scored = window.snapshot()
        count, mention = window.pending()
        return _json(
            {
                "ok": True,
                "character": character,
                "pending": count,
                "has_mention": mention,
                "window": [
                    {"message": message_json(m), "score": round(float(s), 3)} for m, s in scored
                ],
            }
        )

    # --- memory -----------------------------------------------------------------------------
    def _store(self, character: str) -> MemoryStore:
        store = self._memory.get(character)
        if store is None:
            raise _BadRequest(f"no memory store for {character!r}", 404)
        return store

    async def _memory_list(self, request: web.Request) -> web.Response:
        character = self._character(request)
        store = self._store(character)
        kind = request.query.get("kind") or None
        status = request.query.get("status") or None
        if kind is not None and kind not in _MEM_KINDS:
            return _error(f"kind must be one of {', '.join(_MEM_KINDS)}")
        if status is not None and status not in _MEM_STATUSES:
            return _error(f"status must be one of {', '.join(_MEM_STATUSES)}")
        async with deadline(5.0, what="memory list", clock=self._clock):
            items = await store.list_memories(
                kind=cast("MemKind | None", kind), status=cast("MemStatus | None", status)
            )
        return _json({"ok": True, "character": character, "items": to_jsonable(items)})

    async def _find_memory(self, store: MemoryStore, memory_id: int) -> MemoryItem | None:
        async with deadline(5.0, what="memory read", clock=self._clock):
            items = await store.list_memories()
        return next((m for m in items if m.id == memory_id), None)

    @staticmethod
    def _memory_id(request: web.Request) -> int:
        try:
            return int(request.match_info["id"])
        except ValueError:
            raise _BadRequest("memory id must be an integer") from None

    async def _memory_patch(self, request: web.Request) -> web.Response:
        memory_id = self._memory_id(request)
        body = await self._body(request)
        if not isinstance(body, Mapping):
            return _error("body must be a JSON object")
        character = self._character(request, body)
        commands: list[OpCommand] = []
        edits = {
            k: body[k]
            for k in ("text", "subject", "importance", "locked", "pinned")
            if body.get(k) is not None
        }
        if edits:
            commands.append(self._op(OpKind.MEMORY_EDIT, {"id": memory_id, **edits}, character))
        if body.get("status") is not None:
            status_args = {"id": memory_id, "status": body["status"]}
            commands.append(self._op(OpKind.MEMORY_STATUS, status_args, character))
        if not commands:
            return _error("nothing to change")
        return await self._memory_commands(commands, character, memory_id)

    async def _memory_delete(self, request: web.Request) -> web.Response:
        memory_id = self._memory_id(request)
        character = self._character(request)
        cmd = self._op(OpKind.MEMORY_STATUS, {"id": memory_id, "status": "deleted"}, character)
        return await self._memory_commands([cmd], character, memory_id)

    def _op(self, kind: OpKind, args: Mapping[str, Any], character: str | None) -> OpCommand:
        payload = {"kind": kind.value, "args": dict(args), "character": character}
        try:
            return parse_op_command(payload, operator="panel")
        except ValueError as exc:
            raise _BadRequest(str(exc)) from None

    async def _memory_commands(
        self, commands: list[OpCommand], character: str, memory_id: int
    ) -> web.Response:
        results: list[dict[str, Any]] = []
        ok = True
        for cmd in commands:
            try:
                async with deadline(self._cmd_timeout, what="memory command", clock=self._clock):
                    result = await self._control.execute(cmd)
            except DeadlineExceeded:
                result = OpResult(False, "timeout")
            results.append({"kind": cmd.kind.value, "ok": result.ok, "detail": result.detail})
            if not result.ok:
                ok = False
                break
        item = None
        store = self._memory.get(character)
        if store is not None:
            try:
                item = await self._find_memory(store, memory_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("reading memory %d back failed: %r", memory_id, exc)
        body = {"ok": ok, "results": results, "item": to_jsonable(item)}
        return _json(body, 200 if ok else 409)

    def _memory_edit_fields(self) -> list[str]:
        """Fields every store's operator ``edit`` accepts (the SPA hides the other buttons)."""
        common: set[str] | None = None
        for store in self._memory.values():
            edit = getattr(store, "edit", None)
            try:
                params = set(inspect.signature(edit).parameters) if callable(edit) else set()
            except (TypeError, ValueError):
                params = set()
            common = params if common is None else common & params
        return [f for f in _MEM_EDIT_FIELDS if common and f in common]

    # --- moderation -------------------------------------------------------------------------
    async def _moderation_feed(self, request: web.Request) -> web.Response:
        limit = self._int_query(request, "limit", 100, 1, 500)
        if self._moderation_log is not None:
            raw = request.query.get("since_ts", "")
            try:
                since_ts = float(raw) if raw else 0.0
            except ValueError:
                raise _BadRequest("since_ts must be a number") from None
            if not math.isfinite(since_ts):
                raise _BadRequest("since_ts must be a number")
            records = [_moderation_item(r) for r in list(self._moderation_log())]
            items = [i for i in records if (i["ts"] or 0.0) > since_ts][-limit:]
            return _json({"ok": True, "source": "live", "items": items})
        if self._ops is None:
            return _error("no moderation log", 503)
        since = self._int_query(request, "since", 0, 0, 2**62)
        async with deadline(5.0, what="ops moderation_log", clock=self._clock):
            rows = await self._ops.audit_rows("moderation_log", since_id=since, limit=limit)
        return _json({"ok": True, "source": "ops", "items": [_moderation_item(r) for r in rows]})

    async def _moderation_handler(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        if not isinstance(body, Mapping):
            return _error("body must be a JSON object")
        action = body.get("action")
        character = body.get("character")
        if character is not None and not isinstance(character, str):
            return _error("character must be a string")
        if action == "mute_user":
            platform, user_id = body.get("platform"), body.get("user_id")
            if not isinstance(platform, str) or not isinstance(user_id, str) or not user_id:
                return _error("mute_user needs platform and user_id")
            minutes = body.get("minutes", 10)
            if isinstance(minutes, bool) or not isinstance(minutes, int | float) or minutes <= 0:
                return _error("minutes must be a positive number")
            mute_args = {
                "platform": platform,
                "user_id": user_id,
                "name": str(body.get("name") or ""),
                "minutes": min(float(minutes), 24 * 60.0),
            }
            return await self._execute(
                OpCommand(
                    kind=OpKind.MUTE_USER,
                    args=mute_args,
                    character=character,
                    operator="panel",
                    id=new_command_id(),
                )
            )
        if self._moderation is None:
            return _error("moderation actions are not available", 503)
        t0 = self._clock.now()
        if action == "blocklist":
            term = body.get("term")
            category = body.get("category") or "custom"
            if not isinstance(term, str) or not term.strip() or len(term) > 200:
                return _error("blocklist needs a term (1-200 characters)")
            if not isinstance(category, str):
                return _error("category must be a string")
            args: dict[str, Any] = {"term": term.strip(), "category": category}
            if character:
                args["character"] = character
            call = self._moderation.add_to_blocklist(
                term.strip(), category=category, character=character
            )
        elif action == "false_positive":
            ref, note = body.get("ref"), body.get("note") or ""
            if not isinstance(ref, str) or not ref or not isinstance(note, str):
                return _error("false_positive needs a ref")
            args = {"ref": ref, "note": note[:500]}
            call = self._moderation.mark_false_positive(ref, note=note[:500])
        else:
            return _error("action must be mute_user, blocklist or false_positive")
        try:
            async with deadline(self._cmd_timeout, what=f"moderation {action}", clock=self._clock):
                result = await call
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("moderation action %s failed", action)
            result = OpResult(False, f"{type(exc).__name__}: {exc}")
        latency = round((self._clock.now() - t0) * 1000.0, 2)
        result = OpResult(result.ok, result.detail, latency)
        await self._audit(f"moderation.{action}", args, result)
        body_out = {"ok": result.ok, "detail": result.detail, "latency_ms": latency}
        return _json(body_out, 200 if result.ok else 409)

    # --- live events ------------------------------------------------------------------------
    async def _pump(self, sub: Subscription) -> None:
        """Mirror bus events into the panel state and every client's queue."""
        async for event in sub:
            try:
                self._on_event(event)
            except Exception:
                self.serialize_errors += 1
                if self.serialize_errors <= 10:
                    log.exception("panel could not mirror %s", type(event).__name__)

    def _on_event(self, event: Event) -> None:
        self.events_seen += 1
        data = event_to_json(event)
        if isinstance(event, ChatReceived | SupportReceived):
            message = data.get("message")
            if isinstance(message, dict):
                message.pop("raw", None)
            self._chat.append(data)
        elif isinstance(event, HealthChanged):
            self._health[event.health.component] = data["health"]
        elif isinstance(event, Alert):
            self._alerts.append(data)
        elif isinstance(event, TurnTraceReady):
            self._traces.append(dict(event.trace))
        elif isinstance(event, Filtered | ChatDropped):
            self._mod_feed.append(data)
        elif isinstance(event, OperatorAction):
            self._op_feed.append(data)
        elif isinstance(event, StateChanged):
            self._states[event.character or ""] = event.new
        if not self._clients:
            return
        payload = dumps(data)
        name = type(event).__name__
        for client in self._clients:
            client.offer(name, event.character, payload)

    def _hello(self) -> str:
        return dumps(
            {
                "kind": "hello",
                "now": self._clock.now(),
                "health": self._health_list(),
                "states": dict(self._states),
                "ws_max_hz": self._ws_max_hz,
            }
        )

    async def _send(self, client: _Client, frame: str) -> None:
        async with deadline(self._ws_send_timeout, what="panel ws send", clock=self._clock):
            await client.ws.send_str(frame)
        client.sent += 1

    async def _ws_writer(self, client: _Client) -> None:
        """Send batched frames at most ``ws_max_hz`` per second; close the socket on failure."""
        gap = 1.0 / self._ws_max_hz
        try:
            await self._send(client, self._hello())
            while not client.ws.closed:
                if not client.queue:
                    client.wake.clear()
                    await client.wake.wait()
                    continue
                items: list[str] = []
                while client.queue and len(items) < _MAX_BATCH:
                    items.append(client.queue.popleft())
                dropped, client.dropped = client.dropped, 0
                head = dumps({"kind": "events", "now": self._clock.now(), "dropped": dropped})
                await self._send(client, f'{head[:-1]},"events":[{",".join(items)}]}}')
                await self._clock.sleep(gap)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # slow or vanished client: DeadlineExceeded, ConnectionReset
            log.info("panel websocket dropped: %r", exc)
            await self._close_ws(client.ws, WSCloseCode.TRY_AGAIN_LATER, b"slow consumer")

    def _client_command(self, client: _Client, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if isinstance(msg, dict) and msg.get("kind") == "subscribe":
            types = msg.get("types")
            if types is None:
                client.types = None
            elif isinstance(types, list):
                client.types = frozenset(t for t in types if t in EVENT_TYPES)

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        raw_types = request.query.get("types", "")
        types: frozenset[str] | None = None
        if raw_types:
            names = {t.strip() for t in raw_types.split(",") if t.strip()}
            unknown = sorted(names - set(EVENT_TYPES))
            if unknown:
                return _error(f"unknown event types: {', '.join(unknown)}")
            types = frozenset(names)
        ws = web.WebSocketResponse(
            timeout=2.0, heartbeat=15.0, max_msg_size=64 * 1024, compress=False
        )
        await ws.prepare(request)
        client = _Client(ws, types, request.query.get("character") or None, self._ws_queue)
        self._clients.add(client)
        writer = asyncio.get_running_loop().create_task(
            self._ws_writer(client), name="panel-ws-writer"
        )
        try:
            async for msg in ws:
                if msg.type is WSMsgType.TEXT:
                    self._client_command(client, msg.data)
                elif msg.type is WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(client)
            writer.cancel()
            await asyncio.wait({writer})
            await self._close_ws(ws, WSCloseCode.GOING_AWAY, b"")
        return ws


def _health_json(health: Health) -> dict[str, Any]:
    """``Health`` as the JSON object ``HealthChanged`` events carry."""
    return {
        "component": health.component,
        "state": health.state.value,
        "detail": health.detail,
        "since": health.since,
    }


def _moderation_item(record: Any) -> dict[str, Any]:
    """A moderation feed item from a ``safety.ModerationRecord`` or a ``moderation_log`` row."""
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        row: Mapping[str, Any] = {
            f.name: getattr(record, f.name) for f in dataclasses.fields(record)
        }
    elif isinstance(record, Mapping):
        row = record
    else:
        row = {}
    sha = str(row.get("text_sha256") or "")
    raw_ts = row.get("ts")
    ts = (
        float(raw_ts)
        if isinstance(raw_ts, int | float)
        and not isinstance(raw_ts, bool)
        and math.isfinite(raw_ts)
        else None
    )
    source = row.get("source")
    chat_input = (
        row.get("direction") in ("in", "name") and isinstance(source, str) and source in _PLATFORMS
    )
    author = row.get("author")
    item: dict[str, Any] = {k: to_jsonable(row.get(k)) for k in _MOD_FIELDS}
    item.update(
        id=row.get("id") if isinstance(row.get("id"), int) else None,
        key=f"{ts}:{sha[:16]}:{row.get('direction')}:{row.get('verdict')}",
        ts=ts,
        text=str(row.get("text_masked") or ""),
        ref=sha[:16] or None,
        turn_id=row.get("turn_id"),
        platform=source if chat_input else None,
        user_id=str(author) if chat_input and author else None,
    )
    return item


def _audit_table(name: str) -> AuditTable:
    for table in _AUDIT_TABLES:
        if table == name:
            return table
    raise _BadRequest(f"unknown audit table {name!r}")
