"""The emergency endpoint on ``127.0.0.1:<ports.emergency>`` (8779, §2.11), ``http.server``.

It works when the core is wedged: it lives in the launcher, which only runs the standard
library. Routes (all need the token, sent as ``Authorization: Bearer``, ``X-Aivtube-Token`` or
``?token=``; anything else gets 403):

- ``POST /hardkill``: ``TerminateProcess`` on the voice worker, held down until ``/rearm``; the
  core is asked to FREEZE (best effort, in the background, so the kill never waits for it).
- ``POST /rearm``: release and restart the voice worker.
- ``POST /restart/<name>``: restart ``core``/``voice`` or a llama server by its config name.
- ``POST /llm/ensure/<server>?timeout_s=``: start or adopt a llama server and wait until it is
  ready (200) or not (503). ``POST /llm/stop/<server>``.
- ``GET /status``: children, restarts, llama servers and GPU telemetry as JSON.

Only the panel's origin (``http://127.0.0.1:<panel>`` / ``http://localhost:<panel>``) gets CORS
headers; a request carrying any other ``Origin`` is refused, and so is a ``Host`` that is not
loopback (DNS rebinding).
"""

from __future__ import annotations

import hmac
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

__all__ = ["EmergencyServer", "notify_freeze"]

log = logging.getLogger("aivtube.launcher.emergency")

MAX_BODY = 64 * 1024
"""Request bodies are not used (parameters travel in the query); bigger ones get 413."""
SOCKET_TIMEOUT_S = 10.0
VOICE = "voice"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _Supervisor(Protocol):
    @property
    def names(self) -> list[str]: ...

    def terminate(self, name: str, *, hold: bool) -> float: ...

    def rearm(self, name: str) -> bool: ...

    def restart(self, name: str) -> bool: ...

    def status(self) -> dict[str, dict[str, Any]]: ...


class _Llama(Protocol):
    def ensure(self, timeout_s: float) -> bool: ...

    def stop(self) -> None: ...

    def restart(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...


def notify_freeze(panel_url: str, token: str, *, reason: str, timeout_s: float = 1.0) -> bool:
    """``POST <panel>/api/cmd {"kind": "freeze"}``; ``False`` if the core did not accept it."""
    body = json.dumps(
        {"kind": "freeze", "args": {"reason": reason}, "operator": "launcher"}
    ).encode()
    req = urllib.request.Request(
        panel_url.rstrip("/") + "/api/cmd",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Aivtube-Token": token,
            "Origin": panel_url.rstrip("/"),
        },
    )
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            resp.read(4096)
            return bool(200 <= resp.status < 300)
    except (urllib.error.URLError, OSError, ValueError):
        return False


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR on Windows lets a second process bind the same port: never set it there.
    # (SO_EXCLUSIVEADDRUSE is not used either: it blocks a quick rebind after a restart while
    # closed connections linger in TIME_WAIT.)
    allow_reuse_address = sys.platform != "win32"
    request_queue_size = 16


class EmergencyServer:
    """Serves the routes above on its own thread (see the module docstring)."""

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        supervisor: _Supervisor,
        llama: Mapping[str, _Llama] | None = None,
        *,
        panel_port: int | None = None,
        freeze: Callable[[str], bool] | None = None,
        gpu: Callable[[], Mapping[str, Any]] | None = None,
        extra_status: Callable[[], Mapping[str, Any]] | None = None,
        voice: str = VOICE,
    ) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("the emergency endpoint binds loopback only")
        if not token:
            raise ValueError("the emergency endpoint needs a token")
        self.host = host
        self.port = port
        self.token = token
        self.supervisor = supervisor
        self.llama = dict(llama or {})
        self.panel_port = panel_port
        self.freeze = freeze
        self.gpu = gpu
        self.extra_status = extra_status
        self.voice = voice
        self.hardkills = 0
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Bind and serve on a daemon thread. Raises ``OSError`` if the port is taken."""
        if self._httpd is not None:
            return
        handler = self._handler_class()
        httpd = _Server((self.host, self.port), handler)
        self.port = int(httpd.server_address[1])
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, name="emergency-http",
            daemon=True,
        )
        self._thread.start()
        log.info("emergency endpoint on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- actions (also used by the console keys) ---------------------------------------------

    def hardkill(self, reason: str = "hardkill") -> dict[str, Any]:
        """Kill and hold the voice worker, then ask the core to FREEZE in the background."""
        present = self.voice in self.supervisor.names
        elapsed = self.supervisor.terminate(self.voice, hold=True) if present else 0.0
        self.hardkills += 1
        if self.freeze is not None:
            freeze = self.freeze
            threading.Thread(
                target=self._freeze, args=(freeze, reason), name="freeze-notify", daemon=True
            ).start()
        return {"ok": True, "voice": "down" if present else "absent",
                "elapsed_ms": round(elapsed * 1000, 1)}

    @staticmethod
    def _freeze(freeze: Callable[[str], bool], reason: str) -> None:
        try:
            ok = freeze(reason)
        except Exception:
            log.exception("FREEZE notification failed")
            return
        log.info("core FREEZE after hard kill: %s", "accepted" if ok else "not reachable")

    def rearm(self) -> dict[str, Any]:
        if self.voice not in self.supervisor.names:
            return {"ok": False, "voice": "absent"}
        ok = self.supervisor.rearm(self.voice)
        return {"ok": ok, "voice": "up" if ok else self.supervisor.status()[self.voice]["state"]}

    def status(self) -> dict[str, Any]:
        children = self.supervisor.status()
        voice = children.get(self.voice)
        if voice is None:
            word = "absent"
        else:
            word = "down" if voice.get("held") or voice.get("state") != "running" else "up"
        out: dict[str, Any] = {
            "ok": True,
            "voice": word,
            "held": bool(voice and voice.get("held")),
            "children": children,
            "restarts": {n: c.get("restarts", 0) for n, c in children.items()},
            "llm": {n: c.status() for n, c in self.llama.items()},
            "vram": dict(self.gpu()) if self.gpu is not None else {},
            "hardkills": self.hardkills,
            "t": time.perf_counter(),
        }
        if self.extra_status is not None:
            out.update(self.extra_status())
        return out

    # -- HTTP --------------------------------------------------------------------------------

    def _allowed_origins(self) -> set[str]:
        if self.panel_port is None:
            return set()
        return {f"http://127.0.0.1:{self.panel_port}", f"http://localhost:{self.panel_port}"}

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "aivtube-launcher"
            sys_version = ""
            protocol_version = "HTTP/1.1"
            timeout = SOCKET_TIMEOUT_S  # an idle client cannot pin a handler thread

            def log_message(self, format: str, *args: Any) -> None:
                log.debug("%s %s", self.address_string(), format % args)

            # plumbing ----------------------------------------------------------------------

            def _drain(self) -> bool:
                """Read and drop the body; ``False`` (after a 413) when it is too large.

                Always read it all: on Windows, closing a socket with unread input sends RST
                and the client loses our response.
                """
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if length > MAX_BODY:
                    self._send(413, {"ok": False, "error": "body too large"})
                    return False
                remaining = max(0, length)
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 16384))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                return True

            def _cors(self) -> None:
                origin = self.headers.get("Origin")
                if origin and origin in server._allowed_origins():
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")

            def _send(self, code: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self._cors()
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
                self.close_connection = True

            def _origin_ok(self) -> bool:
                origin = self.headers.get("Origin")
                return origin is None or origin in server._allowed_origins()

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").strip().lower()
                if not host:
                    return True
                name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
                return name in ("127.0.0.1", "localhost", "[::1]")

            def _authorised(self, query: Mapping[str, list[str]]) -> bool:
                auth = self.headers.get("Authorization", "")
                candidates = [
                    auth[7:].strip() if auth[:7].lower() == "bearer " else "",
                    self.headers.get("X-Aivtube-Token", ""),
                    (query.get("token") or [""])[0],
                ]
                return any(
                    c and hmac.compare_digest(c.encode(), server.token.encode()) for c in candidates
                )

            def _gate(self) -> tuple[str, dict[str, list[str]]] | None:
                parts = urlsplit(self.path)
                query = parse_qs(parts.query)
                if not self._host_ok() or not self._origin_ok():
                    self._send(403, {"ok": False, "error": "forbidden origin"})
                    return None
                if not self._authorised(query):
                    self._send(403, {"ok": False, "error": "forbidden"})
                    return None
                return parts.path.rstrip("/") or "/", query

            # verbs -------------------------------------------------------------------------

            def do_OPTIONS(self) -> None:
                if not self._drain():
                    return
                origin = self.headers.get("Origin")
                if not origin or origin not in server._allowed_origins():
                    self._send(403, {"ok": False, "error": "forbidden origin"})
                    return
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers", "Authorization, X-Aivtube-Token, Content-Type"
                )
                self.send_header("Access-Control-Max-Age", "600")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def do_GET(self) -> None:
                if not self._drain():
                    return
                gated = self._gate()
                if gated is None:
                    return
                path, _ = gated
                if path == "/status":
                    self._send(200, server.status())
                else:
                    self._send(404, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:
                if not self._drain():
                    return
                gated = self._gate()
                if gated is None:
                    return
                path, query = gated
                try:
                    self._route_post(path, query)
                except Exception as exc:  # never let a handler kill the endpoint
                    log.exception("emergency %s failed", path)
                    self._send(500, {"ok": False, "error": type(exc).__name__})

            def _route_post(self, path: str, query: Mapping[str, list[str]]) -> None:
                if path == "/hardkill":
                    self._send(200, server.hardkill())
                elif path == "/rearm":
                    result = server.rearm()
                    self._send(200 if result["ok"] else 409, result)
                elif path.startswith("/restart/"):
                    name = path.removeprefix("/restart/")
                    if name in server.llama:
                        ok = server.llama[name].restart()
                    elif name in server.supervisor.names:
                        ok = server.supervisor.restart(name)
                    else:
                        self._send(404, {"ok": False, "error": f"unknown process {name!r}"})
                        return
                    self._send(200 if ok else 409, {"ok": ok, "name": name})
                elif path.startswith("/llm/ensure/"):
                    name = path.removeprefix("/llm/ensure/")
                    llama = server.llama.get(name)
                    if llama is None:
                        self._send(404, {"ok": False, "server": name, "error": "unknown server"})
                        return
                    try:
                        timeout = float((query.get("timeout_s") or ["60"])[0])
                    except ValueError:
                        timeout = 60.0
                    ok = llama.ensure(min(max(timeout, 0.0), 600.0))
                    self._send(200 if ok else 503, {"ok": ok, "server": name, **llama.status()})
                elif path.startswith("/llm/stop/"):
                    name = path.removeprefix("/llm/stop/")
                    llama = server.llama.get(name)
                    if llama is None:
                        self._send(404, {"ok": False, "server": name, "error": "unknown server"})
                        return
                    llama.stop()
                    self._send(200, {"ok": True, "server": name})
                else:
                    self._send(404, {"ok": False, "error": "not found"})

        return Handler
