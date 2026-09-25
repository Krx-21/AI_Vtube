"""``FakeLauncher`` (a ``LocalServerManager``) and the launcher's emergency HTTP endpoint (§2.11)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from aiohttp import web

from aivtube.contracts.infra import Clock
from aivtube.testing.fakes._http import HttpFixture

__all__ = ["DEFAULT_PROPS", "FakeEmergencyServer", "FakeLauncher"]

DEFAULT_PROPS: Mapping[str, Any] = {
    "chat_template_caps": {
        "supports_tool_calls": True,
        "supports_parallel_tool_calls": True,
        "supports_tools": True,
        "supports_system_role": True,
    },
    "model_alias": "pailin-30b",
    "model_path": "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
    "total_slots": 3,
    "build_info": "b11177-fake",
}


class FakeLauncher:
    """``LocalServerManager`` that simulates llama-server instances and slot files.

    ``fail_start`` names servers that never come up; ``start_delay_s`` delays ``ensure_running``
    (a delay longer than ``timeout_s`` fails it). Slot files live in memory: ``restore_slot``
    succeeds only for a file saved earlier on the same server.
    """

    def __init__(
        self,
        servers: Mapping[str, Mapping[str, Any]] | Sequence[str] = ("local30b", "local4b"),
        *,
        fail_start: Sequence[str] = (),
        start_delay_s: float = 0.0,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(servers, Mapping):
            self.props_by_server = {k: {**DEFAULT_PROPS, **v} for k, v in servers.items()}
        else:
            self.props_by_server = {k: dict(DEFAULT_PROPS) for k in servers}
        self.running: dict[str, bool] = {k: False for k in self.props_by_server}
        self.fail_start = set(fail_start)
        self.start_delay_s = start_delay_s
        self.saved: dict[str, set[str]] = {k: set() for k in self.props_by_server}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._sleep: Callable[[float], Awaitable[None]] = clock.sleep if clock else asyncio.sleep

    async def ensure_running(self, server: str, timeout_s: float) -> bool:
        self.calls.append(("ensure_running", (server, timeout_s)))
        if server not in self.running:
            return False
        if self.running[server]:
            return True
        if server in self.fail_start or self.start_delay_s > timeout_s:
            await self._sleep(min(timeout_s, self.start_delay_s))
            return False
        await self._sleep(self.start_delay_s)
        self.running[server] = True
        return True

    async def stop(self, server: str) -> None:
        self.calls.append(("stop", (server,)))
        if server in self.running:
            self.running[server] = False

    async def props(self, server: str) -> Mapping[str, Any]:
        self.calls.append(("props", (server,)))
        if not self.running.get(server):
            raise ConnectionError(f"{server} is not running")
        return dict(self.props_by_server[server])

    async def save_slot(self, server: str, slot: int, filename: str) -> bool:
        self.calls.append(("save_slot", (server, slot, filename)))
        if not self.running.get(server):
            return False
        self.saved[server].add(filename)
        return True

    async def restore_slot(self, server: str, slot: int, filename: str) -> bool:
        self.calls.append(("restore_slot", (server, slot, filename)))
        return bool(self.running.get(server)) and filename in self.saved.get(server, set())

    def crash(self, server: str) -> None:
        """Simulate the llama-server process dying."""
        self.running[server] = False


class FakeEmergencyServer(HttpFixture):
    """The launcher's loopback emergency endpoint (``127.0.0.1:8779``), backed by a FakeLauncher.

    Routes: ``POST /hardkill``, ``/rearm``, ``/restart/{name}``, ``/llm/ensure/{server}``,
    ``/llm/stop/{server}`` and ``GET /status``. The token is accepted as ``Authorization:
    Bearer``, ``X-Aivtube-Token`` or ``?token=``; anything else gets 403.
    """

    def __init__(
        self,
        launcher: FakeLauncher | None = None,
        *,
        token: str = "test-token",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        super().__init__(host, port)
        self.launcher = launcher or FakeLauncher()
        self.token = token
        self.hits: list[tuple[str, str]] = []
        self.voice_down = False
        self.restarts: list[str] = []

    async def start(self) -> str:
        return await self._start_http()

    async def stop(self) -> None:
        await self._stop_http()

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/hardkill", self._hardkill)
        app.router.add_post("/rearm", self._rearm)
        app.router.add_post("/restart/{name}", self._restart)
        app.router.add_post("/llm/ensure/{server}", self._ensure)
        app.router.add_post("/llm/stop/{server}", self._llm_stop)
        app.router.add_get("/status", self._status)
        return app

    def _authorised(self, request: web.Request) -> bool:
        self.hits.append((request.method, request.path))
        auth = request.headers.get("Authorization", "")
        candidates = {
            auth.removeprefix("Bearer ").strip(),
            request.headers.get("X-Aivtube-Token", ""),
            request.query.get("token", ""),
        }
        return self.token in candidates

    def _deny(self) -> web.Response:
        return web.json_response({"ok": False, "error": "forbidden"}, status=403)

    async def _hardkill(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        self.voice_down = True
        return web.json_response({"ok": True, "voice": "down"})

    async def _rearm(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        self.voice_down = False
        return web.json_response({"ok": True, "voice": "up"})

    async def _restart(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        self.restarts.append(request.match_info["name"])
        return web.json_response({"ok": True})

    async def _ensure(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        server = request.match_info["server"]
        timeout = float(request.query.get("timeout_s", "60"))
        ok = await self.launcher.ensure_running(server, timeout)
        return web.json_response({"ok": ok, "server": server}, status=200 if ok else 503)

    async def _llm_stop(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        await self.launcher.stop(request.match_info["server"])
        return web.json_response({"ok": True})

    async def _status(self, request: web.Request) -> web.Response:
        if not self._authorised(request):
            return self._deny()
        return web.json_response(
            {
                "voice": "down" if self.voice_down else "up",
                "llm": {k: ("up" if v else "down") for k, v in self.launcher.running.items()},
                "restarts": list(self.restarts),
            }
        )
