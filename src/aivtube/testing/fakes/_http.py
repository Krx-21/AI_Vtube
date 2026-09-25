"""Shared start/stop plumbing for the aiohttp-based fixture servers."""

from __future__ import annotations

from typing import Any

from aiohttp import web


class HttpFixture:
    """Owns an aiohttp ``AppRunner`` bound to ``host:port`` (``port=0`` picks a free port)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None

    def build_app(self) -> web.Application:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._runner is not None

    async def _start_http(self) -> str:
        if self._runner is not None:
            return self.root_url
        runner = web.AppRunner(
            self.build_app(), handler_cancellation=True, access_log=None, shutdown_timeout=0.5
        )
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        addr: Any = runner.addresses[0]
        self.port = int(addr[1])
        self._runner = runner
        return self.root_url

    async def _stop_http(self) -> None:
        if self._runner is not None:
            runner, self._runner = self._runner, None
            await runner.cleanup()

    async def __aenter__(self) -> HttpFixture:
        await self._start_http()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stop_http()
