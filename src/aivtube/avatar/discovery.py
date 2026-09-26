"""VTube Studio instance discovery over UDP 47779 (§2.2, avatar brief).

Every VTS instance broadcasts ``VTubeStudioAPIStateBroadcast`` ``{active, port, instanceID,
windowTitle}`` on UDP 47779 every 2 s, even while its API is off. Extra instances bind the next
free port and are titled "VTube Studio Window 2", and so on, so ports are never hard-coded:
the character's ``vts_window_title`` picks the instance (twins), with the configured
``vts_url`` as the fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from aivtube.contracts.infra import Clock
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline

__all__ = [
    "DISCOVERY_PORT",
    "MAIN_WINDOW_TITLE",
    "VTSDiscovery",
    "discover_vts",
    "parse_broadcast",
    "pick_instance",
    "resolve_vts_url",
    "url_port",
]

log = logging.getLogger("aivtube.avatar.discovery")

DISCOVERY_PORT = 47779
MAIN_WINDOW_TITLE = "VTube Studio"
_BROADCAST = "VTubeStudioAPIStateBroadcast"


def parse_broadcast(data: bytes | str, host: str = "") -> dict[str, Any] | None:
    """Parse one datagram; ``None`` unless it is a valid ``VTubeStudioAPIStateBroadcast``.

    Returns ``{"active", "port", "instanceID", "windowTitle", "host"}``.
    """
    try:
        msg = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict) or msg.get("messageType") != _BROADCAST:
        return None
    body = msg.get("data")
    if not isinstance(body, dict):
        return None
    port = body.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        return None
    return {
        "active": bool(body.get("active", False)),
        "port": port,
        "instanceID": str(body.get("instanceID") or f"{host}:{port}"),
        "windowTitle": str(body.get("windowTitle") or ""),
        "host": host,
    }


def url_port(url: str) -> int | None:
    try:
        return urlsplit(url).port
    except ValueError:
        return None


def pick_instance(
    instances: Iterable[Mapping[str, Any]],
    window_title: str = "",
    *,
    prefer_port: int | None = None,
) -> Mapping[str, Any] | None:
    """Choose the VTS instance to connect to (only instances whose API is active).

    With ``window_title``: the exact (case-insensitive) title match, else ``None``. Without:
    the instance on ``prefer_port``, else the only one, else the main "VTube Studio" window,
    else ``None`` (ambiguous: use the configured URL).
    """
    active = sorted((i for i in instances if i.get("active")), key=lambda i: int(i["port"]))
    title = window_title.strip().casefold()
    if title:
        return next((i for i in active if str(i["windowTitle"]).strip().casefold() == title), None)
    if prefer_port is not None:
        match = next((i for i in active if i["port"] == prefer_port), None)
        if match is not None:
            return match
    if len(active) == 1:
        return active[0]
    main = MAIN_WINDOW_TITLE.casefold()
    return next((i for i in active if str(i["windowTitle"]).strip().casefold() == main), None)


def resolve_vts_url(
    fallback_url: str, window_title: str, instances: Iterable[Mapping[str, Any]]
) -> str:
    """The ``ws://127.0.0.1:<port>`` of the chosen instance, else ``fallback_url``."""
    inst = pick_instance(instances, window_title, prefer_port=url_port(fallback_url))
    if inst is None:
        return fallback_url
    return f"ws://127.0.0.1:{int(inst['port'])}"


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, owner: VTSDiscovery) -> None:
        self._owner = owner

    def datagram_received(self, data: bytes, addr: tuple[str | Any, int]) -> None:
        self._owner.feed(data, str(addr[0]))

    def error_received(self, exc: Exception) -> None:
        log.debug("discovery socket error: %r", exc)


class VTSDiscovery:
    """A long-lived listener that remembers every instance it has heard recently."""

    def __init__(
        self,
        *,
        port: int = DISCOVERY_PORT,
        host: str = "0.0.0.0",
        clock: Clock | None = None,
        stale_s: float = 6.0,
    ) -> None:
        self._bind = (host, port)
        self._clock: Clock = clock or SystemClock()
        self.stale_s = stale_s
        self._seen: dict[str, tuple[float, dict[str, Any]]] = {}
        self._transport: asyncio.DatagramTransport | None = None
        self._changed = asyncio.Event()
        self.port = port
        self.received = 0

    @property
    def running(self) -> bool:
        return self._transport is not None and not self._transport.is_closing()

    async def start(self) -> None:
        """Bind the UDP socket (``SO_REUSEADDR`` so twins and VTS tools can share it)."""
        if self.running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self._bind)
            sock.setblocking(False)
        except OSError:
            sock.close()
            raise
        self.port = int(sock.getsockname()[1])
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(lambda: _Protocol(self), sock=sock)
        self._transport = transport

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def feed(self, data: bytes | str, host: str = "") -> dict[str, Any] | None:
        """Handle one datagram (the socket calls this; tests may too)."""
        inst = parse_broadcast(data, host)
        if inst is None:
            return None
        self.received += 1
        self._seen[inst["instanceID"]] = (self._clock.now(), inst)
        self._changed.set()
        return inst

    def instances(self, *, max_age_s: float | None = None) -> list[dict[str, Any]]:
        """Instances heard within ``max_age_s`` (default ``stale_s``), active or not."""
        limit = self.stale_s if max_age_s is None else max_age_s
        now = self._clock.now()
        return [dict(i) for t, i in self._seen.values() if now - t <= limit]

    def pick(
        self, window_title: str = "", *, prefer_port: int | None = None
    ) -> Mapping[str, Any] | None:
        return pick_instance(self.instances(), window_title, prefer_port=prefer_port)

    async def wait_for(
        self,
        window_title: str = "",
        *,
        timeout: float,  # noqa: ASYNC109 - bounded wait is the point of this helper
        prefer_port: int | None = None,
        after: float | None = None,
    ) -> Mapping[str, Any] | None:
        """Wait until a matching active instance is heard (after ``after``), or time out."""

        def match() -> Mapping[str, Any] | None:
            fresh = [
                i
                for t, i in self._seen.values()
                if (after is None or t > after) and self._clock.now() - t <= self.stale_s
            ]
            return pick_instance(fresh, window_title, prefer_port=prefer_port)

        found = match()
        if found is not None or timeout <= 0:
            return found
        try:
            async with deadline(timeout, what="VTS discovery", clock=self._clock):
                while found is None:
                    self._changed.clear()
                    await self._changed.wait()
                    found = match()
        except DeadlineExceeded:
            return None
        return found


async def discover_vts(
    timeout_s: float = 2.5, *, port: int = DISCOVERY_PORT, clock: Clock | None = None
) -> list[dict[str, Any]]:
    """Listen for ``timeout_s`` and return every instance heard (``[]`` if the port is busy)."""
    clock = clock or SystemClock()
    listener = VTSDiscovery(port=port, clock=clock)
    try:
        await listener.start()
    except OSError as exc:
        log.warning("VTS discovery: cannot listen on UDP %d: %s", port, exc)
        return []
    try:
        await clock.sleep(timeout_s)
    finally:
        with contextlib.suppress(Exception):
            listener.close()
    return listener.instances(max_age_s=float("inf"))
