"""Helpers for the IPC tests: a running server, a client task and a raw (misbehaving) peer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve

from aivtube.contracts import ipc
from aivtube.contracts.infra import Clock, EventBus
from aivtube.ipc import IpcClient, IpcServer

TOKEN = "test-token-1234567890"


@contextlib.asynccontextmanager
async def running_server(clock: Clock, bus: EventBus, **kw: Any) -> AsyncIterator[IpcServer]:
    server = IpcServer("127.0.0.1", 0, kw.pop("token", TOKEN), clock, bus, **kw)
    task = asyncio.create_task(server.serve(), name="test-ipc-server")
    try:
        async with asyncio.timeout(5):
            await server.wait_started()
        yield server
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@contextlib.asynccontextmanager
async def running_client(client: IpcClient) -> AsyncIterator[IpcClient]:
    task = asyncio.create_task(client.run(), name="test-ipc-client")
    try:
        yield client
    finally:
        await client.aclose()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def wait_for(pred: Callable[[], bool], timeout: float = 5.0, step: float = 0.01) -> None:
    """Poll ``pred`` in real time."""
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(step)


def frame(
    mtype: str, data: dict[str, Any], *, mid: str = "raw-1", corr: str | None = None, v: int = 1
) -> str:
    return json.dumps(
        {"v": v, "type": mtype, "id": mid, "corr": corr, "ts": 1.0, "data": data},
        ensure_ascii=False,
    )


def hello(token: str = TOKEN, version: Any = 1, role: str = "voice", mid: str = "h-1") -> str:
    data = {"role": role, "pid": os.getpid(), "version": version, "token": token, "caps": {}}
    return frame(ipc.HELLO, data, mid=mid)


@dataclass
class RawPeer:
    """A hand-driven client: says hello, answers pings (optionally) and never heartbeats."""

    ws: ClientConnection
    answer_pings: bool = True
    received: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    async def _pump(self) -> None:
        with contextlib.suppress(Exception):
            async for raw in self.ws:
                msg = json.loads(raw)
                self.received.append(msg)
                if msg["type"] == ipc.PING and self.answer_pings:
                    await self.ws.send(
                        frame(ipc.PONG, {"ping_ts": msg["ts"], "seq": 0}, mid="p", corr=msg["id"])
                    )

    def start(self) -> None:
        self.task = asyncio.create_task(self._pump())

    def types(self) -> list[str]:
        return [m["type"] for m in self.received]

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


async def raw_peer(url: str, *, token: str = TOKEN, answer_pings: bool = True) -> RawPeer:
    ws = await connect(url, proxy=None, ping_interval=None, compression=None)
    await ws.send(hello(token))
    reply = json.loads(await ws.recv())
    assert reply["type"] == ipc.REPLY and reply["data"]["status"] == "ok", reply
    peer = RawPeer(ws, answer_pings=answer_pings)
    peer.start()
    return peer


@contextlib.asynccontextmanager
async def silent_core(token: str = TOKEN) -> AsyncIterator[str]:
    """A fake core that accepts the hello and then says nothing at all."""

    async def handler(ws: ServerConnection) -> None:
        msg = json.loads(await ws.recv())
        await ws.send(frame(ipc.REPLY, {"status": "ok"}, mid="r", corr=msg["id"]))
        with contextlib.suppress(Exception):
            async for _ in ws:
                pass

    async with serve(handler, "127.0.0.1", 0, ping_interval=None, compression=None) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}/bus"
