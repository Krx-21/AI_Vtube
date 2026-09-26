"""IPC handshake, clock check, loopback-only rules and reconnects (Appendix A)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from ipc_kit import TOKEN, frame, hello, running_client, running_server, wait_for
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from aivtube.contracts import ipc
from aivtube.contracts.events import HealthChanged
from aivtube.contracts.types import HealthState
from aivtube.infra import AsyncEventBus, SystemClock
from aivtube.ipc import (
    CLOSE_BAD_TOKEN,
    CLOSE_REPLACED,
    CLOSE_VERSION,
    IpcClient,
    IpcPeer,
    IpcServer,
    major_version,
)


class SkewedClock(SystemClock):
    """A perf_counter clock running ``skew`` seconds ahead (another process's view)."""

    __slots__ = ("skew",)

    def __init__(self, skew: float) -> None:
        self.skew = skew

    def now(self) -> float:
        return super().now() + self.skew


@pytest.fixture
def clock() -> SystemClock:
    return SystemClock()


@pytest.fixture
def bus(clock: SystemClock) -> AsyncEventBus:
    return AsyncEventBus(clock)


def client_for(server: IpcServer, clock: Any, **kw: Any) -> IpcClient:
    return IpcClient(server.url, kw.pop("token", TOKEN), "voice", clock, backoff=(0.05, 0.2), **kw)


async def test_good_hello_connects_measures_rtt_and_reports_ready(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    health = bus.subscribe(HealthChanged, name="t")
    ready: list[IpcPeer] = []
    async with running_server(clock, bus) as server:
        server.on_peer_ready(ready.append)
        client = client_for(server, clock, caps={"fake_audio": True})
        ups: list[int] = []
        client.on_link_up = lambda: ups.append(1)
        async with running_client(client):
            await wait_for(lambda: bool(ready))
            peer = server.peer("voice")
            assert peer is ready[0] and peer.connected and client.connected
            assert peer.caps == {"fake_audio": True} and peer.version == 1
            assert 0.0 <= peer.rtt_s < 0.5
            assert abs(peer.measured_offset_s) < 0.002  # same host: perf_counter agrees
            assert peer.clock_offset_s == 0.0 and peer.to_local(5.0) == 5.0
            assert ups == [1] and client.core_info["role"] == "core"
            assert server.url.endswith(f":{server.port}/bus")
    events = health.drain()  # type: ignore[attr-defined]
    states = [(e.health.component, e.health.state) for e in events]
    assert ("ipc.voice", HealthState.OK) in states
    assert states[-1] == ("ipc.voice", HealthState.DOWN)


async def test_wrong_token_is_rejected_with_4001(clock: SystemClock, bus: AsyncEventBus) -> None:
    lost: list[tuple[str, str]] = []
    async with running_server(clock, bus) as server:
        server.on_peer_lost(lambda role, why: lost.append((role, why)))
        client = client_for(server, clock, token="wrong-token")
        async with running_client(client):
            await wait_for(lambda: client.last_reject is not None)
            assert client.last_reject is not None
            assert client.last_reject.code == CLOSE_BAD_TOKEN
            assert not client.connected and server.peer("voice") is None
        assert server.stats[f"rejected_{CLOSE_BAD_TOKEN}"] >= 1
    assert lost == []  # never connected, so never lost


@pytest.mark.parametrize("version", [2, "2.0", "0.9"])
async def test_other_major_version_is_rejected_with_4002(
    clock: SystemClock, bus: AsyncEventBus, version: Any
) -> None:
    async with running_server(clock, bus) as server:
        client = client_for(server, clock, version=version)
        async with running_client(client):
            await wait_for(lambda: client.last_reject is not None)
            assert client.last_reject is not None and client.last_reject.code == CLOSE_VERSION


async def test_same_major_with_a_minor_is_accepted(clock: SystemClock, bus: AsyncEventBus) -> None:
    async with running_server(clock, bus) as server:
        client = client_for(server, clock, version="1.3")
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            peer = server.peer("voice")
            assert peer is not None and peer.version == "1.3"


async def test_raw_rejections_close_with_the_right_codes(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus, hello_timeout_s=0.3) as server:
        cases = [
            (hello(token="nope"), CLOSE_BAD_TOKEN),
            (hello(version=7), CLOSE_VERSION),
            ("{not json", 4000),
            (json.dumps({"v": 1, "type": ipc.HEARTBEAT, "id": "x", "ts": 0, "data": {}}), 4000),
            (frame(ipc.HELLO, {"role": "voice", "pid": 1, "version": 1, "token": TOKEN}), 4000),
        ]
        for first, code in cases:
            async with connect(server.url, proxy=None) as ws:
                await ws.send(first)
                with pytest.raises(ConnectionClosed) as info:
                    await ws.recv()
                assert info.value.rcvd is not None and info.value.rcvd.code == code, first
        async with connect(server.url, proxy=None) as ws:  # says nothing at all
            with pytest.raises(ConnectionClosed) as info:
                await ws.recv()
            assert info.value.rcvd is not None and info.value.rcvd.code == 4000


async def test_wrong_path_and_browser_origin_are_refused(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus) as server:
        with pytest.raises(InvalidStatus) as info:
            await connect(f"ws://127.0.0.1:{server.port}/other", proxy=None)
        assert info.value.response.status_code == 404
        with pytest.raises(InvalidStatus) as info:
            await connect(server.url, proxy=None, origin="http://evil.example")  # type: ignore[arg-type]
        assert info.value.response.status_code == 403


def test_loopback_only(clock: SystemClock, bus: AsyncEventBus) -> None:
    with pytest.raises(ValueError, match="loopback"):
        IpcServer("0.0.0.0", 8771, TOKEN, clock, bus)
    with pytest.raises(ValueError, match="token"):
        IpcServer("127.0.0.1", 8771, "", clock, bus)
    with pytest.raises(ValueError, match="loopback"):
        IpcClient("ws://192.168.1.5:8771/bus", TOKEN, "voice", clock)
    with pytest.raises(ValueError):
        IpcClient("wss://127.0.0.1:8771/bus", TOKEN, "voice", clock)
    IpcServer("::1", 0, TOKEN, clock, bus)
    IpcClient("ws://localhost:8771/bus", TOKEN, "voice", clock)
    assert major_version("1.2") == 1 and major_version(3) == 3
    assert major_version(True) is None and major_version("x") is None


async def test_clock_offset_over_2ms_is_warned_and_corrected(
    clock: SystemClock, bus: AsyncEventBus, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="aivtube.ipc")
    ready: list[IpcPeer] = []
    async with running_server(clock, bus) as server:
        server.on_peer_ready(ready.append)
        client = client_for(server, SkewedClock(0.010))
        async with running_client(client):
            await wait_for(lambda: bool(ready))  # ready comes after the clock check
            peer = ready[0]
            assert peer.measured_offset_s == pytest.approx(0.010, abs=0.002)
            assert peer.clock_offset_s == peer.measured_offset_s
            assert peer.to_local(100.010) == pytest.approx(100.0, abs=0.002)
    assert any("clock offset" in r.getMessage() for r in caplog.records)


async def test_client_reconnects_after_the_link_drops(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    ready: list[IpcPeer] = []
    lost: list[str] = []
    async with running_server(clock, bus) as server:
        server.on_peer_ready(ready.append)
        server.on_peer_lost(lambda role, why: lost.append(why))
        client = client_for(server, clock)
        events: list[str] = []
        client.on_link_up = lambda: events.append("up")
        client.on_link_lost = lambda: events.append("lost")
        async with running_client(client):
            await wait_for(lambda: len(ready) == 1)
            ready[0].close(reason="test drop")
            await wait_for(lambda: len(ready) == 2)
            assert events == ["up", "lost", "up"]
            assert lost and "test drop" in lost[0]
            client.disconnect()  # the worker side can drop it too
            await wait_for(lambda: len(ready) == 3)
            assert events == ["up", "lost", "up", "lost", "up"]


async def test_client_reconnects_to_a_restarted_core(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus) as server:
        port = server.port
        client = client_for(server, clock)
        ups: list[int] = []
        client.on_link_up = lambda: ups.append(1)
        async with running_client(client):
            await wait_for(lambda: client.connected)
    # the core is gone; the client keeps retrying until a new core listens on the port
    async with running_client(client):
        await asyncio.sleep(0.3)
        assert not client.connected
        again = IpcServer("127.0.0.1", port, TOKEN, clock, bus)
        task = asyncio.create_task(again.serve())
        try:
            await wait_for(lambda: again.peer("voice") is not None)
            assert client.stats["connect_failed"] >= 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_a_new_connection_for_the_same_role_replaces_the_old(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    lost: list[str] = []
    ready: list[IpcPeer] = []
    async with running_server(clock, bus) as server:
        server.on_peer_lost(lambda role, why: lost.append(why))
        server.on_peer_ready(ready.append)
        first = await connect(server.url, proxy=None)
        await first.send(hello(mid="a"))
        assert json.loads(await first.recv())["data"]["status"] == "ok"
        second = client_for(server, clock)
        async with running_client(second):
            await wait_for(lambda: len(ready) == 1)  # the raw one never answered its pings
            with pytest.raises(ConnectionClosed) as info:
                while True:
                    await first.recv()
            assert info.value.rcvd is not None and info.value.rcvd.code == CLOSE_REPLACED
            await asyncio.sleep(0.05)
            assert lost == []  # a replaced connection is not a lost peer
            assert server.peer("voice") is ready[0] and second.connected
