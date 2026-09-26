"""VTSClient against FakeVTSServer (real localhost sockets): auth, matching, pipelining."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from aivtube.avatar import VTSAPIError, VTSClient, VTSDisconnected
from aivtube.infra.clock import DeadlineExceeded
from aivtube.testing.fakes import FakeVTSServer

NAME, DEV = "AI_Vtube Brain", "AI_Vtube"


def make_client(url: str, token_path: Path, **kw: Any) -> VTSClient:
    return VTSClient(url, NAME, DEV, token_path, **kw)


async def connected_client(server: FakeVTSServer, token_path: Path, **kw: Any) -> VTSClient:
    client = make_client(server.url, token_path, **kw)
    await client.connect()
    assert await client.authenticate()
    return client


async def test_token_is_requested_persisted_and_reused(
    vts_server: FakeVTSServer, token_path: Path
) -> None:
    c = await connected_client(vts_server, token_path)
    assert c.authenticated and c.connected
    assert vts_server.token_requests == 1 and vts_server.auth_requests == 1
    assert token_path.read_text(encoding="ascii") == "tok-123"
    await c.aclose()
    assert not c.connected and not c.authenticated

    c2 = await connected_client(vts_server, token_path)  # a new session reuses the token
    assert vts_server.token_requests == 1 and vts_server.auth_requests == 2
    await c2.aclose()


async def test_invalid_stored_token_is_replaced(
    vts_server: FakeVTSServer, token_path: Path
) -> None:
    token_path.parent.mkdir(parents=True)
    token_path.write_text("revoked-token", encoding="ascii")
    c = await connected_client(vts_server, token_path)
    assert vts_server.token_requests == 1 and vts_server.auth_requests == 2
    assert token_path.read_text(encoding="ascii") == "tok-123"
    await c.aclose()


async def test_denied_token_returns_false(token_path: Path) -> None:
    async with FakeVTSServer(deny_token=True) as server:
        c = make_client(server.url, token_path)
        await c.connect()
        assert await c.authenticate() is False
        assert "denied" in c.auth_error and not token_path.exists()
        await c.aclose()


async def test_request_before_auth_is_error_8(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = make_client(vts_server.url, token_path)
    await c.connect()
    with pytest.raises(VTSAPIError) as err:
        await c.request("HotkeyTriggerRequest", {"hotkeyID": "Happy"})
    assert err.value.error_id == 8
    await c.aclose()


async def test_event_carrying_subscription_id_is_not_the_response(
    vts_server: FakeVTSServer, token_path: Path
) -> None:
    c = await connected_client(vts_server, token_path)
    data = await c.subscribe("TestEvent", {"testMessageForEvent": "hi"})
    assert data["subscribedEvents"] == ["TestEvent"]  # not the early TestEvent
    await asyncio.sleep(0.1)
    events = [c.events.get_nowait() for _ in range(c.events.qsize())]
    assert [e["data"]["counter"] for e in events if e["messageType"] == "TestEvent"] == [0, 1, 2, 3]
    await c.aclose()


async def test_api_errors_raise_with_error_id(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = await connected_client(vts_server, token_path)
    reply = await c.request("HotkeyTriggerRequest", {"hotkeyID": "happy"})
    assert reply["hotkeyID"] == "hk-happy"
    with pytest.raises(VTSAPIError) as err:
        await c.request("HotkeyTriggerRequest", {"hotkeyID": "nope"})
    assert err.value.error_id == 202 and err.value.request_type == "HotkeyTriggerRequest"
    await c.aclose()


async def test_connect_uses_localhost_safe_options(token_path: Path) -> None:
    seen: dict[str, Any] = {}

    async def fake_connect(url: str, **kw: Any) -> Any:
        seen.update(kw, url=url)
        raise ConnectionRefusedError("no VTS")

    c = make_client("ws://127.0.0.1:8001", token_path, connect=fake_connect)
    with pytest.raises(ConnectionRefusedError):
        await c.connect()
    assert seen["url"] == "ws://127.0.0.1:8001"
    assert seen["proxy"] is None and seen["compression"] is None and seen["ping_interval"] is None
    assert seen["open_timeout"] == 3.0 and seen["max_size"] == 16 * 2**20
    assert not c.connected
    with pytest.raises(VTSDisconnected):
        await c.request("APIStateRequest")
    assert c.inject({"MouthOpen": 1.0}) is False


async def test_request_has_a_deadline(token_path: Path) -> None:
    async def silent(ws: ServerConnection) -> None:
        async for _ in ws:
            pass

    async with serve(silent, "127.0.0.1", 0, compression=None) as srv:
        port = next(iter(srv.sockets)).getsockname()[1]
        c = make_client(f"ws://127.0.0.1:{port}", token_path, request_timeout=0.2)
        await c.connect()
        t0 = time.perf_counter()
        with pytest.raises(DeadlineExceeded):
            await c.request("APIStateRequest")
        assert time.perf_counter() - t0 < 1.5
        assert not c._pending
        await c.aclose()


async def test_invalid_frames_are_ignored(token_path: Path) -> None:
    async def noisy(ws: ServerConnection) -> None:
        async for raw in ws:
            req = json.loads(raw)
            await ws.send("not json")
            await ws.send("[1, 2]")
            await ws.send(json.dumps({"requestID": "other", "messageType": "APIError", "data": {}}))
            reply = {"requestID": req["requestID"], "messageType": "APIStateResponse"}
            await ws.send(json.dumps({**reply, "data": {"active": True}}))

    async with serve(noisy, "127.0.0.1", 0, compression=None) as srv:
        port = next(iter(srv.sockets)).getsockname()[1]
        c = make_client(f"ws://127.0.0.1:{port}", token_path)
        await c.connect()
        assert (await c.request("APIStateRequest"))["active"] is True
        assert c.events.empty()
        await c.aclose()


async def test_at_most_8_injections_in_flight(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = await connected_client(vts_server, token_path)
    vts_server.frame_hz = 1.0  # VTS stalls: answers once per second
    await asyncio.sleep(0.05)
    before = len(vts_server.injected)
    results = [c.inject({"MouthOpen": i / 20}) for i in range(20)]
    assert results.count(True) == 8 and c.dropped_frames == 12
    assert c.in_flight == 8
    await asyncio.sleep(0.2)
    assert len(vts_server.injected) - before == 8  # the dropped frames were never sent
    deadline = time.perf_counter() + 3.0
    while c.in_flight and time.perf_counter() < deadline:
        await asyncio.sleep(0.02)
    assert c.in_flight == 0 and c.acked_frames == 8
    await c.aclose()


async def test_fire_and_forget_error_surfaces(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = await connected_client(vts_server, token_path)
    seen: list[tuple[int, str]] = []
    c.on_ff_error = lambda eid, msg: seen.append((eid, msg))
    assert c.inject({"NotAParam": 1.0, "MouthOpen": float("nan")})
    deadline = time.perf_counter() + 2.0
    while not seen and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert c.last_ff_error == 453 and seen and seen[0][0] == 453
    assert c.in_flight == 0
    injected = [p for p in vts_server.received if p["messageType"] == "InjectParameterDataRequest"]
    assert injected[-1]["data"]["parameterValues"] == [{"id": "NotAParam", "value": 1.0}]
    assert injected[-1]["data"]["faceFound"] is True and injected[-1]["data"]["mode"] == "set"
    await c.aclose()


async def test_drop_fails_pending_requests(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = await connected_client(vts_server, token_path)
    vts_server.frame_hz = 0.5
    await asyncio.sleep(0.05)
    pending = asyncio.ensure_future(c.request("APIStateRequest", timeout=5.0))
    await asyncio.sleep(0.05)
    await vts_server.drop()
    with pytest.raises(VTSDisconnected):
        await pending
    await asyncio.wait_for(c.wait_closed(), 2.0)
    assert not c.connected and not c.authenticated
    vts_server.frame_hz = 60.0
    await c.connect()  # the same client reconnects
    assert await c.authenticate() and vts_server.token_requests == 1
    await c.aclose()


@pytest.mark.timing
async def test_120_frames_at_60hz_none_dropped(vts_server: FakeVTSServer, token_path: Path) -> None:
    c = await connected_client(vts_server, token_path)
    before = len(vts_server.injected)
    t0 = time.perf_counter()
    for i in range(120):
        assert c.inject({"MouthOpen": (i % 10) / 10, "MouthSmile": 0.5})
        await asyncio.sleep(max(0.0, t0 + (i + 1) / 60 - time.perf_counter()))
    deadline = time.perf_counter() + 2.0
    while c.acked_frames < 120 and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert c.sent_frames == 120 and c.dropped_frames == 0 and c.acked_frames == 120
    assert len(vts_server.injected) - before == 120
    await c.aclose()
