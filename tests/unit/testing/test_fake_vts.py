"""FakeVTSServer: token auth, parameter injection, errors, events, frame batching, reconnects."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect

from aivtube.testing.fakes import FakeVTSServer

API = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0"}
IDENT = {"pluginName": "AI_Vtube Brain", "pluginDeveloper": "AI_Vtube"}


class Client:
    """A tiny request/response client that also collects unsolicited frames."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.frames: list[dict[str, Any]] = []

    async def send(
        self, mtype: str, data: dict[str, Any] | None = None, rid: str | None = None
    ) -> str:
        rid = rid or uuid.uuid4().hex
        await self.ws.send(
            json.dumps({**API, "requestID": rid, "messageType": mtype, "data": data or {}})
        )
        return rid

    async def recv(self) -> dict[str, Any]:
        msg: dict[str, Any] = json.loads(await asyncio.wait_for(self.ws.recv(), 2.0))
        self.frames.append(msg)
        return msg

    async def request(self, mtype: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        rid = await self.send(mtype, data)
        want = mtype.removesuffix("Request") + "Response"
        while True:
            msg = await self.recv()
            if msg["requestID"] == rid and msg["messageType"] in (want, "APIError"):
                return msg

    async def auth(self, token: str | None = None) -> bool:
        if token is None:
            token = (await self.request("AuthenticationTokenRequest", IDENT))["data"][
                "authenticationToken"
            ]
        msg = await self.request("AuthenticationRequest", {**IDENT, "authenticationToken": token})
        return bool(msg["data"]["authenticated"])


@pytest.fixture
async def vts() -> AsyncIterator[FakeVTSServer]:
    server = FakeVTSServer(frame_hz=60.0)
    await server.start()
    yield server
    await server.stop()


async def _client(server: FakeVTSServer) -> Client:
    ws = await connect(server.url, proxy=None, compression=None, ping_interval=None, open_timeout=3)
    return Client(ws)


async def test_token_flow_and_reuse(vts: FakeVTSServer) -> None:
    c = await _client(vts)
    state = await c.request("APIStateRequest")
    assert state["data"]["currentSessionAuthenticated"] is False
    denied = await c.request(
        "InjectParameterDataRequest", {"parameterValues": [{"id": "MouthOpen", "value": 1}]}
    )
    assert denied["messageType"] == "APIError" and denied["data"]["errorID"] == 8
    assert await c.auth() is True and vts.token_requests == 1
    await c.ws.close()

    c2 = await _client(vts)  # a new session reuses the stored token: no new popup
    assert await c2.auth(vts.token) is True and vts.token_requests == 1
    vts.revoke_token()
    c3 = await _client(vts)
    assert await c3.auth("tok-123") is False
    await c2.ws.close()
    await c3.ws.close()


async def test_token_request_errors() -> None:
    async with FakeVTSServer(frame_hz=0, deny_token=True) as server:
        c = await _client(server)
        msg = await c.request("AuthenticationTokenRequest", IDENT)
        assert msg["data"]["errorID"] == 50
        bad = await c.request(
            "AuthenticationTokenRequest", {"pluginName": "x", "pluginDeveloper": "AI_Vtube"}
        )
        assert bad["data"]["errorID"] == 52
        await c.ws.close()


async def test_injection_at_60hz_is_batched_per_frame(vts: FakeVTSServer) -> None:
    c = await _client(vts)
    assert await c.auth()
    n = 120
    t0 = time.perf_counter()
    for i in range(n):
        await c.send(
            "InjectParameterDataRequest",
            {
                "faceFound": True,
                "mode": "set",
                "parameterValues": [{"id": "MouthOpen", "value": (i % 10) / 10}],
            },
            rid=f"ff-{i}",
        )
        await asyncio.sleep(max(0.0, t0 + (i + 1) / 60 - time.perf_counter()))
    acks = 0
    while acks < n:
        msg = await c.recv()
        if msg["messageType"] == "InjectParameterDataResponse":
            acks += 1
    assert len(vts.injected) == n and vts.param_values["MouthOpen"] == pytest.approx(0.9)
    await c.ws.close()


async def test_injection_errors(vts: FakeVTSServer) -> None:
    vts.held_params.add("MouthSmile")
    c = await _client(vts)
    assert await c.auth()

    async def inject(values: list[dict[str, Any]], mode: str = "set") -> dict[str, Any]:
        return await c.request(
            "InjectParameterDataRequest", {"mode": mode, "parameterValues": values}
        )

    assert (await inject([{"id": "NotAParam", "value": 1}]))["data"]["errorID"] == 453
    assert (await inject([{"id": "MouthSmile", "value": 1}]))["data"]["errorID"] == 454
    assert (await inject([{"id": "MouthSmile", "value": 0.2}], mode="add"))["messageType"] == (
        "InjectParameterDataResponse"
    )
    assert (await inject([{"id": "MouthOpen", "value": 2e6}]))["data"]["errorID"] == 451
    assert (await inject([{"id": "MouthOpen", "value": 1}], mode="toggle"))["data"][
        "errorID"
    ] == 455
    assert (await inject([]))["data"]["errorID"] == 450
    created = await c.request(
        "ParameterCreationRequest",
        {"parameterName": "PailinMouthOpen", "min": 0, "max": 1, "defaultValue": 0},
    )
    assert created["data"]["parameterName"] == "PailinMouthOpen"
    assert (await inject([{"id": "PailinMouthOpen", "value": 0.7}]))["messageType"] == (
        "InjectParameterDataResponse"
    )
    await c.ws.close()


async def test_event_with_subscription_request_id_precedes_response(vts: FakeVTSServer) -> None:
    c = await _client(vts)
    assert await c.auth()
    rid = await c.send(
        "EventSubscriptionRequest", {"eventName": "TestEvent", "subscribe": True, "config": {}}
    )
    frames = [await c.recv() for _ in range(5)]
    assert all(f["requestID"] == rid for f in frames)
    types = [f["messageType"] for f in frames]
    assert types[0] == "TestEvent" and types[1] == "EventSubscriptionResponse"
    assert [f["data"]["counter"] for f in frames if f["messageType"] == "TestEvent"] == [0, 1, 2, 3]
    assert vts.emit("TestEvent", {"yourTestMessage": "later", "counter": 9}) == 1
    later = await c.recv()
    assert later["requestID"] == rid and later["data"]["counter"] == 9
    await c.ws.close()


async def test_hotkeys_expressions_and_moves(vts: FakeVTSServer) -> None:
    c = await _client(vts)
    assert await c.auth()
    await c.request(
        "EventSubscriptionRequest", {"eventName": "HotkeyTriggeredEvent", "subscribe": True}
    )
    ok = await c.request("HotkeyTriggerRequest", {"hotkeyID": "happy"})  # by name, any case
    assert ok["data"]["hotkeyID"] == "hk-happy" and vts.triggered == ["hk-happy"]
    event = await c.recv()
    assert event["messageType"] == "HotkeyTriggeredEvent" and event["data"]["hotkeyName"] == "Happy"
    missing = await c.request("HotkeyTriggerRequest", {"hotkeyID": "nope"})
    assert missing["data"]["errorID"] == 202
    listed = await c.request("HotkeysInCurrentModelRequest")
    assert {h["name"] for h in listed["data"]["availableHotkeys"]} >= {"Happy", "Wave"}

    await c.request(
        "ExpressionActivationRequest", {"expressionFile": "exp_03.exp3.json", "active": False}
    )
    await c.request(
        "ExpressionActivationRequest", {"expressionFile": "exp_05.exp3.json", "active": True}
    )
    state = await c.request("ExpressionStateRequest", {"details": False})
    active = {e["file"] for e in state["data"]["expressions"] if e["active"]}
    assert active == {"exp_05.exp3.json"}
    assert (await c.request("ExpressionActivationRequest", {"expressionFile": "bad"}))["data"][
        "errorID"
    ] == 650
    assert (await c.request("ExpressionActivationRequest", {"expressionFile": "x.exp3.json"}))[
        "data"
    ]["errorID"] == 651

    for _ in range(4):
        await c.request(
            "MoveModelRequest",
            {"timeInSeconds": 0.1, "valuesAreRelativeToModel": True, "rotation": 90},
        )
    assert vts.model_position["rotation"] == 360 and len(vts.moves) == 4
    stats = await c.request("StatisticsRequest")
    assert stats["data"]["framerate"] == 60
    unknown = await c.request("NoSuchRequest")
    assert unknown["data"]["errorID"] == 7
    await c.ws.close()


async def test_drop_closes_clients_and_accepts_reconnects(vts: FakeVTSServer) -> None:
    c = await _client(vts)
    assert await c.auth()
    await vts.drop()
    with pytest.raises(Exception):  # noqa: B017 - any ConnectionClosed flavour
        await asyncio.wait_for(c.ws.recv(), 2.0)
    c2 = await _client(vts)
    assert await c2.auth(vts.token)
    assert vts.connections_total == 2
    await c2.ws.close()
