"""VTSSink against FakeVTSServer: contract suite, expressions, reconnect, 454 fallback, re-auth."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from aivtube.avatar import VTSClient, VTSDiscovery, VTSSink
from aivtube.contracts.avatar import AvatarSink
from aivtube.contracts.types import HealthState
from aivtube.infra import SystemClock
from aivtube.testing.contracts import avatar_sink_suite, case_id
from aivtube.testing.fakes import FakeVTSServer

Spawn = Callable[[Coroutine[Any, Any, Any]], asyncio.Task[Any]]

PAILIN_MAP: dict[str, Any] = {
    "neutral": {"expressions": [], "smile": 0.5, "brows": 0.5},
    "happy": {"expressions": ["exp_03.exp3.json"], "smile": 0.9, "brows": 0.7},
    "sad": {"expressions": ["exp_05.exp3.json"], "smile": 0.15, "brows": 0.2},
}


async def until(predicate: Callable[[], bool], timeout: float = 5.0, *, what: str = "") -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise AssertionError(f"timed out waiting for {what or 'condition'}")
        await asyncio.sleep(0.01)


def make_sink(
    url: str,
    token_path: Path,
    *,
    emotion_map: dict[str, Any] | None = None,
    request_timeout: float = 2.0,
    **kw: Any,
) -> VTSSink:
    client = VTSClient(
        url, "AI_Vtube Brain", "AI_Vtube", token_path, request_timeout=request_timeout
    )
    kw.setdefault("reconnect_backoff", (0.05, 0.2))
    kw.setdefault("poll_s", 0.05)
    return VTSSink(client, PAILIN_MAP if emotion_map is None else emotion_map, SystemClock(), **kw)


async def running(sink: VTSSink, spawn: Spawn) -> VTSSink:
    spawn(sink.run())
    await until(lambda: sink.connected, what="sink connected")
    return sink


def requests_of(server: FakeVTSServer, mtype: str) -> list[dict[str, Any]]:
    return [r for r in server.received if r.get("messageType") == mtype]


# --- contract suite --------------------------------------------------------------------------

_ENV: dict[str, Any] = {}


@pytest.fixture
async def suite_env(vts_server: FakeVTSServer, token_path: Path) -> Any:
    _ENV.update(server=vts_server, token=token_path)
    yield vts_server
    _ENV.clear()


def _suite_factory() -> VTSSink:
    return make_sink(_ENV["server"].url, _ENV["token"])


@pytest.mark.parametrize("case", avatar_sink_suite(_suite_factory, hotkey="happy"), ids=case_id)
async def test_vts_sink_contract(case: Callable[[], Awaitable[None]], suite_env: Any) -> None:
    await case()


# --- behaviour -------------------------------------------------------------------------------


async def test_connects_authenticates_and_injects(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = make_sink(vts_server.url, token_path)
    assert isinstance(sink, AvatarSink)
    assert sink.health().state is HealthState.STARTING and not sink.connected
    await running(sink, spawn)
    health = sink.health()
    assert health.state is HealthState.OK and health.component == "avatar:vts"
    sink.set_params({"MouthOpen": 0.4, "MouthSmile": 0.6})
    await until(lambda: vts_server.param_values.get("MouthOpen") == 0.4, what="injection")
    frame = vts_server.injected[-1]
    assert frame["mode"] == "set" and frame["faceFound"] is True
    assert requests_of(vts_server, "EventSubscriptionRequest")[0]["data"]["eventName"] == (
        "ModelLoadedEvent"
    )


async def test_set_emotion_uses_explicit_expression_state(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path), spawn)
    await sink.set_emotion("happy")
    await sink.set_emotion("happy")  # idempotent: no second request
    acts = requests_of(vts_server, "ExpressionActivationRequest")
    assert [(a["data"]["expressionFile"], a["data"]["active"]) for a in acts] == [
        ("exp_03.exp3.json", True)
    ]
    assert acts[0]["data"]["fadeTime"] == 0.3
    assert vts_server.expressions["exp_03.exp3.json"] is True
    await sink.set_emotion("sad", fade_s=0.5)
    assert vts_server.expressions["exp_03.exp3.json"] is False
    assert vts_server.expressions["exp_05.exp3.json"] is True
    await sink.set_emotion("neutral")
    assert not any(vts_server.expressions.values())
    assert not requests_of(vts_server, "HotkeyTriggerRequest")  # never a toggle hotkey
    assert all(
        isinstance(a["data"]["active"], bool)
        for a in requests_of(vts_server, "ExpressionActivationRequest")
    )


async def test_emotion_set_while_offline_is_applied_on_connect(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = make_sink(vts_server.url, token_path)
    await sink.set_emotion("happy")  # not connected yet: remembered
    await running(sink, spawn)
    await until(lambda: vts_server.expressions["exp_03.exp3.json"], what="reconciled expression")
    # not driven by a driver: the emotion baseline is injected by the sink itself
    await until(lambda: vts_server.param_values.get("MouthSmile") == 0.9, what="baseline")
    assert vts_server.param_values.get("Brows") == 0.7


async def test_missing_expression_is_reported_once(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    emap = {"neutral": {}, "happy": {"expressions": ["exp_99.exp3.json"]}}
    sink = await running(make_sink(vts_server.url, token_path, emotion_map=emap), spawn)
    await sink.set_emotion("happy")
    await sink.set_emotion("neutral")
    await sink.set_emotion("happy")
    # at most one attempt (if it raced the initial reconcile); afterwards it is known missing
    assert len(requests_of(vts_server, "ExpressionActivationRequest")) <= 1
    assert sink.connected


async def test_reconnects_after_vts_restart_with_stored_token(
    token_path: Path, spawn: Spawn
) -> None:
    server = FakeVTSServer(frame_hz=60.0)
    await server.start()
    port = server.port
    sink = await running(make_sink(server.url, token_path, hold_s=30.0), spawn)
    await sink.set_emotion("happy")
    sink.set_params({"MouthOpen": 0.3, "MouthSmile": 0.5})
    await until(lambda: server.param_values.get("MouthOpen") == 0.3, what="first injection")
    await server.stop()
    await until(lambda: not sink.connected, what="disconnect noticed")
    await until(lambda: sink.health().state is HealthState.DOWN, what="DOWN")
    assert "unreachable" in sink.health().detail or "closed" in sink.health().detail

    restarted = FakeVTSServer(port=port, frame_hz=60.0)  # VTS comes back, fresh state
    await restarted.start()
    try:
        await until(lambda: sink.connected, timeout=5.0, what="reconnected")
        assert restarted.token_requests == 0 and restarted.auth_requests == 1  # token reused
        await until(lambda: restarted.expressions["exp_03.exp3.json"], what="expressions re-synced")
        await until(lambda: restarted.param_values.get("MouthOpen") == 0.3, what="state restored")
        assert sink.connects == 2 and sink.health().state is HealthState.OK
    finally:
        await restarted.stop()


async def test_param_held_by_other_plugin_switches_to_custom_param(
    token_path: Path, spawn: Spawn
) -> None:
    async with FakeVTSServer(held_params={"MouthOpen"}) as server:
        sink = await running(make_sink(server.url, token_path), spawn)
        for i in range(60):
            sink.set_params({"MouthOpen": 0.5, "MouthSmile": 0.4 + i / 1000})
            if "PailinMouthOpen" in server.param_values:
                break
            await asyncio.sleep(1 / 30)
        await until(lambda: "PailinMouthOpen" in server.param_values, what="custom param used")
        assert sink.client.last_ff_error == 454
        assert server.created_params["PailinMouthOpen"]["min"] == 0.0
        assert sink.remapped == {"MouthOpen": "PailinMouthOpen"}
        health = sink.health()
        assert health.state is HealthState.DEGRADED and "PailinMouthOpen" in health.detail
        sink.set_params({"MouthOpen": 0.8, "MouthSmile": 0.6})
        await until(lambda: server.param_values.get("PailinMouthOpen") == 0.8, what="remapped")
        assert server.param_values["MouthSmile"] == 0.6


async def test_uninjectable_default_param_gets_custom_param(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path), spawn)
    sink.set_params({"VoiceA": 0.5, "MouthOpen": 0.2})  # VoiceA: 453 on this model
    await until(lambda: "PailinVoiceA" in vts_server.created_params, what="custom VoiceA")
    await until(lambda: vts_server.param_values.get("PailinVoiceA") == 0.5, what="VoiceA sent")
    assert sink.remapped == {"VoiceA": "PailinVoiceA"}


async def test_reauthenticates_when_token_is_revoked(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path), spawn)
    vts_server.revoke_token("tok-new")
    for _ in range(100):
        sink.set_params({"MouthOpen": 0.25})
        if vts_server.token_requests >= 2 and sink.connected:
            break
        await asyncio.sleep(0.02)
    await until(lambda: token_path.read_text(encoding="ascii") == "tok-new", what="new token")
    sink.set_params({"MouthOpen": 0.75})
    await until(lambda: vts_server.param_values.get("MouthOpen") == 0.75, what="resumed")
    assert vts_server.token_requests == 2 and vts_server.connections_total == 1


async def test_trigger_move_and_spin(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path), spawn)
    assert await sink.trigger("Wave") is True and await sink.trigger("nope") is False
    assert vts_server.triggered == ["hk-wave"]
    await sink.move(rotation=90.0, seconds=0.1)
    assert vts_server.moves[-1] == {
        "timeInSeconds": 0.1,
        "valuesAreRelativeToModel": True,
        "rotation": 90.0,
    }
    await sink.move(x=0.5, y=-0.25, size=500.0, seconds=5.0, relative=False)
    assert vts_server.moves[-1] == {
        "timeInSeconds": 2.0,
        "valuesAreRelativeToModel": False,
        "positionX": 0.5,
        "positionY": -0.25,
        "rotation": 0.0,
        "size": 100.0,
    }
    before = len(vts_server.moves)
    await sink.spin(seconds=0.2)
    spins = vts_server.moves[before:]
    assert len(spins) == 4 and all(m["rotation"] == 90.0 for m in spins)


async def test_keepalive_resends_before_the_vts_timeout(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path, keepalive_s=0.3, hold_s=1.2), spawn)
    sink.set_params({"MouthOpen": 0.1})
    await asyncio.sleep(1.0)
    count = sum(1 for f in vts_server.injected if f["parameterValues"][0]["id"] == "MouthOpen")
    assert count >= 3  # re-sent at least every 0.3-0.35 s, well inside VTS's 1 s
    await asyncio.sleep(1.0)  # held values are released after hold_s without updates
    settled = len(vts_server.injected)
    await asyncio.sleep(0.6)
    assert len(vts_server.injected) == settled


async def test_request_timeout_sends_sink_back_to_reconnect(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn
) -> None:
    sink = await running(make_sink(vts_server.url, token_path, request_timeout=0.3), spawn)
    vts_server.frame_hz = 0.5  # VTS hangs
    await asyncio.sleep(0.05)
    assert await sink.trigger("Happy") is False
    vts_server.frame_hz = 60.0
    await until(lambda: vts_server.connections_total >= 2 and sink.connected, what="reconnect")


async def test_unreachable_vts_reports_down(
    free_tcp_port: int, token_path: Path, spawn: Spawn
) -> None:
    sink = make_sink(f"ws://127.0.0.1:{free_tcp_port}", token_path)
    spawn(sink.run())
    await until(lambda: sink.health().state is HealthState.DOWN, what="DOWN")
    assert "unreachable" in sink.health().detail and not sink.connected
    assert await sink.trigger("happy") is False
    sink.set_params({"MouthOpen": 1.0})  # fire-and-forget: silently kept for later
    await sink.set_emotion("happy")
    await sink.move(rotation=10.0)


async def test_denied_plugin_reports_down(token_path: Path, spawn: Spawn) -> None:
    async with FakeVTSServer(deny_token=True) as server:
        sink = make_sink(server.url, token_path)
        spawn(sink.run())
        await until(lambda: "not authorised" in sink.health().detail, what="auth failure")
        assert sink.health().state is HealthState.DOWN


def _broadcast(port: int, title: str) -> bytes:
    data = {"active": True, "port": port, "instanceID": f"id{port}", "windowTitle": title}
    return json.dumps({"messageType": "VTubeStudioAPIStateBroadcast", "data": data}).encode()


def _send(port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(payload, ("127.0.0.1", port))


async def test_discovery_finds_the_twin_window(
    vts_server: FakeVTSServer, token_path: Path, spawn: Spawn, free_tcp_port: int
) -> None:
    disc = VTSDiscovery(port=0, host="127.0.0.1")
    sink = make_sink(
        f"ws://127.0.0.1:{free_tcp_port}",  # configured URL: nothing listens there
        token_path,
        window_title="VTube Studio Window 2",
        discovery=disc,
        discovery_timeout_s=3.0,
    )
    spawn(sink.run())
    await until(lambda: disc.running, what="discovery listening")
    _send(disc.port, _broadcast(free_tcp_port, "VTube Studio"))
    _send(disc.port, _broadcast(vts_server.port, "VTube Studio Window 2"))
    await until(lambda: sink.connected, what="connected via discovery")
    assert sink.client.url == vts_server.url


async def test_discovery_broadcast_cuts_the_backoff_short(
    token_path: Path, spawn: Spawn, free_tcp_port: int
) -> None:
    disc = VTSDiscovery(port=0, host="127.0.0.1")
    url = f"ws://127.0.0.1:{free_tcp_port}"
    sink = make_sink(url, token_path, discovery=disc, reconnect_backoff=(8.0, 10.0))
    spawn(sink.run())
    await until(lambda: sink.health().state is HealthState.DOWN, what="first failure")
    async with FakeVTSServer(port=free_tcp_port) as server:  # VTS starts during the 8 s backoff
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        _send(disc.port, _broadcast(server.port, "VTube Studio"))
        await until(lambda: sink.connected, timeout=4.0, what="woken by broadcast")
        assert loop.time() - t0 < 2.0
