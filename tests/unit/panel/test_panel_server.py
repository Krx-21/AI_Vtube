"""``PanelServer`` over real loopback sockets: auth, routing, WS mirroring, ingest, memory."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from panel_testkit import EditableMemory, RecordingBrain

from aivtube.chat import ScoredChatWindow
from aivtube.contracts.control import OpKind, OpResult
from aivtube.contracts.events import (
    Alert,
    ChatReceived,
    Filtered,
    HealthChanged,
    SegmentStarted,
    StateChanged,
    TurnTraceReady,
)
from aivtube.contracts.infra import EventBus
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.types import ChatMessage, Health, HealthState, MsgKind
from aivtube.panel.control import CoreControl
from aivtube.panel.server import PanelServer, _Client
from aivtube.testing.fakes import (
    FakeControlSurface,
    FakeLLM,
    FakeLLMRouter,
    FakeSafetyGate,
    FakeSpeechOutput,
    FakeTaskSupervisor,
    FakeToolRegistry,
    make_chat_message,
)

TOKEN = "panel-test-token-0123456789"
EMERGENCY = "emergency-token-abcdef"


class FakeOpsDb:
    """The ``PanelOps`` subset of ``OpsDb``, in memory."""

    def __init__(self) -> None:
        self.traces: list[Mapping[str, Any]] = []
        self.rows: dict[str, list[dict[str, Any]]] = {
            "op_audit": [],
            "tool_audit": [],
            "moderation_log": [],
        }

    async def recent_traces(self, n: int = 20) -> list[Mapping[str, Any]]:
        return self.traces[-n:]

    async def audit_rows(
        self, table: Any, *, since_id: int = 0, limit: int = 100
    ) -> list[Mapping[str, Any]]:
        rows: list[Mapping[str, Any]] = [r for r in self.rows[table] if r["id"] > since_id]
        return rows[:limit]

    async def log_op(self, **row: Any) -> None:
        self.rows["op_audit"].append({"id": len(self.rows["op_audit"]) + 1, **row})


class FakeModeration:
    def __init__(self) -> None:
        self.blocked: list[tuple[str, str, str | None]] = []
        self.false_positives: list[tuple[str, str]] = []

    async def add_to_blocklist(
        self, term: str, *, category: str, character: str | None
    ) -> OpResult:
        self.blocked.append((term, category, character))
        return OpResult(True, "added")

    async def mark_false_positive(self, ref: str, *, note: str) -> OpResult:
        self.false_positives.append((ref, note))
        return OpResult(True)


@dataclass
class Panel:
    server: PanelServer
    control: Any
    bus: EventBus
    session: aiohttp.ClientSession
    ingested: list[ChatMessage] = field(default_factory=list)

    @property
    def url(self) -> str:
        return self.server.url

    def auth(self, **extra: str) -> dict[str, str]:
        return {"X-Aivtube-Token": TOKEN, **extra}

    async def get(self, path: str, **kw: Any) -> tuple[int, Any]:
        headers = kw.pop("headers", self.auth())
        async with self.session.get(self.url + path, headers=headers, **kw) as resp:
            return resp.status, await _body(resp)

    async def post(self, path: str, body: Any = None, **kw: Any) -> tuple[int, Any]:
        headers = kw.pop("headers", self.auth())
        async with self.session.post(self.url + path, json=body, headers=headers, **kw) as resp:
            return resp.status, await _body(resp)

    async def send(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        async with self.session.request(
            method, self.url + path, json=body, headers=self.auth()
        ) as resp:
            return resp.status, await _body(resp)


async def _body(resp: aiohttp.ClientResponse) -> Any:
    text = await resp.text()
    try:
        return json.loads(text)
    except ValueError:
        return text


async def _start(
    bus: EventBus, control: Any, *, ingest: Any = None, **kw: Any
) -> tuple[PanelServer, list[ChatMessage]]:
    ingested: list[ChatMessage] = []
    server = PanelServer(
        "127.0.0.1",
        0,
        TOKEN,
        control,
        bus,
        emergency_url="http://127.0.0.1:8779",
        emergency_token=EMERGENCY,
        ingest=ingest or ingested.append,
        **kw,
    )
    await server.start()
    return server, ingested


@pytest.fixture
async def panel(bus: EventBus) -> AsyncIterator[Panel]:
    control = FakeControlSurface()
    server, ingested = await _start(bus, control, characters=["pailin"])
    session = aiohttp.ClientSession()
    try:
        yield Panel(server, control, bus, session, ingested)
    finally:
        await session.close()
        await server.aclose()


# --- construction and binding ---------------------------------------------------------------


def test_panel_refuses_non_loopback_hosts_and_empty_tokens(bus: EventBus) -> None:
    kw: dict[str, Any] = {
        "emergency_url": "http://127.0.0.1:8779",
        "emergency_token": "x",
        "ingest": lambda m: None,
    }
    with pytest.raises(ValueError, match="loopback"):
        PanelServer("0.0.0.0", 0, TOKEN, FakeControlSurface(), bus, **kw)
    with pytest.raises(ValueError, match="token"):
        PanelServer("127.0.0.1", 0, "", FakeControlSurface(), bus, **kw)


async def test_binds_loopback_only(panel: Panel) -> None:
    runner = panel.server._runner
    assert runner is not None
    hosts = {addr[0] for addr in runner.addresses}
    assert hosts == {"127.0.0.1"}
    assert panel.server.port > 0
    assert panel.server.health().state is HealthState.OK


async def test_port_in_use_raises_and_reports_down(panel: Panel) -> None:
    other = PanelServer(
        "127.0.0.1",
        panel.server.port,
        TOKEN,
        FakeControlSurface(),
        panel.bus,
        emergency_url="http://127.0.0.1:8779",
        emergency_token=EMERGENCY,
        ingest=lambda m: None,
    )
    with pytest.raises(OSError):
        await other.start()
    assert other.health().state is HealthState.DOWN


# --- security -------------------------------------------------------------------------------


async def test_healthz_needs_no_token(panel: Panel) -> None:
    status, body = await panel.get("/healthz", headers={})
    assert status == 200 and body["ok"] is True


async def test_api_requires_the_token(panel: Panel) -> None:
    assert (await panel.get("/api/state", headers={}))[0] == 401
    assert (await panel.get("/api/state", headers={"X-Aivtube-Token": "wrong"}))[0] == 401
    assert (await panel.post("/api/cmd", {"kind": "freeze"}, headers={}))[0] == 401
    assert (await panel.get("/hotkey/freeze", headers={}))[0] == 401
    assert (await panel.post("/api/event", {"user": "a"}, headers={}))[0] == 401
    assert panel.control.commands == [] and panel.ingested == []
    assert (await panel.get("/api/state"))[0] == 200
    assert (await panel.get("/api/state", headers={"Authorization": f"Bearer {TOKEN}"}))[0] == 200
    assert (await panel.get(f"/api/state?token={TOKEN}", headers={}))[0] == 200


async def test_bad_origin_and_host_are_rejected(panel: Panel) -> None:
    evil = panel.auth(Origin="http://evil.example")
    assert (await panel.post("/api/cmd", {"kind": "freeze"}, headers=evil))[0] == 403
    null = panel.auth(Origin="null")
    assert (await panel.post("/api/cmd", {"kind": "freeze"}, headers=null))[0] == 403
    other_port = panel.auth(Origin="http://127.0.0.1:1")
    assert (await panel.get("/api/state", headers=other_port))[0] == 403
    rebinding = panel.auth(Host=f"attacker.example:{panel.server.port}")
    assert (await panel.get("/api/state", headers=rebinding))[0] == 403
    assert panel.control.commands == []
    same = panel.auth(Origin=f"http://127.0.0.1:{panel.server.port}")
    assert (await panel.post("/api/cmd", {"kind": "freeze"}, headers=same))[0] == 200
    local = panel.auth(Origin=f"http://localhost:{panel.server.port}")
    assert (await panel.get("/api/ping", headers=local))[0] == 200


async def test_security_headers(panel: Panel) -> None:
    async with panel.session.get(panel.url + "/api/ping", headers=panel.auth()) as resp:
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Cache-Control"] == "no-store"
        assert resp.headers["Referrer-Policy"] == "no-referrer"


# --- pages ----------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/overlay/captions", "/overlay/status"])
async def test_pages_are_served_with_a_csp(panel: Panel, path: str) -> None:
    async with panel.session.get(panel.url + path) as resp:
        assert resp.status == 200
        assert resp.content_type == "text/html"
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'none'" in csp and "http://127.0.0.1:8779" in csp
        text = await resp.text()
        assert "<html" in text.lower() and "http://" not in text.split("<script", 1)[0]


async def test_config_exposes_the_emergency_endpoint(panel: Panel) -> None:
    status, body = await panel.get("/api/config")
    assert status == 200
    assert body["emergency_url"] == "http://127.0.0.1:8779"
    assert body["emergency_token"] == EMERGENCY
    assert body["characters"] == ["pailin"] and body["default_character"] == "pailin"
    assert "freeze" in body["hotkeys"] and "SegmentStarted" in body["event_types"]


# --- commands -------------------------------------------------------------------------------


async def test_cmd_routes_to_the_control_surface(panel: Panel) -> None:
    status, body = await panel.post(
        "/api/cmd", {"kind": "say", "args": {"text": "สวัสดีค่ะ"}, "character": "pailin"}
    )
    assert status == 200 and body["ok"] is True and body["kind"] == "say"
    cmd = panel.control.commands[-1]
    assert (cmd.kind, cmd.args, cmd.character, cmd.operator) == (
        OpKind.SAY,
        {"text": "สวัสดีค่ะ"},
        "pailin",
        "panel",
    )
    assert (await panel.post("/api/cmd", {"kind": "freeze"}))[0] == 200
    assert panel.control.snapshot()["paused"] is True
    assert (await panel.post("/api/cmd", {"kind": "nope"}))[0] == 400
    assert (await panel.post("/api/cmd", {"kind": "say", "args": {}}))[0] == 400
    async with panel.session.post(
        panel.url + "/api/cmd", data="{not json", headers=panel.auth()
    ) as resp:
        assert resp.status == 400


async def test_failed_commands_answer_409(bus: EventBus) -> None:
    control = FakeControlSurface({OpKind.GO_LIVE: OpResult(False, "LLM not ready")})
    server, _ = await _start(bus, control)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                server.url + "/api/cmd",
                json={"kind": "go_live"},
                headers={"X-Aivtube-Token": TOKEN},
            ) as resp,
        ):
            assert resp.status == 409
            assert (await resp.json())["detail"] == "LLM not ready"
    finally:
        await server.aclose()


async def test_mic_endpoint(panel: Panel) -> None:
    assert (await panel.post("/api/mic", {"mode": "ptt"}))[0] == 200
    assert (await panel.post("/api/mic", {"ptt": True}))[0] == 200
    kinds = [(c.kind, dict(c.args)) for c in panel.control.commands]
    assert kinds == [(OpKind.MIC_MODE, {"mode": "ptt"}), (OpKind.PTT, {"active": True})]
    assert (await panel.post("/api/mic", {"mode": "loud"}))[0] == 400
    assert (await panel.post("/api/mic", {"mode": "open", "ptt": False}))[0] == 400
    assert (await panel.post("/api/mic", {}))[0] == 400


async def test_hotkeys_with_query_token(panel: Panel) -> None:
    status, body = await panel.get(f"/hotkey/freeze?token={TOKEN}", headers={})
    assert status == 200 and body["kind"] == "freeze"
    assert panel.control.commands[-1].operator == "hotkey"
    await panel.get(f"/hotkey/mute_toggle?token={TOKEN}", headers={})
    await panel.get(f"/hotkey/mute_toggle?token={TOKEN}", headers={})
    assert [c.kind for c in panel.control.commands[-2:]] == [OpKind.MUTE, OpKind.UNMUTE]
    assert (await panel.post(f"/hotkey/ptt_down?token={TOKEN}", headers={}))[0] == 200
    status, body = await panel.get(f"/hotkey/explode?token={TOKEN}", headers={})
    assert status == 404 and "freeze" in body["hotkeys"]


# --- alert ingest ---------------------------------------------------------------------------


async def test_event_ingest_creates_a_support_message(panel: Panel) -> None:
    status, body = await panel.post(
        "/api/event",
        {"user": "ต้นกล้า", "amount": 100, "currency": "THB", "text": "สู้ๆนะ", "id": "tip-1"},
    )
    assert status == 200 and body == {"ok": True, "id": "alert:tip-1", "kind": "donation"}
    (msg,) = panel.ingested
    assert msg.kind is MsgKind.DONATION and msg.user.name == "ต้นกล้า" and msg.amount == 100
    again = await panel.post("/api/event", {"user": "ต้นกล้า", "amount": 100, "id": "tip-1"})
    assert again[1]["duplicate"] is True and len(panel.ingested) == 1
    assert (await panel.post("/api/event", {"user": "x", "kind": "bribe"}))[0] == 400
    sub = await panel.post("/api/event", {"user": "bob", "kind": "sub", "months": 3})
    assert sub[0] == 200 and panel.ingested[-1].kind is MsgKind.SUB


async def test_event_ingest_failure_is_503_and_can_retry(bus: EventBus) -> None:
    calls: list[ChatMessage] = []

    def flaky(msg: ChatMessage) -> None:
        calls.append(msg)
        if len(calls) == 1:
            raise RuntimeError("intake down")

    server, _ = await _start(bus, FakeControlSurface(), ingest=flaky)
    try:
        async with aiohttp.ClientSession(headers={"X-Aivtube-Token": TOKEN}) as session:
            body = {"user": "amy", "amount": 5, "currency": "USD", "id": "x1"}
            async with session.post(server.url + "/api/event", json=body) as resp:
                assert resp.status == 503
            async with session.post(server.url + "/api/event", json=body) as resp:
                assert resp.status == 200 and "duplicate" not in await resp.json()
        assert len(calls) == 2
    finally:
        await server.aclose()


# --- websocket ------------------------------------------------------------------------------


async def _next_events(
    ws: aiohttp.ClientWebSocketResponse, timeout: float = 5.0
) -> list[dict[str, Any]]:
    while True:
        msg = await asyncio.wait_for(ws.receive(), timeout)
        assert msg.type is aiohttp.WSMsgType.TEXT, msg
        frame = json.loads(msg.data)
        if frame["kind"] == "events":
            return list(frame["events"])


async def test_ws_requires_the_token(panel: Panel) -> None:
    with pytest.raises(aiohttp.WSServerHandshakeError) as info:
        await panel.session.ws_connect(panel.url + "/ws")
    assert info.value.status == 401
    with pytest.raises(aiohttp.WSServerHandshakeError) as info:
        await panel.session.ws_connect(
            panel.url + f"/ws?token={TOKEN}", headers={"Origin": "http://evil.example"}
        )
    assert info.value.status == 403


async def test_ws_receives_bus_events(panel: Panel) -> None:
    async with panel.session.ws_connect(panel.url + f"/ws?token={TOKEN}") as ws:
        hello = json.loads((await asyncio.wait_for(ws.receive(), 5)).data)
        assert hello["kind"] == "hello" and hello["ws_max_hz"] == 20.0
        msg = make_chat_message("ไพลินจ๋า", user="tom")
        panel.bus.publish(ChatReceived(message=msg, character="pailin"))
        panel.bus.publish(Alert(level="warn", message="TTS ช้า"))
        events: list[dict[str, Any]] = []
        while len(events) < 2:
            events += await _next_events(ws)
        assert [e["type"] for e in events] == ["ChatReceived", "Alert"]
        assert events[0]["message"]["text"] == "ไพลินจ๋า"
        assert "raw" not in events[0]["message"]


def _segment(utt_id: str, caption: str, character: str) -> SegmentStarted:
    return SegmentStarted(
        utt_id=utt_id,
        seq=0,
        t_audible=1.0,
        duration_s=1.2,
        backend="edge",
        silent=False,
        caption=caption,
        emotion=None,
        character=character,
    )


async def test_ws_type_and_character_filters(panel: Panel) -> None:
    url = panel.url + f"/ws?token={TOKEN}&types=SegmentStarted,Filtered&character=pailin"
    async with panel.session.ws_connect(url) as ws:
        await asyncio.wait_for(ws.receive(), 5)  # hello
        panel.bus.publish(Alert(level="info", message="ignored"))
        panel.bus.publish(_segment("u1", "สวัสดีค่ะ", "twin"))
        panel.bus.publish(_segment("u2", "ไพลินเองค่ะ", "pailin"))
        panel.bus.publish(
            Filtered(direction="out", tier="tier0", category="slur", rule="r1", ref="m1")
        )
        events: list[dict[str, Any]] = []
        while len(events) < 2:
            events += await _next_events(ws)
        assert [(e["type"], e.get("caption")) for e in events] == [
            ("SegmentStarted", "ไพลินเองค่ะ"),
            ("Filtered", None),
        ]
    with pytest.raises(aiohttp.WSServerHandshakeError) as info:
        await panel.session.ws_connect(panel.url + f"/ws?token={TOKEN}&types=Bogus")
    assert info.value.status == 400


async def test_ws_frames_are_rate_limited_and_batched(panel: Panel) -> None:
    async with panel.session.ws_connect(panel.url + f"/ws?token={TOKEN}") as ws:
        await asyncio.wait_for(ws.receive(), 5)
        for i in range(50):
            panel.bus.publish(Alert(level="info", message=f"a{i}"))
        frames = 0
        seen: list[str] = []
        while len(seen) < 50:
            seen += [e["message"] for e in await _next_events(ws)]
            frames += 1
        assert seen == [f"a{i}" for i in range(50)]
        assert frames <= 3  # batched, not one frame per event


def test_ws_client_queue_drops_oldest() -> None:
    client = _Client(ws=None, types=None, character=None, maxlen=3)  # type: ignore[arg-type]
    for i in range(5):
        client.offer("Alert", None, f'"{i}"')
    assert list(client.queue) == ['"2"', '"3"', '"4"']
    assert client.dropped == 2 and client.dropped_total == 2
    client.offer("Alert", "twin", '"5"')
    assert client.dropped == 3


async def test_shutdown_closes_connected_clients(bus: EventBus) -> None:
    server, _ = await _start(bus, FakeControlSurface())
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(server.url + f"/ws?token={TOKEN}")
        await asyncio.wait_for(ws.receive(), 5)
        await asyncio.wait_for(server.aclose(), 5)
        msg = await asyncio.wait_for(ws.receive(), 5)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
        await ws.close()
    assert server.health().state is HealthState.DOWN


async def test_run_serves_until_cancelled(bus: EventBus) -> None:
    server = PanelServer(
        "127.0.0.1",
        0,
        TOKEN,
        FakeControlSurface(),
        bus,
        emergency_url="http://127.0.0.1:8779",
        emergency_token=EMERGENCY,
        ingest=lambda m: None,
    )
    task = asyncio.ensure_future(server.run())
    for _ in range(200):
        if server.health().state is HealthState.OK:
            break
        await asyncio.sleep(0.01)
    async with aiohttp.ClientSession() as session, session.get(server.url + "/healthz") as resp:
        assert resp.status == 200
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.health().state is HealthState.DOWN


# --- snapshot, traces, audit, chat ------------------------------------------------------------


async def test_state_mirrors_bus_feeds(panel: Panel) -> None:
    panel.bus.publish(HealthChanged(health=Health("voice", HealthState.DEGRADED, "tts slow")))
    panel.bus.publish(StateChanged(old="idle", new="speaking", character="pailin"))
    panel.bus.publish(ChatReceived(message=make_chat_message("hi", user="tom")))
    panel.bus.publish(Filtered(direction="in", tier="tier0", category="pii", rule=None, ref="m9"))
    await asyncio.sleep(0.05)
    status, body = await panel.get("/api/state")
    assert status == 200
    assert body["control"]["paused"] is False
    health = {h["component"]: h for h in body["health"]}
    assert health["voice"]["state"] == "degraded" and health["panel"]["state"] == "ok"
    assert body["states"] == {"pailin": "speaking"}
    assert body["recent"]["chat"][0]["message"]["text"] == "hi"
    assert body["recent"]["moderation"][0]["ref"] == "m9"
    assert (await panel.get("/api/snapshot"))[1]["states"] == {"pailin": "speaking"}


async def test_state_merges_the_health_source_with_bus_updates(
    bus: EventBus, real_clock: Any
) -> None:
    t0 = real_clock.now()
    current = [
        Health("llm", HealthState.OK, "local-30b", t0),
        Health("voice", HealthState.OK, "", t0),
    ]
    server, _ = await _start(bus, FakeControlSurface(), health_source=lambda: current)
    session = aiohttp.ClientSession()
    panel = Panel(server, None, bus, session)
    try:
        bus.publish(HealthChanged(health=Health("voice", HealthState.DOWN, "crash", t0 + 1.0)))
        await asyncio.sleep(0.05)
        status, body = await panel.get("/api/state")
        assert status == 200
        states = {h["component"]: (h["state"], h["detail"]) for h in body["health"]}
        assert states["llm"] == ("ok", "local-30b")
        assert states["voice"] == ("down", "crash")  # the newer bus event wins
        assert [h["component"] for h in body["health"]] == ["llm", "panel", "voice"]
        current[1] = Health("voice", HealthState.OK, "restarted", t0 + 2.0)
        status, body = await panel.get("/api/state")
        states = {h["component"]: (h["state"], h["detail"]) for h in body["health"]}
        assert states["voice"] == ("ok", "restarted")
    finally:
        await session.close()
        await server.aclose()


def _trace(i: int, ttfa: float) -> dict[str, Any]:
    return {
        "turn_id": f"t{i}",
        "kind": "voice",
        "character": "pailin",
        "stages_ms": {"vad_end": 0.0, "first_audible": ttfa},
        "ttfa_ms": ttfa,
    }


async def test_traces_from_the_bus(panel: Panel) -> None:
    for i in range(3):
        panel.bus.publish(TurnTraceReady(trace=_trace(i, 1200.0 + i)))
    await asyncio.sleep(0.05)
    status, body = await panel.get("/api/traces?n=2")
    assert status == 200 and body["source"] == "bus"
    assert [r["turn_id"] for r in body["rows"]] == ["t1", "t2"]
    assert body["badges"]["voice"]["status"] == "ok"
    assert (await panel.get("/api/traces?n=abc"))[0] == 400


async def test_traces_and_audit_from_ops(bus: EventBus) -> None:
    ops = FakeOpsDb()
    ops.traces = [_trace(i, 3500.0) for i in range(4)]
    ops.rows["moderation_log"] = [{"id": 1, "category": "slur", "author": "twitch:tom"}]
    server, _ = await _start(bus, FakeControlSurface(), ops=ops)
    try:
        async with aiohttp.ClientSession(headers={"X-Aivtube-Token": TOKEN}) as session:
            async with session.get(server.url + "/api/traces") as resp:
                body = await resp.json()
                assert body["source"] == "ops" and body["badges"]["voice"]["status"] == "over"
            async with session.get(server.url + "/api/audit?table=moderation_log") as resp:
                assert (await resp.json())["rows"][0]["author"] == "twitch:tom"
            async with session.get(server.url + "/api/audit?table=users") as resp:
                assert resp.status == 400
    finally:
        await server.aclose()


async def test_audit_without_ops_is_unavailable(panel: Panel) -> None:
    assert (await panel.get("/api/audit"))[0] == 503


async def test_chat_window_with_scores(bus: EventBus, fake_clock: Any) -> None:
    window = ScoredChatWindow(fake_clock, name_matcher=["ไพลิน"])
    window.add(make_chat_message("ไพลินกินข้าวยัง?", user="tom", clock=fake_clock))
    window.add(make_chat_message("555", user="amy", clock=fake_clock))
    server, _ = await _start(bus, FakeControlSurface(), windows={"pailin": window})
    try:
        async with aiohttp.ClientSession(headers={"X-Aivtube-Token": TOKEN}) as session:
            async with session.get(server.url + "/api/chat") as resp:
                body = await resp.json()
            async with session.get(server.url + "/api/chat?character=twin") as resp:
                assert resp.status == 404
        assert body["character"] == "pailin" and body["has_mention"] is True
        texts = [row["message"]["text"] for row in body["window"]]
        assert texts[0] == "ไพลินกินข้าวยัง?" and len(texts) == 2
        assert body["window"][0]["score"] > body["window"][1]["score"]
    finally:
        await server.aclose()


# --- memory and moderation (through the real CoreControl) -------------------------------------


@pytest.fixture
async def core_panel(bus: EventBus, real_clock: Any) -> AsyncIterator[Panel]:
    memory = EditableMemory("pailin", real_clock)
    speech = FakeSpeechOutput(bus, real_clock)
    tasks = FakeTaskSupervisor(real_clock)
    ops = FakeOpsDb()
    control = CoreControl(
        {"pailin": RecordingBrain()},
        router=FakeLLMRouter([FakeLLM([], name="local-30b")]),
        registry=FakeToolRegistry(),
        speech=speech,
        memory_by_char={"pailin": memory},
        safety=FakeSafetyGate(),
        ops=ops,
        bus=bus,
        clock=real_clock,
        restart=lambda name: asyncio.sleep(0),
        tasks=tasks,
    )
    server, ingested = await _start(
        bus, control, memory={"pailin": memory}, ops=ops, moderation=FakeModeration()
    )
    session = aiohttp.ClientSession()
    panel = Panel(server, control, bus, session, ingested)
    panel.memory = memory  # type: ignore[attr-defined]
    panel.speech = speech  # type: ignore[attr-defined]
    panel.ops = ops  # type: ignore[attr-defined]
    try:
        yield panel
    finally:
        await session.close()
        await server.aclose()
        await speech.aclose()
        await tasks.aclose()


async def test_memory_approve_edit_delete_round_trip(core_panel: Panel) -> None:
    memory = core_panel.memory  # type: ignore[attr-defined]
    item = await memory.remember(
        MemoryItem(id=None, kind="fact", text="ต้นกล้าชอบแมว", status="quarantined", source="model")
    )
    status, body = await core_panel.get("/api/memory?status=quarantined")
    assert status == 200 and [i["id"] for i in body["items"]] == [item.id]
    status, body = await core_panel.send("PATCH", f"/api/memory/{item.id}", {"status": "active"})
    assert status == 200 and body["item"]["status"] == "active"
    status, body = await core_panel.send(
        "PATCH", f"/api/memory/{item.id}", {"text": "ต้นกล้าชอบหมา", "locked": True}
    )
    assert status == 200 and body["item"]["text"] == "ต้นกล้าชอบหมา" and body["item"]["locked"]
    status, body = await core_panel.send("DELETE", f"/api/memory/{item.id}")
    assert status == 409 and "locked" in body["results"][0]["detail"]
    await core_panel.send("PATCH", f"/api/memory/{item.id}", {"locked": False})
    status, body = await core_panel.send("DELETE", f"/api/memory/{item.id}")
    assert status == 200 and body["item"]["status"] == "deleted"
    assert (await core_panel.get("/api/memory?status=gone"))[0] == 400
    assert (await core_panel.send("PATCH", "/api/memory/abc", {"status": "active"}))[0] == 400
    assert (await core_panel.send("PATCH", f"/api/memory/{item.id}", {}))[0] == 400
    assert (await core_panel.get("/api/memory?character=twin"))[0] == 404
    await asyncio.sleep(0.05)
    commands = [r["command"] for r in core_panel.ops.rows["op_audit"]]  # type: ignore[attr-defined]
    assert commands == [
        "memory_status",
        "memory_edit",
        "memory_status",
        "memory_edit",
        "memory_status",
    ]


async def test_freeze_via_panel_reaches_speech_stop_quickly(core_panel: Panel) -> None:
    speech = core_panel.speech  # type: ignore[attr-defined]
    status, body = await core_panel.post("/api/cmd", {"kind": "freeze"})
    assert status == 200 and body["ok"] is True
    assert body["latency_ms"] < 150.0
    assert ("stop", (None, "now", "operator_freeze", 30)) in speech.calls


async def test_moderation_actions(core_panel: Panel) -> None:
    status, body = await core_panel.post(
        "/api/moderation",
        {"action": "mute_user", "platform": "twitch", "user_id": "u-1", "name": "troll"},
    )
    brain = core_panel.control._brains["pailin"]
    assert status == 200 and brain.commands[-1].kind is OpKind.MUTE_USER
    assert brain.commands[-1].args["minutes"] == 10.0
    status, body = await core_panel.post(
        "/api/moderation", {"action": "blocklist", "term": "สล็อตเว็บตรง", "category": "gambling"}
    )
    assert status == 200 and body["ok"] is True
    status, _ = await core_panel.post(
        "/api/moderation", {"action": "false_positive", "ref": "mod-7", "note": "หีบ = box"}
    )
    assert status == 200
    assert (await core_panel.post("/api/moderation", {"action": "ban"}))[0] == 400
    assert (await core_panel.post("/api/moderation", {"action": "blocklist"}))[0] == 400
    await asyncio.sleep(0.05)
    rows = core_panel.ops.rows["op_audit"]  # type: ignore[attr-defined]
    assert [r["command"] for r in rows] == [
        "mute_user",
        "moderation.blocklist",
        "moderation.false_positive",
    ]
    assert rows[1]["args"] == {"term": "สล็อตเว็บตรง", "category": "gambling"}


async def test_moderation_without_callbacks_is_unavailable(panel: Panel) -> None:
    status, _ = await panel.post("/api/moderation", {"action": "blocklist", "term": "x"})
    assert status == 503
    status, _ = await panel.post(
        "/api/moderation", {"action": "mute_user", "platform": "twitch", "user_id": "u"}
    )
    assert status == 200  # mute-user goes through the control surface


# --- launcher, moderation feed, config ---------------------------------------------------------


async def test_launcher_notify_freeze_is_audited_as_the_launcher(panel: Panel) -> None:
    """The launcher's own ``notify_freeze`` (stdlib urllib) against the real server."""
    from aivtube.launcher.emergency import notify_freeze

    ok = await asyncio.to_thread(notify_freeze, panel.url, TOKEN, reason="hardkill")
    assert ok
    cmd = panel.control.commands[-1]
    assert cmd.kind is OpKind.FREEZE and cmd.operator == "launcher"
    assert cmd.args == {"reason": "hardkill"}
    status, body = await panel.post("/api/cmd", {"kind": "freeze", "operator": "Evil Name"})
    assert status == 400 and "operator" in body["error"]


async def test_moderation_feed_from_the_live_gate_audit(bus: EventBus, real_clock: Any) -> None:
    from aivtube.safety import LayeredSafetyGate, ModerationAudit
    from aivtube.testing.fakes import FakeTextFilter

    audit = ModerationAudit(None, clock=real_clock)
    gate = LayeredSafetyGate(FakeTextFilter(["คำต้องห้าม"]), bus=bus, clock=real_clock, audit=audit)
    server, _ = await _start(
        bus, FakeControlSurface(), moderation_log=lambda: audit.recent, characters=["pailin"]
    )
    session = aiohttp.ClientSession()
    panel = Panel(server, None, bus, session)
    filtered_sub = bus.subscribe(Filtered, name="test-filtered")
    try:
        status, cfg = await panel.get("/api/config")
        assert status == 200 and cfg["moderation_log"] is True
        msg = make_chat_message("คำต้องห้าม นะ", user="troll", user_id="u-9")
        verdict, _name = gate.check_input(msg, character="pailin")
        assert verdict.verdict.value == "drop"
        await gate.check_output("พูดคำต้องห้าม", character="pailin", prev_tail="")
        status, body = await panel.get("/api/moderation")
        assert status == 200 and body["source"] == "live"
        chat, speech = body["items"]
        assert (chat["direction"], chat["verdict"], chat["platform"], chat["user_id"]) == (
            "in",
            "drop",
            "twitch",
            "u-9",
        )
        assert chat["text"] == "คำต้องห้าม นะ" and chat["category"] == "fake"
        assert len(chat["ref"]) == 16 and chat["ts"] > 0
        assert (speech["direction"], speech["platform"], speech["user_id"]) == ("out", None, None)
        # the ref joins the bus event (Filtered.ref) to its feed item
        refs: set[str | None] = set()
        async with asyncio.timeout(5):
            async for event in filtered_sub:
                assert isinstance(event, Filtered)
                refs.add(event.ref)
                if len(refs) == 2:
                    break
        assert refs == {chat["ref"], speech["ref"]}
        status, body = await panel.get(f"/api/moderation?since_ts={speech['ts']}")
        assert status == 200 and body["items"] == []
        assert (await panel.get("/api/moderation?since_ts=nan"))[0] == 400
    finally:
        filtered_sub.close()
        await session.close()
        await server.aclose()


async def test_moderation_feed_from_ops_and_unavailable(bus: EventBus) -> None:
    ops = FakeOpsDb()
    ops.rows["moderation_log"] = [
        {
            "id": i,
            "ts": 1000.0 + i,
            "character": "pailin",
            "direction": "in",
            "source": "youtube",
            "tier": "tier0",
            "category": "spam",
            "rule": "r",
            "verdict": "review",
            "text_masked": f"ข้อความ {i}",
            "text_sha256": f"{i:064x}",
            "author": f"UC{i}",
            "turn_id": None,
        }
        for i in (1, 2, 3)
    ]
    server, _ = await _start(bus, FakeControlSurface(), ops=ops)
    session = aiohttp.ClientSession()
    panel = Panel(server, None, bus, session)
    try:
        status, body = await panel.get("/api/moderation?since=1")
        assert status == 200 and body["source"] == "ops"
        assert [(i["id"], i["verdict"], i["user_id"]) for i in body["items"]] == [
            (2, "review", "UC2"),
            (3, "review", "UC3"),
        ]
    finally:
        await session.close()
        await server.aclose()
    server, _ = await _start(bus, FakeControlSurface())
    session = aiohttp.ClientSession()
    try:
        async with session.get(
            server.url + "/api/moderation", headers={"X-Aivtube-Token": TOKEN}
        ) as resp:
            assert resp.status == 503
    finally:
        await session.close()
        await server.aclose()


async def test_config_lists_the_editable_memory_fields(core_panel: Panel) -> None:
    status, cfg = await core_panel.get("/api/config")
    assert status == 200
    assert cfg["memory_edit_fields"] == ["text", "subject", "importance", "locked"]
