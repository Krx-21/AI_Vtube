"""``FakeVTSServer``: the VTube Studio Public API over a real local WebSocket (§3.7, §10).

Ported from the avatar brief's ``test_avatar.py`` and extended. Like VTS (a Unity main thread),
it answers once per render frame (``frame_hz``; 0 answers at once). It implements the token
flow (errors 50/52/53, 8 before auth), ``InjectParameterDataRequest`` (errors 450-455, custom
parameters), hotkeys (202), expressions (650/651), model moves, statistics and event
subscriptions. A ``TestEvent`` subscription reproduces the adversarial case where an event that
carries the subscription's requestID arrives *before* the subscription response.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

__all__ = ["DEFAULT_EXPRESSIONS", "DEFAULT_HOTKEYS", "DEFAULT_PARAMS", "FakeVTSServer"]

API = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0"}

DEFAULT_PARAMS: frozenset[str] = frozenset(
    {
        "MouthOpen", "MouthSmile", "MouthX", "Brows", "BrowLeftY", "BrowRightY",
        "FaceAngleX", "FaceAngleY", "FaceAngleZ", "FacePositionX", "FacePositionY",
        "FacePositionZ", "EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY", "EyeOpenLeft",
        "EyeOpenRight", "CheekPuff", "TongueOut", "VoiceVolume", "VoiceFrequency",
        "VoiceVolumePlusMouthOpen", "VoiceFrequencyPlusMouthSmile",
    }
)  # fmt: skip

DEFAULT_HOTKEYS: tuple[Mapping[str, str], ...] = (
    {
        "name": "Happy",
        "hotkeyID": "hk-happy",
        "type": "ToggleExpression",
        "file": "exp_03.exp3.json",
    },
    {
        "name": "Wave",
        "hotkeyID": "hk-wave",
        "type": "TriggerAnimation",
        "file": "wave.motion3.json",
    },
    {"name": "Spin", "hotkeyID": "hk-spin", "type": "MoveModel", "file": ""},
)

DEFAULT_EXPRESSIONS: tuple[str, ...] = tuple(f"exp_0{i}.exp3.json" for i in range(1, 6))

_EVENTS = frozenset(
    {
        "TestEvent", "ModelLoadedEvent", "TrackingStatusChangedEvent", "BackgroundChangedEvent",
        "ModelConfigChangedEvent", "ModelMovedEvent", "ModelOutlineEvent", "HotkeyTriggeredEvent",
        "ModelAnimationEvent", "ItemEvent", "ModelClickedEvent", "PostProcessingEvent",
        "Live2DCubismEditorConnectedEvent",
    }
)  # fmt: skip


@dataclass(eq=False)
class _Session:
    ws: ServerConnection
    outbox: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    authed: bool = False
    plugin: tuple[str, str] | None = None
    subscriptions: dict[str, str] = field(default_factory=dict)  # event name -> requestID


class FakeVTSServer:
    """VTube Studio stand-in. ``start()`` returns ``ws://127.0.0.1:<port>``.

    Instrumentation: ``received`` (every request), ``injected`` (successful injection
    payloads), ``param_values``, ``token_requests``, ``auth_requests``, ``triggered`` (hotkey
    IDs), ``expressions`` (file -> active), ``moves``, ``created_params``.
    Controls: ``deny_token``, ``held_params`` (error 454 for ``set`` mode), ``revoke_token()``,
    ``emit(event, data)``, ``drop()`` (close every connection, keep listening).
    """

    def __init__(
        self,
        port: int = 0,
        *,
        frame_hz: float = 60.0,
        host: str = "127.0.0.1",
        token: str = "tok-123",
        hotkeys: Sequence[Mapping[str, str]] = DEFAULT_HOTKEYS,
        expressions: Iterable[str] = DEFAULT_EXPRESSIONS,
        params: Iterable[str] = DEFAULT_PARAMS,
        held_params: Iterable[str] = (),
        deny_token: bool = False,
        model_loaded: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_hz = frame_hz
        self.token = token
        self.hotkeys = [dict(h) for h in hotkeys]
        self.expressions: dict[str, bool] = {e: False for e in expressions}
        self.params: set[str] = set(params)
        self.created_params: dict[str, Mapping[str, Any]] = {}
        self.held_params = set(held_params)
        self.deny_token = deny_token
        self.model_loaded = model_loaded
        self.received: list[dict[str, Any]] = []
        self.injected: list[dict[str, Any]] = []
        self.param_values: dict[str, float] = {}
        self.token_requests = 0
        self.auth_requests = 0
        self.triggered: list[str] = []
        self.moves: list[dict[str, Any]] = []
        self.connections_total = 0
        self.model_position = {"positionX": 0.0, "positionY": 0.0, "rotation": 0.0, "size": 0.0}
        self._sessions: set[_Session] = set()
        self._server: Server | None = None
        self._started = time.time()

    # --- lifecycle --------------------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def start(self) -> str:
        if self._server is None:
            self._server = await serve(
                self._handler,
                self.host,
                self.port,
                compression=None,
                ping_interval=None,
                max_size=16 * 2**20,
            )
            sock = next(iter(self._server.sockets))
            self.port = int(sock.getsockname()[1])
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            server.close()
            await server.wait_closed()

    async def drop(self) -> None:
        """Close every client connection (VTS restarting); keep accepting new ones."""
        for s in list(self._sessions):
            await s.ws.close(1001, "VTube Studio closed")

    async def __aenter__(self) -> FakeVTSServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @property
    def connected_clients(self) -> int:
        return len(self._sessions)

    def revoke_token(self, new_token: str | None = None) -> None:
        """The streamer revoked the plugin: stored tokens stop working."""
        self.token = new_token or f"tok-{uuid.uuid4().hex[:8]}"
        for s in self._sessions:
            s.authed = False

    def emit(self, event: str, data: Mapping[str, Any]) -> int:
        """Send ``event`` to every session subscribed to it; returns the number of recipients."""
        n = 0
        for s in self._sessions:
            rid = s.subscriptions.get(event)
            if rid is not None:
                s.outbox.put_nowait(self._msg(rid, event, data))
                n += 1
        return n

    # --- protocol ---------------------------------------------------------------------------
    @staticmethod
    def _msg(request_id: str, message_type: str, data: Mapping[str, Any]) -> str:
        return json.dumps(
            {
                **API,
                "timestamp": int(time.time() * 1000),
                "requestID": request_id,
                "messageType": message_type,
                "data": dict(data),
            },
            ensure_ascii=False,
        )

    def _err(self, request_id: str, error_id: int, message: str) -> str:
        return self._msg(request_id, "APIError", {"errorID": error_id, "message": message})

    async def _handler(self, ws: ServerConnection) -> None:
        session = _Session(ws)
        self._sessions.add(session)
        self.connections_total += 1
        pump = asyncio.get_running_loop().create_task(self._frame_pump(session))
        try:
            async for raw in ws:
                for out in self._handle(session, raw):
                    session.outbox.put_nowait(out)
        except ConnectionClosed:
            pass
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
            self._sessions.discard(session)

    async def _frame_pump(self, session: _Session) -> None:
        """Flush responses once per render frame (all queued messages at once)."""
        try:
            while True:
                if self.frame_hz > 0:
                    await asyncio.sleep(1.0 / self.frame_hz)
                    batch = []
                    while not session.outbox.empty():
                        batch.append(session.outbox.get_nowait())
                else:
                    batch = [await session.outbox.get()]
                for msg in batch:
                    await session.ws.send(msg)
        except ConnectionClosed:
            return

    def _handle(self, s: _Session, raw: str | bytes) -> list[str]:
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return [self._err("", 2, "JSONInvalid")]
        if not isinstance(req, dict):
            return [self._err("", 2, "JSONInvalid")]
        self.received.append(req)
        rid = str(req.get("requestID") or "")
        if req.get("apiName") != API["apiName"]:
            return [self._err(rid, 3, "APINameInvalid")]
        if len(rid) > 64 or not rid.isascii():
            return [self._err(rid[:64], 5, "RequestIDInvalid")]
        mtype = req.get("messageType")
        if not mtype:
            return [self._err(rid, 6, "RequestTypeMissingOrEmpty")]
        data = req.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if mtype == "APIStateRequest":
            return [self._resp(rid, mtype, self._api_state(s))]
        if mtype == "AuthenticationTokenRequest":
            return [self._token_request(rid, data)]
        if mtype == "AuthenticationRequest":
            return [self._auth_request(s, rid, data)]
        if not s.authed:
            return [self._err(rid, 8, "RequestRequiresAuthetication")]
        handler = getattr(self, f"_on_{mtype}", None)
        if handler is None:
            return [self._err(rid, 7, f"RequestTypeUnknown: {mtype}")]
        result: list[str] = handler(s, rid, data)
        return result

    @staticmethod
    def _resp(rid: str, request_type: str, data: Mapping[str, Any]) -> str:
        return FakeVTSServer._msg(rid, request_type.removesuffix("Request") + "Response", data)

    def _api_state(self, s: _Session) -> dict[str, Any]:
        return {
            "active": True,
            "vTubeStudioVersion": "1.35.10",
            "currentSessionAuthenticated": s.authed,
        }

    def _token_request(self, rid: str, data: Mapping[str, Any]) -> str:
        name = str(data.get("pluginName", ""))
        dev = str(data.get("pluginDeveloper", ""))
        if not 3 <= len(name) <= 32:
            return self._err(rid, 52, "TokenRequestPluginNameInvalid")
        if not 3 <= len(dev) <= 32:
            return self._err(rid, 53, "TokenRequestDeveloperNameInvalid")
        self.token_requests += 1
        if self.deny_token:
            return self._err(rid, 50, "TokenRequestDenied")
        return self._resp(rid, "AuthenticationTokenRequest", {"authenticationToken": self.token})

    def _auth_request(self, s: _Session, rid: str, data: Mapping[str, Any]) -> str:
        self.auth_requests += 1
        if not data.get("authenticationToken"):
            return self._err(rid, 100, "AuthenticationTokenMissing")
        if not data.get("pluginName"):
            return self._err(rid, 101, "AuthenticationPluginNameMissing")
        ok = data.get("authenticationToken") == self.token
        s.authed = ok
        s.plugin = (str(data.get("pluginName")), str(data.get("pluginDeveloper", "")))
        reason = (
            "Token valid. The plugin is authenticated for the duration of this session."
            if ok
            else "Token invalid. The plugin is not authenticated."
        )
        return self._resp(rid, "AuthenticationRequest", {"authenticated": ok, "reason": reason})

    # --- authenticated requests (dispatched by name) ------------------------------------------
    def _on_InjectParameterDataRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        mode = data.get("mode") or "set"
        if mode not in ("set", "add"):
            return [self._err(rid, 455, "InjectDataModeUnknown")]
        values = data.get("parameterValues") or []
        if not values:
            return [self._err(rid, 450, "InjectDataNoDataProvided")]
        unknown = [str(p.get("id")) for p in values if p.get("id") not in self.params]
        if unknown:
            return [self._err(rid, 453, f"InjectDataParamNameNotFound: {unknown}")]
        for p in values:
            v = p.get("value")
            if isinstance(v, bool) or not isinstance(v, int | float) or abs(v) > 1e6:
                return [self._err(rid, 451, f"InjectDataValueInvalid: {p.get('id')}")]
            w = p.get("weight", 1.0)
            if not isinstance(w, int | float) or not 0.0 <= w <= 1.0:
                return [self._err(rid, 452, f"InjectDataWeightInvalid: {p.get('id')}")]
        if mode == "set":
            held = [str(p["id"]) for p in values if p["id"] in self.held_params]
            if held:
                return [self._err(rid, 454, f"InjectDataParamControlledByOtherPlugin: {held}")]
        self.injected.append(dict(data))
        for p in values:
            prev = self.param_values.get(p["id"], 0.0) if mode == "add" else 0.0
            self.param_values[p["id"]] = prev + float(p["value"])
        return [self._resp(rid, "InjectParameterDataRequest", {})]

    def _on_HotkeyTriggerRequest(self, s: _Session, rid: str, data: Mapping[str, Any]) -> list[str]:
        if not self.model_loaded:
            return [self._err(rid, 201, "HotkeyExecutionFailedBecauseNoModelLoaded")]
        key = str(data.get("hotkeyID", ""))
        hk = next(
            (h for h in self.hotkeys if h["hotkeyID"] == key or h["name"].lower() == key.lower()),
            None,
        )
        if hk is None:
            return [self._err(rid, 202, "HotkeyIDNotFoundInModel")]
        self.triggered.append(hk["hotkeyID"])
        if hk["type"] == "ToggleExpression" and hk["file"] in self.expressions:
            self.expressions[hk["file"]] = not self.expressions[hk["file"]]
        # the response goes out before the HotkeyTriggeredEvent it causes
        s.outbox.put_nowait(self._resp(rid, "HotkeyTriggerRequest", {"hotkeyID": hk["hotkeyID"]}))
        self.emit(
            "HotkeyTriggeredEvent",
            {
                "hotkeyID": hk["hotkeyID"],
                "hotkeyName": hk["name"],
                "hotkeyAction": hk["type"],
                "hotkeyFile": hk["file"],
                "hotkeyTriggeredByAPI": True,
                "modelID": "fake-model",
                "modelName": "Pailin",
                "isLive2DItem": False,
            },
        )
        return []

    def _on_HotkeysInCurrentModelRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        hotkeys = [
            {
                "name": h["name"],
                "type": h["type"],
                "description": h["type"],
                "file": h["file"],
                "hotkeyID": h["hotkeyID"],
                "keyCombination": [],
                "onScreenButtonID": -1,
            }
            for h in self.hotkeys
        ]
        return [
            self._resp(
                rid,
                "HotkeysInCurrentModelRequest",
                {
                    "modelLoaded": self.model_loaded,
                    "modelName": "Pailin",
                    "modelID": "fake-model",
                    "availableHotkeys": hotkeys,
                },
            )
        ]

    def _on_ExpressionStateRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        only = data.get("expressionFile")
        exprs = [
            {
                "name": f.removesuffix(".exp3.json"),
                "file": f,
                "active": active,
                "deactivateWhenKeyIsLetGo": False,
                "autoDeactivateAfterSeconds": False,
                "secondsRemaining": 0,
                "usedInHotkeys": [],
                "parameters": [],
            }
            for f, active in self.expressions.items()
            if not only or f == only
        ]
        return [
            self._resp(
                rid,
                "ExpressionStateRequest",
                {
                    "modelLoaded": self.model_loaded,
                    "modelName": "Pailin",
                    "modelID": "fake-model",
                    "expressions": exprs,
                },
            )
        ]

    def _on_ExpressionActivationRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        if not self.model_loaded:
            return [self._err(rid, 652, "ExpressionActivationRequestNoModelLoaded")]
        f = str(data.get("expressionFile", ""))
        if not f.endswith(".exp3.json"):
            return [self._err(rid, 650, "ExpressionActivationRequestInvalidFilename")]
        if f not in self.expressions:
            return [self._err(rid, 651, "ExpressionActivationRequestFileNotFound")]
        self.expressions[f] = bool(data.get("active", True))
        return [self._resp(rid, "ExpressionActivationRequest", {})]

    def _on_CurrentModelRequest(self, s: _Session, rid: str, data: Mapping[str, Any]) -> list[str]:
        return [
            self._resp(
                rid,
                "CurrentModelRequest",
                {
                    "modelLoaded": self.model_loaded,
                    "modelName": "Pailin",
                    "modelID": "fake-model",
                    "vtsModelName": "Pailin.vtube.json",
                    "live2DModelName": "Pailin.model3.json",
                    "numberOfLive2DParameters": len(self.params),
                    "modelPosition": dict(self.model_position),
                },
            )
        ]

    def _on_MoveModelRequest(self, s: _Session, rid: str, data: Mapping[str, Any]) -> list[str]:
        if not self.model_loaded:
            return [self._err(rid, 300, "MoveModelRequestNoModelLoaded")]
        self.moves.append(dict(data))
        rel = bool(data.get("valuesAreRelativeToModel", False))
        for k in ("positionX", "positionY", "rotation", "size"):
            if k in data:
                v = float(data[k])
                self.model_position[k] = self.model_position[k] + v if rel else v
        return [self._resp(rid, "MoveModelRequest", {})]

    def _on_StatisticsRequest(self, s: _Session, rid: str, data: Mapping[str, Any]) -> list[str]:
        return [
            self._resp(
                rid,
                "StatisticsRequest",
                {
                    "uptime": int((time.time() - self._started) * 1000),
                    "framerate": int(self.frame_hz) if self.frame_hz else 60,
                    "vTubeStudioVersion": "1.35.10",
                    "allowedPlugins": 1,
                    "connectedPlugins": len(self._sessions),
                    "startedWithSteam": True,
                    "windowWidth": 1920,
                    "windowHeight": 1080,
                    "windowIsFullscreen": False,
                },
            )
        ]

    def _on_FaceFoundRequest(self, s: _Session, rid: str, data: Mapping[str, Any]) -> list[str]:
        return [self._resp(rid, "FaceFoundRequest", {"found": True})]

    def _on_ParameterCreationRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        name = str(data.get("parameterName", ""))
        if not (4 <= len(name) <= 32 and name.isascii() and name.isalnum()):
            return [self._err(rid, 350, "CustomParamNameInvalid")]
        self.params.add(name)
        self.created_params[name] = dict(data)
        return [self._resp(rid, "ParameterCreationRequest", {"parameterName": name})]

    def _on_InputParameterListRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        def entry(name: str, added_by: str) -> dict[str, Any]:
            return {
                "name": name,
                "addedBy": added_by,
                "value": self.param_values.get(name, 0.0),
                "min": -1e6,
                "max": 1e6,
                "defaultValue": 0.0,
            }

        return [
            self._resp(
                rid,
                "InputParameterListRequest",
                {
                    "modelLoaded": self.model_loaded,
                    "modelName": "Pailin",
                    "modelID": "fake-model",
                    "customParameters": [entry(n, "plugin") for n in sorted(self.created_params)],
                    "defaultParameters": [
                        entry(n, "VTube Studio")
                        for n in sorted(self.params - set(self.created_params))
                    ],
                },
            )
        ]

    def _on_EventSubscriptionRequest(
        self, s: _Session, rid: str, data: Mapping[str, Any]
    ) -> list[str]:
        name = str(data.get("eventName", ""))
        subscribe = bool(data.get("subscribe", True))
        if name and name not in _EVENTS:
            return [self._err(rid, 7, f"unknown event {name}")]
        out: list[str] = []
        if subscribe and name:
            s.subscriptions[name] = rid
            if name == "TestEvent":  # adversarial: an event with this requestID comes first
                msg = str((data.get("config") or {}).get("testMessageForEvent", "early"))
                out.append(self._msg(rid, "TestEvent", {"yourTestMessage": msg, "counter": 0}))
        elif name:
            s.subscriptions.pop(name, None)
        else:
            s.subscriptions.clear()
        out.append(
            self._resp(
                rid,
                "EventSubscriptionRequest",
                {
                    "subscribedEventCount": len(s.subscriptions),
                    "subscribedEvents": sorted(s.subscriptions),
                },
            )
        )
        if subscribe and name == "TestEvent":
            msg = str((data.get("config") or {}).get("testMessageForEvent", "hi"))
            out += [
                self._msg(rid, "TestEvent", {"yourTestMessage": msg, "counter": i})
                for i in (1, 2, 3)
            ]
        return out
