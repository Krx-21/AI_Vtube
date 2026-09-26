"""``VTSSink``: the ``AvatarSink`` for VTube Studio (§3.7, §2.8 "VTS closed").

``run()`` is the connect → authenticate → serve → reconnect loop (backoff 1 → 10 s, woken early
by a UDP discovery broadcast once VTS is back). Per connection it restores the last injected
values, subscribes to ``ModelLoadedEvent`` and reconciles expressions with
``ExpressionStateRequest``; while connected it re-sends held parameters well inside VTS's 1 s
"lost" window, heartbeats a silent link and re-authenticates when VTS reports error 8.

Parameters: ``set_params`` is fire-and-forget (the client keeps at most 8 frames in flight).
If VTS rejects a default input (454: another plugin controls it, e.g. "VTS Desktop Audio"
holding ``MouthOpen``; 453: not injectable), the sink probes each parameter, creates a custom
parameter such as ``PailinMouthOpen`` and injects that instead, and reports DEGRADED until the
streamer maps it in the model settings.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Any

from websockets.exceptions import WebSocketException

from aivtube.avatar.discovery import VTSDiscovery, url_port
from aivtube.avatar.emotion import EmotionController
from aivtube.avatar.vts_client import VTSAPIError, VTSClient, VTSDisconnected, VTSErrorID
from aivtube.contracts.infra import Clock
from aivtube.contracts.types import Health, HealthState
from aivtube.infra.clock import DeadlineExceeded
from aivtube.infra.tasks import backoff_delay

__all__ = ["PARAM_RANGES", "VTSSink"]

log = logging.getLogger("aivtube.avatar.sink")

#: (min, max, default) for custom parameters created in place of VTS default inputs.
PARAM_RANGES: Mapping[str, tuple[float, float, float]] = {
    "MouthOpen": (0.0, 1.0, 0.0),
    "MouthSmile": (0.0, 1.0, 0.5),
    "Brows": (0.0, 1.0, 0.5),
    "EyeOpenLeft": (0.0, 1.0, 1.0),
    "EyeOpenRight": (0.0, 1.0, 1.0),
    "FaceAngleX": (-30.0, 30.0, 0.0),
    "FaceAngleY": (-30.0, 30.0, 0.0),
    "FaceAngleZ": (-30.0, 30.0, 0.0),
    "FacePositionX": (-15.0, 15.0, 0.0),
    "FacePositionY": (-15.0, 15.0, 0.0),
    "FacePositionZ": (-10.0, 10.0, 0.0),
    "EyeLeftX": (-1.0, 1.0, 0.0),
    "EyeLeftY": (-1.0, 1.0, 0.0),
    "EyeRightX": (-1.0, 1.0, 0.0),
    "EyeRightY": (-1.0, 1.0, 0.0),
}
_DEFAULT_RANGE = (-1.0, 1.0, 0.0)
_SOFT_ERRORS = (VTSAPIError, VTSDisconnected, DeadlineExceeded)


class _AuthFailed(Exception):
    pass


class _Reconnect(Exception):
    pass


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(v)))


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class VTSSink:
    """VTube Studio renderer sink. Use it from the event-loop thread only."""

    def __init__(
        self,
        client: VTSClient,
        emotion_map: Mapping[str, Mapping[str, Any]],
        clock: Clock,
        *,
        name: str = "vts",
        component: str | None = None,
        fade_s: float = 0.3,
        reconnect_backoff: tuple[float, float] = (1.0, 10.0),
        face_found: bool = True,
        keepalive_s: float = 0.5,
        hold_s: float = 5.0,
        heartbeat_s: float = 5.0,
        poll_s: float = 0.25,
        custom_prefix: str = "Pailin",
        window_title: str = "",
        discovery: VTSDiscovery | None = None,
        discovery_timeout_s: float = 2.5,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.name = name
        self.component = component or f"avatar:{name}"
        self._client = client
        self._clock = clock
        self._emotions = EmotionController(emotion_map)
        self.fade_s = fade_s
        self._backoff = reconnect_backoff
        self._face_found = face_found
        self._keepalive_s = keepalive_s
        self._hold_s = hold_s
        self._heartbeat_s = heartbeat_s
        self._poll_s = max(0.05, poll_s)
        self._prefix = "".join(c for c in custom_prefix if c.isascii() and c.isalnum()) or "Aiv"
        self._fallback_url = client.url
        self._window_title = window_title
        self._discovery = discovery
        self._own_discovery = False
        self._discovery_timeout_s = discovery_timeout_s
        self._on_event = on_event
        client.on_ff_error = self._on_ff_error
        # parameter state
        self._state: dict[str, tuple[float, float]] = {}  # name -> (value, updated)
        self._baseline: dict[str, float] = {}
        self._remap: dict[str, str] = {}  # default input -> custom parameter
        self._disabled: dict[str, str] = {}  # parameter -> reason
        self._missing_expressions: set[str] = set()
        self._last_inject = 0.0
        self._last_external = -1e9
        # link state
        self._ready = False
        self._was_ready = False
        self._reauth_needed = False
        self._probe_needed = False
        self._reconnect_needed = False
        self._health = Health(self.component, HealthState.STARTING, "not started", clock.now())
        self.connects = 0
        self.probes = 0

    # --- AvatarSink ---------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._ready and self._client.authenticated

    @property
    def client(self) -> VTSClient:
        return self._client

    @property
    def remapped(self) -> Mapping[str, str]:
        return dict(self._remap)

    def health(self) -> Health:
        if self.connected:
            notes = [f"{k}→{v}" for k, v in sorted(self._remap.items())]
            notes += [f"{k} off ({why})" for k, why in sorted(self._disabled.items())]
            if notes:
                detail = "custom parameters (map them in VTS model settings): " + ", ".join(notes)
                return self._set_health(HealthState.DEGRADED, detail)
            return self._set_health(HealthState.OK, self._client.url)
        return self._health

    async def run(self) -> None:
        """Connect, authenticate, serve, and reconnect forever (cancel to stop)."""
        failures = 0
        try:
            await self._start_discovery()
            while True:
                detail, wake_on_broadcast = await self._attempt(failures)
                failures = 1 if self._was_ready else failures + 1
                self._set_health(HealthState.DOWN, detail)
                delay = backoff_delay(failures, self._backoff)
                log.info("VTS: %s; retrying in %.1f s", detail, delay)
                await self._wait_backoff(delay, wake_on_broadcast)
        finally:
            self._ready = False
            await self._client.aclose()
            if self._own_discovery and self._discovery is not None:
                self._discovery.close()
            self._set_health(HealthState.DOWN, "stopped")

    async def _attempt(self, failures: int) -> tuple[str, bool]:
        """One connection's lifetime; returns (health detail, wake early on a broadcast)."""
        self._was_ready = False
        try:
            await self._session()
            return "VTS connection closed", True
        except _AuthFailed as exc:
            return f"plugin not authorised; click Allow in VTube Studio ({exc})", False
        except (OSError, TimeoutError, WebSocketException) as exc:
            return f"VTS unreachable at {self._client.url}: {_describe(exc)}", True
        except (VTSAPIError, _Reconnect) as exc:
            return f"VTS link reset: {_describe(exc)}", True
        except Exception as exc:
            log.warning("VTS session failed", exc_info=failures == 0)
            return _describe(exc), True
        finally:
            self._was_ready = self._ready
            self._ready = False
            await self._client.aclose()

    def set_params(self, values: Mapping[str, float]) -> None:
        now = self._clock.now()
        self._last_external = now
        clean: dict[str, float] = {}
        for key, value in values.items():
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                log.debug("VTS: ignoring non-numeric %s=%r", key, value)
                continue
            self._state[key] = (clean[key], now)
        if self._ready and clean:
            self._inject(clean)

    async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None:
        before = self._emotions.current
        activate, deactivate = self._emotions.apply(emotion)
        changed = self._emotions.current != before
        self._baseline = self._emotions.baseline()
        if not self._ready:
            return  # reconciled on connect
        if not self._driven():
            self._inject(self._baseline)
        await self._sync_expressions(activate, deactivate, fade_s)
        hotkey = self._emotions.hotkey()
        if changed and hotkey:
            await self.trigger(hotkey)

    async def trigger(self, hotkey: str) -> bool:
        if not self._ready:
            return False
        try:
            await self._call("HotkeyTriggerRequest", {"hotkeyID": hotkey})
        except _SOFT_ERRORS as exc:  # queue full / cooldown / unknown: cosmetic, never fatal
            log.info("VTS hotkey %r failed: %s", hotkey, _describe(exc))
            return False
        return True

    async def move(
        self,
        *,
        rotation: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        size: float = 0.0,
        seconds: float = 0.25,
        relative: bool = True,
    ) -> None:
        if not self._ready:
            return
        fields = {
            "positionX": _clamp(x, -1000.0, 1000.0),
            "positionY": _clamp(y, -1000.0, 1000.0),
            "rotation": _clamp(rotation, -360.0, 360.0),
            "size": _clamp(size, -100.0, 100.0),
        }
        data: dict[str, Any] = {
            "timeInSeconds": _clamp(seconds, 0.0, 2.0),
            "valuesAreRelativeToModel": relative,
        }
        data.update({k: v for k, v in fields.items() if v != 0.0} if relative else fields)
        try:
            await self._call("MoveModelRequest", data)
        except _SOFT_ERRORS as exc:
            log.info("VTS move failed: %s", _describe(exc))

    async def spin(self, turns: int = 1, seconds: float = 1.0) -> None:
        """Spin the model: ``4 × turns`` relative 90° moves (a single 360° is unverified)."""
        quarter = max(0.0, seconds) / 4.0
        for _ in range(4 * max(1, turns)):
            if not self._ready:
                return
            await self.move(rotation=90.0, seconds=quarter)
            await self._clock.sleep(quarter)

    # --- session ------------------------------------------------------------------------------
    async def _session(self) -> None:
        url = await self._resolve_url()
        await self._client.connect(url)
        if not await self._client.authenticate():
            raise _AuthFailed(self._client.auth_error or "rejected")
        self._ready = True
        self._reauth_needed = self._probe_needed = self._reconnect_needed = False
        self.connects += 1
        self._set_health(HealthState.OK, self._client.url)
        log.info("VTS connected and authenticated at %s", self._client.url)
        await self._on_connected()
        await self._serve()

    async def _on_connected(self) -> None:
        self._restore()
        try:
            await self._call("EventSubscriptionRequest", _subscription("ModelLoadedEvent"))
        except _SOFT_ERRORS as exc:
            log.info("VTS: cannot subscribe to ModelLoadedEvent: %s", _describe(exc))
        await self._reconcile_expressions()

    async def _serve(self) -> None:
        client = self._client
        while client.connected:
            if self._reconnect_needed:
                raise _Reconnect("an awaited VTS request timed out")
            if self._reauth_needed:
                self._reauth_needed = False
                self._ready = False
                if not await client.authenticate():
                    raise _AuthFailed(client.auth_error or "token revoked")
                self._ready = True
                self._restore()
            if self._probe_needed:
                self._probe_needed = False
                await self._probe_params()
            await self._drain_events()
            now = self._clock.now()
            if now - self._last_inject >= self._keepalive_s:
                self._restore()
            if now - client.last_rx >= self._heartbeat_s:
                await self._call("APIStateRequest")  # raises on timeout → reconnect
            await self._clock.sleep(self._poll_s)

    async def _drain_events(self) -> None:
        events = self._client.events
        while not events.empty():
            event = events.get_nowait()
            if event.get("messageType") == "ModelLoadedEvent":
                data = event.get("data") or {}
                if data.get("modelLoaded"):
                    log.info("VTS model loaded: %s", data.get("modelName", "?"))
                    self._missing_expressions.clear()
                    self._restore()
                    await self._reconcile_expressions()
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:
                    log.exception("VTS on_event callback failed")

    async def _call(
        self, message_type: str, data: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """An awaited request; a timeout (2 s) sends the sink back to the reconnect loop."""
        try:
            return await self._client.request(message_type, data)
        except DeadlineExceeded:
            log.warning("VTS %s timed out; reconnecting", message_type)
            self._reconnect_needed = True
            raise
        except VTSAPIError as exc:
            if exc.error_id == VTSErrorID.REQUEST_REQUIRES_AUTH:
                self._reauth_needed = True
            raise

    # --- parameters ---------------------------------------------------------------------------
    def _driven(self) -> bool:
        """Someone (the driver) is streaming parameters; it owns the emotion baseline then."""
        return self._clock.now() - self._last_external < 1.0

    def _held_values(self) -> dict[str, float]:
        now = self._clock.now()
        values = {} if self._driven() else dict(self._baseline)
        values.update({k: v for k, (v, t) in self._state.items() if now - t <= self._hold_s})
        return values

    def _restore(self) -> None:
        values = self._held_values()
        if values:
            self._inject(values)

    def _inject(self, values: Mapping[str, float]) -> bool:
        mapped: dict[str, float] = {}
        for key, value in values.items():
            if key not in self._disabled:
                mapped[self._remap.get(key, key)] = value
        if not mapped:
            return False
        ok = self._client.inject(mapped, face_found=self._face_found)
        if ok:
            self._last_inject = self._clock.now()
        return ok

    def _on_ff_error(self, error_id: int, message: str) -> None:
        if error_id == VTSErrorID.REQUEST_REQUIRES_AUTH:
            self._reauth_needed = True
        elif error_id in (VTSErrorID.INJECT_PARAM_HELD, VTSErrorID.INJECT_PARAM_NOT_FOUND):
            self._probe_needed = True

    async def _probe_params(self) -> None:
        """Find which parameters VTS rejects (one awaited injection each) and remap them."""
        self.probes += 1
        values = self._held_values()
        names = [k for k in values if k not in self._disabled]
        results = await asyncio.gather(
            *(self._probe_one(k, values[k]) for k in names), return_exceptions=True
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, VTSAPIError) and result.error_id in (
                VTSErrorID.INJECT_PARAM_HELD,
                VTSErrorID.INJECT_PARAM_NOT_FOUND,
            ):
                await self._fallback(name, result)
            elif isinstance(result, BaseException):
                log.debug("VTS probe of %s: %r", name, result)

    async def _probe_one(self, name: str, value: float) -> None:
        target = self._remap.get(name, name)
        await self._call(
            "InjectParameterDataRequest",
            {
                "faceFound": self._face_found,
                "mode": "set",
                "parameterValues": [{"id": target, "value": value}],
            },
        )

    async def _fallback(self, name: str, error: VTSAPIError) -> None:
        custom = self._remap.get(name)
        held = error.error_id == VTSErrorID.INJECT_PARAM_HELD
        if held and custom is not None:
            self._disabled[name] = f"{custom} is controlled by another plugin"
            log.warning("VTS: %s", self._disabled[name])
            return
        custom = custom or self._custom_name(name)
        lo, hi, default = PARAM_RANGES.get(name, _DEFAULT_RANGE)
        try:
            await self._call(
                "ParameterCreationRequest",
                {
                    "parameterName": custom,
                    "explanation": f"AI_Vtube {name}"[:256],
                    "min": lo,
                    "max": hi,
                    "defaultValue": default,
                },
            )
        except _SOFT_ERRORS as exc:
            self._disabled[name] = f"cannot create {custom}: {_describe(exc)}"
            log.warning("VTS: %s is not injectable and %s", name, self._disabled[name])
            return
        self._remap[name] = custom
        why = "is controlled by another plugin" if held else "is not injectable"
        log.warning(
            "VTS: %s %s; injecting custom parameter %s instead. Map %s to your model's "
            "parameter in VTS model settings. / พารามิเตอร์ %s ใช้ไม่ได้ ให้ผูก %s ในหน้าตั้งค่าโมเดล",
            name, why, custom, custom, name, custom,
        )  # fmt: skip
        self._restore()

    def _custom_name(self, name: str) -> str:
        base = "".join(c for c in name if c.isascii() and c.isalnum())
        return (self._prefix + base)[:32]

    # --- expressions --------------------------------------------------------------------------
    async def _sync_expressions(
        self, activate: list[str], deactivate: list[str], fade_s: float
    ) -> None:
        for file in deactivate:
            await self._expression(file, False, fade_s)
        for file in activate:
            await self._expression(file, True, fade_s)

    async def _expression(self, file: str, active: bool, fade_s: float) -> None:
        if file in self._missing_expressions or not self._ready:
            return
        data = {"expressionFile": file, "active": active, "fadeTime": _clamp(fade_s, 0.0, 2.0)}
        try:
            await self._call("ExpressionActivationRequest", data)
        except VTSAPIError as exc:
            if exc.error_id in (
                VTSErrorID.EXPRESSION_BAD_FILENAME,
                VTSErrorID.EXPRESSION_NOT_FOUND,
            ):
                self._missing_expressions.add(file)
                log.warning(
                    "VTS: expression %s is not in the loaded model; fix avatar.emotion_map", file
                )
            else:
                log.info("VTS expression %s → %s failed: %s", file, active, exc)
        except (VTSDisconnected, DeadlineExceeded) as exc:
            log.info("VTS expression %s → %s failed: %s", file, active, _describe(exc))

    async def _reconcile_expressions(self) -> None:
        try:
            data = await self._call("ExpressionStateRequest", {"details": False})
        except _SOFT_ERRORS as exc:
            log.info("VTS: cannot read expression state: %s", _describe(exc))
            return
        present: dict[str, bool] = {}
        for item in data.get("expressions") or ():
            if isinstance(item, Mapping) and item.get("file"):
                present[str(item["file"])] = bool(item.get("active"))
        if data.get("modelLoaded", True):
            self._missing_expressions = {f for f in self._emotions.managed if f not in present}
            for file in sorted(self._missing_expressions & self._emotions.active):
                log.warning("VTS: expression %s is not in the loaded model", file)
        activate, deactivate = self._emotions.reconcile(present)
        await self._sync_expressions(activate, deactivate, self.fade_s)

    # --- discovery / backoff ------------------------------------------------------------------
    async def _start_discovery(self) -> None:
        if self._discovery is None or self._discovery.running:
            return
        try:
            await self._discovery.start()
            self._own_discovery = True
        except OSError as exc:
            log.warning("VTS discovery disabled (UDP %d busy?): %s", self._discovery.port, exc)
            self._discovery = None

    async def _resolve_url(self) -> str:
        disc = self._discovery
        if disc is None:
            return self._fallback_url
        prefer = url_port(self._fallback_url)
        inst = disc.pick(self._window_title, prefer_port=prefer)
        if inst is None and self._window_title:
            inst = await disc.wait_for(
                self._window_title, timeout=self._discovery_timeout_s, prefer_port=prefer
            )
            if inst is None:
                log.warning(
                    "VTS: no window titled %r found by discovery; using %s",
                    self._window_title,
                    self._fallback_url,
                )
        if inst is None:
            return self._fallback_url
        return f"ws://127.0.0.1:{int(inst['port'])}"

    async def _wait_backoff(self, delay: float, wake_on_broadcast: bool) -> None:
        disc = self._discovery
        if disc is None or not wake_on_broadcast:
            await self._clock.sleep(delay)
            return
        await disc.wait_for(
            self._window_title,
            timeout=delay,
            prefer_port=url_port(self._fallback_url),
            after=self._clock.now(),
        )

    def _set_health(self, state: HealthState, detail: str) -> Health:
        if self._health.state is not state or self._health.detail != detail:
            since = self._health.since if self._health.state is state else self._clock.now()
            self._health = Health(self.component, state, detail, since)
        return self._health


def _subscription(event: str) -> dict[str, Any]:
    return {"eventName": event, "subscribe": True, "config": {}}
