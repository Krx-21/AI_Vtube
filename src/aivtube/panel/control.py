"""``CoreControl``: the core's ``ControlSurface`` (ARCHITECTURE.md §2.11, §3.12, invariant I4).

Every operator command from the panel, hotkeys and the text console goes through
:meth:`CoreControl.execute`, which routes it, publishes ``OperatorAction`` on the bus and writes
one ``op_audit`` row (through ``TaskSupervisor.track``, so a slow database never delays a
FREEZE).

Routing:

- **Brain commands** (SKIP, FREEZE, RESUME, GO_LIVE, CHAT_INTAKE, SAY, DIRECT, APPROVE,
  MUTE_USER, TTS_IDENTITY, END_STREAM, and FAKE_CHAT/INJECT_EVENT when no ``ingest`` is wired)
  go to ``Brain.control``. With ``character=None`` the global ones go to every brain; the
  others go to the default character.
- **FREEZE** additionally stops all speech itself (``speech.stop(None, "now")``) in parallel with
  the brains and switches the tool registry to ``off``, so silence and "no tools" do not
  depend on a healthy brain. RESUME restores the tools mode (a TOOLS_MODE sent while frozen is
  applied then). **SKIP** falls back to a direct stop when no brain answers.
- **MUTE/UNMUTE, MIC_MODE, PTT** act on ``SpeechOutput`` directly and then notify the brains
  (their flags), ignoring the brains' answers.
- **APPROVE** ``{"request": id, "approved": bool}`` answers a pending tool call in the
  ``approvals`` queue (``ToolApprovalQueue``); any other APPROVE goes to the brains.
- **STRICT** ``{"on": bool}`` → ``set_strict(character, on, reason="operator")`` on the gate
  (``LayeredSafetyGate``) for one or every character, then notifies the brains; a gate without
  ``set_strict`` leaves STRICT to the brains.
- **LLM_USE/LLM_ROLLBACK** → ``LLMRouter``; **TOOLS_MODE/TOOL_ENABLE** → ``ToolRegistry``;
  **MEMORY_EDIT/MEMORY_STATUS** → the character's ``MemoryStore``; **FILTER_RELOAD** →
  ``SafetyGate.reload`` in a worker thread (a bad list file answers ``ok=False`` and the old
  lists stay active); **RESTART** → the injected ``restart`` callback.
- ``handlers`` overrides the routing for any kind (e.g. TTS_IDENTITY wired by the app).

FREEZE, SKIP, MUTE, UNMUTE and PTT take the fast path, and APPROVE is answered at once: they
never wait behind another command. All other commands run one at a time in arrival order.
Every awaited call has a deadline (I2). Every command, even a cancelled one, is audited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final, Literal, Protocol, runtime_checkable

from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.events import OperatorAction
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.llm import LLMRouter
from aivtube.contracts.memory import MemoryStore, SlotsFull
from aivtube.contracts.safety import SafetyGate
from aivtube.contracts.speech import MicMode, SpeechOutput, VoicePolicy
from aivtube.contracts.tools import ToolRegistry
from aivtube.contracts.types import ChatMessage
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.panel._json import to_jsonable
from aivtube.panel.approvals import ToolApprovalQueue
from aivtube.panel.commands import FAST_KINDS, validate_args
from aivtube.panel.ingest import alert_message, chat_message

__all__ = [
    "BRAIN_KINDS",
    "GLOBAL_KINDS",
    "BrainControl",
    "CoreControl",
    "OpAuditSink",
    "OpHandler",
]

log = logging.getLogger("aivtube.panel.control")

OpHandler = Callable[[OpCommand], Awaitable[OpResult]]

#: Commands owned by ``Brain.control``.
BRAIN_KINDS: Final = frozenset(
    {
        OpKind.SKIP,
        OpKind.FREEZE,
        OpKind.RESUME,
        OpKind.GO_LIVE,
        OpKind.CHAT_INTAKE,
        OpKind.SAY,
        OpKind.DIRECT,
        OpKind.FAKE_CHAT,
        OpKind.INJECT_EVENT,
        OpKind.APPROVE,
        OpKind.MUTE_USER,
        OpKind.STRICT,
        OpKind.TTS_IDENTITY,
        OpKind.END_STREAM,
    }
)
#: Commands that apply to every character when ``cmd.character`` is None.
GLOBAL_KINDS: Final = frozenset(
    {
        OpKind.SKIP,
        OpKind.FREEZE,
        OpKind.RESUME,
        OpKind.GO_LIVE,
        OpKind.CHAT_INTAKE,
        OpKind.APPROVE,
        OpKind.MUTE_USER,
        OpKind.STRICT,
        OpKind.END_STREAM,
        OpKind.MUTE,
        OpKind.UNMUTE,
        OpKind.MIC_MODE,
        OpKind.PTT,
    }
)
_EDIT_FIELDS: Final = ("text", "subject", "importance", "locked", "pinned")
#: Commands that never wait behind another one: the fast path (I4) plus APPROVE, whose
#: answer is instant and time-critical (the tool call or review gate is waiting).
_UNLOCKED: Final = FAST_KINDS | {OpKind.APPROVE}


@runtime_checkable
class BrainControl(Protocol):
    """What ``CoreControl`` needs from ``brain.loop.Brain`` (modules.json ``brain.loop``)."""

    async def control(self, cmd: OpCommand) -> OpResult: ...

    def snapshot(self) -> Any: ...


class OpAuditSink(Protocol):
    """``OpsDb.log_op``: one ``op_audit`` row (never raises)."""

    async def log_op(self, **row: Any) -> None: ...


class _Refused(Exception):
    """A command that cannot run as given (unknown character, missing store, …)."""


class CoreControl:
    """The core's ``ControlSurface``; see the module docstring for the routing rules."""

    def __init__(
        self,
        brains: Mapping[str, BrainControl],
        *,
        router: LLMRouter,
        registry: ToolRegistry,
        speech: SpeechOutput,
        memory_by_char: Mapping[str, MemoryStore],
        safety: SafetyGate,
        ops: OpAuditSink | None,
        bus: EventBus,
        clock: Clock,
        restart: Callable[[str], Awaitable[None]],
        tasks: TaskSupervisor,
        ingest: Callable[[ChatMessage], None] | None = None,
        handlers: Mapping[OpKind, OpHandler] | None = None,
        approvals: ToolApprovalQueue | None = None,
        voice_policy: VoicePolicy | None = None,
        tools_mode: str = "live",
        default_character: str | None = None,
        fast_timeout_s: float = 0.5,
        timeout_s: float = 10.0,
        restart_timeout_s: float = 30.0,
    ) -> None:
        self._brains = dict(brains)
        self._router = router
        self._registry = registry
        self._speech = speech
        self._memory = dict(memory_by_char)
        self._safety = safety
        self._ops = ops
        self._bus = bus
        self._clock = clock
        self._restart = restart
        self._tasks = tasks
        self._ingest = ingest
        self._approvals = approvals
        self._handlers = dict(handlers or {})
        self._policy = voice_policy or VoicePolicy()
        self._before_deafen: MicMode = (
            self._policy.mic_mode if self._policy.mic_mode != "deafened" else "open"
        )
        self._tools_mode = tools_mode
        characters = list(self._brains) or list(self._memory)
        self._default = default_character or (characters[0] if characters else None)
        self._fast_timeout = fast_timeout_s
        self._timeout = timeout_s
        self._restart_timeout = restart_timeout_s
        self._lock = asyncio.Lock()
        self._frozen = False
        self._muted = False
        self._tools_resume: str | None = None  # the tools mode to restore at RESUME
        self._routes: dict[OpKind, OpHandler] = {
            OpKind.FREEZE: self._freeze,
            OpKind.SKIP: self._skip,
            OpKind.RESUME: self._resume,
            OpKind.MUTE: self._mute,
            OpKind.UNMUTE: self._mute,
            OpKind.MIC_MODE: self._mic_mode,
            OpKind.PTT: self._ptt,
            OpKind.LLM_USE: self._llm_use,
            OpKind.LLM_ROLLBACK: self._llm_rollback,
            OpKind.TOOLS_MODE: self._tools_mode_cmd,
            OpKind.TOOL_ENABLE: self._tool_enable,
            OpKind.MEMORY_EDIT: self._memory_edit,
            OpKind.MEMORY_STATUS: self._memory_status,
            OpKind.FILTER_RELOAD: self._filter_reload,
            OpKind.RESTART: self._restart_cmd,
            OpKind.FAKE_CHAT: self._fake_chat,
            OpKind.INJECT_EVENT: self._fake_chat,
            OpKind.APPROVE: self._approve,
            OpKind.STRICT: self._strict,
        }

    # --- ControlSurface ---------------------------------------------------------------------
    async def execute(self, cmd: OpCommand) -> OpResult:
        """Run ``cmd``; never raises (except cancellation). The latency covers the whole call.

        Every command is audited, including one cancelled mid-way (e.g. by a caller's timeout).
        """
        t0 = self._clock.now()
        try:
            cmd = dataclasses.replace(cmd, args=validate_args(cmd.kind, cmd.args))
            if cmd.kind in _UNLOCKED:
                result = await self._dispatch(cmd)
            else:
                async with self._lock:
                    result = await self._dispatch(cmd)
        except asyncio.CancelledError:
            latency_ms = round((self._clock.now() - t0) * 1000.0, 2)
            self._record(cmd, OpResult(False, "cancelled", latency_ms))
            raise
        except (_Refused, ValueError) as exc:
            result = OpResult(False, str(exc))
        except DeadlineExceeded as exc:
            result = OpResult(False, f"timeout: {exc.what}")
        except Exception as exc:
            log.exception("operator command %s failed", cmd.kind.value)
            result = OpResult(False, f"{type(exc).__name__}: {exc}")
        latency_ms = round((self._clock.now() - t0) * 1000.0, 2)
        final = OpResult(result.ok, result.detail, latency_ms)
        self._record(cmd, final)
        return final

    def snapshot(self) -> Mapping[str, Any]:
        """Operator view of the core; each part is best-effort and never raises."""
        snap: dict[str, Any] = {
            "default_character": self._default,
            "characters": {},
            "frozen": self._frozen,
            "muted": self._muted,
            "mic_mode": self._policy.mic_mode,
            "mic_mode_before_deafen": self._before_deafen,
            "ptt_active": self._policy.ptt_active,
            "voice_policy": to_jsonable(self._policy),
            "tools": self._safe(self._tools_snapshot, {"mode": self._tools_mode}),
            "approvals": self._safe(self._approvals.snapshot, []) if self._approvals else [],
            "tts": {},
            "strict": {},
        }
        for name, brain in self._brains.items():
            snap["characters"][name] = self._safe(functools.partial(_brain_snapshot, brain), {})
        chars = list(self._brains) or ([self._default] if self._default else [])
        for name in chars:
            snap["tts"][name] = self._safe(
                functools.partial(_constraints, self._speech, name), None
            )
            snap["strict"][name] = self._safe(
                functools.partial(self._safety.strict_mode, name), False
            )
        status = getattr(self._safety, "status", None)
        if callable(status):
            snap["safety"] = self._safe(lambda: to_jsonable(status()), None)
        snap["speech_ready"] = self._safe(self._speech.ready, False)
        snap["llm"] = {
            "active": self._safe(self._router.active, None),
            "providers": self._safe(lambda: to_jsonable(self._router.status()), []),
        }
        return snap

    # --- dispatch ---------------------------------------------------------------------------
    async def _dispatch(self, cmd: OpCommand) -> OpResult:
        handler = self._handlers.get(cmd.kind) or self._routes.get(cmd.kind)
        timeout = self._timeout
        if cmd.kind is OpKind.RESTART:
            timeout = self._restart_timeout
        if handler is not None:
            async with deadline(timeout, what=f"op {cmd.kind.value}", clock=self._clock):
                return await handler(cmd)
        if cmd.kind in BRAIN_KINDS:
            return await self._to_brains(cmd, self._targets(cmd), timeout)
        return OpResult(False, f"unsupported command {cmd.kind.value}")

    def _targets(self, cmd: OpCommand) -> list[tuple[str, BrainControl]]:
        if cmd.character is not None:
            brain = self._brains.get(cmd.character)
            if brain is None:
                raise _Refused(f"unknown character {cmd.character!r}")
            return [(cmd.character, brain)]
        if cmd.kind in GLOBAL_KINDS:
            return list(self._brains.items())
        if self._default is not None and self._default in self._brains:
            return [(self._default, self._brains[self._default])]
        return []

    async def _guard(self, aw: Awaitable[Any], what: str, limit_s: float) -> OpResult:
        """Await ``aw`` under a deadline; failures become ``OpResult(False, ...)``."""
        try:
            async with deadline(limit_s, what=what, clock=self._clock):
                value = await aw
        except asyncio.CancelledError:
            raise
        except DeadlineExceeded:
            log.warning("%s timed out after %.2f s", what, limit_s)
            return OpResult(False, f"{what}: timeout")
        except Exception as exc:
            log.warning("%s failed: %s", what, exc, exc_info=True)
            return OpResult(False, f"{what}: {type(exc).__name__}: {exc}")
        return value if isinstance(value, OpResult) else OpResult(True)

    async def _to_brains(
        self, cmd: OpCommand, targets: list[tuple[str, BrainControl]], limit_s: float
    ) -> OpResult:
        if not targets:
            return OpResult(False, "no brain for this command")
        results = await asyncio.gather(
            *(self._guard(brain.control(cmd), f"brain {name}", limit_s) for name, brain in targets)
        )
        return _combine([(name, r) for (name, _), r in zip(targets, results, strict=True)])

    async def _notify_brains(self, cmd: OpCommand, targets: list[tuple[str, BrainControl]]) -> None:
        results = await asyncio.gather(
            *(
                self._guard(brain.control(cmd), f"brain {name}", self._fast_timeout)
                for name, brain in targets
            )
        )
        for (name, _), r in zip(targets, results, strict=True):
            if not r.ok:
                log.debug("brain %s did not take %s: %s", name, cmd.kind.value, r.detail)

    # --- kill ladder --------------------------------------------------------------------------
    async def _freeze(self, cmd: OpCommand) -> OpResult:
        targets = self._targets(cmd)
        jobs: list[Awaitable[OpResult]] = [
            self._guard(brain.control(cmd), f"brain {name}", self._fast_timeout)
            for name, brain in targets
        ]
        if cmd.character is None:
            stop = self._speech.stop(None, "now", "operator_freeze")
            jobs.append(self._guard(stop, "speech stop", self._fast_timeout))
            self._tools_off_for_freeze()
        results = await asyncio.gather(*jobs)
        names = [name for name, _ in targets] + (["speech"] if cmd.character is None else [])
        if cmd.character is None:
            self._frozen = True
        return _combine(list(zip(names, results, strict=True)))

    def _tools_off_for_freeze(self) -> None:
        """Switch the registry off until RESUME (never raises: FREEZE must not fail on it)."""
        try:
            current = self._registry_mode()
            if self._tools_resume is None:
                self._tools_resume = current
            if current != "off":
                self._registry.set_mode("off")
                self._tools_mode = "off"
        except Exception:
            log.exception("switching tools off for FREEZE failed")

    async def _skip(self, cmd: OpCommand) -> OpResult:
        targets = self._targets(cmd)
        if targets:
            result = await self._to_brains(cmd, targets, self._fast_timeout)
            if result.ok or cmd.character is not None:
                return result
        stop = self._speech.stop(None, "now", "operator_skip")
        return await self._guard(stop, "speech stop", self._fast_timeout)

    async def _resume(self, cmd: OpCommand) -> OpResult:
        result = await self._to_brains(cmd, self._targets(cmd), self._timeout)
        if cmd.character is None and (result.ok or not self._brains):
            self._frozen = False
            resume, self._tools_resume = self._tools_resume, None
            if resume is not None and resume != self._registry_mode():
                try:
                    self._registry.set_mode(_tools_mode(resume))
                    self._tools_mode = resume
                except Exception as exc:
                    log.exception("restoring tools mode %s after RESUME failed", resume)
                    return OpResult(False, f"resumed, but tools mode {resume} failed: {exc}")
        return result

    async def _mute(self, cmd: OpCommand) -> OpResult:
        targets = self._targets(cmd)  # validates the character before acting
        on = cmd.kind is OpKind.MUTE
        result = await self._guard(self._speech.mute(on), "speech mute", self._fast_timeout)
        if result.ok:
            self._muted = on
        await self._notify_brains(cmd, targets)
        return result

    async def _mic_mode(self, cmd: OpCommand) -> OpResult:
        targets = self._targets(cmd)
        mode: MicMode = cmd.args["mode"]
        policy = dataclasses.replace(self._policy, mic_mode=mode)
        result = await self._set_policy(policy)
        if result.ok:
            if mode == "deafened" and self._policy.mic_mode != "deafened":
                self._before_deafen = self._policy.mic_mode
            self._policy = policy
            await self._notify_brains(cmd, targets)
        return result

    async def _ptt(self, cmd: OpCommand) -> OpResult:
        targets = self._targets(cmd)
        policy = dataclasses.replace(self._policy, ptt_active=bool(cmd.args["active"]))
        result = await self._set_policy(policy)
        if result.ok:
            self._policy = policy
            await self._notify_brains(cmd, targets)
        return result

    async def _set_policy(self, policy: VoicePolicy) -> OpResult:
        return await self._guard(
            self._speech.set_policy(policy), "speech policy", self._fast_timeout
        )

    # --- services -------------------------------------------------------------------------------
    async def _llm_use(self, cmd: OpCommand) -> OpResult:
        name = str(cmd.args["name"])
        try:
            self._router.promote(name)
        except KeyError:
            return OpResult(False, f"unknown provider {name!r}")
        except Exception as exc:  # e.g. ConsentRequired for a cloud provider
            return OpResult(False, str(exc))
        return OpResult(True, f"active at the next decision: {name}")

    async def _llm_rollback(self, cmd: OpCommand) -> OpResult:
        before = self._router.active()
        self._router.rollback()
        after = self._router.active()
        return OpResult(True, f"{before} -> {after}" if after != before else "nothing to roll back")

    async def _tools_mode_cmd(self, cmd: OpCommand) -> OpResult:
        mode = _tools_mode(cmd.args["mode"])
        if self._tools_resume is not None:  # frozen: tools stay off until RESUME
            self._tools_resume = mode
            return OpResult(True, f"{mode} after RESUME (tools are off while frozen)")
        self._registry.set_mode(mode)
        self._tools_mode = mode
        return OpResult(True, mode)

    async def _tool_enable(self, cmd: OpCommand) -> OpResult:
        name, enabled = str(cmd.args["name"]), bool(cmd.args["enabled"])
        try:
            self._registry.set_enabled(name, enabled)
        except KeyError:
            return OpResult(False, f"unknown tool {name!r}")
        return OpResult(True, f"{name} {'enabled' if enabled else 'disabled'}")

    def _store(self, cmd: OpCommand) -> MemoryStore:
        character = cmd.character or self._default
        store = self._memory.get(character) if character is not None else None
        if store is None:
            raise _Refused(f"no memory store for {character!r}")
        return store

    async def _memory_status(self, cmd: OpCommand) -> OpResult:
        store = self._store(cmd)
        memory_id, status = int(cmd.args["id"]), cmd.args["status"]
        try:
            await store.set_status(memory_id, status, by=f"operator:{cmd.operator}")
        except SlotsFull:
            return OpResult(False, "all core slots are full; replace or delete one first")
        except PermissionError as exc:  # MemoryLocked, ViewerOptedOut
            return OpResult(False, f"refused: {exc}")
        except LookupError:
            return OpResult(False, f"unknown memory {memory_id}")
        return OpResult(True, f"memory {memory_id} -> {status}")

    async def _memory_edit(self, cmd: OpCommand) -> OpResult:
        store = self._store(cmd)
        edit = getattr(store, "edit", None)
        if not callable(edit):
            return OpResult(False, "this memory store does not support edits")
        memory_id = int(cmd.args["id"])
        changes = {k: cmd.args[k] for k in _EDIT_FIELDS if cmd.args.get(k) is not None}
        if not changes:
            return OpResult(False, "nothing to change")
        params = inspect.signature(edit).parameters
        unsupported = [k for k in changes if k not in params]
        if unsupported:
            return OpResult(False, f"cannot edit {', '.join(unsupported)} on this store")
        try:
            await edit(memory_id, by=f"operator:{cmd.operator}", **changes)
        except PermissionError as exc:
            return OpResult(False, f"refused: {exc}")
        except LookupError:
            return OpResult(False, f"unknown memory {memory_id}")
        return OpResult(True, f"memory {memory_id} edited")

    async def _filter_reload(self, cmd: OpCommand) -> OpResult:
        # Blocking file I/O: off the loop. The tier-0 filter swaps its rules atomically, so a
        # bad file (``safety.FilterListError``, a ``ValueError``) keeps the previous lists.
        try:
            await asyncio.to_thread(self._safety.reload)
        except (ValueError, OSError) as exc:
            log.warning("filter reload failed; previous lists stay active: %s", exc)
            return OpResult(False, f"filter lists not reloaded: {exc}")
        return OpResult(True, "filters reloaded")

    async def _strict(self, cmd: OpCommand) -> OpResult:
        set_strict = getattr(self._safety, "set_strict", None)
        targets = self._targets(cmd)
        if not callable(set_strict):
            return await self._to_brains(cmd, targets, self._timeout)
        on = bool(cmd.args["on"])
        characters = [cmd.character] if cmd.character is not None else self._characters()
        if not characters:
            return OpResult(False, "no character for this command")
        takes_reason = "reason" in inspect.signature(set_strict).parameters
        for character in characters:
            if takes_reason:
                set_strict(character, on, reason="operator")
            else:
                set_strict(character, on)
        await self._notify_brains(cmd, targets)
        return OpResult(True, f"strict {'on' if on else 'off'}: {', '.join(characters)}")

    async def _approve(self, cmd: OpCommand) -> OpResult:
        request = cmd.args.get("request")
        if request is None:
            return await self._to_brains(cmd, self._targets(cmd), self._timeout)
        if self._approvals is None:
            return OpResult(False, "no tool approval queue")
        pending = self._approvals.get(str(request))
        if pending is None or not self._approvals.resolve(str(request), cmd.args["approved"]):
            return OpResult(False, f"no pending approval {request!r}")
        verdict = "approved" if cmd.args["approved"] else "denied"
        return OpResult(True, f"{pending.tool} {verdict}")

    async def _restart_cmd(self, cmd: OpCommand) -> OpResult:
        component = str(cmd.args["component"])
        await self._restart(component)
        return OpResult(True, f"restarting {component}")

    async def _fake_chat(self, cmd: OpCommand) -> OpResult:
        if self._ingest is None:
            return await self._to_brains(cmd, self._targets(cmd), self._timeout)
        if cmd.kind is OpKind.INJECT_EVENT:
            msg = alert_message(cmd.args, clock=self._clock)
        else:
            msg = chat_message(
                str(cmd.args.get("user") or cmd.args.get("name") or "operator"),
                str(cmd.args.get("text") or ""),
                clock=self._clock,
            )
        self._ingest(msg)
        return OpResult(True, msg.id)

    # --- audit ----------------------------------------------------------------------------------
    def _record(self, cmd: OpCommand, result: OpResult) -> None:
        args = to_jsonable(dict(cmd.args))
        try:
            self._bus.publish(
                OperatorAction(
                    kind=cmd.kind.value,
                    args=args,
                    ok=result.ok,
                    latency_ms=result.latency_ms,
                    character=cmd.character,
                )
            )
        except Exception:  # the bus never raises, but a fake might
            log.exception("publishing OperatorAction failed")
        if cmd.kind is OpKind.FREEZE or not result.ok:
            log.warning(
                "operator %s: %s %s -> ok=%s %s (%.1f ms)",
                cmd.operator,
                cmd.kind.value,
                args,
                result.ok,
                result.detail,
                result.latency_ms,
            )
        else:
            log.info("operator %s: %s -> %s", cmd.operator, cmd.kind.value, result.detail or "ok")
        if self._ops is None:
            return
        row = {
            "ts": self._clock.wall(),
            "operator": cmd.operator,
            "command": cmd.kind.value,
            "args": {**args, **({"character": cmd.character} if cmd.character else {})},
            "result": "ok" if result.ok else f"error: {result.detail}",
            "latency_ms": result.latency_ms,
        }
        self._tasks.track(self._write_audit(row), name=f"op-audit:{cmd.kind.value}")

    async def _write_audit(self, row: Mapping[str, Any]) -> None:
        assert self._ops is not None
        try:
            async with deadline(5.0, what="op_audit write", clock=self._clock):
                await self._ops.log_op(**row)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("op_audit write failed for %s", row.get("command"))

    def _registry_mode(self) -> str:
        """The registry's own mode when it exposes one (``PolicyToolRegistry.mode``)."""
        mode = getattr(self._registry, "mode", None)
        return mode if isinstance(mode, str) else self._tools_mode

    def _tools_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        extra = getattr(self._registry, "snapshot", None)  # PolicyToolRegistry: per-tool state
        if callable(extra):
            value = to_jsonable(extra())
            if isinstance(value, dict):
                snap.update(value)
        snap["mode"] = self._registry_mode()
        if self._tools_resume is not None:
            snap["resume_mode"] = self._tools_resume
        return snap

    def _characters(self) -> list[str]:
        names = list(self._brains) or list(self._memory)
        if not names and self._default is not None:
            names = [self._default]
        return names

    @staticmethod
    def _safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            log.debug("snapshot part failed", exc_info=True)
            return default


def _brain_snapshot(brain: BrainControl) -> Any:
    return to_jsonable(brain.snapshot())


def _constraints(speech: SpeechOutput, character: str) -> Any:
    return to_jsonable(speech.constraints(character))


def _tools_mode(value: Any) -> Literal["live", "dry_run", "off"]:
    if value == "live":
        return "live"
    if value == "dry_run":
        return "dry_run"
    if value == "off":
        return "off"
    raise ValueError(f"unknown tools mode {value!r}")


def _combine(results: list[tuple[str, OpResult]]) -> OpResult:
    ok = all(r.ok for _, r in results)
    details = [f"{name}: {r.detail}" for name, r in results if r.detail]
    return OpResult(ok, "; ".join(details))
