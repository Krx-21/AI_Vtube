"""``PolicyToolRegistry``: the policy-enforcing tool registry (ARCHITECTURE.md §3.11, §4.9).

``execute`` runs every call through the same pipeline, in this order:

1. lookup: a tool the character does not list is "unknown";
2. parse: ``raw_arguments`` with ``json.loads``, then ``json_repair``. A string cut off
   mid-value (an unterminated string, i.e. a truncated generation) is refused rather than
   repaired, because acting on half a sentence is worse than asking again;
3. JSON Schema Draft 2020-12 validation (messages name the rule, never echo the value);
4. ``SafetyGate.check_args`` on every free-text value (direction ``tool.arg_direction``,
   default ``"tool"``). BLOCK/DROP/REVIEW refuse the call; MASK/REPLACE rewrite the value;
5. policy: enabled (a disabled tool stays listed and answers "unavailable now"), mode
   ``off``, required channel capabilities, rate limit, ``dry_run`` (logs, never executes),
   then ``requires_approval`` (the approval callback, default deny after
   ``approval_timeout_s``);
6. the tool itself under ``min(policy.timeout_s, timeout_max_s)``. Nothing is ever retried,
   so side-effect tools never run twice; a failure becomes the result text;
7. one ``tool_audit`` row for every call, whatever the outcome.

Results are ``ToolResult``s whose ``content`` is a JSON string for the model. Failures never
raise (cancellation does). ``ToolRequested`` is published for every call, then either
``ToolRejected`` (refused before running) or ``ToolExecuted`` (ran, or dry run).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import json_repair
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aivtube.contracts.events import ToolExecuted, ToolRejected, ToolRequested
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.llm import ToolCall, ToolSpec
from aivtube.contracts.safety import FilterResult, SafetyGate, Verdict
from aivtube.contracts.tools import Tool, ToolContext, ToolResult
from aivtube.infra.clock import DeadlineExceeded, deadline

__all__ = [
    "UNAVAILABLE",
    "ApprovalCallback",
    "PolicyToolRegistry",
    "ToolAuditSink",
    "ToolMode",
    "free_text_values",
    "parse_tool_arguments",
    "tool_error",
]

log = logging.getLogger("aivtube.tools")

UNAVAILABLE = "unavailable now"
ToolMode = Literal["live", "dry_run", "off"]
ApprovalCallback = Callable[[str, ToolCall], Awaitable[bool]]
ArgDirection = Literal["tool", "memory", "game"]
_DIRECTIONS: frozenset[str] = frozenset({"tool", "memory", "game"})
_REFUSING = frozenset({Verdict.BLOCK, Verdict.DROP, Verdict.REVIEW})
_REWRITING = frozenset({Verdict.MASK, Verdict.REPLACE})
_BOUND_RULES = frozenset(
    {
        "maxLength",
        "minLength",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "minItems",
        "multipleOf",
    }
)
_AUDIT_RESULT_MAX = 2000
_AUDIT_ARGS_MAX = 4000


class ToolAuditSink(Protocol):
    """Where audit rows go (``OpsDb`` in the app). ``log_tool`` must not raise."""

    async def log_tool(self, **row: Any) -> None: ...


def tool_error(message: str, **extra: Any) -> ToolResult:
    """A refusal/failure result: ``{"ok": false, "error": message, ...}``."""
    body = {"ok": False, "error": message, **extra}
    return ToolResult(False, json.dumps(body, ensure_ascii=False), note=message)


# --- argument parsing ----------------------------------------------------------------------------
def _unterminated_string(text: str) -> bool:
    """True when ``text`` ends inside a double-quoted JSON string (a truncated value)."""
    in_str = escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    return in_str


def parse_tool_arguments(call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    """``(arguments, None)`` or ``(None, reason)``; see step 2 of the module docstring.

    ``raw_arguments`` is authoritative when present (the provider may already have repaired
    ``arguments``); with no raw text, ``arguments`` is used (``None`` means ``{}``).
    """
    raw = (call.raw_arguments or "").strip()
    if not raw:
        given: Any = call.arguments if call.arguments is not None else {}
        if not isinstance(given, Mapping):
            return None, "arguments are not a valid JSON object"
        return dict(given), None
    try:
        value: Any = json.loads(raw)
    except (ValueError, RecursionError):
        if _unterminated_string(raw):
            return None, "arguments were cut off (unterminated string); send the call again"
        try:
            value = json_repair.loads(raw)
        except Exception:  # json_repair is best effort
            value = None
    if not isinstance(value, dict):
        return None, "arguments are not a valid JSON object"
    return value, None


def free_text_values(args: Mapping[str, Any]) -> list[tuple[tuple[Any, ...], str]]:
    """Every string value in ``args`` (recursively) with its path, in a stable order."""
    out: list[tuple[tuple[Any, ...], str]] = []

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, str):
            if value.strip():
                out.append((path, value))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, (*path, k))
        elif isinstance(value, list | tuple):
            for i, v in enumerate(value):
                walk(v, (*path, i))

    walk(args, ())
    return out


def _set_path(root: Any, path: tuple[Any, ...], value: str) -> None:
    target = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _describe(err: ValidationError) -> str:
    """A validation message that names the rule but never echoes the offending value."""
    where = "/".join(str(p) for p in err.absolute_path) or "arguments"
    rule = err.validator
    if rule == "required" and isinstance(err.instance, Mapping):
        missing = [n for n in err.validator_value if n not in err.instance]
        return f"missing required argument(s): {', '.join(map(str, missing))}"
    if rule == "additionalProperties" and isinstance(err.instance, Mapping):
        known = set(err.schema.get("properties", {}))
        extra = sorted(str(k) for k in err.instance if k not in known)
        return f"unexpected argument(s): {', '.join(extra)}"
    if rule == "type":
        return f"{where}: must be of type {err.validator_value}"
    if rule == "enum":
        return f"{where}: must be one of {json.dumps(err.validator_value, ensure_ascii=False)}"
    if rule in _BOUND_RULES:
        return f"{where}: {rule} is {err.validator_value}"
    return f"{where}: fails {rule}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _Entry:
    tool: Tool
    validator: Draft202012Validator
    direction: ArgDirection
    enabled: bool = True


class PolicyToolRegistry:
    """``ToolRegistry`` enforcing validation, argument filtering and policy (see module doc).

    ``enabled_by_character`` lists the tools each character may use (``[tools] enabled`` in
    ``character.toml``); ``specs(character)`` is those tools in registration order, fixed
    for the session. ``set_enabled`` toggles a tool at run time without changing ``specs``.
    Extra keyword arguments beyond modules.json: ``mode``, ``approval_timeout_s``,
    ``timeout_max_s``, ``filter_timeout_s`` and ``tasks`` (to audit calls cancelled
    mid-flight without delaying the cancellation).
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        *,
        gate: SafetyGate,
        ops: ToolAuditSink | None,
        bus: EventBus,
        clock: Clock,
        enabled_by_character: Mapping[str, Sequence[str]],
        approval: ApprovalCallback | None = None,
        mode: ToolMode = "live",
        approval_timeout_s: float = 20.0,
        timeout_max_s: float = 600.0,
        filter_timeout_s: float = 2.0,
        tasks: TaskSupervisor | None = None,
    ) -> None:
        self._entries: dict[str, _Entry] = {}
        for tool in tools:
            name = tool.spec.name
            if name in self._entries:
                raise ValueError(f"duplicate tool {name!r}")
            schema = dict(tool.spec.parameters)
            if schema.get("type") != "object":
                raise ValueError(f"tool {name!r}: parameters must be a JSON Schema object")
            Draft202012Validator.check_schema(schema)
            direction: Any = getattr(tool, "arg_direction", "tool")
            if direction not in _DIRECTIONS:
                raise ValueError(f"tool {name!r}: bad arg_direction {direction!r}")
            self._entries[name] = _Entry(tool, Draft202012Validator(schema), direction)
        self._gate = gate
        self._ops = ops
        self._bus = bus
        self._clock = clock
        self._approval = approval
        self._allowed: dict[str, frozenset[str]] = {}
        for character, names in enabled_by_character.items():
            unknown = sorted(set(names) - set(self._entries))
            if unknown:
                log.warning("character %s lists unknown tools %s (ignored)", character, unknown)
            self._allowed[character] = frozenset(names) & frozenset(self._entries)
        self._specs: dict[str, tuple[ToolSpec, ...]] = {}
        self._mode: ToolMode = "live"
        self.set_mode(mode)
        self.approval_timeout_s = approval_timeout_s
        self.timeout_max_s = timeout_max_s
        self.filter_timeout_s = filter_timeout_s
        self.audit_timeout_s = 2.0
        self._tasks = tasks
        self._calls: dict[tuple[str, str], deque[float]] = {}

    @classmethod
    def from_config(
        cls,
        tools: Sequence[Tool],
        tools_cfg: Any,
        characters: Sequence[Any],
        *,
        gate: SafetyGate,
        ops: ToolAuditSink | None,
        bus: EventBus,
        clock: Clock,
        approval: ApprovalCallback | None = None,
        tasks: TaskSupervisor | None = None,
    ) -> PolicyToolRegistry:
        """Build from ``AppConfig.tools`` and the ``CharacterConfig``s (``[tools] enabled``)."""
        return cls(
            tools,
            gate=gate,
            ops=ops,
            bus=bus,
            clock=clock,
            enabled_by_character={c.id: list(c.tools.enabled) for c in characters},
            approval=approval,
            mode=tools_cfg.mode,
            approval_timeout_s=float(tools_cfg.approval_timeout_s),
            timeout_max_s=float(tools_cfg.timeout_max_s),
            tasks=tasks,
        )

    # --- ToolRegistry ---------------------------------------------------------------------------
    def specs(self, character: str) -> tuple[ToolSpec, ...]:
        cached = self._specs.get(character)
        if cached is None:
            allowed = self._allowed.get(character, frozenset())
            cached = tuple(e.tool.spec for n, e in self._entries.items() if n in allowed)
            self._specs[character] = cached
        return cached

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._entries:
            raise KeyError(name)
        self._entries[name].enabled = enabled

    def set_mode(self, mode: ToolMode) -> None:
        if mode not in ("live", "dry_run", "off"):
            raise ValueError(f"unknown tools mode {mode!r}")
        self._mode = mode

    # --- extras for the panel -------------------------------------------------------------------
    @property
    def mode(self) -> ToolMode:
        return self._mode

    def is_enabled(self, name: str) -> bool:
        return self._entries[name].enabled

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "mode": self._mode,
            "tools": {n: {"enabled": e.enabled} for n, e in self._entries.items()},
        }

    # --- execute --------------------------------------------------------------------------------
    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        self._bus.publish(
            ToolRequested(
                character=ctx.character,
                turn_id=ctx.turn_id,
                tool=call.name,
                args=dict(call.arguments) if isinstance(call.arguments, Mapping) else {},
            )
        )
        audit = _Audit(call.name)
        try:
            result = await self._pipeline(call, ctx, audit)
        except asyncio.CancelledError:
            audit.verdict = "cancelled"
            self._audit_in_background(ctx, audit)
            raise
        except Exception:  # a bug in a step: still never raise into the brain
            log.exception("tool pipeline failed for %s", call.name)
            result = tool_error("failed: internal error")
            audit.verdict, audit.result = "failed", result.content
        await self._write_audit(ctx, audit)
        return result

    async def _pipeline(self, call: ToolCall, ctx: ToolContext, audit: _Audit) -> ToolResult:
        name = call.name
        entry = self._entries.get(name)
        if entry is None or name not in self._allowed.get(ctx.character, frozenset()):
            return self._refuse(ctx, audit, "unknown", f"unknown tool {name!r}")
        raw_args, reason = parse_tool_arguments(call)
        if raw_args is None:
            audit.args = {"_unparsed_sha256": _digest(call.raw_arguments or "")}
            return self._refuse(ctx, audit, "invalid", reason or "invalid arguments")
        audit.args = raw_args
        errors = self._validate(entry, raw_args)
        if errors:
            return self._refuse(ctx, audit, "invalid", "invalid arguments", details=errors)
        args, filtered = await self._filter(entry, raw_args, ctx.character)
        if args is None:
            audit.args = {"_blocked_sha256": _digest(json.dumps(raw_args, ensure_ascii=False))}
            rule = filtered.category if filtered is not None else None
            return self._refuse(
                ctx, audit, "blocked", "arguments blocked by the content filter", category=rule
            )
        if args is not raw_args:
            audit.args = args
            errors = self._validate(entry, args)
            if errors:
                return self._refuse(ctx, audit, "invalid", "invalid arguments", details=errors)
        policy = entry.tool.policy
        if not entry.enabled:
            return self._refuse(ctx, audit, "unavailable", UNAVAILABLE)
        if self._mode == "off":
            return self._refuse(ctx, audit, "off", UNAVAILABLE, detail="tools are switched off")
        caps: set[str] = set()
        for channel in ctx.channels.values():
            caps |= set(channel.capabilities)
        missing = sorted(policy.requires - caps)
        if missing:
            return self._refuse(ctx, audit, "unavailable", UNAVAILABLE, needs=missing)
        limit = policy.rate_limit
        window: deque[float] | None = None
        if limit is not None:
            now = self._clock.now()
            window = self._window(ctx.character, name, limit.per_s, now)
            if len(window) >= limit.max_calls:
                retry = round(max(0.0, window[0] + limit.per_s - now))
                return self._refuse(ctx, audit, "rate_limited", "rate limited", retry_in_s=retry)
        if self._mode == "dry_run":
            audit.verdict, audit.dry_run = "dry_run", True
            result = ToolResult(
                True,
                json.dumps({"ok": True, "dry_run": True, "tool": name}, ensure_ascii=False),
                note="dry run: not executed",
            )
            audit.result = result.content
            self._executed(ctx, name, result, dry_run=True)
            return result
        if policy.requires_approval:
            approved, why = await self._approve(ctx.character, replace(call, arguments=args))
            if not approved:
                return self._refuse(ctx, audit, "denied", "not approved", detail=why)
            audit.approved_by = "operator"
        if window is not None:
            window.append(self._clock.now())
        return await self._run(entry, args, ctx, audit)

    async def _run(
        self, entry: _Entry, args: dict[str, Any], ctx: ToolContext, audit: _Audit
    ) -> ToolResult:
        name = entry.tool.spec.name
        timeout = min(entry.tool.policy.timeout_s, self.timeout_max_s)
        try:
            async with deadline(timeout, what=f"tool {name}", clock=self._clock):
                result = await entry.tool(args, ctx)
        except DeadlineExceeded:
            result = tool_error(f"timed out after {timeout:g} s")
            audit.verdict = "timeout"
        except Exception as exc:
            log.warning("tool %s failed", name, exc_info=True)
            result = tool_error(f"failed: {type(exc).__name__}")
            result = replace(result, note=f"{type(exc).__name__}: {exc}"[:500])
            audit.verdict = "failed"
        else:
            if not isinstance(result, ToolResult):
                result = tool_error("failed: the tool returned no result")
                audit.verdict = "failed"
            else:
                audit.verdict = "ok" if result.ok else "error"
        audit.result = result.content
        self._executed(ctx, name, result, dry_run=False)
        return result

    # --- steps ----------------------------------------------------------------------------------
    @staticmethod
    def _validate(entry: _Entry, args: Mapping[str, Any]) -> list[str]:
        errors = sorted(
            entry.validator.iter_errors(args), key=lambda e: [str(p) for p in e.absolute_path]
        )
        return [_describe(e) for e in errors[:5]]

    async def _filter(
        self, entry: _Entry, args: dict[str, Any], character: str
    ) -> tuple[dict[str, Any] | None, FilterResult | None]:
        """``(args, verdict)``: ``args`` possibly rewritten, or ``None`` when refused."""
        texts = free_text_values(args)
        if not texts:
            return args, None
        res = await self._check(entry.direction, [t for _, t in texts], character)
        if res is None or res.verdict in _REFUSING:
            return None, res
        if res.verdict not in _REWRITING:
            return args, res
        rewritten = copy.deepcopy(args)
        for path, text in texts:  # rare: find which values to mask, one by one
            one = await self._check(entry.direction, [text], character)
            if one is None or one.verdict in _REFUSING:
                return None, one
            if one.verdict in _REWRITING:
                _set_path(rewritten, path, one.text)
        return rewritten, res

    async def _check(
        self, direction: ArgDirection, texts: list[str], character: str
    ) -> FilterResult | None:
        """The gate's verdict, or ``None`` when the check itself failed (fail closed)."""
        try:
            async with deadline(self.filter_timeout_s, what="check_args", clock=self._clock):
                return await self._gate.check_args(direction, texts, character=character)
        except Exception:
            log.warning("argument check failed; refusing the call", exc_info=True)
            return None

    async def _approve(self, character: str, call: ToolCall) -> tuple[bool, str]:
        if self._approval is None:
            return False, "no approval queue"
        try:
            async with deadline(self.approval_timeout_s, what="tool approval", clock=self._clock):
                ok = await self._approval(character, call)
        except DeadlineExceeded:
            return False, f"no answer within {self.approval_timeout_s:g} s"
        except Exception:
            log.warning("approval callback failed; denying", exc_info=True)
            return False, "approval failed"
        return (True, "") if ok is True else (False, "denied by the operator")

    def _window(self, character: str, name: str, horizon: float, now: float) -> deque[float]:
        """Recent executions of ``name`` for ``character`` within ``horizon`` seconds."""
        window = self._calls.setdefault((character, name), deque())
        while window and now - window[0] >= horizon:
            window.popleft()
        return window

    # --- results, events, audit -----------------------------------------------------------------
    def _refuse(
        self, ctx: ToolContext, audit: _Audit, verdict: str, message: str, **extra: Any
    ) -> ToolResult:
        extra = {k: v for k, v in extra.items() if v is not None}
        result = tool_error(message, **extra)
        audit.verdict, audit.result = verdict, result.content
        self._bus.publish(
            ToolRejected(
                character=ctx.character,
                turn_id=ctx.turn_id,
                tool=audit.tool,
                reason=f"{verdict}: {message}",
            )
        )
        return result

    def _executed(self, ctx: ToolContext, name: str, result: ToolResult, *, dry_run: bool) -> None:
        self._bus.publish(
            ToolExecuted(
                character=ctx.character,
                turn_id=ctx.turn_id,
                tool=name,
                ok=result.ok,
                content=result.content,
                dry_run=dry_run,
            )
        )

    def _audit_row(self, ctx: ToolContext, audit: _Audit) -> dict[str, Any]:
        args = json.dumps(audit.args, ensure_ascii=False, default=str, sort_keys=True)
        return {
            "character": ctx.character,
            "turn_id": ctx.turn_id,
            "tool": audit.tool,
            "args": args[:_AUDIT_ARGS_MAX],
            "verdict": audit.verdict,
            "result": (audit.result or "")[:_AUDIT_RESULT_MAX],
            "approved_by": audit.approved_by,
            "dry_run": audit.dry_run,
        }

    async def _write_audit(self, ctx: ToolContext, audit: _Audit) -> None:
        if self._ops is None:
            return
        try:
            async with deadline(self.audit_timeout_s, what="tool audit", clock=self._clock):
                await self._ops.log_tool(**self._audit_row(ctx, audit))
        except Exception:
            log.exception("tool audit write failed (%s %s)", audit.tool, audit.verdict)

    def _audit_in_background(self, ctx: ToolContext, audit: _Audit) -> None:
        if self._ops is None:
            return
        if self._tasks is None:
            log.info("tool %s cancelled (not audited: no task supervisor)", audit.tool)
            return
        self._tasks.track(self._write_audit(ctx, audit), name=f"tool-audit:{audit.tool}")


@dataclass(slots=True)
class _Audit:
    tool: str
    args: Any = None
    verdict: str = "pending"
    result: str | None = None
    approved_by: str | None = None
    dry_run: bool = False
