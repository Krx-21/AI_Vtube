"""Tool fakes: ``FakeTool`` and ``FakeToolRegistry`` (§3.11, §4.9)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

import jsonschema

from aivtube.contracts.events import ToolExecuted, ToolRejected, ToolRequested
from aivtube.contracts.llm import ToolCall, ToolSpec
from aivtube.contracts.safety import SafetyGate, Verdict
from aivtube.contracts.tools import Tool, ToolContext, ToolPolicy, ToolResult

__all__ = ["ECHO_SPEC", "UNAVAILABLE", "FakeTool", "FakeToolRegistry"]

UNAVAILABLE = "unavailable now"

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Repeat a short text (test tool).",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "maxLength": 120}},
        "required": ["text"],
        "additionalProperties": False,
    },
)


def _json(obj: Mapping[str, Any]) -> str:
    return json.dumps(dict(obj), ensure_ascii=False)


class FakeTool:
    """A ``Tool`` that records its calls and returns ``result`` (or runs ``handler``)."""

    def __init__(
        self,
        spec: ToolSpec = ECHO_SPEC,
        policy: ToolPolicy | None = None,
        *,
        result: ToolResult | None = None,
        handler: Callable[[Mapping[str, Any], ToolContext], Awaitable[ToolResult]] | None = None,
        delay_s: float = 0.0,
        raises: BaseException | None = None,
    ) -> None:
        self.spec = spec
        self.policy = policy or ToolPolicy()
        self.result = result
        self.handler = handler
        self.delay_s = delay_s
        self.raises = raises
        self.calls: list[Mapping[str, Any]] = []

    async def __call__(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raises is not None:
            raise self.raises
        if self.handler is not None:
            return await self.handler(args, ctx)
        if self.result is not None:
            return self.result
        return ToolResult(True, _json({"ok": True, "echo": dict(args)}))


class FakeToolRegistry:
    """``ToolRegistry`` with the real order of checks: mode, known, enabled, JSON-schema
    validation, argument filtering (through ``gate``), policy (capabilities, rate limit,
    approval), then the call under ``policy.timeout_s``. Failures never raise: they return
    ``ToolResult(ok=False)`` and publish ``ToolRejected`` on ``ctx.bus``."""

    def __init__(
        self,
        tools: Sequence[Tool] = (),
        *,
        gate: SafetyGate | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.spec.name in self._tools:
                raise ValueError(f"duplicate tool {t.spec.name!r}")
            self._tools[t.spec.name] = t
        self.gate = gate
        self.approve = approve
        self.enabled: dict[str, bool] = {name: True for name in self._tools}
        self.mode: Literal["live", "dry_run", "off"] = "live"
        self.executed: list[ToolCall] = []
        self.rejected: list[tuple[ToolCall, str]] = []
        self._calls_at: dict[str, list[float]] = {}

    def specs(self, character: str) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self._tools.values())

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._tools:
            raise KeyError(name)
        self.enabled[name] = enabled

    def set_mode(self, mode: Literal["live", "dry_run", "off"]) -> None:
        self.mode = mode

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        ctx.bus.publish(
            ToolRequested(
                character=ctx.character,
                turn_id=ctx.turn_id,
                tool=call.name,
                args=dict(call.arguments or {}),
            )
        )
        if self.mode == "off":
            return self._reject(call, ctx, "tools are off")
        tool = self._tools.get(call.name)
        if tool is None:
            return self._reject(call, ctx, f"unknown tool {call.name!r}")
        if not self.enabled.get(call.name, False):
            return self._reject(call, ctx, UNAVAILABLE)
        if call.arguments is None:
            return self._reject(call, ctx, "arguments are not valid JSON")
        try:
            jsonschema.validate(dict(call.arguments), dict(tool.spec.parameters))
        except jsonschema.ValidationError as exc:
            return self._reject(call, ctx, f"invalid arguments: {exc.message}")
        texts = [v for v in call.arguments.values() if isinstance(v, str)]
        if self.gate is not None and texts:
            res = await self.gate.check_args("tool", texts, character=ctx.character)
            if res.verdict in (Verdict.BLOCK, Verdict.DROP):
                return self._reject(call, ctx, "arguments blocked by the filter")
        policy = tool.policy
        caps = (
            set().union(*(ch.capabilities for ch in ctx.channels.values()))
            if ctx.channels
            else set()
        )
        missing = sorted(policy.requires - caps)
        if missing:
            return self._reject(call, ctx, f"needs {', '.join(missing)}")
        if policy.rate_limit is not None:
            now = ctx.clock.now()
            recent = [
                t for t in self._calls_at.get(call.name, []) if now - t < policy.rate_limit.per_s
            ]
            if len(recent) >= policy.rate_limit.max_calls:
                return self._reject(call, ctx, "rate limited")
            self._calls_at[call.name] = [*recent, now]
        if policy.requires_approval and (
            self.approve is None or not await self.approve(ctx.character, call)
        ):
            return self._reject(call, ctx, "not approved")
        if self.mode == "dry_run":
            self.executed.append(call)
            result = ToolResult(True, _json({"dry_run": True}), note="dry run")
            ctx.bus.publish(
                ToolExecuted(
                    character=ctx.character,
                    turn_id=ctx.turn_id,
                    tool=call.name,
                    ok=True,
                    content=result.content,
                    dry_run=True,
                )
            )
            return result
        try:
            result = await asyncio.wait_for(tool(call.arguments, ctx), policy.timeout_s)
        except TimeoutError:
            return self._reject(call, ctx, f"timed out after {policy.timeout_s:g} s")
        except Exception as exc:
            return self._reject(call, ctx, f"failed: {type(exc).__name__}")
        self.executed.append(call)
        ctx.bus.publish(
            ToolExecuted(
                character=ctx.character,
                turn_id=ctx.turn_id,
                tool=call.name,
                ok=result.ok,
                content=result.content,
            )
        )
        return result

    def _reject(self, call: ToolCall, ctx: ToolContext, reason: str) -> ToolResult:
        self.rejected.append((call, reason))
        ctx.bus.publish(
            ToolRejected(
                character=ctx.character, turn_id=ctx.turn_id, tool=call.name, reason=reason
            )
        )
        return ToolResult(False, _json({"error": reason}), note=reason)
