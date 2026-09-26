"""Tool execution after the reply stream (ARCHITECTURE.md §4.9).

1. The request carries the static tools; text streams to speech while tool calls accumulate.
2. After the stream ends, :meth:`ToolFlow.run` executes every call **in call order** through
   ``ToolRegistry.execute`` (parse, validate, ``check_args``, policy, timeout, audit). Side-effect
   tools never retry; a failure becomes the result text.
3. :meth:`ToolFlow.tool_messages` renders the assistant message and one ``tool`` message per
   result, in call order (the Typhoon template has no ``tool_call_id``; the id is still sent for
   providers that want it).
4. :meth:`ToolFlow.needs_follow_up`: one follow-up completion (``/u2``) when the reply had no
   spoken content or a tool asks for one (``ToolPolicy.follow_up``); never more than
   ``max_rounds`` completions per decision.

Execution happens only after the stream ended, so it never delays speech already queued.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final

from aivtube.contracts.infra import Clock
from aivtube.contracts.llm import ToolCall
from aivtube.contracts.tools import ToolContext, ToolRegistry, ToolResult
from aivtube.infra.clock import DeadlineExceeded, deadline

__all__ = ["UNAVAILABLE_RESULT", "ToolFlow", "assistant_message_copy", "error_result"]

log = logging.getLogger("aivtube.brain.tools")

_CALL_TIMEOUT_S: Final = 30.0  # outer bound; the registry applies each tool's own timeout


def error_result(reason: str) -> ToolResult:
    """A failed call as the model sees it (``{"ok": false, "error": …}``)."""
    body = json.dumps({"ok": False, "error": reason}, ensure_ascii=False)
    return ToolResult(False, body, note=reason)


#: The result given to calls that are not executed (tools off, strict mode, paused).
UNAVAILABLE_RESULT: Final = error_result("unavailable now")


class ToolFlow:
    """Runs one decision's tool calls; see the module docstring.

    ``follow_up_tools`` names the tools whose policy has ``follow_up=True`` (the Protocol has
    no policy accessor; build it with :meth:`follow_up_names`). ``clock`` drives the outer
    per-call deadline (invariant I2).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_rounds: int = 2,
        follow_up_tools: Collection[str] = (),
        clock: Clock | None = None,
        call_timeout_s: float = _CALL_TIMEOUT_S,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.registry = registry
        self.max_rounds = max_rounds
        self.follow_up_tools = frozenset(follow_up_tools)
        self._clock = clock
        self._call_timeout_s = call_timeout_s

    @staticmethod
    def follow_up_names(tools: Sequence[Any]) -> frozenset[str]:
        """Names of the ``Tool`` objects whose ``policy.follow_up`` is set."""
        out: set[str] = set()
        for tool in tools:
            policy = getattr(tool, "policy", None)
            spec = getattr(tool, "spec", None)
            if policy is not None and spec is not None and getattr(policy, "follow_up", False):
                out.add(str(spec.name))
        return frozenset(out)

    async def run(
        self, calls: Sequence[ToolCall], ctx: ToolContext
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Execute ``calls`` one after another, in order. Never raises except cancellation."""
        results: list[tuple[ToolCall, ToolResult]] = []
        for call in calls:
            results.append((call, await self._execute(call, ctx)))
        return results

    async def _execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        try:
            async with deadline(self._call_timeout_s, what=f"tool {call.name}", clock=self._clock):
                return await self.registry.execute(call, ctx)
        except DeadlineExceeded:
            log.warning("tool %s did not finish within %.0f s", call.name, self._call_timeout_s)
            return error_result("timeout")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the registry never raises by contract; stay safe anyway
            log.exception("tool %s raised", call.name)
            return error_result(f"failed: {type(exc).__name__}")

    def needs_follow_up(
        self,
        results: Sequence[tuple[ToolCall, ToolResult]],
        spoke: bool,
        *,
        rounds: int = 1,
    ) -> bool:
        """Whether to run one more completion after ``rounds`` completions so far."""
        if not results or rounds >= self.max_rounds:
            return False
        if not spoke:
            return True
        return any(call.name in self.follow_up_tools for call, _ in results)

    def tool_messages(
        self,
        assistant_message: Mapping[str, Any],
        results: Sequence[tuple[ToolCall, ToolResult]],
        *,
        keep_extra: bool,
    ) -> list[Mapping[str, Any]]:
        """The assistant message (``extra_content`` stripped unless ``keep_extra``), then one
        ``tool`` message per result in call order. Content is always a plain string."""
        msg = assistant_message_copy(assistant_message, keep_extra=keep_extra)
        out: list[Mapping[str, Any]] = [msg]
        for call, result in results:
            out.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
        return out


def assistant_message_copy(message: Mapping[str, Any], *, keep_extra: bool) -> dict[str, Any]:
    """A deep copy of an assistant message with string content; ``extra_content`` (Gemini
    ``thought_signature``) is removed from the message and its tool calls unless kept."""
    msg: dict[str, Any] = copy.deepcopy(dict(message))
    msg["role"] = "assistant"
    content = msg.get("content")
    msg["content"] = content if isinstance(content, str) else ""
    if keep_extra:
        return msg
    msg.pop("extra_content", None)
    calls = msg.get("tool_calls")
    if isinstance(calls, list):
        cleaned: list[Any] = []
        for call in calls:
            if isinstance(call, dict):
                call = {k: v for k, v in call.items() if k != "extra_content"}
            cleaned.append(call)
        msg["tool_calls"] = cleaned
    return msg
