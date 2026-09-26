"""The operator's tool-approval queue (ARCHITECTURE.md §4.9: ``requires_approval``).

:class:`ToolApprovalQueue` is the ``approval`` callback of ``PolicyToolRegistry``
(``(character, ToolCall) -> Awaitable[bool]``). Each call waiting for the operator becomes a
pending :class:`ApprovalRequest`: the panel lists them (``CoreControl.snapshot()["approvals"]``)
and answers with ``OpKind.APPROVE {"request": <id>, "approved": true|false}``. A new request
raises a panel ``Alert``.

The registry wraps the wait in its own deadline (``tools.approval_timeout_s``, deny after 20 s);
``timeout_s`` here is only a backstop for other callers. However the wait ends (answer,
timeout, cancellation), the request leaves the queue, and anything but an explicit approval is
a denial. Everything runs on the event loop; nothing blocks. Standard library only.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from aivtube.contracts.events import Alert
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.llm import ToolCall
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.panel._json import to_jsonable

__all__ = ["ApprovalRequest", "ToolApprovalQueue"]

log = logging.getLogger("aivtube.panel.approvals")

_MAX_VALUE_CHARS: Final = 300


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One tool call waiting for the operator. ``created`` is perf_counter seconds."""

    id: str
    character: str
    tool: str
    args: Mapping[str, Any]
    created: float


def _preview(value: Any) -> Any:
    """Arguments for display: plain JSON with long strings shortened."""
    if isinstance(value, str):
        return value if len(value) <= _MAX_VALUE_CHARS else value[:_MAX_VALUE_CHARS] + "…"
    if isinstance(value, Mapping):
        return {str(k): _preview(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_preview(v) for v in value]
    return value


class ToolApprovalQueue:
    """Pending ``requires_approval`` tool calls, answered from the panel (see module doc)."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus | None = None,
        max_pending: int = 8,
        timeout_s: float = 60.0,
    ) -> None:
        if max_pending < 1 or timeout_s <= 0:
            raise ValueError("max_pending must be >= 1 and timeout_s > 0")
        self._clock = clock
        self._bus = bus
        self._max_pending = max_pending
        self._timeout_s = timeout_s
        self._ids = itertools.count(1)
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[bool]]] = {}
        self.approved = 0
        self.denied = 0
        self.expired = 0

    async def __call__(self, character: str, call: ToolCall) -> bool:
        """Wait for the operator's answer to ``call``; ``False`` unless approved."""
        if len(self._pending) >= self._max_pending:
            log.warning("tool approval queue full; denying %s for %s", call.name, character)
            self.denied += 1
            return False
        request = ApprovalRequest(
            id=f"ap{next(self._ids)}",
            character=character,
            tool=call.name,
            args=_preview(to_jsonable(dict(call.arguments or {}))),
            created=self._clock.now(),
        )
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[request.id] = (request, future)
        self._alert(request)
        try:
            async with deadline(self._timeout_s, what="tool approval", clock=self._clock):
                return await future
        except DeadlineExceeded:
            log.info("tool approval %s (%s) expired; denied", request.id, request.tool)
            return False
        finally:
            self._pending.pop(request.id, None)
            if not future.done():
                future.cancel()
            if future.cancelled():  # cancelled or timed out: never answered
                self.expired += 1

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Answer a pending request; ``False`` when it is unknown or already answered."""
        entry = self._pending.get(request_id)
        if entry is None or entry[1].done():
            return False
        request, future = entry
        future.set_result(bool(approved))
        if approved:
            self.approved += 1
        else:
            self.denied += 1
        log.info(
            "tool %s for %s %s by the operator",
            request.tool,
            request.character,
            "approved" if approved else "denied",
        )
        return True

    def get(self, request_id: str) -> ApprovalRequest | None:
        entry = self._pending.get(request_id)
        return None if entry is None else entry[0]

    def pending(self) -> list[ApprovalRequest]:
        """Waiting requests, oldest first."""
        return [request for request, future in self._pending.values() if not future.done()]

    def snapshot(self) -> list[dict[str, Any]]:
        """Waiting requests for the panel, with their age in seconds."""
        now = self._clock.now()
        return [
            {
                "id": r.id,
                "character": r.character,
                "tool": r.tool,
                "args": r.args,
                "age_s": round(max(0.0, now - r.created), 1),
            }
            for r in self.pending()
        ]

    def _alert(self, request: ApprovalRequest) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Alert(
                level="warn",
                message=f"เครื่องมือ {request.tool} รออนุมัติจากผู้ควบคุม (tool approval)",
                character=request.character,
            )
        )
