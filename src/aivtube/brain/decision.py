"""``DecisionSlot``: the single path to an LLM decision (ARCHITECTURE.md §4.1).

One slot per character. Live (and, in M2, speculative and pipelined) decisions all run through
:meth:`DecisionSlot.run`, which holds a lock so at most one decision is in flight. The decision
runs as a tracked child task and is awaited with ``asyncio.wait``, which does not swallow
cancellation of the caller: if the Brain task is cancelled, the child is cancelled and the
``CancelledError`` propagates. A child cancelled through :meth:`DecisionSlot.cancel` returns an
``aborted`` outcome instead of raising.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from aivtube.contracts.infra import TaskSupervisor

__all__ = ["DecisionOutcome", "DecisionResult", "DecisionSlot"]

OutcomeStatus = Literal["ok", "aborted", "failed"]


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """What a finished decision produced (the details live in the brain's turn record)."""

    turn_id: str
    utt_id: str | None = None
    emitted_text: str = ""
    spoke: bool = False
    filtered: bool = False
    provider: str | None = None
    tool_rounds: int = 0


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    turn_id: str
    status: OutcomeStatus
    reason: str | None
    result: DecisionResult | None
    error: BaseException | None = field(default=None, compare=False)

    @classmethod
    def ok(cls, turn_id: str, result: DecisionResult) -> DecisionOutcome:
        return cls(turn_id, "ok", None, result)

    @classmethod
    def aborted(cls, turn_id: str, reason: str) -> DecisionOutcome:
        return cls(turn_id, "aborted", reason, None)

    @classmethod
    def failed(cls, turn_id: str, exc: BaseException) -> DecisionOutcome:
        text = str(exc)
        reason = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
        return cls(turn_id, "failed", reason, None, exc)


class DecisionSlot:
    """Runs one decision at a time; see the module docstring."""

    def __init__(self, tasks: TaskSupervisor) -> None:
        self._tasks = tasks
        self._lock = asyncio.Lock()
        self._current: tuple[str, asyncio.Task[Any]] | None = None
        self._reasons: dict[str, str] = {}

    @property
    def busy(self) -> bool:
        """A decision is in flight."""
        return self._current is not None

    @property
    def current_turn(self) -> str | None:
        return self._current[0] if self._current is not None else None

    async def run(self, turn_id: str, coro: Coroutine[Any, Any, DecisionResult]) -> DecisionOutcome:
        try:
            await self._lock.acquire()
        except BaseException:
            coro.close()  # never started: do not leak an un-awaited coroutine
            raise
        try:
            task = self._tasks.track(coro, name=f"decide:{turn_id}")
            self._current = (turn_id, task)
            try:
                await asyncio.wait({task})  # does NOT swallow cancellation of the Brain task
            finally:
                self._current = None
                if not task.done():
                    task.cancel()  # Brain cancelled -> cancel the child, then propagate
            if task.cancelled():
                return DecisionOutcome.aborted(turn_id, self._reasons.pop(turn_id, "cancelled"))
            self._reasons.pop(turn_id, None)
            if (exc := task.exception()) is not None:
                return DecisionOutcome.failed(turn_id, exc)
            result = task.result()
            if not isinstance(result, DecisionResult):
                return DecisionOutcome.failed(
                    turn_id, TypeError(f"decision returned {type(result).__name__}")
                )
            return DecisionOutcome.ok(turn_id, result)
        finally:
            self._lock.release()

    def cancel(self, reason: str) -> bool:
        """Cancel the decision in flight (if any); ``reason`` becomes the abort reason."""
        current = self._current
        if current is None:
            return False
        turn_id, task = current
        if task.done():
            return False
        self._reasons.setdefault(turn_id, reason)
        task.cancel()
        return True
