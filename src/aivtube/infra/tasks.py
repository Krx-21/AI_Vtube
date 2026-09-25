"""``SupervisedTasks``: long-running supervised tasks and tracked fire-and-forget tasks.

Invariant I3: every long-running task is supervised; a crash loop marks it FAILED and never
propagates. A failing ``critical`` task calls ``on_critical_failure`` (the app exits with 70).

Backoff after the n-th consecutive failure is ``min(max, min * 2**(n-1))``; it resets once a
run has lasted ``max`` seconds. The breaker ``(n, window)`` trips on the n-th failure inside
``window`` seconds, so the default ``(6, 120)`` allows 5 restarts per 120 s (§2.8/§2.9).
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aivtube.contracts.events import Alert, ComponentRestarted, HealthChanged
from aivtube.contracts.infra import Clock, EventBus, RestartPolicy
from aivtube.contracts.types import Health, HealthState

__all__ = ["SupervisedTasks", "backoff_delay"]

log = logging.getLogger("aivtube.tasks")


def backoff_delay(failures: int, backoff: tuple[float, float]) -> float:
    """Delay before the restart that follows ``failures`` consecutive failures (>= 1)."""
    lo, hi = backoff
    return float(min(hi, lo * 2 ** max(0, failures - 1)))


@dataclass(slots=True, eq=False)
class _Entry:
    name: str
    factory: Callable[[], Awaitable[None]]
    restart: RestartPolicy
    backoff: tuple[float, float]
    breaker: tuple[int, float]
    critical: bool
    runner: asyncio.Task[None] | None = None
    state: HealthState = HealthState.STARTING
    detail: str = ""
    since: float = 0.0
    restarts: int = 0
    consecutive: int = 0
    failures: collections.deque[float] = field(default_factory=collections.deque)
    stopping: bool = False


class SupervisedTasks:
    """The core's ``TaskSupervisor``. Use it only from the event-loop thread."""

    def __init__(
        self,
        clock: Clock,
        bus: EventBus,
        *,
        on_critical_failure: Callable[[str, BaseException], None],
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._on_critical = on_critical_failure
        self._entries: dict[str, _Entry] = {}
        self._tracked: set[asyncio.Task[Any]] = set()
        self._closing = False
        self.tracked_errors = 0

    # -- supervised tasks ----------------------------------------------------------------

    def spawn(
        self,
        name: str,
        factory: Callable[[], Awaitable[None]],
        *,
        restart: RestartPolicy = "on_error",
        backoff: tuple[float, float] = (0.5, 30.0),
        breaker: tuple[int, float] = (6, 120.0),
        critical: bool = False,
    ) -> None:
        """Start ``factory()`` under supervision. Names are unique while running."""
        if self._closing:
            raise RuntimeError("supervisor is closed")
        if restart not in ("never", "on_error", "always"):
            raise ValueError(f"unknown restart policy {restart!r}")
        if not 0 < backoff[0] <= backoff[1] or breaker[0] < 1 or breaker[1] <= 0:
            raise ValueError("invalid backoff or breaker")
        old = self._entries.get(name)
        if old is not None and old.runner is not None and not old.runner.done():
            raise ValueError(f"task {name!r} is already running")
        entry = _Entry(name, factory, restart, backoff, breaker, critical)
        self._entries[name] = entry
        self._start(entry)

    def _start(self, entry: _Entry) -> None:
        entry.stopping = False
        entry.runner = asyncio.get_running_loop().create_task(
            self._supervise(entry), name=f"supervised:{entry.name}"
        )

    async def _supervise(self, entry: _Entry) -> None:
        while True:
            started = self._clock.now()
            self._set(entry, HealthState.OK, "running")
            error: BaseException | None = None
            try:
                await entry.factory()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if entry.stopping or (task is not None and task.cancelling()):
                    raise
                error = RuntimeError("cancelled from inside")  # a stray CancelledError
            except Exception as exc:
                error = exc
            if entry.stopping:
                return
            now = self._clock.now()
            if now - started >= entry.backoff[1]:
                entry.consecutive = 0

            if error is not None:
                log.error("task %r failed", entry.name, exc_info=error)
            if entry.critical:
                cause = error or RuntimeError(f"critical task {entry.name!r} exited")
                self._set(entry, HealthState.FAILED, f"critical: {_describe(cause)}")
                self._critical(entry.name, cause)
                return
            if error is None and entry.restart != "always":
                self._set(entry, HealthState.DOWN, "finished")
                return
            if error is not None and entry.restart == "never":
                self._set(entry, HealthState.FAILED, _describe(error))
                return

            limit, window = entry.breaker
            entry.failures.append(now)
            while entry.failures and now - entry.failures[0] > window:
                entry.failures.popleft()
            if len(entry.failures) >= limit:
                detail = f"crash loop: {len(entry.failures)} failures in {window:g} s"
                if error is not None:
                    detail += f" (last: {_describe(error)})"
                self._set(entry, HealthState.FAILED, detail)
                self._publish(Alert(level="error", message=f"{entry.name}: {detail}"))
                return

            entry.consecutive += 1
            delay = backoff_delay(entry.consecutive, entry.backoff)
            reason = _describe(error) if error is not None else "exited"
            self._set(entry, HealthState.DEGRADED, f"restarting in {delay:g} s ({reason})")
            await self._clock.sleep(delay)
            entry.restarts += 1
            self._publish(ComponentRestarted(name=entry.name, count=entry.restarts))

    def _critical(self, name: str, exc: BaseException) -> None:
        try:
            self._on_critical(name, exc)
        except Exception:
            log.exception("on_critical_failure raised for %r", name)

    async def restart(self, name: str) -> None:
        """Operator restart: stop the task, clear its breaker and start it again now."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        await self._stop(entry)
        entry.failures.clear()
        entry.consecutive = 0
        entry.restarts += 1
        self._publish(ComponentRestarted(name=name, count=entry.restarts))
        self._start(entry)

    def status(self) -> list[Health]:
        return [
            Health(component=e.name, state=e.state, detail=e.detail, since=e.since)
            for e in self._entries.values()
        ]

    def names(self) -> list[str]:
        return list(self._entries)

    # -- fire-and-forget -----------------------------------------------------------------

    def track(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
        """Run ``coro`` as a task, holding a strong reference until it finishes.

        asyncio keeps only weak references to tasks, so an unreferenced task can be garbage
        collected mid-flight. Exceptions (not cancellation) are logged when the task ends.
        """
        if not asyncio.iscoroutine(coro):
            coro = _await(coro)
        task: asyncio.Task[Any] = asyncio.get_running_loop().create_task(coro, name=name)
        self._tracked.add(task)
        task.add_done_callback(self._tracked_done)
        return task

    def _tracked_done(self, task: asyncio.Task[Any]) -> None:
        self._tracked.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.tracked_errors += 1
            log.error("tracked task %r failed", task.get_name(), exc_info=exc)

    @property
    def tracked(self) -> int:
        """Number of tracked tasks still running."""
        return len(self._tracked)

    # -- shutdown ------------------------------------------------------------------------

    async def aclose(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109 - frozen §3.3
        """Cancel everything and wait up to ``timeout`` seconds for it to finish."""
        self._closing = True
        tasks: list[asyncio.Task[Any]] = []
        for entry in self._entries.values():
            entry.stopping = True
            if entry.runner is not None and not entry.runner.done():
                entry.runner.cancel()
                tasks.append(entry.runner)
        for task in list(self._tracked):
            task.cancel()
            tasks.append(task)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                log.warning(
                    "%d task(s) did not stop within %g s: %s",
                    len(pending),
                    timeout,
                    ", ".join(sorted(t.get_name() for t in pending)),
                )
        for entry in self._entries.values():
            if entry.state not in (HealthState.FAILED, HealthState.DOWN):
                self._set(entry, HealthState.DOWN, "stopped")

    async def _stop(self, entry: _Entry) -> None:
        runner = entry.runner
        entry.stopping = True
        if runner is not None and not runner.done():
            runner.cancel()
            await asyncio.wait({runner})

    # -- helpers -------------------------------------------------------------------------

    def _set(self, entry: _Entry, state: HealthState, detail: str) -> None:
        if state is entry.state and detail == entry.detail:
            return
        entry.state = state
        entry.detail = detail
        entry.since = self._clock.now()
        self._publish(
            HealthChanged(
                health=Health(component=entry.name, state=state, detail=detail, since=entry.since)
            )
        )

    def _publish(self, event: Any) -> None:
        try:
            self._bus.publish(event)
        except Exception:  # the bus should never raise; be safe anyway
            log.exception("publishing %s failed", type(event).__name__)


async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _describe(exc: BaseException) -> str:
    text = str(exc)
    name = type(exc).__name__
    return f"{name}: {text}" if text else name
