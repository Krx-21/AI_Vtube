"""In-process EventBus, Subscription and TaskSupervisor fakes (ARCHITECTURE.md §3.3).

They depend only on the contracts, so tests can run before (or without) ``aivtube.infra``.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from aivtube.contracts.events import Event
from aivtube.contracts.infra import Clock, Overflow, RestartPolicy
from aivtube.contracts.types import Health, HealthState
from aivtube.testing.fakes.clock import RealClock

__all__ = ["FakeComponent", "FakeEventBus", "FakeSubscription", "FakeTaskSupervisor"]

log = logging.getLogger(__name__)
E = TypeVar("E", bound=Event)


class FakeSubscription:
    """A bounded per-subscriber queue with DROP_OLDEST / DROP_NEWEST overflow."""

    def __init__(
        self,
        bus: FakeEventBus,
        types: tuple[type[Event], ...],
        *,
        name: str,
        maxsize: int,
        overflow: Overflow,
    ) -> None:
        self.name = name
        self.dropped = 0
        self.types = types
        self.maxsize = maxsize
        self.overflow = overflow
        self._bus = bus
        self._q: collections.deque[Event] = collections.deque()
        self._wakeup = asyncio.Event()
        self.closed = False

    def matches(self, event: Event) -> bool:
        return not self.types or isinstance(event, self.types)

    def _offer(self, event: Event) -> None:
        if self.closed:
            return
        if len(self._q) >= self.maxsize:
            self.dropped += 1
            if self.overflow is Overflow.DROP_NEWEST:
                return
            self._q.popleft()
        self._q.append(event)
        self._wakeup.set()

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event]:
        while True:
            while self._q:
                yield self._q.popleft()
            if self.closed:
                return
            self._wakeup.clear()
            await self._wakeup.wait()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._wakeup.set()
            self._bus._remove(self)

    # --- test helpers -----------------------------------------------------------------------
    def qsize(self) -> int:
        return len(self._q)

    def get_nowait(self) -> Event | None:
        return self._q.popleft() if self._q else None

    def drain(self) -> list[Event]:
        items = list(self._q)
        self._q.clear()
        return items

    async def next(self, timeout: float = 5.0) -> Event:  # noqa: ASYNC109 - test helper
        """Wait for the next event (real-time ``timeout``)."""
        async with asyncio.timeout(timeout):
            while not self._q:
                if self.closed:
                    raise EOFError(f"subscription {self.name!r} is closed")
                self._wakeup.clear()
                await self._wakeup.wait()
        return self._q.popleft()


class FakeEventBus:
    """``EventBus`` that also keeps a full ``history`` for assertions.

    ``publish`` stamps ``ts`` with ``clock.now()`` when it is 0, never blocks and never raises.
    """

    def __init__(self, clock: Clock | None = None, *, history_limit: int = 100_000) -> None:
        self.clock: Clock = clock or RealClock()
        self.history: collections.deque[Event] = collections.deque(maxlen=history_limit)
        self._subs: list[FakeSubscription] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self.errors = 0

    # --- EventBus protocol ------------------------------------------------------------------
    def publish(self, event: Event) -> None:
        try:
            self._capture_loop()
            if event.ts == 0.0:
                event = dataclasses.replace(event, ts=self.clock.now())
            self.history.append(event)
            for sub in list(self._subs):
                if sub.matches(event):
                    sub._offer(event)
        except Exception:  # the contract: publish never raises
            self.errors += 1
            log.exception("FakeEventBus.publish failed for %r", type(event).__name__)

    def publish_threadsafe(self, event: Event) -> None:
        loop = self._loop
        if loop is None or loop.is_closed() or threading.get_ident() == self._loop_thread:
            self.publish(event)
            return
        try:
            loop.call_soon_threadsafe(self.publish, event)
        except RuntimeError:  # loop closed between the check and the call
            self.errors += 1

    def subscribe(
        self,
        *types: type[Event],
        name: str,
        maxsize: int = 1024,
        overflow: Overflow = Overflow.DROP_OLDEST,
    ) -> FakeSubscription:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._capture_loop()
        sub = FakeSubscription(self, tuple(types), name=name, maxsize=maxsize, overflow=overflow)
        self._subs.append(sub)
        return sub

    # --- test helpers -----------------------------------------------------------------------
    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Pin the loop used by ``publish_threadsafe`` (normally captured automatically)."""
        self._loop = loop
        self._loop_thread = None

    def of_type(self, cls: type[E]) -> list[E]:
        return [e for e in self.history if isinstance(e, cls)]

    def names(self) -> list[str]:
        return [type(e).__name__ for e in self.history]

    def clear(self) -> None:
        self.history.clear()

    @property
    def subscriptions(self) -> list[FakeSubscription]:
        return list(self._subs)

    async def wait_for(
        self, cls: type[E], predicate: Callable[[E], bool] | None = None, *, within: float = 5.0
    ) -> E:
        """Return the first (past or future) event of ``cls`` matching ``predicate``; waits at
        most ``within`` real seconds."""
        for e in self.history:
            if isinstance(e, cls) and (predicate is None or predicate(e)):
                return e
        sub = self.subscribe(cls, name=f"wait_for:{cls.__name__}", maxsize=100_000)
        try:
            async with asyncio.timeout(within):
                async for e in sub:
                    assert isinstance(e, cls)
                    if predicate is None or predicate(e):
                        return e
        finally:
            sub.close()
        raise EOFError("subscription closed")  # pragma: no cover - close() only happens above

    def _capture_loop(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._loop_thread = threading.get_ident()

    def _remove(self, sub: FakeSubscription) -> None:
        with contextlib.suppress(ValueError):
            self._subs.remove(sub)


@dataclasses.dataclass
class _Spawned:
    name: str
    factory: Callable[[], Awaitable[None]]
    restart: RestartPolicy
    backoff: tuple[float, float]
    breaker: tuple[int, float]
    critical: bool
    task: asyncio.Task[None] | None = None
    restarts: list[float] = dataclasses.field(default_factory=list)
    state: HealthState = HealthState.STARTING
    detail: str = ""
    since: float = 0.0


class FakeTaskSupervisor:
    """A small but honest ``TaskSupervisor``: restarts with backoff, breaker, strong refs."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock: Clock = clock or RealClock()
        self.tracked: set[asyncio.Task[Any]] = set()
        self.errors: list[tuple[str, BaseException]] = []
        self.critical_failures: list[tuple[str, BaseException]] = []
        self._spawned: dict[str, _Spawned] = {}

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
        if name in self._spawned and self._spawned[name].task is not None:
            raise ValueError(f"task {name!r} already spawned")
        spec = _Spawned(name, factory, restart, backoff, breaker, critical)
        spec.since = self.clock.now()
        self._spawned[name] = spec
        spec.task = asyncio.get_running_loop().create_task(self._run(spec), name=f"sup:{name}")

    def track(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
        task: asyncio.Task[Any] = asyncio.ensure_future(coro)
        with contextlib.suppress(AttributeError):
            task.set_name(name)
        self.tracked.add(task)
        task.add_done_callback(self._on_tracked_done)
        return task

    async def restart(self, name: str) -> None:
        spec = self._spawned[name]
        if spec.task is not None:
            spec.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await spec.task
        spec.restarts.clear()
        spec.state = HealthState.STARTING
        spec.task = asyncio.get_running_loop().create_task(self._run(spec), name=f"sup:{name}")

    def status(self) -> list[Health]:
        return [Health(s.name, s.state, s.detail, s.since) for s in self._spawned.values()]

    async def aclose(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109 - frozen §3.3
        tasks = [s.task for s in self._spawned.values() if s.task is not None]
        tasks += list(self.tracked)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        for s in self._spawned.values():
            s.state = HealthState.DOWN

    async def _run(self, spec: _Spawned) -> None:
        delay = spec.backoff[0]
        while True:
            spec.state = HealthState.OK
            spec.since = self.clock.now()
            try:
                await spec.factory()
                failed: BaseException | None = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed = exc
                self.errors.append((spec.name, exc))
                log.warning("supervised task %s failed: %r", spec.name, exc)
            if failed is not None and spec.critical:
                spec.state = HealthState.FAILED
                spec.detail = repr(failed)
                self.critical_failures.append((spec.name, failed))
                return
            if spec.restart == "never" or (spec.restart == "on_error" and failed is None):
                spec.state = HealthState.DOWN if failed is not None else HealthState.DISABLED
                return
            now = self.clock.now()
            spec.restarts = [t for t in spec.restarts if now - t <= spec.breaker[1]]
            spec.restarts.append(now)
            if len(spec.restarts) > spec.breaker[0]:
                spec.state = HealthState.FAILED
                spec.detail = "crash loop"
                return
            spec.state = HealthState.DEGRADED
            await self.clock.sleep(delay)
            delay = min(delay * 2, spec.backoff[1])

    def _on_tracked_done(self, task: asyncio.Task[Any]) -> None:
        self.tracked.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.errors.append((task.get_name(), exc))
            log.error("tracked task %s failed: %r", task.get_name(), exc)


class FakeComponent:
    """``Component`` that records start/aclose and reports a settable health state."""

    def __init__(self, name: str = "fake", *, fail_start: BaseException | None = None) -> None:
        self.name = name
        self.fail_start = fail_start
        self.started = 0
        self.closed = 0
        self.state = HealthState.STARTING
        self.detail = ""

    async def start(self) -> None:
        self.started += 1
        if self.fail_start is not None:
            self.state = HealthState.FAILED
            raise self.fail_start
        self.state = HealthState.OK

    async def aclose(self) -> None:
        self.closed += 1
        self.state = HealthState.DOWN

    def health(self) -> Health:
        return Health(self.name, self.state, self.detail)
