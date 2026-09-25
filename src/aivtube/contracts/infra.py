"""Infrastructure protocols: Clock, Component, TaskSupervisor, EventBus (ARCHITECTURE.md §3.3).

Protocol definitions only; implementations live in ``aivtube.infra`` and fakes in
``aivtube.testing.fakes``. All protocols are ``runtime_checkable`` so fakes and contract
suites can use ``isinstance`` (which checks member presence, not signatures).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.events import Event
from aivtube.contracts.types import Health

if TYPE_CHECKING:
    import asyncio

__all__ = [
    "Clock",
    "Component",
    "EventBus",
    "Overflow",
    "RestartPolicy",
    "Subscription",
    "TaskSupervisor",
]


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds from ``time.perf_counter()``; comparable across processes."""
        ...

    def wall(self) -> float:
        """Wall-clock seconds (``time.time()``); only for storage, never for pacing."""
        ...

    async def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class Component(Protocol):
    name: str

    async def start(self) -> None: ...

    async def aclose(self) -> None: ...

    def health(self) -> Health: ...


RestartPolicy: TypeAlias = Literal["never", "on_error", "always"]


@runtime_checkable
class TaskSupervisor(Protocol):
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
        """Run ``factory()`` as a supervised long-running task.

        Restarts follow ``restart`` with exponential ``backoff`` (min, max seconds). More than
        ``breaker[0]`` restarts within ``breaker[1]`` seconds marks the task FAILED. A failing
        ``critical`` task ends the process (exit code 70).
        """
        ...

    def track(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
        """Start a fire-and-forget task and hold a STRONG reference until it finishes.

        asyncio keeps only weak references to tasks, so an unreferenced task can be garbage
        collected mid-flight. The supervisor also logs any exception the task raises.
        This is the only allowed way to fire and forget.
        """
        ...

    async def restart(self, name: str) -> None: ...

    def status(self) -> list[Health]: ...

    async def aclose(self, timeout: float = 5.0) -> None: ...  # noqa: ASYNC109 - frozen §3.3


class Overflow(StrEnum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


@runtime_checkable
class Subscription(Protocol):
    name: str
    dropped: int  # events lost to the overflow policy

    def __aiter__(self) -> AsyncIterator[Event]: ...

    def close(self) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    def publish(self, event: Event) -> None:
        """Publish from the loop thread. Never blocks and never raises."""
        ...

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any other thread (via ``call_soon_threadsafe``)."""
        ...

    def subscribe(
        self,
        *types: type[Event],
        name: str,
        maxsize: int = 1024,
        overflow: Overflow = Overflow.DROP_OLDEST,
    ) -> Subscription:
        """Deliver events that are instances of any of ``types``; no types means all events."""
        ...
