"""``AsyncEventBus``: in-process pub/sub with bounded per-subscriber queues (§3.3).

``publish`` never blocks and never raises: a slow subscriber loses events according to its
``Overflow`` policy (counted in ``Subscription.dropped``) instead of stalling the publisher.
Events whose ``ts`` is 0 are stamped with ``clock.now()`` (perf_counter).
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aivtube.contracts.events import Event
from aivtube.contracts.infra import Clock, Overflow

if TYPE_CHECKING:
    from aivtube.infra.flight import FlightRecorder

__all__ = ["AsyncEventBus", "BusSubscription"]

log = logging.getLogger("aivtube.bus")


class BusSubscription:
    """A subscriber's queue. Iterate it with ``async for``; one consumer per subscription."""

    def __init__(
        self,
        bus: AsyncEventBus,
        types: tuple[type[Event], ...],
        name: str,
        maxsize: int,
        overflow: Overflow,
    ) -> None:
        self.name = name
        self.types = types
        self.maxsize = maxsize
        self.overflow = overflow
        self.dropped = 0
        self._bus = bus
        self._queue: collections.deque[Event] = collections.deque()
        self._ready = asyncio.Event()
        self._closed = False

    def accepts(self, cls: type[Event]) -> bool:
        return not self.types or issubclass(cls, self.types)

    def _offer(self, event: Event) -> None:
        """Enqueue on the loop thread, applying the overflow policy."""
        if self._closed:
            return
        if len(self._queue) >= self.maxsize:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                log.warning(
                    "bus subscriber %r is full (%d); dropped %d event(s) so far (%s)",
                    self.name,
                    self.maxsize,
                    self.dropped,
                    self.overflow.value,
                )
            if self.overflow is Overflow.DROP_NEWEST:
                return
            self._queue.popleft()
        self._queue.append(event)
        self._ready.set()

    def __aiter__(self) -> BusSubscription:
        return self

    async def __anext__(self) -> Event:
        while not self._queue:
            if self._closed:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()
        return self._queue.popleft()

    def get_nowait(self) -> Event | None:
        return self._queue.popleft() if self._queue else None

    def drain(self) -> list[Event]:
        """Remove and return everything queued."""
        items = list(self._queue)
        self._queue.clear()
        return items

    @property
    def pending(self) -> int:
        return len(self._queue)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Unsubscribe; a waiting ``async for`` ends. Queued events are discarded."""
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        self._ready.set()
        self._bus._unsubscribe(self)

    def __repr__(self) -> str:
        return (
            f"<BusSubscription {self.name!r} pending={self.pending} dropped={self.dropped} "
            f"{self.overflow.value}>"
        )


class AsyncEventBus:
    """The core's event bus. Bound to the event loop it is first used on."""

    def __init__(
        self,
        clock: Clock,
        *,
        flight: FlightRecorder | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._clock = clock
        self._flight = flight
        self._loop = loop
        self._subs: list[BusSubscription] = []
        self._routes: dict[type[Event], tuple[BusSubscription, ...]] = {}
        self.published = 0
        self.errors = 0
        self.lost = 0  # threadsafe publishes that found no running loop

    # -- publishing ----------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Publish from the loop thread (other threads are routed through the loop)."""
        try:
            where = self._where()
            if where == "other":
                self.publish_threadsafe(event)
                return
            self._deliver(self._stamp(event))
        except Exception:
            self._error("publish", event)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread; delivery happens on the loop thread."""
        try:
            stamped = self._stamp(event)
            if self._loop is None and self._where() == "none":
                self._deliver(stamped)  # no event loop exists yet: deliver synchronously
                return
            loop = self._loop
            if loop is None or loop.is_closed():
                self.lost += 1
                return
            loop.call_soon_threadsafe(self._deliver_guarded, stamped)
        except RuntimeError:  # loop closed between the check and the call
            self.lost += 1
        except Exception:
            self._error("publish_threadsafe", event)

    def _where(self) -> str:
        """'loop' on the bus loop, 'other' on another thread, 'none' if no loop exists yet."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return "none" if self._loop is None else "other"
        if self._loop is None:
            self._loop = running
        return "loop" if running is self._loop else "other"

    def _stamp(self, event: Event) -> Event:
        if not isinstance(event, Event):
            raise TypeError(f"not an Event: {type(event).__name__}")
        if event.ts == 0.0:
            return dataclasses.replace(event, ts=self._clock.now())
        return event

    def _deliver_guarded(self, event: Event) -> None:
        try:
            self._deliver(event)
        except Exception:
            self._error("deliver", event)

    def _deliver(self, event: Event) -> None:
        if self._flight is not None:
            try:
                self._flight.record(event)
            except Exception:
                self._error("flight", event)
        for sub in self._route(type(event)):
            sub._offer(event)
        self.published += 1

    def _route(self, cls: type[Event]) -> tuple[BusSubscription, ...]:
        subs = self._routes.get(cls)
        if subs is None:
            subs = tuple(s for s in self._subs if s.accepts(cls))
            self._routes[cls] = subs
        return subs

    def _error(self, where: str, event: object) -> None:
        self.errors += 1
        if self.errors <= 10 or self.errors % 1000 == 0:
            log.exception(
                "bus %s failed for %s (error #%d)", where, type(event).__name__, self.errors
            )

    # -- subscribing ---------------------------------------------------------------------

    def subscribe(
        self,
        *types: type[Event],
        name: str,
        maxsize: int = 1024,
        overflow: Overflow = Overflow.DROP_OLDEST,
    ) -> BusSubscription:
        """Queue events that are instances of ``types`` (all events when none are given)."""
        for tp in types:
            if not (isinstance(tp, type) and issubclass(tp, Event)):
                raise TypeError(f"subscribe() takes Event subclasses, got {tp!r}")
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._where()
        sub = BusSubscription(self, tuple(types), name, maxsize, Overflow(overflow))
        self._subs.append(sub)
        self._routes.clear()
        return sub

    def _unsubscribe(self, sub: BusSubscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
            self._routes.clear()

    def subscribers(self) -> list[Mapping[str, Any]]:
        """Per-subscriber stats (panel / metrics)."""
        return [
            {
                "name": s.name,
                "pending": s.pending,
                "dropped": s.dropped,
                "maxsize": s.maxsize,
                "overflow": s.overflow.value,
            }
            for s in self._subs
        ]

    def close(self) -> None:
        """Close every subscription (shutdown)."""
        for sub in list(self._subs):
            sub.close()
