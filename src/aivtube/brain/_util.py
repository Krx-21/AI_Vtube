"""Small async helpers shared by the brain modules (clock-driven waits, guarded awaits).

Every wait goes through the injected ``Clock`` so tests drive time with ``FakeClock``; with the
real ``SystemClock`` the plain ``asyncio.wait(timeout=...)`` path is used (no helper task).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from typing import Any, TypeVar

from aivtube.contracts.infra import Clock
from aivtube.infra.clock import SystemClock, deadline

__all__ = ["guarded", "new_id", "wait_event", "wait_future"]

log = logging.getLogger("aivtube.brain")

T = TypeVar("T")


def new_id(prefix: str) -> str:
    """A short unique id such as ``t-3f9c1a2b``."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def wait_future(fut: asyncio.Future[Any], clock: Clock, seconds: float | None) -> bool:
    """Wait until ``fut`` is done or ``seconds`` clock seconds pass. Never cancels ``fut``.

    Returns ``True`` when ``fut`` is done. Cancellation of the caller propagates.
    """
    if fut.done():
        return True
    if seconds is None:
        await asyncio.wait({fut})
        return True
    if seconds <= 0:
        await asyncio.sleep(0)
        return fut.done()
    if isinstance(clock, SystemClock):
        await asyncio.wait({fut}, timeout=seconds)
        return fut.done()
    sleeper = asyncio.ensure_future(clock.sleep(seconds))
    try:
        await asyncio.wait({fut, sleeper}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sleeper.cancel()
    return fut.done()


async def wait_event(event: asyncio.Event, clock: Clock, seconds: float | None) -> bool:
    """Wait for ``event`` at most ``seconds`` clock seconds; ``True`` if it is set."""
    if event.is_set():
        return True
    waiter = asyncio.ensure_future(event.wait())
    try:
        return await wait_future(waiter, clock, seconds)
    finally:
        waiter.cancel()


async def guarded(aw: Awaitable[T], seconds: float, *, what: str, clock: Clock) -> T:
    """Await ``aw`` under a deadline (invariant I2); raises ``DeadlineExceeded``."""
    async with deadline(seconds, what=what, clock=clock):
        return await aw
