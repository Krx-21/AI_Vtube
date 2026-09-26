"""Small async helpers shared by the chat sources."""

from __future__ import annotations

import asyncio

from aivtube.contracts.infra import Clock

__all__ = ["sleep_unless_set"]


async def sleep_unless_set(clock: Clock, seconds: float, event: asyncio.Event) -> bool:
    """Sleep ``seconds`` on ``clock`` (fake clocks included) unless ``event`` is set first.

    Returns ``True`` when the full sleep elapsed with ``event`` still clear. Both helper tasks
    are cancelled and reaped before returning, so nothing outlives the call.
    """
    if event.is_set():
        return False
    if seconds <= 0:
        await asyncio.sleep(0)
        return not event.is_set()
    sleeper = asyncio.ensure_future(clock.sleep(seconds))
    waiter = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (sleeper, waiter):
            task.cancel()
        await asyncio.gather(sleeper, waiter, return_exceptions=True)
    return not event.is_set()
