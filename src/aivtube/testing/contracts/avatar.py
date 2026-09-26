"""Contract suite for ``AvatarSink`` (§3.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from aivtube.contracts.avatar import AvatarSink
from aivtube.contracts.types import Health, HealthState
from aivtube.testing.contracts._base import AsyncCase, _Cases, await_until, check, maybe_await

__all__ = ["avatar_sink_suite"]


def avatar_sink_suite(
    factory: Callable[[], AvatarSink | Awaitable[AvatarSink]],
    *,
    hotkey: str = "happy",
    missing_hotkey: str = "no-such-hotkey-xyz",
    connect_within: float = 5.0,
) -> list[AsyncCase]:
    """``factory`` returns a sink whose renderer is reachable (e.g. ``VTSSink`` on a running
    ``FakeVTSServer``) and which knows ``hotkey``."""
    cases = _Cases("avatar_sink")

    async def running() -> tuple[AvatarSink, asyncio.Task[None]]:
        sink = await maybe_await(factory())
        task = asyncio.ensure_future(sink.run())
        await await_until(lambda: sink.connected, within=connect_within, what="connected")
        return sink, task

    async def stop(task: asyncio.Task[None]) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @cases
    async def attributes_before_run() -> None:
        sink = await maybe_await(factory())
        check(isinstance(sink.name, str) and sink.name, "name")
        check(isinstance(sink.connected, bool), "connected must be a bool")
        check(isinstance(sink.health(), Health), "health() must return Health")

    @cases
    async def run_connects_and_reports_ok() -> None:
        sink, task = await running()
        try:
            check(sink.health().state is HealthState.OK, f"health {sink.health()!r}")
        finally:
            await stop(task)

    @cases
    async def set_params_is_fire_and_forget() -> None:
        sink = await maybe_await(factory())
        sink.set_params({"MouthOpen": 0.3})  # before run(): dropped, never raises
        sink2, task = await running()
        try:
            for i in range(30):
                sink2.set_params({"MouthOpen": (i % 10) / 10.0, "MouthSmile": 0.5})
            await asyncio.sleep(0.1)
        finally:
            await stop(task)

    @cases
    async def trigger_returns_bool() -> None:
        sink, task = await running()
        try:
            check(await sink.trigger(hotkey) is True, f"trigger({hotkey!r}) failed")
            check(await sink.trigger(missing_hotkey) is False, "unknown hotkey must be False")
        finally:
            await stop(task)

    @cases
    async def emotion_and_move_do_not_raise() -> None:
        sink, task = await running()
        try:
            await sink.set_emotion("happy")
            await sink.set_emotion("happy")  # idempotent
            await sink.set_emotion("neutral", fade_s=0.1)
            await sink.move(rotation=90.0, seconds=0.1)
        finally:
            await stop(task)

    return cases.items
