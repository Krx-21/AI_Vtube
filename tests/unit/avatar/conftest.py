"""Fixtures for the avatar tests: a recording sink, a FakeClock-driven ticker, FakeVTS."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from aivtube.testing.fakes import FakeAvatarSink, FakeClock, FakeVTSServer

#: Pailin's emotion map as in characters/pailin/character.toml (Mao sample expressions).
PAILIN_MAP: dict[str, dict[str, Any]] = {
    "neutral": {"expressions": [], "smile": 0.5, "brows": 0.5},
    "happy": {"expressions": ["exp_03.exp3.json"], "smile": 0.9, "brows": 0.7},
    "sad": {"expressions": ["exp_05.exp3.json"], "smile": 0.15, "brows": 0.2},
    "angry": {"expressions": [], "smile": 0.2, "brows": 0.1},
}


class RecordingSink(FakeAvatarSink):
    """A connected ``FakeAvatarSink`` that timestamps every frame with the fake clock."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__("rec")
        self.clock = clock
        self.frames: list[tuple[float, dict[str, float]]] = []
        self.connect()

    def set_params(self, values: Mapping[str, float]) -> None:
        super().set_params(values)
        self.frames.append((self.clock.now(), dict(values)))


class ClockTicker:
    """``TickerLike`` driven by a FakeClock: ``callback(now)`` every ``1/hz`` fake seconds."""

    def __init__(
        self,
        hz: float,
        callback: Callable[[float], None],
        loop: asyncio.AbstractEventLoop,
        *,
        clock: FakeClock,
        p95_ms: float,
        name: str = "ticker",
    ) -> None:
        self.hz = hz
        self.callback = callback
        self.loop = loop
        self.clock = clock
        self.p95_ms = p95_ms
        self.name = name
        self.ticks = 0
        self.hz_changes: list[float] = []
        self.stopped = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = self.loop.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await self.clock.sleep(1.0 / self.hz)
            self.ticks += 1
            self.callback(self.clock.now())

    def stop(self, timeout: float = 1.0) -> None:
        self.stopped = True
        if self._task is not None:
            self._task.cancel()

    def set_hz(self, hz: float) -> None:
        self.hz = hz
        self.hz_changes.append(hz)

    def jitter_stats(self) -> dict[str, float]:
        return {"hz": self.hz, "ticks": float(self.ticks), "p95_ms": self.p95_ms}


class ClockTickerFactory:
    """``ticker_factory`` for ``LiveAvatarDriver``; keeps every ticker it made."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.p95_ms = 0.5
        self.tickers: list[ClockTicker] = []

    def __call__(
        self,
        hz: float,
        callback: Callable[[float], None],
        loop: asyncio.AbstractEventLoop,
        *,
        name: str = "ticker",
    ) -> ClockTicker:
        ticker = ClockTicker(hz, callback, loop, clock=self.clock, p95_ms=self.p95_ms, name=name)
        self.tickers.append(ticker)
        return ticker


@pytest.fixture
def rec_sink(fake_clock: FakeClock) -> RecordingSink:
    return RecordingSink(fake_clock)


@pytest.fixture
def ticker_factory(fake_clock: FakeClock) -> ClockTickerFactory:
    return ClockTickerFactory(fake_clock)


@pytest.fixture
def pailin_map() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in PAILIN_MAP.items()}


@pytest.fixture
def token_path(tmp_path: Path) -> Path:
    return tmp_path / "tokens" / "vts_test.txt"


@pytest.fixture
async def vts_server() -> AsyncIterator[FakeVTSServer]:
    server = FakeVTSServer(frame_hz=60.0)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def spawn() -> AsyncIterator[Callable[[Coroutine[Any, Any, Any]], asyncio.Task[Any]]]:
    """Run coroutines in the background; cancelled and awaited at teardown."""
    tasks: list[asyncio.Task[Any]] = []

    def _spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task

    yield _spawn
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def until(
    predicate: Callable[[], bool], timeout: float = 5.0, *, what: str = "condition"
) -> None:
    """Poll ``predicate`` in real time (real sockets)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


@pytest.fixture
def wait_until() -> Callable[..., Awaitable[None]]:
    return until


@pytest.fixture
def free_udp_port() -> Iterator[int]:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    yield port


@pytest.fixture
def free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
