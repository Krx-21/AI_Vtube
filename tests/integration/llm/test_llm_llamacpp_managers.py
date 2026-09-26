"""The ``LocalServerManager`` implementations over real localhost sockets and processes."""

from __future__ import annotations

import asyncio
import socket
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from aivtube.config.schema import LlamaServerConfig
from aivtube.contracts.llm import LocalServerManager
from aivtube.llm.llama_args import build_llama_argv
from aivtube.llm.llamacpp import (
    LauncherServerManager,
    LlamaCppAdmin,
    LlamaServerManager,
    LlamaServerSpec,
    TemplateCapsError,
)
from aivtube.testing.contracts import case_id, local_server_manager_suite
from aivtube.testing.fakes import DEFAULT_PROPS, FakeEmergencyServer, FakeLauncher, SseFixtureServer

FAKE_SERVER = Path(__file__).with_name("fake_llama_server.py")
MODEL = str(DEFAULT_PROPS["model_path"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fake_spec(port: int, *, env: dict[str, str] | None = None, tmp: Path) -> LlamaServerSpec:
    """A spec whose command is the real llama-server argv, run by the fake server script."""
    cfg = LlamaServerConfig(
        model=MODEL, alias="pailin-30b", port=port, ctx=4096, parallel=3, threads=2
    )
    argv = build_llama_argv(cfg, root=tmp)
    return LlamaServerSpec(
        name="local30b",
        base_url=f"http://127.0.0.1:{port}",
        alias="pailin-30b",
        model=MODEL,
        argv=(sys.executable, str(FAKE_SERVER), *argv[1:]),
        cwd=tmp,
        log_path=tmp / "logs" / "llama-local30b.log",
        env=env,
    )


@pytest.fixture
async def sse_server() -> AsyncIterator[SseFixtureServer]:
    srv = SseFixtureServer()
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


# --- contract suites ----------------------------------------------------------------------------


class Factory:
    """A suite factory whose target is set per test (the fixtures provide the URLs)."""

    def __init__(self) -> None:
        self.make: Callable[[], Any] | None = None
        self.made: list[Any] = []

    def __call__(self) -> LocalServerManager:
        assert self.make is not None
        manager: LocalServerManager = self.make()
        self.made.append(manager)
        return manager

    async def close(self) -> None:
        for m in self.made:
            await m.aclose()
        self.made.clear()


ADOPT, LAUNCHER, SPAWN = Factory(), Factory(), Factory()


@pytest.mark.parametrize("case", local_server_manager_suite(ADOPT), ids=case_id)
async def test_adopt_mode_contract(case: Callable[[], Any], sse_server: SseFixtureServer) -> None:
    spec = LlamaServerSpec("local30b", sse_server.root_url, "pailin-30b", MODEL)
    ADOPT.make = lambda: LlamaServerManager([spec], mode="adopt")
    try:
        await case()
    finally:
        await ADOPT.close()


@pytest.mark.parametrize("case", local_server_manager_suite(LAUNCHER), ids=case_id)
async def test_launcher_mode_contract(
    case: Callable[[], Any], sse_server: SseFixtureServer
) -> None:
    emergency = FakeEmergencyServer(FakeLauncher(["local30b"]), token="tok")
    url = await emergency.start()
    LAUNCHER.make = lambda: LauncherServerManager(
        url, "tok", {"local30b": LlamaCppAdmin(sse_server.root_url)}
    )
    try:
        await case()
    finally:
        await LAUNCHER.close()
        await emergency.stop()


@pytest.mark.parametrize("case", local_server_manager_suite(SPAWN), ids=case_id)
async def test_spawn_mode_contract(case: Callable[[], Any], tmp_path: Path) -> None:
    spec = fake_spec(free_port(), tmp=tmp_path)
    SPAWN.make = lambda: LlamaServerManager([spec], mode="spawn")
    try:
        await case()
    finally:
        await SPAWN.close()


# --- spawn mode ---------------------------------------------------------------------------------


async def test_spawn_waits_for_loading_then_stops_the_process(tmp_path: Path) -> None:
    spec = fake_spec(free_port(), env={"FAKE_LLAMA_LOAD_S": "0.6"}, tmp=tmp_path)
    m = LlamaServerManager([spec], mode="spawn")
    slot_dir = tmp_path / "data" / "kv"
    assert not slot_dir.exists()
    try:
        assert await m.ensure_running("local30b", 15.0) is True
        assert slot_dir.is_dir()  # llama-server refuses a missing --slot-save-path
        proc = m.process("local30b")
        assert proc is not None and proc.returncode is None and not m.adopted("local30b")
        assert (await m.props("local30b"))["model_alias"] == "pailin-30b"
        assert spec.log_path is not None and spec.log_path.exists()
        # a second manager (e.g. after a core restart) adopts the running server
        other = LlamaServerManager([spec], mode="spawn")
        assert await other.ensure_running("local30b", 5.0) is True and other.adopted("local30b")
        await other.aclose()  # leaves the adopted server alone
        assert proc.returncode is None
        await m.stop("local30b")
        assert proc.returncode is not None and m.process("local30b") is None
    finally:
        await m.aclose()


async def test_spawn_reports_a_process_that_exits(tmp_path: Path) -> None:
    m = LlamaServerManager([fake_spec(free_port(), env={"FAKE_LLAMA_EXIT": "3"}, tmp=tmp_path)])
    try:
        assert await m.ensure_running("local30b", 10.0) is False
        assert m.process("local30b") is None
    finally:
        await m.aclose()


async def test_spawn_timeout_leaves_a_loading_server_to_finish(tmp_path: Path) -> None:
    spec = fake_spec(free_port(), env={"FAKE_LLAMA_LOAD_S": "2.5"}, tmp=tmp_path)
    m = LlamaServerManager([spec], graceful_timeout_s=2.0)
    try:
        assert await m.ensure_running("local30b", 1.0) is False  # still loading (503)
        proc = m.process("local30b")
        assert proc is not None and proc.returncode is None
        # the next call (e.g. the router's half-open retry) picks up the same process
        assert await m.ensure_running("local30b", 10.0) is True
        assert m.process("local30b") is proc and not m.adopted("local30b")
    finally:
        await m.aclose()


async def test_spawn_kills_a_server_loading_past_the_load_timeout(tmp_path: Path) -> None:
    spec = fake_spec(free_port(), env={"FAKE_LLAMA_LOAD_S": "30"}, tmp=tmp_path)
    m = LlamaServerManager([spec], graceful_timeout_s=2.0, load_timeout_s=0.5)
    try:
        assert await m.ensure_running("local30b", 1.0) is False
        if m.process("local30b") is not None:
            # The probe before the spawn ate the budget (a refused loopback connect takes ~2 s
            # on Windows), so the server was still young. Once it is past the load timeout,
            # the next call stops it.
            await asyncio.sleep(0.6)
            assert await m.ensure_running("local30b", 1.0) is False
        assert m.process("local30b") is None
    finally:
        await m.aclose()


async def test_spawn_mode_asserts_tool_caps(tmp_path: Path) -> None:
    spec = fake_spec(free_port(), env={"FAKE_LLAMA_NO_TOOLS": "1"}, tmp=tmp_path)
    m = LlamaServerManager([spec])
    try:
        with pytest.raises(TemplateCapsError):
            await m.ensure_running("local30b", 10.0)
    finally:
        await m.aclose()


async def test_missing_executable_is_false(tmp_path: Path) -> None:
    spec = LlamaServerSpec(
        "local30b",
        f"http://127.0.0.1:{free_port()}",
        "pailin-30b",
        argv=(str(tmp_path / "no-such-llama-server"),),
    )
    m = LlamaServerManager([spec])
    try:
        assert await m.ensure_running("local30b", 2.0) is False
    finally:
        await m.aclose()


# --- launcher mode ------------------------------------------------------------------------------


async def test_launcher_is_asked_when_the_server_is_down(sse_server: SseFixtureServer) -> None:
    class Launcher(FakeLauncher):
        async def ensure_running(self, server: str, timeout_s: float) -> bool:
            sse_server.loading = False  # the launcher starts the model
            return await super().ensure_running(server, timeout_s)

    emergency = FakeEmergencyServer(Launcher(["local30b"]), token="tok")
    await emergency.start()
    sse_server.loading = True
    m = LauncherServerManager(
        emergency.root_url, "tok", {"local30b": LlamaCppAdmin(sse_server.root_url)}
    )
    try:
        assert await m.ensure_running("local30b", 5.0) is True
        assert ("POST", "/llm/ensure/local30b") in emergency.hits
        await m.stop("local30b")
        assert ("POST", "/llm/stop/local30b") in emergency.hits
        # a wrong token is refused by the launcher
        sse_server.loading = True
        bad = LauncherServerManager(
            emergency.root_url, "wrong", {"local30b": LlamaCppAdmin(sse_server.root_url)}
        )
        assert await bad.ensure_running("local30b", 1.0) is False
        await bad.aclose()
    finally:
        await m.aclose()
        await emergency.stop()


async def test_launcher_unreachable_is_false(sse_server: SseFixtureServer) -> None:
    sse_server.loading = True
    m = LauncherServerManager(
        f"http://127.0.0.1:{free_port()}", "tok", {"local30b": LlamaCppAdmin(sse_server.root_url)}
    )
    try:
        assert await m.ensure_running("local30b", 1.0) is False
        await m.stop("local30b")  # never raises
    finally:
        await m.aclose()
