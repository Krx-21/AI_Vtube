"""The core's ``LauncherServerManager`` against the real launcher endpoint and controller.

This runs the ``LocalServerManager`` contract suite end to end: core client → emergency HTTP
endpoint → ``LlamaServerController`` → a dummy llama-server process.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import DUMMY_LLAMA, free_port

from aivtube.contracts.llm import LocalServerManager
from aivtube.launcher.emergency import EmergencyServer
from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.llama import LlamaServerController
from aivtube.launcher.supervisor import Supervisor
from aivtube.llm.llamacpp import LauncherServerManager, LlamaCppAdmin
from aivtube.testing.contracts import case_id, local_server_manager_suite

TOKEN = "integration-token-0123456789"


class Factory:
    def __init__(self) -> None:
        self.make: Callable[[], Any] | None = None
        self.made: list[Any] = []

    def __call__(self) -> LocalServerManager:
        assert self.make is not None
        manager: LocalServerManager = self.make()
        self.made.append(manager)
        return manager


FACTORY = Factory()


@pytest.mark.parametrize("case", local_server_manager_suite(FACTORY), ids=case_id)
async def test_launcher_endpoint_contract(case: Callable[[], Any], tmp_path: Path) -> None:
    port = free_port()
    cfg = {
        "model": "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
        "alias": "pailin-30b",
        "port": port,
        "ctx": 4096,
        "threads": 2,
        "placement": "fit",
    }
    llama = LlamaServerController(
        "local30b",
        cfg,
        root=tmp_path,
        state_dir=tmp_path / "state",
        job=JobObject(),
        command_prefix=[sys.executable, str(DUMMY_LLAMA)],
        poll_s=0.05,
        backoff=(0.05, 0.2),
        graceful_timeout_s=2.0,
    )
    sup = Supervisor([], None)
    srv = EmergencyServer("127.0.0.1", 0, TOKEN, sup, {"local30b": llama})
    srv.start()
    FACTORY.make = lambda: LauncherServerManager(
        srv.url, TOKEN, {"local30b": LlamaCppAdmin(f"http://127.0.0.1:{port}")}
    )
    try:
        await case()
    finally:
        for m in FACTORY.made:
            await m.aclose()
        FACTORY.made.clear()
        srv.stop()
        llama.close()
