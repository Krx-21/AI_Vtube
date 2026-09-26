"""``LlamaServerController`` with a dummy llama-server process (§2.3, §2.8, §4.11)."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import DUMMY_LLAMA, free_port, lines, wait_until

from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.llama import LlamaServerController, TuningStore, props_match

MODEL = "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf"


def server_cfg(port: int, **kw: Any) -> dict[str, Any]:
    return {
        "exe": "vendor/llama.cpp/llama-server.exe",
        "model": MODEL,
        "alias": "pailin-30b",
        "port": port,
        "ctx": 4096,
        "parallel": 3,
        "threads": 2,
        "placement": "fit",
        "slot_save_dir": "data/kv",
        "extra_args": [],
        **kw,
    }


def controller(tmp: Path, cfg: dict[str, Any], env: dict[str, str] | None = None,
               **kw: Any) -> LlamaServerController:
    return LlamaServerController(
        "local30b",
        cfg,
        root=tmp,
        state_dir=tmp / "data" / "state",
        log_dir=tmp / "logs",
        job=JobObject(),
        command_prefix=[sys.executable, str(DUMMY_LLAMA)],
        env=env,
        **{
            "backoff": (0.05, 0.2),
            "health_interval_s": 0.2,
            "poll_s": 0.05,
            "graceful_timeout_s": 2.0,
            **kw,
        },
    )


class _PropsServer:
    """An in-process server answering /health and /props (an already running llama-server)."""

    def __init__(self, props: dict[str, Any]) -> None:
        body = json.dumps(props).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                data = b'{"status":"ok"}' if self.path == "/health" else body
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def running_server() -> Iterator[_PropsServer]:
    srv = _PropsServer({"model_alias": "pailin-30b", "model_path": "C:\\ai\\" + Path(MODEL).name,
                        "chat_template_caps": {"supports_tool_calls": True}})
    yield srv
    srv.close()


def test_props_match_rules() -> None:
    props = {"model_alias": "pailin-30b,x", "model_path": "/m/typhoon.gguf"}
    assert props_match(props, alias="pailin-30b", model="models/typhoon.gguf")
    assert not props_match(props, alias="pailin-4b", model="models/typhoon.gguf")
    assert not props_match(props, alias="pailin-30b", model="models/other.gguf")
    assert props_match({"model_alias": "a"}, alias="a", model="whatever.gguf")


def test_adopt_accepts_matching_server(tmp_path: Path, running_server: _PropsServer) -> None:
    c = controller(tmp_path, server_cfg(running_server.port))
    try:
        assert c.adopt() is True
        assert c.status()["phase"] == "adopted"
        assert c.ensure(2.0) is True
        assert c.proc is None  # nothing spawned
        c.stop()
        assert c.status()["phase"] == "released"  # left running, not killed
        assert c.health() == "ok"
    finally:
        c.close()


@pytest.mark.parametrize(
    "override",
    [{"alias": "pailin-4b"}, {"model": "models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf"}],
)
def test_adopt_rejects_mismatch(
    tmp_path: Path, running_server: _PropsServer, override: dict[str, str]
) -> None:
    c = controller(tmp_path, server_cfg(running_server.port, **override))
    try:
        assert c.adopt() is False
        assert c.ensure(3.0) is False
        st = c.status()
        assert st["phase"] == "failed" and "taken by another server" in st["detail"]
        assert c.proc is None
    finally:
        c.close()


def test_start_ready_and_stop(tmp_path: Path) -> None:
    port = free_port()
    c = controller(tmp_path, server_cfg(port), env={"DUMMY_LLAMA_LOAD_S": "0.3"})
    try:
        assert c.ensure(15.0) is True
        st = c.status()
        assert st["phase"] == "ready" and st["pid"]
        assert (tmp_path / "data" / "kv").is_dir()  # --slot-save-path created first
        proc = c.proc
        assert proc is not None
        c.stop()
        assert proc.poll() is not None
        assert c.status()["phase"] == "stopped"
        logs = list((tmp_path / "logs").rglob("llama-local30b.log"))
        assert logs, "llama output is logged"
    finally:
        c.close()


def test_closed_controller_never_starts_again(tmp_path: Path) -> None:
    """A late /llm/ensure during launcher shutdown must not respawn an orphan server."""
    c = controller(tmp_path, server_cfg(free_port()), env={"DUMMY_LLAMA_LOAD_S": "0.1"})
    c.close()
    assert c.ensure(1.0) is False
    assert c.restart() is False
    assert c.want() is False
    assert c.proc is None and c.status()["phase"] in ("idle", "stopped")
    assert not (tmp_path / "logs").exists()  # nothing was spawned


def test_ensure_timeout_leaves_it_loading(tmp_path: Path) -> None:
    c = controller(tmp_path, server_cfg(free_port()), env={"DUMMY_LLAMA_LOAD_S": "1.0"})
    try:
        assert c.ensure(0.2) is False
        assert c.status()["phase"] == "loading"
        assert c.ensure(15.0) is True
    finally:
        c.close()


def test_pinned_load_failure_raises_n_and_persists(tmp_path: Path) -> None:
    argv_log = tmp_path / "argv.jsonl"
    env = {"DUMMY_LLAMA_ARGV_LOG": str(argv_log), "DUMMY_LLAMA_FAIL_UNTIL_NCMOE": "34"}
    cfg = server_cfg(free_port(), placement="pinned", pinned_n_cpu_moe=30, n_cpu_moe_step=2)
    c = controller(tmp_path, cfg, env=env, max_load_failures=3)
    try:
        assert c.ensure(20.0) is True
        runs = [json.loads(x) for x in lines(argv_log)]
        ns = [r[r.index("--n-cpu-moe") + 1] for r in runs]
        assert ns == ["30", "32", "34"]
        assert all("--fit" in r and r[r.index("--fit") + 1] == "off" for r in runs)
        saved = json.loads((tmp_path / "data" / "state" / "llama_tuning.json").read_text())
        entry = saved["servers"]["local30b"]
        assert entry["n_cpu_moe"] == 34 and entry["base"] == 30
        assert entry["model"] == Path(MODEL).name
        assert c.status()["n_cpu_moe"] == 34
    finally:
        c.close()
    # the next launch starts at the saved N ...
    again = controller(tmp_path, cfg, env=env)
    assert again.n_cpu_moe == 34
    assert "34" in again.build_argv()
    # ... unless the configured N changed (bench llm --tune wrote a new one)
    retuned = controller(tmp_path, {**cfg, "pinned_n_cpu_moe": 38}, env=env)
    assert retuned.n_cpu_moe == 38


def test_n_steps_up_to_cpu_moe(tmp_path: Path) -> None:
    cfg = server_cfg(free_port(), placement="pinned", pinned_n_cpu_moe=46, n_cpu_moe_step=2)
    c = controller(tmp_path, cfg)
    seen = []
    for _ in range(3):  # the third has nothing further to give
        c.on_load_failure()
        seen.append(c.n_cpu_moe)
        if len(seen) == 2:
            assert "--cpu-moe" in c.build_argv() and "--n-cpu-moe" not in c.build_argv()
    assert seen == [48, "all", "all"]
    assert TuningStore(tmp_path / "data" / "state" / "llama_tuning.json").get("local30b")


def test_fit_placement_load_failure_does_not_touch_n(tmp_path: Path) -> None:
    c = controller(tmp_path, server_cfg(free_port()))
    c.on_load_failure()
    assert c.n_cpu_moe is None
    assert not (tmp_path / "data" / "state" / "llama_tuning.json").exists()


def test_repeated_load_failures_mark_failed(tmp_path: Path) -> None:
    argv_log = tmp_path / "argv.jsonl"
    c = controller(tmp_path, server_cfg(free_port()),
                   env={"DUMMY_LLAMA_EXIT": "1", "DUMMY_LLAMA_ARGV_LOG": str(argv_log)},
                   max_load_failures=3)
    try:
        assert c.ensure(20.0) is False
        st = c.status()
        assert st["phase"] == "failed" and st["load_failures"] == 3
        assert len(lines(argv_log)) == 3
        assert c.ensure(0.5) is False  # stays FAILED until the operator retries
        assert len(lines(argv_log)) == 3
        c.restart()
        assert wait_until(lambda: len(lines(argv_log)) > 3, timeout=10)
    finally:
        c.close()


def test_crash_after_ready_restarts(tmp_path: Path) -> None:
    c = controller(tmp_path, server_cfg(free_port()))
    try:
        assert c.ensure(15.0) is True
        first = c.proc
        assert first is not None
        first.kill()
        assert wait_until(lambda: c.status()["restarts"] == 1, timeout=10)
        assert c.ensure(15.0) is True
        assert c.proc is not None and c.proc.pid != first.pid
    finally:
        c.close()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP simulates a hang on POSIX")
def test_hung_server_is_restarted(tmp_path: Path) -> None:
    c = controller(tmp_path, server_cfg(free_port()), http_timeout_s=0.2, health_failures=3)
    try:
        assert c.ensure(15.0) is True
        hung = c.proc
        assert hung is not None
        os.kill(hung.pid, signal.SIGSTOP)
        assert wait_until(lambda: c.status()["restarts"] == 1, timeout=15)
        assert hung.poll() is not None
        assert c.ensure(15.0) is True
    finally:
        c.close()


def test_missing_executable_fails(tmp_path: Path) -> None:
    c = LlamaServerController(
        "local4b",
        server_cfg(free_port(), exe=str(tmp_path / "nope" / "llama-server.exe")),
        root=tmp_path,
        state_dir=tmp_path / "state",
        poll_s=0.05,
    )
    try:
        assert c.ensure(5.0) is False
        assert "cannot start" in c.status()["detail"]
    finally:
        c.close()
