"""Heartbeat, console keys, GPU telemetry, settings/tokens, child helpers and platform shims."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import wait_until

from aivtube.config.errors import ConfigError
from aivtube.launcher import win32
from aivtube.launcher.childside import start_dump_request_watcher
from aivtube.launcher.console import KEY_HELP, ConsoleKeys, help_text
from aivtube.launcher.gpu import GpuMonitor, driver_major, parse_nvidia_smi, query_gpus
from aivtube.launcher.heartbeat import CoreHeartbeat, http_alive, request_dump
from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.logs import TokenRedactor
from aivtube.launcher.settings import load_settings, load_tokens

REPO = Path(__file__).resolve().parents[3]


# --- heartbeat --------------------------------------------------------------------------------


class FakeTarget:
    def __init__(self) -> None:
        self.pid = 100
        self.state = "running"
        self.killed: list[str] = []

    def info(self, name: str) -> tuple[str, int | None, float]:
        return self.state, self.pid, 0.0

    def kill(self, name: str, reason: str) -> None:
        self.killed.append(reason)
        self.pid += 1  # the supervisor restarts it


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _drive(hb: CoreHeartbeat, clock: Clock, answers: list[bool], step: float = 1.0) -> None:
    """Run the heartbeat loop body synchronously for each probe answer."""
    it = iter(answers)
    hb._probe = lambda url, timeout: next(it)
    hb.interval_s = 0.0
    calls = {"n": 0}
    real_wait = hb._halt.wait

    def fake_wait(timeout: float | None = None) -> bool:
        calls["n"] += 1
        clock.t += step
        return calls["n"] > len(answers)

    hb._halt.wait = fake_wait  # type: ignore[method-assign]
    try:
        hb.run()
    finally:
        hb._halt.wait = real_wait  # type: ignore[method-assign]


def test_heartbeat_kills_after_timeout_once_armed() -> None:
    target, clock = FakeTarget(), Clock()
    hb = CoreHeartbeat(target, "core", "http://x", timeout_s=5.0, startup_grace_s=60.0,
                       clock=clock)
    # answers for 2 s, then silence: killed once 5 s have passed without an answer
    _drive(hb, clock, [True, True] + [False] * 6)
    assert len(target.killed) == 1 and "no answer" in target.killed[0]


def test_heartbeat_gives_a_starting_core_the_grace_period() -> None:
    target, clock = FakeTarget(), Clock()
    hb = CoreHeartbeat(target, "core", "http://x", timeout_s=5.0, startup_grace_s=10.0,
                       clock=clock)
    _drive(hb, clock, [False] * 9)  # 8 s of silence since the start: still in the grace
    assert target.killed == []
    target2, clock2 = FakeTarget(), Clock()
    hb2 = CoreHeartbeat(target2, "core", "http://x", timeout_s=5.0, startup_grace_s=10.0,
                        clock=clock2)
    _drive(hb2, clock2, [False] * 12)
    assert len(target2.killed) == 1 and "never answered" in target2.killed[0]


def test_heartbeat_ignores_children_that_are_not_running() -> None:
    target, clock = FakeTarget(), Clock()
    target.state = "held"
    hb = CoreHeartbeat(target, "core", "http://x", timeout_s=1.0, startup_grace_s=1.0,
                       clock=clock)
    _drive(hb, clock, [False] * 5)
    assert target.killed == []


def test_http_alive_counts_any_answer(tmp_path: Path) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *a: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert http_alive(f"http://127.0.0.1:{httpd.server_address[1]}/healthz", 2.0)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert http_alive("http://127.0.0.1:9/healthz", 0.5) is False


def test_dump_request_round_trip(tmp_path: Path) -> None:
    req = tmp_path / "dump_request.core"
    fault = tmp_path / "core.fault"
    with fault.open("w", encoding="utf-8") as out:  # faulthandler needs a real file
        thread = start_dump_request_watcher(req, file=out, interval_s=0.02)
        assert thread is not None
        assert request_dump(req, os.getpid(), wait_s=3.0) is True
        assert not req.exists()
    text = fault.read_text(encoding="utf-8")
    assert "stack dump requested by the launcher" in text
    assert "test_dump_request_round_trip" in text  # every thread's stack, ours included
    assert start_dump_request_watcher("", file=io.StringIO()) is None


def test_unanswered_dump_request_is_cleaned_up(tmp_path: Path) -> None:
    req = tmp_path / "dump_request.voice"
    assert request_dump(req, 1, wait_s=0.1) is False
    assert not req.exists()


# --- console keys -----------------------------------------------------------------------------


def test_console_keys_from_lines() -> None:
    hits: list[str] = []
    handlers = {k: (lambda k=k: hits.append(k)) for k in "kafrsqh"}
    keys = ConsoleKeys(handlers, stdin=io.StringIO("k\n\nS\n?\nx\nboom\nq\n"))
    keys.start()
    keys.join(5)
    assert hits == ["k", "s", "h", "q"]
    assert keys.dispatch("R") is True and hits[-1] == "r"
    assert keys.dispatch("z") is False


def test_console_handler_errors_do_not_stop_the_reader() -> None:
    hits: list[str] = []

    def boom() -> None:
        raise RuntimeError("x")

    keys = ConsoleKeys({"k": boom, "q": lambda: hits.append("q")}, stdin=io.StringIO("k\nq\n"))
    keys.start()
    keys.join(5)
    assert hits == ["q"]


def test_help_text_is_bilingual() -> None:
    text = help_text()
    for key in KEY_HELP:
        assert f"[{key.upper()}]" in text
    assert "HARD KILL" in text and "ตัดเสียง" in text


# --- GPU --------------------------------------------------------------------------------------

SMI = "0, NVIDIA GeForce RTX 4070, 581.42, 12282, 8000, 4282, 37, 51\n1, Other, 581.42, [N/A], 1, 2, [N/A], 40\n"


def test_parse_nvidia_smi() -> None:
    gpus = parse_nvidia_smi(SMI)
    assert gpus[0] == {"index": 0, "name": "NVIDIA GeForce RTX 4070", "driver": "581.42",
                       "total_mib": 12282.0, "used_mib": 8000.0, "free_mib": 4282.0,
                       "util_pct": 37.0, "temp_c": 51.0}
    assert gpus[1]["total_mib"] is None
    assert driver_major("581.42") == 581 and driver_major("") is None


def test_query_gpus_with_a_runner() -> None:
    def runner(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert argv[1].startswith("--query-gpu=")
        return subprocess.CompletedProcess(argv, 0, SMI, "")

    gpus = query_gpus("nvidia-smi", runner=runner)
    assert gpus is not None and gpus[0]["free_mib"] == 4282.0

    def failing(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    assert query_gpus("nvidia-smi", runner=failing) is None
    if shutil.which("nvidia-smi") is None:
        assert query_gpus() is None


def test_gpu_monitor_alarm_is_rate_limited() -> None:
    alarms: list[dict[str, Any]] = []
    readings = iter([[{"free_mib": 400.0}], [{"free_mib": 300.0}], None])
    mon = GpuMonitor(alarm_mib=500, on_alarm=lambda g: alarms.append(dict(g)),
                     query=lambda: next(readings), alarm_every_s=60.0)
    assert mon.poll_once() is not None
    assert mon.poll_once() is not None
    assert len(alarms) == 1 and mon.alarms == 1
    assert mon.latest()["free_mib"] == 300.0
    assert mon.poll_once() is None and mon.available is False


# --- settings and tokens ----------------------------------------------------------------------


def test_settings_from_the_real_defaults() -> None:
    s = load_settings(REPO, env={})
    assert s.ports == {"panel": 8770, "bus": 8771, "emergency": 8779, "neuro_sdk": 8000}
    assert s.autostart_servers() == ["local30b"]
    assert s.llm_servers_in_chain() == ["local30b", "local4b"]  # cloud entries need consent
    assert s.restart_backoff_s == (0.5, 30.0)
    assert s.crash_loop_restarts == 5 and s.core_timeout_s == 5.0
    light = load_settings(REPO, profile="light", env={})
    assert light.autostart_servers() == ["local4b"]
    faked = load_settings(REPO, cli_overrides={"app.fakes": True}, env={})
    assert faked.autostart_servers() == []


def test_settings_type_errors_are_config_errors() -> None:
    with pytest.raises(ConfigError) as err:
        load_settings(REPO, cli_overrides={"launcher.core_timeout_s": "soon"}, env={})
    assert "ตัวเลข" in err.value.message_th
    with pytest.raises(ConfigError):
        load_settings(REPO, cli_overrides={"launcher.restart_backoff_s": [5, 1]}, env={})


def test_tokens_persist_but_the_bus_token_is_fresh(tmp_path: Path) -> None:
    a = load_tokens(tmp_path, env={})
    b = load_tokens(tmp_path, env={})
    assert a.panel == b.panel and a.emergency == b.emergency
    assert a.bus != b.bus
    assert len(a.emergency) >= 24
    saved = json.loads((tmp_path / "tokens.json").read_text())
    assert set(saved) == {"panel", "emergency"}
    env = load_tokens(tmp_path, env={"AIVTUBE_EMERGENCY_TOKEN": "from-env-123456789"})
    assert env.emergency == "from-env-123456789"


def test_token_redactor() -> None:
    red = TokenRedactor(["supersecret123"])
    text = red.redact("got supersecret123 with Bearer abc.def and ?token=xyz&x=1")
    assert "supersecret123" not in text and "abc.def" not in text and "xyz" not in text


# --- platform ---------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows no-op behaviour")
def test_platform_shims_are_noops_elsewhere() -> None:
    assert win32.keep_awake(True) is False
    assert win32.session_id() is None
    assert win32.process_creation_flags("above_normal") == 0
    job = JobObject()
    assert job.active is False and job.assign(os.getpid()) is False
    job.close()


@pytest.mark.windows
def test_windows_flags_and_session() -> None:
    flags = win32.process_creation_flags("above_normal")
    assert flags & win32.CREATE_NEW_PROCESS_GROUP and flags & win32.ABOVE_NORMAL_PRIORITY_CLASS
    sid = win32.session_id()
    assert sid is not None and sid >= 0
    assert win32.keep_awake(True) is True
    assert win32.keep_awake(False) is True


@pytest.mark.windows
def test_job_object_kills_children_on_close() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job = JobObject()
    try:
        assert job.active and job.assign(child.pid)
        job.close()
        assert child.wait(timeout=10) is not None
    finally:
        if child.poll() is None:
            child.kill()


LAUNCHER_LIKE = """
import subprocess, sys, time
sys.path.insert(0, {src!r})
from aivtube.launcher.jobobject import JobObject
job = JobObject()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
job.assign(child.pid)
print(child.pid, flush=True)
time.sleep(60)
"""


@pytest.mark.windows
def test_killing_the_launcher_kills_its_children(tmp_path: Path) -> None:
    """Acceptance: TerminateProcess on the launcher takes its children with it."""
    script = tmp_path / "launcher_like.py"
    script.write_text(LAUNCHER_LIKE.format(src=str(REPO / "src")), encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert parent.stdout is not None
    grandchild = int(parent.stdout.readline())
    parent.kill()  # TerminateProcess: no cleanup code runs in the launcher
    parent.wait(10)

    def gone() -> bool:
        done = subprocess.run(["tasklist", "/FI", f"PID eq {grandchild}"], capture_output=True,
                              text=True, check=False)
        return str(grandchild) not in done.stdout

    assert wait_until(gone, timeout=10)


def test_perf_counter_is_the_clock() -> None:
    from aivtube.launcher.proc import monotonic

    a = monotonic()
    time.sleep(0.01)
    assert monotonic() - a >= 0.009
