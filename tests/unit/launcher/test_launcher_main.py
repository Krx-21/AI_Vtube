"""The whole launcher with dummy core/voice processes (§2.6, §2.8, §2.9, §2.11)."""

from __future__ import annotations

import io
import json
import shutil
import socket
import threading
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import child_argv, free_port, lines, wait_until

from aivtube.launcher import main as M
from aivtube.launcher.settings import load_settings, load_tokens

REPO = Path(__file__).resolve().parents[3]
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    return tmp_path


class Run:
    """A launcher running on a thread. ``core`` may be a function of the ports."""

    def __init__(self, root: Path, *, core: list[str] | Callable[[dict[str, int]], list[str]],
                 voice: list[str] | None = None, extra: dict[str, Any] | None = None,
                 argv: list[str] | None = None, grace: float = 30.0, preflight: bool = False,
                 start: bool = True) -> None:
        self.ports = {"panel": free_port(), "bus": free_port(), "emergency": free_port()}
        overrides = {f"ports.{k}": v for k, v in self.ports.items()}
        overrides.update(extra or {})
        args = M.build_parser().parse_args(
            (argv or ["--fake-llm"]) + ([] if preflight else ["--no-preflight"])
        )
        overrides.update(M.cli_overrides(args))
        self.settings = load_settings(root, cli_overrides=overrides, env={})
        self.tokens = load_tokens(self.settings.state_dir, env={})
        self.out = io.StringIO()
        self.launcher = M.Launcher(
            self.settings, self.tokens, args,
            core_argv=core(self.ports) if callable(core) else core,
            voice_argv=voice, out=self.out, console=False,
            heartbeat_grace_s=grace, backoff=(0.05, 0.2), install_signals=False,
            log_console=False,
            preflight_kw={"check_config": None, "session": lambda: None,
                          "gpus": lambda: None},
        )
        self.code: int | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self.thread.start()

    def _run(self) -> None:
        self.code = self.launcher.run()

    def call(self, path: str, method: str = "POST") -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.ports['emergency']}{path}", method=method,
            headers={"Authorization": f"Bearer {self.tokens.emergency}"},
        )
        try:
            with _OPENER.open(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read() or b"{}")
            exc.close()
            return exc.code, body

    def finish(self, timeout: float = 30.0) -> int | None:
        self.launcher.request_quit(0)
        self.thread.join(timeout)
        return self.code


@pytest.fixture
def cleanup() -> Iterator[list[Run]]:
    runs: list[Run] = []
    yield runs
    for r in runs:
        if r.thread.is_alive():
            r.finish()


def test_runs_supervises_hardkills_and_quits(root: Path, cleanup: list[Run]) -> None:
    voice_starts = root / "voice-starts"
    core_starts = root / "core-starts"
    run = Run(root, core=child_argv("--starts", str(core_starts)),
              voice=child_argv("--starts", str(voice_starts)))
    cleanup.append(run)
    assert run.launcher.started.wait(20)
    assert wait_until(lambda: len(lines(voice_starts)) == 1 and len(lines(core_starts)) == 1)
    code, status = run.call("/status", "GET")
    assert code == 200 and status["voice"] == "up" and status["profile"] == "stream"
    code, body = run.call("/hardkill")
    assert code == 200 and body["voice"] == "down"
    code, status = run.call("/status", "GET")
    assert status["held"] is True and status["children"]["core"]["state"] == "running"
    code, _ = run.call("/rearm")
    assert code == 200
    assert wait_until(lambda: len(lines(voice_starts)) == 2)
    assert run.finish() == 0
    st = run.launcher.supervisor.status()  # type: ignore[union-attr]
    assert {v["state"] for v in st.values()} == {"stopped"}
    assert "aivtube running" in run.out.getvalue()
    logs = list((root / "logs").rglob("launcher.log"))
    assert logs and run.tokens.emergency not in logs[0].read_text(encoding="utf-8")


def test_children_get_the_tokens(root: Path, cleanup: list[Run]) -> None:
    run = Run(root, core=child_argv())
    cleanup.append(run)
    assert run.launcher.started.wait(20)
    spec = {s.name: s for s in run.launcher._specs()}
    env = spec["core"].env
    assert env["AIVTUBE_BUS_TOKEN"] == run.tokens.bus
    assert env["AIVTUBE_EMERGENCY_TOKEN"] == run.tokens.emergency
    assert env["AIVTUBE_PANEL_TOKEN"] == run.tokens.panel
    assert env["AIVTUBE_LAUNCHER_URL"] == f"http://127.0.0.1:{run.ports['emergency']}"
    assert env["AIVTUBE_DUMP_REQUEST"].endswith("dump_request.core")
    assert spec["voice"].priority == "above_normal"
    run.finish()


def test_core_exit_3_requests_a_restart(root: Path, cleanup: list[Run]) -> None:
    run = Run(root, core=child_argv("--exit", "3", "--after", "0.5"), voice=child_argv())
    cleanup.append(run)
    run.thread.join(30)
    assert run.code == M.EXIT_RESTART


def test_core_exit_0_quits(root: Path, cleanup: list[Run]) -> None:
    run = Run(root, core=child_argv("--exit", "0", "--after", "0.3"), voice=child_argv())
    cleanup.append(run)
    run.thread.join(30)
    assert run.code == M.EXIT_OK


def test_text_mode_without_speak_has_no_voice_worker(root: Path, cleanup: list[Run]) -> None:
    run = Run(root, core=child_argv(), argv=["--fake-llm", "--text"])
    cleanup.append(run)
    assert run.launcher.started.wait(20)
    assert list(run.launcher.supervisor.status()) == ["core"]  # type: ignore[union-attr]
    assert run.launcher._specs()[0].inherit_stdin is True
    code, body = run.call("/hardkill")  # nothing to kill, but it must not fail
    assert code == 200 and body["voice"] == "absent"
    code, status = run.call("/status", "GET")
    assert status["voice"] == "absent"
    run.finish()


def test_operator_restart_of_a_core_that_exits_0_does_not_quit(
    root: Path, cleanup: list[Run]
) -> None:
    starts = root / "core-starts"
    run = Run(root, core=child_argv("--term-code", "0", "--starts", str(starts)),
              voice=child_argv())
    cleanup.append(run)
    assert run.launcher.started.wait(20)
    assert wait_until(lambda: len(lines(starts)) == 1)
    code, _ = run.call("/restart/core")
    assert code == 200
    assert wait_until(lambda: len(lines(starts)) == 2)
    assert run.thread.is_alive() and run.code is None  # still running
    assert run.finish() == 0


def test_wedged_core_is_dumped_killed_and_restarted(root: Path, cleanup: list[Run]) -> None:
    starts = root / "core-starts"
    dump = root / "core.fault"

    def core(ports: dict[str, int]) -> list[str]:
        # serves /healthz on the panel port, then stops answering after 1.5 s
        return child_argv("--http", str(ports["panel"]), "--wedge-after", "1.5",
                          "--starts", str(starts), "--dump-file", str(dump))

    run = Run(root, core=core, voice=child_argv(), extra={"launcher.core_timeout_s": 1.0},
              grace=20.0)
    cleanup.append(run)
    assert run.launcher.started.wait(20)
    assert wait_until(lambda: len(lines(starts)) >= 2, timeout=30), "wedged core not restarted"
    hb = run.launcher.heartbeat
    assert hb is not None and hb.kills >= 1 and hb.armed is not None
    assert wait_until(
        lambda: dump.exists() and "stack dump requested" in dump.read_text(encoding="utf-8"),
        timeout=5,
    )
    assert run.finish() == 0


def test_preflight_failure_exits_2(root: Path, cleanup: list[Run]) -> None:
    run = Run(root, core=child_argv(), preflight=True, extra={"app.voice_worker": False},
              start=False)
    cleanup.append(run)
    with socket.socket() as sock:  # the panel port is busy
        sock.bind(("127.0.0.1", run.ports["panel"]))
        sock.listen(1)
        run.thread.start()
        run.thread.join(30)
    assert run.code == M.EXIT_PREFLIGHT
    out = run.out.getvalue()
    assert "พอร์ต" in out and "Preflight failed" in out
    assert run.launcher.supervisor is None  # nothing was started


def test_main_reports_config_errors(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "config" / "user.toml").write_text("this is = = not toml", encoding="utf-8")
    assert M.main(["--root", str(root), "--no-console"]) == M.EXIT_PREFLIGHT
    out = capsys.readouterr().out
    assert "TOML" in out and "ไฟล์" in out


def test_forwarded_args_and_overrides() -> None:
    args = M.build_parser().parse_args(["--profile", "light", "--text", "--speak", "--fake-llm"])
    assert M.forwarded_args(args) == ["--profile", "light", "--text", "--speak", "--fake-llm"]
    assert M.cli_overrides(args) == {"app.fakes": True}
    safe = M.build_parser().parse_args(["--safe"])
    assert M.cli_overrides(safe) == {"app.voice_worker": False}
