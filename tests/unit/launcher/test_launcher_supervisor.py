"""``Supervisor`` with real dummy child processes (§2.8, §2.9, §2.11)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import child_argv, lines, wait_until

from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.supervisor import ProcessSpec, Supervisor, backoff_delay

FAST = (0.05, 0.2)


def spec(name: str, *args: str, tmp: Path, **kw: Any) -> ProcessSpec:
    return ProcessSpec(name, child_argv(*args), {}, tmp, **{"backoff": FAST, **kw})


@pytest.fixture
def events() -> list[tuple[str, str, dict[str, Any]]]:
    return []


def make(specs: list[ProcessSpec], events: list[Any]) -> Supervisor:
    return Supervisor(
        specs, JobObject(), on_event=lambda n, k, i: events.append((n, k, dict(i))), poll_s=0.02
    )


def test_backoff_delay_sequence() -> None:
    seq = [backoff_delay(n, (0.5, 30.0)) for n in range(1, 9)]
    assert seq == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_spec_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProcessSpec("x", [], {}, tmp_path)
    with pytest.raises(ValueError):
        ProcessSpec("x", ["a"], {}, tmp_path, restart="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProcessSpec("x", ["a"], {}, tmp_path, backoff=(0.0, 1.0))
    with pytest.raises(ValueError):
        Supervisor([spec("a", tmp=tmp_path), spec("a", tmp=tmp_path)], None)


def test_crash_restarts_with_backoff(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("voice", "--exit", "1", "--after", "0.1", "--starts", str(starts),
                     tmp=tmp_path, backoff=(0.05, 0.4), breaker=(50, 120.0))], events)
    sup.start()
    try:
        assert wait_until(lambda: len(lines(starts)) >= 4, timeout=15)
    finally:
        sup.stop(1.0)
    delays = [i["delay_s"] for n, k, i in events if k == "restarting"]
    assert delays[:3] == [0.05, 0.1, 0.2]
    assert all(d <= 0.4 for d in delays)
    assert sup.status()["voice"]["restarts"] >= 3
    assert sup.status()["voice"]["state"] == "stopped"


def test_crash_loop_marks_failed_then_retry(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("core", "--exit", "1", "--starts", str(starts), tmp=tmp_path,
                     backoff=(0.02, 0.05), breaker=(5, 120.0))], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("core") == "failed", timeout=20)
        # 1 initial start + 5 restarts, then FAILED (no 6th restart)
        assert len(lines(starts)) == 6
        time.sleep(0.3)
        assert len(lines(starts)) == 6
        st = sup.status()["core"]
        assert st["state"] == "failed" and "crash loop" in st["detail"]
        assert any(k == "failed" for _, k, _ in events)
        # R on the console: retry clears the breaker and starts it again
        assert sup.retry_failed() == ["core"]
        assert wait_until(lambda: len(lines(starts)) >= 7, timeout=10)
    finally:
        sup.stop(1.0)


def test_config_error_exit_is_not_restarted(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("core", "--exit", "2", "--starts", str(starts), tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("core") == "failed", timeout=10)
        time.sleep(0.3)
        assert len(lines(starts)) == 1
        assert "configuration" in sup.status()["core"]["detail"]
    finally:
        sup.stop(1.0)


def test_clean_exit_is_not_restarted_on_error_policy(tmp_path: Path, events: list[Any]) -> None:
    sup = make([spec("x", "--exit", "0", tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("x") == "exited", timeout=10)
        assert sup.status()["x"]["exit_code"] == 0
    finally:
        sup.stop(1.0)


def test_always_policy_restarts_clean_exits(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("x", "--exit", "0", "--starts", str(starts), tmp=tmp_path,
                     restart="always")], events)
    sup.start()
    try:
        assert wait_until(lambda: len(lines(starts)) >= 2, timeout=10)
    finally:
        sup.stop(1.0)


def test_graceful_stop(tmp_path: Path, events: list[Any]) -> None:
    sup = make([spec("core", tmp=tmp_path), spec("voice", tmp=tmp_path)], events)
    sup.start()
    assert wait_until(lambda: all(s["state"] == "running" for s in sup.status().values()))
    t0 = time.perf_counter()
    sup.stop(5.0)
    assert time.perf_counter() - t0 < 4.0  # SIGTERM / CTRL_BREAK is honoured quickly
    st = sup.status()
    assert {s["state"] for s in st.values()} == {"stopped"}
    stopped = [n for n, k, _ in events if k == "stopped"]
    assert stopped == ["core", "voice"]  # spec order: core first


def test_stop_kills_after_the_graceful_timeout(tmp_path: Path, events: list[Any]) -> None:
    sup = make([spec("stubborn", "--ignore-term", tmp=tmp_path)], events)
    sup.start()
    assert wait_until(lambda: sup.state("stubborn") == "running")
    time.sleep(0.5)  # let the child install its handlers
    t0 = time.perf_counter()
    sup.stop(0.5)
    took = time.perf_counter() - t0
    # Without a console (some Windows CI runners) CTRL_BREAK cannot be delivered at all, and
    # the child is killed at once instead of after the timeout.
    assert (0.4 if sys.platform != "win32" else 0.0) <= took < 4.0
    assert sup.state("stubborn") == "stopped"


def test_hardkill_holds_until_rearm(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("voice", "--starts", str(starts), tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: len(lines(starts)) == 1)
        sup.terminate("voice", hold=True)
        st = sup.status()["voice"]
        assert st["state"] == "held" and st["held"] and st["pid"] is None
        time.sleep(0.5)
        assert sup.state("voice") == "held"
        assert len(lines(starts)) == 1  # not restarted while held
        assert sup.restart("voice") is False  # restart refuses a held child
        assert sup.rearm("voice") is True
        assert wait_until(lambda: len(lines(starts)) == 2)
        assert sup.state("voice") == "running"
        assert sup.rearm("voice") is False  # not held any more
        kinds = [k for n, k, _ in events if n == "voice"]
        assert "held" in kinds and "rearmed" in kinds
    finally:
        sup.stop(1.0)


@pytest.mark.timing
def test_terminate_is_fast(tmp_path: Path, events: list[Any]) -> None:
    sup = make([spec("voice", tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("voice") == "running")
        assert sup.terminate("voice", hold=True) < 0.3
    finally:
        sup.stop(1.0)


def test_kill_without_hold_restarts(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("core", "--starts", str(starts), tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: len(lines(starts)) == 1)
        sup.kill("core", "wedged")
        assert wait_until(lambda: len(lines(starts)) == 2)
        assert sup.status()["core"]["restarts"] == 1
    finally:
        sup.stop(1.0)


def test_operator_restart(tmp_path: Path, events: list[Any]) -> None:
    starts = tmp_path / "starts"
    sup = make([spec("core", "--starts", str(starts), tmp=tmp_path)], events)
    sup.start()
    try:
        assert wait_until(lambda: len(lines(starts)) == 1)
        pid1 = sup.status()["core"]["pid"]
        assert sup.restart("core") is True
        pid2 = sup.status()["core"]["pid"]
        assert pid2 and pid2 != pid1
        assert len(lines(starts)) >= 1
        with pytest.raises(KeyError):
            sup.restart("nope")
    finally:
        sup.stop(1.0)


def test_unstartable_command_goes_to_backoff_then_failed(tmp_path: Path, events: list[Any]) -> None:
    bad = ProcessSpec("ghost", [str(tmp_path / "missing.exe")], {}, tmp_path,
                      backoff=(0.02, 0.05), breaker=(2, 60.0))
    sup = make([bad], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("ghost") == "failed", timeout=10)
        assert "cannot start" in sup.status()["ghost"]["detail"]
    finally:
        sup.stop(1.0)


def test_log_path_captures_output(tmp_path: Path, events: list[Any]) -> None:
    log = tmp_path / "logs" / "child.log"
    s = ProcessSpec("x", child_argv("--bogus-flag"), {}, tmp_path, log_path=log, restart="never")
    sup = make([s], events)
    sup.start()
    try:
        assert wait_until(lambda: sup.state("x") == "failed", timeout=10)  # argparse exits 2
        assert "unrecognized arguments" in log.read_text(encoding="utf-8")
    finally:
        sup.stop(1.0)
