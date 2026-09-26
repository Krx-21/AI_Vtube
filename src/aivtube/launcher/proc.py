"""Process start/stop helpers shared by the supervisor and the llama-server controller.

- Children start in their own process group: ``CREATE_NEW_PROCESS_GROUP`` on Windows (so
  ``CTRL_BREAK_EVENT`` reaches exactly one child and a Ctrl+C in the launcher console does
  not), a new session on POSIX. Each is assigned to the Job Object right away.
- Graceful stop: ``CTRL_BREAK_EVENT`` (Windows) or ``SIGTERM`` (POSIX), then
  ``TerminateProcess``/``SIGKILL`` after the timeout (§2.9).
- Hard kill: ``TerminateProcess``/``SIGKILL`` at once (HARD KILL, §2.11).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Protocol

from aivtube.launcher.win32 import IS_WINDOWS, process_creation_flags

__all__ = [
    "JobLike",
    "child_env",
    "hard_kill",
    "monotonic",
    "request_stop",
    "spawn",
    "stop_process",
    "wait_gone",
]

log = logging.getLogger("aivtube.launcher.proc")


class JobLike(Protocol):
    def assign(self, pid: int) -> bool: ...


def child_env(extra: Mapping[str, str], *, inherit: bool = True) -> dict[str, str]:
    """The environment for a child: ours (when ``inherit``) plus ``extra``; UTF-8 stdio so
    Thai log lines survive a Windows console."""
    env = dict(os.environ) if inherit else {}
    if IS_WINDOWS and "SYSTEMROOT" not in {k.upper() for k in env}:
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update({str(k): str(v) for k, v in extra.items()})
    return env


def spawn(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str],
    job: JobLike | None = None,
    priority: str = "normal",
    log_path: Path | None = None,
    inherit_stdin: bool = False,
) -> subprocess.Popen[bytes]:
    """Start ``argv`` in its own process group and put it into ``job``.

    Output goes to ``log_path`` (appended) when given, else to the launcher's console.
    Raises ``OSError`` when the executable cannot be started.
    """
    out: IO[bytes] | None = None
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = process_creation_flags(priority)
    else:
        kwargs["start_new_session"] = True
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env),
            stdin=None if inherit_stdin else subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT if out is not None else None,
            close_fds=True,
            **kwargs,
        )
    finally:
        if out is not None:
            out.close()  # the child holds its own handle
    if job is not None:
        job.assign(proc.pid)
    if not IS_WINDOWS and priority != "normal":
        log.debug("process priority %r is only applied on Windows", priority)
    return proc


def request_stop(proc: subprocess.Popen[Any]) -> bool:
    """Ask a child to exit (``CTRL_BREAK_EVENT`` / ``SIGTERM``). ``False`` if it could not be
    asked (already gone, or no console to deliver the event on Windows)."""
    if proc.poll() is not None:
        return False
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined,unused-ignore]
        else:
            proc.send_signal(signal.SIGTERM)
    except (ProcessLookupError, OSError) as exc:
        log.debug("could not ask pid %s to stop: %s", proc.pid, exc)
        return False
    return True


def hard_kill(proc: subprocess.Popen[Any]) -> None:
    """``TerminateProcess`` / ``SIGKILL`` right away (never raises)."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


def stop_process(proc: subprocess.Popen[Any], timeout_s: float, *, what: str = "") -> int | None:
    """Graceful stop, then kill after ``timeout_s``. Returns the exit code (``None`` if the
    process could not be reaped within a further 5 s)."""
    if proc.poll() is not None:
        return proc.returncode
    name = what or f"pid {proc.pid}"
    if request_stop(proc):
        try:
            return proc.wait(timeout=max(0.0, timeout_s))
        except subprocess.TimeoutExpired:
            log.warning("%s ignored the stop request for %.1f s; killing it", name, timeout_s)
    hard_kill(proc)
    try:
        return proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        log.error("%s (pid %s) did not exit after kill", name, proc.pid)
        return None


def wait_gone(proc: subprocess.Popen[Any], timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for ``proc`` to exit; whether it did."""
    try:
        proc.wait(timeout=max(0.0, timeout_s))
    except subprocess.TimeoutExpired:
        return False
    return True


def monotonic() -> float:
    """The launcher's time source (``perf_counter``, §2.5)."""
    return time.perf_counter()
