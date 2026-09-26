"""``Supervisor``: keeps the core and the voice worker running (§2.8, §2.9).

- Restart backoff ``min(max, min * 2**(n-1))`` after the n-th consecutive failure (0.5 s → 30 s);
  it resets once a run has lasted ``backoff[1]`` seconds.
- Crash-loop breaker ``(n, window)``: at most ``n`` restarts within ``window`` seconds; the next
  failure marks the child FAILED (``R`` on the console retries it). The default ``(5, 120)`` is
  §2.8's "more than 5 restarts in 120 s".
- ``terminate(name, hold=True)`` is HARD KILL: ``TerminateProcess``/``SIGKILL`` at once, and the
  child stays down until ``rearm(name)`` (§2.11).
- An exit code in ``fatal_exit_codes`` (2: bad config or usage) is not retried (FAILED); one in
  ``final_exit_codes`` ends the child normally (the core's 0 = quit and 3 = restart all).
- ``stop()`` stops children in spec order (core first, so it can finish its §2.9 shutdown while
  the voice worker still plays), each gracefully with a kill after the timeout.

Thread model: one monitor thread polls the children every ``poll_s``; every public method is
thread-safe (the emergency server and the console keys call them from their own threads).
Callbacks (``on_event``) run outside the lock.
"""

from __future__ import annotations

import collections
import logging
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aivtube.launcher import proc as _proc
from aivtube.launcher.proc import JobLike

__all__ = [
    "EXIT_CONFIG",
    "ChildState",
    "ProcessSpec",
    "Supervisor",
    "backoff_delay",
]

EXIT_CONFIG = 2
"""Exit code for a configuration or usage error: never restarted."""

ChildState = Literal["idle", "running", "backoff", "held", "failed", "exited", "stopping", "stopped"]
RestartPolicy = Literal["on_error", "always", "never"]
EventCallback = Callable[[str, str, Mapping[str, Any]], None]


def backoff_delay(failures: int, backoff: tuple[float, float]) -> float:
    """Delay before the restart that follows ``failures`` consecutive failures (>= 1)."""
    lo, hi = backoff
    return float(min(hi, lo * 2 ** max(0, failures - 1)))


@dataclass
class ProcessSpec:
    """One supervised child process."""

    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path = field(default_factory=Path.cwd)
    restart: RestartPolicy = "on_error"
    backoff: tuple[float, float] = (0.5, 30.0)
    breaker: tuple[int, float] = (5, 120.0)
    priority: str = "normal"
    log_path: Path | None = None
    stop_timeout_s: float | None = None
    fatal_exit_codes: frozenset[int] = frozenset({EXIT_CONFIG})
    final_exit_codes: frozenset[int] = frozenset()
    autostart: bool = True
    inherit_stdin: bool = False

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError(f"{self.name}: empty argv")
        if self.restart not in ("on_error", "always", "never"):
            raise ValueError(f"{self.name}: unknown restart policy {self.restart!r}")
        lo, hi = self.backoff
        if not 0 < lo <= hi or self.breaker[0] < 0 or self.breaker[1] <= 0:
            raise ValueError(f"{self.name}: invalid backoff or breaker")


@dataclass(eq=False)
class _Child:
    spec: ProcessSpec
    proc: subprocess.Popen[bytes] | None = None
    state: ChildState = "idle"
    detail: str = ""
    started_at: float = 0.0
    next_start: float = 0.0
    restarts: int = 0
    consecutive: int = 0
    exit_code: int | None = None
    restart_times: collections.deque[float] = field(default_factory=collections.deque)
    starts: int = 0


class Supervisor:
    """Starts, watches and restarts the child processes in ``specs``."""

    def __init__(
        self,
        specs: Sequence[ProcessSpec],
        job: JobLike | None,
        log: logging.Logger | None = None,
        *,
        on_event: EventCallback | None = None,
        clock: Callable[[], float] = time.perf_counter,
        poll_s: float = 0.05,
    ) -> None:
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate process names: {names}")
        self._children: dict[str, _Child] = {s.name: _Child(s) for s in specs}
        self._job = job
        self.log = log or logging.getLogger("aivtube.launcher.supervisor")
        self._on_event = on_event
        self._clock = clock
        self._poll_s = poll_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopping = False

    # -- lifecycle -------------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(self._children)

    def start(self) -> None:
        """Start every ``autostart`` child and the monitor thread."""
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            if self._thread is not None:
                return
            self._stopping = False
            self._quit.clear()
            for child in self._children.values():
                if child.spec.autostart:
                    self._spawn(child, events)
            self._thread = threading.Thread(
                target=self._run, name="launcher-supervisor", daemon=True
            )
            self._thread.start()
        self._emit(events)

    def stop(self, graceful_timeout: float = 5.0) -> None:
        """Stop every child (spec order, each gracefully then killed) and the monitor."""
        with self._lock:
            self._stopping = True
            order = list(self._children.values())
            for child in order:
                if child.state in ("backoff", "idle"):
                    child.state = "stopped"
        for child in order:
            self._stop_child(child, graceful_timeout)
        self._quit.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _stop_child(self, child: _Child, graceful_timeout: float) -> None:
        with self._lock:
            proc = child.proc
            if proc is None:
                if child.state not in ("failed", "held"):
                    child.state = "stopped"
                return
            if child.state != "held":
                child.state = "stopping"
        timeout = child.spec.stop_timeout_s if child.spec.stop_timeout_s is not None else graceful_timeout
        code = _proc.stop_process(proc, timeout, what=child.spec.name)
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            if child.proc is proc:
                child.proc = None
                child.exit_code = code
                if child.state == "stopping":
                    child.state = "stopped"
                events.append((child.spec.name, "stopped", {"exit_code": code}))
        self._emit(events)

    # -- operator actions --------------------------------------------------------------------

    def restart(self, name: str) -> bool:
        """Operator restart: stop gracefully (if running) and start again at once, clearing a
        FAILED state and the crash-loop history. A held child is not restarted (rearm it)."""
        child = self._child(name)
        with self._lock:
            if child.state == "held" or self._stopping:
                return False
            proc = child.proc
            if proc is not None:
                child.state = "stopping"
        if proc is not None:
            timeout = child.spec.stop_timeout_s if child.spec.stop_timeout_s is not None else 5.0
            code = _proc.stop_process(proc, timeout, what=name)
            with self._lock:
                if child.proc is proc:
                    child.proc = None
                    child.exit_code = code
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            if self._held(child) or self._stopping:  # a hard kill may have come in meanwhile
                return False
            if child.proc is not None and child.proc.poll() is None:
                return True  # a concurrent restart already started it
            child.consecutive = 0
            child.restart_times.clear()
            child.detail = "restarted by the operator"
            self._spawn(child, events)
        self._emit(events)
        self._wake.set()
        return child.state == "running"

    def retry_failed(self) -> list[str]:
        """Restart every FAILED child (console key ``R``). Returns their names."""
        with self._lock:
            failed = [n for n, c in self._children.items() if c.state == "failed"]
        return [n for n in failed if self.restart(n)]

    def terminate(self, name: str, *, hold: bool) -> float:
        """Kill ``name`` right away. With ``hold`` it stays down until :meth:`rearm`.

        Returns the seconds it took until the process was gone (HARD KILL target: ≤ 0.3 s).
        """
        t0 = self._clock()
        child = self._child(name)
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            proc = child.proc
            if hold:
                child.state = "held"
                child.detail = "hard kill: held down until rearm"
                events.append((name, "held", {"pid": proc.pid if proc else None}))
        if proc is not None:
            _proc.hard_kill(proc)
            gone = _proc.wait_gone(proc, 2.0)
            with self._lock:
                if gone and hold and child.proc is proc and child.state == "held":
                    child.proc = None
                    child.exit_code = proc.returncode
        elapsed = self._clock() - t0
        self._emit(events)
        self._wake.set()
        self.log.warning(
            "%s %s in %.0f ms%s", name, "hard-killed" if hold else "killed", elapsed * 1000,
            " (held until rearm)" if hold else "",
        )
        return elapsed

    def rearm(self, name: str) -> bool:
        """Release a held child and start it now. ``False`` if it was not held."""
        child = self._child(name)
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            if child.state != "held":
                return False
            if child.proc is not None and child.proc.poll() is None:
                return False  # still dying; the caller may retry
            child.proc = None
            child.consecutive = 0
            child.restart_times.clear()
            child.detail = "rearmed"
            events.append((name, "rearmed", {}))
            if not self._stopping:
                self._spawn(child, events)
            else:
                child.state = "stopped"
        self._emit(events)
        self._wake.set()
        return True

    def kill(self, name: str, reason: str) -> None:
        """Kill ``name`` as a failure (e.g. a wedged core); it restarts with backoff."""
        child = self._child(name)
        with self._lock:
            proc = child.proc
            if proc is None:
                return
            child.detail = reason
        self.log.error("killing %s: %s", name, reason)
        _proc.hard_kill(proc)
        self._wake.set()

    # -- queries -----------------------------------------------------------------------------

    def status(self) -> dict[str, dict[str, Any]]:
        now = self._clock()
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for name, c in self._children.items():
                running = c.proc is not None and c.proc.poll() is None
                out[name] = {
                    "state": c.state,
                    "pid": c.proc.pid if c.proc is not None else None,
                    "restarts": c.restarts,
                    "exit_code": c.exit_code,
                    "uptime_s": round(now - c.started_at, 3) if running else 0.0,
                    "held": c.state == "held",
                    "detail": c.detail,
                    "retry_in_s": round(max(0.0, c.next_start - now), 3)
                    if c.state == "backoff"
                    else None,
                }
        return out

    def info(self, name: str) -> tuple[ChildState, int | None, float]:
        """``(state, pid, started_at)`` of one child (for the heartbeat)."""
        child = self._child(name)
        with self._lock:
            pid = child.proc.pid if child.proc is not None else None
            return child.state, pid, child.started_at

    def state(self, name: str) -> ChildState:
        return self.info(name)[0]

    # -- internals ---------------------------------------------------------------------------

    @staticmethod
    def _held(child: _Child) -> bool:
        return child.state == "held"

    def _child(self, name: str) -> _Child:
        try:
            return self._children[name]
        except KeyError:
            raise KeyError(f"unknown process {name!r}") from None

    def _spawn(self, child: _Child, events: list[tuple[str, str, dict[str, Any]]]) -> None:
        spec = child.spec
        try:
            proc = _proc.spawn(
                spec.argv,
                cwd=spec.cwd,
                env=spec.env,
                job=self._job,
                priority=spec.priority,
                log_path=spec.log_path,
                inherit_stdin=spec.inherit_stdin,
            )
        except OSError as exc:
            self.log.error("cannot start %s: %s", spec.name, exc)
            child.proc = None
            child.exit_code = None
            self._after_failure(child, f"cannot start: {exc}", events)
            return
        child.proc = proc
        child.state = "running"
        child.started_at = self._clock()
        child.starts += 1
        child.exit_code = None
        self.log.info("started %s (pid %d)", spec.name, proc.pid)
        events.append((spec.name, "started", {"pid": proc.pid}))

    def _run(self) -> None:
        while not self._quit.is_set():
            try:
                self._tick()
            except Exception:  # the monitor must never die
                self.log.exception("supervisor tick failed")
            self._wake.wait(self._poll_s)
            self._wake.clear()

    def _tick(self) -> None:
        events: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            now = self._clock()
            for child in self._children.values():
                proc = child.proc
                if proc is not None and proc.poll() is not None:
                    self._on_exit(child, proc.returncode, now, events)
                if (
                    child.state == "backoff"
                    and not self._stopping
                    and now >= child.next_start
                ):
                    self._spawn(child, events)
        self._emit(events)

    def _on_exit(
        self,
        child: _Child,
        code: int,
        now: float,
        events: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        spec = child.spec
        child.proc = None
        child.exit_code = code
        uptime = now - child.started_at
        info = {"exit_code": code, "uptime_s": round(uptime, 3)}
        if child.state in ("held", "stopping", "stopped") or self._stopping:
            # an exit we asked for (stop, restart, hard kill): not the child's own decision
            events.append((spec.name, "stopped", info))
            if child.state == "stopping":
                child.state = "stopped"
            return
        events.append((spec.name, "exited", info))
        if code in spec.fatal_exit_codes:
            child.state = "failed"
            child.detail = f"exited with code {code} (configuration error; not restarted)"
            self.log.error("%s %s", spec.name, child.detail)
            events.append((spec.name, "failed", {"exit_code": code, "detail": child.detail}))
            return
        if (
            spec.restart == "never"
            or (spec.restart == "on_error" and code == 0)
            or code in spec.final_exit_codes
        ):
            child.state = "exited"
            child.detail = f"exited with code {code}"
            self.log.info("%s exited with code %s", spec.name, code)
            return
        if uptime >= spec.backoff[1]:
            child.consecutive = 0
        self.log.warning("%s exited with code %s after %.1f s", spec.name, code, uptime)
        self._after_failure(child, f"exited with code {code}", events)

    def _after_failure(
        self, child: _Child, why: str, events: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        spec = child.spec
        now = self._clock()
        limit, window = spec.breaker
        while child.restart_times and now - child.restart_times[0] > window:
            child.restart_times.popleft()
        if len(child.restart_times) >= limit:
            child.state = "failed"
            child.detail = (
                f"crash loop: {why}; {len(child.restart_times)} restarts within {window:g} s"
            )
            self.log.error("%s FAILED (%s); press R to retry", spec.name, child.detail)
            events.append((spec.name, "failed", {"detail": child.detail}))
            return
        child.consecutive += 1
        delay = backoff_delay(child.consecutive, spec.backoff)
        child.restart_times.append(now)
        child.restarts += 1
        child.state = "backoff"
        child.detail = why
        child.next_start = now + delay
        events.append((spec.name, "restarting", {"delay_s": delay, "why": why}))
        self.log.info("restarting %s in %.2f s", spec.name, delay)

    def _emit(self, events: list[tuple[str, str, dict[str, Any]]]) -> None:
        if self._on_event is None:
            return
        for name, kind, info in events:
            try:
                self._on_event(name, kind, info)
            except Exception:
                self.log.exception("supervisor event callback failed (%s %s)", name, kind)
