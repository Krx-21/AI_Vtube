"""Core heartbeat (§2.8 "Core loop wedged"): poll the panel's ``/healthz``; when the core has
not answered for ``timeout_s`` (5 s), ask it for a ``faulthandler`` stack dump, then kill it so
the supervisor restarts it.

Any HTTP answer, even 401 or 404, proves the core's event loop is alive; only silence (a
timeout, a refused or reset connection) counts. Until the first answer after a (re)start the
limit is ``startup_grace_s`` instead, so a slow start is not mistaken for a wedge.

The stack dump is requested through a file (``AIVTUBE_DUMP_REQUEST`` in the child's
environment): a wedged event loop cannot run a handler, but a plain thread in the child can
(``aivtube.launcher.childside.start_dump_request_watcher``). The same mechanism works on Windows,
which has no ``SIGUSR1``.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

__all__ = ["CoreHeartbeat", "http_alive", "request_dump"]

log = logging.getLogger("aivtube.launcher.heartbeat")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_alive(url: str, timeout_s: float) -> bool:
    """Whether an HTTP server answered ``GET url`` (any status) within ``timeout_s``."""
    try:
        with _OPENER.open(url, timeout=timeout_s) as resp:
            resp.read(1024)
        return True
    except urllib.error.HTTPError:
        return True  # it answered (e.g. 401 without a token): the loop is alive
    except (urllib.error.URLError, OSError, ValueError):
        return False


def request_dump(path: Path, pid: int, *, wait_s: float = 1.5) -> bool:
    """Ask a child for a stack dump by creating ``path``; wait until the child removes it.

    Returns whether the child acknowledged (removed the file) within ``wait_s``.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(pid), encoding="utf-8")
    except OSError:
        log.warning("cannot write the dump request %s", path, exc_info=True)
        return False
    end = time.perf_counter() + wait_s
    while time.perf_counter() < end:
        if not path.exists():
            return True
        time.sleep(0.05)
    with contextlib.suppress(OSError):
        path.unlink()
    return False


class _Target(Protocol):
    def info(self, name: str) -> tuple[str, int | None, float]: ...

    def kill(self, name: str, reason: str) -> None: ...


class CoreHeartbeat(threading.Thread):
    """Watches one supervised child through an HTTP endpoint (see the module docstring)."""

    def __init__(
        self,
        supervisor: _Target,
        name: str,
        url: str,
        *,
        timeout_s: float = 5.0,
        interval_s: float = 1.0,
        probe_timeout_s: float = 2.0,
        startup_grace_s: float = 90.0,
        dump_request: Path | None = None,
        dump_wait_s: float = 1.5,
        probe: Callable[[str, float], bool] = http_alive,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(name=f"heartbeat-{name}", daemon=True)
        self.supervisor = supervisor
        self.child = name
        self.url = url
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        self.probe_timeout_s = min(probe_timeout_s, timeout_s)
        self.startup_grace_s = startup_grace_s
        self.dump_request = dump_request
        self.dump_wait_s = dump_wait_s
        self._probe = probe
        self._clock = clock
        self._halt = threading.Event()
        self.kills = 0
        self.last_ok = 0.0
        self.armed = False

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        pid_seen: int | None = None
        since = 0.0
        while not self._halt.wait(self.interval_s):
            try:
                state, pid, _ = self.supervisor.info(self.child)
                if state != "running" or pid is None:
                    pid_seen = None
                    continue
                if pid != pid_seen:
                    pid_seen, self.armed = pid, False
                    since = self._clock()
                ok = self._probe(self.url, self.probe_timeout_s)
                now = self._clock()
                if ok:
                    self.armed, since, self.last_ok = True, now, now
                    continue
                limit = self.timeout_s if self.armed else self.startup_grace_s
                silent = now - since
                if silent >= limit:
                    self._wedged(pid, silent)
                    pid_seen = None
            except Exception:  # the watchdog itself must never die
                log.exception("heartbeat check failed")

    def _wedged(self, pid: int, silent_s: float) -> None:
        what = "no answer" if self.armed else "never answered after start"
        log.error("%s (pid %d): %s for %.1f s; dumping stacks and restarting it",
                  self.child, pid, what, silent_s)
        if self.dump_request is not None:
            acked = request_dump(self.dump_request, pid, wait_s=self.dump_wait_s)
            log.info("stack dump %s", "written by the child" if acked else "not acknowledged")
        self.kills += 1
        self.supervisor.kill(self.child, f"wedged: {what} for {silent_s:.1f} s")


def port_open(host: str, port: int, timeout_s: float = 0.3) -> bool:
    """Whether something accepts TCP connections on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
