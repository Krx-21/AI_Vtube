"""``LlamaServerController``: start, adopt, watch and restart one llama-server (§2.3, §2.8, §4.11).

- **Adopt.** A server already listening on the port whose ``/props`` reports the expected
  alias and model file is adopted, never started twice; an adopted server is left running when
  the launcher exits (it keeps its warm KV cache). A different server on the port is an error.
- **Start.** The command comes from ``aivtube.llm.llama_args.build_llama_argv`` (the same
  builder the standalone manager uses); output goes to ``logs/<date>/llama-<name>.log``.
- **Load failures** (the process exits before ``/health`` reaches 200, or it never gets there
  within ``load_timeout_s``): with ``placement = "pinned"`` the launcher raises
  ``--n-cpu-moe`` by ``n_cpu_moe_step``, up to ``--cpu-moe``, saves the new N to
  ``data/state/llama_tuning.json`` and retries; after ``max_load_failures`` in a row the
  server is FAILED and the router serves from the next provider.
- **Crash or hang** after it was ready: restart with backoff; a hang is ``health_failures``
  failed ``/health`` checks ``health_interval_s`` apart. A crash loop marks it FAILED.

A saved N applies only while the configured ``pinned_n_cpu_moe`` and model are the ones it
was derived from; ``bench llm --tune`` writing a new N supersedes it.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from aivtube.launcher import proc as _proc
from aivtube.launcher.proc import JobLike
from aivtube.launcher.supervisor import backoff_delay
from aivtube.llm.llama_args import LOOPBACK, build_llama_argv, slot_save_path

__all__ = [
    "MAX_N_CPU_MOE",
    "LlamaServerController",
    "TuningStore",
    "fetch_json",
    "props_match",
]

MAX_N_CPU_MOE = 48
"""MoE layers of Qwen3-30B-A3B; any higher N means every expert on the CPU (``--cpu-moe``)."""

Phase = Literal[
    "idle", "adopting", "adopted", "loading", "ready", "backoff", "failed", "stopped", "released"
]
HealthWord = Literal["ok", "loading", "down"]
NCpuMoe = int | Literal["all"]

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_json(url: str, timeout_s: float) -> tuple[int, Any]:
    """``GET url`` → ``(status, parsed JSON or None)``; ``(0, None)`` when unreachable."""
    try:
        with _OPENER.open(url, timeout=timeout_s) as resp:
            status, body = resp.status, resp.read(1 << 20)
    except urllib.error.HTTPError as exc:
        status, body = exc.code, b""
        try:
            body = exc.read(1 << 16)
        except OSError:
            pass
        finally:
            exc.close()
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(body) if body else None
    except ValueError:
        return status, None


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def props_match(props: Mapping[str, Any], *, alias: str, model: str = "") -> bool:
    """The running server is ours: ``model_alias`` lists ``alias`` and ``model_path`` has the
    same file name as ``model`` (when both are known). Mirrors ``aivtube.llm.props_match``."""
    got_alias = props.get("model_alias")
    aliases = {a.strip() for a in str(got_alias).split(",")} if got_alias else set()
    if alias and alias not in aliases:
        return False
    got_model = props.get("model_path")
    if model and isinstance(got_model, str) and got_model:
        return _basename(got_model).lower() == _basename(model).lower()
    return True


class TuningStore:
    """``data/state/llama_tuning.json``: the N the launcher raised after load failures."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, server: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._read().get("servers", {}).get(server)
        return entry if isinstance(entry, dict) else None

    def set(self, server: str, **fields: Any) -> None:
        with self._lock:
            data = self._read()
            data["version"] = 1
            servers = data.setdefault("servers", {})
            if not isinstance(servers, dict):
                servers = data["servers"] = {}
            servers[server] = {
                **fields,
                "updated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)

    def clear(self, server: str) -> None:
        with self._lock:
            data = self._read()
            servers = data.get("servers")
            if isinstance(servers, dict) and servers.pop(server, None) is not None:
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(tmp, self.path)


class LlamaServerController:
    """One ``[llm.servers.<name>]`` entry (see the module docstring). Thread-safe."""

    def __init__(
        self,
        name: str,
        cfg: Mapping[str, Any],
        log: logging.Logger | None = None,
        *,
        root: Path,
        state_dir: Path,
        log_dir: Path | None = None,
        job: JobLike | None = None,
        health_interval_s: float = 2.0,
        health_failures: int = 3,
        load_timeout_s: float = 300.0,
        max_load_failures: int = 3,
        graceful_timeout_s: float = 5.0,
        backoff: tuple[float, float] = (0.5, 30.0),
        breaker: tuple[int, float] = (5, 120.0),
        command_prefix: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        http_timeout_s: float = 1.0,
        poll_s: float = 0.25,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.cfg = dict(cfg)
        self.log = log or logging.getLogger(f"aivtube.launcher.llama.{name}")
        self.root = Path(root)
        self.state_dir = Path(state_dir)
        self.log_dir = log_dir
        self.job = job
        self.health_interval_s = health_interval_s
        self.health_failures = max(1, health_failures)
        self.load_timeout_s = load_timeout_s
        self.max_load_failures = max(1, max_load_failures)
        self.graceful_timeout_s = graceful_timeout_s
        self.backoff = backoff
        self.breaker = breaker
        self.command_prefix = list(command_prefix) if command_prefix else None
        self.env = dict(env or {})
        self.http_timeout_s = http_timeout_s
        self.poll_s = poll_s
        self._clock = clock
        self.port = int(self.cfg.get("port", 0))
        self.alias = str(self.cfg.get("alias", ""))
        self.model = str(self.cfg.get("model", ""))
        self.placement = str(self.cfg.get("placement", "fit"))
        self.adopt_existing = bool(self.cfg.get("adopt_existing", True))
        self.tuning = TuningStore(self.state_dir / "llama_tuning.json")
        self.n_cpu_moe: NCpuMoe | None = self._initial_n()

        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._wake = threading.Event()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.phase: Phase = "idle"
        self.detail = ""
        self.desired = False
        self.closed = False
        self.load_failures = 0
        self.restarts = 0
        self._consecutive = 0
        self._restart_times: collections.deque[float] = collections.deque()
        self._phase_since = 0.0
        self._next_start = 0.0
        self._next_health = 0.0
        self._health_fails = 0

    # -- config ------------------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def _pinned_base(self) -> int:
        try:
            return int(self.cfg.get("pinned_n_cpu_moe", 0))
        except (TypeError, ValueError):
            return 0

    def _initial_n(self) -> NCpuMoe | None:
        if self.placement != "pinned":
            return None
        base = self._pinned_base()
        entry = self.tuning.get(self.name)
        if entry and entry.get("base") == base and entry.get("model") == _basename(self.model):
            saved = entry.get("n_cpu_moe")
            if saved == "all" or (isinstance(saved, int) and saved >= base):
                self.log.info("%s: using the saved --n-cpu-moe %s (config %d)", self.name, saved, base)
                return "all" if saved == "all" else int(saved)
        return base

    def build_argv(self) -> list[str]:
        """The llama-server command line (with the current N for pinned placement)."""
        argv = build_llama_argv(self.cfg, root=self.root, n_cpu_moe=self.n_cpu_moe)
        if self.command_prefix:
            argv = [*self.command_prefix, *argv[1:]]
        return argv

    # -- probes ------------------------------------------------------------------------------

    def health(self) -> HealthWord:
        """``ok`` (200), ``loading`` (503) or ``down``."""
        status, _ = fetch_json(self.base_url + "/health", self.http_timeout_s)
        if status == 200:
            return "ok"
        if status == 503:
            return "loading"
        return "down"

    def props(self) -> Mapping[str, Any] | None:
        status, body = fetch_json(self.base_url + "/props", self.http_timeout_s)
        return body if status == 200 and isinstance(body, Mapping) else None

    def adopt(self) -> bool:
        """Adopt a healthy server on the port whose ``/props`` alias and model match."""
        if self.health() != "ok":
            return False
        props = self.props()
        if props is None or not props_match(props, alias=self.alias, model=self.model):
            return False
        with self._lock:
            if self.proc is None:
                self._set_phase("adopted", "adopted a running server")
                self._health_fails = 0
                self._next_health = self._clock() + self.health_interval_s
        self.log.info("%s: adopted the running server at %s", self.name, self.base_url)
        return True

    # -- control -----------------------------------------------------------------------------

    def start(self) -> None:
        """Start the watcher thread (idempotent). Nothing runs until ``ensure``/``want``."""
        with self._lock:
            if self._thread is not None or self.closed:
                return
            self._quit.clear()
            self._thread = threading.Thread(target=self._run, name=f"llama-{self.name}", daemon=True)
            self._thread.start()

    def want(self) -> bool:
        """Keep the server running from now on (autostart) without waiting for it.
        ``False`` once the controller is closed (the launcher is shutting down)."""
        self.start()
        with self._lock:
            if self.closed:
                return False
            self.desired = True
            if self.phase == "released":
                self._set_phase("idle", "wanted again")
        self._wake.set()
        return True

    def ensure(self, timeout_s: float) -> bool:
        """Make sure the server runs (adopt or start it) and wait up to ``timeout_s`` for it to
        be ready. A server still loading keeps loading after a ``False``."""
        if not self.want():
            return False
        end = self._clock() + max(0.0, timeout_s)
        with self._changed:
            while self.phase not in ("ready", "adopted", "failed"):
                if not self.desired or self._quit.is_set():  # stop() or close() meanwhile
                    return False
                remaining = end - self._clock()
                if remaining <= 0:
                    break
                self._changed.wait(min(remaining, 0.25))
            return self.phase in ("ready", "adopted")

    def stop(self) -> None:
        """Stop wanting the server. Ours is stopped (gracefully, then killed); an adopted one
        is left running."""
        with self._lock:
            self.desired = False
            proc, self.proc = self.proc, None
            if proc is None and self.phase == "adopted":
                self._set_phase("released", "adopted server left running")
            elif self.phase != "failed":
                self._set_phase("stopped", "stopped")
        if proc is not None:
            _proc.stop_process(proc, self.graceful_timeout_s, what=f"llama-server {self.name}")
        self._wake.set()

    def restart(self) -> bool:
        """Operator retry: clear FAILED and the failure counters, restart our process.
        ``False`` once the controller is closed."""
        with self._lock:
            if self.closed:
                return False
            proc, self.proc = self.proc, None
            self.load_failures = 0
            self._consecutive = 0
            self._restart_times.clear()
            self.desired = True
            self._set_phase("idle", "restart requested")
        if proc is not None:
            _proc.stop_process(proc, self.graceful_timeout_s, what=f"llama-server {self.name}")
        self.start()
        self._wake.set()
        return True

    def close(self) -> None:
        """Shut down for good: stop our process (an adopted server stays up) and the watcher.
        Later ``want``/``ensure``/``restart`` calls do nothing."""
        with self._lock:
            self.closed = True
        self.stop()
        self._quit.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def on_load_failure(self) -> None:
        """Pinned placement: raise N by the step (up to ``--cpu-moe``) and persist it."""
        if self.placement != "pinned":
            self.log.warning("%s: load failure (placement %s)", self.name, self.placement)
            return
        current = self.n_cpu_moe if self.n_cpu_moe is not None else self._pinned_base()
        if isinstance(current, str):  # "all": every expert is on the CPU already
            self.log.error("%s: load failed even with every expert on the CPU", self.name)
            return
        step = max(1, int(self.cfg.get("n_cpu_moe_step", 2) or 2))
        bumped = current + step
        new: NCpuMoe = "all" if bumped > MAX_N_CPU_MOE else bumped
        self.n_cpu_moe = new
        self.log.warning(
            "%s: load failed; raising --n-cpu-moe from %s to %s (saved to %s)",
            self.name, current, new, self.tuning.path,
        )
        try:
            self.tuning.set(
                self.name,
                n_cpu_moe=new,
                base=self._pinned_base(),
                model=_basename(self.model),
                reason="load failure",
            )
        except OSError:
            self.log.warning("cannot save %s", self.tuning.path, exc_info=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            proc = self.proc
            return {
                "phase": self.phase,
                "health": "up" if self.phase in ("ready", "adopted") else "down",
                "pid": proc.pid if proc is not None else None,
                "adopted": self.phase in ("adopted", "released"),
                "port": self.port,
                "alias": self.alias,
                "placement": self.placement,
                "n_cpu_moe": self.n_cpu_moe,
                "load_failures": self.load_failures,
                "restarts": self.restarts,
                "detail": self.detail,
            }

    # -- watcher -----------------------------------------------------------------------------

    def _set_phase(self, phase: Phase, detail: str = "") -> None:
        # caller holds the lock
        self.phase = phase
        self.detail = detail
        self._phase_since = self._clock()
        self._changed.notify_all()

    def _run(self) -> None:
        while not self._quit.is_set():
            try:
                self._tick()
            except Exception:
                self.log.exception("%s: watcher tick failed", self.name)
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def _tick(self) -> None:
        with self._lock:
            if not self.desired:
                return
            phase, proc, now = self.phase, self.proc, self._clock()
        if phase in ("failed", "released"):
            return
        if phase in ("idle", "stopped") or (phase == "backoff" and now >= self._next_start):
            self._bring_up()
        elif phase == "adopting":
            self._tick_adopting(now)
        elif phase == "adopted":
            self._tick_adopted(now)
        elif phase == "loading" and proc is not None:
            self._tick_loading(proc, now)
        elif phase == "ready" and proc is not None:
            self._tick_ready(proc, now)

    def _bring_up(self) -> None:
        state = self.health()
        if state != "down":
            if not self.adopt_existing:
                with self._lock:
                    self._set_phase(
                        "failed", f"port {self.port} is in use and adopt_existing = false"
                    )
                self.log.error("%s: %s", self.name, self.detail)
                return
            if state == "ok" and self.adopt():
                return
            if state == "loading":
                with self._lock:
                    self._set_phase("adopting", "a server on the port is still loading")
                return
            props = self.props() or {}
            with self._lock:
                self._set_phase(
                    "failed",
                    f"port {self.port} is taken by another server (alias "
                    f"{props.get('model_alias')!r}, model {props.get('model_path')!r}); expected "
                    f"{self.alias!r}",
                )
            self.log.error("%s: %s", self.name, self.detail)
            return
        self._spawn()

    def _spawn(self) -> None:
        argv = self.build_argv()
        try:
            slot_save_path(self.cfg, root=self.root).mkdir(parents=True, exist_ok=True)
        except OSError:
            self.log.warning("cannot create the slot save directory", exc_info=True)
        log_path = None
        if self.log_dir is not None:
            log_path = self.log_dir / _dt.date.today().isoformat() / f"llama-{self.name}.log"
        try:
            proc = _proc.spawn(
                argv,
                cwd=self.root,
                env=_proc.child_env(self.env),
                job=self.job,
                log_path=log_path,
            )
        except OSError as exc:
            self.log.error("%s: cannot start llama-server: %s", self.name, exc)
            with self._lock:
                self._set_phase("failed", f"cannot start {argv[0]}: {exc}")
            return
        with self._lock:
            if not self.desired:  # stop() raced with the start
                _proc.hard_kill(proc)
                return
            self.proc = proc
            self._set_phase("loading", f"loading (pid {proc.pid})")
        self.log.info("%s: started llama-server (pid %d): %s", self.name, proc.pid, " ".join(argv))

    def _tick_adopting(self, now: float) -> None:
        state = self.health()
        if state == "ok":
            if not self.adopt():
                with self._lock:
                    if self.phase == "adopting":
                        self._set_phase("idle")  # re-evaluated next tick (foreign → failed)
            return
        if state == "down":
            with self._lock:
                if self.phase == "adopting":
                    self._set_phase("idle", "the loading server went away")
            return
        if now - self._phase_since > self.load_timeout_s:
            with self._lock:
                self._set_phase("failed", "a server on the port never finished loading")

    def _tick_adopted(self, now: float) -> None:
        if now < self._next_health:
            return
        self._next_health = now + self.health_interval_s
        if self.health() == "ok":
            self._health_fails = 0
            return
        self._health_fails += 1
        if self._health_fails >= self.health_failures:
            self.log.warning("%s: the adopted server stopped answering /health", self.name)
            with self._lock:
                if self.phase == "adopted":
                    self._set_phase("idle", "adopted server lost")
                    self._health_fails = 0

    def _tick_loading(self, proc: subprocess.Popen[bytes], now: float) -> None:
        code = proc.poll()
        if code is not None:
            self._load_failed(proc, f"exited with code {code} while loading")
            return
        if self.health() == "ok":
            with self._lock:
                if self.proc is proc:
                    self.load_failures = 0
                    self._health_fails = 0
                    self._next_health = self._clock() + self.health_interval_s
                    self._set_phase("ready", f"ready (pid {proc.pid})")
            self.log.info("%s: ready after %.1f s", self.name, now - self._phase_since)
            return
        if now - self._phase_since > self.load_timeout_s:
            _proc.hard_kill(proc)
            _proc.wait_gone(proc, 5.0)
            self._load_failed(proc, f"not ready within {self.load_timeout_s:g} s")

    def _load_failed(self, proc: subprocess.Popen[bytes], why: str) -> None:
        with self._lock:
            if self.proc is not proc:
                return
            self.proc = None
            self.load_failures += 1
            failures = self.load_failures
        self.log.error("%s: load failure %d/%d: %s", self.name, failures, self.max_load_failures, why)
        self.on_load_failure()
        with self._lock:
            if failures >= self.max_load_failures:
                self._set_phase("failed", f"{failures} load failures in a row ({why})")
            else:
                delay = backoff_delay(failures, self.backoff)
                self._next_start = self._clock() + delay
                self._set_phase("backoff", f"load failed ({why}); retrying in {delay:g} s")

    def _tick_ready(self, proc: subprocess.Popen[bytes], now: float) -> None:
        code = proc.poll()
        if code is not None:
            self._crashed(proc, f"exited with code {code}")
            return
        if now < self._next_health:
            return
        self._next_health = now + self.health_interval_s
        if self.health() == "ok":
            self._health_fails = 0
            return
        self._health_fails += 1
        if self._health_fails >= self.health_failures:
            self.log.error("%s: %d failed /health checks; restarting it", self.name, self._health_fails)
            _proc.hard_kill(proc)
            _proc.wait_gone(proc, 5.0)
            self._crashed(proc, f"{self._health_fails} failed /health checks")

    def _crashed(self, proc: subprocess.Popen[bytes], why: str) -> None:
        with self._lock:
            if self.proc is not proc:
                return
            self.proc = None
            now = self._clock()
            uptime = now - self._phase_since
            if uptime >= self.backoff[1]:
                self._consecutive = 0
            limit, window = self.breaker
            while self._restart_times and now - self._restart_times[0] > window:
                self._restart_times.popleft()
            if len(self._restart_times) >= limit:
                self._set_phase("failed", f"crash loop: {why}")
                self.log.error("%s FAILED: %s", self.name, self.detail)
                return
            self._consecutive += 1
            self._restart_times.append(now)
            self.restarts += 1
            delay = backoff_delay(self._consecutive, self.backoff)
            self._next_start = now + delay
            self._set_phase("backoff", f"{why}; restarting in {delay:g} s")
        self.log.warning("%s: %s; restarting in %.2f s", self.name, why, delay)
