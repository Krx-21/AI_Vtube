"""llama-server administration and ``LocalServerManager`` implementations (§2.6, §4.8, §4.11).

- ``LlamaCppAdmin``: the server-root endpoints (public, no ``/v1``): ``GET /health`` (200 ok,
  503 loading), ``GET /props``, ``GET /slots`` and ``POST /slots/{id}?action=save|restore|erase``.
- ``check_tool_caps``: the startup assertion ``/props → chat_template_caps.supports_tool_calls``.
  A GGUF without a tool-capable template silently ignores tools, so anything else is a hard
  ``TemplateCapsError``.
- ``LlamaServerManager``: works without the launcher. ``mode="spawn"`` (standalone) adopts a
  running server whose ``/props`` alias and model match, otherwise spawns llama-server itself
  and polls ``/health``; ``mode="adopt"`` never spawns or stops anything and only waits for a
  matching server (the launcher, or the user, runs it).
- ``LauncherServerManager``: asks the launcher's loopback emergency endpoint
  (``POST /llm/ensure/<server>``, ``/llm/stop/<server>``) and talks to the servers directly for
  props and slots.

Slot files are named ``<char>-<prefix_hash>.bin`` (``slot_filename``) and live under
``--slot-save-path``; llama-server accepts only a bare file name.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

import httpx2

from aivtube.contracts.infra import Clock
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.llm.llama_args import build_llama_argv, root_url, server_root_url

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig

__all__ = [
    "HealthWord",
    "LauncherServerManager",
    "LlamaAdminError",
    "LlamaCppAdmin",
    "LlamaServerManager",
    "LlamaServerSpec",
    "TemplateCapsError",
    "check_tool_caps",
    "props_match",
    "slot_filename",
]

log = logging.getLogger(__name__)

HealthWord = Literal["ok", "loading", "down"]

_SLOT_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


class TemplateCapsError(RuntimeError):
    """The loaded model's chat template cannot call tools (§2.6 step 5). A hard error."""


class LlamaAdminError(RuntimeError):
    """A llama-server admin request failed (unreachable, timed out or an HTTP error)."""


def slot_filename(character: str, prefix_hash: str) -> str:
    """``<char>-<prefix_hash>.bin``; raises ``ValueError`` for anything but a bare file name."""
    name = f"{character}-{prefix_hash}.bin"
    if not _valid_slot_file(name):
        raise ValueError(f"invalid slot file name {name!r}")
    return name


def _valid_slot_file(name: str) -> bool:
    return bool(_SLOT_FILE.match(name)) and ".." not in name


def check_tool_caps(props: Mapping[str, Any], *, server: str = "llama-server") -> None:
    """Raise ``TemplateCapsError`` unless ``chat_template_caps.supports_tool_calls`` is true."""
    caps = props.get("chat_template_caps")
    if isinstance(caps, Mapping) and caps.get("supports_tool_calls") is True:
        return
    raise TemplateCapsError(
        f"{server}: the model's chat template does not support tool calls, so tools would be "
        "silently ignored. Use the mradermacher GGUF (it embeds the Typhoon template) or add "
        "--chat-template-file <chat_template.jinja> to extra_args. / เทมเพลตแชตของโมเดลนี้"
        "ไม่รองรับการเรียกเครื่องมือ ให้ใช้ไฟล์ GGUF ของ mradermacher หรือเพิ่ม "
        "--chat-template-file ใน extra_args"
    )


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def props_match(props: Mapping[str, Any], *, alias: str, model: str = "") -> bool:
    """Whether a running server is the one we expect: its ``model_alias`` lists ``alias`` and
    its ``model_path`` has the same file name as ``model`` (when both are known)."""
    got_alias = props.get("model_alias")
    aliases = {a.strip() for a in str(got_alias).split(",")} if got_alias else set()
    if alias and alias not in aliases:
        return False
    got_model = props.get("model_path")
    if model and isinstance(got_model, str) and got_model:
        return _basename(got_model).lower() == _basename(model).lower()
    return True


class LlamaCppAdmin:
    """Client for one llama-server's root endpoints. Every call has a deadline (I2)."""

    def __init__(
        self,
        base_url: str,
        http: httpx2.AsyncClient | None = None,
        *,
        clock: Clock | None = None,
        timeout_s: float = 2.0,
        slot_timeout_s: float = 120.0,
    ) -> None:
        self.base_url = root_url(base_url)
        self._owns_http = http is None
        self.http: httpx2.AsyncClient = (
            http
            if http is not None
            else httpx2.AsyncClient(
                trust_env=False, timeout=httpx2.Timeout(slot_timeout_s, connect=timeout_s)
            )
        )
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.timeout_s = timeout_s
        self.slot_timeout_s = slot_timeout_s

    def __repr__(self) -> str:
        return f"LlamaCppAdmin({self.base_url!r})"

    async def _request(
        self, method: str, path: str, *, timeout_s: float, **kw: Any
    ) -> httpx2.Response:
        url = self.base_url + path
        try:
            async with deadline(timeout_s, what=f"{method} {url}", clock=self.clock):
                resp = await self.http.request(method, url, **kw)
                await resp.aread()
        except DeadlineExceeded as exc:
            raise LlamaAdminError(f"{method} {url}: no answer within {timeout_s:g} s") from exc
        except httpx2.HTTPError as exc:
            raise LlamaAdminError(f"{method} {url}: {type(exc).__name__}: {exc}") from exc
        return resp

    async def health(self) -> HealthWord:
        """``ok`` (200), ``loading`` (503 while the model loads) or ``down``."""
        try:
            resp = await self._request("GET", "/health", timeout_s=self.timeout_s)
        except LlamaAdminError:
            return "down"
        if resp.status_code == 200:
            return "ok"
        if resp.status_code == 503:
            return "loading"
        return "down"

    async def props(self) -> Mapping[str, Any]:
        resp = await self._request("GET", "/props", timeout_s=self.timeout_s)
        if resp.status_code != 200:
            raise LlamaAdminError(f"GET {self.base_url}/props returned {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LlamaAdminError(f"GET {self.base_url}/props: not JSON") from exc
        if not isinstance(data, Mapping):
            raise LlamaAdminError(f"GET {self.base_url}/props: not an object")
        return data

    async def assert_tool_caps(self) -> None:
        """Raise ``TemplateCapsError`` unless the template supports tool calls."""
        check_tool_caps(await self.props(), server=self.base_url)

    async def slots(self) -> list[Mapping[str, Any]]:
        """``GET /slots`` (per-slot state, e.g. ``is_processing``)."""
        resp = await self._request("GET", "/slots", timeout_s=self.timeout_s)
        if resp.status_code != 200:
            raise LlamaAdminError(f"GET {self.base_url}/slots returned {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LlamaAdminError(f"GET {self.base_url}/slots: not JSON") from exc
        return [s for s in data if isinstance(s, Mapping)] if isinstance(data, list) else []

    async def save_slot(self, slot: int, filename: str) -> bool:
        """``POST /slots/{slot}?action=save`` with ``{"filename": …}``."""
        return await self._slot_action(slot, "save", filename)

    async def restore_slot(self, slot: int, filename: str) -> bool:
        """``POST /slots/{slot}?action=restore``; ``False`` when the file is missing."""
        return await self._slot_action(slot, "restore", filename)

    async def erase_slot(self, slot: int) -> bool:
        return await self._slot_action(slot, "erase", None)

    async def _slot_action(self, slot: int, action: str, filename: str | None) -> bool:
        if filename is not None and not _valid_slot_file(filename):
            log.warning("refusing slot %s of %r: not a bare file name", action, filename)
            return False
        body = {"filename": filename} if filename is not None else {}
        try:
            resp = await self._request(
                "POST",
                f"/slots/{int(slot)}",
                params={"action": action},
                json=body,
                timeout_s=self.slot_timeout_s,
            )
        except LlamaAdminError as exc:
            log.warning("slot %s failed: %s", action, exc)
            return False
        if resp.status_code != 200:
            log.warning(
                "slot %s %s on %s returned %s: %s",
                action,
                filename or "",
                self.base_url,
                resp.status_code,
                resp.text[:200],
            )
            return False
        return True

    async def aclose(self) -> None:
        if self._owns_http:
            await self.http.aclose()


class _AdminManager:
    """Props and slot calls shared by both managers (servers are addressed by name)."""

    def __init__(self, admins: Mapping[str, LlamaCppAdmin], *, clock: Clock | None) -> None:
        self._admins = dict(admins)
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._locks: dict[str, asyncio.Lock] = {}

    def admin(self, server: str) -> LlamaCppAdmin | None:
        return self._admins.get(server)

    def _lock(self, server: str) -> asyncio.Lock:
        lock = self._locks.get(server)
        if lock is None:
            lock = self._locks[server] = asyncio.Lock()
        return lock

    async def props(self, server: str) -> Mapping[str, Any]:
        admin = self._admins.get(server)
        if admin is None:
            raise LlamaAdminError(f"unknown llama server {server!r}")
        return await admin.props()

    async def save_slot(self, server: str, slot: int, filename: str) -> bool:
        admin = self._admins.get(server)
        return admin is not None and await admin.save_slot(slot, filename)

    async def restore_slot(self, server: str, slot: int, filename: str) -> bool:
        admin = self._admins.get(server)
        return admin is not None and await admin.restore_slot(slot, filename)

    async def _wait_healthy(self, admin: LlamaCppAdmin, end: float, poll_s: float) -> bool:
        while True:
            if await admin.health() == "ok":
                return True
            remaining = end - self._clock.now()
            if remaining <= 0:
                return False
            await self._clock.sleep(min(poll_s, remaining))


def _make_slot_dirs(argv: Sequence[str], cwd: Path | None) -> None:
    """Create the ``--slot-save-path`` directory: llama-server exits when it is missing."""
    for i, arg in enumerate(argv):
        flag, _, value = arg.partition("=")
        if flag != "--slot-save-path":
            continue
        if not value and i + 1 < len(argv):
            value = argv[i + 1]
        if value:
            path = Path(value)
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class LlamaServerSpec:
    """One llama-server for ``LlamaServerManager``: where it listens, what it must report in
    ``/props``, and (spawn mode) the command that starts it."""

    name: str
    base_url: str
    alias: str
    model: str = ""
    argv: tuple[str, ...] = ()
    cwd: Path | None = None
    log_path: Path | None = None
    env: Mapping[str, str] | None = None


class LlamaServerManager(_AdminManager):
    """``LocalServerManager`` without the launcher (see the module docstring).

    ``check_caps`` makes ``ensure_running`` raise ``TemplateCapsError`` for a server whose
    template cannot call tools. ``stop()`` only ever stops a process this manager spawned; an
    adopted server is left running (it keeps its warm KV cache).

    When ``ensure_running`` times out, a spawned process that is still loading (``/health``
    503) keeps loading in the background, like under the launcher, so a later call (e.g. the
    router's half-open retry) finds it ready; it is killed once it has been loading for
    ``load_timeout_s`` or when it never answered at all.
    """

    def __init__(
        self,
        specs: Sequence[LlamaServerSpec] | Mapping[str, LlamaServerSpec],
        *,
        mode: Literal["spawn", "adopt"] = "spawn",
        http: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
        check_caps: bool = True,
        poll_interval_s: float = 0.25,
        graceful_timeout_s: float = 5.0,
        load_timeout_s: float = 300.0,
    ) -> None:
        items = list(specs.values()) if isinstance(specs, Mapping) else list(specs)
        self._specs = {s.name: s for s in items}
        self._owns_http = http is None
        self._http = http if http is not None else httpx2.AsyncClient(trust_env=False)
        super().__init__(
            {s.name: LlamaCppAdmin(s.base_url, self._http, clock=clock) for s in items},
            clock=clock,
        )
        self.mode = mode
        self._check_caps = check_caps
        self._poll_s = poll_interval_s
        self._graceful_s = graceful_timeout_s
        self._load_timeout_s = load_timeout_s
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._spawned_at: dict[str, float] = {}
        self._adopted: set[str] = set()

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        *,
        mode: Literal["spawn", "adopt"] = "spawn",
        servers: Sequence[str] | None = None,
        log_dir: Path | None = None,
        n_cpu_moe: Mapping[str, int | Literal["all"]] | None = None,
        http: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
        check_caps: bool = True,
    ) -> LlamaServerManager:
        """Specs for ``cfg.llm.servers`` (or just ``servers``), with the llama-server command
        from ``build_llama_argv`` and logs in ``log_dir/llama-<name>.log``."""
        specs: list[LlamaServerSpec] = []
        for name, server in cfg.llm.servers.items():
            if servers is not None and name not in servers:
                continue
            argv = build_llama_argv(server, root=cfg.root, n_cpu_moe=(n_cpu_moe or {}).get(name))
            specs.append(
                LlamaServerSpec(
                    name=name,
                    base_url=server_root_url(server),
                    alias=server.alias,
                    model=str(cfg.resolve_path(server.model)),
                    argv=tuple(argv),
                    cwd=cfg.root,
                    log_path=log_dir / f"llama-{name}.log" if log_dir is not None else None,
                )
            )
        return cls(
            specs,
            mode=mode,
            http=http,
            clock=clock,
            check_caps=check_caps,
            graceful_timeout_s=cfg.launcher.graceful_timeout_s,
            load_timeout_s=cfg.launcher.llm_load_timeout_s,
        )

    def adopted(self, server: str) -> bool:
        """Whether ``server`` was already running when we found it (not spawned by us)."""
        return server in self._adopted

    def process(self, server: str) -> asyncio.subprocess.Process | None:
        return self._procs.get(server)

    async def ensure_running(self, server: str, timeout_s: float) -> bool:
        spec = self._specs.get(server)
        if spec is None:
            return False
        async with self._lock(server):
            return await self._ensure(spec, timeout_s)

    async def _ensure(self, spec: LlamaServerSpec, timeout_s: float) -> bool:
        admin = self._admins[spec.name]
        end = self._clock.now() + timeout_s
        state = await admin.health()
        if state == "ok":
            return await self._accept(spec, admin)
        if state == "down" and self.mode == "spawn" and not self._alive(spec.name):
            if not spec.argv:
                log.error("llama server %s is down and has no command to start it", spec.name)
                return False
            try:
                await self._spawn(spec)
            except OSError as exc:
                log.error("could not start llama server %s: %s", spec.name, exc)
                return False
        while True:
            proc = self._procs.get(spec.name)
            if proc is not None and proc.returncode is not None:
                self._procs.pop(spec.name, None)
                self._spawned_at.pop(spec.name, None)
                log.error("llama server %s exited with code %s", spec.name, proc.returncode)
                return False
            remaining = end - self._clock.now()
            if remaining <= 0:
                break
            await self._clock.sleep(min(self._poll_s, remaining))
            state = await admin.health()
            if state == "ok":
                return await self._accept(spec, admin)
        if self._alive(spec.name):
            age = self._clock.now() - self._spawned_at.get(spec.name, self._clock.now())
            if age < self._load_timeout_s:
                # Loading (503) or still starting (port not bound yet: on Windows a refused
                # loopback connect itself takes ~2 s). Either way it is young, so let it finish.
                log.warning(
                    "llama server %s not ready after %.1f s (%s); it keeps loading",
                    spec.name,
                    age,
                    state,
                )
                return False
            log.error("llama server %s did not become healthy; stopping it", spec.name)
            await self._terminate(spec.name)
            return False
        log.error("llama server %s was not healthy within %.1f s", spec.name, timeout_s)
        return False

    async def _accept(self, spec: LlamaServerSpec, admin: LlamaCppAdmin) -> bool:
        try:
            props = await admin.props()
        except LlamaAdminError as exc:
            log.warning("llama server %s: %s", spec.name, exc)
            return False
        if not props_match(props, alias=spec.alias, model=spec.model):
            log.error(
                "port of %s is taken by another server (alias %r, model %r); expected %r / %r",
                spec.name,
                props.get("model_alias"),
                props.get("model_path"),
                spec.alias,
                spec.model,
            )
            return False
        if not self._alive(spec.name) and spec.name not in self._adopted:
            self._adopted.add(spec.name)
            log.info("adopted running llama server %s at %s", spec.name, spec.base_url)
        if self._check_caps:
            check_tool_caps(props, server=spec.name)
        return True

    def _alive(self, server: str) -> bool:
        proc = self._procs.get(server)
        return proc is not None and proc.returncode is None

    async def _spawn(self, spec: LlamaServerSpec) -> None:
        _make_slot_dirs(spec.argv, spec.cwd)
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        env = {**os.environ, **spec.env} if spec.env else None
        out: IO[bytes] | int = subprocess.DEVNULL
        if spec.log_path is not None:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            out = spec.log_path.open("ab")
        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.argv,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT if spec.log_path is not None else subprocess.DEVNULL,
                cwd=spec.cwd,
                env=env,
                **kwargs,
            )
        finally:
            if not isinstance(out, int):
                out.close()
        self._procs[spec.name] = proc
        self._spawned_at[spec.name] = self._clock.now()
        self._adopted.discard(spec.name)
        log.info("started llama server %s (pid %s): %s", spec.name, proc.pid, " ".join(spec.argv))

    async def stop(self, server: str) -> None:
        """Stop a server this manager spawned (graceful, then kill). Adopted servers and
        ``mode="adopt"`` are left alone."""
        async with self._lock(server):
            if server in self._procs:
                await self._terminate(server)
            elif server in self._adopted:
                log.info("leaving adopted llama server %s running", server)

    async def _terminate(self, server: str) -> None:
        proc = self._procs.pop(server, None)
        self._spawned_at.pop(server, None)
        if proc is None or proc.returncode is not None:
            return
        graceful = True
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        except OSError as exc:  # e.g. no console to deliver CTRL_BREAK to
            log.warning("could not ask llama server %s to stop (%s); killing it", server, exc)
            graceful = False
        if graceful:
            try:
                async with deadline(self._graceful_s, what=f"{server} exit", clock=self._clock):
                    await proc.wait()
                return
            except DeadlineExceeded:
                log.warning("llama server %s ignored the stop request; killing it", server)
            except asyncio.CancelledError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()  # never leave an orphan holding VRAM and the port
                raise
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        try:
            async with deadline(self._graceful_s, what=f"{server} kill", clock=self._clock):
                await proc.wait()
        except DeadlineExceeded:
            log.error("llama server %s (pid %s) did not exit after kill", server, proc.pid)

    async def aclose(self, *, stop_servers: bool = True) -> None:
        """Stop the servers this manager spawned (unless ``stop_servers=False``) and close
        the HTTP client."""
        if stop_servers:
            for name in list(self._procs):
                await self.stop(name)
        if self._owns_http:
            await self._http.aclose()


class LauncherServerManager(_AdminManager):
    """``LocalServerManager`` that lets the launcher start and stop llama-server (§2.3).

    ``ensure_running`` returns at once when the server is already healthy; otherwise it calls
    ``POST <emergency_url>/llm/ensure/<server>?timeout_s=…`` and waits for ``/health``. The
    token is sent as ``Authorization: Bearer`` and ``X-Aivtube-Token``.
    """

    def __init__(
        self,
        emergency_url: str,
        token: str,
        admins: Mapping[str, LlamaCppAdmin],
        *,
        http: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
        check_caps: bool = True,
        poll_interval_s: float = 0.25,
        request_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(admins, clock=clock)
        self.emergency_url = emergency_url.rstrip("/")
        self._token = token
        self._owns_http = http is None
        self._http = http if http is not None else httpx2.AsyncClient(trust_env=False)
        self._check_caps = check_caps
        self._poll_s = poll_interval_s
        self._request_timeout_s = request_timeout_s

    def __repr__(self) -> str:
        return f"LauncherServerManager({self.emergency_url!r}, servers={sorted(self._admins)})"

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        *,
        token: str,
        http: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
        check_caps: bool = True,
    ) -> LauncherServerManager:
        """Admins for every ``cfg.llm.servers`` entry; the launcher at ``ports.emergency``."""
        admins = {
            name: LlamaCppAdmin(server_root_url(server), http, clock=clock)
            for name, server in cfg.llm.servers.items()
        }
        return cls(
            f"http://127.0.0.1:{cfg.ports.emergency}",
            token,
            admins,
            http=http,
            clock=clock,
            check_caps=check_caps,
        )

    async def _call(self, path: str, *, timeout_s: float, **kw: Any) -> bool:
        url = self.emergency_url + path
        headers = {"Authorization": f"Bearer {self._token}", "X-Aivtube-Token": self._token}
        try:
            async with deadline(timeout_s, what=f"launcher {path}", clock=self._clock):
                resp = await self._http.post(url, headers=headers, **kw)
                await resp.aread()
        except (DeadlineExceeded, httpx2.HTTPError) as exc:
            log.warning("launcher %s failed: %s", path, exc)
            return False
        if resp.status_code != 200:
            log.warning("launcher %s returned %s", path, resp.status_code)
            return False
        return True

    async def ensure_running(self, server: str, timeout_s: float) -> bool:
        admin = self._admins.get(server)
        if admin is None:
            return False
        async with self._lock(server):
            end = self._clock.now() + timeout_s
            if await admin.health() == "ok":
                return await self._accept(server, admin)
            asked = await self._call(
                f"/llm/ensure/{server}",
                params={"timeout_s": f"{timeout_s:g}"},
                timeout_s=timeout_s + 2.0,
            )
            if not asked:
                return False
            if not await self._wait_healthy(admin, end, self._poll_s):
                log.warning("llama server %s not healthy within %.1f s", server, timeout_s)
                return False
            return await self._accept(server, admin)

    async def _accept(self, server: str, admin: LlamaCppAdmin) -> bool:
        if not self._check_caps:
            return True
        try:
            props = await admin.props()
        except LlamaAdminError as exc:
            log.warning("llama server %s: %s", server, exc)
            return False
        check_tool_caps(props, server=server)
        return True

    async def stop(self, server: str) -> None:
        if server in self._admins:
            await self._call(f"/llm/stop/{server}", timeout_s=self._request_timeout_s)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
        for admin in self._admins.values():
            await admin.aclose()
