"""``aivtube run``: the launcher process (P0, §2.1, §2.6, §2.9, §2.11).

Order of work: read the raw config → preflight (exit 2 with a hint on any problem) → Job
Object → emergency endpoint → GPU monitor and keep-awake → llama-server (adopt or start, in the
background) → core and voice worker (in parallel) → core heartbeat → console keys. It then
waits for ``Q``, Ctrl+C, a termination signal, or the core exiting (0 = quit, 3 = restart all).

Exit codes: 0 normal quit, 2 preflight or configuration problem, 3 restart requested (``run.bat``
loops on it), 1 internal error.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from aivtube.config.errors import ConfigError
from aivtube.config.layers import find_root
from aivtube.launcher import childside
from aivtube.launcher.console import ConsoleKeys, help_text
from aivtube.launcher.emergency import EmergencyServer, notify_freeze
from aivtube.launcher.gpu import GpuMonitor
from aivtube.launcher.heartbeat import CoreHeartbeat
from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.llama import LlamaServerController
from aivtube.launcher.logs import setup_launcher_logging
from aivtube.launcher.preflight import format_problems, run_preflight
from aivtube.launcher.proc import child_env
from aivtube.launcher.settings import LauncherSettings, Tokens, load_settings, load_tokens
from aivtube.launcher.supervisor import ProcessSpec, Supervisor
from aivtube.launcher.win32 import IS_WINDOWS, keep_awake

__all__ = [
    "CORE_SUBCOMMAND",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_PREFLIGHT",
    "EXIT_RESTART",
    "VOICE_SUBCOMMAND",
    "Launcher",
    "build_parser",
    "main",
]

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_PREFLIGHT = 2
EXIT_RESTART = 3
CORE_SUBCOMMAND = "_core"
VOICE_SUBCOMMAND = "_voice"
HEARTBEAT_PATH = "/healthz"  # the panel's token-free liveness route

log = logging.getLogger("aivtube.launcher")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aivtube run",
        description="Start aivtube: llama-server, the core and the voice worker, supervised.",
    )
    p.add_argument("--profile", default=None, help="stream | light | gaming | text | offline | ci")
    p.add_argument("--text", action="store_true", help="text console instead of the microphone")
    p.add_argument("--speak", action="store_true", help="with --text: still speak the replies")
    p.add_argument("--safe", action="store_true", help="no network, no GPU, no audio (I6)")
    p.add_argument("--fake-llm", action="store_true", help="fake LLM/TTS/STT (app.fakes)")
    p.add_argument("--root", type=Path, default=None, help="the AI_Vtube folder")
    p.add_argument("--no-preflight", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-console", action="store_true", help="do not read console keys")
    p.add_argument("--no-llm", action="store_true", help="do not start or adopt llama-server")
    return p


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Config overrides implied by the ``run`` flags (the children derive the same)."""
    out: dict[str, Any] = {}
    if args.fake_llm:
        out["app.fakes"] = True
    if args.safe or (args.text and not args.speak):
        out["app.voice_worker"] = False
    return out


def forwarded_args(args: argparse.Namespace) -> list[str]:
    """The ``run`` flags passed on to ``_core`` and ``_voice``."""
    out: list[str] = []
    if args.profile:
        out += ["--profile", args.profile]
    for flag in ("text", "speak", "safe", "fake_llm"):
        if getattr(args, flag):
            out.append("--" + flag.replace("_", "-"))
    return out


def _say(out: IO[str], text: str) -> None:
    try:
        out.write(text + "\n")
        out.flush()
    except UnicodeEncodeError:  # a console code page without Thai
        out.write(text.encode("ascii", "replace").decode("ascii") + "\n")
        out.flush()
    except (OSError, ValueError):
        pass


class Launcher:
    """One launcher run (see the module docstring). ``run()`` blocks until quit."""

    def __init__(
        self,
        settings: LauncherSettings,
        tokens: Tokens,
        args: argparse.Namespace,
        *,
        core_argv: Sequence[str] | None = None,
        voice_argv: Sequence[str] | None = None,
        llama_prefix: Sequence[str] | None = None,
        preflight_kw: Mapping[str, Any] | None = None,
        out: IO[str] | None = None,
        console: bool | None = None,
        heartbeat_grace_s: float = 90.0,
        backoff: tuple[float, float] | None = None,
        install_signals: bool = True,
        log_console: bool = True,
    ) -> None:
        self.settings = settings
        self.tokens = tokens
        self.args = args
        self.out: IO[str] = out if out is not None else sys.stdout
        self._core_argv = list(core_argv) if core_argv else None
        self._voice_argv = list(voice_argv) if voice_argv else None
        self._llama_prefix = list(llama_prefix) if llama_prefix else None
        self._preflight_kw = dict(preflight_kw or {})
        self._console = console
        self._grace = heartbeat_grace_s
        self._backoff = backoff or settings.restart_backoff_s
        self._install_signals = install_signals
        self._log_console = log_console
        self._quit = threading.Event()
        self._exit_code = EXIT_OK
        self._code_lock = threading.Lock()
        self.supervisor: Supervisor | None = None
        self.emergency: EmergencyServer | None = None
        self.llama: dict[str, LlamaServerController] = {}
        self.gpu: GpuMonitor | None = None
        self.heartbeat: CoreHeartbeat | None = None
        self.keys: ConsoleKeys | None = None
        self.job: JobObject | None = None
        self.started = threading.Event()

    # -- decisions ---------------------------------------------------------------------------

    @property
    def voice_enabled(self) -> bool:
        return self.settings.voice_worker and not self.args.safe

    def llama_to_start(self) -> list[str]:
        if self.args.no_llm or self.args.safe:
            return []
        return self.settings.autostart_servers()

    def request_quit(self, code: int = EXIT_OK) -> None:
        with self._code_lock:
            if not self._quit.is_set():
                self._exit_code = code
        self._quit.set()

    # -- run ---------------------------------------------------------------------------------

    def run(self) -> int:
        s = self.settings
        setup_launcher_logging(s.log_dir, secrets=self.tokens.values(), console=self._log_console)
        log.info("launcher starting (profile %s, root %s)", s.profile, s.root)
        if not self.args.no_preflight:
            problems = run_preflight(
                s,
                cli_overrides=cli_overrides(self.args),
                llama_servers=self.llama_to_start(),
                voice=self.voice_enabled,
                **self._preflight_kw,
            )
            warnings = [p for p in problems if not p.fatal]
            fatal = [p for p in problems if p.fatal]
            if warnings:
                _say(self.out, format_problems(warnings))
            if fatal:
                _say(self.out, "\nPreflight failed / ตรวจความพร้อมไม่ผ่าน:\n" + format_problems(fatal))
                log.error("preflight failed: %s", "; ".join(p.key for p in fatal))
                return EXIT_PREFLIGHT
        try:
            return self._run_supervised()
        finally:
            self._shutdown()

    def _run_supervised(self) -> int:
        s = self.settings
        self.job = JobObject()
        self._build_llama()
        self.supervisor = Supervisor(
            self._specs(), self.job, log.getChild("supervisor"), on_event=self._on_child_event
        )
        self.gpu = GpuMonitor(
            interval_s=s.gpu_poll_s, alarm_mib=s.vram_alarm_mib, on_alarm=self._on_vram_alarm
        )
        self.emergency = EmergencyServer(
            "127.0.0.1",
            s.ports["emergency"],
            self.tokens.emergency,
            self.supervisor,
            self.llama,
            panel_port=s.ports["panel"],
            freeze=self._freeze,
            gpu=self.gpu.latest,
            extra_status=lambda: {"profile": s.profile},
        )
        try:
            self.emergency.start()
        except OSError as exc:
            _say(self.out, f"✖ Cannot open the emergency endpoint on port {s.ports['emergency']}: "
                           f"{exc}\n  เปิดพอร์ตฉุกเฉิน {s.ports['emergency']} ไม่ได้ (ถูกใช้อยู่หรือเปล่า)")
            return EXIT_PREFLIGHT
        if not self.args.safe:  # without nvidia-smi it idles; --safe means no GPU at all
            self.gpu.start()
        if s.keep_awake:
            keep_awake(True)
        for name in self.llama_to_start():
            self.llama[name].want()
        self._install_signal_handlers()
        self.supervisor.start()
        self.heartbeat = CoreHeartbeat(
            self.supervisor,
            "core",
            s.panel_url + HEARTBEAT_PATH,
            timeout_s=s.core_timeout_s,
            startup_grace_s=self._grace,
            dump_request=self._dump_request("core"),
        )
        self.heartbeat.start()
        if self._console_wanted():
            self.keys = ConsoleKeys(self._key_handlers())
            self.keys.start()
        self._banner()
        self.started.set()
        while not self._quit.wait(0.2):
            pass
        with self._code_lock:
            return self._exit_code

    def _shutdown(self) -> None:
        """Stop everything; one failing step never skips the others (§2.9 order)."""

        def step(what: str, fn: Callable[[], object]) -> None:
            try:
                fn()
            except Exception:
                log.exception("shutdown: %s failed", what)

        if self.keys is not None:
            step("console keys", self.keys.stop)
        if self.heartbeat is not None:
            step("heartbeat", self.heartbeat.stop)
        if self.supervisor is not None:
            _say(self.out, "Stopping… / กำลังปิด…")
            sup = self.supervisor
            step("children", lambda: sup.stop(self.settings.graceful_timeout_s))
        for name, ctl in self.llama.items():
            step(f"llama-server {name}", functools.partial(self._close_llama, name, ctl))
        if self.emergency is not None:
            step("emergency endpoint", self.emergency.stop)
        if self.gpu is not None:
            step("gpu monitor", self.gpu.stop)
        if self.settings.keep_awake:
            step("keep-awake", lambda: keep_awake(False))
        if self.job is not None:
            step("job object", self.job.close)  # kills anything still in the job
        log.info("launcher stopped (exit %d)", self._exit_code)

    def _close_llama(self, name: str, ctl: LlamaServerController) -> None:
        adopted = ctl.status()["phase"] == "adopted"
        ctl.close()  # ours stop; an adopted server keeps running (§2.9 step 8)
        if adopted and not self.settings.keep_llm:
            log.warning("llama-server %s was adopted, so it keeps running (launcher.keep_llm "
                        "= false cannot stop a process the launcher did not start)", name)

    # -- building ----------------------------------------------------------------------------

    def _dump_request(self, name: str) -> Path:
        return self.settings.state_dir / f"dump_request.{name}"

    def _child_env(self, name: str) -> dict[str, str]:
        s = self.settings
        return child_env({
            childside.BUS_TOKEN_ENV: self.tokens.bus,
            childside.EMERGENCY_TOKEN_ENV: self.tokens.emergency,
            childside.PANEL_TOKEN_ENV: self.tokens.panel,
            childside.LAUNCHER_URL_ENV: s.emergency_url,
            childside.DUMP_REQUEST_ENV: str(self._dump_request(name)),
            childside.ROOT_ENV: str(s.root),
            childside.LAUNCHED_ENV: "1",
        })

    def _specs(self) -> list[ProcessSpec]:
        s = self.settings
        fwd = forwarded_args(self.args)
        breaker = (s.crash_loop_restarts, s.crash_loop_window_s)
        core = ProcessSpec(
            "core",
            self._core_argv or [sys.executable, "-m", "aivtube", CORE_SUBCOMMAND, *fwd],
            self._child_env("core"),
            s.root,
            restart="always",
            backoff=self._backoff,
            breaker=breaker,
            final_exit_codes=frozenset({EXIT_OK, EXIT_RESTART}),
            inherit_stdin=bool(self.args.text),
            stop_timeout_s=s.graceful_timeout_s + 3.0,  # it lets the utterance finish (§2.9)
        )
        specs = [core]
        if self.voice_enabled:
            specs.append(ProcessSpec(
                "voice",
                self._voice_argv or [sys.executable, "-m", "aivtube", VOICE_SUBCOMMAND, *fwd],
                self._child_env("voice"),
                s.root,
                restart="always",
                backoff=self._backoff,
                breaker=breaker,
                priority="above_normal",
            ))
        return specs

    def _build_llama(self) -> None:
        s = self.settings
        for name, cfg in s.servers.items():
            self.llama[name] = LlamaServerController(
                name,
                cfg,
                log.getChild(f"llama.{name}"),
                root=s.root,
                state_dir=s.state_dir,
                log_dir=s.log_dir,
                job=self.job,
                health_interval_s=s.llm_health_interval_s,
                health_failures=s.llm_health_failures,
                load_timeout_s=s.llm_load_timeout_s,
                max_load_failures=s.llm_max_load_failures,
                graceful_timeout_s=s.graceful_timeout_s,
                backoff=self._backoff,
                breaker=(s.crash_loop_restarts, s.crash_loop_window_s),
                command_prefix=self._llama_prefix,
            )

    # -- callbacks ---------------------------------------------------------------------------

    def _on_child_event(self, name: str, kind: str, info: Mapping[str, Any]) -> None:
        if kind == "failed":
            _say(self.out, f"✖ {name} FAILED: {info.get('detail', '')}\n"
                           f"  {name} หยุดทำงานซ้ำหลายครั้ง กด R เพื่อลองใหม่ / press R to retry")
        elif kind == "exited" and name == "core":
            code = info.get("exit_code")
            if code == EXIT_RESTART:
                log.info("the core asked for a full restart")
                self.request_quit(EXIT_RESTART)
            elif code == EXIT_OK:
                log.info("the core exited normally; quitting")
                self.request_quit(EXIT_OK)

    def _on_vram_alarm(self, gpu: Mapping[str, Any]) -> None:
        _say(self.out, f"! VRAM low: {gpu.get('free_mib')} MiB free / หน่วยความจำการ์ดจอใกล้เต็ม")

    def _freeze(self, reason: str) -> bool:
        return notify_freeze(self.settings.panel_url, self.tokens.panel, reason=reason)

    def _console_wanted(self) -> bool:
        if self._console is not None:
            return self._console
        if self.args.no_console or self.args.text:
            return False
        stdin = sys.stdin
        return IS_WINDOWS or (stdin is not None and stdin.isatty())

    def _key_handlers(self) -> dict[str, Callable[[], None]]:
        assert self.emergency is not None and self.supervisor is not None
        emergency, sup = self.emergency, self.supervisor

        def hardkill() -> None:
            result = emergency.hardkill("console")
            _say(self.out, f"HARD KILL: voice down in {result['elapsed_ms']:.0f} ms "
                           "(กด A เพื่อเปิดเสียงกลับ / press A to rearm)")

        def rearm() -> None:
            ok = emergency.rearm()["ok"]
            _say(self.out, "voice rearmed / เปิดเสียงกลับแล้ว" if ok else "voice was not held")

        def freeze() -> None:
            threading.Thread(target=self._freeze_and_report, daemon=True).start()

        def retry() -> None:
            names = sup.retry_failed()
            for name, ctl in self.llama.items():
                if ctl.status()["phase"] == "failed" and ctl.desired:
                    ctl.restart()
                    names.append(name)
            _say(self.out, f"retrying: {', '.join(names) or 'nothing FAILED'}")

        return {
            "k": hardkill,
            "a": rearm,
            "f": freeze,
            "r": retry,
            "s": lambda: _say(self.out, self.status_text()),
            "q": lambda: self.request_quit(EXIT_OK),
            "h": lambda: _say(self.out, help_text()),
        }

    def _freeze_and_report(self) -> None:
        ok = self._freeze("console")
        _say(self.out, "FREEZE sent / สั่งหยุดแล้ว" if ok else
             "✖ the core did not answer FREEZE (use K) / สมองไม่ตอบ ใช้ K แทน")

    def status_text(self) -> str:
        lines = [f"profile {self.settings.profile}"]
        if self.supervisor is not None:
            for name, st in self.supervisor.status().items():
                lines.append(
                    f"  {name:8s} {st['state']:9s} pid={st['pid']} restarts={st['restarts']} "
                    f"{st['detail']}"
                )
        for name, ctl in self.llama.items():
            st = ctl.status()
            if ctl.desired or st["phase"] not in ("idle", "stopped"):
                lines.append(f"  {name:8s} {st['phase']:9s} pid={st['pid']} {st['detail']}")
        if self.gpu is not None and self.gpu.latest():
            g = self.gpu.latest()
            lines.append(f"  gpu      free {g.get('free_mib')} MiB / {g.get('total_mib')} MiB")
        return "\n".join(lines)

    def _banner(self) -> None:
        s = self.settings
        text = [
            f"aivtube running (profile {s.profile}) / ไพลินพร้อมแล้ว",
            f"  panel      {s.panel_url}/",
            f"  emergency  {s.emergency_url}/hardkill (token in data/state/tokens.json)",
        ]
        if self.keys is not None:
            text.append(help_text())
        _say(self.out, "\n".join(text))

    def _install_signal_handlers(self) -> None:
        if not self._install_signals or threading.current_thread() is not threading.main_thread():
            return

        def handler(signum: int, frame: Any) -> None:
            if self._quit.is_set():
                return
            log.info("signal %s: quitting", signum)
            self.request_quit(EXIT_OK)

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(OSError, ValueError):
                    signal.signal(sig, handler)


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            with contextlib.suppress(OSError, ValueError, AttributeError):
                stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``aivtube run`` (the CLI hands over its arguments)."""
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        root = Path(args.root).resolve() if args.root else find_root()
        settings = load_settings(root, profile=args.profile, cli_overrides=cli_overrides(args))
    except ConfigError as exc:
        _say(sys.stdout, str(exc))
        return EXIT_PREFLIGHT
    tokens = load_tokens(settings.state_dir)
    started = time.perf_counter()
    try:
        code = Launcher(settings, tokens, args).run()
    except KeyboardInterrupt:
        code = EXIT_OK
    except Exception:
        log.exception("launcher crashed")
        return EXIT_INTERNAL
    log.info("ran for %.0f s", time.perf_counter() - started)
    return code


if __name__ == "__main__":
    sys.exit(main())
