"""Launcher preflight (§2.6 step 1): refuse to start with a clear Thai + English hint.

Checks, each independent:

- not in Windows Session 0 (no CUDA context, no audio devices there);
- the configuration validates (``aivtube.ops.configcheck`` in a subprocess: pydantic is not
  imported into the launcher);
- the launcher's ports are free (panel, IPC bus, emergency; the SDK hub when games are on); a
  llama-server port may be taken only by a server that can be adopted;
- the model files of the servers that start now, and the VAD model, exist with the right size
  and (when cheap: small files, or a matching ``models/.verified.json`` stamp) sha256;
- the llama-server executable exists;
- free VRAM (``nvidia-smi``, optional: a warning only).

``preflight`` returns the fatal problems as printable strings (empty = go); ``run_preflight``
returns every :class:`Problem` including the warnings.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivtube.launcher.gpu import GpuInfo, query_gpus
from aivtube.launcher.heartbeat import port_open
from aivtube.launcher.llama import fetch_json, props_match
from aivtube.launcher.settings import LOOPBACK, LauncherSettings
from aivtube.launcher.win32 import IS_WINDOWS, session_id

__all__ = ["Problem", "format_problems", "port_in_use", "preflight", "run_preflight"]

log = logging.getLogger("aivtube.launcher.preflight")

EXIT_PREFLIGHT = 2
VRAM_MARGIN_MIB = 1024


@dataclass(frozen=True)
class Problem:
    key: str
    message_en: str
    message_th: str
    hint_en: str = ""
    hint_th: str = ""
    fatal: bool = True

    def __str__(self) -> str:
        lines = [f"{'✖' if self.fatal else '!'} {self.message_en}", f"  {self.message_th}"]
        if self.hint_en:
            lines.append(f"  hint: {self.hint_en}")
        if self.hint_th:
            lines.append(f"  วิธีแก้: {self.hint_th}")
        return "\n".join(lines)


def format_problems(problems: Sequence[Problem]) -> str:
    return "\n".join(str(p) for p in problems)


def port_in_use(port: int, host: str = LOOPBACK) -> bool:
    """Whether ``host:port`` is taken: something accepts connections, or we cannot bind it."""
    if port_open(host, port, 0.25):
        return True
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if not IS_WINDOWS:  # ignore TIME_WAIT leftovers; Linux still refuses a live listener
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        return True
    finally:
        sock.close()
    return False


ConfigRunner = Callable[[LauncherSettings, Mapping[str, Any]], list[dict[str, Any]]]


def run_config_check(
    settings: LauncherSettings, cli_overrides: Mapping[str, Any], *, timeout_s: float = 90.0
) -> list[dict[str, Any]]:
    """Validate the config in a subprocess (see ``aivtube.ops.configcheck``)."""
    argv = [
        sys.executable,
        "-m",
        "aivtube.ops.configcheck",
        "--root",
        str(settings.root),
        "--profile",
        settings.profile,
        "--overrides",
        json.dumps(dict(cli_overrides)),
    ]
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, cwd=settings.root,
            encoding="utf-8", errors="replace", check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"path": "config", "message_en": f"The config check could not run: {exc}",
                 "message_th": f"ตรวจไฟล์ตั้งค่าไม่ได้: {exc}", "hint": "", "hint_th": ""}]
    try:
        result = json.loads(done.stdout.strip().splitlines()[-1])
        errors = result.get("errors", [])
        return [e for e in errors if isinstance(e, dict)]
    except (ValueError, IndexError, AttributeError):
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
        return [{
            "path": "config",
            "message_en": f"The config check crashed (exit {done.returncode}): {' | '.join(tail)}",
            "message_th": f"การตรวจไฟล์ตั้งค่าล้มเหลว (exit {done.returncode})",
            "hint": "Run: aivtube doctor",
            "hint_th": "รันคำสั่ง: aivtube doctor",
        }]


def _manifest(settings: LauncherSettings) -> Any:
    try:
        from aivtube.ops.models import MANIFEST, ModelManifest

        return ModelManifest.load(settings.root / MANIFEST, root=settings.root)
    except Exception as exc:  # a broken manifest must not stop the preflight itself
        log.warning("cannot read the model manifest: %s", exc)
        return None


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _check_file(
    manifest: Any, key: str, path: Path, what: str, problems: list[Problem], *, fatal: bool
) -> None:
    """``path`` exists; when the manifest pins it, size (and cheap sha256) match."""
    entry = None
    if manifest is not None:
        for e in manifest.required_by(key):
            if not e.archive and _same(manifest.dest_path(e), path):
                entry = e
                break
    if not path.is_file():
        problems.append(Problem(
            f"model:{key}",
            f"{what} is missing: {path}",
            f"ไม่พบไฟล์ {what}: {path}",
            "Run: aivtube models pull (or aivtube setup)." + (
                " Use --profile light until the 30B has downloaded." if "30b" in key else ""
            ),
            "รันคำสั่ง: aivtube models pull (หรือ aivtube setup)" + (
                " ใช้ --profile light ระหว่างรอดาวน์โหลดโมเดล 30B" if "30b" in key else ""
            ),
            fatal=fatal,
        ))
        return
    if entry is None:
        return
    result = manifest.check(entry.name, full=False)
    if result.status in ("size", "sha"):
        problems.append(Problem(
            f"model:{key}",
            f"{what} is damaged ({result.detail}): {path}",
            f"ไฟล์ {what} เสียหาย ({result.detail}): {path}",
            f"Delete it and run: aivtube models pull {entry.name}",
            f"ลบไฟล์นี้แล้วรันคำสั่ง: aivtube models pull {entry.name}",
            fatal=fatal,
        ))


def run_preflight(
    settings: LauncherSettings,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    llama_servers: Sequence[str] | None = None,
    voice: bool | None = None,
    check_config: ConfigRunner | None = run_config_check,
    session: Callable[[], int | None] = session_id,
    in_use: Callable[[int], bool] = port_in_use,
    gpus: Callable[[], list[GpuInfo] | None] = query_gpus,
    props: Callable[[int], Mapping[str, Any] | None] | None = None,
    health: Callable[[int], int] | None = None,
) -> list[Problem]:
    """Every problem found (fatal and warnings). ``llama_servers`` are the servers the
    launcher will start now (default: ``settings.autostart_servers()``); ``voice`` whether
    the voice worker runs (default: ``settings.voice_worker``)."""
    problems: list[Problem] = []
    servers = list(settings.autostart_servers() if llama_servers is None else llama_servers)
    voice_on = settings.voice_worker if voice is None else voice

    if session() == 0:
        problems.append(Problem(
            "session0",
            "Running in Windows Session 0 (a service or an SSH login): no GPU or audio there.",
            "กำลังรันใน Session 0 ของ Windows (บริการหรือ SSH) ซึ่งใช้ GPU และเสียงไม่ได้",
            "Start run.bat from your own desktop login.",
            "เปิด run.bat จากหน้าจอเดสก์ท็อปของผู้ใช้เอง",
        ))

    if check_config is not None:
        for err in check_config(settings, dict(cli_overrides or {})):
            where = str(err.get("path") or "config")
            if err.get("source"):
                where += f" ({err['source']})"
            problems.append(Problem(
                f"config:{err.get('path')}",
                f"Config error at {where}: {err.get('message_en', '')}",
                f"ไฟล์ตั้งค่าผิดที่ {where}: {err.get('message_th', '')}",
                str(err.get("hint") or ""),
                str(err.get("hint_th") or ""),
            ))

    ports = {"panel": settings.ports["panel"], "bus": settings.ports["bus"],
             "emergency": settings.ports["emergency"]}
    if settings.games_enabled:
        ports["neuro_sdk"] = settings.ports["neuro_sdk"]
    for name, port in ports.items():
        if in_use(port):
            problems.append(Problem(
                f"port:{name}",
                f"Port {port} (ports.{name}) is already in use. Is aivtube already running?",
                f"พอร์ต {port} (ports.{name}) ถูกใช้อยู่แล้ว aivtube เปิดซ้อนอยู่หรือเปล่า",
                "Close the other aivtube window, or change the port in config/user.toml.",
                "ปิดหน้าต่าง aivtube อีกอัน หรือเปลี่ยนพอร์ตใน config/user.toml",
            ))

    get_props = props or (lambda port: _props(port))
    get_health = health or (lambda port: _health(port))
    manifest = _manifest(settings) if servers or voice_on else None
    for name in servers:
        server = settings.servers.get(name, {})
        port = int(server.get("port", 0))
        alias = str(server.get("alias", ""))
        model = str(server.get("model", ""))
        adoptable = False
        if port and in_use(port):
            found = get_props(port)
            adoptable = (
                found is not None
                and bool(server.get("adopt_existing", True))
                and props_match(found, alias=alias, model=model)
            )
            if (
                not adoptable
                and found is None
                and bool(server.get("adopt_existing", True))
                and get_health(port) == 503
            ):
                problems.append(Problem(
                    f"port:llm.{name}",
                    f"A llama-server on port {port} (llm.servers.{name}) is still loading; it is "
                    "adopted if its alias and model match, otherwise the LLM stays red.",
                    f"มี llama-server ที่พอร์ต {port} กำลังโหลดโมเดลอยู่ ถ้าตรงกับที่ตั้งค่าไว้จะใช้ตัวนั้นเลย",
                    fatal=False,
                ))
                continue  # the controller decides once it has loaded
            if not adoptable:
                problems.append(Problem(
                    f"port:llm.{name}",
                    f"Port {port} (llm.servers.{name}) is taken by another program"
                    + (f" (a llama-server with alias {found.get('model_alias')!r})" if found else "")
                    + ".",
                    f"พอร์ต {port} (llm.servers.{name}) ถูกโปรแกรมอื่นใช้อยู่",
                    f"Close it, or give llm.servers.{name}.port another port in config/user.toml.",
                    f"ปิดโปรแกรมนั้น หรือเปลี่ยน llm.servers.{name}.port ใน config/user.toml",
                ))
        if adoptable:
            continue  # a running, matching server: nothing to start, nothing to check
        exe = settings.resolve(str(server.get("exe", "vendor/llama.cpp/llama-server.exe")))
        if not exe.is_file():
            problems.append(Problem(
                f"exe:llm.{name}",
                f"llama-server is missing: {exe}",
                f"ไม่พบโปรแกรม llama-server: {exe}",
                "Run: aivtube setup (it downloads llama.cpp into vendor/).",
                "รันคำสั่ง: aivtube setup (จะดาวน์โหลด llama.cpp ไว้ใน vendor/)",
            ))
        _check_file(manifest, f"llm.servers.{name}", settings.resolve(model),
                    f"the model of llm.servers.{name}", problems, fatal=True)

    raw = settings.raw
    vad = raw.get("vad", {}) if isinstance(raw.get("vad"), Mapping) else {}
    if voice_on and not settings.fakes and vad.get("backend", "silero_ort") != "energy":
        _check_file(manifest, "vad.model", settings.resolve(str(vad.get("model", ""))),
                    "the Silero VAD model", problems, fatal=True)
        stt = raw.get("stt", {}) if isinstance(raw.get("stt"), Mapping) else {}
        chain = stt.get("chain", []) if isinstance(stt.get("chain"), list) else []
        backends = stt.get("backends", {}) if isinstance(stt.get("backends"), Mapping) else {}
        rt = backends.get("typhoon_rt", {}) if isinstance(backends, Mapping) else {}
        if "typhoon_rt" in chain and isinstance(rt, Mapping) and rt.get("enabled", True):
            model_dir = settings.resolve(str(rt.get("model_dir", "")))
            if not (model_dir / "tokens.txt").is_file():
                problems.append(Problem(
                    "model:stt.typhoon_rt",
                    f"Typhoon ASR Realtime is not downloaded ({model_dir}); STT uses the next "
                    "backend in the chain.",
                    f"ยังไม่มีโมเดล Typhoon ASR Realtime ({model_dir}) จะใช้ตัวถอดเสียงตัวถัดไปแทน",
                    "Run: aivtube models pull",
                    "รันคำสั่ง: aivtube models pull",
                    fatal=False,
                ))

    if servers:
        _vram(settings, servers, gpus, problems)
    return problems


def _props(port: int) -> Mapping[str, Any] | None:
    status, body = fetch_json(f"http://{LOOPBACK}:{port}/props", 1.5)
    return body if status == 200 and isinstance(body, Mapping) else None


def _health(port: int) -> int:
    """HTTP status of ``GET /health`` (503 while llama-server loads; 0 = no answer)."""
    return fetch_json(f"http://{LOOPBACK}:{port}/health", 1.5)[0]


def _vram(
    settings: LauncherSettings,
    servers: Sequence[str],
    gpus: Callable[[], list[GpuInfo] | None],
    problems: list[Problem],
) -> None:
    try:
        found = gpus()
    except Exception:
        found = None
    if not found:
        problems.append(Problem(
            "vram",
            "nvidia-smi is not available: free VRAM was not checked.",
            "ไม่พบ nvidia-smi จึงไม่ได้ตรวจหน่วยความจำการ์ดจอ",
            "Install or update the NVIDIA driver (R580 or newer).",
            "ติดตั้งหรืออัปเดตไดรเวอร์ NVIDIA (R580 ขึ้นไป)",
            fatal=False,
        ))
        return
    free = found[0].get("free_mib")
    fit = [
        int(settings.servers.get(n, {}).get("fit_target_mib", 3584))
        for n in servers
        if settings.servers.get(n, {}).get("placement", "fit") == "fit"
    ]
    need = (max(fit) if fit else 0) + VRAM_MARGIN_MIB
    if isinstance(free, float) and free < need:
        problems.append(Problem(
            "vram",
            f"Only {free:.0f} MiB of VRAM is free (want at least {need} MiB before loading).",
            f"หน่วยความจำการ์ดจอเหลือเพียง {free:.0f} MiB (ควรมีอย่างน้อย {need} MiB)",
            "Close GPU-heavy programs, or use --profile light / gaming.",
            "ปิดโปรแกรมที่ใช้การ์ดจอหนัก ๆ หรือใช้ --profile light / gaming",
            fatal=False,
        ))


def preflight(settings: LauncherSettings, **kw: Any) -> list[str]:
    """The fatal problems as printable Thai + English messages (empty: go ahead)."""
    return [str(p) for p in run_preflight(settings, **kw) if p.fatal]
