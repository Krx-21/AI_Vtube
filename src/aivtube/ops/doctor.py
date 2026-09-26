"""``aivtube doctor``: diagnostics with ✔/✖ and Thai + English fix hints (§2.2, §2.6, §9).

Every check is independent and never raises: a check that crashes is itself reported. Live
checks (``live=True``) talk to running programs: llama-server ``/props`` tool caps, VTube
Studio discovery, ``nvidia-smi`` and, when asked, a short TTS time-to-first-audio sample.
Everything that touches the machine goes through :class:`DoctorEnv`, so tests fake it all.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import platform
import shutil
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig, Secrets
    from aivtube.contracts.voice import AudioBackend

__all__ = ["Check", "DoctorEnv", "format_report", "run_doctor"]

Level = Literal["info", "warn", "error"]

VTS_PORTS = range(8001, 8010)
FREE_DISK_GB_30B = 25.0
FREE_DISK_GB = 5.0

EXTRAS: Mapping[str, tuple[str, ...]] = {
    "voice": ("sounddevice", "soxr", "onnxruntime", "sherpa_onnx", "edge_tts", "av"),
    "aec": ("livekit",),
    "tts-azure": ("azure.cognitiveservices.speech",),
    "stt-pythaiasr": ("pythaiasr",),
}


@dataclass(frozen=True)
class Check:
    """One doctor finding."""

    name: str
    ok: bool
    level: Level
    detail: str
    hint_en: str = ""
    hint_th: str = ""


def _ok(name: str, detail: str) -> Check:
    return Check(name, True, "info", detail)


def _bad(name: str, level: Level, detail: str, hint_en: str = "", hint_th: str = "") -> Check:
    return Check(name, False, level, detail, hint_en, hint_th)


def _default_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _default_http_json(url: str, timeout_s: float) -> tuple[int, Any]:
    from aivtube.launcher.llama import fetch_json

    return fetch_json(url, timeout_s)


def _default_port_in_use(port: int) -> bool:
    from aivtube.launcher.preflight import port_in_use

    return port_in_use(port)


def _default_gpus() -> list[dict[str, Any]] | None:
    from aivtube.launcher.gpu import query_gpus

    return query_gpus()


def _default_session() -> int | None:
    from aivtube.launcher.win32 import session_id

    return session_id()


def _default_vts(timeout_s: float) -> list[dict[str, Any]]:
    from aivtube.avatar.discovery import discover_vts

    return asyncio.run(discover_vts(timeout_s))


def _default_audio_backend() -> AudioBackend:
    from aivtube.voice.audio_io import default_backend

    return default_backend()


@dataclass
class DoctorEnv:
    """The machine as the doctor sees it; every field can be replaced in tests."""

    find_spec: Callable[[str], bool] = _default_spec
    audio_backend: Callable[[], AudioBackend] = _default_audio_backend
    port_in_use: Callable[[int], bool] = _default_port_in_use
    http_json: Callable[[str, float], tuple[int, Any]] = _default_http_json
    gpus: Callable[[], list[dict[str, Any]] | None] = _default_gpus
    session_id: Callable[[], int | None] = _default_session
    discover_vts: Callable[[float], list[dict[str, Any]]] = _default_vts
    disk_free_gb: Callable[[Path], float] = field(
        default=lambda p: shutil.disk_usage(p).free / 2**30
    )
    secrets: Secrets | None = None
    tts_sample: Callable[[AppConfig], Mapping[str, Any]] | None = None
    python: tuple[int, int] = field(default_factory=lambda: sys.version_info[:2])


def run_doctor(
    cfg: AppConfig,
    *,
    live: bool = True,
    env: DoctorEnv | None = None,
    tts_sample: bool = False,
) -> list[Check]:
    """Every check (see the module docstring)."""
    env = env or DoctorEnv()
    secrets = env.secrets
    if secrets is None:
        try:
            from aivtube.config import load_secrets

            secrets = load_secrets(cfg.root)
        except Exception:
            secrets = None
    checks: list[Callable[[], Iterable[Check]]] = [
        lambda: _python(env),
        lambda: _extras(cfg, env),
        lambda: _session(env),
        lambda: _ports(cfg, env),
        lambda: _audio(cfg, env),
        lambda: _models(cfg),
        lambda: _llama_exe(cfg),
        lambda: _fts5(),
        lambda: _disk(cfg, env),
        lambda: _clock(),
        lambda: _secrets(cfg, secrets),
        lambda: _licences(cfg),
        lambda: _consent(cfg),
    ]
    if live:
        checks += [lambda: _llama_live(cfg, env), lambda: _vts(cfg, env), lambda: _gpu(cfg, env)]
        if tts_sample:
            checks.append(lambda: _tts(cfg, env))
    out: list[Check] = []
    for check in checks:
        try:
            out.extend(check())
        except Exception as exc:  # a broken check must not hide the others
            out.append(_bad("doctor", "warn", f"a check crashed: {type(exc).__name__}: {exc}"))
    return out


def format_report(checks: Sequence[Check]) -> str:
    """✔ / ! / ✖ lines, with the hints under the failures and a summary line."""
    lines: list[str] = []
    for c in checks:
        mark = "✔" if c.ok else ("✖" if c.level == "error" else "!")
        lines.append(f"{mark} {c.name}: {c.detail}")
        if not c.ok:
            if c.hint_en:
                lines.append(f"    hint: {c.hint_en}")
            if c.hint_th:
                lines.append(f"    วิธีแก้: {c.hint_th}")
    errors = sum(1 for c in checks if not c.ok and c.level == "error")
    warns = sum(1 for c in checks if not c.ok and c.level == "warn")
    lines.append(
        f"\n{errors} error(s), {warns} warning(s) / ข้อผิดพลาด {errors} รายการ คำเตือน {warns} รายการ"
    )
    return "\n".join(lines)


# --- checks -----------------------------------------------------------------------------------


def _python(env: DoctorEnv) -> Iterable[Check]:
    major, minor = env.python
    detail = f"Python {major}.{minor} on {platform.system()} {platform.release()}"
    if (major, minor) < (3, 11):
        yield _bad("python", "error", detail, "Python 3.11 or newer is required (3.12 recommended).",
                   "ต้องใช้ Python 3.11 ขึ้นไป (แนะนำ 3.12) รัน setup.ps1 ใหม่")
    else:
        yield _ok("python", detail)


def _extras(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    needed = {"voice": cfg.app.voice_worker and not cfg.app.fakes}
    identities = cfg.tts.identity_chain
    backends = {b for i in identities if i in cfg.tts.identities
                for b in cfg.tts.identities[i].backends}
    needed["tts-azure"] = "azure" in backends and cfg.tts.backends.get("azure") is not None
    needed["aec"] = cfg.app.voice_worker and not cfg.audio.headphones
    needed["stt-pythaiasr"] = "pythaiasr" in cfg.stt.chain
    for extra, modules in EXTRAS.items():
        missing = [m for m in modules if not env.find_spec(m)]
        if not missing:
            yield _ok(f"extra {extra}", "installed")
        elif needed.get(extra):
            level: Level = "error" if extra == "voice" else "warn"
            yield _bad(f"extra {extra}", level, f"missing: {', '.join(missing)}",
                       f"Run: uv sync --frozen --extra {extra}  (or --extra pc)",
                       f"รันคำสั่ง: uv sync --frozen --extra {extra} (หรือ --extra pc)")
        else:
            yield Check(f"extra {extra}", True, "info", f"not installed (not needed): {', '.join(missing)}")


def _session(env: DoctorEnv) -> Iterable[Check]:
    sid = env.session_id()
    if sid == 0:
        yield _bad("session", "error", "running in Windows Session 0 (service/SSH)",
                   "Run from your own desktop login: CUDA and audio do not work in Session 0.",
                   "รันจากหน้าจอเดสก์ท็อปของผู้ใช้เอง เพราะ Session 0 ใช้ GPU และเสียงไม่ได้")
    else:
        yield _ok("session", "interactive session" if sid is not None else "not Windows")


def _ports(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    for key, port in cfg.port_items():
        if port in VTS_PORTS:
            yield _bad(f"port {port}", "error", f"{key} uses {port}, reserved for VTube Studio",
                       "Use another port (see ARCHITECTURE §2.2).",
                       "พอร์ต 8001–8009 สงวนไว้ให้ VTube Studio ให้เปลี่ยนไปใช้พอร์ตอื่น")
        elif key.startswith("ports.browser_renderer"):
            continue  # M6, reserved only
        elif env.port_in_use(port):
            yield _bad(f"port {port}", "warn", f"{key} ({port}) is in use",
                       "Is aivtube already running? Otherwise close the program using it, or "
                       "change the port in config/user.toml.",
                       "aivtube เปิดอยู่แล้วหรือเปล่า ถ้าไม่ใช่ให้ปิดโปรแกรมที่ใช้พอร์ตนี้ "
                       "หรือเปลี่ยนพอร์ตใน config/user.toml")
        else:
            yield _ok(f"port {port}", f"{key} free")


def _audio(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    if not cfg.app.voice_worker:
        yield _ok("audio", "voice worker off (text profile)")
        return
    try:
        from aivtube.voice.audio_io import WASAPI, list_devices, resolve_device

        backend = env.audio_backend()
        devices = list_devices(backend)
    except (OSError, ImportError) as exc:
        level: Level = "error" if sys.platform == "win32" and not cfg.app.fakes else "warn"
        yield _bad("audio", level, f"PortAudio is not available: {exc}",
                   "Install the voice extra: uv sync --frozen --extra voice",
                   "ติดตั้งส่วนเสียง: uv sync --frozen --extra voice")
        return
    wasapi = [d for d in devices if d.hostapi == WASAPI]
    listed = wasapi or devices
    outs = sorted({d.name for d in listed if d.max_out > 0})
    ins = sorted({d.name for d in listed if d.max_in > 0})
    api = "WASAPI" if wasapi else "all host APIs"
    yield _ok("audio devices", f"{api}: outputs {outs}; inputs {ins}")
    pairs: list[tuple[Literal["input", "output"], str]] = [
        ("output", cfg.audio.output_device),
        ("input", cfg.audio.input_device),
    ]
    for kind, query in pairs:
        try:
            index = resolve_device(query or None, kind, backend)
            yield _ok(f"audio {kind}", f"{query or '(default)'} → {devices[index].name}")
        except LookupError:
            names = outs if kind == "output" else ins
            yield _bad(f"audio {kind}", "error", f"no {kind} device matches {query!r}",
                       f"Set audio.{kind}_device to part of one of: {', '.join(names)}",
                       f"ตั้ง audio.{kind}_device ให้เป็นส่วนหนึ่งของชื่ออุปกรณ์: {', '.join(names)}")
    if cfg.audio.mirror_output_device:
        try:
            resolve_device(cfg.audio.mirror_output_device, "output", backend)
            yield _ok("audio mirror", cfg.audio.mirror_output_device)
        except LookupError:
            yield _bad("audio mirror", "warn",
                       f"no output matches {cfg.audio.mirror_output_device!r}",
                       "Install VB-CABLE or clear audio.mirror_output_device.",
                       "ติดตั้ง VB-CABLE หรือลบค่า audio.mirror_output_device")


def _manifest(cfg: AppConfig) -> Any:
    from aivtube.ops.models import MANIFEST, ModelManifest

    return ModelManifest.load(cfg.root / MANIFEST, root=cfg.root)


def _chain_servers(cfg: AppConfig) -> list[str]:
    out: list[str] = []
    for name in cfg.llm.effective_chain():
        prov = cfg.llm.providers.get(name)
        if prov is not None and prov.server and prov.server not in out:
            out.append(prov.server)
    return out


def _models(cfg: AppConfig) -> Iterable[Check]:
    manifest = _manifest(cfg)
    wanted: dict[str, Level] = {}
    if cfg.app.voice_worker and not cfg.app.fakes:
        for e in manifest.required_by("vad.model"):
            wanted[e.name] = "error"
        if "typhoon_rt" in cfg.stt.chain:
            for e in manifest.required_by("stt.backends.typhoon_rt"):
                wanted.setdefault(e.name, "warn")
    if not cfg.app.fakes:
        for server in _chain_servers(cfg):
            level: Level = "error" if cfg.llm.servers[server].autostart == "always" else "warn"
            for e in manifest.required_by(f"llm.servers.{server}"):
                wanted[e.name] = level
    for name, level in wanted.items():
        r = manifest.check(name, full=False)
        entry = manifest.entry(name)
        if r.status == "ok":
            yield _ok(f"model {name}", f"{entry.dest} ({r.detail})")
        elif r.status == "unpinned":
            yield Check(f"model {name}", True, "info", f"{entry.dest}: {r.detail}")
        elif r.status in ("size", "sha"):
            yield _bad(f"model {name}", "error", f"{entry.dest} is damaged: {r.detail}",
                       f"Delete it and run: aivtube models pull {name}",
                       f"ลบไฟล์แล้วรันคำสั่ง: aivtube models pull {name}")
        else:
            what = "downloading" if r.status == "partial" else "missing"
            yield _bad(f"model {name}", level, f"{entry.dest} is {what} ({r.detail})".rstrip(" ()"),
                       f"Run: aivtube models pull {name}",
                       f"รันคำสั่ง: aivtube models pull {name}")


def _llama_exe(cfg: AppConfig) -> Iterable[Check]:
    if cfg.app.fakes:
        return
    for server in _chain_servers(cfg):
        exe = cfg.resolve_path(cfg.llm.servers[server].exe)
        if exe.is_file():
            yield _ok(f"llama-server {server}", str(exe))
        else:
            yield _bad(f"llama-server {server}", "error", f"missing: {exe}",
                       "Run: aivtube setup (downloads llama.cpp into vendor/).",
                       "รันคำสั่ง: aivtube setup (จะดาวน์โหลด llama.cpp ไว้ใน vendor/)")


def _fts5() -> Iterable[Check]:
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
            con.execute("INSERT INTO t VALUES ('ไพลินชอบแมว')")
            hits = con.execute("SELECT count(*) FROM t WHERE t MATCH 'ชอบแมว'").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as exc:
        yield _bad("sqlite fts5", "error", f"SQLite {sqlite3.sqlite_version}: {exc}",
                   "Use the Python from `uv python install 3.12` (SQLite 3.34+ with FTS5).",
                   "ใช้ Python ที่ติดตั้งผ่าน uv python install 3.12 (SQLite 3.34 ขึ้นไป)")
        return
    if hits != 1:
        yield _bad("sqlite fts5", "error", "the trigram tokenizer does not match Thai text")
    else:
        yield _ok("sqlite fts5", f"SQLite {sqlite3.sqlite_version} with the trigram tokenizer")


def _disk(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    free = env.disk_free_gb(cfg.root)
    big = cfg.resolve_path(cfg.llm.servers["local30b"].model) if "local30b" in cfg.llm.servers else None
    need = FREE_DISK_GB_30B if big is not None and not big.is_file() else FREE_DISK_GB
    if free < need:
        yield _bad("disk", "warn", f"{free:.1f} GB free (want {need:.0f} GB)",
                   "Free some disk space (the 30B model alone is 18.6 GB).",
                   "เพิ่มพื้นที่ว่างในดิสก์ (โมเดล 30B ใช้ 18.6 GB)")
    else:
        yield _ok("disk", f"{free:.1f} GB free")


def _clock() -> Iterable[Check]:
    info = time.get_clock_info("perf_counter")
    a = time.perf_counter()
    b = time.perf_counter()
    if not info.monotonic or b < a or info.resolution > 1e-5:
        yield _bad("clock", "error", f"perf_counter: monotonic={info.monotonic} "
                                     f"resolution={info.resolution}")
    elif time.gmtime().tm_year < 2025:
        yield _bad("clock", "warn", "the system date looks wrong",
                   "Turn on automatic time in Windows settings.",
                   "เปิดการตั้งเวลาอัตโนมัติในการตั้งค่า Windows")
    else:
        yield _ok("clock", f"perf_counter resolution {info.resolution * 1e9:.0f} ns")


def _secrets(cfg: AppConfig, secrets: Secrets | None) -> Iterable[Check]:
    def have(name: str | None) -> bool:
        return bool(name) and secrets is not None and bool(secrets.get(str(name)))

    for name in cfg.llm.effective_chain():
        prov = cfg.llm.providers[name]
        if prov.cloud and prov.enabled and prov.api_key_env:
            if have(prov.api_key_env):
                yield _ok(f"key {prov.api_key_env}", f"set (llm {name})")
            else:
                yield _bad(f"key {prov.api_key_env}", "warn",
                           f"llm provider {name} is enabled but {prov.api_key_env} is not set",
                           f"Add {prov.api_key_env}=... to .env",
                           f"เพิ่ม {prov.api_key_env}=... ในไฟล์ .env")
    backends = {b for i in cfg.tts.identity_chain if i in cfg.tts.identities
                for b in cfg.tts.identities[i].backends}
    azure = cfg.tts.backends.get("azure")
    if "azure" in backends and azure is not None and azure.enabled:
        for key in (azure.key_env, azure.region_env):
            if key and not have(key):
                yield _bad(f"key {key}", "warn", f"Azure TTS is in the voice chain but {key} is not set",
                           f"Add {key}=... to .env (Azure is the fallback for edge-tts).",
                           f"เพิ่ม {key}=... ในไฟล์ .env (Azure เป็นเสียงสำรองของ edge-tts)")
    if cfg.chat.enabled and "youtube_poll" in cfg.chat.sources:
        key = cfg.chat.youtube_poll.api_key_env
        if not have(key):
            yield _bad(f"key {key}", "error", "YouTube chat is enabled but no API key is set",
                       f"Add {key}=... to .env", f"เพิ่ม {key}=... ในไฟล์ .env")


def _licences(cfg: AppConfig) -> Iterable[Check]:
    if not cfg.app.monetised:
        return
    for ident in cfg.tts.identity_chain:
        spec = cfg.tts.identities.get(ident)
        if spec is not None and "piper" in spec.backends:
            yield _bad(f"voice {ident}", "warn",
                       f"the {ident!r} voice (Piper th_TH-tsync2) is non-commercial and "
                       "app.monetised = true",
                       "Remove it from tts.identity_chain on monetised streams.",
                       "สตรีมที่มีรายได้ห้ามใช้เสียงนี้ ให้ลบออกจาก tts.identity_chain")
    if cfg.avatar.sink == "vts":
        yield Check("vts licence", True, "info",
                    "monetised streams need the VTube Studio DLC (no watermark licence)")


def _consent(cfg: AppConfig) -> Iterable[Check]:
    blocked = cfg.consent_blocked()
    if blocked:
        yield Check("privacy", True, "info",
                    f"cloud fallbacks off until consent: {', '.join(blocked)}")


def _llama_live(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    if cfg.app.fakes:
        return
    from aivtube.launcher.llama import props_match

    for server in _chain_servers(cfg):
        sc = cfg.llm.servers[server]
        base = f"http://127.0.0.1:{sc.port}"
        status, _ = env.http_json(base + "/health", 1.5)
        if status == 0:
            yield Check(f"llama {server}", True, "info", "not running (the launcher starts it)")
            continue
        if status == 503:
            yield Check(f"llama {server}", True, "info", "loading")
            continue
        status, props = env.http_json(base + "/props", 2.0)
        if status != 200 or not isinstance(props, Mapping):
            yield _bad(f"llama {server}", "warn", f"/props answered {status}")
            continue
        caps = props.get("chat_template_caps") or {}
        if not (isinstance(caps, Mapping) and caps.get("supports_tool_calls") is True):
            yield _bad(f"llama {server}", "error", "the chat template cannot call tools",
                       "Use the mradermacher GGUF (it embeds the Typhoon template).",
                       "ใช้ไฟล์ GGUF ของ mradermacher ซึ่งมีเทมเพลตแชตที่รองรับเครื่องมือ")
        elif not props_match(props, alias=sc.alias, model=sc.model):
            yield _bad(f"llama {server}", "warn",
                       f"port {sc.port} serves alias {props.get('model_alias')!r}, "
                       f"expected {sc.alias!r}")
        else:
            yield _ok(f"llama {server}", f"tool calls supported ({props.get('build_info', '')})")


def _vts(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    if cfg.avatar.sink != "vts":
        return
    found = env.discover_vts(cfg.avatar.discovery_timeout_s)
    active = [i for i in found if i.get("active")]
    if active:
        desc = ", ".join(f"{i.get('windowTitle') or 'VTube Studio'} :{i.get('port')}" for i in active)
        yield _ok("vtube studio", f"API on: {desc}")
    elif found:
        yield _bad("vtube studio", "warn", "VTube Studio is running but its API is off",
                   "VTube Studio → Settings → Start API (port 8001).",
                   "ใน VTube Studio ไปที่ Settings แล้วเปิด Start API (พอร์ต 8001)")
    else:
        yield _bad("vtube studio", "warn", "no VTube Studio found (UDP 47779)",
                   "Start VTube Studio and turn on its API; the avatar reconnects by itself.",
                   "เปิด VTube Studio และเปิด API ระบบจะเชื่อมต่อให้เอง")


def _gpu(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    from aivtube.launcher.gpu import driver_major
    from aivtube.ops.models import cuda_variant

    gpus = env.gpus()
    if not gpus:
        level: Level = "error" if not cfg.app.fakes and _chain_servers(cfg) else "warn"
        yield _bad("gpu", level if sys.platform == "win32" else "warn", "nvidia-smi not available",
                   "Install the NVIDIA driver (R580 or newer for the CUDA 13.4 build).",
                   "ติดตั้งไดรเวอร์ NVIDIA (R580 ขึ้นไปสำหรับ CUDA 13.4)")
        return
    g = gpus[0]
    driver = str(g.get("driver", ""))
    variant = cuda_variant(driver_major(driver))
    yield _ok("gpu", f"{g.get('name')} driver {driver} → llama.cpp {variant}")
    free = g.get("free_mib")
    if isinstance(free, float):
        need = cfg.launcher.vram_alarm_mib
        fit = cfg.llm.servers.get("local30b")
        want = (fit.fit_target_mib if fit is not None else 0) + 1024
        if free < need:
            yield _bad("vram", "error", f"{free:.0f} MiB free", "Close GPU-heavy programs.",
                       "ปิดโปรแกรมที่ใช้การ์ดจอหนัก ๆ")
        elif free < want:
            yield _bad("vram", "warn", f"{free:.0f} MiB free (want {want} MiB before loading)",
                       "Start OBS and VTube Studio first; consider --profile light.",
                       "เปิด OBS และ VTube Studio ก่อน หรือใช้ --profile light")
        else:
            yield _ok("vram", f"{free:.0f} MiB free")


def _tts(cfg: AppConfig, env: DoctorEnv) -> Iterable[Check]:
    if env.tts_sample is not None:
        result = env.tts_sample(cfg)
    else:
        from aivtube.ops.bench import bench_tts

        result = asyncio.run(bench_tts(cfg, n=2))
    for backend, stats in result.get("backends", {}).items():
        p50 = stats.get("ttfa_p50_s")
        if stats.get("ok", 0):
            yield _ok(f"tts {backend}", f"first audio p50 {p50:.2f} s over {stats.get('n')} tries")
        else:
            yield _bad(f"tts {backend}", "warn", f"no audio ({stats.get('last_error', '')})",
                       "Check the internet connection, or use Azure (bench tts).",
                       "ตรวจการเชื่อมต่ออินเทอร์เน็ต หรือเปลี่ยนไปใช้ Azure")


def main(argv: list[str] | None = None) -> int:
    """``python -m aivtube.ops.doctor [--offline] [--tts]``: print the report; exit 1 on errors."""
    import argparse

    from aivtube.config import ConfigError, find_root, load_config

    parser = argparse.ArgumentParser(prog="aivtube doctor")
    parser.add_argument("--offline", action="store_true", help="skip live checks")
    parser.add_argument("--tts", action="store_true", help="also sample TTS first-audio time")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(find_root(), profile=args.profile)
    except ConfigError as exc:
        print(exc.format_all())
        return 1
    checks = run_doctor(cfg, live=not args.offline, tts_sample=args.tts)
    print(format_report(checks))
    return 1 if any(not c.ok and c.level == "error" for c in checks) else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
