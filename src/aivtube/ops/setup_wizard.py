"""``aivtube setup``: the idempotent, resumable install wizard (§9).

Steps (``--only`` picks some, e.g. ``aivtube setup --only audio``):

1. ``driver``: ``nvidia-smi`` driver check, the llama.cpp zips for it (CUDA 13.4 for R580+,
   else 12.4) into ``vendor/``, and ``llama-server --list-devices`` must show the GPU.
2. ``models``: Silero, the Typhoon RT export (PyThaiASR until it is published), the 4B, then
   the 30B as a resumable background download. ``active_profile`` stays ``light`` until the
   30B verifies (the background download switches it to ``stream``).
3. ``audio``: WASAPI devices by name, a test tone, the mic level, "headphones?".
4. ``chat``: Twitch or YouTube, and the channel.
5. ``vts``: VTube Studio discovery and plugin authorisation (the streamer clicks Allow).
6. ``privacy``: consent for cloud fallbacks (default no).
7. ``tts``: ``bench tts`` and the G1 recommendation (edge or Azure first).
8. ``phrases``: pre-synthesise the cached phrases (only with the voice extras installed).
9. ``finish``: save, and run ``doctor``.

Answers are written to ``config/user.toml`` through ``write_user_overrides`` after each step, so
an interrupted setup keeps what it learned; ``data/state/setup.json`` records finished steps.
``--yes`` answers every question with its default (or the value given as a flag).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as _dt
import importlib.util
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aivtube.ops.models import (
    MANIFEST,
    ModelBusy,
    ModelError,
    ModelManifest,
    ProgressLine,
    cuda_variant,
)

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig
    from aivtube.contracts.voice import AudioBackend

__all__ = [
    "STEPS",
    "ConsolePrompter",
    "Prompter",
    "SetupDeps",
    "SetupOptions",
    "SetupWizard",
    "main",
    "run_setup",
]

log = logging.getLogger("aivtube.ops.setup")

STEPS = ("driver", "models", "audio", "chat", "vts", "privacy", "tts", "phrases", "finish")
MODEL_ORDER = ("silero_vad", "typhoon_rt_tokens", "typhoon_rt_encoder", "typhoon_rt_decoder",
               "typhoon_rt_joiner", "typhoon_rt_readme", "llm_4b")
BIG_MODEL = "llm_30b"
PRESYNTH_TIMEOUT_S = 300.0
"""Upper bound for pre-synthesising one character's cached phrases (network TTS)."""


class Prompter(Protocol):
    def say(self, text: str) -> None: ...

    def ask(self, question: str, default: str) -> str: ...


class ConsolePrompter:
    """``input()`` questions; with ``yes`` every question takes its default."""

    def __init__(self, *, yes: bool = False) -> None:
        self.yes = yes

    def say(self, text: str) -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)

    def ask(self, question: str, default: str) -> str:
        if self.yes:
            self.say(f"{question} [{default}]")
            return default
        try:
            answer = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        return answer or default


def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes", "ใช่", "ช", "1", "true")


@dataclass
class SetupOptions:
    only: Sequence[str] | None = None
    skip: Sequence[str] = ()
    output_device: str | None = None
    input_device: str | None = None
    headphones: bool | None = None
    chat: str | None = None
    channel: str | None = None
    video: str | None = None
    cloud_llm_consent: bool | None = None
    cloud_stt_consent: bool | None = None
    variant: str | None = None
    big_model: bool = True
    bench: bool | None = None


_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _default_spawn(argv: list[str], log_path: Path) -> int | None:
    """Start a detached background process (it outlives setup); returns its pid.

    On Windows ``uv run`` may hold setup in a kill-on-close Job Object, which would take the
    download down with it: the child breaks away from the job when the job allows that.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]]
    if sys.platform == "win32":
        base = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
        attempts = [{"creationflags": base | _CREATE_BREAKAWAY_FROM_JOB},
                    {"creationflags": base}]
    else:
        attempts = [{"start_new_session": True}]
    with log_path.open("ab") as out:
        for i, kwargs in enumerate(attempts):
            try:
                proc = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                    **kwargs,
                )
            except OSError:
                if i + 1 == len(attempts):
                    raise
                continue  # the job forbids breakaway (access denied): start it inside
            return proc.pid
    return None


def _default_gpus() -> list[dict[str, Any]] | None:
    from aivtube.launcher.gpu import query_gpus

    return query_gpus()


def _default_audio_backend() -> AudioBackend:
    from aivtube.voice.audio_io import default_backend

    return default_backend()


def _default_vts(timeout_s: float) -> list[dict[str, Any]]:
    from aivtube.avatar.discovery import discover_vts

    return asyncio.run(discover_vts(timeout_s))


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


async def _default_bench(cfg: AppConfig) -> dict[str, Any]:
    from aivtube.ops.bench import bench_tts

    return await bench_tts(cfg, n=40)


async def _default_presynth(cfg: AppConfig) -> dict[str, dict[str, bool]]:
    from aivtube.config import load_characters, load_secrets
    from aivtube.infra.clock import deadline
    from aivtube.voice.tts.factory import build_tts_router

    chars = load_characters(cfg)
    secrets = load_secrets(cfg.root)
    router = build_tts_router(
        cfg.tts.model_dump(),
        {cid: cfg.tts_chain_for(c) for cid, c in chars.items()},
        root=cfg.root,
        secrets=secrets.get,
    )
    out: dict[str, dict[str, bool]] = {}
    try:
        for cid, c in chars.items():
            async with deadline(PRESYNTH_TIMEOUT_S, what=f"pre-synthesis for {cid}"):
                out[cid] = dict(await router.presynthesize(cid, c.cached_phrases))
        return out
    finally:
        async with deadline(10.0, what="tts router close"):
            await router.aclose()


def _default_play_tone(cfg: AppConfig) -> None:
    """One second of a quiet 440 Hz tone on the configured output device."""
    import time

    import numpy as np

    from aivtube.voice.audio_io import StreamingPlayer

    sr = cfg.audio.samplerate
    player = StreamingPlayer(cfg.audio.output_device or None, samplerate=sr,
                             blocksize=sr * cfg.audio.block_ms // 1000,
                             latency=cfg.audio.output_latency_s)
    player.start()
    try:
        t = np.arange(sr) / sr
        player.play((0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        time.sleep(1.3)
    finally:
        player.close()


def _default_mic_level(cfg: AppConfig) -> float:
    """The loudest block (dBFS) over three seconds of the configured microphone."""
    import math
    import time

    import numpy as np

    from aivtube.voice.audio_io import MicCapture

    peak = [1e-9]

    def on_frame(block: Any, t: float) -> None:
        peak[0] = max(peak[0], float(np.sqrt(np.mean(np.square(block))) if len(block) else 0.0))

    mic = MicCapture(cfg.audio.input_device or None, samplerate=cfg.audio.samplerate,
                     blocksize=cfg.audio.samplerate * cfg.audio.block_ms // 1000)
    mic.start(on_frame)
    try:
        time.sleep(3.0)
    finally:
        mic.close()
    return 20 * math.log10(max(peak[0], 1e-9))


def _default_doctor(cfg: AppConfig) -> str:
    from aivtube.ops.doctor import format_report, run_doctor

    return format_report(run_doctor(cfg, live=True))


@dataclass
class SetupDeps:
    """Everything setup touches outside the project folder (replaced in tests)."""

    gpus: Callable[[], list[dict[str, Any]] | None] = _default_gpus
    manifest: Callable[[Path], ModelManifest] = field(
        default=lambda root: ModelManifest.load(root / MANIFEST, root=root)
    )
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    spawn_background: Callable[[list[str], Path], int | None] = _default_spawn
    audio_backend: Callable[[], AudioBackend] = _default_audio_backend
    find_spec: Callable[[str], bool] = _find_spec
    discover_vts: Callable[[float], list[dict[str, Any]]] = _default_vts
    bench_tts: Callable[[AppConfig], Coroutine[Any, Any, dict[str, Any]]] = _default_bench
    presynth: Callable[
        [AppConfig], Coroutine[Any, Any, Mapping[str, Mapping[str, bool]]]
    ] = _default_presynth
    doctor: Callable[[AppConfig], str] = _default_doctor
    play_tone: Callable[[AppConfig], None] | None = _default_play_tone
    mic_level: Callable[[AppConfig], float] | None = _default_mic_level
    platform: str = sys.platform


@dataclass
class StepResult:
    ok: bool
    detail: str
    updates: dict[str, Any] = field(default_factory=dict)


class SetupWizard:
    """Runs the steps (see the module docstring)."""

    def __init__(
        self,
        root: Path,
        options: SetupOptions | None = None,
        prompter: Prompter | None = None,
        deps: SetupDeps | None = None,
    ) -> None:
        self.root = Path(root)
        self.opt = options or SetupOptions()
        self.io: Prompter = prompter or ConsolePrompter()
        self.deps = deps or SetupDeps()
        self.state_path = self.root / "data" / "state" / "setup.json"
        self.results: dict[str, StepResult] = {}

    # -- plumbing ----------------------------------------------------------------------------

    def _cfg(self) -> AppConfig:
        from aivtube.config import load_config

        return load_config(self.root)

    def _state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, name: str, result: StepResult, **extra: Any) -> None:
        data = self._state()
        steps = data.setdefault("steps", {})
        steps[name] = {"ok": result.ok, "detail": result.detail,
                       "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        data.update(extra)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _write(self, updates: Mapping[str, Any]) -> None:
        if not updates:
            return
        from aivtube.config import write_user_overrides

        write_user_overrides(self.root, updates)

    def selected(self) -> list[str]:
        names = list(self.opt.only) if self.opt.only else list(STEPS)
        unknown = [n for n in names if n not in STEPS]
        if unknown:
            raise ValueError(f"unknown setup step(s): {', '.join(unknown)}")
        return [n for n in STEPS if n in names and n not in self.opt.skip]

    def run(self) -> int:
        self.io.say("aivtube setup / ตั้งค่า aivtube")
        failed = 0
        for name in self.selected():
            step: Callable[[], StepResult] = getattr(self, f"step_{name}")
            self.io.say(f"\n== {name} ==")
            try:
                result = step()
                self._write(result.updates)
            except KeyboardInterrupt:
                self.io.say("stopped; run setup again to continue / หยุดแล้ว รัน setup อีกครั้งเพื่อทำต่อ")
                return 130
            except Exception as exc:
                log.exception("setup step %s failed", name)
                result = StepResult(False, f"{type(exc).__name__}: {exc}")
            self.results[name] = result
            self._save_state(name, result)
            self.io.say(f"{'✔' if result.ok else '✖'} {name}: {result.detail}")
            failed += 0 if result.ok else 1
        return 0 if not failed else 1

    # -- steps -------------------------------------------------------------------------------

    def step_driver(self) -> StepResult:
        from aivtube.launcher.gpu import driver_major

        gpus = self.deps.gpus() or []
        driver = str(gpus[0].get("driver", "")) if gpus else ""
        variant = self.opt.variant or cuda_variant(driver_major(driver) if driver else None)
        if not gpus:
            self.io.say("! nvidia-smi not found: install the NVIDIA driver (R580+) / "
                        "ไม่พบไดรเวอร์ NVIDIA ให้ติดตั้งไดรเวอร์ R580 ขึ้นไป")
        if self.deps.platform != "win32":
            return StepResult(True, f"not Windows: llama.cpp zips skipped (would use {variant})")
        manifest = self.deps.manifest(self.root)
        names = [n for n, e in manifest.entries.items()
                 if e.variant == variant and e.archive and not e.optional]
        for name in names:
            manifest.pull_one(name, progress=ProgressLine())
        exe = self._cfg().resolve_path(self._cfg().llm.servers["local4b"].exe)
        try:
            done = self.deps.run([str(exe), "--list-devices"], capture_output=True, text=True,
                                 timeout=60, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return StepResult(False, f"llama-server did not run: {exc}")
        devices = (done.stdout or "") + (done.stderr or "")
        if "CUDA" not in devices:
            return StepResult(False, f"llama-server sees no CUDA GPU ({variant}); update the driver")
        gpu = gpus[0].get("name", "GPU") if gpus else "GPU"
        return StepResult(True, f"driver {driver or '?'} → {variant}; llama-server sees {gpu}")

    def step_models(self) -> StepResult:
        manifest = self.deps.manifest(self.root)
        missing: list[str] = []
        for name in MODEL_ORDER:
            if name not in manifest.entries:
                continue
            try:
                manifest.pull_one(name, progress=ProgressLine())
            except ModelBusy as exc:
                self.io.say(f"! {exc}; run setup again when it has finished")
                missing.append(name)
                continue
            except ModelError as exc:
                if name.startswith("typhoon_rt"):
                    self.io.say(f"! {name}: {exc} (STT uses PyThaiASR until it is published)")
                    missing.append(name)
                    continue
                raise
        updates: dict[str, Any] = {}
        big = manifest.check(BIG_MODEL, full=False) if BIG_MODEL in manifest.entries else None
        detail = "models ready"
        if big is not None and big.status != "ok" and self.opt.big_model:
            updates["active_profile"] = "light"
            log_path = self.root / "logs" / "models-pull.log"
            if manifest.downloading(BIG_MODEL):  # an earlier setup's download is still running
                detail = (f"the 30B is already downloading in the background (log "
                          f"{log_path.name}); profile light until it verifies / "
                          "โมเดล 30B กำลังดาวน์โหลดอยู่แล้ว")
            else:
                argv = [sys.executable, "-m", "aivtube.ops.models", "pull", BIG_MODEL,
                        "--root", str(self.root), "--then-profile", "stream"]
                pid = self.deps.spawn_background(argv, log_path)
                detail = (f"the 30B downloads in the background (pid {pid}, log "
                          f"{log_path.name}); profile light until it verifies / "
                          "โมเดล 30B กำลังดาวน์โหลดอยู่เบื้องหลัง")
        elif big is not None and big.status == "ok":
            cfg = self._cfg()
            if cfg.active_profile == "light" and self._state().get("profile_pending_30b"):
                updates["active_profile"] = "stream"
                self._save_state("models", StepResult(True, "30B verified"),
                                 profile_pending_30b=False)
            detail = "all models verified"
        if updates.get("active_profile") == "light":
            self._save_state("models", StepResult(True, detail), profile_pending_30b=True)
        if missing:
            detail += f"; not available yet: {', '.join(missing)}"
        return StepResult(True, detail, updates)

    def step_audio(self) -> StepResult:
        from aivtube.voice.audio_io import WASAPI, list_devices

        cfg = self._cfg()
        try:
            devices = list_devices(self.deps.audio_backend())
        except (OSError, ImportError) as exc:
            return StepResult(False, f"PortAudio is not available ({exc}); install --extra voice")
        wasapi = [d for d in devices if d.hostapi == WASAPI] or devices
        outs = sorted({d.name for d in wasapi if d.max_out > 0})
        ins = sorted({d.name for d in wasapi if d.max_in > 0})
        updates: dict[str, Any] = {}
        out = self._pick("output", outs, self.opt.output_device, cfg.audio.output_device)
        inp = self._pick("input", ins, self.opt.input_device, cfg.audio.input_device)
        updates["audio.output_device"] = out
        updates["audio.input_device"] = inp
        checked = cfg.model_copy(
            update={"audio": cfg.audio.model_copy(update={"output_device": out,
                                                          "input_device": inp})}
        )
        if self.deps.play_tone is not None and _yes(self.io.ask(
            "Play a test tone? / เล่นเสียงทดสอบไหม (y/n)", "n"
        )):
            try:
                self.deps.play_tone(checked)
            except Exception as exc:
                return StepResult(False, f"cannot play on {out or 'the default device'}: {exc}")
        if self.deps.mic_level is not None and _yes(self.io.ask(
            "Say something to test the mic? / ทดสอบไมค์ไหม (พูดอะไรก็ได้ 3 วินาที) (y/n)", "n"
        )):
            try:
                level = self.deps.mic_level(checked)
            except Exception as exc:
                return StepResult(False, f"cannot open the mic {inp or '(default)'}: {exc}")
            self.io.say(f"mic peak {level:.0f} dBFS" + (
                "  (too quiet: check the mic / เบาเกินไป ตรวจไมค์)" if level < -45 else ""
            ))
        if self.opt.headphones is not None:
            headphones = self.opt.headphones
        else:
            headphones = _yes(self.io.ask(
                "Do you use headphones? / ใช้หูฟังไหม (y/n)", "y" if cfg.audio.headphones else "n"
            ))
        updates["audio.headphones"] = headphones
        return StepResult(True, f"output {out or '(default)'}, input {inp or '(default)'}, "
                                f"headphones {headphones}", updates)

    def _pick(self, kind: str, names: Sequence[str], flag: str | None, current: str) -> str:
        if flag is not None:
            return flag
        for i, name in enumerate(names, 1):
            self.io.say(f"  {i}. {name}")
        default = current or ""
        answer = self.io.ask(
            f"{kind} device: number, part of the name, or empty for the Windows default / "
            f"เลือกอุปกรณ์ {'เสียงออก' if kind == 'output' else 'ไมค์'}",
            default,
        )
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        return answer

    def step_chat(self) -> StepResult:
        cfg = self._cfg()
        current = "youtube" if "youtube_poll" in cfg.chat.sources else "twitch"
        platform = self.opt.chat or self.io.ask(
            "Chat platform: twitch / youtube / none / แพลตฟอร์มแชต", current
        )
        platform = platform.strip().lower()
        if platform == "none":
            return StepResult(True, "chat off", {"chat.enabled": False})
        if platform == "youtube":
            video = self.opt.video if self.opt.video is not None else self.io.ask(
                "YouTube video id or @handle / ไอดีวิดีโอหรือ @handle", cfg.chat.youtube_poll.video_id
            )
            key = "chat.youtube_poll.handle" if video.startswith("@") else "chat.youtube_poll.video_id"
            return StepResult(True, f"YouTube {video or '(set later)'}", {
                "chat.enabled": True, "chat.sources": ["youtube_poll"], key: video,
            })
        if platform != "twitch":
            return StepResult(False, f"unknown platform {platform!r}")
        channel = self.opt.channel if self.opt.channel is not None else self.io.ask(
            "Twitch channel (login name) / ชื่อช่อง Twitch", cfg.chat.twitch_irc.channel
        )
        return StepResult(True, f"Twitch #{channel or '(set later)'}", {
            "chat.enabled": True, "chat.sources": ["twitch_irc"],
            "chat.twitch_irc.channel": channel.strip().lstrip("#").lower(),
        })

    def step_vts(self) -> StepResult:
        cfg = self._cfg()
        if cfg.avatar.sink != "vts":
            return StepResult(True, f"avatar sink is {cfg.avatar.sink}")
        found = self.deps.discover_vts(3.0)
        active = [i for i in found if i.get("active")]
        if not active:
            return StepResult(True, "VTube Studio not found: start it, turn on the API, and run "
                                    "setup --only vts / เปิด VTube Studio และเปิด API แล้วรันอีกครั้ง")
        names = ", ".join(f"{i.get('windowTitle') or 'VTube Studio'} :{i.get('port')}" for i in active)
        return StepResult(True, f"found {names}; the plugin asks for access on the first run "
                                "(click Allow in VTube Studio) / กด Allow ใน VTube Studio")

    def step_privacy(self) -> StepResult:
        cfg = self._cfg()
        self.io.say(
            "Cloud fallbacks send what is said on stream to other companies: the Typhoon API may "
            "train on it and Gemini's free tier is used by Google.\n"
            "ระบบสำรองบนคลาวด์จะส่งข้อความบนสตรีมไปยังบริษัทอื่น (Typhoon API อาจนำไปฝึกโมเดล "
            "และ Gemini ฟรีจะถูก Google ใช้ข้อมูล)"
        )
        llm = self.opt.cloud_llm_consent
        if llm is None:
            llm = _yes(self.io.ask("Allow cloud LLM fallbacks? / อนุญาต LLM บนคลาวด์ไหม (y/n)",
                                   "y" if cfg.privacy.cloud_llm_consent else "n"))
        stt = self.opt.cloud_stt_consent
        if stt is None:
            stt = _yes(self.io.ask("Allow cloud speech-to-text? / อนุญาตถอดเสียงบนคลาวด์ไหม (y/n)",
                                   "y" if cfg.privacy.cloud_stt_consent else "n"))
        return StepResult(True, f"cloud LLM {'on' if llm else 'off'}, cloud STT "
                                f"{'on' if stt else 'off'}",
                          {"privacy.cloud_llm_consent": llm, "privacy.cloud_stt_consent": stt})

    def step_tts(self) -> StepResult:
        run = self.opt.bench
        if run is None:
            run = self.deps.find_spec("edge_tts")
        if not run:
            return StepResult(True, "skipped (voice extras not installed, or --no-bench)")
        from aivtube.ops.bench import apply_tts_recommendation, write_recommendations

        cfg = self._cfg()
        self.io.say("Measuring TTS first-audio time (about 2 minutes) / กำลังวัดความเร็วเสียงพูด")
        result = asyncio.run(self.deps.bench_tts(cfg))
        write_recommendations(self.root, {"tts": result})
        rec = result.get("recommendation", {})
        edge = result.get("backends", {}).get("edge", {})
        detail = f"edge p50 {edge.get('ttfa_p50_s')} s, failures {edge.get('fail_rate')}"
        if rec.get("change") and _yes(self.io.ask(
            f"{rec.get('reason')}: use {' → '.join(rec['backends'])}? / ใช้ลำดับนี้ไหม (y/n)", "y"
        )):
            apply_tts_recommendation(self.root, str(result.get("identity")), rec)
            detail += f"; voice order {' → '.join(rec['backends'])}"
        return StepResult(True, detail)

    def step_phrases(self) -> StepResult:
        if not all(self.deps.find_spec(m) for m in ("edge_tts", "av")):
            return StepResult(True, "skipped: the voice extras are not installed")
        results = asyncio.run(self.deps.presynth(self._cfg()))
        missing = [f"{c}:{p}" for c, per in results.items() for p, ok in per.items() if not ok]
        done = sum(1 for per in results.values() for ok in per.values() if ok)
        if missing:
            return StepResult(False, f"{done} cached, failed: {', '.join(missing)}")
        return StepResult(True, f"{done} phrases cached (including Filtered.)")

    def step_finish(self) -> StepResult:
        cfg = self._cfg()
        report = self.deps.doctor(cfg)
        self.io.say(report)
        self.io.say("run.bat sets HF_HOME=models\\hf; start aivtube with run.bat / "
                    "เริ่มใช้งานด้วย run.bat")
        errors = sum(1 for line in report.splitlines() if line.startswith("✖"))
        return StepResult(True, f"config saved; doctor: {errors} error(s)")


def run_setup(
    root: Path,
    *,
    only: Sequence[str] | None = None,
    non_interactive: bool = False,
    options: SetupOptions | None = None,
    deps: SetupDeps | None = None,
    prompter: Prompter | None = None,
) -> int:
    """Run the wizard; 0 when every step succeeded."""
    opts = options or SetupOptions()
    if only is not None:
        opts.only = list(only)
    wizard = SetupWizard(root, opts, prompter or ConsolePrompter(yes=non_interactive), deps)
    return wizard.run()


def _tristate(value: str | None) -> bool | None:
    return None if value is None else _yes(value)


def main(argv: list[str] | None = None) -> int:
    from aivtube.config import find_root

    p = argparse.ArgumentParser(prog="aivtube setup")
    p.add_argument("--yes", "-y", action="store_true", help="accept every default")
    p.add_argument("--only", action="append", choices=STEPS, help="run only this step")
    p.add_argument("--skip", action="append", default=[], choices=STEPS)
    p.add_argument("--audio", action="store_true", help="same as --only audio")
    p.add_argument("--output-device")
    p.add_argument("--input-device")
    p.add_argument("--headphones", choices=["y", "n"])
    p.add_argument("--chat", choices=["twitch", "youtube", "none"])
    p.add_argument("--channel")
    p.add_argument("--video")
    p.add_argument("--cloud-llm-consent", choices=["y", "n"])
    p.add_argument("--cloud-stt-consent", choices=["y", "n"])
    p.add_argument("--variant", choices=["cuda-13.4", "cuda-12.4"])
    p.add_argument("--no-30b", action="store_true", help="do not download the 30B model")
    p.add_argument("--no-bench", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    only = (args.only or []) + (["audio"] if args.audio else [])
    opts = SetupOptions(
        only=only or None, skip=args.skip, output_device=args.output_device,
        input_device=args.input_device, headphones=_tristate(args.headphones), chat=args.chat,
        channel=args.channel, video=args.video,
        cloud_llm_consent=_tristate(args.cloud_llm_consent),
        cloud_stt_consent=_tristate(args.cloud_stt_consent), variant=args.variant,
        big_model=not args.no_30b, bench=False if args.no_bench else None,
    )
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    return run_setup(find_root(), options=opts, non_interactive=args.yes)


if __name__ == "__main__":
    sys.exit(main())
