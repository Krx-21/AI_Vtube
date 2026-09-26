"""``aivtube report``: a bug-report zip (§2.10) with no secrets in it.

Contents: the last ``days`` of logs (``.log``, ``.jsonl``, ``.fault``; big files keep their
tail), the redacted ``config/user.toml`` and effective config, the doctor report (offline
checks), versions, the launcher's state files and the newest flight-recorder dump. ``.env`` and
``data/state/tokens.json`` are never included, and every text file goes through the same
redaction as the logs (``aivtube.infra.logging.Redactor``) with the ``.env`` values and the
launcher tokens added.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = ["build_report", "collect_secrets"]

log = logging.getLogger("aivtube.ops.report")

MAX_FILE_BYTES = 8 * 1024 * 1024
"""Larger log files keep only their last 8 MiB."""
TEXT_SUFFIXES = {".log", ".jsonl", ".fault", ".txt", ".json", ".toml", ".md"}
NEVER = {".env", "tokens.json"}
PACKAGES = (
    "numpy", "websockets", "openai", "httpx2", "aiohttp", "pydantic", "pythainlp",
    "sounddevice", "soxr", "onnxruntime", "sherpa-onnx", "edge-tts", "av", "livekit",
    "azure-cognitiveservices-speech",
)


def _dirs(root: Path) -> tuple[Path, Path]:
    """``(logs, data)`` folders from ``logging.dir`` / ``app.data_dir`` (defaults on error)."""
    logs, data = "logs", "data"
    try:
        from aivtube.config.layers import collect_layers

        merged = collect_layers(root).merged
        logs = str(merged.get("logging", {}).get("dir", logs))
        data = str(merged.get("app", {}).get("data_dir", data))
    except Exception:
        log.debug("config layers unreadable; using the default folders", exc_info=True)

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    return resolve(logs), resolve(data)


def collect_secrets(root: Path, env: dict[str, str] | None = None) -> list[str]:
    """Every secret value we know of: ``.env`` / environment keys and the launcher tokens."""
    values: set[str] = set()
    try:
        from aivtube.config import load_secrets

        values.update(load_secrets(root, env=env).redaction_values())
    except Exception:
        log.warning("could not read .env for redaction", exc_info=True)
    tokens = _dirs(Path(root))[1] / "state" / "tokens.json"
    try:
        data = json.loads(tokens.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            values.update(str(v) for v in data.values() if isinstance(v, str))
    except (OSError, ValueError):
        pass
    source = os.environ if env is None else env
    values.update(v for k, v in source.items() if k.startswith("AIVTUBE_") and "TOKEN" in k and v)
    return sorted(v for v in values if v)


def _redactor(secrets: Iterable[str]) -> Callable[[str], str]:
    from aivtube.infra.logging import Redactor

    return Redactor(secrets)


def _read_text(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > MAX_FILE_BYTES:
            fh.seek(size - MAX_FILE_BYTES)
            head = f"[… {size - MAX_FILE_BYTES} bytes cut …]\n"
        else:
            head = ""
        return head + fh.read().decode("utf-8", "replace")


def _log_files(logs: Path, days: int) -> list[Path]:
    if not logs.is_dir():
        return []
    today = _dt.date.today()
    out: list[Path] = []
    for folder in sorted(logs.iterdir()):
        try:
            day = _dt.date.fromisoformat(folder.name)
        except ValueError:
            continue
        if folder.is_dir() and (today - day).days < max(1, days):
            out += sorted(p for p in folder.rglob("*") if p.is_file())
    return out


def _newest_flight(logs: Path, data: Path) -> Path | None:
    found: list[tuple[float, Path]] = []
    for base in (logs, data):
        if base.is_dir():
            for p in base.rglob("flight-*.json"):
                try:
                    found.append((p.stat().st_mtime, p))
                except OSError:
                    continue
    return max(found)[1] if found else None


def _versions(root: Path) -> dict[str, Any]:
    import aivtube

    pkgs: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "aivtube": aivtube.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": pkgs,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
    }


def _doctor_text(root: Path) -> str:
    from aivtube.config import load_config
    from aivtube.ops.doctor import format_report, run_doctor

    try:
        cfg = load_config(root)
    except Exception as exc:
        return f"config did not load: {exc}"
    return format_report(run_doctor(cfg, live=False))


def _effective_config(root: Path) -> str:
    from aivtube.config import load_config

    try:
        cfg = load_config(root)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False)


def build_report(
    root: Path,
    out: Path | None = None,
    *,
    days: int = 2,
    doctor: str | Callable[[Path], str] | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Write the report zip and return its path (default ``data/reports/``). Blocking."""
    root = Path(root)
    logs, data = _dirs(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(out) if out is not None else data / "reports" / f"aivtube-report-{stamp}.zip"
    if target.is_dir() or target.suffix.lower() != ".zip":  # a folder to put it in
        target = target / f"aivtube-report-{stamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    redact = _redactor(collect_secrets(root, env))
    tmp = target.with_name(target.name + ".tmp")

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        def add_text(name: str, text: str) -> None:
            zf.writestr(name, redact(text))

        def add_file(name: str, path: Path) -> None:
            if path.name in NEVER:
                return
            try:
                if path.suffix.lower() in TEXT_SUFFIXES:
                    add_text(name, _read_text(path))
                elif path.stat().st_size <= MAX_FILE_BYTES:
                    zf.write(path, name)
            except OSError as exc:
                add_text(name + ".error.txt", f"could not read: {exc}")

        for path in _log_files(logs, days):
            add_file("logs/" + path.relative_to(logs).as_posix(), path)
        user = root / "config" / "user.toml"
        if user.is_file():
            add_file("config/user.toml", user)
        add_text("config/effective.json", _effective_config(root))
        doctor_text = doctor(root) if callable(doctor) else doctor
        add_text("doctor.txt", doctor_text if doctor_text is not None else _doctor_text(root))
        add_text("versions.json", json.dumps(_versions(root), indent=2, ensure_ascii=False))
        state = data / "state"
        if state.is_dir():
            for path in sorted(state.glob("*.json")):
                add_file(f"state/{path.name}", path)
        flight = _newest_flight(logs, data)
        if flight is not None:
            add_file(f"flight/{flight.name}", flight)
        add_text("README.txt", (
            "aivtube bug report. Secrets (.env values, tokens, Bearer/oauth strings) were "
            "replaced with ***.\nรายงานปัญหาของ aivtube ข้อมูลลับถูกแทนด้วย ***\n"
        ))
    os.replace(tmp, target)
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    from aivtube.config import find_root

    parser = argparse.ArgumentParser(prog="aivtube report")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--days", type=int, default=2)
    args = parser.parse_args(argv)
    path = build_report(find_root(), args.out, days=args.days)
    print(f"report: {path}\nส่งไฟล์นี้ให้ผู้ดูแลได้เลย (ไม่มีรหัสลับอยู่ในไฟล์)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
