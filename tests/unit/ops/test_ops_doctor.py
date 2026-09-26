"""``doctor`` with everything faked (ports, audio, GPU, VTS, llama, secrets, models)."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from aivtube.config import load_config, load_secrets
from aivtube.ops.doctor import Check, DoctorEnv, format_report, run_doctor
from aivtube.testing.fakes import FakeSD

REPO = Path(__file__).resolve().parents[3]
BLOBS = {
    "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf": b"GGUF30" + bytes(1000),
    "models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf": b"GGUF4" + bytes(700),
    "models/vad/silero_vad.onnx": b"onnx" + bytes(300),
}
REQUIRED = {
    "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf": "llm.servers.local30b",
    "models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf": "llm.servers.local4b",
    "models/vad/silero_vad.onnx": "vad.model",
}
PROPS = {
    "model_alias": "pailin-30b",
    "model_path": "C:/ai/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
    "chat_template_caps": {"supports_tool_calls": True},
    "build_info": "b11177",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    lines = ["schema_version = 1"]
    for i, (dest, data) in enumerate(BLOBS.items()):
        lines += [
            f"[models.m{i}]",
            'url = "https://example.invalid/x"',
            f'sha256 = "{hashlib.sha256(data).hexdigest()}"',
            f"size = {len(data)}",
            'licence = "MIT"',
            f'required_by = ["{REQUIRED[dest]}"]',
            f'dest = "{dest}"',
        ]
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.toml").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def install(root: Path) -> None:
    for dest, data in BLOBS.items():
        path = root / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    exe = root / "vendor" / "llama.cpp" / "llama-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"MZ")
    tok = root / "models" / "stt" / "typhoon-asr-rt-int8" / "tokens.txt"
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text("x", encoding="utf-8")


def http_json(url: str, timeout: float) -> tuple[int, Any]:
    if url.endswith("/health"):
        return (200, {"status": "ok"}) if ":8080" in url else (0, None)
    return 200, PROPS


def env(**over: Any) -> DoctorEnv:
    base: dict[str, Any] = {
        "find_spec": lambda name: True,
        "audio_backend": lambda: FakeSD(),
        "port_in_use": lambda port: False,
        "http_json": http_json,
        "gpus": lambda: [{"name": "NVIDIA GeForce RTX 4070", "driver": "581.42",
                          "free_mib": 9000.0, "total_mib": 12282.0}],
        "session_id": lambda: 1,
        "discover_vts": lambda timeout: [{"active": True, "port": 8001,
                                          "windowTitle": "VTube Studio"}],
        "disk_free_gb": lambda path: 200.0,
        "secrets": load_secrets(None, env={"AZURE_SPEECH_KEY": "az-123456",
                                           "AZURE_SPEECH_REGION": "southeastasia"}),
    }
    base.update(over)
    return DoctorEnv(**base)


def cfg(root: Path, **overrides: Any) -> Any:
    return load_config(root, cli_overrides=overrides, env={})


def problems(checks: list[Check], level: str | None = None) -> dict[str, Check]:
    return {c.name: c for c in checks if not c.ok and (level is None or c.level == level)}


def test_everything_healthy(root: Path) -> None:
    install(root)
    checks = run_doctor(cfg(root), env=env())
    assert problems(checks) == {}, format_report(checks)
    names = {c.name for c in checks}
    assert {"python", "sqlite fts5", "audio output", "audio input", "llama local30b",
            "vtube studio", "gpu", "vram", "clock", "disk"} <= names
    gpu = next(c for c in checks if c.name == "gpu")
    assert "cuda-13.4" in gpu.detail
    report = format_report(checks)
    assert "✔" in report and "0 error(s)" in report and "ข้อผิดพลาด 0" in report


def test_lists_wasapi_devices_by_name(root: Path) -> None:
    install(root)
    checks = run_doctor(cfg(root, **{"audio.output_device": "Speakers"}), env=env(), live=False)
    devices = next(c for c in checks if c.name == "audio devices")
    assert devices.detail.startswith("WASAPI")
    assert "CABLE Input (VB-Audio Virtual Cable)" in devices.detail  # the full WASAPI name
    out = next(c for c in checks if c.name == "audio output")
    assert out.ok and "Speakers (Realtek(R) Audio)" in out.detail


def test_unknown_device_name_is_an_error_with_the_choices(root: Path) -> None:
    install(root)
    checks = run_doctor(cfg(root, **{"audio.input_device": "Blue Yeti"}), env=env(), live=False)
    bad = problems(checks, "error")["audio input"]
    assert "Blue Yeti" in bad.detail and "Microphone (USB Mic)" in bad.hint_en
    assert "ชื่ออุปกรณ์" in bad.hint_th


def test_no_portaudio(root: Path) -> None:
    install(root)

    def missing() -> Any:
        raise OSError("PortAudio library not found")

    checks = run_doctor(cfg(root), env=env(audio_backend=missing), live=False)
    assert "PortAudio" in problems(checks)["audio"].detail


def test_ports_in_use_and_vts_range_misuse(root: Path) -> None:
    install(root)
    c = cfg(root)
    checks = run_doctor(c, env=env(port_in_use=lambda p: p == 8770), live=False)
    assert problems(checks, "warn")["port 8770"].detail.startswith("ports.panel")
    bad_ports = c.ports.model_copy(update={"panel": 8001})
    misused = c.model_copy(update={"ports": bad_ports})
    checks = run_doctor(misused, env=env(), live=False)
    err = problems(checks, "error")["port 8001"]
    assert "VTube Studio" in err.detail and "8001–8009" in err.hint_th


def test_missing_and_corrupt_models(root: Path) -> None:
    checks = run_doctor(cfg(root), env=env(), live=False)
    errs = problems(checks, "error")
    assert {"model m0", "model m2", "llama-server local30b"} <= set(errs)
    assert "missing" in errs["model m0"].detail
    assert "models pull m0" in errs["model m0"].hint_en
    assert problems(checks, "warn")["model m1"]  # the 4B is on demand: a warning
    install(root)
    (root / "models" / "vad" / "silero_vad.onnx").write_bytes(b"xxxx" + bytes(300))
    errs = problems(run_doctor(cfg(root), env=env(), live=False), "error")
    assert list(errs) == ["model m2"] and "damaged" in errs["model m2"].detail


def test_sqlite_without_fts5_trigram(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(root)
    real = sqlite3.connect

    class NoFts:
        def __init__(self) -> None:
            self.con = real(":memory:")

        def execute(self, sql: str, *a: Any) -> Any:
            if "fts5" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return self.con.execute(sql, *a)

        def close(self) -> None:
            self.con.close()

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: NoFts())
    checks = run_doctor(cfg(root), env=env(), live=False)
    assert "fts5" in problems(checks, "error")["sqlite fts5"].detail


def test_session0(root: Path) -> None:
    install(root)
    checks = run_doctor(cfg(root), env=env(session_id=lambda: 0), live=False)
    assert "Session 0" in problems(checks, "error")["session"].detail


def test_missing_env_keys_for_enabled_backends(root: Path) -> None:
    install(root)
    c = cfg(root, **{"chat.sources": ["youtube_poll"], "chat.youtube_poll.video_id": "abc",
                     "privacy.cloud_llm_consent": True})
    checks = run_doctor(c, env=env(secrets=load_secrets(None, env={})), live=False)
    errs, warns = problems(checks, "error"), problems(checks, "warn")
    assert "key YOUTUBE_API_KEY" in errs
    assert {"key TYPHOON_API_KEY", "key GEMINI_API_KEY", "key AZURE_SPEECH_KEY"} <= set(warns)
    keyed = env(secrets=load_secrets(None, env={
        "YOUTUBE_API_KEY": "yt-key-123456", "TYPHOON_API_KEY": "t-123456",
        "GEMINI_API_KEY": "g-123456", "AZURE_SPEECH_KEY": "az-123456",
        "AZURE_SPEECH_REGION": "southeastasia",
    }))
    assert not [n for n in problems(run_doctor(c, env=keyed, live=False)) if n.startswith("key")]


def test_monetised_flags_non_commercial_voices(root: Path) -> None:
    install(root)
    c = cfg(root, **{"app.monetised": True, "tts.identity_chain": ["premwadee", "offline"]})
    checks = run_doctor(c, env=env(), live=False)
    assert "non-commercial" in problems(checks, "warn")["voice offline"].detail
    plain = run_doctor(cfg(root, **{"tts.identity_chain": ["premwadee", "offline"]}),
                       env=env(), live=False)
    assert "voice offline" not in problems(plain)


def test_missing_extras(root: Path) -> None:
    install(root)
    checks = run_doctor(cfg(root), env=env(find_spec=lambda n: n not in ("soxr", "av")),
                        live=False)
    err = problems(checks, "error")["extra voice"]
    assert "soxr" in err.detail and "--extra voice" in err.hint_en


def test_live_llama_without_tool_caps_and_vts_off(root: Path) -> None:
    install(root)

    def no_tools(url: str, timeout: float) -> tuple[int, Any]:
        if url.endswith("/health"):
            return 200, {}
        return 200, {**PROPS, "chat_template_caps": {"supports_tool_calls": False}}

    checks = run_doctor(cfg(root), env=env(
        http_json=no_tools,
        discover_vts=lambda t: [{"active": False, "port": 8001}],
        gpus=lambda: [{"name": "RTX", "driver": "560.10", "free_mib": 300.0}],
    ))
    errs, warns = problems(checks, "error"), problems(checks, "warn")
    assert "tool" in errs["llama local30b"].detail
    assert "API is off" in warns["vtube studio"].detail
    assert errs["vram"].detail.startswith("300")
    assert "cuda-12.4" in next(c for c in checks if c.name == "gpu").detail


def test_a_crashing_check_is_reported_not_raised(root: Path) -> None:
    install(root)

    def boom(path: Path) -> float:
        raise RuntimeError("disk exploded")

    checks = run_doctor(cfg(root), env=env(disk_free_gb=boom), live=False)
    crashed = problems(checks)["doctor"]
    assert "disk exploded" in crashed.detail
    assert any(c.name == "clock" for c in checks)  # later checks still ran


def test_tts_sample_is_optional(root: Path) -> None:
    install(root)
    sample = {"backends": {"edge": {"n": 2, "ok": 2, "ttfa_p50_s": 0.4}}}
    checks = run_doctor(cfg(root), env=env(tts_sample=lambda c: sample), tts_sample=True)
    assert next(c for c in checks if c.name == "tts edge").ok
    assert not any(c.name.startswith("tts ") for c in run_doctor(cfg(root), env=env()))
