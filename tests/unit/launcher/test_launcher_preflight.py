"""Launcher preflight: ports, models (size/sha256), Session 0, config, VRAM (§2.6 step 1)."""

from __future__ import annotations

import hashlib
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from aivtube.launcher.preflight import port_in_use, preflight, run_config_check, run_preflight
from aivtube.launcher.settings import LauncherSettings, load_settings

REPO = Path(__file__).resolve().parents[3]
MODEL_BYTES = b"GGUF" + bytes(2000)
VAD_BYTES = b"onnx" + bytes(500)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.toml").write_text(
        f"""schema_version = 1
[models.llm_4b]
url = "https://example.invalid/4b.gguf"
sha256 = "{_sha(MODEL_BYTES)}"
size = {len(MODEL_BYTES)}
licence = "Apache-2.0"
required_by = ["llm.servers.local4b"]
dest = "models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf"

[models.silero_vad]
url = "https://example.invalid/vad.onnx"
sha256 = "{_sha(VAD_BYTES)}"
size = {len(VAD_BYTES)}
licence = "MIT"
required_by = ["vad.model"]
dest = "models/vad/silero_vad.onnx"
""",
        encoding="utf-8",
    )
    return tmp_path


def install(root: Path, *, model: bytes = MODEL_BYTES, vad: bytes = VAD_BYTES,
            exe: bool = True, stt: bool = True) -> None:
    m = root / "models" / "llm" / "typhoon2.5-qwen3-4b.Q4_K_M.gguf"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_bytes(model)
    v = root / "models" / "vad" / "silero_vad.onnx"
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_bytes(vad)
    if exe:
        e = root / "vendor" / "llama.cpp" / "llama-server.exe"
        e.parent.mkdir(parents=True, exist_ok=True)
        e.write_bytes(b"MZ")
    if stt:
        t = root / "models" / "stt" / "typhoon-asr-rt-int8" / "tokens.txt"
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("<blk> 0\n", encoding="utf-8")


def settings(root: Path, **overrides: Any) -> LauncherSettings:
    return load_settings(root, profile="light", cli_overrides=overrides, env={})


def run(s: LauncherSettings, **kw: Any) -> list[Any]:
    base: dict[str, Any] = {
        "check_config": None,
        "session": lambda: None,
        "in_use": lambda port: False,
        "gpus": lambda: [{"free_mib": 9000.0}],
        "props": lambda port: None,
        "health": lambda port: 0,
    }
    return run_preflight(s, **{**base, **kw})


def fatal(problems: list[Any]) -> list[str]:
    return [p.key for p in problems if p.fatal]


@pytest.fixture
def listener() -> Iterator[int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield int(sock.getsockname()[1])
    sock.close()


def test_all_good(root: Path) -> None:
    install(root)
    s = settings(root)
    assert s.autostart_servers() == ["local4b"]
    assert run(s) == []
    assert preflight(s, check_config=None, session=lambda: None, in_use=lambda p: False,
                     gpus=lambda: [{"free_mib": 9000.0}]) == []


def test_port_in_use_is_detected(listener: int) -> None:
    assert port_in_use(listener) is True
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = int(s.getsockname()[1])
    assert port_in_use(free) is False


def test_ports_in_use_fail_with_bilingual_hint(root: Path, listener: int) -> None:
    install(root)
    s = settings(root, **{"ports.panel": listener})
    problems = run_preflight(s, check_config=None, session=lambda: None,
                             gpus=lambda: [{"free_mib": 9000.0}])
    assert fatal(problems) == ["port:panel"]
    text = str(problems[0])
    assert str(listener) in text and "พอร์ต" in text and "hint:" in text


def test_missing_and_corrupt_models(root: Path) -> None:
    s = settings(root)
    problems = run(s)
    assert set(fatal(problems)) == {"exe:llm.local4b", "model:llm.servers.local4b",
                                    "model:vad.model"}
    install(root, model=b"GGUF" + b"\x01" * 2000)  # right size, wrong content
    problems = run(s)
    assert fatal(problems) == ["model:llm.servers.local4b"]
    assert "damaged" in problems[0].message_en and "sha256" in problems[0].message_en
    install(root, vad=b"short")
    assert "size" in str(next(p for p in run(s) if p.key == "model:vad.model"))


def test_missing_stt_model_is_only_a_warning(root: Path) -> None:
    install(root, stt=False)
    problems = run(settings(root))
    assert [(p.key, p.fatal) for p in problems] == [("model:stt.typhoon_rt", False)]


def test_voice_off_or_fakes_skip_the_vad_model(root: Path) -> None:
    install(root)
    (root / "models" / "vad" / "silero_vad.onnx").unlink()
    assert fatal(run(settings(root, **{"app.voice_worker": False}))) == []
    faked = settings(root, **{"app.fakes": True})
    assert faked.autostart_servers() == []
    assert fatal(run(faked)) == []


def test_session0_is_refused(root: Path) -> None:
    install(root)
    problems = run(settings(root), session=lambda: 0)
    assert fatal(problems) == ["session0"]
    assert "Session 0" in problems[0].message_th


def test_llama_port_taken_by_an_adoptable_server_is_fine(root: Path) -> None:
    s = settings(root)  # nothing installed: adoption needs neither exe nor model
    port = s.servers["local4b"]["port"]
    props = {"model_alias": "pailin-4b", "model_path": "x/typhoon2.5-qwen3-4b.Q4_K_M.gguf"}
    problems = run(s, in_use=lambda p: p == port, props=lambda p: props)
    assert "port:llm.local4b" not in fatal(problems)
    assert "exe:llm.local4b" not in fatal(problems)


def test_llama_port_taken_by_another_server(root: Path) -> None:
    install(root)
    s = settings(root)
    port = s.servers["local4b"]["port"]
    props = {"model_alias": "someone-else", "model_path": "other.gguf"}
    problems = run(s, in_use=lambda p: p == port, props=lambda p: props)
    assert fatal(problems) == ["port:llm.local4b"]
    assert "someone-else" in problems[0].message_en


def test_llama_port_with_a_server_still_loading_is_only_a_warning(root: Path) -> None:
    s = settings(root)
    port = s.servers["local4b"]["port"]
    problems = run(s, in_use=lambda p: p == port, health=lambda p: 503 if p == port else 0)
    assert "port:llm.local4b" not in fatal(problems)
    warn = [p for p in problems if p.key == "port:llm.local4b"]
    assert warn and not warn[0].fatal and "still loading" in warn[0].message_en
    # something else that is not llama-server (no /health answer) is still fatal
    problems = run(s, in_use=lambda p: p == port)
    assert "port:llm.local4b" in fatal(problems)


def test_low_vram_is_only_a_warning(root: Path) -> None:
    install(root)
    s = settings(root, **{"llm.servers.local4b.placement": "fit"})
    problems = run(s, gpus=lambda: [{"free_mib": 1200.0}])
    assert fatal(problems) == []
    assert [p.key for p in problems] == ["vram"]
    assert [p.key for p in run(s, gpus=lambda: None)] == ["vram"]


def test_config_errors_come_from_the_subprocess(root: Path) -> None:
    install(root)
    (root / "config" / "user.toml").write_text(
        "schema_version = 1\n[ports]\npanel = 8001\n", encoding="utf-8"
    )
    s = settings(root)
    errors = run_config_check(s, {})
    assert errors and errors[0]["path"].startswith("ports")
    problems = run_preflight(s, session=lambda: None, in_use=lambda p: False,
                             gpus=lambda: [{"free_mib": 9000.0}])
    keys = fatal(problems)
    assert keys and keys[0].startswith("config:")
    assert "VTube Studio" in str(problems[0])
    assert "ไฟล์ตั้งค่าผิด" in str(problems[0])


def test_valid_config_passes_the_subprocess_check(root: Path) -> None:
    assert run_config_check(settings(root), {"app.fakes": True}) == []
