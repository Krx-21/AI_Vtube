"""Packaging scaffolding: model manifest, fixtures, install scripts, workflows, licences, tools."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = {
    "url": str,
    "sha256": str,
    "size": int,
    "licence": str,
    "required_by": list,
    "dest": str,
}


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tool_{name}", ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    with (ROOT / "models" / "manifest.toml").open("rb") as f:
        return tomllib.load(f)


def test_manifest_schema(manifest: dict[str, Any]) -> None:
    assert manifest["schema_version"] == 1
    models = manifest["models"]
    assert {"silero_vad", "llm_4b", "llm_30b", "typhoon_rt_encoder", "typhoon_rt_tokens"} <= set(
        models
    )
    for name, entry in models.items():
        for key, typ in REQUIRED.items():
            assert isinstance(entry.get(key), typ), f"{name}.{key}"
        assert entry["url"].startswith("https://"), name
        assert "/master/" not in entry["url"] and "/main/" not in entry["url"], (
            f"{name}: unpinned URL"
        )
        assert entry["sha256"] == "" or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert entry["size"] >= 0 and (entry["sha256"] == "") == (entry["size"] == 0), name
        dest = Path(entry["dest"])
        assert not dest.is_absolute() and ".." not in dest.parts, name
        assert dest.parts[0] in ("models", "vendor"), name
        assert entry["required_by"] and all(isinstance(r, str) for r in entry["required_by"])


def test_manifest_llama_variants_and_typhoon_release(manifest: dict[str, Any]) -> None:
    models = manifest["models"]
    variants = {e.get("variant") for e in models.values() if e.get("unpack") == "zip"}
    assert variants == {"cuda-13.4", "cuda-12.4"}
    for name in (
        "typhoon_rt_encoder",
        "typhoon_rt_decoder",
        "typhoon_rt_joiner",
        "typhoon_rt_tokens",
    ):
        assert "/releases/download/models-typhoon-rt-v1/" in models[name]["url"]
        assert models[name]["licence"] == "CC-BY-4.0"
    for name in ("llm_4b", "llm_30b"):
        assert "huggingface.co/mradermacher/" in models[name]["url"]


def test_silero_fixture_matches_manifest(manifest: dict[str, Any], silero_onnx: Path) -> None:
    entry = manifest["models"]["silero_vad"]
    data = silero_onnx.read_bytes()
    assert len(data) == entry["size"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert (silero_onnx.parent / "silero_vad.LICENSE.txt").read_text(encoding="utf-8").count("MIT")


def _doc_block(lang: str) -> str:
    text = (ROOT / "docs" / "design" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    m = re.search(rf"```{lang}\n(.*?)```", text, re.S)
    assert m, lang
    return m.group(1)


@pytest.mark.parametrize(("path", "lang"), [("setup.ps1", "powershell"), ("run.bat", "bat")])
def test_install_scripts_match_section_9(path: str, lang: str) -> None:
    raw = (ROOT / path).read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    assert text == _doc_block(lang)
    assert b"\r\n" in raw, "Windows scripts use CRLF"
    assert raw.startswith(b"\xef\xbb\xbf") == (path == "setup.ps1"), "BOM only for PowerShell"


def test_ci_workflow() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for needle in (
        "permissions:\n  contents: read",
        "astral-sh/setup-uv@",
        "uv sync --frozen --extra voice --extra aec --extra dev",
        "sudo apt-get install -y libportaudio2",
        "uv run --frozen ruff check",
        "uv run --frozen mypy",
        'pytest -m "not nightly and not hardware"',
    ):
        assert needle in text, needle
    matrix = re.findall(r"- os: (\S+)\n\s+python-version: \"([\d.]+)\"", text)
    assert matrix == [
        ("ubuntu-latest", "3.11"),
        ("ubuntu-latest", "3.12"),
        ("windows-latest", "3.12"),
    ]
    order = [text.index(s) for s in ("uv sync", "ruff check", "mypy", "pytest -m")]
    assert order == sorted(order)


def test_models_export_workflow() -> None:
    text = (ROOT / ".github" / "workflows" / "models-export.yml").read_text(encoding="utf-8")
    for needle in (
        "workflow_dispatch:",
        "runs-on: ubuntu-latest",
        "permissions:\n  contents: write",
        "https://download.pytorch.org/whl/cpu",
        '"nemo_toolkit[asr]==3.0.0"',
        "tools/export_typhoon_rt_onnx.py",
        "tools/verify_typhoon_rt_onnx.py",
        "gh release create",
        "gh release upload",
        "secrets.GITHUB_TOKEN",
        "default: models-typhoon-rt-v1",
    ):
        assert needle in text, needle
    assert "${{ inputs." not in text.split("steps:")[1], "inputs must reach scripts via env"


def test_licence_and_notice() -> None:
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("MIT License") and "Copyright (c) 2025-2026 Krx-21" in licence
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for needle in (
        "typhoon-ai/typhoon-asr-realtime",
        "CC-BY-4.0",
        "Silero",
        "Live2D",
        "compatible with the Neuro Game SDK protocol",
    ):
        assert needle in notice, needle


def test_plugin_entry_point_is_a_no_op() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        eps = tomllib.load(f)["project"]["entry-points"]["aivtube.plugins"]
    module, _, attr = eps["builtin"].partition(":")
    register = getattr(importlib.import_module(module), attr)
    assert register(object()) is None


def test_export_tool_is_lazy_and_writes_attribution(tmp_path: Path) -> None:
    before = set(sys.modules)
    mod = _load_tool("export_typhoon_rt_onnx")
    assert not {"torch", "nemo", "onnx"} & (set(sys.modules) - before)
    for name in mod.OUTPUTS:
        (tmp_path / name).write_bytes(name.encode())
    assert mod.main(["--out", str(tmp_path), "--readme-only"]) == 0
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "CC-BY-4.0" in readme and "typhoon-ai/typhoon-asr-realtime" in readme
    assert mod.DEFAULT_REVISION in readme
    sums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(sums) == 4 and sums[-1].endswith("  tokens.txt")
    assert mod.main(["--out", str(tmp_path / "empty"), "--readme-only"]) == 1


def test_verify_tool_cer() -> None:
    mod = _load_tool("verify_typhoon_rt_onnx")
    assert mod.cer("สวัสดี ค่ะ", "สวัสดีค่ะ") == 0.0
    assert mod.cer("abcd", "abxd") == pytest.approx(0.25)
    assert mod.cer("abc", "") == 1.0
