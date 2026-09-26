"""Raw layering helpers (stdlib only) and import hygiene."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aivtube.config import ConfigError, deep_merge, env_overrides, expand_character, find_root
from aivtube.config.layers import expand_dotted, parse_env_value

REPO = Path(__file__).resolve().parents[3]


def test_deep_merge_merges_tables_replaces_lists_and_copies() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    over = {"a": {"c": [3], "e": {"f": 1}}, "g": 2}
    merged = deep_merge(base, over)
    assert merged == {"a": {"b": 1, "c": [3], "e": {"f": 1}}, "d": 1, "g": 2}
    merged["a"]["e"]["f"] = 99
    merged["a"]["c"].append(4)
    assert over == {"a": {"c": [3], "e": {"f": 1}}, "g": 2}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 1}


def test_deep_merge_scalar_replaces_table_and_back() -> None:
    assert deep_merge({"a": {"b": 1}}, {"a": 3}) == {"a": 3}
    assert deep_merge({"a": 3}, {"a": {"b": 1}}) == {"a": {"b": 1}}


def test_expand_dotted() -> None:
    assert expand_dotted({"a.b.c": 1, "a.b.d": 2, "x": {"y.z": 3}}) == {
        "a": {"b": {"c": 1, "d": 2}},
        "x": {"y": {"z": 3}},
    }
    with pytest.raises(ConfigError):
        expand_dotted({"a..b": 1})


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("8080", 8080),
        ("0.5", 0.5),
        ("true", True),
        ('["a", "b"]', ["a", "b"]),
        ('"quoted"', "quoted"),
        ("ptt", "ptt"),
        ("Headphones (USB)", "Headphones (USB)"),
        ("", ""),
        ("{ a = 1 }", {"a": 1}),
    ],
)
def test_parse_env_value(raw: str, value: object) -> None:
    assert parse_env_value(raw) == value


def test_env_overrides_nest_and_ignore_other_variables() -> None:
    env = {
        "AIVTUBE__LLM__SERVERS__LOCAL30B__PORT": "8090",
        "AIVTUBE__MIC__MODE": "ptt",
        "AIVTUBE_BUS_TOKEN": "x",
        "PATH": "/bin",
    }
    assert env_overrides(env) == {
        "llm": {"servers": {"local30b": {"port": 8090}}},
        "mic": {"mode": "ptt"},
    }


def test_expand_character_walks_nested_values() -> None:
    data = {"a": "data/{character}.db", "b": ["{character}", 3], "c": {"d": "x{character}"}}
    assert expand_character(data, "pailin") == {
        "a": "data/pailin.db",
        "b": ["pailin", 3],
        "c": {"d": "xpailin"},
    }


def test_find_root(tmp_path: Path) -> None:
    assert find_root(REPO / "src" / "aivtube") == REPO
    assert find_root(tmp_path) == REPO  # falls back to the package's own checkout


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True
    )
    return out.stdout.strip()


def test_layers_import_without_pydantic() -> None:
    code = (
        "import sys, aivtube.config.layers, aivtube.config.errors; "
        "print(sorted(m for m in ('pydantic', 'pydantic_core') if m in sys.modules))"
    )
    assert _run(code) == "[]"


def test_config_and_infra_import_no_heavy_native_deps() -> None:
    heavy = (
        "sounddevice",
        "sherpa_onnx",
        "onnxruntime",
        "av",
        "edge_tts",
        "azure",
        "livekit",
        "soxr",
        "numpy",
    )
    code = (
        "import sys, aivtube.config, aivtube.infra; "
        "from aivtube.config import AppConfig, load_config; AppConfig(); "
        f"print(sorted(m for m in {heavy!r} if m in sys.modules))"
    )
    assert _run(code) == "[]"
