"""Import hygiene of aivtube.contracts: stdlib-only types, lazy numpy, no cycles, exports."""

from __future__ import annotations

import ast
import graphlib
import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aivtube.contracts as contracts

PKG_DIR = Path(contracts.__file__).parent
MODULES = sorted(p.stem for p in PKG_DIR.glob("*.py") if p.stem != "__init__")
# Native or optional stacks that must never be imported by a contract module.
FORBIDDEN = {
    "sounddevice",
    "sherpa_onnx",
    "onnxruntime",
    "av",
    "edge_tts",
    "azure",
    "livekit",
    "soxr",
    "websockets",
    "aiohttp",
    "pydantic",
}
# The only third-party imports allowed, per module.
ALLOWED_THIRD_PARTY = {"voice": {"numpy"}, "ipc": {"jsonschema"}}


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=60,
        check=False,
    )


def _imports(path: Path) -> list[str]:
    """Absolute module names imported anywhere in the file (incl. TYPE_CHECKING/functions)."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: use absolute imports"
            assert node.module is not None
            if node.module == "aivtube.contracts":
                names.extend(f"aivtube.contracts.{a.name}" for a in node.names)
            else:
                names.append(node.module)
    return names


def test_expected_modules_exist() -> None:
    assert set(MODULES) == {
        "types",
        "events",
        "infra",
        "voice",
        "speech",
        "avatar",
        "llm",
        "chat",
        "memory",
        "safety",
        "tools",
        "control",
        "games",
        "ipc",
    }


def test_types_imports_only_the_stdlib() -> None:
    tops = {name.split(".")[0] for name in _imports(PKG_DIR / "types.py")}
    assert tops <= set(sys.stdlib_module_names), tops - set(sys.stdlib_module_names)


def test_importing_contracts_types_does_not_import_numpy() -> None:
    proc = _run(
        """
        import sys
        import aivtube.contracts.types
        assert "numpy" not in sys.modules, "numpy imported by aivtube.contracts.types"
        import aivtube.contracts
        assert "numpy" not in sys.modules, "numpy imported by aivtube.contracts"
        assert "jsonschema" not in sys.modules, "jsonschema imported eagerly"
        assert "sounddevice" not in sys.modules
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_voice_names_are_exported_lazily() -> None:
    proc = _run(
        """
        import sys
        import aivtube.contracts as c
        assert "numpy" not in sys.modules
        from aivtube.contracts import AudioOut, F32
        import aivtube.contracts.voice as v
        assert "numpy" in sys.modules
        assert F32 is v.F32 and AudioOut is v.AudioOut
        assert "sounddevice" not in sys.modules
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_main_stub_imports_cli_lazily() -> None:
    proc = _run(
        """
        import sys
        import aivtube.__main__
        assert "aivtube.cli" not in sys.modules
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("module", MODULES)
def test_no_forbidden_or_unexpected_third_party_imports(module: str) -> None:
    stdlib = set(sys.stdlib_module_names)
    for name in _imports(PKG_DIR / f"{module}.py"):
        top = name.split(".")[0]
        assert top not in FORBIDDEN, f"{module} imports {name}"
        if top in stdlib or top == "aivtube":
            if top == "aivtube":
                assert name.startswith("aivtube.contracts."), f"{module} imports {name}"
            continue
        assert top in ALLOWED_THIRD_PARTY.get(module, set()), f"{module} imports {name}"


def test_jsonschema_is_imported_lazily_by_ipc() -> None:
    tree = ast.parse((PKG_DIR / "ipc.py").read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in top_level:
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
        assert all(not (m or "").startswith("jsonschema") for m in mods)


def test_import_graph_has_no_cycles() -> None:
    graph: dict[str, set[str]] = {}
    for module in MODULES:
        deps = {
            name.removeprefix("aivtube.contracts.")
            for name in _imports(PKG_DIR / f"{module}.py")
            if name.startswith("aivtube.contracts.")
        }
        graph[module] = deps & set(MODULES)
    order = list(graphlib.TopologicalSorter(graph).static_order())  # raises CycleError
    assert order.index("types") < order.index("events") < order.index("infra")
    assert graph["types"] == set()
    assert graph["ipc"] == set()


@pytest.mark.parametrize("module", ["__init__", *MODULES])
def test_all_exports_resolve(module: str) -> None:
    name = "aivtube.contracts" if module == "__init__" else f"aivtube.contracts.{module}"
    mod = importlib.import_module(name)
    exported = list(mod.__all__)
    assert len(exported) == len(set(exported)), "duplicate names in __all__"
    for attr in exported:
        assert hasattr(mod, attr), f"{name}.{attr}"


def test_package_reexports_are_the_same_objects() -> None:
    for module in MODULES:
        if module == "ipc":
            continue  # only the envelope types are re-exported at package level
        mod = importlib.import_module(f"aivtube.contracts.{module}")
        for attr in mod.__all__:
            assert getattr(contracts, attr) is getattr(mod, attr), f"{module}.{attr}"
    from aivtube.contracts import ipc

    for attr in ("IPC_VERSION", "MESSAGE_SCHEMAS", "Envelope", "IpcError"):
        assert getattr(contracts, attr) is getattr(ipc, attr)
    assert "F32" in dir(contracts)
