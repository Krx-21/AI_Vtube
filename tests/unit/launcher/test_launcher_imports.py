"""The launcher must import only the standard library (§2.3): it can never crash on native code."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "src" / "aivtube" / "launcher"

# first-party modules the launcher may use; each is itself standard-library only
ALLOWED_FIRST_PARTY = {
    "aivtube",
    "aivtube.config",
    "aivtube.config.errors",
    "aivtube.config.layers",
    "aivtube.llm",
    "aivtube.llm.llama_args",
    "aivtube.ops",
    "aivtube.ops.models",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_every_import_in_the_source_is_stdlib_or_allowed() -> None:
    bad: dict[str, list[str]] = {}
    for path in sorted(LAUNCHER.glob("*.py")):
        for mod in _imports(path):
            top = mod.split(".")[0]
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            if mod.startswith("aivtube.launcher") or mod in ALLOWED_FIRST_PARTY:
                continue
            bad.setdefault(path.name, []).append(mod)
    assert bad == {}


PROBE = """
import sys
before = set(sys.modules)  # whatever site/sitecustomize loaded at startup
import importlib, json, pkgutil
import aivtube.launcher as pkg
for info in pkgutil.iter_modules(pkg.__path__):
    importlib.import_module(f"aivtube.launcher.{info.name}")
import aivtube.ops.models
from aivtube.launcher.main import build_parser
build_parser()
mods = sorted(set(sys.modules) - before)
third = sorted({m.split('.')[0] for m in mods} - set(sys.stdlib_module_names) - {"aivtube"})
first = sorted(m for m in mods if m.startswith("aivtube") and not m.startswith("aivtube.launcher"))
print(json.dumps({"third": third, "first": first}))
"""


def test_importing_the_launcher_loads_no_third_party_module() -> None:
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                          check=True, cwd=REPO)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["third"] == [], result["third"]
    assert set(result["first"]) <= ALLOWED_FIRST_PARTY, set(result["first"]) - ALLOWED_FIRST_PARTY


def test_allowed_first_party_modules_are_stdlib_only() -> None:
    for mod in ALLOWED_FIRST_PARTY - {"aivtube", "aivtube.config", "aivtube.llm", "aivtube.ops"}:
        rel = Path(*mod.split(".")).with_suffix(".py")
        path = REPO / "src" / rel
        for imported in _imports(path):
            top = imported.split(".")[0]
            ok = top in sys.stdlib_module_names or top == "__future__" or imported.startswith(
                "aivtube."
            )
            assert ok, f"{mod} imports {imported}"
