"""mypy-level Protocol conformance for the fakes and the real infra classes.

``isinstance`` against a ``runtime_checkable`` Protocol only checks that members exist;
``aivtube/testing/_conformance.py`` assigns each implementation to its Protocol so mypy checks
signatures, keyword names, defaults and attribute types as well.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE = ROOT / "src" / "aivtube" / "testing" / "_conformance.py"


def test_fakes_and_infra_statically_satisfy_their_protocols() -> None:
    pytest.importorskip("mypy")
    out = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(CONFORMANCE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    errors = [line for line in out.stdout.splitlines() if ": error:" in line or ": note:" in line]
    errors = [line for line in errors if "unused section" not in line]
    assert out.returncode == 0, "\n".join(errors) or out.stdout + out.stderr


def test_conformance_module_covers_every_protocol() -> None:
    import importlib

    names: set[str] = set()
    for mod_name in (
        "infra",
        "voice",
        "speech",
        "llm",
        "avatar",
        "chat",
        "memory",
        "safety",
        "tools",
        "control",
        "games",
    ):
        mod = importlib.import_module(f"aivtube.contracts.{mod_name}")
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == mod.__name__
                and getattr(obj, "_is_protocol", False)
            ):
                names.add(f"{mod_name}.{obj.__name__}:")
    text = CONFORMANCE.read_text(encoding="utf-8")
    missing = sorted(n.rstrip(":") for n in names if f"-> {n}" not in text)
    assert not missing, f"add static checks for: {missing}"
