"""The launcher (P0, ARCHITECTURE.md §2.1–§2.3, §2.6–§2.11). Standard library only.

It runs the preflight, starts or adopts llama-server, supervises the core and the voice
worker, holds the Windows Job Object, serves the emergency hard-kill endpoint, reads console
keys, polls GPU telemetry and keeps the PC awake. It must never crash on native code, so
nothing under this package imports a third-party module (a test enforces it).

Names load lazily, so a child importing ``aivtube.launcher.childside`` pulls in nothing else.
The entry point is ``aivtube.launcher.main.main`` and the preflight
``aivtube.launcher.preflight.preflight`` (both share their submodule's name, so they are not
re-exported here).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aivtube.launcher.console import ConsoleKeys
    from aivtube.launcher.emergency import EmergencyServer
    from aivtube.launcher.gpu import GpuMonitor
    from aivtube.launcher.jobobject import JobObject
    from aivtube.launcher.llama import LlamaServerController
    from aivtube.launcher.main import Launcher
    from aivtube.launcher.supervisor import ProcessSpec, Supervisor
    from aivtube.launcher.win32 import keep_awake

_LAZY: dict[str, str] = {
    "ConsoleKeys": "console",
    "EmergencyServer": "emergency",
    "GpuMonitor": "gpu",
    "JobObject": "jobobject",
    "Launcher": "main",
    "LlamaServerController": "llama",
    "ProcessSpec": "supervisor",
    "Supervisor": "supervisor",
    "keep_awake": "win32",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ConsoleKeys",
    "EmergencyServer",
    "GpuMonitor",
    "JobObject",
    "Launcher",
    "LlamaServerController",
    "ProcessSpec",
    "Supervisor",
    "keep_awake",
]
