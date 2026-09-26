"""Small Windows helpers over ``ctypes`` (standard library only; no-ops elsewhere).

- :func:`keep_awake`: ``SetThreadExecutionState`` so the PC does not sleep mid-stream (§2.9).
- :func:`session_id`: the Terminal Services session of this process. Session 0 (services, SSH)
  cannot create a CUDA context or open audio devices, so preflight refuses it (§2.6, §9).
- :func:`process_creation_flags`: ``CREATE_NEW_PROCESS_GROUP`` (so ``CTRL_BREAK_EVENT`` reaches
  one child only) plus the priority class.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

__all__ = [
    "ABOVE_NORMAL_PRIORITY_CLASS",
    "CREATE_NEW_PROCESS_GROUP",
    "CREATE_NO_WINDOW",
    "ES_CONTINUOUS",
    "ES_DISPLAY_REQUIRED",
    "ES_SYSTEM_REQUIRED",
    "IS_WINDOWS",
    "keep_awake",
    "process_creation_flags",
    "session_id",
]

log = logging.getLogger("aivtube.launcher.win32")

IS_WINDOWS = sys.platform == "win32"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
HIGH_PRIORITY_CLASS = 0x00000080

_PRIORITY = {
    "normal": 0,
    "above_normal": ABOVE_NORMAL_PRIORITY_CLASS,
    "below_normal": BELOW_NORMAL_PRIORITY_CLASS,
    "high": HIGH_PRIORITY_CLASS,
}


def _kernel32() -> Any:
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]


def keep_awake(on: bool) -> bool:
    """Keep the system and display awake (``on``) or restore normal sleep. Call it from a
    thread that lives as long as the launcher (the state belongs to the calling thread).
    Returns whether the call took effect (always ``False`` off Windows)."""
    if not IS_WINDOWS:
        return False
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else ES_CONTINUOUS
    try:
        k32 = _kernel32()
        import ctypes

        k32.SetThreadExecutionState.restype = ctypes.c_uint32
        k32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        ok = bool(k32.SetThreadExecutionState(flags))
    except (OSError, AttributeError):
        log.warning("SetThreadExecutionState is not available", exc_info=True)
        return False
    if not ok:
        log.warning("SetThreadExecutionState(%#x) failed", flags)
    return ok


def session_id(pid: int | None = None) -> int | None:
    """The Windows session id of ``pid`` (default: this process); ``None`` off Windows or on
    error. Session 0 is the non-interactive services session."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = _kernel32()
        k32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k32.ProcessIdToSessionId.restype = wintypes.BOOL
        out = wintypes.DWORD(0)
        if not k32.ProcessIdToSessionId(pid if pid is not None else os.getpid(), ctypes.byref(out)):
            return None
        return int(out.value)
    except (OSError, AttributeError):
        return None


def process_creation_flags(priority: str = "normal", *, new_group: bool = True) -> int:
    """``creationflags`` for ``subprocess.Popen`` on Windows (``0`` elsewhere)."""
    if not IS_WINDOWS:
        return 0
    flags = CREATE_NEW_PROCESS_GROUP if new_group else 0
    return flags | _PRIORITY.get(priority, 0)
