"""Helpers for the supervised children (core, voice), standard library only.

The launcher passes these environment variables to every child:

- ``AIVTUBE_BUS_TOKEN``: the core ↔ voice IPC token (new every launcher run).
- ``AIVTUBE_EMERGENCY_TOKEN`` and ``AIVTUBE_LAUNCHER_URL``: the launcher's emergency endpoint
  (``POST /hardkill``, ``/llm/ensure/<server>`` …), for the panel's fallback and the core's
  ``LauncherServerManager``.
- ``AIVTUBE_PANEL_TOKEN``: the token the panel should require (stable across runs, so OBS docks
  and Stream Deck URLs keep working).
- ``AIVTUBE_DUMP_REQUEST``: a file path. When it appears, the child dumps all thread stacks
  with ``faulthandler`` and deletes it (the launcher's wedged-core handling).
- ``AIVTUBE_ROOT`` and ``AIVTUBE_LAUNCHED=1``.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import sys
import threading
from pathlib import Path
from typing import IO

__all__ = [
    "BUS_TOKEN_ENV",
    "DUMP_REQUEST_ENV",
    "EMERGENCY_TOKEN_ENV",
    "LAUNCHED_ENV",
    "LAUNCHER_URL_ENV",
    "PANEL_TOKEN_ENV",
    "ROOT_ENV",
    "start_dump_request_watcher",
]

BUS_TOKEN_ENV = "AIVTUBE_BUS_TOKEN"
EMERGENCY_TOKEN_ENV = "AIVTUBE_EMERGENCY_TOKEN"
PANEL_TOKEN_ENV = "AIVTUBE_PANEL_TOKEN"
LAUNCHER_URL_ENV = "AIVTUBE_LAUNCHER_URL"
DUMP_REQUEST_ENV = "AIVTUBE_DUMP_REQUEST"
ROOT_ENV = "AIVTUBE_ROOT"
LAUNCHED_ENV = "AIVTUBE_LAUNCHED"


def start_dump_request_watcher(
    path: str | Path | None = None,
    *,
    file: IO[str] | None = None,
    interval_s: float = 0.25,
) -> threading.Thread | None:
    """Start a daemon thread that answers the launcher's stack-dump requests.

    ``path`` defaults to ``$AIVTUBE_DUMP_REQUEST`` (``None`` is returned when neither is set);
    ``file`` defaults to ``sys.stderr`` (pass the ``<proc>.fault`` file from
    ``aivtube.infra.logging.fault_file()``). The thread holds no locks and does not use the
    event loop, so it still works while the loop is wedged.
    """
    raw = path if path is not None else os.environ.get(DUMP_REQUEST_ENV, "")
    if not raw:
        return None
    target = Path(raw)
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(interval_s):
            if not target.exists():
                continue
            out = file if file is not None else sys.stderr
            try:
                if out is not None:
                    out.write(f"\n--- stack dump requested by the launcher (pid {os.getpid()}) ---\n")
                    out.flush()
                    faulthandler.dump_traceback(file=out, all_threads=True)
            except (OSError, ValueError, AttributeError):
                pass
            with contextlib.suppress(OSError):
                target.unlink()

    thread = threading.Thread(target=watch, name="dump-request-watcher", daemon=True)
    thread.start()
    return thread
