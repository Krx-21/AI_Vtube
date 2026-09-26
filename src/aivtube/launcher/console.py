"""Launcher console keys (§2.11, Appendix C): ``msvcrt`` on Windows, a stdin line reader elsewhere.

Default bindings (see ``KEY_HELP``): ``K`` hard kill, ``A`` rearm, ``F`` freeze, ``R`` retry
FAILED components, ``S`` status, ``Q`` quit, ``H``/``?`` help. The handlers are callbacks, so the
thread only reads keys; a failing handler is logged and the reader keeps going.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable, Mapping
from typing import IO

from aivtube.launcher.win32 import IS_WINDOWS

__all__ = ["KEY_HELP", "ConsoleKeys"]

log = logging.getLogger("aivtube.launcher.console")

KEY_HELP: Mapping[str, str] = {
    "k": "HARD KILL: silence the voice worker now / ตัดเสียงทันที (ค้างไว้จนกด A)",
    "a": "rearm the voice worker / เปิดเสียงกลับมา",
    "f": "FREEZE the brain / หยุดสมองทั้งหมด",
    "r": "retry FAILED components / ลองเริ่มส่วนที่ล้มเหลวใหม่",
    "s": "status / แสดงสถานะ",
    "q": "quit / ออก",
    "h": "help / วิธีใช้",
}


def help_text() -> str:
    return "\n".join(f"  [{k.upper()}] {v}" for k, v in KEY_HELP.items())


class ConsoleKeys(threading.Thread):
    """Reads single keys and calls ``handlers[key]`` (keys are lower-cased)."""

    def __init__(
        self,
        handlers: Mapping[str, Callable[[], None]],
        *,
        stdin: IO[str] | None = None,
        use_msvcrt: bool | None = None,
        poll_s: float = 0.05,
    ) -> None:
        super().__init__(name="console-keys", daemon=True)
        self.handlers = {k.lower(): v for k, v in handlers.items()}
        self._stdin = stdin
        self._msvcrt = IS_WINDOWS and stdin is None if use_msvcrt is None else use_msvcrt
        self._poll_s = poll_s
        self._halt = threading.Event()
        self.seen: list[str] = []

    def stop(self) -> None:
        self._halt.set()

    def dispatch(self, key: str) -> bool:
        """Run the handler for ``key``; whether one existed."""
        k = key.strip().lower()[:1]
        if k == "?":
            k = "h"
        handler = self.handlers.get(k)
        if handler is None:
            return False
        self.seen.append(k)
        try:
            handler()
        except Exception:
            log.exception("console key %r failed", k)
        return True

    def run(self) -> None:
        if self._msvcrt:
            self._run_msvcrt()
        else:
            self._run_lines()

    def _run_msvcrt(self) -> None:
        import msvcrt

        while not self._halt.wait(self._poll_s):
            try:
                while msvcrt.kbhit():  # type: ignore[attr-defined,unused-ignore]
                    ch = msvcrt.getwch()  # type: ignore[attr-defined,unused-ignore]
                    if ch in ("\x00", "\xe0"):  # function/arrow key prefix
                        msvcrt.getwch()  # type: ignore[attr-defined,unused-ignore]
                        continue
                    self.dispatch(ch)
            except OSError:
                log.warning("console input is not available; keys disabled")
                return

    def _run_lines(self) -> None:
        stream = self._stdin if self._stdin is not None else sys.stdin
        if stream is None:
            return
        while not self._halt.is_set():
            try:
                line = stream.readline()
            except (OSError, ValueError):
                return
            if not line:
                return  # EOF
            if line.strip():
                self.dispatch(line)
