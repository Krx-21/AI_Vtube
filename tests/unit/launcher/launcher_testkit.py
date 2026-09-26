"""Shared helpers for the launcher tests."""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUMMY_CHILD = HERE / "dummy_child.py"
DUMMY_LLAMA = HERE / "dummy_llama.py"


def child_argv(*args: str) -> list[str]:
    return [sys.executable, str(DUMMY_CHILD), *args]


def wait_until(cond: Callable[[], bool], timeout: float = 10.0, step: float = 0.02) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def lines(path: Path) -> list[str]:
    try:
        return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []
