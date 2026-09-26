"""GPU telemetry through ``nvidia-smi`` (§2.10): free VRAM, utilisation, temperature, driver.

``GpuMonitor`` polls every ``interval_s`` on its own thread and alarms (log + callback) when
free VRAM drops below ``alarm_mib`` (500 MiB). Without ``nvidia-smi`` (CI, no NVIDIA driver)
everything reports "unavailable" and nothing fails.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from aivtube.launcher.win32 import CREATE_NO_WINDOW, IS_WINDOWS

__all__ = ["GpuInfo", "GpuMonitor", "parse_nvidia_smi", "query_gpus"]

log = logging.getLogger("aivtube.launcher.gpu")

GpuInfo = dict[str, Any]
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_FIELDS = (
    "index",
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
)
_KEYS = ("index", "name", "driver", "total_mib", "used_mib", "free_mib", "util_pct", "temp_c")


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse ``--query-gpu=<_FIELDS> --format=csv,noheader,nounits`` output."""
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_FIELDS):
            continue
        info: GpuInfo = {}
        for key, raw in zip(_KEYS, parts, strict=False):
            if key in ("name", "driver"):
                info[key] = raw
            elif key == "index":
                info[key] = int(_num(raw) or 0)
            else:
                info[key] = _num(raw)
        gpus.append(info)
    return gpus


def driver_major(driver: str) -> int | None:
    """``"581.42"`` → 581."""
    try:
        return int(driver.split(".", 1)[0])
    except (ValueError, AttributeError):
        return None


def query_gpus(
    exe: str | None = None, *, timeout_s: float = 5.0, runner: Runner | None = None
) -> list[GpuInfo] | None:
    """Every NVIDIA GPU as a dict, or ``None`` when ``nvidia-smi`` is missing or fails."""
    path = exe or shutil.which("nvidia-smi")
    if runner is None and not path:
        return None
    argv = [
        path or "nvidia-smi",
        f"--query-gpu={','.join(_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    run: Runner = runner or subprocess.run
    try:
        done = run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("nvidia-smi failed: %s", exc)
        return None
    if done.returncode != 0:
        log.debug("nvidia-smi exited with %s: %s", done.returncode, (done.stderr or "")[:200])
        return None
    return parse_nvidia_smi(done.stdout or "")


class GpuMonitor(threading.Thread):
    """Polls ``query`` and keeps the latest reading of GPU 0 (see the module docstring)."""

    def __init__(
        self,
        *,
        interval_s: float = 5.0,
        alarm_mib: float = 500.0,
        on_alarm: Callable[[Mapping[str, Any]], None] | None = None,
        query: Callable[[], list[GpuInfo] | None] = query_gpus,
        alarm_every_s: float = 60.0,
    ) -> None:
        super().__init__(name="gpu-monitor", daemon=True)
        self.interval_s = interval_s
        self.alarm_mib = alarm_mib
        self.on_alarm = on_alarm
        self._query = query
        self._alarm_every_s = alarm_every_s
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._latest: GpuInfo = {}
        self.available: bool | None = None
        self.alarms = 0
        self._last_alarm = -1e9

    def latest(self) -> dict[str, Any]:
        """The last reading of GPU 0 (``{}`` before the first or without ``nvidia-smi``)."""
        with self._lock:
            return dict(self._latest)

    def poll_once(self) -> GpuInfo | None:
        gpus = self._query()
        if not gpus:
            if self.available is None:
                log.info("nvidia-smi is not available; no GPU telemetry")
            self.available = False
            return None
        self.available = True
        gpu = dict(gpus[0])
        gpu["t"] = time.perf_counter()
        with self._lock:
            self._latest = gpu
        free = gpu.get("free_mib")
        now = time.perf_counter()
        if (
            isinstance(free, float)
            and free < self.alarm_mib
            and now - self._last_alarm >= self._alarm_every_s
        ):
            self._last_alarm = now
            self.alarms += 1
            log.warning("free VRAM is %.0f MiB (alarm below %.0f MiB)", free, self.alarm_mib)
            if self.on_alarm is not None:
                try:
                    self.on_alarm(gpu)
                except Exception:
                    log.exception("VRAM alarm callback failed")
        return gpu

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        misses = 0
        while not self._halt.is_set():
            try:
                ok = self.poll_once() is not None
            except Exception:
                log.exception("GPU poll failed")
                ok = False
            misses = 0 if ok else misses + 1
            # without nvidia-smi, check again only once a minute
            wait = self.interval_s if ok or misses < 2 else max(self.interval_s, 60.0)
            self._halt.wait(wait)
