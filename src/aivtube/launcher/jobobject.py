"""Windows Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` (§2.3), via ``ctypes``.

Every child is assigned to the job right after it starts. When the launcher exits for any
reason (including a crash or ``TerminateProcess``), Windows closes the job handle and kills
every process still in it, so no orphan keeps the mic, the audio device or VRAM.
``JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION`` also stops a crashed native child from hanging
on a Windows Error Reporting dialog instead of exiting (the supervisor then restarts it).

On other platforms the class is a no-op (``active`` is ``False``).
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

__all__ = ["JobObject"]

log = logging.getLogger("aivtube.launcher.job")

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


def _structs() -> tuple[Any, Any]:
    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return ExtendedLimit, wintypes


class JobObject:
    """Owns one kill-on-close job. ``assign(pid)`` never raises; failures are logged."""

    def __init__(self) -> None:
        self._handle: Any = None
        self._k32: Any = None
        self._lock = threading.Lock()
        self.assigned: list[int] = []
        if sys.platform != "win32":
            return
        try:
            self._create()
        except OSError:
            log.warning("could not create the Job Object; children may outlive a crash", exc_info=True)
            self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _create(self) -> None:
        import ctypes

        extended_limit, wintypes = _structs()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined,unused-ignore]
        info = extended_limit()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        ok = k32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()  # type: ignore[attr-defined,unused-ignore]
            k32.CloseHandle(handle)
            raise ctypes.WinError(err)  # type: ignore[attr-defined,unused-ignore]
        self._k32 = k32
        self._handle = handle

    def assign(self, pid: int) -> bool:
        """Put process ``pid`` into the job. Returns ``False`` (logged) when it cannot."""
        with self._lock:
            if self._handle is None:
                return False
            import ctypes

            proc = self._k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            if not proc:
                log.warning(
                    "OpenProcess(%d) failed (error %d); not in the Job Object",
                    pid,
                    ctypes.get_last_error(),  # type: ignore[attr-defined,unused-ignore]
                )
                return False
            try:
                if not self._k32.AssignProcessToJobObject(self._handle, proc):
                    log.warning(
                        "AssignProcessToJobObject(%d) failed (error %d)",
                        pid,
                        ctypes.get_last_error(),  # type: ignore[attr-defined,unused-ignore]
                    )
                    return False
            finally:
                self._k32.CloseHandle(proc)
            self.assigned.append(pid)
            return True

    def close(self) -> None:
        """Close the job handle: Windows kills every process still assigned to it."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                self._k32.CloseHandle(handle)

    def __enter__(self) -> JobObject:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
