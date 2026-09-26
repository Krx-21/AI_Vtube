"""Process logging (§2.10): QueueHandler → ``logs/<date>/<proc>.log`` + ``.jsonl``.

Records are redacted on the emitting thread (configured secrets, ``oauth:…``, ``Bearer …``,
API keys in URLs and headers), then written by a ``QueueListener`` thread so logging never
blocks the event loop on disk I/O. Files rotate at 20 MB × 5; day folders older than
``keep_days`` are removed. ``faulthandler`` writes native crash stacks to ``<proc>.fault``.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import datetime as _dt
import faulthandler
import json
import logging
import logging.handlers
import queue
import re
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

__all__ = [
    "JsonFormatter",
    "RedactionFilter",
    "Redactor",
    "add_secrets",
    "current_redactor",
    "fault_file",
    "setup_logging",
    "shutdown_logging",
]

_MASK = "***"
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(oauth:)[a-z0-9]+"), r"\1" + _MASK),
    (re.compile(r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]+"), r"\1" + _MASK),
    (
        re.compile(r"(?i)([?&](?:key|api_?key|access_token|token|client_secret|sig)=)[^&\s\"']+"),
        r"\1" + _MASK,
    ),
    (
        re.compile(
            r"(?i)((?:ocp-apim-subscription-key|x-api-key|x-goog-api-key)[\"']?\s*[:=]\s*"
            r"[\"']?)[^\s\"',;]+"
        ),
        r"\1" + _MASK,
    ),
    (
        re.compile(
            r"(?i)([\"']?\b\w*(?:api_?key|secret|password|token)[\"']?\s*[:=]\s*[\"']?)"
            r"[^\s\"',;&]{6,}"
        ),
        r"\1" + _MASK,
    ),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), _MASK),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), _MASK),
)

_STD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "perf",
    "taskName",
}


class Redactor:
    """Masks secrets in text. Configured secret values shorter than ``min_len`` are ignored
    (masking them would mangle ordinary words)."""

    def __init__(self, secrets: Iterable[str] = (), *, min_len: int = 6) -> None:
        self._min_len = min_len
        self._values: set[str] = set()
        self._regex: re.Pattern[str] | None = None
        self.add(*secrets)

    def add(self, *secrets: str | None) -> None:
        values = {s for s in secrets if s and len(s) >= self._min_len}
        if values - self._values:
            self._values |= values
            alternation = "|".join(
                re.escape(v) for v in sorted(self._values, key=len, reverse=True)
            )
            self._regex = re.compile(alternation)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        if self._regex is not None:
            text = self._regex.sub(_MASK, text)
        for pattern, repl in _PATTERNS:
            text = pattern.sub(repl, text)
        return text


class RedactionFilter(logging.Filter):
    """Formats ``msg % args`` and the traceback into text, then redacts them in place."""

    _exc_formatter = logging.Formatter()

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = f"{record.msg!r} % {record.args!r}"
        record.msg = self.redactor(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = self._exc_formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redactor(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redactor(record.stack_info)
        for key, value in list(record.__dict__.items()):
            if key not in _STD_ATTRS and isinstance(value, str):
                record.__dict__[key] = self.redactor(value)
        if not hasattr(record, "perf"):
            record.perf = time.perf_counter()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra=`` fields are included."""

    def __init__(self, proc: str) -> None:
        super().__init__()
        self.proc = proc

    def format(self, record: logging.LogRecord) -> str:
        created = _dt.datetime.fromtimestamp(record.created).astimezone()
        out: dict[str, Any] = {
            "ts": created.isoformat(timespec="milliseconds"),
            "t": getattr(record, "perf", None),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "proc": self.proc,
            "pid": record.process,
            "thread": record.threadName,
        }
        if record.exc_text:
            out["exc"] = record.exc_text
        if record.stack_info:
            out["stack"] = record.stack_info
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and key not in out:
                out[key] = value
        return json.dumps(out, ensure_ascii=False, default=str)


class _QueueHandler(logging.handlers.QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # The RedactionFilter already merged args and rendered the traceback; drop live
        # objects (exc_info holds frames) before the record crosses to the listener thread.
        record = copy.copy(record)
        record.exc_info = None
        return record


@dataclass
class _State:
    listener: logging.handlers.QueueListener
    handler: _QueueHandler
    targets: list[logging.Handler]
    redactor: Redactor
    fault: IO[str] | None = None
    fault_was_enabled: bool = False
    quieted: dict[str, int] = field(default_factory=dict)


_state: _State | None = None
_atexit_registered = False
_NOISY = (
    "websockets",
    "httpx",
    "httpx2",
    "httpcore",
    "hpack",
    "urllib3",
    "aiohttp.access",
    "openai",
    "asyncio",
    "pythainlp",
)


def setup_logging(
    proc: str,
    log_dir: Path,
    *,
    secrets: Sequence[str],
    level: str = "INFO",
    console: bool = True,
    jsonl: bool = True,
    max_bytes: int = 20 * 1024 * 1024,
    backups: int = 5,
    keep_days: int = 14,
    fault: bool = True,
) -> None:
    """Configure the root logger for process ``proc`` (``core``, ``voice``, …).

    Safe to call again (the previous setup is shut down first). Handlers that others added to
    the root logger (e.g. pytest's) are left alone.
    """
    global _state, _atexit_registered
    shutdown_logging()
    today = _dt.date.today()
    day_dir = Path(log_dir) / today.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    _prune(Path(log_dir), today, keep_days)

    redactor = Redactor(secrets)
    targets: list[logging.Handler] = []
    text = logging.handlers.RotatingFileHandler(
        day_dir / f"{proc}.log", maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    text.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    targets.append(text)
    if jsonl:
        js = logging.handlers.RotatingFileHandler(
            day_dir / f"{proc}.jsonl", maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        js.setFormatter(JsonFormatter(proc))
        targets.append(js)
    if console:
        con = logging.StreamHandler(sys.stderr)
        con.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        targets.append(con)

    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    handler = _QueueHandler(log_queue)
    handler.addFilter(RedactionFilter(redactor))
    listener = logging.handlers.QueueListener(log_queue, *targets, respect_handler_level=True)
    listener.start()

    root = logging.getLogger()
    root.addHandler(handler)
    numeric = logging.getLevelName(level.upper())
    root.setLevel(numeric if isinstance(numeric, int) else logging.INFO)
    state = _State(listener=listener, handler=handler, targets=targets, redactor=redactor)
    if root.level > logging.DEBUG:
        for name in _NOISY:
            lib = logging.getLogger(name)
            state.quieted[name] = lib.level
            lib.setLevel(max(logging.WARNING, root.level))

    if fault:
        try:
            state.fault_was_enabled = faulthandler.is_enabled()
            fh = (day_dir / f"{proc}.fault").open("a", encoding="utf-8")
            faulthandler.enable(file=fh, all_threads=True)
            state.fault = fh
        except (OSError, ValueError):
            logging.getLogger("aivtube.logging").warning(
                "faulthandler file not enabled", exc_info=True
            )
    _state = state
    if not _atexit_registered:
        atexit.register(shutdown_logging)
        _atexit_registered = True


def shutdown_logging() -> None:
    """Flush and remove what ``setup_logging`` installed (idempotent)."""
    global _state
    state = _state
    if state is None:
        return
    _state = None
    logging.getLogger().removeHandler(state.handler)
    with contextlib.suppress(Exception):  # listener already stopped or its thread died
        state.listener.stop()
    for target in state.targets:
        if isinstance(target, logging.StreamHandler) and not isinstance(
            target, logging.FileHandler
        ):
            target.flush()
        else:
            target.close()
    for name, lvl in state.quieted.items():
        logging.getLogger(name).setLevel(lvl)
    if state.fault is not None:
        try:
            faulthandler.disable()
            if state.fault_was_enabled:
                faulthandler.enable(file=sys.__stderr__ or sys.stderr, all_threads=True)
        except (OSError, ValueError, AttributeError):
            pass
        state.fault.close()


def current_redactor() -> Redactor | None:
    """The active redactor (``aivtube report`` reuses it for config and logs)."""
    return _state.redactor if _state is not None else None


def add_secrets(*values: str | None) -> None:
    """Register secrets created after setup (e.g. the IPC token)."""
    if _state is not None:
        _state.redactor.add(*values)


def fault_file() -> IO[str] | None:
    """The ``<proc>.fault`` file, for extra ``faulthandler`` dumps (loop-lag watchdog)."""
    return _state.fault if _state is not None else None


def _prune(log_dir: Path, today: _dt.date, keep_days: int) -> None:
    try:
        entries = list(log_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            day = _dt.date.fromisoformat(entry.name)
        except ValueError:
            continue
        if entry.is_dir() and (today - day).days > keep_days:
            shutil.rmtree(entry, ignore_errors=True)
