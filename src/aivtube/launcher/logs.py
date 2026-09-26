"""Launcher logging: ``logs/<date>/launcher.log`` (20 MB × 5) plus the console, with the tokens
redacted. Standard library only (the core's ``aivtube.infra.logging`` is not imported here)."""

from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import re
import sys
from collections.abc import Iterable
from pathlib import Path

__all__ = ["TokenRedactor", "setup_launcher_logging"]

_PATTERNS = (
    (re.compile(r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]+"), r"\1***"),
    (re.compile(r"(?i)([?&]token=)[^&\s\"']+"), r"\1***"),
)


class TokenRedactor(logging.Filter):
    """Replaces the given secret values (and Bearer / ``?token=`` values) with ``***``."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        values = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)
        self._regex = re.compile("|".join(re.escape(v) for v in values)) if values else None

    def redact(self, text: str) -> str:
        if self._regex is not None:
            text = self._regex.sub("***", text)
        for pattern, repl in _PATTERNS:
            text = pattern.sub(repl, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = self.redact(message)
        record.args = None
        return True


def setup_launcher_logging(
    log_dir: Path,
    *,
    secrets: Iterable[str] = (),
    level: str = "INFO",
    console: bool = True,
    max_bytes: int = 20 * 1024 * 1024,
    backups: int = 5,
) -> logging.Logger:
    """Configure the ``aivtube`` logger tree for the launcher process; returns its logger."""
    logger = logging.getLogger("aivtube")
    for handler in list(logger.handlers):
        if getattr(handler, "_aivtube_launcher", False):
            logger.removeHandler(handler)
            handler.close()
    redactor = TokenRedactor(secrets)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    day = Path(log_dir) / _dt.date.today().isoformat()
    try:
        day.mkdir(parents=True, exist_ok=True)
        fh: logging.Handler | None = logging.handlers.RotatingFileHandler(
            day / "launcher.log", maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
    except OSError:
        fh = None
    if fh is not None:
        fh.setFormatter(fmt)
        fh.addFilter(redactor)
        fh._aivtube_launcher = True  # type: ignore[attr-defined]
        logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        ch.addFilter(redactor)
        ch.setLevel(logging.WARNING)
        ch._aivtube_launcher = True  # type: ignore[attr-defined]
        logger.addHandler(ch)
    numeric = logging.getLevelName(level.upper())
    logger.setLevel(numeric if isinstance(numeric, int) else logging.INFO)
    return logging.getLogger("aivtube.launcher")
