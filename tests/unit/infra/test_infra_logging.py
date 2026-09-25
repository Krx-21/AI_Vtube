"""setup_logging and the redaction filter."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from aivtube.infra import (
    RedactionFilter,
    Redactor,
    add_secrets,
    current_redactor,
    setup_logging,
    shutdown_logging,
)
from aivtube.infra.logging import fault_file

SECRET = "sk-live-TYPHOON-1234567890abcdef"


@pytest.fixture
def logs(tmp_path: Path) -> Iterator[Path]:
    root_level = logging.getLogger().level
    setup_logging("core", tmp_path, secrets=[SECRET, "abc"], level="DEBUG", console=False)
    try:
        yield tmp_path
    finally:
        shutdown_logging()
        logging.getLogger().setLevel(root_level)


def test_redactor_masks_secrets_and_token_shapes() -> None:
    r = Redactor([SECRET, "short"])
    text = (
        f"key={SECRET} irc PASS oauth:abcdef123456 hdr Authorization: Bearer eyJhbGci.x-y_z "
        "url https://www.googleapis.com/youtube/v3/videos?part=x&key=AIzaSyA1234567890 "
        "short stays"
    )
    out = r(text)
    assert SECRET not in out
    assert "oauth:***" in out and "abcdef123456" not in out
    assert "Bearer ***" in out and "eyJhbGci" not in out
    assert "AIzaSyA1234567890" not in out and "part=x" in out
    assert "short stays" in out  # secrets under 6 characters are not masked


@pytest.mark.parametrize(
    "text",
    [
        '{"api_key": "abcdefghijkl"}',
        "bus_token=0123456789abcdef",
        "password: hunter2hunter2",
        "Ocp-Apim-Subscription-Key: 0123456789abcdef",
        "AIzaSyD" + "x" * 32,
        "sk-" + "a" * 24,
    ],
)
def test_redactor_generic_patterns(text: str) -> None:
    out = Redactor()(text)
    assert "***" in out


@pytest.mark.parametrize(
    "text", ["prompt_tokens: 312", "max_tokens=256", "token_file = data/tokens/vts_pailin.txt"]
)
def test_redactor_leaves_ordinary_text(text: str) -> None:
    assert Redactor()(text) == text


def test_redaction_filter_renders_args_and_tracebacks() -> None:
    record = logging.LogRecord(
        "x", logging.ERROR, __file__, 1, "sent %s to %s", (SECRET, "api"), None
    )
    try:
        raise ValueError(f"bad key {SECRET}")
    except ValueError:
        record.exc_info = sys.exc_info()
    record.token_hint = f"Bearer {SECRET}"
    assert RedactionFilter(Redactor([SECRET])).filter(record)
    assert record.getMessage() == "sent *** to api"
    assert record.exc_text is not None and SECRET not in record.exc_text
    assert SECRET not in record.token_hint


def test_bad_format_args_do_not_raise() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%d items", ("many",), None)
    assert RedactionFilter(Redactor()).filter(record)
    assert "many" in record.getMessage()


def test_setup_logging_writes_redacted_text_and_jsonl(logs: Path) -> None:
    log = logging.getLogger("aivtube.test")
    log.info("connecting with %s", SECRET, extra={"turn_id": "t1"})
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    add_secrets("late-secret-value-42")
    log.warning("late-secret-value-42 appeared")
    shutdown_logging()
    day = logs / dt.date.today().isoformat()
    text = (day / "core.log").read_text(encoding="utf-8")
    assert "connecting with ***" in text and SECRET not in text
    assert "RuntimeError: boom" in text and "late-secret-value-42" not in text
    lines = [
        json.loads(line) for line in (day / "core.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first = lines[0]
    assert first["msg"] == "connecting with ***" and first["turn_id"] == "t1"
    assert first["level"] == "INFO" and first["proc"] == "core" and isinstance(first["t"], float)
    assert "RuntimeError: boom" in lines[1]["exc"]
    assert (day / "core.fault").is_file()


def test_setup_is_repeatable_and_leaves_foreign_handlers(
    logs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = logging.getLogger()
    setup_logging("core", logs, secrets=[], level="INFO", console=False)
    ours = [h for h in root.handlers if type(h).__name__ == "_QueueHandler"]
    assert len(ours) == 1
    assert caplog.handler in root.handlers
    assert logging.getLogger("websockets").level == logging.WARNING
    assert current_redactor() is not None and fault_file() is not None
    shutdown_logging()
    assert not [h for h in root.handlers if type(h).__name__ == "_QueueHandler"]
    assert caplog.handler in root.handlers
    assert logging.getLogger("websockets").level == logging.NOTSET
    assert current_redactor() is None
    shutdown_logging()  # idempotent


def test_old_day_folders_are_pruned(tmp_path: Path) -> None:
    old = tmp_path / (dt.date.today() - dt.timedelta(days=30)).isoformat()
    recent = tmp_path / (dt.date.today() - dt.timedelta(days=3)).isoformat()
    other = tmp_path / "notes"
    for d in (old, recent, other):
        d.mkdir()
    setup_logging("voice", tmp_path, secrets=[], console=False, keep_days=14, fault=False)
    shutdown_logging()
    assert not old.exists() and recent.exists() and other.exists()
