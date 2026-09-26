"""``report`` (a secret scan of the zip) and ``db`` backup/restore."""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest

from aivtube.ops.db import DbError, db_backup, db_restore, list_backups
from aivtube.ops.report import build_report, collect_secrets

REPO = Path(__file__).resolve().parents[3]
SECRETS = {
    "TYPHOON_API_KEY": "sk-typhoon-SECRET-1234567890abcdef",
    "GEMINI_API_KEY": "AIzaSyD-SECRET-gemini-key-000000000000",
    "AZURE_SPEECH_KEY": "azure-SECRET-0123456789abcdef",
    "YOUTUBE_API_KEY": "yt-SECRET-key-abcdef123456",
    "TWITCH_CLIENT_ID": "twitch-SECRET-client-id-9999",
}
PANEL_TOKEN = "panel-SECRET-token-abcdefghijkl"
EMERGENCY_TOKEN = "emergency-SECRET-token-mnopqrstuv"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    (tmp_path / ".env").write_text(
        "\n".join(f"{k}={v}" for k, v in SECRETS.items()) + "\nAZURE_SPEECH_REGION=southeastasia\n",
        encoding="utf-8",
    )
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    (state / "tokens.json").write_text(
        json.dumps({"panel": PANEL_TOKEN, "emergency": EMERGENCY_TOKEN}), encoding="utf-8"
    )
    (state / "llama_tuning.json").write_text('{"servers": {}}', encoding="utf-8")
    (tmp_path / "config" / "user.toml").write_text(
        "schema_version = 1\n[chat.twitch_irc]\nchannel = \"pailin_ch\"\n", encoding="utf-8"
    )
    return tmp_path


def _leaky_logs(root: Path) -> None:
    today = root / "logs" / _dt.date.today().isoformat()
    today.mkdir(parents=True)
    lines = [
        f"calling gemini with key={SECRETS['GEMINI_API_KEY']}",
        f"Authorization: Bearer {SECRETS['TYPHOON_API_KEY']}",
        "twitch PASS oauth:abcdef0123456789",
        f"launcher token {EMERGENCY_TOKEN} and ?token={PANEL_TOKEN}",
        f"azure {SECRETS['AZURE_SPEECH_KEY']} youtube {SECRETS['YOUTUBE_API_KEY']}",
        f"client {SECRETS['TWITCH_CLIENT_ID']}",
        "ไพลินพูดว่า สวัสดีค่ะ",
    ]
    (today / "core.log").write_text("\n".join(lines), encoding="utf-8")
    (today / "core.jsonl").write_text(json.dumps({"msg": lines[0]}), encoding="utf-8")
    (today / "core.fault").write_text("Fatal Python error\n", encoding="utf-8")
    old = root / "logs" / "2020-01-01"
    old.mkdir()
    (old / "core.log").write_text("ancient", encoding="utf-8")
    flight = root / "data" / "flight"
    flight.mkdir(parents=True)
    (flight / "flight-20260101-000000-1.json").write_text(
        json.dumps({"events": [], "llm": [{"key": SECRETS["GEMINI_API_KEY"]}]}), encoding="utf-8"
    )


def test_report_contents_and_secret_scan(root: Path) -> None:
    _leaky_logs(root)
    assert set(SECRETS.values()) | {PANEL_TOKEN, EMERGENCY_TOKEN} <= set(
        collect_secrets(root, env={})
    )
    path = build_report(root, root / "out", env={})
    assert path.suffix == ".zip" and path.parent == root / "out"
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        blobs = {n: zf.read(n) for n in names}
    today = _dt.date.today().isoformat()
    assert {f"logs/{today}/core.log", f"logs/{today}/core.jsonl", f"logs/{today}/core.fault",
            "config/user.toml", "config/effective.json", "doctor.txt", "versions.json",
            "state/llama_tuning.json", "flight/flight-20260101-000000-1.json"} <= names
    assert "logs/2020-01-01/core.log" not in names  # older than --days
    assert not any(n.endswith(("tokens.json", ".env")) for n in names)
    everything = b"\n".join(blobs.values()).decode("utf-8", "replace")
    for secret in [*SECRETS.values(), PANEL_TOKEN, EMERGENCY_TOKEN, "abcdef0123456789"]:
        assert secret not in everything, f"secret leaked: {secret[:10]}…"
    assert "ไพลินพูดว่า" in blobs[f"logs/{today}/core.log"].decode()
    assert "pailin_ch" in blobs["config/user.toml"].decode()
    versions = json.loads(blobs["versions.json"])
    assert versions["aivtube"] and "numpy" in versions["packages"]
    assert "error(s)" in blobs["doctor.txt"].decode()
    effective = json.loads(blobs["config/effective.json"])
    assert effective["ports"]["emergency"] == 8779


def test_report_trims_huge_logs(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import aivtube.ops.report as R

    monkeypatch.setattr(R, "MAX_FILE_BYTES", 1000)
    today = root / "logs" / _dt.date.today().isoformat()
    today.mkdir(parents=True)
    (today / "voice.log").write_text("x" * 5000 + "THE END", encoding="utf-8")
    path = build_report(root, doctor="doctor skipped", env={})
    with zipfile.ZipFile(path) as zf:
        text = zf.read(f"logs/{today.name}/voice.log").decode()
        assert text.endswith("THE END") and "bytes cut" in text and len(text) < 1100
        assert zf.read("doctor.txt").decode() == "doctor skipped"


def _make_db(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS m (t TEXT)")
    con.execute("DELETE FROM m")
    con.executemany("INSERT INTO m VALUES (?)", [(r,) for r in rows])
    con.commit()
    con.close()


def _rows(path: Path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT t FROM m ORDER BY rowid")]
    finally:
        con.close()


def test_db_backup_and_restore(root: Path) -> None:
    mem = root / "data" / "memory" / "pailin.sqlite"
    ops = root / "data" / "ops.db"
    _make_db(mem, ["ไพลินชอบแมว"])
    _make_db(ops, ["audit"])
    made = db_backup(root)
    assert {p.name.split("-")[0] for p in made} == {"pailin", "ops"}
    folder = root / "data" / "backups"
    assert all(p.parent == folder for p in made)
    backup = next(p for p in made if p.name.startswith("pailin"))
    _make_db(mem, ["something else"])
    db_restore(root, backup, is_running=lambda r: False)
    assert _rows(mem) == ["ไพลินชอบแมว"]
    # the state before the restore was saved too
    assert len(list_backups(folder, "pailin")) == 2
    with pytest.raises(DbError, match="running"):
        db_restore(root, backup, is_running=lambda r: True)
    bogus = folder / "nope-20260101-000000.sqlite"
    bogus.write_bytes(b"not a database")
    with pytest.raises(DbError):
        db_restore(root, bogus, is_running=lambda r: False)
    with pytest.raises(DbError):
        db_restore(root, root / "random.sqlite", is_running=lambda r: False)


def test_db_backup_rotation(root: Path) -> None:
    (root / "config" / "user.toml").write_text(
        "schema_version = 1\n[memory]\nbackup_keep = 2\n", encoding="utf-8"
    )
    _make_db(root / "data" / "memory" / "pailin.sqlite", ["a"])
    for _ in range(4):
        db_backup(root)
    assert len(list_backups(root / "data" / "backups", "pailin")) == 2


def test_db_cli_backup_list_restore(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import socket

    from aivtube.ops import db as D

    ports = []
    for _ in range(2):  # free ports, so "is aivtube running?" says no
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(int(s.getsockname()[1]))
    (root / "config" / "user.toml").write_text(
        f"schema_version = 1\n[ports]\npanel = {ports[0]}\nemergency = {ports[1]}\n",
        encoding="utf-8",
    )
    mem = root / "data" / "memory" / "pailin.sqlite"
    _make_db(mem, ["ก่อน"])
    assert D.main(["--root", str(root), "backup"]) == 0
    listed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("✔")]
    backup = Path(listed[0].removeprefix("✔ ").strip())
    assert backup.is_file()
    _make_db(mem, ["หลัง"])
    assert D.main(["--root", str(root), "list"]) == 0
    assert str(backup) in capsys.readouterr().out
    assert D.main(["--root", str(root), "restore", str(backup)]) == 0
    assert _rows(mem) == ["ก่อน"]
    assert D.main(["--root", str(root), "restore", str(root / "nope.sqlite")]) == 1
    assert "✖" in capsys.readouterr().out
