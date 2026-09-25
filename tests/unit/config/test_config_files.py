"""Secrets, migrate, write_user_overrides and the example files."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from aivtube.config import (
    ConfigError,
    Secrets,
    load_config,
    load_secrets,
    migrate,
    write_user_overrides,
)
from aivtube.config.layers import collect_layers
from aivtube.config.load import validate_layers

REPO = Path(__file__).resolve().parents[3]


def make_root(tmp_path: Path, user: str | None = None) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    if user is not None:
        (tmp_path / "config" / "user.toml").write_text(user, encoding="utf-8")
    return tmp_path


# --- example files ----------------------------------------------------------------------


def test_env_example_lists_exactly_the_secrets() -> None:
    keys = {
        line.split("=", 1)[0]
        for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert keys == set(Secrets.model_fields)


def test_user_toml_example_is_valid() -> None:
    data = tomllib.loads((REPO / "config" / "user.toml.example").read_text(encoding="utf-8"))
    cfg = validate_layers(REPO, collect_layers(REPO, env={}, user_data=data))
    assert cfg.chat.twitch_irc.channel == "your_channel"


# --- secrets ----------------------------------------------------------------------------


def test_secrets_from_dotenv_and_environment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "TYPHOON_API_KEY=sk-file-key-123456\nGEMINI_API_KEY=\nAZURE_SPEECH_REGION=southeastasia\n"
        "OPENROUTER_API_KEY=or-key-abcdef\n",
        encoding="utf-8",
    )
    secrets = load_secrets(
        tmp_path,
        env={"AZURE_SPEECH_KEY": "azure-key-999999", "TYPHOON_API_KEY": "sk-env-wins-0000"},
    )
    assert secrets.get("TYPHOON_API_KEY") == "sk-env-wins-0000"
    assert secrets.get("gemini_api_key") is None  # empty means unset
    assert secrets.get("AZURE_SPEECH_KEY") == "azure-key-999999"
    assert secrets.get("OPENROUTER_API_KEY") == "or-key-abcdef"  # custom provider keys work
    assert secrets.get("NOPE") is None
    assert "sk-env-wins" not in repr(secrets) and "azure-key" not in str(secrets)
    values = secrets.redaction_values()
    assert "sk-env-wins-0000" in values and "or-key-abcdef" in values
    assert "southeastasia" not in values


def test_secrets_are_hermetic_with_an_env_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPHOON_API_KEY", "from-the-real-env")
    assert load_secrets(tmp_path, env={}).get("TYPHOON_API_KEY") is None
    assert load_secrets(tmp_path).get("TYPHOON_API_KEY") == "from-the-real-env"


# --- migrate ----------------------------------------------------------------------------


def test_migrate_without_user_toml_does_nothing(tmp_path: Path) -> None:
    assert migrate(make_root(tmp_path)) == []


def test_migrate_stamps_the_schema_version_with_a_backup(tmp_path: Path) -> None:
    original = '# my notes\n[mic]\nmode = "ptt"\n'
    root = make_root(tmp_path, user=original)
    user = root / "config" / "user.toml"
    assert migrate(root, dry_run=True) == ["set schema_version = 1"]
    assert user.read_text(encoding="utf-8") == original  # dry run writes nothing
    assert migrate(root) == ["set schema_version = 1"]
    assert (root / "config" / "user.toml.bak").read_text(encoding="utf-8") == original
    data = tomllib.loads(user.read_text(encoding="utf-8"))
    assert data == {"schema_version": 1, "mic": {"mode": "ptt"}}
    assert migrate(root) == []  # idempotent
    assert load_config(root, env={}).mic.mode == "ptt"


def test_migrate_runs_registered_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aivtube.config import layers, migration

    def v1_to_v2(data: dict[str, object]) -> list[str]:
        mic = data.setdefault("mic", {})
        assert isinstance(mic, dict)
        mic["mode"] = mic.pop("old_mode", "open")
        return ["renamed mic.old_mode to mic.mode"]

    monkeypatch.setattr(layers, "CURRENT_SCHEMA_VERSION", 2)
    monkeypatch.setitem(migration.MIGRATIONS, 1, v1_to_v2)
    root = make_root(tmp_path, user='schema_version = 1\n[mic]\nold_mode = "ptt"\n')
    notes = migrate(root)
    assert notes == ["v1→v2: renamed mic.old_mode to mic.mode", "set schema_version = 2"]
    data = tomllib.loads((root / "config" / "user.toml").read_text(encoding="utf-8"))
    assert data == {"schema_version": 2, "mic": {"mode": "ptt"}}
    assert (root / "config" / "user.toml.bak").is_file()


def test_migrate_refuses_newer_files(tmp_path: Path) -> None:
    root = make_root(tmp_path, user="schema_version = 7\n")
    with pytest.raises(ConfigError) as info:
        migrate(root)
    assert info.value.message_th and info.value.hint


def test_migrate_needs_every_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aivtube.config import layers

    monkeypatch.setattr(layers, "CURRENT_SCHEMA_VERSION", 3)
    root = make_root(tmp_path, user="schema_version = 1\n")
    with pytest.raises(ConfigError, match="No migration"):
        migrate(root)


# --- write_user_overrides ---------------------------------------------------------------


def test_write_user_overrides_creates_merges_and_backs_up(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    path = write_user_overrides(
        root,
        {"llm.servers.local30b.placement": "pinned", "llm.servers.local30b.pinned_n_cpu_moe": 38},
    )
    assert path == root / "config" / "user.toml"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# config/user.toml")
    data = tomllib.loads(text)
    assert data["schema_version"] == 1
    assert data["llm"]["servers"]["local30b"] == {"placement": "pinned", "pinned_n_cpu_moe": 38}

    write_user_overrides(root, {"audio": {"output_device": "Headphones"}})
    assert (root / "config" / "user.toml.bak").read_text(encoding="utf-8") == text
    cfg = load_config(root, env={})
    assert cfg.llm.servers["local30b"].pinned_n_cpu_moe == 38
    assert cfg.audio.output_device == "Headphones"

    write_user_overrides(root, {"llm.servers.local30b.pinned_n_cpu_moe": None})
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["llm"]["servers"]["local30b"] == {"placement": "pinned"}  # None deletes


def test_invalid_overrides_are_not_written(tmp_path: Path) -> None:
    root = make_root(tmp_path, user="schema_version = 1\n")
    before = (root / "config" / "user.toml").read_text(encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        write_user_overrides(root, {"ports.panel": 8005})
    assert info.value.source == "config/user.toml"
    assert (root / "config" / "user.toml").read_text(encoding="utf-8") == before
    assert not (root / "config" / "user.toml.bak").exists()


def test_overrides_store_paths_and_tuples(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_user_overrides(root, {"app.data_dir": Path("D:/stream/data"), "llm.chain": ("local-4b",)})
    cfg = load_config(root, env={})
    assert cfg.app.data_dir == "D:/stream/data" and cfg.llm.chain == ["local-4b"]
