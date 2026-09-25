"""load_config: profiles, layering, env/CLI overrides, defaults drift, error translation."""

from __future__ import annotations

import pickle
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aivtube.config import AppConfig, ConfigError, load_config
from aivtube.config.layers import collect_layers

REPO = Path(__file__).resolve().parents[3]
PROFILES = ["stream", "light", "gaming", "text", "offline", "ci"]


def make_root(tmp_path: Path, user: str | None = None) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    if user is not None:
        (tmp_path / "config" / "user.toml").write_text(user, encoding="utf-8")
    return tmp_path


def test_defaults_define_exactly_the_documented_profiles() -> None:
    data = tomllib.loads((REPO / "config" / "defaults.toml").read_text(encoding="utf-8"))
    assert set(data["profiles"]) == set(PROFILES)
    assert data["active_profile"] == "stream"


@pytest.mark.parametrize("profile", PROFILES)
def test_defaults_load_and_validate_for_every_profile(profile: str) -> None:
    cfg = load_config(REPO, profile=profile, env={})
    assert cfg.active_profile == profile
    assert cfg.llm.effective_chain(), "every profile needs at least one usable LLM"
    assert cfg.stt.effective_chain()
    assert cfg.root == REPO
    assert cfg.profile_names() == sorted(PROFILES)


def test_profile_overlays_apply() -> None:
    light = load_config(REPO, profile="light", env={})
    assert light.llm.chain[0] == "local-4b"
    assert light.llm.servers["local30b"].autostart == "never"
    assert light.llm.servers["local4b"].autostart == "always"
    gaming = load_config(REPO, profile="gaming", env={})
    assert gaming.llm.servers["local4b"].ctx == 8192
    assert load_config(REPO, profile="text", env={}).app.voice_worker is False
    assert load_config(REPO, profile="offline", env={}).tts.identity_chain == ["offline"]
    ci = load_config(REPO, profile="ci", env={})
    assert ci.app.fakes is True and ci.chat.sources == ["fake"]


def test_schema_defaults_mirror_defaults_toml() -> None:
    loaded = load_config(REPO, profile="stream", env={})
    skip = {"profiles", "active_profile"}
    assert AppConfig().model_dump(exclude=skip) == loaded.model_dump(exclude=skip)


def test_layer_precedence(tmp_path: Path) -> None:
    root = make_root(
        tmp_path, user="[brain]\nchat_k = 4\nidle_after_s = 30.0\n[chat]\nmax_msg_chars = 250\n"
    )
    cfg = load_config(root, profile="light", env={})
    assert cfg.llm.chain[0] == "local-4b"  # profile over defaults
    assert cfg.brain.chat_k == 4  # user over defaults
    env = {"AIVTUBE__BRAIN__CHAT_K": "5", "AIVTUBE__BRAIN__IDLE_AFTER_S": "40"}
    cfg = load_config(root, env=env)
    assert cfg.brain.chat_k == 5 and cfg.brain.idle_after_s == 40.0  # env over user
    cfg = load_config(root, env=env, cli_overrides={"brain.chat_k": 6})
    assert cfg.brain.chat_k == 6  # CLI over env
    assert cfg.brain.idle_after_s == 40.0
    assert cfg.chat.max_msg_chars == 250


def test_user_layer_beats_profile(tmp_path: Path) -> None:
    root = make_root(tmp_path, user='[llm]\nchain = ["local-30b"]\n')
    assert load_config(root, profile="light", env={}).llm.chain == ["local-30b"]


def test_active_profile_comes_from_the_highest_layer(tmp_path: Path) -> None:
    root = make_root(tmp_path, user='active_profile = "light"\n')
    assert load_config(root, env={}).active_profile == "light"
    assert load_config(root, env={"AIVTUBE__ACTIVE_PROFILE": "gaming"}).active_profile == "gaming"
    cfg = load_config(
        root, env={"AIVTUBE__ACTIVE_PROFILE": "gaming"}, cli_overrides={"active_profile": "text"}
    )
    assert cfg.active_profile == "text"
    assert load_config(root, profile="ci", env={}).active_profile == "ci"


def test_user_file_can_define_its_own_profile(tmp_path: Path) -> None:
    root = make_root(tmp_path, user="[profiles.quiet.brain]\nidle_after_s = 90.0\n")
    assert load_config(root, profile="quiet", env={}).brain.idle_after_s == 90.0


def test_unknown_profile_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as info:
        load_config(REPO, profile="streem", env={})
    assert info.value.path == "active_profile"
    assert "streem" in info.value.message_en and info.value.message_th


def test_env_values_are_parsed_as_toml_literals() -> None:
    env = {
        "AIVTUBE__LLM__CHAIN": '["local-4b"]',
        "AIVTUBE__MIC__MODE": "ptt",
        "AIVTUBE__CHAT__ENABLED": "false",
        "AIVTUBE__AUDIO__OUTPUT_DEVICE": "Headphones (Realtek(R) Audio)",
        "aivtube__brain__chat_k": "2",  # names are case-insensitive
        "AIVTUBE_BUS_TOKEN": "not-config",  # single underscore: ignored
    }
    cfg = load_config(REPO, env=env)
    assert cfg.llm.chain == ["local-4b"]
    assert cfg.mic.mode == "ptt"
    assert cfg.chat.enabled is False
    assert cfg.audio.output_device == "Headphones (Realtek(R) Audio)"
    assert cfg.brain.chat_k == 2


def test_malformed_env_name_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as info:
        load_config(REPO, env={"AIVTUBE__BRAIN____CHAT_K": "2"})
    assert info.value.source == "environment"


def test_os_environ_is_read_when_env_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIVTUBE__BRAIN__CHAT_K", "7")
    assert load_config(REPO).brain.chat_k == 7


def test_cli_overrides_accept_nested_and_dotted_keys() -> None:
    cfg = load_config(REPO, env={}, cli_overrides={"app": {"fakes": True}, "mic.mode": "ptt"})
    assert cfg.app.fakes is True and cfg.mic.mode == "ptt"


def test_toml_syntax_error_names_the_file_in_both_languages(tmp_path: Path) -> None:
    root = make_root(tmp_path, user="[brain\nchat_k = 3\n")
    with pytest.raises(ConfigError) as info:
        load_config(root, env={})
    err = info.value
    assert err.path.endswith("user.toml")
    assert "TOML" in err.message_en and "TOML" in err.message_th and err.hint


def test_missing_defaults_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_config(tmp_path, env={})
    assert info.value.message_th


@pytest.mark.parametrize(("version", "needle"), [(0, "positive"), (99, "newer")])
def test_user_schema_version_is_checked(tmp_path: Path, version: int, needle: str) -> None:
    root = make_root(tmp_path, user=f"schema_version = {version}\n")
    with pytest.raises(ConfigError) as info:
        load_config(root, env={})
    text = (info.value.message_en + info.value.hint).lower()
    assert needle in text
    assert info.value.message_th and info.value.source == "config/user.toml"


def test_old_user_schema_asks_for_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aivtube.config import layers

    monkeypatch.setattr(layers, "CURRENT_SCHEMA_VERSION", 2)
    root = make_root(tmp_path, user="schema_version = 1\n")
    with pytest.raises(ConfigError) as info:
        load_config(root, env={})
    assert "aivtube config migrate" in info.value.hint
    assert "aivtube config migrate" in info.value.hint_th


def test_unknown_key_suggests_the_closest_name() -> None:
    with pytest.raises(ConfigError) as info:
        load_config(REPO, env={}, cli_overrides={"brain.chat_min_intervl_s": 3.0})
    err = info.value
    assert err.path == "brain.chat_min_intervl_s"
    assert err.source == "command line"
    assert "chat_min_interval_s" in err.hint and "chat_min_interval_s" in err.hint_th
    assert err.message_th


def test_unknown_key_inside_a_named_entry() -> None:
    with pytest.raises(ConfigError) as info:
        load_config(REPO, env={}, cli_overrides={"llm.providers.gemini.modle": "x"})
    assert info.value.path == "llm.providers.gemini.modle"
    assert "'model'" in info.value.hint


def test_every_issue_is_reported(tmp_path: Path) -> None:
    root = make_root(tmp_path, user='[mic]\nmode = "loud"\n[brain]\nchat_k = "x"\n')
    with pytest.raises(ConfigError) as info:
        load_config(root, env={})
    err = info.value
    paths = {i.path for i in err.issues}
    assert {"mic.mode", "brain.chat_k"} <= paths
    assert all(i.message_th and i.hint and i.hint_th for i in err.issues)
    assert all(i.source == "config/user.toml" for i in err.issues)
    assert "mic.mode" in err.format_all() and "brain.chat_k" in err.format_all()
    assert "(+" in str(err)


def test_config_error_pickles_and_prints_both_languages() -> None:
    err = ConfigError(
        "a.b", "English text", "ข้อความไทย", "do this", hint_th="ทำแบบนี้", source="config/user.toml"
    )
    again = pickle.loads(pickle.dumps(err))
    assert (again.path, again.message_th, again.hint_th, again.source) == (
        "a.b",
        "ข้อความไทย",
        "ทำแบบนี้",
        "config/user.toml",
    )
    text = str(err)
    assert "English text" in text and "ข้อความไทย" in text and "do this" in text


def test_config_is_frozen() -> None:
    cfg = AppConfig()
    with pytest.raises(ValidationError):
        cfg.brain.chat_k = 9  # type: ignore[misc]
    changed = cfg.model_copy(update={"brain": cfg.brain.model_copy(update={"chat_k": 9})})
    assert changed.brain.chat_k == 9 and cfg.brain.chat_k == 3


def test_resolve_path_expands_character_placeholders() -> None:
    cfg = load_config(REPO, env={})
    lexicon = cfg.tts.backends["piper"].lexicon
    assert "{character}" in lexicon
    assert (
        cfg.resolve_path(lexicon, character="pailin")
        == REPO / "characters" / "pailin" / "lexicon.toml"
    )
    assert cfg.resolve_path("data") == REPO / "data"
    with pytest.raises(ValueError):
        cfg.resolve_path(lexicon)


def test_collect_layers_reports_sources() -> None:
    layers = collect_layers(
        REPO, env={"AIVTUBE__MIC__MODE": "ptt"}, cli_overrides={"brain.chat_k": 2}
    )
    assert layers.source_of(("mic", "mode")) == "environment"
    assert layers.source_of(("brain", "chat_k")) == "command line"
    assert layers.source_of(("vad", "threshold")) == "config/defaults.toml"
    assert layers.merged["active_profile"] == "stream"


def _roundtrip(cfg: AppConfig) -> dict[str, Any]:
    return AppConfig.model_validate(cfg.model_dump()).model_dump()


def test_validated_config_revalidates_from_its_own_dump() -> None:
    for profile in PROFILES:
        cfg = load_config(REPO, profile=profile, env={})
        assert _roundtrip(cfg) == cfg.model_dump()
