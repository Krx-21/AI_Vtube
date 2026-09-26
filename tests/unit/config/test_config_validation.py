"""Validation rules: ports, Azure F0, cloud consent, device names and other cross-checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivtube.config import AppConfig, ConfigError, load_config

REPO = Path(__file__).resolve().parents[3]


def load(**cli: Any) -> AppConfig:
    return load_config(REPO, env={}, cli_overrides=cli)


def error(**cli: Any) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        load(**cli)
    err = info.value
    assert err.message_en and err.message_th and err.hint and err.hint_th
    return err


# --- ports ------------------------------------------------------------------------------


@pytest.mark.parametrize("port", range(8001, 8010))
def test_vts_ports_are_rejected(port: int) -> None:
    err = error(**{"ports.panel": port})
    assert err.path == "ports.panel"
    assert "VTube Studio" in err.message_en and "VTube Studio" in err.message_th
    assert err.source == "command line"


@pytest.mark.parametrize("key", ["llm.servers.local4b.port", "ports.neuro_sdk", "ports.emergency"])
def test_vts_ports_are_rejected_everywhere(key: str) -> None:
    assert error(**{key: 8005}).path == key


def test_vts_port_in_a_worker_url_is_rejected() -> None:
    err = error(**{"stt.backends.whisper_gpu.url": "http://127.0.0.1:8003"})
    assert err.path == "stt.backends.whisper_gpu.url"


@pytest.mark.parametrize("port", [0, 70000])
def test_out_of_range_ports_are_rejected(port: int) -> None:
    assert error(**{"ports.bus": port}).path == "ports.bus"


def test_duplicate_ports_are_rejected_and_blame_the_changed_key() -> None:
    err = error(**{"ports.bus": 8770})
    assert err.path == "ports.bus"
    assert err.source == "command line"
    assert "8770" in err.message_en and "8770" in err.message_th
    err = error(**{"ports.panel": 8771})
    assert err.path == "ports.panel"  # the clash is blamed on the key the user changed


def test_duplicate_between_sections_is_rejected() -> None:
    err = error(**{"ports.panel": 8080})
    assert err.path == "ports.panel"
    assert "llm.servers.local30b.port" in err.message_en


def test_server_and_provider_ports_must_match() -> None:
    err = error(**{"llm.servers.local30b.port": 8082})
    assert err.path == "llm.servers.local30b.port"
    assert "8082" in err.message_en


def test_twin_sdk_port_8010_is_fine() -> None:
    assert load(**{"ports.neuro_sdk": 8010}).ports.neuro_sdk == 8010


# --- Azure and cloud consent ------------------------------------------------------------


def test_azure_f0_hedge_is_rejected() -> None:
    err = error(**{"tts.backends.azure.hedge_first_chunk": True})
    assert err.path == "tts.backends.azure.hedge_first_chunk"
    assert "S0" in err.message_en and "S0" in err.message_th


def test_azure_s0_may_hedge() -> None:
    cfg = load(**{"tts.backends.azure.hedge_first_chunk": True, "tts.backends.azure.tier": "S0"})
    assert cfg.tts.backends["azure"].hedge_first_chunk is True


def test_azure_f0_quota_is_enforced() -> None:
    assert (
        error(**{"tts.backends.azure.requests_per_min": 25}).path
        == "tts.backends.azure.requests_per_min"
    )
    assert load(
        **{"tts.backends.azure.tier": "S0", "tts.backends.azure.requests_per_min": 25}
    ).tts.backends["azure"]


def test_cloud_providers_without_consent_are_disabled_not_errors() -> None:
    cfg = load()
    assert cfg.privacy.cloud_llm_consent is False
    assert "typhoon-api" in cfg.llm.chain and "gemini" in cfg.llm.chain
    assert cfg.llm.providers["typhoon-api"].enabled is False
    assert cfg.llm.providers["gemini"].enabled is False
    assert cfg.llm.providers["local-30b"].enabled is True
    assert cfg.llm.effective_chain() == ["local-30b", "local-4b"]
    assert cfg.stt.backends["typhoon_api"].enabled is False
    assert cfg.stt.effective_chain() == ["typhoon_rt", "pythaiasr"]
    assert set(cfg.consent_blocked()) == {
        "llm.providers.typhoon-api",
        "llm.providers.gemini",
        "stt.backends.typhoon_api",
    }


def test_light_profile_without_consent_still_loads() -> None:
    cfg = load_config(REPO, profile="light", env={})
    assert cfg.llm.effective_chain() == ["local-4b"]


def test_consent_enables_cloud_providers() -> None:
    cfg = load(**{"privacy.cloud_llm_consent": True, "privacy.cloud_stt_consent": True})
    assert cfg.llm.effective_chain() == ["local-30b", "local-4b", "typhoon-api", "gemini"]
    assert cfg.stt.effective_chain() == ["typhoon_rt", "pythaiasr", "typhoon_api"]
    assert cfg.consent_blocked() == []


def test_consent_is_per_kind_and_user_disable_wins() -> None:
    cfg = load(**{"privacy.cloud_llm_consent": True, "llm.providers.gemini.enabled": False})
    assert cfg.llm.effective_chain() == ["local-30b", "local-4b", "typhoon-api"]
    assert cfg.stt.backends["typhoon_api"].enabled is False


def test_consent_applies_to_models_built_in_code() -> None:
    assert AppConfig().llm.providers["gemini"].enabled is False


def test_off_machine_provider_must_be_marked_cloud() -> None:
    err = error(
        **{
            "llm.providers.lan": {
                "kind": "openai_compat",
                "base_url": "http://192.168.1.20:8080/v1",
                "model": "m",
            }
        }
    )
    assert err.path == "llm.providers.lan.cloud"
    cfg = load(
        **{
            "llm.providers.lan": {
                "kind": "openai_compat",
                "cloud": True,
                "model": "m",
                "base_url": "http://192.168.1.20:8080/v1",
            }
        }
    )
    assert cfg.llm.providers["lan"].enabled is False  # cloud without consent


# --- audio device names ----------------------------------------------------------------


@pytest.mark.parametrize("value", [3, "2", " 12 ", 1.5, True])
def test_device_numbers_are_rejected_with_thai_message_and_hint(value: Any) -> None:
    err = error(**{"audio.output_device": value})
    assert err.path == "audio.output_device"
    assert "ชื่อ" in err.message_th  # "name"
    assert "setup --audio" in err.hint and "setup --audio" in err.hint_th


@pytest.mark.parametrize("value", ["   ", "Head\nphones", "x" * 201])
def test_malformed_device_names_are_rejected(value: str) -> None:
    err = error(**{"audio.input_device": value})
    assert err.path == "audio.input_device"


def test_device_names_are_trimmed_and_mirror_must_differ() -> None:
    cfg = load(**{"audio.output_device": "  Headphones ", "audio.mirror_output_device": "CABLE"})
    assert cfg.audio.output_device == "Headphones"
    err = error(**{"audio.output_device": "Speakers", "audio.mirror_output_device": "speakers"})
    assert err.path == "audio.mirror_output_device"


def test_device_error_from_env_names_the_environment() -> None:
    with pytest.raises(ConfigError) as info:
        load_config(REPO, env={"AIVTUBE__AUDIO__INPUT_DEVICE": "5"})
    assert info.value.source == "environment"
    assert info.value.message_th and info.value.hint


# --- other rules ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"llm.chain": ["local-30b", "nope"]}, "llm.chain"),
        ({"stt.chain": ["nope"]}, "stt.chain"),
        ({"tts.identity_chain": ["nope"]}, "tts.identity_chain"),
        ({"tts.identities.premwadee.backends": ["nope"]}, "tts.identities.premwadee.backends"),
        ({"llm.providers.local-4b.server": "nope"}, "llm.providers.local-4b.server"),
        ({"llm.chain": ["local-4b", "local-4b"]}, "llm.chain"),
        ({"vad.neg_threshold": 0.6}, "vad.threshold"),
        ({"vad.speculative_endpoint_ms": 700}, "vad.speculative_endpoint_ms"),
        ({"tts.chunker.first_max_chars": 200}, "tts.chunker"),
        ({"llm.slots.background": 0}, "llm.slots"),
        (
            {"llm.servers.local30b.extra_args": ["--port", "9000"]},
            "llm.servers.local30b.extra_args",
        ),
        ({"panel.host": "0.0.0.0"}, "panel.host"),
        ({"safety.fail_closed_categories": []}, "safety.fail_closed_categories"),
        ({"safety.tier1_review": 0.9}, "safety.tier1_review"),
        ({"tools.timeout_max_s": 601}, "tools.timeout_max_s"),
        ({"mic.mode": "loud"}, "mic.mode"),
        ({"chat.sources": ["twitch_ric"]}, "chat.sources"),
        ({"chat.twitch_irc.channel": "bad name!"}, "chat.twitch_irc.channel"),
        ({"stt.backends.pythaiasr.model_dir": "x"}, "stt.backends.pythaiasr.model_dir"),
        ({"tts.backends.edge.tier": "S0"}, "tts.backends.edge.tier"),
        ({"llm.providers.local-4b.kind": "ollama"}, "llm.providers.local-4b.kind"),
        ({"llm.providers.gemini.api_key_env": "sk-abc"}, "llm.providers.gemini.api_key_env"),
        ({"tts.identities.premwadee.rate": "fast"}, "tts.identities.premwadee.rate"),
        ({"brain.idle_jitter_s": 30.0}, "brain.idle_after_s"),
        ({"avatar.jitter_fallback_fps": 90}, "avatar.jitter_fallback_fps"),
        ({"characters": ["Pailin"]}, "characters.0"),
        ({"logging.level": "LOUD"}, "logging.level"),
    ],
)
def test_rules_reject_bad_values(overrides: dict[str, Any], path: str) -> None:
    assert error(**overrides).path == path


def test_games_need_a_fourth_llama_slot() -> None:
    err = error(**{"games.enabled": True})
    assert err.path == "llm.servers.local30b.parallel"
    cfg = load(
        **{
            "games.enabled": True,
            "llm.servers.local30b.parallel": 4,
            "llm.servers.local4b.parallel": 4,
        }
    )
    assert cfg.games.enabled


def test_friendly_normalisation() -> None:
    cfg = load(**{"chat.twitch_irc.channel": "#SomeStreamer", "logging.level": "debug"})
    assert cfg.chat.twitch_irc.channel == "somestreamer"
    assert cfg.logging.level == "DEBUG"


def test_plugin_kinds_are_namespaced() -> None:
    cfg = load(**{"llm.providers.custom": {"kind": "mypkg:llm", "model": "x"}})
    assert cfg.llm.providers["custom"].kind == "mypkg:llm"
    assert load(**{"chat.sources": ["mypkg:discord"]}).chat.sources == ["mypkg:discord"]


def test_endpointer_config_mirrors_vad_section() -> None:
    cfg = load(**{"vad.end_silence_ms": 700})
    ep = cfg.vad.endpointer_config()
    assert ep.end_silence_ms == 700 and ep.threshold == cfg.vad.threshold


def test_voice_spec_from_identity() -> None:
    spec = load().tts.identities["premwadee"].voice_spec("premwadee")
    assert (spec.voice, spec.rate, spec.pitch) == ("th-TH-PremwadeeNeural", "+8%", "+20Hz")
