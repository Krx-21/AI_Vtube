"""Layered, validated configuration (ARCHITECTURE.md §8).

Layers, later wins: ``config/defaults.toml`` → active ``[profiles.<name>]`` →
``characters/<id>/character.toml`` (character-scoped; see ``load_character``) →
``config/user.toml`` → env ``AIVTUBE__SECTION__KEY`` → CLI. Secrets live only in ``.env``.

``ConfigError`` and the raw layering helpers (``aivtube.config.layers``) use only the standard
library; the pydantic models load lazily on first use, so the stdlib-only launcher can import
``aivtube.config.layers`` safely.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from aivtube.config.errors import ConfigError
from aivtube.config.layers import (
    CURRENT_SCHEMA_VERSION,
    collect_layers,
    deep_merge,
    env_overrides,
    expand_character,
    find_root,
)

if TYPE_CHECKING:
    from aivtube.config.load import (
        config_error_from_validation,
        load_character,
        load_characters,
        load_config,
        load_secrets,
        validate_stage,
    )
    from aivtube.config.migration import dumps_toml, migrate, write_user_overrides
    from aivtube.config.schema import (
        AppConfig,
        CharacterConfig,
        Secrets,
        load_lexicon,
    )

_LAZY: dict[str, str] = {
    "AppConfig": "schema",
    "CharacterConfig": "schema",
    "Secrets": "schema",
    "load_lexicon": "schema",
    "config_error_from_validation": "load",
    "load_character": "load",
    "load_characters": "load",
    "load_config": "load",
    "load_secrets": "load",
    "validate_stage": "load",
    "dumps_toml": "migration",
    "migrate": "migration",
    "write_user_overrides": "migration",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "AppConfig",
    "CharacterConfig",
    "ConfigError",
    "Secrets",
    "collect_layers",
    "config_error_from_validation",
    "deep_merge",
    "dumps_toml",
    "env_overrides",
    "expand_character",
    "find_root",
    "load_character",
    "load_characters",
    "load_config",
    "load_lexicon",
    "load_secrets",
    "migrate",
    "validate_stage",
    "write_user_overrides",
]
