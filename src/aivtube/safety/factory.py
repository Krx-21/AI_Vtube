"""Build the tier-0 filter and the gate from the validated config (for the app wiring).

Both builders block (they read and compile every list file and warm pythainlp): call them in a
thread at startup, e.g. ``await asyncio.to_thread(build_keyword_filter, cfg, chars)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from aivtube.config.schema import AppConfig, CharacterConfig
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.safety import Classifier, TextFilter
from aivtube.safety.audit import ModerationAudit, ModerationSink
from aivtube.safety.gate import LayeredSafetyGate
from aivtube.safety.keyword import KeywordRegexFilter

__all__ = ["build_keyword_filter", "build_safety_gate"]


def build_keyword_filter(
    cfg: AppConfig,
    characters: Iterable[CharacterConfig] = (),
    *,
    warm: bool = True,
) -> KeywordRegexFilter:
    """Base + private lists, ``safety.platform_overlays`` and each character's
    ``filters.toml``; the character's Latin aliases exempt its own ``@handle``."""
    s = cfg.safety
    chars = list(characters)
    platform: dict[str, Path] = {
        str(p): cfg.resolve_path(path) for p, path in s.platform_overlays.items()
    }
    character = {c.id: c.char_path(c.safety.overlay) for c in chars}
    aliases = {c.id: [*c.aliases, c.id, c.display_name] for c in chars}
    return KeywordRegexFilter(
        cfg.resolve_path(s.base_lists),
        cfg.resolve_path(s.private_lists),
        platform_overlays=platform,
        character_overlays=character,
        politics=s.politics,
        fail_closed=s.fail_closed_categories,
        mask_text=s.mask_text,
        handle_aliases=aliases,
        warm=warm,
    )


def build_safety_gate(
    cfg: AppConfig,
    characters: Iterable[CharacterConfig] = (),
    *,
    bus: EventBus,
    clock: Clock,
    tasks: TaskSupervisor,
    audit_sink: ModerationSink | None = None,
    classifier: Classifier | None = None,
    tier0: TextFilter | None = None,
    on_auto_strict: Callable[[str], None] | None = None,
    on_auto_freeze: Callable[[str], None] | None = None,
    warm: bool = True,
) -> LayeredSafetyGate:
    """The gate with its audit; ``audit_sink`` is usually ``ops_sink(ops_db)``."""
    filt = tier0 if tier0 is not None else build_keyword_filter(cfg, characters, warm=warm)
    audit = ModerationAudit(audit_sink, clock=clock, tasks=tasks, mask_text=cfg.safety.mask_text)
    return LayeredSafetyGate(
        filt,
        bus=bus,
        clock=clock,
        cfg=cfg.safety,
        audit=audit,
        classifier=classifier,
        on_auto_strict=on_auto_strict,
        on_auto_freeze=on_auto_freeze,
    )
