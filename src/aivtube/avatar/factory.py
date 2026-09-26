"""Build the avatar sink and driver for one character from the validated config (§8)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from aivtube.avatar.discovery import VTSDiscovery
from aivtube.avatar.driver import LiveAvatarDriver, TickerLike
from aivtube.avatar.null_sink import NullSink
from aivtube.avatar.vts_client import VTSClient
from aivtube.avatar.vts_sink import VTSSink
from aivtube.contracts.avatar import AvatarSink
from aivtube.contracts.infra import Clock, TaskSupervisor
from aivtube.infra.ticker import PrecisionTicker

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig, CharacterConfig

__all__ = ["build_avatar_driver", "build_avatar_sink"]

log = logging.getLogger("aivtube.avatar")


def build_avatar_sink(
    app: AppConfig,
    character: CharacterConfig,
    clock: Clock,
    *,
    tasks: TaskSupervisor | None = None,
    discovery: VTSDiscovery | bool = True,
) -> AvatarSink:
    """``VTSSink`` for ``avatar.sink = "vts"``, otherwise ``NullSink``.

    ``discovery=True`` gives the sink its own UDP 47779 listener; pass one ``VTSDiscovery``
    to share it between twins, or ``False`` to always use ``vts_url``.
    """
    component = f"avatar:{character.id}"
    kind = app.avatar.sink
    if kind != "vts":
        if kind == "browser":
            log.warning("avatar.sink = 'browser' arrives in M6; the avatar is disabled for now")
        return NullSink(component=component)
    av = character.avatar
    cfg = app.avatar
    client = VTSClient(
        av.vts_url,
        av.plugin_name,
        av.plugin_developer,
        character.resolve_path(av.token_file),
        clock=clock,
        request_timeout=cfg.request_timeout_s,
        max_inflight=cfg.max_inflight,
        tasks=tasks,
    )
    if discovery is True:
        listener: VTSDiscovery | None = VTSDiscovery(clock=clock)
    elif discovery is False:
        listener = None
    else:
        listener = discovery
    backoff = (float(cfg.reconnect_backoff_s[0]), float(cfg.reconnect_backoff_s[-1]))
    return VTSSink(
        client,
        character.emotion_map,
        clock,
        name="vts",
        component=component,
        fade_s=cfg.emotion_fade_s,
        reconnect_backoff=backoff,
        custom_prefix=character.display_name,
        window_title=av.vts_window_title,
        discovery=listener,
        discovery_timeout_s=cfg.discovery_timeout_s,
    )


def build_avatar_driver(
    app: AppConfig,
    character: CharacterConfig,
    sink: AvatarSink,
    clock: Clock,
    *,
    ticker_factory: Callable[..., TickerLike] = PrecisionTicker,
    seed: int = 0,
) -> LiveAvatarDriver:
    cfg = app.avatar
    return LiveAvatarDriver(
        sink,
        clock,
        fps=cfg.fps,
        lead_ms=float(cfg.lead_ms),
        jitter_fallback_fps=cfg.jitter_fallback_fps,
        ticker_factory=ticker_factory,
        jitter_limit_ms=cfg.jitter_limit_ms,
        emotion_map=character.emotion_map,
        emotion_fade_s=cfg.emotion_fade_s,
        idle_motion=cfg.idle_motion,
        seed=seed,
    )
