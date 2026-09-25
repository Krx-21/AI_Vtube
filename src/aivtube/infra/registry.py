"""``PluginRegistry``: named factories per kind (llm, stt, tts, chat, tool, avatar, …).

Built-ins and third-party packages register through the ``aivtube.plugins`` entry-point
group; each entry point is a ``register(registry)`` callable. A plugin that fails to load is
logged and recorded in ``load_errors``; it never stops startup.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["Factory", "PluginRegistry"]

log = logging.getLogger("aivtube.plugins")

Factory = Callable[[Mapping[str, Any], Any], Any]


class PluginRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Factory]] = {}
        self.load_errors: list[tuple[str, str]] = []
        self.loaded: list[str] = []

    def register(self, kind: str, name: str, factory: Factory, *, replace: bool = False) -> None:
        """Register ``factory(cfg, services)`` as ``kind``/``name``."""
        if not callable(factory):
            raise TypeError(f"factory for {kind}/{name} is not callable")
        table = self._factories.setdefault(kind, {})
        if name in table and not replace:
            raise ValueError(f"{kind}/{name} is already registered")
        table[name] = factory

    def create(self, kind: str, name: str, cfg: Mapping[str, Any], services: Any) -> Any:
        factory = self._factories.get(kind, {}).get(name)
        if factory is None:
            known = ", ".join(self.names(kind)) or "none"
            raise KeyError(f"no {kind} plugin named {name!r} (registered: {known})")
        return factory(cfg, services)

    def names(self, kind: str) -> list[str]:
        return sorted(self._factories.get(kind, {}))

    def kinds(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, key: object) -> bool:
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        kind, name = key
        return name in self._factories.get(kind, {})

    def load_entry_points(self, group: str = "aivtube.plugins") -> None:
        """Call every ``register(registry)`` entry point in ``group`` (built-in first)."""
        points = sorted(
            importlib.metadata.entry_points(group=group),
            key=lambda ep: (ep.name != "builtin", ep.name),
        )
        for ep in points:
            if ep.name in self.loaded:
                continue
            try:
                register = ep.load()
                register(self)
            except Exception as exc:
                self.load_errors.append((ep.name, f"{type(exc).__name__}: {exc}"))
                log.exception("plugin %r (%s) failed to load", ep.name, ep.value)
            else:
                self.loaded.append(ep.name)
