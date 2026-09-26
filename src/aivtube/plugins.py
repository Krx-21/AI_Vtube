"""Built-in plugin hook, exposed as the ``aivtube.plugins`` entry point ``builtin``.

``infra.PluginRegistry.load_entry_points()`` calls every ``register(registry)`` in the group.
Third-party packages register their own factories the same way. The built-in factories are
wired here by the app work package; until then this hook deliberately registers nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
    """Register the built-in adapter factories with ``registry`` (currently a no-op)."""
    del registry
