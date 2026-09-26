"""Operator commands (ARCHITECTURE.md §9): ``doctor``, ``bench``, ``setup``, ``models``,
``report`` and ``db``.

Submodules load lazily; ``aivtube.ops.models`` uses only the standard library so the launcher
preflight can import it.
"""

from __future__ import annotations

__all__: list[str] = []
