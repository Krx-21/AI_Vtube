"""Game protocols, Neuro-SDK compatible (ARCHITECTURE.md §3.13, §4.15). Frozen at M0, used
from M4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from aivtube.contracts.types import Priority

__all__ = ["GameAction", "GameBrain", "GameForce", "GameServer"]


@dataclass(frozen=True, slots=True)
class GameAction:
    game: str
    name: str
    description: str
    schema: Mapping[str, Any] | None  # JSON Schema of the action's data; None = no data


@dataclass(frozen=True, slots=True)
class GameForce:
    """An ``actions/force`` request: the game needs one of ``action_names`` now."""

    id: str
    game: str
    query: str
    state: str | None
    action_names: tuple[str, ...]
    priority: Priority
    ephemeral: bool
    attempt: int = 0


@runtime_checkable
class GameServer(Protocol):
    character: str
    port: int

    async def serve(self) -> None: ...

    def games(self) -> list[str]: ...

    def actions(self, game: str | None = None) -> list[GameAction]: ...

    def active_force(self) -> GameForce | None: ...

    async def execute(
        self, game: str, name: str, data: Mapping[str, Any] | None, force_id: str | None
    ) -> str:
        """Send an action to the game; returns the action id."""
        ...

    async def speech_finished(
        self, is_final: bool, cancelled: bool = False, reason: str | None = None
    ) -> None: ...


@runtime_checkable
class GameBrain(Protocol):
    async def on_context(self, game: str, message: str, silent: bool) -> None: ...

    async def on_force(self, force: GameForce, actions: Sequence[GameAction]) -> None: ...

    async def on_force_dropped(self, force: GameForce, reason: str) -> None: ...

    async def on_action_result(
        self, game: str, name: str, success: bool, message: str | None, forced: bool
    ) -> None: ...

    async def on_actions_changed(self, game: str) -> None: ...
