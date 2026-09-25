"""Control and game fakes: ``FakeControlSurface``, ``FakeGameServer``, ``FakeGameBrain``."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Mapping, Sequence
from typing import Any

from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.games import GameAction, GameForce

__all__ = ["FakeControlSurface", "FakeGameBrain", "FakeGameServer"]


class FakeControlSurface:
    """``ControlSurface`` that records commands; ``results`` scripts per-kind outcomes."""

    def __init__(self, results: Mapping[OpKind, OpResult] | None = None) -> None:
        self.results = dict(results or {})
        self.commands: list[OpCommand] = []
        self.state: dict[str, Any] = {"paused": False, "muted": False, "live": False}

    async def execute(self, cmd: OpCommand) -> OpResult:
        t0 = time.perf_counter()
        self.commands.append(cmd)
        if cmd.kind is OpKind.FREEZE:
            self.state["paused"] = True
        elif cmd.kind is OpKind.RESUME:
            self.state["paused"] = False
        elif cmd.kind in (OpKind.MUTE, OpKind.UNMUTE):
            self.state["muted"] = cmd.kind is OpKind.MUTE
        elif cmd.kind is OpKind.GO_LIVE:
            self.state["live"] = True
        res = self.results.get(cmd.kind, OpResult(True))
        return OpResult(res.ok, res.detail, (time.perf_counter() - t0) * 1000.0)

    def snapshot(self) -> Mapping[str, Any]:
        return {**self.state, "commands": len(self.commands)}


class FakeGameServer:
    """``GameServer`` (M4) with in-memory games, actions and forces."""

    def __init__(self, character: str = "pailin", port: int = 8000) -> None:
        self.character = character
        self.port = port
        self._actions: dict[str, dict[str, GameAction]] = {}
        self._force: GameForce | None = None
        self.executed: list[tuple[str, str, Mapping[str, Any] | None, str | None, str]] = []
        self.speech: list[tuple[bool, bool, str | None]] = []
        self._ids = itertools.count(1)
        self._stop = asyncio.Event()

    async def serve(self) -> None:
        await self._stop.wait()

    def stop(self) -> None:
        self._stop.set()

    def register(self, game: str, actions: Sequence[GameAction]) -> None:
        bucket = self._actions.setdefault(game, {})
        for a in actions:
            bucket[a.name] = a

    def force(self, force: GameForce | None) -> None:
        self._force = force

    def games(self) -> list[str]:
        return sorted(self._actions)

    def actions(self, game: str | None = None) -> list[GameAction]:
        games = [game] if game is not None else self.games()
        return [a for g in games for a in self._actions.get(g, {}).values()]

    def active_force(self) -> GameForce | None:
        return self._force

    async def execute(
        self, game: str, name: str, data: Mapping[str, Any] | None, force_id: str | None
    ) -> str:
        if name not in self._actions.get(game, {}):
            raise KeyError(f"{game} has no action {name!r}")
        action_id = f"act-{next(self._ids)}"
        self.executed.append((game, name, data, force_id, action_id))
        if force_id is not None and self._force is not None and self._force.id == force_id:
            self._force = None
        return action_id

    async def speech_finished(
        self, is_final: bool, cancelled: bool = False, reason: str | None = None
    ) -> None:
        self.speech.append((is_final, cancelled, reason))


class FakeGameBrain:
    """``GameBrain`` that records every callback."""

    def __init__(self) -> None:
        self.contexts: list[tuple[str, str, bool]] = []
        self.forces: list[tuple[GameForce, tuple[GameAction, ...]]] = []
        self.dropped: list[tuple[GameForce, str]] = []
        self.results: list[tuple[str, str, bool, str | None, bool]] = []
        self.changed: list[str] = []

    async def on_context(self, game: str, message: str, silent: bool) -> None:
        self.contexts.append((game, message, silent))

    async def on_force(self, force: GameForce, actions: Sequence[GameAction]) -> None:
        self.forces.append((force, tuple(actions)))

    async def on_force_dropped(self, force: GameForce, reason: str) -> None:
        self.dropped.append((force, reason))

    async def on_action_result(
        self, game: str, name: str, success: bool, message: str | None, forced: bool
    ) -> None:
        self.results.append((game, name, success, message, forced))

    async def on_actions_changed(self, game: str) -> None:
        self.changed.append(game)
