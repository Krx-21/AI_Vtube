"""The brain state machine (ARCHITECTURE.md §4.5).

```
BOOTING → PRE_SHOW ──GO_LIVE──► IDLE ◄──────── PAUSED ◄── FREEZE / HARD KILL (any state)
                                 │  ▲              │ RESUME
                     stimulus    │  │ UtteranceDone & nothing eligible
                                 ▼  │
                              DECIDING ──first segment audible──► SPEAKING
                                 ▲  └─abort/merge─┘                  │ HIGH/CRITICAL/barge-in
```

PRE_SHOW doubles as "idle while not live": operator and voice decisions may run before Go Live
and return to PRE_SHOW. Every change publishes ``StateChanged`` and is forwarded to the avatar
driver as a pose (``listening`` while the streamer speaks and she is not speaking).
"""

from __future__ import annotations

import logging
from typing import Final, Literal, TypeAlias

from aivtube.contracts.avatar import AvatarDriver, AvatarState
from aivtube.contracts.events import StateChanged
from aivtube.contracts.infra import EventBus

__all__ = ["ALLOWED", "BrainState", "BrainStateMachine", "avatar_state"]

log = logging.getLogger("aivtube.brain.state")

BrainState: TypeAlias = Literal["booting", "pre_show", "idle", "deciding", "speaking", "paused"]

ALLOWED: Final[dict[BrainState, frozenset[BrainState]]] = {
    "booting": frozenset({"pre_show", "idle", "paused"}),
    "pre_show": frozenset({"idle", "deciding", "speaking", "paused"}),
    "idle": frozenset({"pre_show", "deciding", "speaking", "paused"}),
    "deciding": frozenset({"idle", "pre_show", "speaking", "paused"}),
    "speaking": frozenset({"idle", "pre_show", "deciding", "paused"}),
    "paused": frozenset({"idle", "pre_show"}),
}


def avatar_state(state: BrainState, *, user_speaking: bool) -> AvatarState:
    """The avatar pose for a brain state."""
    if state == "paused":
        return "paused"
    if state == "speaking":
        return "speaking"
    if user_speaking:
        return "listening"
    if state == "deciding":
        return "thinking"
    return "idle"


class BrainStateMachine:
    """Holds the state, validates transitions, publishes ``StateChanged``, drives the avatar."""

    def __init__(
        self,
        *,
        character: str,
        bus: EventBus,
        avatar: AvatarDriver | None = None,
        initial: BrainState = "booting",
    ) -> None:
        self.character = character
        self._bus = bus
        self._avatar = avatar
        self._state: BrainState = initial
        self._pose: AvatarState | None = None
        self._user_speaking = False

    @property
    def state(self) -> BrainState:
        return self._state

    def set(self, new: BrainState, *, reason: str = "") -> bool:
        """Move to ``new``; returns ``False`` (and changes nothing) for a forbidden move."""
        old = self._state
        if new == old:
            return True
        if new not in ALLOWED[old]:
            log.warning(
                "brain %s: refused state change %s -> %s (%s)", self.character, old, new, reason
            )
            return False
        self._state = new
        log.debug("brain %s: %s -> %s (%s)", self.character, old, new, reason)
        self._bus.publish(StateChanged(character=self.character, old=old, new=new))
        self._refresh()
        return True

    def set_user_speaking(self, on: bool) -> None:
        if on != self._user_speaking:
            self._user_speaking = on
            self._refresh()

    def _refresh(self) -> None:
        pose = avatar_state(self._state, user_speaking=self._user_speaking)
        if pose == self._pose or self._avatar is None:
            self._pose = pose
            return
        self._pose = pose
        try:
            self._avatar.set_state(pose)
        except Exception:  # the avatar is a sink: it must never break the brain
            log.exception("avatar set_state(%s) failed", pose)
