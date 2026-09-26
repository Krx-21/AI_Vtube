"""Pending-stimulus store and selection (ARCHITECTURE.md §4.3).

When the brain is free, the eligible stimulus with the lowest ``Rank`` wins; ties go to the
oldest. Waiting earns an aging bonus of −1 rank per 5 s, so nothing starves. Expired stimuli
are dropped and announced with ``StimulusExpired``.

Cadence rules:

- While the streamer speaks (``user_speaking``) only OPERATOR stimuli are eligible.
- While her own audio still plays (``speaking``) only OPERATOR stimuli and priority ≥ HIGH are
  eligible: LOW and MEDIUM wait for ``UtteranceDone`` (§4.4). LOW/MEDIUM stimuli also wait
  ``post_speech_gap_s`` after the brain becomes free.
- Chat (CHAT and MENTION) decisions are at least ``chat_min_interval_s`` apart (4 s, or 8 s
  after a voice turn in the last 20 s). The chat window additionally gathers for 1 s after the
  brain becomes free (or after its first message) before deciding.
- Window chat is one coalesced CHAT stimulus; ``drain_context`` attaches ``window.select(k)``
  to a CHAT decision, and to any other non-voice decision once the chat interval allows.
- At most one SUPPORT acknowledgement per decision: SUPPORT is never merged.

A stimulus arriving mid-decision stays pending and is merged into the next decision by
``drain_context`` (T1 semantics); the two abort-restart exceptions live in ``brain.preempt``,
and an aborted decision's context comes back through :meth:`Arbiter.restore`.

``next()`` waits on an ``asyncio.Event`` (poked by ``push``/``poke``) plus clock timers of at
least 50 ms; it returns ``None`` once the idle deadline passes with nothing eligible.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from aivtube.brain._util import wait_event
from aivtube.contracts.chat import ChatSelection, ChatWindow
from aivtube.contracts.events import StimulusExpired, StimulusQueued
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.types import Priority, Stimulus, StimulusKind

__all__ = ["Arbiter", "ArbiterConfig", "MergedContext", "is_say"]

log = logging.getLogger("aivtube.brain.arbiter")

IdleDeadline = float | None | Callable[[], float | None]

_MERGEABLE = frozenset(
    {
        StimulusKind.VOICE,
        StimulusKind.MENTION,
        StimulusKind.CHARACTER,
        StimulusKind.VISION,
        StimulusKind.OPERATOR,  # DIRECT only; SAY is never merged
    }
)
_CHATTY = frozenset({StimulusKind.CHAT, StimulusKind.MENTION})
# Voice turns focus on the streamer (and have the small 250-token tail): no window chat.
_NO_CHAT = frozenset({StimulusKind.VOICE, StimulusKind.GAME_FORCE})


def is_say(s: Stimulus) -> bool:
    """An operator SAY (spoken verbatim, bypasses the LLM)."""
    return s.kind is StimulusKind.OPERATOR and s.payload.get("op") == "say"


@dataclass(frozen=True, slots=True)
class MergedContext:
    """The primary stimulus plus everything merged into this decision."""

    primary: Stimulus
    merged: tuple[Stimulus, ...] = ()
    chat: ChatSelection | None = None
    support: tuple[Stimulus, ...] = ()
    game_context: tuple[Stimulus, ...] = ()

    @property
    def stimuli(self) -> tuple[Stimulus, ...]:
        """Primary first, then merged and game-context stimuli."""
        return (self.primary, *self.merged, *self.game_context)

    @property
    def merged_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in (*self.merged, *self.game_context))

    def has_kind(self, kind: StimulusKind) -> bool:
        return any(s.kind is kind for s in self.stimuli)


@dataclass(frozen=True, slots=True)
class ArbiterConfig:
    chat_min_interval_s: float = 4.0
    chat_min_interval_voice_s: float = 8.0
    recent_voice_s: float = 20.0
    chat_gather_s: float = 1.0
    chat_k: int = 3
    post_speech_gap_s: float = 0.3
    aging_s: float = 5.0
    min_wait_s: float = 0.05
    recheck_s: float = 0.5

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> ArbiterConfig:
        """Pick the known keys of ``cfg`` (e.g. the flattened ``[brain]`` section)."""
        if not cfg:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})


class Arbiter:
    """See the module docstring. Use from the event-loop thread only."""

    def __init__(
        self,
        clock: Clock,
        window: ChatWindow,
        cfg: Mapping[str, Any] | ArbiterConfig | None = None,
        *,
        bus: EventBus | None = None,
        character: str | None = None,
    ) -> None:
        self._clock = clock
        self._window = window
        self.cfg = cfg if isinstance(cfg, ArbiterConfig) else ArbiterConfig.from_mapping(cfg)
        self._bus = bus
        self._character = character
        self._pending: dict[str, Stimulus] = {}
        self._chat: Stimulus | None = None  # the coalesced window stimulus
        self._carry_chat: ChatSelection | None = None
        self._wake = asyncio.Event()
        self._free_since = -math.inf
        self._last_chat: float | None = None
        self._last_voice: float | None = None
        self._accept: Callable[[Stimulus], bool] | None = None
        self.expired = 0

    @property
    def window(self) -> ChatWindow:
        return self._window

    # --- intake side ------------------------------------------------------------------------
    def push(self, s: Stimulus) -> None:
        """Queue a stimulus (sync, never blocks). Window chat is coalesced into one."""
        if s.kind is StimulusKind.CHAT:
            if self._chat is not None:
                self.poke()
                return
            self._chat = s
        else:
            self._pending[s.id] = s
        self._publish(StimulusQueued(character=s.character, stimulus=s))
        self.poke()

    def remove(self, stimulus_id: str) -> bool:
        """Withdraw a pending stimulus (read-aloud dedupe of a mention)."""
        if self._chat is not None and self._chat.id == stimulus_id:
            self._chat = None
            return True
        return self._pending.pop(stimulus_id, None) is not None

    def restore(self, ctx: MergedContext) -> None:
        """Put an aborted decision's stimuli back so the restart merges them (§4.3b)."""
        for s in ctx.stimuli:
            if s.kind is StimulusKind.CHAT:
                if self._chat is None:
                    self._chat = s
            elif s.kind is not StimulusKind.IDLE:
                self._pending.setdefault(s.id, s)
        for s in ctx.support:
            self._pending.setdefault(s.id, s)
        if ctx.chat is not None:
            self._carry_chat = _combine(self._carry_chat, ctx.chat)
        self.poke()

    def poke(self) -> None:
        """Re-evaluate eligibility (state, speaking or user-speaking flags changed)."""
        self._wake.set()

    def mark_free(self, t: float) -> None:
        """The brain became free at ``t`` (playback ended); starts the gap and chat gather."""
        self._free_since = max(self._free_since, t)
        self.poke()

    def note_voice_turn(self, t: float) -> None:
        self._last_voice = t

    # --- selection --------------------------------------------------------------------------
    async def next(
        self,
        *,
        idle_deadline: IdleDeadline,
        user_speaking: Callable[[], bool],
        speaking: Callable[[], bool] | None = None,
        accept: Callable[[Stimulus], bool] | None = None,
    ) -> Stimulus | None:
        """Wait for the best eligible stimulus; ``None`` when the idle deadline passes first.

        ``speaking`` (extra): her audio still plays, so only ≥ HIGH may start (§4.4).
        ``accept`` (extra): state filter (PAUSED accepts nothing, PRE_SHOW only operator and
        voice). ``idle_deadline`` may be a callable so a deadline armed while waiting counts.
        """
        self._accept = accept
        self._free_since = max(self._free_since, self._clock.now())
        while True:
            self._wake.clear()
            now = self._clock.now()
            self._expire(now)
            us = user_speaking()
            sp = speaking() if speaking is not None else False
            best: Stimulus | None = None
            best_key = (math.inf, math.inf)
            wake_at = math.inf
            soon = False
            for s in self._candidates():
                ready = self._ready_at(s, now, us, sp, accept)
                if ready is None:
                    wake_at = min(wake_at, now + self.cfg.recheck_s)
                    continue
                soon = True
                if ready > now:
                    wake_at = min(wake_at, ready)
                    continue
                key = (s.rank - (now - s.created) / self.cfg.aging_s, s.created)
                if key < best_key:
                    best, best_key = s, key
            if best is not None:
                if best is not self._chat:  # the chat stimulus stays until drain_context
                    self._pending.pop(best.id, None)
                return best
            idle_at = idle_deadline() if callable(idle_deadline) else idle_deadline
            if idle_at is not None and not soon and not us:
                if now >= idle_at:
                    return None
                wake_at = min(wake_at, idle_at)
            for s in self._pending.values():
                if s.ttl_s is not None:
                    wake_at = min(wake_at, s.created + s.ttl_s + 1e-3)
            seconds = None if math.isinf(wake_at) else max(self.cfg.min_wait_s, wake_at - now)
            await wait_event(self._wake, self._clock, seconds)

    def drain_context(self, primary: Stimulus) -> MergedContext:
        """Everything queued since the last decision, merged into ``primary``'s decision."""
        now = self._clock.now()
        self._pending.pop(primary.id, None)
        accept = self._accept
        merged: list[Stimulus] = []
        game: list[Stimulus] = []
        if not is_say(primary):
            for s in list(self._pending.values()):
                if accept is not None and not accept(s):
                    continue
                if s.kind is StimulusKind.GAME_CONTEXT:
                    game.append(s)
                elif s.kind in _MERGEABLE and not is_say(s):
                    merged.append(s)
                else:
                    continue
                del self._pending[s.id]
        chat: ChatSelection | None = None
        take_chat = primary.kind is StimulusKind.CHAT
        if (
            not take_chat
            and self._chat is not None
            and primary.kind not in _NO_CHAT
            and not is_say(primary)
            and (accept is None or accept(self._chat))
            and (self._last_chat is None or now >= self._last_chat + self._chat_interval(now))
        ):
            take_chat = self._window.pending()[0] > 0  # a decision starts anyway: no gather
        if take_chat:
            chat = self._window.select(now, self.cfg.chat_k)
            if primary.kind is StimulusKind.CHAT or self._chat is not None:
                self._chat = None
        if self._carry_chat is not None and not is_say(primary):
            chat = _combine(self._carry_chat, chat)
            self._carry_chat = None
        kinds = {primary.kind, *(s.kind for s in merged)}
        if chat is not None or kinds & _CHATTY:
            self._last_chat = now
        if StimulusKind.VOICE in kinds:
            self._last_voice = now
        support = (primary,) if primary.kind is StimulusKind.SUPPORT else ()
        return MergedContext(primary, tuple(merged), chat, support, tuple(game))

    def pending(self) -> list[Stimulus]:
        """Pending stimuli, best first (the coalesced chat stimulus included)."""
        now = self._clock.now()
        items = list(self._candidates())
        items.sort(key=lambda s: (s.rank - (now - s.created) / self.cfg.aging_s, s.created))
        return items

    def clear(self, *, keep: frozenset[StimulusKind] = frozenset({StimulusKind.SUPPORT})) -> None:
        """Drop everything pending except ``keep`` kinds, and empty the chat window."""
        self._pending = {k: s for k, s in self._pending.items() if s.kind in keep}
        if StimulusKind.CHAT not in keep:
            self._chat = None
            self._carry_chat = None
            self._window.select(self._clock.now(), 0)  # consumes the window, picks nobody
        self.poke()

    # --- internals --------------------------------------------------------------------------
    def _candidates(self) -> list[Stimulus]:
        items = list(self._pending.values())
        if self._chat is not None:
            items.append(self._chat)
        return items

    def _ready_at(
        self,
        s: Stimulus,
        now: float,
        user_speaking: bool,
        speaking: bool,
        accept: Callable[[Stimulus], bool] | None,
    ) -> float | None:
        """When ``s`` becomes eligible; ``None`` while a flag blocks it (re-checked on poke)."""
        if accept is not None and not accept(s):
            return None
        operator = s.kind is StimulusKind.OPERATOR
        if user_speaking and not operator:
            return None
        urgent = operator or s.priority >= Priority.HIGH
        if speaking and not urgent:
            return None
        t = now
        if not urgent:
            t = max(t, self._free_since + self.cfg.post_speech_gap_s)
        if s.kind is StimulusKind.CHAT:
            ready = self._chat_ready_at(s, now)
            if ready is None:
                return None
            t = max(t, ready)
        elif s.kind is StimulusKind.MENTION and self._last_chat is not None:
            t = max(t, self._last_chat + self._chat_interval(now))
        return t

    def _chat_ready_at(self, s: Stimulus, now: float) -> float | None:
        count, _ = self._window.pending()
        if count <= 0:
            if s is self._chat:
                self._chat = None  # the window emptied (expired or consumed)
            return None
        t = max(self._free_since, s.created) + self.cfg.chat_gather_s
        if self._last_chat is not None:
            t = max(t, self._last_chat + self._chat_interval(now))
        return t

    def _chat_interval(self, now: float) -> float:
        recent_voice = self._last_voice is not None and now - self._last_voice < (
            self.cfg.recent_voice_s
        )
        return self.cfg.chat_min_interval_voice_s if recent_voice else self.cfg.chat_min_interval_s

    def _expire(self, now: float) -> None:
        for s in list(self._pending.values()):
            if s.ttl_s is not None and now - s.created > s.ttl_s:
                del self._pending[s.id]
                self.expired += 1
                log.debug("stimulus %s (%s) expired", s.id, s.kind.value)
                self._publish(StimulusExpired(character=s.character, stimulus_id=s.id))

    def _publish(self, event: StimulusQueued | StimulusExpired) -> None:
        if self._bus is not None:
            self._bus.publish(event)


def _combine(a: ChatSelection | None, b: ChatSelection | None) -> ChatSelection | None:
    if a is None:
        return b
    if b is None:
        return a

    def merge(x: tuple[Any, ...], y: tuple[Any, ...]) -> tuple[Any, ...]:
        seen: set[str] = set()
        out = []
        for m in (*x, *y):
            if m.id not in seen:
                seen.add(m.id)
                out.append(m)
        return tuple(out)

    return ChatSelection(
        merge(a.must_ack, b.must_ack),
        merge(a.candidates, b.candidates),
        merge(a.ambient, b.ambient),
    )
