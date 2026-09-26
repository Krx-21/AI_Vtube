"""``ScoredChatWindow``: the limited chat window (ARCHITECTURE.md §3.8, §4.2, §4.3).

Neuro "picks what to respond to within a limited window" of chat (T1); how many messages she
considers and how they are weighted is unknown, so this scoring is our own design:

    2·exp(−age/15 s) + 3·mention + 0.8·question + 0.7·sub + 0.5·mod + 0.3·vip + 0.5·first_msg
    − 1.5·ln(dup) − 5·[user picked < 60 s ago]

``add()`` routes each message: donations, subs, gift subs, raids and redeems go to the
must-acknowledge queue ("priority"); plain chat goes to the rolling window ("window");
duplicates, shared-chat messages from other channels, ``!commands``, empty, oversized and
rate-limited messages are dropped. ``select()`` softmax-samples up to ``k`` candidates (one
per user and one per near-duplicate group) with a seedable RNG, then consumes the window.
Everything runs on the event loop; no method blocks or awaits.
"""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from aivtube.chat._text import AliasMatcher, dedupe_key, is_question
from aivtube.contracts.chat import ChatSelection
from aivtube.contracts.infra import Clock
from aivtube.contracts.types import ChatMessage, MsgKind

__all__ = [
    "SUPPORT_KINDS",
    "WEIGHTS",
    "ScoredChatWindow",
    "WindowConfig",
]

SUPPORT_KINDS: frozenset[MsgKind] = frozenset(
    {MsgKind.DONATION, MsgKind.SUB, MsgKind.GIFT_SUB, MsgKind.RAID, MsgKind.REDEEM}
)
"""Kinds that go to the must-acknowledge queue (§4.2: SUPPORT stimuli)."""

WEIGHTS: dict[str, float] = {
    "recency": 2.0,
    "mention": 3.0,
    "question": 0.8,
    "sub": 0.7,
    "mod": 0.5,
    "vip": 0.3,
    "first_msg": 0.5,
    "dup": 1.5,
    "recent_pick": 5.0,
}
"""The §4.3 score weights (read-only by convention)."""

AddResult = Literal["window", "priority", "dropped"]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Window tuning; the first six fields mirror ``[chat.window]`` and ``chat.max_msg_chars``.

    ``user_burst`` messages per ``user_burst_window_s`` is the per-user rate limit; extra
    messages are dropped as ``rate_limited``. ``max_ambient`` bounds the ambient context.
    """

    horizon_s: float = 40.0
    must_ack_ttl_s: float = 600.0
    temperature: float = 0.6
    user_cooldown_s: float = 60.0
    dedupe_lru: int = 5000
    max_msg_chars: int = 300
    recency_tau_s: float = 15.0
    user_burst: int = 3
    user_burst_window_s: float = 10.0
    max_window: int = 500
    max_ambient: int = 12
    history_s: float = 300.0
    history_max: int = 2000

    def __post_init__(self) -> None:
        if self.temperature <= 0 or self.recency_tau_s <= 0 or self.horizon_s <= 0:
            raise ValueError("temperature, recency_tau_s and horizon_s must be positive")
        if self.dedupe_lru < 1 or self.max_window < 1 or self.user_burst < 1:
            raise ValueError("dedupe_lru, max_window and user_burst must be at least 1")

    @classmethod
    def from_config(cls, chat: Any, **overrides: Any) -> WindowConfig:
        """Build from a ``config.schema.ChatConfig`` (duck-typed: ``.window`` and
        ``.max_msg_chars``); ``overrides`` replace individual fields."""
        w = chat.window
        values: dict[str, Any] = {
            "horizon_s": float(w.horizon_s),
            "must_ack_ttl_s": float(w.must_ack_ttl_s),
            "temperature": float(w.temperature),
            "user_cooldown_s": float(w.user_cooldown_s),
            "dedupe_lru": int(w.dedupe_lru),
            "max_msg_chars": int(chat.max_msg_chars),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(slots=True, eq=False)
class _Entry:
    msg: ChatMessage
    user_key: str
    group: str  # near-duplicate key
    mention: bool
    question: bool


def _key(m: ChatMessage) -> str:
    return f"{m.platform.value}:{m.id}"


def _user_key(m: ChatMessage) -> str:
    return f"{m.platform.value}:{m.user.id}"


class ScoredChatWindow:
    """The real ``ChatWindow`` (§3.8). Pass the clock the adapters stamp ``received`` with.

    ``name_matcher`` is a ``Callable[[str], bool]`` (e.g. ``text.NameMatcher``) or the
    character's aliases, which are wrapped in an :class:`~aivtube.chat._text.AliasMatcher`.
    ``rng`` makes sampling reproducible (``random.Random(seed)``).

    Instrumentation: ``drops`` counts drop reasons; ``last_reason`` is the reason of the most
    recent ``add()`` (``"window"``, ``"priority"`` or a drop reason, for ``ChatDropped``).
    """

    def __init__(
        self,
        clock: Clock,
        cfg: WindowConfig | None = None,
        name_matcher: Callable[[str], bool] | Iterable[str] = (),
        rng: random.Random | None = None,
    ) -> None:
        self._clock = clock
        self.cfg = cfg or WindowConfig()
        self._is_mention: Callable[[str], bool] = (
            name_matcher if callable(name_matcher) else AliasMatcher(name_matcher)
        )
        self._rng = rng or random.Random()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._window: OrderedDict[str, _Entry] = OrderedDict()
        self._must_ack: OrderedDict[str, ChatMessage] = OrderedDict()
        self._groups: Counter[str] = Counter()
        self._user_times: dict[str, deque[float]] = {}
        self._last_picked: dict[str, float] = {}
        self._history: deque[ChatMessage] = deque(maxlen=self.cfg.history_max)
        self.drops: Counter[str] = Counter()
        self.last_reason = ""

    # --- ChatWindow ------------------------------------------------------------------------
    def add(self, m: ChatMessage) -> AddResult:
        now = self._clock.now()
        key = _key(m)
        if key in self._seen:
            self._seen.move_to_end(key)
            return self._drop("duplicate")
        self._remember(key)
        if m.source_channel:
            return self._drop("shared_chat")
        if m.kind in SUPPORT_KINDS:
            self._must_ack[key] = m
            self._record(m, now)
            self.last_reason = "priority"
            return "priority"
        if m.kind is MsgKind.SYSTEM:
            return self._drop("system")
        text = m.text.strip()
        if not text:
            return self._drop("empty")
        if text.startswith("!"):
            return self._drop("command")
        if len(text) > self.cfg.max_msg_chars:
            return self._drop("too_long")
        user_key = _user_key(m)
        if self._rate_limited(user_key, now):
            return self._drop("rate_limited")
        entry = _Entry(m, user_key, dedupe_key(text), self._is_mention(text), is_question(text))
        self._window[key] = entry
        self._groups[entry.group] += 1
        self._record(m, now)
        while len(self._window) > self.cfg.max_window:
            self._remove(next(iter(self._window)))
        self.last_reason = "window"
        return "window"

    def select(self, now: float, k: int = 3) -> ChatSelection:
        """Softmax-sample up to ``k`` candidates, return every live must-ack message sorted by
        ``value_usd`` (then age), and the most recent unpicked messages as ambient context.
        Consumes the window and the must-ack queue."""
        self._expire(now)
        must_ack = tuple(sorted(self._must_ack.values(), key=lambda m: (-m.value_usd, m.ts, m.id)))
        scored = [(e, self._score(e, now)) for e in self._window.values()]
        picked = self._sample(scored, k)
        for e in picked:
            self._last_picked[e.user_key] = now
        chosen = {id(e) for e in picked}
        candidates = tuple(e.msg for e in sorted(picked, key=lambda e: (e.msg.ts, e.msg.received)))
        rest = [e.msg for e, _ in scored if id(e) not in chosen]
        ambient = tuple(rest[-self.cfg.max_ambient :]) if self.cfg.max_ambient > 0 else ()
        self._window.clear()
        self._must_ack.clear()
        self._groups.clear()
        self._prune_picks(now)
        return ChatSelection(must_ack, candidates, ambient)

    def pending(self) -> tuple[int, bool]:
        self._expire(self._clock.now())
        has_mention = any(e.mention for e in self._window.values())
        return len(self._window) + len(self._must_ack), has_mention

    def consume(self, message_id: str) -> ChatMessage | None:
        for key, entry in self._window.items():
            if entry.msg.id == message_id:
                self._remove(key)
                return entry.msg
        for key, msg in self._must_ack.items():
            if msg.id == message_id:
                del self._must_ack[key]
                return msg
        return None

    def recent(self, seconds: float) -> tuple[ChatMessage, ...]:
        now = self._clock.now()
        self._prune_history(now)
        return tuple(m for m in self._history if now - m.received <= seconds)

    def snapshot(self) -> list[tuple[ChatMessage, float]]:
        now = self._clock.now()
        self._expire(now)
        scored = [(e.msg, self._score(e, now)) for e in self._window.values()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    # --- scoring ---------------------------------------------------------------------------
    def score(self, message_id: str, now: float | None = None) -> float | None:
        """Current score of a windowed message (``None`` if it is not in the window)."""
        t = self._clock.now() if now is None else now
        for entry in self._window.values():
            if entry.msg.id == message_id:
                return self._score(entry, t)
        return None

    def _score(self, e: _Entry, now: float) -> float:
        m, w = e.msg, WEIGHTS
        age = max(0.0, now - min(m.ts, m.received))
        s = w["recency"] * math.exp(-age / self.cfg.recency_tau_s)
        s += w["mention"] * e.mention + w["question"] * e.question
        s += w["sub"] * m.user.is_sub + w["mod"] * m.user.is_mod + w["vip"] * m.user.is_vip
        s += w["first_msg"] * m.first_msg
        dup = self._groups.get(e.group, 1)
        if dup > 1:
            s -= w["dup"] * math.log(dup)
        last = self._last_picked.get(e.user_key)
        if last is not None and now - last < self.cfg.user_cooldown_s:
            s -= w["recent_pick"]
        return float(s)

    def _sample(self, scored: list[tuple[_Entry, float]], k: int) -> list[_Entry]:
        best_by_user: dict[str, tuple[_Entry, float]] = {}
        for e, s in scored:
            cur = best_by_user.get(e.user_key)
            if cur is None or s >= cur[1]:  # ties go to the newer message
                best_by_user[e.user_key] = (e, s)
        best_by_group: dict[str, tuple[_Entry, float]] = {}
        for e, s in best_by_user.values():
            cur = best_by_group.get(e.group)
            if cur is None or s > cur[1]:
                best_by_group[e.group] = (e, s)
        pool = list(best_by_group.values())
        picked: list[_Entry] = []
        temp = self.cfg.temperature
        for _ in range(max(0, min(k, len(pool)))):
            top = max(s for _, s in pool)
            weights = [math.exp((s - top) / temp) for _, s in pool]
            r = self._rng.random() * sum(weights)
            index = len(pool) - 1
            for i, wgt in enumerate(weights):
                r -= wgt
                if r < 0:
                    index = i
                    break
            picked.append(pool.pop(index)[0])
        return picked

    # --- bookkeeping -----------------------------------------------------------------------
    def _drop(self, reason: str) -> AddResult:
        self.drops[reason] += 1
        self.last_reason = reason
        return "dropped"

    def _remember(self, key: str) -> None:
        self._seen[key] = None
        while len(self._seen) > self.cfg.dedupe_lru:
            self._seen.popitem(last=False)

    def _record(self, m: ChatMessage, now: float) -> None:
        self._history.append(m)
        self._prune_history(now)

    def _prune_history(self, now: float) -> None:
        horizon = self.cfg.history_s
        while self._history and now - self._history[0].received > horizon:
            self._history.popleft()

    def _rate_limited(self, user_key: str, now: float) -> bool:
        times = self._user_times.setdefault(user_key, deque())
        span = self.cfg.user_burst_window_s
        while times and now - times[0] >= span:
            times.popleft()
        if len(times) >= self.cfg.user_burst:
            return True
        times.append(now)
        if len(self._user_times) > 4 * self.cfg.max_window:
            self._user_times = {
                u: t for u, t in self._user_times.items() if t and now - t[-1] < span
            }
        return False

    def _remove(self, key: str) -> None:
        entry = self._window.pop(key)
        self._groups[entry.group] -= 1
        if self._groups[entry.group] <= 0:
            del self._groups[entry.group]

    def _expire(self, now: float) -> None:
        horizon = self.cfg.horizon_s
        for key in [k for k, e in self._window.items() if now - e.msg.received > horizon]:
            self._remove(key)
        ttl = self.cfg.must_ack_ttl_s
        for key in [k for k, m in self._must_ack.items() if now - m.received > ttl]:
            del self._must_ack[key]
            self.drops["must_ack_expired"] += 1

    def _prune_picks(self, now: float) -> None:
        cooldown = self.cfg.user_cooldown_s
        self._last_picked = {u: t for u, t in self._last_picked.items() if now - t < cooldown}
