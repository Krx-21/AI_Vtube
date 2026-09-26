"""``TurnTraceRecorder`` (§2.10): per-turn latency stage marks and turn facts.

A finished trace is a plain dict shaped like the ``turn_trace`` row in ``ops.db``: ``stages``
maps stage → perf_counter time, ``stages_ms`` maps stage → ms after the trace origin, and
``ttfa_ms`` is time to first audible audio. The origin is ``vad_end`` for voice turns, else
``decision_start``, else ``stimulus_in`` (§5 measures chat turns from the decision start).
Tracing never raises: unknown turn ids are ignored.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable, Mapping
from typing import Any

from aivtube.contracts.events import TurnTraceReady
from aivtube.contracts.infra import Clock, EventBus
from aivtube.infra.clock import SystemClock

__all__ = ["STAGES", "TurnTraceRecorder"]

log = logging.getLogger("aivtube.trace")

STAGES: tuple[str, ...] = (
    "stimulus_in",
    "vad_end",
    "stt_final",
    "decision_start",
    "prompt_built",
    "llm_first_token",
    "first_chunk",
    "filter_done",
    "tts_first_audio",
    "first_audible",
    "last_audible",
    "done",
)
_LAST_WINS = frozenset({"last_audible", "done"})
_ORIGINS = ("vad_end", "decision_start", "stimulus_in")


class TurnTraceRecorder:
    """Collects marks per turn; ``finish`` publishes ``TurnTraceReady`` and returns the trace."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        bus: EventBus | None = None,
        on_finish: Callable[[Mapping[str, Any]], None] | None = None,
        max_open: int = 64,
        keep: int = 200,
    ) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._bus = bus
        self._on_finish = on_finish
        self._max_open = max_open
        self._open: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self._done: collections.deque[dict[str, Any]] = collections.deque(maxlen=keep)

    def begin(self, turn_id: str, kind: str, character: str) -> None:
        if turn_id in self._open:
            return
        while len(self._open) >= self._max_open:
            stale, _ = self._open.popitem(last=False)
            log.debug("dropping unfinished trace %s", stale)
        self._open[turn_id] = {
            "turn_id": turn_id,
            "kind": kind,
            "character": character,
            "stages": {},
            "fallbacks": [],
        }

    def mark(self, turn_id: str, stage: str, t: float | None = None) -> None:
        """Record ``stage`` at ``t`` (default now). First mark wins, except for
        ``last_audible``/``done``, where the latest wins."""
        trace = self._open.get(turn_id)
        if trace is None:
            return
        stages: dict[str, float] = trace["stages"]
        if stage in stages and stage not in _LAST_WINS:
            return
        stages[stage] = self._clock.now() if t is None else t

    def set(self, turn_id: str, **fields: Any) -> None:
        """Attach facts: provider, prompt_n, cache_n, tokens_out, tok_s, tts_backend,
        tts_identity, speculative, opener, outcome, session_id, … ``fallback=`` appends."""
        trace = self._open.get(turn_id)
        if trace is None:
            return
        fallback = fields.pop("fallback", None)
        if fallback is not None:
            trace["fallbacks"].append(fallback)
        trace.update(fields)

    def finish(self, turn_id: str) -> Mapping[str, Any]:
        trace = self._open.pop(turn_id, None)
        if trace is None:
            return {}
        stages: dict[str, float] = trace["stages"]
        stages.setdefault("done", self._clock.now())
        origin = next((stages[s] for s in _ORIGINS if s in stages), min(stages.values()))
        trace["stages_ms"] = {
            k: round((v - origin) * 1000.0, 1)
            for k, v in sorted(stages.items(), key=lambda kv: kv[1])
        }
        audible = stages.get("first_audible")
        trace["ttfa_ms"] = round((audible - origin) * 1000.0, 1) if audible is not None else None
        trace.setdefault("outcome", "ok")
        self._done.append(trace)
        if self._bus is not None:
            try:
                self._bus.publish(
                    TurnTraceReady(trace=trace, character=trace["character"], turn_id=turn_id)
                )
            except Exception:
                log.exception("publishing TurnTraceReady failed")
        if self._on_finish is not None:
            try:
                self._on_finish(trace)
            except Exception:
                log.exception("trace on_finish failed")
        return trace

    def get(self, turn_id: str) -> Mapping[str, Any] | None:
        """An unfinished trace (read-only view for the panel)."""
        return self._open.get(turn_id)

    def recent(self, n: int = 20) -> list[Mapping[str, Any]]:
        """The last ``n`` finished traces, oldest first."""
        done: list[Mapping[str, Any]] = list(self._done)
        return done[-n:] if n > 0 else []

    @staticmethod
    def opener(text: str, n: int = 6) -> str:
        """The reply's opener: its first ``n`` non-space characters (for repetition checks)."""
        return "".join(text.split())[:n]
