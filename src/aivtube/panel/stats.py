"""Latency waterfall rows, p50/p95 badges and trace alarms for the panel (§2.10, §4.6, §4.8, §5).

Traces come either from ``TurnTraceReady`` events (``stages`` = perf_counter times,
``stages_ms`` = offsets) or from ``OpsDb.recent_traces`` (``stages`` = offsets). Standard library
only; every function tolerates missing or malformed fields.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "DEFAULT_BUDGETS",
    "STAGE_ORDER",
    "Budget",
    "budget_for",
    "cache_alarm",
    "cache_stats",
    "opener_alarm",
    "percentile",
    "trace_summary",
    "waterfall_row",
]

STAGE_ORDER: Final = (
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


@dataclass(frozen=True, slots=True)
class Budget:
    """Time-to-first-audible targets in ms (§5). ``p95_ms=None`` means no p95 target."""

    p50_ms: float
    p95_ms: float | None = None


#: §5 targets for the 30B stream profile. Voice turns are measured from ``vad_end``; every
#: other turn kind from ``decision_start`` ("chat turn ... p50 <= 1.0 s").
DEFAULT_BUDGETS: Final[Mapping[str, Budget]] = {
    "voice": Budget(1700.0, 3000.0),
    "chat": Budget(1000.0, None),
}
CACHE_RATIO_MIN: Final = 0.85
CACHE_WINDOW: Final = 5
OPENER_WINDOW: Final = 20
OPENER_MAX_SHARE: Final = 0.30
OPENER_MIN_REPLIES: Final = 5


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100) of the finite values, or ``None``."""
    data = sorted(v for v in values if isinstance(v, int | float) and math.isfinite(v))
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * min(max(q, 0.0), 100.0) / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(data) - 1)
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _stages_ms(trace: Mapping[str, Any]) -> dict[str, float]:
    raw = trace.get("stages_ms")
    if not isinstance(raw, Mapping):
        raw = trace.get("stages")
    if not isinstance(raw, Mapping):
        return {}
    out = {str(k): v for k, v in ((k, _num(v)) for k, v in raw.items()) if v is not None}
    order = {name: i for i, name in enumerate(STAGE_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: (kv[1], order.get(kv[0], 99))))


def _tokens(trace: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(cached, total)`` prompt tokens: llama.cpp ``prompt_n`` counts only the tokens
    processed now and ``cache_n`` those reused from the KV cache (``llm.openai_stream``)."""
    prompt_n, cache_n = _num(trace.get("prompt_n")), _num(trace.get("cache_n"))
    if prompt_n is None or cache_n is None or prompt_n < 0 or cache_n < 0:
        return None
    total = prompt_n + cache_n
    return (cache_n, total) if total > 0 else None


def _cache_ratio(trace: Mapping[str, Any]) -> float | None:
    """``cache_n / (cache_n + prompt_n)`` for one turn, or ``None`` without token counts."""
    tokens = _tokens(trace)
    return None if tokens is None else round(tokens[0] / tokens[1], 3)


def budget_for(kind: str, budgets: Mapping[str, Budget]) -> Budget:
    if kind in budgets:
        return budgets[kind]
    return budgets.get("chat", DEFAULT_BUDGETS["chat"])


def waterfall_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    """One waterfall row: stage offsets in ms, ttfa, provider, TTS, cache ratio, opener."""
    stages = _stages_ms(trace)
    ttfa = _num(trace.get("ttfa_ms"))
    if ttfa is None and "first_audible" in stages:
        ttfa = stages["first_audible"]
    return {
        "turn_id": str(trace.get("turn_id") or ""),
        "kind": str(trace.get("kind") or ""),
        "character": trace.get("character"),
        "stages_ms": stages,
        "ttfa_ms": ttfa,
        "provider": trace.get("provider"),
        "tts_backend": trace.get("tts_backend"),
        "tts_identity": trace.get("tts_identity"),
        "outcome": trace.get("outcome"),
        "cache_ratio": _cache_ratio(trace),
        "opener": trace.get("opener") or "",
        "fallbacks": list(trace.get("fallbacks") or []),
    }


def cache_stats(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Prompt-cache hit ratio over the last 5 turns that report token counts (§4.8)."""
    recent = [t for t in (_tokens(t) for t in traces) if t is not None][-CACHE_WINDOW:]
    if not recent:
        return None
    cached = sum(c for c, _ in recent)
    total = sum(n for _, n in recent)
    return {"ratio": round(cached / total, 3), "turns": len(recent), "min": CACHE_RATIO_MIN}


def cache_alarm(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Alarm when ``cache_n / (cache_n + prompt_n)`` over the last 5 turns is below 0.85."""
    stats = cache_stats(traces)
    if stats is None or stats["turns"] < CACHE_WINDOW or stats["ratio"] >= CACHE_RATIO_MIN:
        return None
    ratio = float(stats["ratio"])
    return {
        "kind": "cache_ratio",
        "level": "warn",
        "value": ratio,
        "message": f"prompt cache ratio {ratio:.2f} < {CACHE_RATIO_MIN} over the last "
        f"{CACHE_WINDOW} turns",
        "message_th": f"อัตรา prompt cache {ratio:.2f} ต่ำกว่า {CACHE_RATIO_MIN} "
        f"ใน {CACHE_WINDOW} เทิร์นล่าสุด",
    }


def opener_alarm(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Alarm when one opener starts more than 30 % of the last 20 replies (§4.6)."""
    openers = [str(t.get("opener")) for t in traces if t.get("opener")][-OPENER_WINDOW:]
    if len(openers) < OPENER_MIN_REPLIES:
        return None
    opener, count = collections.Counter(openers).most_common(1)[0]
    share = count / len(openers)
    if share <= OPENER_MAX_SHARE:
        return None
    return {
        "kind": "opener",
        "level": "warn",
        "value": round(share, 3),
        "opener": opener,
        "message": f"{share:.0%} of the last {len(openers)} replies start with {opener!r}",
        "message_th": f"{share:.0%} ของ {len(openers)} คำตอบล่าสุดขึ้นต้นด้วย {opener!r}",
    }


def trace_summary(
    traces: Sequence[Mapping[str, Any]],
    *,
    budgets: Mapping[str, Budget] | None = None,
    n: int = 20,
) -> dict[str, Any]:
    """Rows for the last ``n`` traces plus p50/p95 badges per turn group and alarms."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    recent = [t for t in traces if isinstance(t, Mapping)][-n:] if n > 0 else []
    rows = [waterfall_row(t) for t in recent]
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        group = "voice" if row["kind"] == "voice" else "chat"
        if row["ttfa_ms"] is not None:
            groups[group].append(row["ttfa_ms"])
    badges: dict[str, dict[str, Any]] = {}
    for group in ("voice", "chat"):
        values = groups.get(group, [])
        budget = budget_for(group, budgets)
        p50, p95 = percentile(values, 50), percentile(values, 95)
        status = "none"
        if p50 is not None:
            over = p50 > budget.p50_ms or (
                budget.p95_ms is not None and p95 is not None and p95 > budget.p95_ms
            )
            status = "over" if over else "ok"
        badges[group] = {
            "n": len(values),
            "p50_ms": None if p50 is None else round(p50, 1),
            "p95_ms": None if p95 is None else round(p95, 1),
            "budget_p50_ms": budget.p50_ms,
            "budget_p95_ms": budget.p95_ms,
            "status": status,
        }
    alarms = [a for a in (cache_alarm(recent), opener_alarm(recent)) if a is not None]
    return {
        "rows": rows,
        "badges": badges,
        "alarms": alarms,
        "cache": cache_stats(recent),
        "stages": list(STAGE_ORDER),
    }
