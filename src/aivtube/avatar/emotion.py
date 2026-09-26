"""Emotion → avatar mapping: an idempotent expression state machine plus parameter baselines.

The character's ``[avatar.emotion_map]`` (§8) maps each emotion to VTS expression files
(``.exp3.json``), baseline values for the injected ``MouthSmile`` and ``Brows`` inputs, optional
extra parameters and an optional one-shot hotkey. Expressions are switched with
``ExpressionActivationRequest`` and an explicit ``active`` flag, never with ToggleExpression
hotkeys: firing a toggle twice turns the expression off (avatar brief, pitfalls).

``EmotionController`` is pure state; the sink sends the requests it computes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DEFAULT_BASELINE", "EmotionController", "EmotionSpec"]

log = logging.getLogger("aivtube.avatar.emotion")

NEUTRAL = "neutral"

#: Baseline for the injected inputs when an emotion does not set them.
DEFAULT_BASELINE: Mapping[str, float] = {"MouthSmile": 0.5, "Brows": 0.5}


def _unit(value: Any, default: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, v))


@dataclass(frozen=True, slots=True)
class EmotionSpec:
    """One ``emotion_map`` entry."""

    expressions: tuple[str, ...] = ()
    smile: float | None = None
    brows: float | None = None
    params: Mapping[str, float] = field(default_factory=dict)
    hotkey: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EmotionSpec:
        exprs = raw.get("expressions") or ()
        if isinstance(exprs, str):
            exprs = (exprs,)
        params_raw = raw.get("params") or {}
        params: dict[str, float] = {}
        if isinstance(params_raw, Mapping):
            for key, value in params_raw.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    params[str(key)] = float(value)
        hotkey = raw.get("hotkey")
        return cls(
            expressions=tuple(dict.fromkeys(str(e) for e in exprs if str(e))),
            smile=_unit(raw.get("smile"), None),
            brows=_unit(raw.get("brows"), None),
            params=params,
            hotkey=str(hotkey) if hotkey else None,
        )


class EmotionController:
    """Tracks which expressions should be active and the parameter baseline per emotion.

    ``apply(emotion)`` returns ``(activate, deactivate)``: the expression files whose state must
    change. Applying the current emotion again returns two empty lists (idempotent). Unknown
    emotions fall back to ``neutral``.
    """

    def __init__(self, emotion_map: Mapping[str, Mapping[str, Any]], *, default: str = NEUTRAL):
        specs: dict[str, EmotionSpec] = {}
        for name, raw in (emotion_map or {}).items():
            if isinstance(raw, EmotionSpec):
                specs[str(name).casefold()] = raw
            elif isinstance(raw, Mapping):
                specs[str(name).casefold()] = EmotionSpec.from_mapping(raw)
        self.default = default.casefold()
        specs.setdefault(self.default, EmotionSpec())
        self._specs = specs
        self._managed = frozenset(f for s in specs.values() for f in s.expressions)
        self.current = self.default
        self.active: frozenset[str] = frozenset()

    @property
    def known(self) -> frozenset[str]:
        return frozenset(self._specs)

    @property
    def managed(self) -> frozenset[str]:
        """Every expression file this map may switch (others are left alone)."""
        return self._managed

    def resolve(self, emotion: str | None) -> str:
        """The known emotion name for ``emotion`` (case-insensitive), else the default."""
        if not emotion:
            return self.default
        key = emotion.strip().casefold()
        if key in self._specs:
            return key
        log.debug("unknown emotion %r; using %r", emotion, self.default)
        return self.default

    def spec(self, emotion: str | None = None) -> EmotionSpec:
        return self._specs[self.resolve(emotion) if emotion is not None else self.current]

    def apply(self, emotion: str | None) -> tuple[list[str], list[str]]:
        """Switch to ``emotion``; returns ``(activate, deactivate)`` expression files."""
        name = self.resolve(emotion)
        want = frozenset(self._specs[name].expressions)
        activate = sorted(want - self.active)
        deactivate = sorted(self.active - want)
        self.current = name
        self.active = want
        return activate, deactivate

    def reconcile(self, present: Mapping[str, bool]) -> tuple[list[str], list[str]]:
        """Compare with the model's real state (``ExpressionStateRequest``: file → active).

        Returns ``(activate, deactivate)`` for managed files that exist in the model and are in
        the wrong state. Files the model lacks are skipped (they would fail with error 651).
        """
        activate = sorted(f for f in self.active if present.get(f) is False)
        deactivate = sorted(
            f for f in self._managed if present.get(f) is True and f not in self.active
        )
        return activate, deactivate

    def baseline(self, emotion: str | None = None) -> dict[str, float]:
        """Injected-parameter baseline: ``MouthSmile``, ``Brows`` and any extra params."""
        spec = self.spec(emotion)
        out = dict(DEFAULT_BASELINE)
        if spec.smile is not None:
            out["MouthSmile"] = spec.smile
        if spec.brows is not None:
            out["Brows"] = spec.brows
        out.update(spec.params)
        return out

    def hotkey(self, emotion: str | None = None) -> str | None:
        return self.spec(emotion).hotkey
