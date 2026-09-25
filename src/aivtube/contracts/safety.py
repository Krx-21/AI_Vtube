"""Safety protocols: filters, classifier and the gate (ARCHITECTURE.md §3.10, §7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from aivtube.contracts.types import ChatMessage

__all__ = [
    "Classifier",
    "Direction",
    "FilterContext",
    "FilterResult",
    "SafetyGate",
    "TextFilter",
    "Verdict",
]

Direction: TypeAlias = Literal["in", "out", "tool", "memory", "name", "game"]


class Verdict(StrEnum):
    PASS = "pass"
    MASK = "mask"
    REPLACE = "replace"
    DROP = "drop"
    BLOCK = "block"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class FilterContext:
    direction: Direction
    character: str
    platform: str | None = None
    user_id: str | None = None
    prev_tail: str = ""  # the last 40 emitted characters, for phrases split across chunks


@dataclass(frozen=True, slots=True)
class FilterResult:
    """``text`` is the (possibly masked or replaced) text to use when the verdict allows it."""

    verdict: Verdict
    text: str
    tier: str
    rule: str | None = None
    category: str | None = None
    score: float | None = None
    fail_closed: bool = False


@runtime_checkable
class TextFilter(Protocol):
    """Tier-0: synchronous and under 1 ms."""

    name: str

    def check(self, text: str, ctx: FilterContext) -> FilterResult: ...

    def reload(self) -> None: ...


@runtime_checkable
class Classifier(Protocol):
    """Tier-1 (M2)."""

    name: str

    async def score(self, texts: Sequence[str], direction: Direction) -> list[float]:
        """``P(harmful)`` per text."""
        ...


@runtime_checkable
class SafetyGate(Protocol):
    def check_input(self, msg: ChatMessage, *, character: str) -> tuple[FilterResult, str]:
        """Returns ``(verdict on the text, display-safe user name)``."""
        ...

    async def check_output(self, chunk: str, *, character: str, prev_tail: str) -> FilterResult: ...

    async def check_args(
        self,
        direction: Literal["tool", "memory", "game"],
        texts: Sequence[str],
        *,
        character: str,
    ) -> FilterResult: ...

    def strict_mode(self, character: str) -> bool: ...

    def reload(self) -> None: ...
