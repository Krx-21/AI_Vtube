"""Safety fakes: a blocklist ``TextFilter``, a keyword ``Classifier`` and a ``SafetyGate`` (§3.10).

Matching is deliberately simple but honours the contract that matters to callers: output checks
look at ``prev_tail + chunk``, so a phrase split across two chunks is still caught. Text is
compared after NFKC, zero-width stripping and casefolding, and also in a despaced form.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from aivtube.contracts.events import Filtered
from aivtube.contracts.infra import EventBus
from aivtube.contracts.safety import Direction, FilterContext, FilterResult, Verdict
from aivtube.contracts.types import ChatMessage

__all__ = ["ANON_NAME", "FakeClassifier", "FakeSafetyGate", "FakeTextFilter", "despace", "norm"]

ANON_NAME = "ใครบางคน"
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
_DESPACE = dict.fromkeys(map(ord, " \t\n.-_*"))


def norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()


def despace(text: str) -> str:
    return norm(text).translate(_DESPACE)


class FakeTextFilter:
    """Tier-0 ``TextFilter`` stand-in: blocklist phrases plus optional replace rules.

    A hit is ``DROP`` for ``direction="in"`` and ``BLOCK`` otherwise (never returning the
    blocked text). ``replace`` maps phrases to spoken replacements (``REPLACE``).
    """

    def __init__(
        self,
        blocklist: Iterable[str] = ("คำต้องห้าม", "badword"),
        *,
        replace: Mapping[str, str] | None = None,
        name: str = "fake-tier0",
        category: str = "fake",
    ) -> None:
        self.name = name
        self.category = category
        self.blocklist = [p for p in blocklist if p.strip()]
        self.replace = dict(replace or {})
        self.checks: list[tuple[str, FilterContext]] = []
        self.reloads = 0

    def find(self, text: str) -> str | None:
        n, d = norm(text), despace(text)
        for phrase in self.blocklist:
            if norm(phrase) in n or (despace(phrase) and despace(phrase) in d):
                return phrase
        return None

    def check(self, text: str, ctx: FilterContext) -> FilterResult:
        self.checks.append((text, ctx))
        hit = self.find(ctx.prev_tail + text)
        if hit is not None:
            verdict = Verdict.DROP if ctx.direction == "in" else Verdict.BLOCK
            return FilterResult(verdict, "", "tier0", rule=hit, category=self.category)
        out = text
        for src, dst in self.replace.items():
            out = out.replace(src, dst)
        if out != text:
            return FilterResult(Verdict.REPLACE, out, "tier0", rule="replace")
        return FilterResult(Verdict.PASS, text, "tier0")

    def reload(self) -> None:
        self.reloads += 1


class FakeClassifier:
    """Tier-1 ``Classifier`` stand-in: the score is the highest keyword score found in a text."""

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        default: float = 0.0,
        delay_s: float = 0.0,
        name: str = "fake-tier1",
    ) -> None:
        self.name = name
        self.scores = {norm(k): v for k, v in (scores or {}).items()}
        self.default = default
        self.delay_s = delay_s
        self.calls: list[tuple[tuple[str, ...], Direction]] = []

    async def score(self, texts: Sequence[str], direction: Direction) -> list[float]:
        self.calls.append((tuple(texts), direction))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        out = []
        for t in texts:
            n = norm(t)
            out.append(max([v for k, v in self.scores.items() if k in n], default=self.default))
        return out


class FakeSafetyGate:
    """Pass-through ``SafetyGate`` with a configurable blocklist.

    Everything passes unchanged unless it contains a blocklisted phrase (output checks include
    ``prev_tail``). Blocked user names become ``ANON_NAME``. With a ``bus`` it publishes
    ``Filtered`` events. ``strict`` lists characters in strict mode.
    """

    def __init__(
        self,
        blocklist: Iterable[str] = (),
        *,
        strict: Iterable[str] = (),
        bus: EventBus | None = None,
    ) -> None:
        self.filter = FakeTextFilter(blocklist)
        self.strict = set(strict)
        self.bus = bus
        self.inputs: list[tuple[ChatMessage, str]] = []
        self.outputs: list[tuple[str, str, str]] = []
        self.arg_checks: list[tuple[str, tuple[str, ...], str]] = []
        self.blocked: list[FilterResult] = []
        self.reloads = 0

    @property
    def blocklist(self) -> list[str]:
        return self.filter.blocklist

    def block(self, *phrases: str) -> None:
        self.filter.blocklist.extend(phrases)

    def set_strict(self, character: str, on: bool = True) -> None:
        (self.strict.add if on else self.strict.discard)(character)

    def check_input(self, msg: ChatMessage, *, character: str) -> tuple[FilterResult, str]:
        self.inputs.append((msg, character))
        ctx = FilterContext("in", character, msg.platform.value, msg.user.id)
        res = self.filter.check(msg.text, ctx)
        name_hit = self.filter.find(msg.user.name)
        self._note(res, "in", character)
        return res, (ANON_NAME if name_hit else msg.user.name)

    async def check_output(self, chunk: str, *, character: str, prev_tail: str) -> FilterResult:
        self.outputs.append((chunk, character, prev_tail))
        res = self.filter.check(chunk, FilterContext("out", character, prev_tail=prev_tail))
        self._note(res, "out", character)
        return res

    async def check_args(
        self,
        direction: Literal["tool", "memory", "game"],
        texts: Sequence[str],
        *,
        character: str,
    ) -> FilterResult:
        self.arg_checks.append((direction, tuple(texts), character))
        for t in texts:
            res = self.filter.check(t, FilterContext(direction, character))
            if res.verdict is Verdict.BLOCK:
                self._note(res, direction, character)
                return res
        return FilterResult(Verdict.PASS, "\n".join(texts), "tier0")

    def strict_mode(self, character: str) -> bool:
        return character in self.strict

    def reload(self) -> None:
        self.reloads += 1
        self.filter.reload()

    def _note(self, res: FilterResult, direction: str, character: str) -> None:
        if res.verdict not in (Verdict.BLOCK, Verdict.DROP):
            return
        self.blocked.append(res)
        if self.bus is not None:
            self.bus.publish(
                Filtered(
                    character=character,
                    direction=direction,
                    tier=res.tier,
                    category=res.category,
                    rule=res.rule,
                    ref=None,
                )
            )
