"""``LayeredSafetyGate``: the core's single safety entry point (ARCHITECTURE.md §3.10, §7).

Layers: the tier-0 ``TextFilter`` (sync, < 1 ms; its own base/private/platform/character
overlays are selected by ``FilterContext``), then the optional tier-1 ``Classifier`` (M2)
under a deadline. A tier-1 timeout or error fails open, except for fail-closed categories,
whose tier-0 hits are already final.

Side effects, all non-blocking: DROP/BLOCK publish ``Filtered``; DROP/BLOCK/REVIEW go to the
moderation audit (PII-masked text + sha256); a self-harm chat message raises a panel
``Alert``. Output BLOCKs feed auto-strict (§2.11): ``auto_strict_after`` blocks within
``auto_strict_window_s`` switch strict mode on for that character and publish an Alert;
``auto_freeze_after`` (0 = off) asks the app to FREEZE. Strict mode turns every REVIEW into a
stop and, with ``tier1 = "strict"``, enables tier-1.

Blocked text is never returned: DROP and BLOCK results carry ``text=""``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal

from aivtube.config.schema import SafetyConfig
from aivtube.contracts.events import Alert, Filtered
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.safety import (
    Classifier,
    Direction,
    FilterContext,
    FilterResult,
    SafetyGate,
    TextFilter,
    Verdict,
)
from aivtube.contracts.types import ChatMessage, ChatUser, Platform
from aivtube.infra.clock import deadline
from aivtube.safety.audit import ModerationAudit, sha256_text
from aivtube.safety.lists import FilterListError
from aivtube.safety.normalize import clean

__all__ = ["ANON_NAME", "LayeredSafetyGate", "check_name"]

log = logging.getLogger("aivtube.safety.gate")

#: Display name used for a user whose own name is flagged ("somebody").
ANON_NAME: Final = "ใครบางคน"
_STOP: Final = frozenset({Verdict.DROP, Verdict.BLOCK})
_CONTROL_RE: Final = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_NAME_CACHE: Final = 4096


class LayeredSafetyGate:
    """Implements ``SafetyGate`` over a tier-0 filter and an optional tier-1 classifier.

    ``cfg`` is the ``[safety]`` section (``SafetyConfig`` or a plain mapping of it).
    ``on_auto_strict`` / ``on_auto_freeze`` receive the character id; the app wires them to
    muting the chatters involved / to ``OpKind.FREEZE``. ``reload()`` blocks (it re-reads the
    list files): call it through ``asyncio.to_thread``.
    """

    def __init__(
        self,
        tier0: TextFilter,
        *,
        bus: EventBus,
        clock: Clock,
        cfg: SafetyConfig | Mapping[str, Any] | None = None,
        audit: ModerationAudit | None = None,
        classifier: Classifier | None = None,
        on_auto_strict: Callable[[str], None] | None = None,
        on_auto_freeze: Callable[[str], None] | None = None,
        name_max_chars: int = 40,
    ) -> None:
        if cfg is None:
            cfg = SafetyConfig()
        elif not isinstance(cfg, SafetyConfig):
            cfg = SafetyConfig.model_validate(dict(cfg))
        self.cfg: SafetyConfig = cfg
        self.tier0 = tier0
        self.classifier = classifier
        self.audit = audit
        self._bus = bus
        self._clock = clock
        self._on_auto_strict = on_auto_strict
        self._on_auto_freeze = on_auto_freeze
        self._name_max = max(1, name_max_chars)
        self._strict: dict[str, str] = {}  # character -> reason ("operator" | "auto")
        self._blocks: dict[str, deque[float]] = {}
        self._names: OrderedDict[tuple[str, str | None, str], str] = OrderedDict()
        self.tier1_timeouts = 0
        self.tier1_errors = 0

    # -- input -------------------------------------------------------------------------

    def check_input(self, msg: ChatMessage, *, character: str) -> tuple[FilterResult, str]:
        platform = msg.platform.value
        ctx = FilterContext("in", character, platform=platform, user_id=msg.user.id)
        res = self._escalate(self.tier0.check(msg.text, ctx), "in", character)
        name = self.display_name(
            msg.user.name, character=character, platform=platform, user_id=msg.user.id
        )
        if res.verdict in _STOP or res.verdict is Verdict.REVIEW:
            self._report(res, "in", character, msg.text, source=platform, author=msg.user.id)
            if res.category == "self_harm" and res.verdict in _STOP:
                self._bus.publish(
                    Alert(
                        level="warn",
                        message=(
                            f"ผู้ชม {name} ({platform}) พิมพ์เรื่องการทำร้ายตัวเอง "
                            "ข้อความถูกซ่อนแล้ว โปรดตรวจสอบแชท (self-harm)"
                        ),
                        character=character,
                    )
                )
        return res, name

    def display_name(
        self,
        name: str,
        *,
        character: str,
        platform: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """A name safe to show and to put in the prompt; flagged names become ``ANON_NAME``."""
        key = (character, platform, name)
        cached = self._names.get(key)
        if cached is not None:
            self._names.move_to_end(key)
            return cached
        safe = " ".join(_CONTROL_RE.sub(" ", clean(name)).split()).removeprefix("@")
        safe = safe[: self._name_max].strip()
        if safe:
            ctx = FilterContext("name", character, platform=platform, user_id=user_id)
            res = self.tier0.check(name, ctx)
            if res.verdict is not Verdict.PASS:
                if res.verdict in _STOP:
                    self._report(
                        res, "name", character, name, source=platform or "", author=user_id
                    )
                safe = ""
        out = safe or ANON_NAME
        self._names[key] = out
        if len(self._names) > _NAME_CACHE:
            self._names.popitem(last=False)
        return out

    # -- output ------------------------------------------------------------------------

    async def check_output(self, chunk: str, *, character: str, prev_tail: str) -> FilterResult:
        keep = self.cfg.prev_tail_chars
        tail = prev_tail[-keep:] if keep and prev_tail else ""
        ctx = FilterContext("out", character, prev_tail=tail)
        res = self._escalate(self.tier0.check(chunk, ctx), "out", character)
        if res.verdict not in _STOP and self._tier1_active(character):
            res = await self._tier1([tail + chunk], "out", character, res)
        if res.verdict in _STOP or res.verdict is Verdict.REVIEW:
            self._report(res, "out", character, tail + chunk, source="llm")
            if res.verdict in _STOP:
                self._count_block(character)
        return res

    async def check_args(
        self,
        direction: Literal["tool", "memory", "game"],
        texts: Sequence[str],
        *,
        character: str,
    ) -> FilterResult:
        results: list[FilterResult] = []
        for text in texts:
            res = self._escalate(
                self.tier0.check(text, FilterContext(direction, character)), direction, character
            )
            if res.verdict in _STOP:
                self._report(res, direction, character, text, source=direction)
                return res
            results.append(res)
        joined = "\n".join(r.text for r in results)
        review = next((r for r in results if r.verdict is Verdict.REVIEW), None)
        replaced = next((r for r in results if r.verdict is not Verdict.PASS), None)
        if review is not None:
            combined = dataclasses.replace(review, text=joined)
        elif replaced is not None:
            combined = dataclasses.replace(replaced, text=joined)
        else:
            combined = FilterResult(Verdict.PASS, joined, "tier0")
        if texts and self._tier1_active(character):
            combined = await self._tier1(list(texts), direction, character, combined)
        if combined.verdict in _STOP or combined.verdict is Verdict.REVIEW:
            self._report(combined, direction, character, "\n".join(texts), source=direction)
        return combined

    # -- strict mode -------------------------------------------------------------------

    def strict_mode(self, character: str) -> bool:
        return character in self._strict

    def set_strict(self, character: str, on: bool, *, reason: str = "operator") -> None:
        """Operator control (``OpKind.STRICT``). Turning it off also resets the block count."""
        if on:
            self._strict[character] = reason
        else:
            self._strict.pop(character, None)
            self._blocks.pop(character, None)
        log.info("strict mode %s for %s (%s)", "on" if on else "off", character, reason)

    def reload(self) -> None:
        """Re-read the filter lists (``FILTER_RELOAD``; blocking, run it in a thread)."""
        self.tier0.reload()
        self._names.clear()

    def reload_if_changed(self) -> bool:
        """Reload when a list file changed (tier-0 filters that can tell); blocking."""
        probe = getattr(self.tier0, "reload_if_changed", None)
        if probe is None or not probe():
            return False
        self._names.clear()
        return True

    async def watch_lists(self, interval_s: float = 5.0) -> None:
        """Hot reload: poll the list files every ``interval_s`` and reload on change.

        Long-running; start it with ``TaskSupervisor.spawn``. File access runs in a worker
        thread under a deadline; a bad file is logged and the previous lists stay active.
        """
        while True:
            await self._clock.sleep(interval_s)
            try:
                async with deadline(30.0, what="filter list reload", clock=self._clock):
                    changed = await asyncio.to_thread(self.reload_if_changed)
            except (FilterListError, TimeoutError):
                log.warning("filter hot reload failed; previous lists stay active", exc_info=True)
                continue
            if changed:
                log.info("filter lists reloaded")
                self._bus.publish(
                    Alert(level="info", message="โหลดรายการคำกรองใหม่แล้ว (filters reloaded)")
                )

    def status(self) -> dict[str, Any]:
        """For the panel: strict characters and recent output blocks per character."""
        now = self._clock.now()
        window = self.cfg.auto_strict_window_s
        return {
            "strict": dict(self._strict),
            "recent_blocks": {
                c: sum(1 for t in q if now - t <= window) for c, q in self._blocks.items()
            },
            "tier1_timeouts": self.tier1_timeouts,
            "tier1_errors": self.tier1_errors,
        }

    # -- internals ---------------------------------------------------------------------

    def _escalate(self, res: FilterResult, direction: Direction, character: str) -> FilterResult:
        """Strict mode: REVIEW becomes a stop (DROP for chat/names, BLOCK otherwise)."""
        if res.verdict is not Verdict.REVIEW or not self.strict_mode(character):
            return res
        stop = Verdict.DROP if direction in ("in", "name") else Verdict.BLOCK
        return dataclasses.replace(res, verdict=stop, text="")

    def _tier1_active(self, character: str) -> bool:
        if self.classifier is None:
            return False
        mode = self.cfg.tier1
        return mode == "on" or (mode == "strict" and self.strict_mode(character))

    async def _tier1(
        self, texts: list[str], direction: Direction, character: str, base: FilterResult
    ) -> FilterResult:
        assert self.classifier is not None
        timeout_s = self.cfg.tier1_timeout_ms / 1000.0
        try:
            async with deadline(timeout_s, what="safety tier-1", clock=self._clock):
                scores = await self.classifier.score(texts, direction)
        except Exception as exc:  # DeadlineExceeded or a classifier failure
            if isinstance(exc, TimeoutError):
                self.tier1_timeouts += 1
                log.warning("tier-1 classifier timed out after %.0f ms", timeout_s * 1000)
            else:
                self.tier1_errors += 1
                log.warning("tier-1 classifier failed", exc_info=True)
            if base.fail_closed:
                return FilterResult(
                    Verdict.BLOCK,
                    "",
                    "tier1",
                    rule="timeout",
                    category=base.category,
                    fail_closed=True,
                )
            return base  # fail open
        score = max(scores, default=0.0)
        strict = self.strict_mode(character)
        stop = Verdict.DROP if direction in ("in", "name") else Verdict.BLOCK
        if score >= self.cfg.tier1_block or (strict and score >= self.cfg.tier1_review):
            return FilterResult(
                stop,
                "",
                "tier1",
                rule="classifier",
                category=base.category or "classifier",
                score=score,
                fail_closed=base.fail_closed,
            )
        if score >= self.cfg.tier1_review:
            if base.verdict is Verdict.REVIEW:
                return dataclasses.replace(base, score=score)
            return FilterResult(
                Verdict.REVIEW,
                base.text,
                "tier1",
                rule="classifier",
                category="classifier",
                score=score,
            )
        return dataclasses.replace(base, score=score)

    def _report(
        self,
        res: FilterResult,
        direction: str,
        character: str,
        text: str,
        *,
        source: str,
        author: str | None = None,
    ) -> None:
        if res.verdict in _STOP:
            self._bus.publish(
                Filtered(
                    character=character,
                    direction=direction,
                    tier=res.tier,
                    category=res.category,
                    rule=res.rule,
                    ref=sha256_text(text)[:16],
                )
            )
        if self.audit is not None:
            try:
                self.audit.record(
                    character=character,
                    direction=direction,
                    source=source,
                    result=res,
                    text=text,
                    author=author,
                )
            except Exception:  # the audit must never break moderation itself
                log.exception("moderation audit failed")

    def _count_block(self, character: str) -> None:
        now = self._clock.now()
        window = self.cfg.auto_strict_window_s
        q = self._blocks.setdefault(character, deque())
        q.append(now)
        while q and now - q[0] > window:
            q.popleft()
        n = len(q)
        after = self.cfg.auto_strict_after
        if after and n >= after and character not in self._strict:
            self._strict[character] = "auto"
            minutes = window / 60.0
            self._bus.publish(
                Alert(
                    level="warn",
                    message=(
                        f"เปิดโหมดเข้มงวดอัตโนมัติ: ถูกกรอง {n} ครั้งใน {minutes:g} นาที (auto-strict on)"
                    ),
                    character=character,
                )
            )
            log.warning("auto-strict on for %s after %d blocks", character, n)
            self._callback(self._on_auto_strict, character)
        freeze = self.cfg.auto_freeze_after
        if freeze and n >= freeze:
            q.clear()
            self._bus.publish(
                Alert(
                    level="error",
                    message=f"หยุดอัตโนมัติ: ถูกกรอง {n} ครั้งใน {window / 60.0:g} นาที (auto-freeze)",
                    character=character,
                )
            )
            log.error("auto-freeze requested for %s after %d blocks", character, n)
            self._callback(self._on_auto_freeze, character)

    @staticmethod
    def _callback(fn: Callable[[str], None] | None, character: str) -> None:
        if fn is None:
            return
        try:
            fn(character)
        except Exception:
            log.exception("safety callback failed")


def check_name(gate: SafetyGate, name: str, character: str) -> str:
    """The display-safe version of ``name`` (``ANON_NAME`` when it is flagged)."""
    if isinstance(gate, LayeredSafetyGate):
        return gate.display_name(name, character=character)
    user = ChatUser(Platform.CONSOLE, "", name)
    probe = ChatMessage(Platform.CONSOLE, "", user, "", 0.0, 0.0)
    return gate.check_input(probe, character=character)[1]
