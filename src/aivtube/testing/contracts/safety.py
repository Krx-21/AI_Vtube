"""Contract suites for ``TextFilter`` and ``SafetyGate`` (§3.10, §7)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence

from aivtube.contracts.safety import FilterContext, FilterResult, SafetyGate, TextFilter, Verdict
from aivtube.contracts.types import ChatMessage, ChatUser, Platform
from aivtube.testing.contracts._base import AsyncCase, SyncCase, _Cases, check, maybe_await

__all__ = ["safety_gate_suite", "text_filter_suite"]

BENIGN: tuple[str, ...] = ("สวัสดีค่ะ วันนี้อากาศดีนะ", "ไพลินชอบกินข้าวมันไก่")
_STOPPING = (Verdict.BLOCK, Verdict.DROP)


def _split(phrase: str) -> tuple[str, str]:
    k = max(1, len(phrase) // 2)
    return phrase[:k], phrase[k:]


def text_filter_suite(
    factory: Callable[[], TextFilter],
    *,
    blocked: str,
    benign: Sequence[str] = BENIGN,
    max_avg_ms: float = 5.0,
) -> list[SyncCase]:
    """``blocked`` must be a phrase the filter blocks in every direction."""
    cases = _Cases("text_filter")

    def ctx(direction: str = "out", prev_tail: str = "") -> FilterContext:
        return FilterContext(direction, "pailin", prev_tail=prev_tail)  # type: ignore[arg-type]

    @cases
    def benign_text_passes_unchanged() -> None:
        f = factory()
        check(isinstance(f.name, str) and f.name, "name")
        for text in benign:
            r = f.check(text, ctx())
            check(isinstance(r, FilterResult), "check() must return FilterResult")
            check(r.verdict is Verdict.PASS and r.text == text, f"{text!r} -> {r!r}")

    @cases
    def blocked_output_never_returns_the_text() -> None:
        r = factory().check(f"ไพลินพูดว่า {blocked} นะ", ctx("out"))
        check(r.verdict is Verdict.BLOCK, f"output verdict {r.verdict}")
        check(blocked not in r.text, "a BLOCK result must not carry the blocked text")

    @cases
    def blocked_input_is_stopped() -> None:
        r = factory().check(f"{blocked} 555", ctx("in"))
        check(r.verdict in _STOPPING, f"input verdict {r.verdict}")

    @cases
    def phrase_split_across_chunks_is_caught() -> None:
        f = factory()
        head, tail = _split(blocked)
        f.check(f"ไพลินพูดว่า {head}", ctx("out"))
        r = f.check(tail, ctx("out", prev_tail=f"ไพลินพูดว่า {head}"[-40:]))
        check(r.verdict is Verdict.BLOCK, f"split phrase passed ({r.verdict})")

    @cases
    def tool_and_memory_arguments_are_checked() -> None:
        f = factory()
        for direction in ("tool", "memory"):
            r = f.check(f"จำไว้ว่า {blocked}", ctx(direction))
            check(r.verdict is Verdict.BLOCK, f"{direction} verdict {r.verdict}")

    @cases
    def reload_keeps_working() -> None:
        f = factory()
        f.reload()
        check(
            f.check(blocked, ctx()).verdict is Verdict.BLOCK, "blocked phrase passed after reload"
        )

    @cases
    def check_is_fast() -> None:
        f = factory()
        f.check(benign[0], ctx())  # warm-up (tokeniser dictionaries)
        t0 = time.perf_counter()
        for _ in range(50):
            f.check(benign[0], ctx())
        avg_ms = (time.perf_counter() - t0) * 1000.0 / 50
        check(avg_ms < max_avg_ms, f"check() averaged {avg_ms:.2f} ms")

    return cases.items


def safety_gate_suite(
    factory: Callable[[], SafetyGate | Awaitable[SafetyGate]],
    *,
    blocked: str,
    benign: Sequence[str] = BENIGN,
    character: str = "pailin",
) -> list[AsyncCase]:
    cases = _Cases("safety_gate")

    def chat(text: str, name: str = "viewer") -> ChatMessage:
        user = ChatUser(Platform.TWITCH, "u-1", name)
        return ChatMessage(Platform.TWITCH, "m-1", user, text, 1.0, 1.0)

    @cases
    async def benign_input_passes_with_a_display_name() -> None:
        gate = await maybe_await(factory())
        res, name = gate.check_input(chat(benign[0], "มะลิ"), character=character)
        check(res.verdict not in _STOPPING, f"benign input {res.verdict}")
        check(isinstance(name, str) and name, "display-safe name")

    @cases
    async def blocked_input_is_stopped() -> None:
        gate = await maybe_await(factory())
        res, _ = gate.check_input(chat(f"{blocked} 555"), character=character)
        check(res.verdict in _STOPPING, f"blocked input {res.verdict}")

    @cases
    async def output_uses_the_previous_tail() -> None:
        gate = await maybe_await(factory())
        ok = await gate.check_output(benign[0], character=character, prev_tail="")
        check(ok.verdict is Verdict.PASS and ok.text, f"benign output {ok.verdict}")
        head, tail = _split(blocked)
        r = await gate.check_output(tail, character=character, prev_tail=f"ไพลินว่า {head}")
        check(r.verdict is Verdict.BLOCK, f"split phrase passed ({r.verdict})")
        check(blocked not in r.text, "BLOCK result carried the blocked text")

    @cases
    async def arguments_are_filtered() -> None:
        gate = await maybe_await(factory())
        ok = await gate.check_args("tool", list(benign), character=character)
        check(ok.verdict not in _STOPPING, f"benign args {ok.verdict}")
        bad = await gate.check_args("memory", ["ปกติ", f"จำไว้ {blocked}"], character=character)
        check(bad.verdict is Verdict.BLOCK, f"blocked memory args {bad.verdict}")

    @cases
    async def strict_mode_and_reload() -> None:
        gate = await maybe_await(factory())
        check(isinstance(gate.strict_mode(character), bool), "strict_mode() must return bool")
        gate.reload()

    return cases.items
