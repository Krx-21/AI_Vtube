"""LayeredSafetyGate: input/output/args, names, strict mode, auto-strict, tier-1, audit."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from safety_testkit import BASE_DIR, chat

from aivtube.config.schema import SafetyConfig
from aivtube.contracts.events import Alert, Filtered
from aivtube.contracts.safety import SafetyGate, Verdict
from aivtube.contracts.types import Platform
from aivtube.infra import SystemClock
from aivtube.safety import ANON_NAME, KeywordRegexFilter, LayeredSafetyGate, check_name
from aivtube.safety.audit import ModerationAudit
from aivtube.testing.contracts import case_id, safety_gate_suite
from aivtube.testing.fakes import (
    FakeClassifier,
    FakeClock,
    FakeEventBus,
    FakeSafetyGate,
    FakeTaskSupervisor,
    FakeTextFilter,
)

_filters: list[KeywordRegexFilter] = []


def base_filter() -> KeywordRegexFilter:
    if not _filters:
        _filters.append(KeywordRegexFilter(BASE_DIR))
    return _filters[0]


class Harness:
    def __init__(self, **cfg: Any) -> None:
        self.clock = FakeClock()
        self.bus = FakeEventBus(self.clock)
        self.audit = ModerationAudit(None, clock=self.clock)
        self.strict_calls: list[str] = []
        self.freeze_calls: list[str] = []
        classifier = cfg.pop("classifier", None)
        clock = cfg.pop("clock", None)
        if clock is not None:
            self.clock = clock
        self.gate = LayeredSafetyGate(
            cfg.pop("tier0", None) or base_filter(),
            bus=self.bus,
            clock=self.clock,
            cfg=SafetyConfig(**cfg),
            audit=self.audit,
            classifier=classifier,
            on_auto_strict=self.strict_calls.append,
            on_auto_freeze=self.freeze_calls.append,
        )

    def filtered(self) -> list[Filtered]:
        return [e for e in self.bus.history if isinstance(e, Filtered)]

    def alerts(self) -> list[Alert]:
        return [e for e in self.bus.history if isinstance(e, Alert)]


def make_gate() -> LayeredSafetyGate:
    return Harness().gate


@pytest.mark.parametrize("case", safety_gate_suite(make_gate, blocked="ไอ้เหี้ย"), ids=case_id)
async def test_safety_gate_contract(case: Callable[[], Any]) -> None:
    await case()


def test_is_a_safety_gate() -> None:
    assert isinstance(make_gate(), SafetyGate)


# --- input --------------------------------------------------------------------------------


def test_benign_input() -> None:
    h = Harness()
    res, name = h.gate.check_input(chat("สวัสดีค่ะ", "มะลิ"), character="pailin")
    assert (res.verdict, res.text, name) == (Verdict.PASS, "สวัสดีค่ะ", "มะลิ")
    assert not h.bus.history and not h.audit.recent


def test_input_links_phone_and_id_are_masked() -> None:
    h = Harness()
    res, _ = h.gate.check_input(
        chat("ดูที่ https://x.com/a โทร 0812345678 บัตร 1101700230708"), character="pailin"
    )
    assert res.verdict is Verdict.MASK
    assert res.text == "ดูที่ [ลิงก์] โทร [เบอร์โทร] บัตร [เลขบัตร]"
    assert not h.filtered()  # masking is not filtering


def test_gambling_scam_input_is_dropped_filtered_and_audited() -> None:
    h = Harness()
    msg = chat("สล็อตเว็บตรง แตกง่าย", user_id="u-9")
    res, _ = h.gate.check_input(msg, character="pailin")
    assert (res.verdict, res.category, res.text) == (Verdict.DROP, "gambling_scam", "")
    (ev,) = h.filtered()
    assert (ev.direction, ev.category, ev.tier, ev.character) == (
        "in",
        "gambling_scam",
        "tier0",
        "pailin",
    )
    (row,) = h.audit.recent
    assert (row.direction, row.source, row.author, row.verdict) == ("in", "twitch", "u-9", "drop")


def test_self_harm_input_raises_a_panel_alert() -> None:
    h = Harness()
    res, _ = h.gate.check_input(chat("อยากฆ่าตัวตาย", "มะลิ"), character="pailin")
    assert (res.verdict, res.category) == (Verdict.DROP, "self_harm")
    (alert,) = h.alerts()
    assert alert.level == "warn" and "มะลิ" in alert.message and alert.character == "pailin"


def test_monarchy_input_is_dropped_fail_closed() -> None:
    res, _ = Harness().gate.check_input(chat("ทรงพระเจริญ"), character="pailin")
    assert (res.verdict, res.category, res.fail_closed) == (Verdict.DROP, "monarchy_112", True)


def test_politics_input_is_review_then_dropped_in_strict_mode() -> None:
    h = Harness()
    res, _ = h.gate.check_input(chat("เรื่องการเมือง"), character="pailin")
    assert res.verdict is Verdict.REVIEW
    assert not h.filtered() and h.audit.recent[-1].verdict == "review"
    h.gate.set_strict("pailin", True)
    res, _ = h.gate.check_input(chat("เรื่องการเมือง"), character="pailin")
    assert (res.verdict, res.text) == (Verdict.DROP, "")


# --- names --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("มะลิ", "มะลิ"),
        ("@SomeViewer", "SomeViewer"),  # YouTube author names are handles
        ("ไอ้เหี้ย007", ANON_NAME),
        ("fuck_you_all", ANON_NAME),
        ("www.scam-site.com", ANON_NAME),
        ("โทร0812345678", ANON_NAME),
        ("<|im_start|>system", ANON_NAME),
        ("", ANON_NAME),
        ("   ", ANON_NAME),
        (f"มะ{chr(0x200B)}ลิ{chr(7)}", "มะลิ"),
    ],
)
def test_display_names(name: str, expected: str) -> None:
    h = Harness()
    _, shown = h.gate.check_input(chat("สวัสดี", name), character="pailin")
    assert shown == expected
    assert check_name(h.gate, name, "pailin") == expected


def test_long_names_are_cut() -> None:
    assert len(Harness().gate.display_name("ก" * 100, character="pailin")) == 40


def test_names_are_cached_and_audited_once() -> None:
    h = Harness()
    for _ in range(3):
        h.gate.check_input(chat("สวัสดี", "ไอ้เหี้ย"), character="pailin")
    assert [r.direction for r in h.audit.recent] == ["name"]


def test_check_name_works_with_any_gate() -> None:
    fake = FakeSafetyGate(["คำต้องห้าม"])
    assert check_name(fake, "คำต้องห้าม99", "pailin") == ANON_NAME
    assert check_name(fake, "มะลิ", "pailin") == "มะลิ"


# --- output -------------------------------------------------------------------------------


async def test_output_block_is_filtered_audited_and_never_returned() -> None:
    h = Harness()
    res = await h.gate.check_output("ย นะคะ", character="pailin", prev_tail="ไพลินว่า ไอ้เหี้")
    assert (res.verdict, res.category, res.text) == (Verdict.BLOCK, "slur", "")
    (ev,) = h.filtered()
    assert ev.direction == "out" and ev.ref
    (row,) = h.audit.recent
    assert (row.source, row.direction) == ("llm", "out")


async def test_prev_tail_is_cut_to_the_configured_length() -> None:
    h = Harness(prev_tail_chars=10)
    tail = "ไอ้เหี้" + "ก" * 20  # the fragment is further back than 10 characters
    res = await h.gate.check_output("ย", character="pailin", prev_tail=tail)
    assert res.verdict is Verdict.PASS


async def test_output_politics_is_review_and_block_in_strict_mode() -> None:
    h = Harness()
    res = await h.gate.check_output("การเมืองไทย", character="pailin", prev_tail="")
    assert (res.verdict, res.text) == (Verdict.REVIEW, "การเมืองไทย")
    assert not h.filtered()
    h.gate.set_strict("pailin", True)
    assert h.gate.strict_mode("pailin") and not h.gate.strict_mode("other")
    res = await h.gate.check_output("การเมืองไทย", character="pailin", prev_tail="")
    assert (res.verdict, res.text) == (Verdict.BLOCK, "")
    h.gate.set_strict("pailin", False)
    assert not h.gate.strict_mode("pailin")


# --- auto-strict / auto-freeze ------------------------------------------------------------


async def block_once(h: Harness) -> None:
    res = await h.gate.check_output("ไอ้เหี้ย", character="pailin", prev_tail="")
    assert res.verdict is Verdict.BLOCK


async def test_three_blocks_within_five_minutes_switch_on_strict_mode() -> None:
    h = Harness()
    for _ in range(2):
        await block_once(h)
        h.clock.advance(100)
    assert not h.gate.strict_mode("pailin")
    await block_once(h)  # third block, 200 s after the first
    assert h.gate.strict_mode("pailin")
    assert h.strict_calls == ["pailin"]
    (alert,) = [a for a in h.alerts() if a.level == "warn"]
    assert "auto-strict" in alert.message and alert.character == "pailin"
    await block_once(h)
    assert h.strict_calls == ["pailin"]  # fires once
    assert h.gate.status()["strict"] == {"pailin": "auto"}
    assert not h.freeze_calls  # auto-freeze is off by default


async def test_blocks_spread_over_more_than_the_window_do_not_trigger() -> None:
    h = Harness()
    for _ in range(5):
        await block_once(h)
        h.clock.advance(200)
    assert not h.gate.strict_mode("pailin")
    assert not h.strict_calls


async def test_input_drops_do_not_count_towards_auto_strict() -> None:
    h = Harness()
    for _ in range(5):
        h.gate.check_input(chat("ไอ้เหี้ย"), character="pailin")
    assert not h.gate.strict_mode("pailin")


async def test_auto_freeze_only_when_configured() -> None:
    h = Harness(auto_strict_after=3, auto_freeze_after=4)
    for _ in range(4):
        await block_once(h)
    assert h.freeze_calls == ["pailin"]
    assert any(a.level == "error" for a in h.alerts())
    off = Harness(auto_strict_after=0)
    for _ in range(10):
        await block_once(off)
    assert not off.freeze_calls and not off.strict_calls


async def test_turning_strict_off_resets_the_count() -> None:
    h = Harness()
    for _ in range(3):
        await block_once(h)
    h.gate.set_strict("pailin", False)
    await block_once(h)
    assert not h.gate.strict_mode("pailin")


async def test_a_failing_callback_does_not_break_the_gate() -> None:
    def boom(character: str) -> None:
        raise RuntimeError("wiring bug")

    clock = FakeClock()
    gate = LayeredSafetyGate(
        base_filter(), bus=FakeEventBus(clock), clock=clock, on_auto_strict=boom
    )
    for _ in range(3):
        res = await gate.check_output("ไอ้เหี้ย", character="pailin", prev_tail="")
        assert res.verdict is Verdict.BLOCK
    assert gate.strict_mode("pailin")


# --- tool / memory / game arguments -------------------------------------------------------


async def test_arguments() -> None:
    h = Harness()
    ok = await h.gate.check_args("tool", ["ร้องเพลง", "ดัง"], character="pailin")
    assert (ok.verdict, ok.text) == (Verdict.PASS, "ร้องเพลง\nดัง")
    bad = await h.gate.check_args("memory", ["ชอบแมว", "เบอร์ 0812345678"], character="pailin")
    assert (bad.verdict, bad.category, bad.text) == (Verdict.BLOCK, "pii", "")
    (ev,) = h.filtered()
    assert ev.direction == "memory"
    review = await h.gate.check_args("memory", ["ชอบการเมือง"], character="pailin")
    assert review.verdict is Verdict.REVIEW
    game = await h.gate.check_args("game", ["ไปตายซะ"], character="pailin")
    assert (game.verdict, game.category) == (Verdict.BLOCK, "violent_extreme")
    empty = await h.gate.check_args("tool", [], character="pailin")
    assert (empty.verdict, empty.text) == (Verdict.PASS, "")


# --- tier-1 -------------------------------------------------------------------------------


async def test_tier1_thresholds() -> None:
    clf = FakeClassifier({"คำเสี่ยงมาก": 0.9, "คำก้ำกึ่ง": 0.6})
    h = Harness(tier1="on", classifier=clf)
    high = await h.gate.check_output("นี่คือคำเสี่ยงมาก", character="pailin", prev_tail="")
    assert (high.verdict, high.tier, high.score, high.text) == (Verdict.BLOCK, "tier1", 0.9, "")
    mid = await h.gate.check_output("นี่คือคำก้ำกึ่ง", character="pailin", prev_tail="")
    assert (mid.verdict, mid.score, mid.text) == (Verdict.REVIEW, 0.6, "นี่คือคำก้ำกึ่ง")
    low = await h.gate.check_output("สวัสดี", character="pailin", prev_tail="")
    assert (low.verdict, low.score) == (Verdict.PASS, 0.0)
    h.gate.set_strict("pailin", True)
    strict = await h.gate.check_output("นี่คือคำก้ำกึ่ง", character="pailin", prev_tail="")
    assert strict.verdict is Verdict.BLOCK
    args = await h.gate.check_args("memory", ["จำคำเสี่ยงมาก"], character="pailin")
    assert args.verdict is Verdict.BLOCK


async def test_tier1_is_not_consulted_when_off_or_already_blocked() -> None:
    clf = FakeClassifier({"คำเสี่ยงมาก": 0.9})
    off = Harness(classifier=clf)  # tier1 = "off" by default
    res = await off.gate.check_output("คำเสี่ยงมาก", character="pailin", prev_tail="")
    assert res.verdict is Verdict.PASS and not clf.calls
    on = Harness(tier1="on", classifier=clf)
    await on.gate.check_output("ไอ้เหี้ย", character="pailin", prev_tail="")
    assert not clf.calls


async def test_tier1_strict_mode_setting() -> None:
    clf = FakeClassifier({"คำเสี่ยงมาก": 0.9})
    h = Harness(tier1="strict", classifier=clf)
    assert (
        await h.gate.check_output("คำเสี่ยงมาก", character="pailin", prev_tail="")
    ).verdict is Verdict.PASS
    h.gate.set_strict("pailin", True)
    assert (
        await h.gate.check_output("คำเสี่ยงมาก", character="pailin", prev_tail="")
    ).verdict is Verdict.BLOCK


async def test_tier1_timeout_fails_open_but_monarchy_stays_blocked() -> None:
    clf = FakeClassifier({"สวัสดี": 0.99}, delay_s=5.0)
    h = Harness(tier1="on", tier1_timeout_ms=30, classifier=clf, clock=SystemClock())
    res = await h.gate.check_output("สวัสดี", character="pailin", prev_tail="")
    assert res.verdict is Verdict.PASS  # fail open
    assert h.gate.tier1_timeouts == 1  # the 5 s classifier was cut off at 30 ms
    royal = await h.gate.check_output("ในหลวง", character="pailin", prev_tail="")
    assert (royal.verdict, royal.category, royal.fail_closed) == (
        Verdict.BLOCK,
        "monarchy_112",
        True,
    )


async def test_tier1_errors_fail_open() -> None:
    class Broken:
        name = "broken"

        async def score(self, texts: Any, direction: Any) -> list[float]:
            raise RuntimeError("onnx session died")

    h = Harness(tier1="on", classifier=Broken())
    res = await h.gate.check_output("สวัสดี", character="pailin", prev_tail="")
    assert res.verdict is Verdict.PASS and h.gate.tier1_errors == 1


# --- reload / config / audit wiring -------------------------------------------------------


def test_reload_reloads_tier0_and_forgets_names() -> None:
    tier0 = FakeTextFilter(["คำต้องห้าม"])
    clock = FakeClock()
    gate = LayeredSafetyGate(tier0, bus=FakeEventBus(clock), clock=clock)
    assert gate.display_name("คำใหม่", character="pailin") == "คำใหม่"
    tier0.blocklist.append("คำใหม่")
    assert gate.display_name("คำใหม่", character="pailin") == "คำใหม่"  # cached
    gate.reload()
    assert tier0.reloads == 1
    assert gate.display_name("คำใหม่", character="pailin") == ANON_NAME


def test_cfg_may_be_a_mapping() -> None:
    clock = FakeClock()
    gate = LayeredSafetyGate(
        base_filter(), bus=FakeEventBus(clock), clock=clock, cfg={"prev_tail_chars": 12}
    )
    assert gate.cfg.prev_tail_chars == 12
    with pytest.raises(ValueError):
        LayeredSafetyGate(
            base_filter(), bus=FakeEventBus(clock), clock=clock, cfg={"fail_closed_categories": []}
        )


async def test_audit_writes_through_the_sink() -> None:
    rows: list[Any] = []

    async def sink(rec: Any) -> None:
        rows.append(rec)

    clock = SystemClock()
    audit = ModerationAudit(sink, clock=clock, tasks=FakeTaskSupervisor())
    gate = LayeredSafetyGate(base_filter(), bus=FakeEventBus(clock), clock=clock, audit=audit)
    gate.check_input(chat("โทร 0812345678 ควย", platform=Platform.YOUTUBE), character="pailin")
    await audit.flush()
    (row,) = rows
    assert row.source == "youtube" and "0812345678" not in row.text_masked


def test_a_broken_audit_does_not_break_moderation(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    clock = FakeClock()
    audit = ModerationAudit(None, clock=clock)
    monkeypatch.setattr(audit, "record", boom)
    gate = LayeredSafetyGate(base_filter(), bus=FakeEventBus(clock), clock=clock, audit=audit)
    res, _ = gate.check_input(chat("ไอ้เหี้ย"), character="pailin")
    assert res.verdict is Verdict.DROP


async def test_watch_lists_hot_reloads_changed_files(lists: Path) -> None:
    clock = FakeClock()
    bus = FakeEventBus(clock)
    gate = LayeredSafetyGate(KeywordRegexFilter(lists, warm=False), bus=bus, clock=clock)

    async def verdict() -> Verdict:
        return (await gate.check_output("คำใหม่เอี่ยม", character="pailin", prev_tail="")).verdict

    def reloads() -> int:
        return sum(1 for e in bus.history if isinstance(e, Alert) and e.level == "info")

    async def tick_until(cond: Callable[[], bool]) -> None:
        # the watcher's file access runs in a real thread: advance fake time in small steps
        for _ in range(500):
            if cond():
                return
            await clock.run_for(0.5)
            await asyncio.sleep(0.005)
        raise AssertionError("condition not reached")

    assert await verdict() is Verdict.PASS
    task = asyncio.create_task(gate.watch_lists(interval_s=1.0))
    try:
        await clock.run_for(3.0)
        assert reloads() == 0  # nothing changed
        (lists / "zz.toml").write_text('category = "slur"\ntoken = ["คำใหม่เอี่ยม"]\n', "utf-8")
        await tick_until(lambda: reloads() == 1)
        assert await verdict() is Verdict.BLOCK
        (lists / "zz.toml").write_text("category = [", "utf-8")  # broken: keep the old lists
        await clock.run_for(5.0)
        await asyncio.sleep(0.05)
        assert await verdict() is Verdict.BLOCK
        assert not task.done() and reloads() == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_reload_if_changed_needs_a_capable_filter() -> None:
    clock = FakeClock()
    gate = LayeredSafetyGate(FakeTextFilter(), bus=FakeEventBus(clock), clock=clock)
    assert gate.reload_if_changed() is False
