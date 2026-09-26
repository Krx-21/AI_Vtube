"""Red-team fixtures: every block line stops in its category, every pass line passes.

``tests/fixtures/redteam/block.txt`` covers zero-width characters, letter-spaced and
leet spellings, Thai digits, stretched letters and phrases split across two output chunks.
``pass.txt`` holds false-positive traps (หีบ, ส้มตำ, idioms with ตาย, game talk with ฆ่า).
"""

from __future__ import annotations

import time

import pytest
from safety_testkit import BlockCase, ctx, load_block_cases, load_pass_lines

from aivtube.contracts.safety import Verdict
from aivtube.safety.keyword import KeywordRegexFilter
from aivtube.safety.pii import find_pii

BLOCK = load_block_cases()
PASS = load_pass_lines()
WHOLE = [c for c in BLOCK if c.head is None]
SPLIT = [c for c in BLOCK if c.head is not None]
INPUT_VERDICT = {"pii": Verdict.MASK, "politics": Verdict.REVIEW}


def test_fixture_sizes() -> None:
    assert len(BLOCK) >= 150
    assert len(SPLIT) >= 10
    assert len(PASS) >= 100
    categories = {c.category for c in BLOCK}
    assert categories >= {
        "slur",
        "sexual",
        "doxx",
        "violent_extreme",
        "self_harm",
        "monarchy_112",
        "gambling_scam",
        "politics",
        "pii",
    }


@pytest.mark.parametrize("case", WHOLE, ids=lambda c: c.id)
def test_block_line_blocks_speech(
    tier0_block_politics: KeywordRegexFilter, case: BlockCase
) -> None:
    r = tier0_block_politics.check(case.text, ctx("out"))
    assert (r.verdict, r.category) == (Verdict.BLOCK, case.category), case.text
    assert r.text == ""


@pytest.mark.parametrize("case", WHOLE, ids=lambda c: c.id)
def test_block_line_stops_chat_input(tier0: KeywordRegexFilter, case: BlockCase) -> None:
    r = tier0.check(case.text, ctx("in"))
    expected = INPUT_VERDICT.get(case.category, Verdict.DROP)
    assert (r.verdict, r.category) == (expected, case.category), case.text
    if expected is Verdict.DROP:
        assert r.text == ""
    if expected is Verdict.MASK:
        assert find_pii(r.text) == [], r.text


@pytest.mark.parametrize("case", SPLIT, ids=lambda c: c.id)
def test_split_phrase_is_caught_with_prev_tail(
    tier0_block_politics: KeywordRegexFilter, case: BlockCase
) -> None:
    assert case.head is not None and case.tail is not None
    r = tier0_block_politics.check(case.tail, ctx("out", prev_tail=case.head[-40:]))
    assert (r.verdict, r.category) == (Verdict.BLOCK, case.category), case.text
    assert r.text == ""


@pytest.mark.parametrize("line", PASS)
def test_pass_line_passes(tier0: KeywordRegexFilter, line: str) -> None:
    for direction in ("in", "out"):
        r = tier0.check(line, ctx(direction))
        assert (r.verdict, r.text) == (Verdict.PASS, line), (direction, r)


@pytest.mark.timing
def test_latency_p95_under_1ms(tier0: KeywordRegexFilter) -> None:
    tail = "วันนี้ไพลินจะมาเล่นเกมสยองขวัญกันนะคะ ทุกคน"[-40:]
    texts = PASS + [c.text for c in BLOCK]
    for text in texts:  # warm caches
        tier0.check(text, ctx("out", prev_tail=tail))
    samples: list[float] = []
    for _ in range(3):
        for text in texts:
            for direction, prev in (("in", ""), ("out", tail)):
                t0 = time.perf_counter()
                tier0.check(text, ctx(direction, prev_tail=prev))
                samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    assert p95 < 1.0, f"p95 {p95:.3f} ms over {len(samples)} checks"
