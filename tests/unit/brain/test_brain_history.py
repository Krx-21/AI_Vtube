"""History: heard-text rendering, freeze-at-first-render, corrections, compaction trigger."""

from __future__ import annotations

from brain_testkit import stim

from aivtube.brain.history import History, render_turn_text
from aivtube.contracts.memory import Turn
from aivtube.contracts.types import StimulusKind
from aivtube.testing.fakes import FakeClock, FakeMemoryStore
from aivtube.text import estimate_tokens


async def make(clock: FakeClock, budget: int = 5000) -> tuple[History, FakeMemoryStore]:
    memory = FakeMemoryStore(clock=clock)
    await memory.start_session()
    return History(memory, budget_tokens=budget, estimate=estimate_tokens, clock=clock), memory


async def assistant(h: History, emitted: str, **kw: object) -> int:
    args: dict[str, object] = {
        "turn_ref": "t1/u1",
        "emitted": emitted,
        "heard": None,
        "interrupted": False,
        "filtered": False,
        "provider": "fake",
        "tool_calls": None,
        "provider_extra": None,
    }
    args.update(kw)
    return await h.append_assistant(**args)  # type: ignore[arg-type]


def test_render_markers() -> None:
    cut = Turn(
        role="assistant", text="สวัสดีค่ะ ทุกคน", source="llm", heard_text="สวัสดีค่ะ ", interrupted=True
    )
    assert render_turn_text(cut) == "สวัสดีค่ะ" + "…" + " [ถูกขัดจังหวะ]"
    filtered = Turn(role="assistant", text="x", source="llm", heard_text="ก่อนหน้า", filtered=True)
    assert render_turn_text(filtered) == "ก่อนหน้า [Filtered.]"
    nothing = Turn(role="assistant", text="x", source="llm", heard_text="", filtered=True)
    assert render_turn_text(nothing) == "[Filtered.]"
    plain = Turn(role="assistant", text="พูดเต็มๆ", source="llm")
    assert render_turn_text(plain) == "พูดเต็มๆ"


async def test_appends_persist_in_order(fake_clock: FakeClock) -> None:
    h, memory = await make(fake_clock)
    u = await h.append_user("[สตรีมเมอร์] สวัสดี", stim(StimulusKind.VOICE), turn_ref="t1")
    a = await assistant(h, "สวัสดีค่ะ")
    n = await h.append_note("[สตรีมเมอร์คุยกับแชท] …")
    assert [t.id for t in h.turns()] == [u, a, n]
    stored = await memory.recent_turns(memory.epoch)
    assert [t.role for t in stored] == ["user", "assistant", "note"]


async def test_heard_before_freeze_updates_the_line(fake_clock: FakeClock) -> None:
    h, _ = await make(fake_clock)
    a = await assistant(h, "หนึ่ง สอง สาม")
    await h.set_heard(a, "หนึ่ง สอง", interrupted=True, filtered=False)
    assert render_turn_text(h.turns()[-1]) == "หนึ่ง สอง… [ถูกขัดจังหวะ]"
    assert h.pending_correction() is None


async def test_after_freeze_a_later_change_becomes_a_correction_note(fake_clock: FakeClock) -> None:
    h, memory = await make(fake_clock)
    a = await assistant(h, "หนึ่ง สอง สาม")
    await h.set_heard(a, "หนึ่ง สอง สาม", interrupted=False, filtered=False, final=False)
    h.freeze_all()
    before = render_turn_text(h.turns()[-1])
    await h.set_heard(a, "หนึ่ง", interrupted=True, filtered=False)
    assert render_turn_text(h.turns()[-1]) == before  # frozen: unchanged
    note = h.pending_correction()
    assert note is not None and "หนึ่ง" in note
    assert h.pending_correction() is None  # consumed
    # the final heard text is still stored for audit, as a hidden note
    audit = [t for t in await memory.recent_turns(memory.epoch) if t.source == "heard"]
    assert audit and audit[-1].heard_text == "หนึ่ง" and audit[-1].interrupted


async def test_same_text_after_freeze_needs_no_correction(fake_clock: FakeClock) -> None:
    h, _ = await make(fake_clock)
    a = await assistant(h, "สวัสดีค่ะ!!")
    await h.set_heard(a, "สวัสดีค่ะ!!", interrupted=False, filtered=False, final=False)
    h.freeze(a)
    assert h.is_frozen(a)
    await h.set_heard(a, "สวัสดีค่ะ!", interrupted=False, filtered=False)  # normalised the same
    assert h.pending_correction() is None


async def test_needs_compaction_above_budget(fake_clock: FakeClock) -> None:
    h, _ = await make(fake_clock, budget=50)
    await h.append_user("[สตรีมเมอร์] " + "ก" * 60, stim(StimulusKind.VOICE))
    assert not h.needs_compaction()
    await assistant(h, "ข" * 60)
    assert h.needs_compaction()


async def test_load_applies_audit_and_hides_it(fake_clock: FakeClock) -> None:
    h, memory = await make(fake_clock)
    await h.append_user("[สตรีมเมอร์] เล่าเรื่อง", stim(StimulusKind.VOICE))
    a = await assistant(h, "เรื่องยาวมากเลย", turn_ref="t9/u1")
    await h.set_heard(a, "เรื่องยาว", interrupted=True, filtered=False)
    fresh = History(memory, budget_tokens=5000, estimate=estimate_tokens, clock=fake_clock)
    await fresh.load(memory.epoch)
    turns = fresh.turns()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert render_turn_text(turns[-1]) == "เรื่องยาว… [ถูกขัดจังหวะ]"


async def test_rebase_drops_compacted_turns(fake_clock: FakeClock) -> None:
    h, _ = await make(fake_clock)
    ids = [await h.append_note(f"n{i}") for i in range(4)]
    await h.rebase(2, ids[1])
    assert [t.text for t in h.turns()] == ["n2", "n3"] and h.epoch == 2


async def test_a_failing_store_keeps_the_turn_in_memory(fake_clock: FakeClock) -> None:
    memory = FakeMemoryStore(clock=fake_clock)  # no session: append_turn raises
    h = History(memory, budget_tokens=100, estimate=estimate_tokens, clock=fake_clock)
    turn_id = await h.append_note("hello")
    assert turn_id < 0 and h.turns()[-1].text == "hello"


async def test_base_id_and_sync_mark_heard(fake_clock: FakeClock) -> None:
    h, memory = await make(fake_clock)
    assert h.base_id() == 0
    u = await h.append_user("[สตรีมเมอร์] หนึ่ง", stim(StimulusKind.VOICE))
    a = await assistant(h, "สอง สาม")
    assert h.base_id() == u - 1  # a new epoch "up to" here keeps every current turn
    audit = h.mark_heard(a, "สอง", interrupted=True, filtered=False, final=True)
    assert audit is not None and audit.source == "heard" and audit.heard_text == "สอง"
    assert render_turn_text(h.turns()[-1]) == "สอง… [ถูกขัดจังหวะ]"
    assert h.mark_heard(a, "สอง", interrupted=True, filtered=False, final=False) is None
    await h.persist_audit(audit)
    h.rebase_now(7, a)
    assert h.turns() == [] and h.base_id() >= a and h.epoch == 7
    assert any(t.source == "heard" for t in await memory.recent_turns(memory.epoch))
