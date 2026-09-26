"""Arbiter: rank/TTL/aging selection, cadence rules and merge-after-decision (§4.3)."""

from __future__ import annotations

import asyncio
import random

import pytest
from brain_testkit import stim

from aivtube.brain.arbiter import Arbiter, MergedContext
from aivtube.chat import ScoredChatWindow
from aivtube.contracts.events import StimulusExpired, StimulusQueued
from aivtube.contracts.types import Priority, Rank, Stimulus, StimulusKind
from aivtube.testing.fakes import FakeClock, FakeEventBus, make_chat_message


def make(clock: FakeClock, bus: FakeEventBus | None = None, **cfg: float) -> Arbiter:
    window = ScoredChatWindow(clock, rng=random.Random(7))
    return Arbiter(clock, window, dict(cfg), bus=bus, character="pailin")


def no() -> bool:
    return False


async def pick(
    arb: Arbiter,
    clock: FakeClock,
    *,
    within: float = 30.0,
    idle: float | None = None,
    user_speaking: bool = False,
    speaking: bool = False,
) -> tuple[Stimulus | None, float]:
    """Run ``next()`` on fake time; return the stimulus and the fake time it came out."""
    t0 = clock.now()
    task = asyncio.create_task(
        arb.next(
            idle_deadline=idle,
            user_speaking=lambda: user_speaking,
            speaking=lambda: speaking,
        )
    )
    await clock.run_until(task.done, within=within, step=0.05)
    return task.result(), clock.now() - t0


async def test_lowest_rank_wins_and_oldest_breaks_ties(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    old = stim(StimulusKind.MENTION, created=999.0, priority=Priority.HIGH)
    new = stim(StimulusKind.MENTION, created=1000.0, priority=Priority.HIGH)
    voice = stim(StimulusKind.VOICE, created=1000.0)
    for s in (new, old, voice):
        arb.push(s)
    assert (await pick(arb, fake_clock))[0] is voice
    assert (await pick(arb, fake_clock))[0] is old
    assert (await pick(arb, fake_clock))[0] is new


async def test_aging_bonus_prevents_starvation(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    waiting = stim(StimulusKind.GAME_CONTEXT, created=900.0, ttl_s=None, priority=Priority.HIGH)
    fresh = stim(StimulusKind.SUPPORT, created=1000.0, priority=Priority.HIGH)
    arb.push(fresh)
    arb.push(waiting)
    # rank 70 aged 100 s -> 70 - 20 = 50 > 30: support first ...
    assert (await pick(arb, fake_clock))[0] is fresh
    fake_clock.set(1100.0)  # 200 s later: 70 - 40 = 30 ties a brand-new support and is older
    arb.push(fresh2 := stim(StimulusKind.SUPPORT, created=1100.0, priority=Priority.HIGH))
    assert (await pick(arb, fake_clock))[0] is waiting
    assert (await pick(arb, fake_clock))[0] is fresh2


async def test_expired_stimuli_emit_stimulus_expired(fake_clock: FakeClock) -> None:
    bus = FakeEventBus(fake_clock)
    arb = make(fake_clock, bus)
    voice = stim(StimulusKind.VOICE, created=fake_clock.now() - 30.0)  # ttl 20 s
    arb.push(voice)
    later = stim(StimulusKind.MENTION, created=fake_clock.now(), priority=Priority.HIGH)
    arb.push(later)
    assert (await pick(arb, fake_clock))[0] is later
    assert [e.stimulus_id for e in bus.of_type(StimulusExpired)] == [voice.id]
    assert len(bus.of_type(StimulusQueued)) == 2


async def test_while_user_speaking_only_operator_is_eligible(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    voice = stim(StimulusKind.VOICE, created=fake_clock.now())
    op = stim(StimulusKind.OPERATOR, created=fake_clock.now(), payload={"op": "direct"})
    arb.push(voice)
    speaking = True
    task = asyncio.create_task(
        arb.next(idle_deadline=None, user_speaking=lambda: speaking, speaking=no)
    )
    await fake_clock.run_for(3.0)
    assert not task.done()  # the voice stimulus waits
    arb.push(op)
    await fake_clock.run_for(0.01)
    assert task.result() is op
    task = asyncio.create_task(
        arb.next(idle_deadline=None, user_speaking=lambda: speaking, speaking=no)
    )
    await fake_clock.run_for(1.0)
    assert not task.done()
    speaking = False
    arb.poke()
    await fake_clock.run_for(0.01)
    assert task.result() is voice


async def test_while_her_audio_plays_only_high_starts(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    low = stim(StimulusKind.MENTION, created=fake_clock.now())
    arb.push(low)
    s, _ = await pick(arb, fake_clock, speaking=False)
    assert s is low
    arb.push(low2 := stim(StimulusKind.MENTION, created=fake_clock.now(), priority=Priority.LOW))
    playing = True
    task = asyncio.create_task(
        arb.next(idle_deadline=None, user_speaking=no, speaking=lambda: playing)
    )
    await fake_clock.run_for(10.0)
    assert not task.done()
    arb.push(high := stim(StimulusKind.VOICE, created=fake_clock.now()))
    await fake_clock.run_for(0.01)
    assert task.result() is high
    playing = False
    arb.mark_free(fake_clock.now())
    s, waited = await pick(arb, fake_clock)
    assert s is low2
    assert waited == pytest.approx(0.3, abs=0.06)  # post-speech gap


async def test_chat_gathers_1s_and_keeps_a_4s_interval(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    now = fake_clock.now()
    for i in range(2):
        m = make_chat_message(f"ข้อความ {i}", user=f"u{i}", clock=fake_clock)
        arb.window.add(m)
        arb.push(stim(StimulusKind.CHAT, created=now))
    s, waited = await pick(arb, fake_clock)
    assert s is not None and s.kind is StimulusKind.CHAT
    assert waited == pytest.approx(1.0, abs=0.06)  # 1 s gather
    ctx = arb.drain_context(s)
    assert ctx.chat is not None and len(ctx.chat.candidates) == 2
    arb.window.add(make_chat_message("อีกข้อความ", user="u9", clock=fake_clock))
    arb.push(stim(StimulusKind.CHAT, created=fake_clock.now()))
    s, waited = await pick(arb, fake_clock)
    assert s is not None and s.kind is StimulusKind.CHAT
    assert waited == pytest.approx(4.0, abs=0.06)  # chat_min_interval_s


async def test_chat_interval_is_8s_after_a_recent_voice_turn(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    voice = stim(StimulusKind.VOICE, created=fake_clock.now())
    arb.push(voice)
    s, _ = await pick(arb, fake_clock)
    arb.drain_context(voice)
    # a mention decision, then chat within the voice window: interval 8 s
    arb.push(m := stim(StimulusKind.MENTION, created=fake_clock.now(), priority=Priority.LOW))
    s, _ = await pick(arb, fake_clock)
    assert s is m
    arb.drain_context(m)
    arb.push(m2 := stim(StimulusKind.MENTION, created=fake_clock.now(), priority=Priority.LOW))
    s, waited = await pick(arb, fake_clock)
    assert s is m2
    assert waited == pytest.approx(8.0, abs=0.06)


async def test_at_most_one_support_per_decision(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    a = stim(StimulusKind.SUPPORT, created=fake_clock.now() - 1)
    b = stim(StimulusKind.SUPPORT, created=fake_clock.now())
    arb.push(a)
    arb.push(b)
    s, _ = await pick(arb, fake_clock)
    ctx = arb.drain_context(s)  # type: ignore[arg-type]
    assert ctx.support == (a,) and ctx.merged == ()
    assert arb.pending() == [b]


async def test_drain_context_merges_everything_queued_and_the_window(
    fake_clock: FakeClock,
) -> None:
    arb = make(fake_clock)
    voice = stim(StimulusKind.VOICE, created=fake_clock.now())
    arb.push(voice)
    s, _ = await pick(arb, fake_clock)
    assert s is voice
    assert arb.drain_context(voice) == MergedContext(voice)
    # mid-decision arrivals: a mention, a second utterance, window chat, a support
    fake_clock.set(fake_clock.now() + 10.0)
    mention = stim(StimulusKind.MENTION, created=fake_clock.now())
    voice2 = stim(StimulusKind.VOICE, "ต่ออีกนิด", created=fake_clock.now())
    support = stim(StimulusKind.SUPPORT, created=fake_clock.now())
    arb.window.add(make_chat_message("สวัสดี", user="tom", clock=fake_clock))
    for x in (mention, voice2, support, stim(StimulusKind.CHAT, created=fake_clock.now())):
        arb.push(x)
    s, _ = await pick(arb, fake_clock)
    assert s is voice2
    ctx = arb.drain_context(voice2)
    assert ctx.merged == (mention,)
    assert ctx.chat is None  # voice turns never consume the window
    s, _ = await pick(arb, fake_clock)
    assert s is support
    ctx = arb.drain_context(support)
    assert ctx.support == (support,) and ctx.merged == ()
    assert ctx.chat is None  # the merged mention counted as a chat turn: 8 s interval
    s, waited = await pick(arb, fake_clock)
    assert s is not None and s.kind is StimulusKind.CHAT
    assert waited == pytest.approx(8.0 - 0.3, abs=0.06)
    ctx = arb.drain_context(s)
    assert ctx.chat is not None and [m.user.name for m in ctx.chat.candidates] == ["tom"]
    assert arb.pending() == []


async def test_non_voice_decisions_take_the_window_when_the_interval_allows(
    fake_clock: FakeClock,
) -> None:
    arb = make(fake_clock)
    arb.window.add(make_chat_message("สวัสดี", user="tom", clock=fake_clock))
    arb.push(stim(StimulusKind.CHAT, created=fake_clock.now()))
    arb.push(support := stim(StimulusKind.SUPPORT, created=fake_clock.now()))
    s, _ = await pick(arb, fake_clock)
    assert s is support
    ctx = arb.drain_context(support)
    assert ctx.chat is not None and len(ctx.chat.candidates) == 1
    assert arb.pending() == []


async def test_idle_deadline_returns_none(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    s, waited = await pick(arb, fake_clock, idle=fake_clock.now() + 25.0)
    assert s is None and waited == pytest.approx(25.0, abs=0.06)


async def test_restore_and_clear(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    a = stim(StimulusKind.MENTION, created=fake_clock.now())
    b = stim(StimulusKind.VOICE, created=fake_clock.now())
    arb.restore(MergedContext(b, (a,)))
    assert {s.id for s in arb.pending()} == {a.id, b.id}
    arb.push(sup := stim(StimulusKind.SUPPORT, created=fake_clock.now()))
    arb.clear()
    assert arb.pending() == [sup]
    assert arb.remove(sup.id) and arb.pending() == []


async def test_accept_filter_blocks_until_poked(fake_clock: FakeClock) -> None:
    arb = make(fake_clock)
    arb.push(m := stim(StimulusKind.MENTION, created=fake_clock.now()))
    live = False
    task = asyncio.create_task(
        arb.next(
            idle_deadline=None,
            user_speaking=no,
            speaking=no,
            accept=lambda s: live or s.kind in (StimulusKind.OPERATOR, StimulusKind.VOICE),
        )
    )
    await fake_clock.run_for(5.0)
    assert not task.done()
    live = True
    arb.poke()
    await fake_clock.run_for(0.5)
    assert task.result() is m


def test_rank_priorities_are_the_documented_defaults() -> None:
    assert stim(StimulusKind.VOICE).rank is Rank.VOICE
    assert stim(StimulusKind.SUPPORT).priority is Priority.MEDIUM
