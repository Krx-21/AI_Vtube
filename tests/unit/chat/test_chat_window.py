"""ScoredChatWindow: the §4.3 score, routing, softmax selection with a seeded RNG, read-aloud."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

import pytest

from aivtube.chat import AliasMatcher, ScoredChatWindow, WindowConfig
from aivtube.contracts.chat import ChatSelection, ChatWindow
from aivtube.contracts.types import ChatMessage, ChatUser, MsgKind, Platform
from aivtube.testing.fakes import FakeClock

ALIASES = ("ไพลิน", "pailin", "ไพ่ลิน", "น้องไพลิน")
_ids = itertools.count(1)


def msg(
    clock: FakeClock,
    text: str,
    user: str = "viewer",
    *,
    kind: MsgKind = MsgKind.TEXT,
    platform: Platform = Platform.TWITCH,
    age: float = 0.0,
    value_usd: float = 0.0,
    amount: float = 0.0,
    source_channel: str | None = None,
    first_msg: bool = False,
    msg_id: str | None = None,
    **flags: Any,
) -> ChatMessage:
    now = clock.now()
    return ChatMessage(
        platform=platform,
        id=msg_id or f"m-{next(_ids)}",
        user=ChatUser(platform, f"id-{user}", user, **flags),
        text=text,
        ts=now - age,
        received=now,
        kind=kind,
        amount=amount,
        value_usd=value_usd,
        first_msg=first_msg,
        source_channel=source_channel,
    )


def window(clock: FakeClock, seed: int = 1, **cfg: Any) -> ScoredChatWindow:
    return ScoredChatWindow(clock, WindowConfig(**cfg), ALIASES, rng=random.Random(seed))


def ids(messages: tuple[ChatMessage, ...]) -> list[str]:
    return [m.id for m in messages]


# --- construction ----------------------------------------------------------------------------
def test_is_a_chat_window_and_accepts_a_matcher_or_aliases(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    assert isinstance(w, ChatWindow)
    custom = ScoredChatWindow(fake_clock, WindowConfig(), lambda text: "zz" in text)
    m = msg(fake_clock, "zz top")
    custom.add(m)
    assert custom.pending() == (1, True)
    via_matcher = ScoredChatWindow(fake_clock, None, AliasMatcher(["pailin"]))
    via_matcher.add(msg(fake_clock, "hi PAILIN"))
    assert via_matcher.pending() == (1, True)


def test_config_validation_and_from_config() -> None:
    with pytest.raises(ValueError):
        WindowConfig(temperature=0)
    with pytest.raises(ValueError):
        WindowConfig(user_burst=0)

    class _W:
        horizon_s = 30.0
        must_ack_ttl_s = 300.0
        temperature = 0.5
        user_cooldown_s = 45.0
        dedupe_lru = 1000

    class _Chat:
        window = _W()
        max_msg_chars = 200

    cfg = WindowConfig.from_config(_Chat(), max_ambient=3)
    assert (cfg.horizon_s, cfg.must_ack_ttl_s, cfg.temperature) == (30.0, 300.0, 0.5)
    assert (cfg.user_cooldown_s, cfg.dedupe_lru, cfg.max_msg_chars, cfg.max_ambient) == (
        45.0,
        1000,
        200,
        3,
    )


def test_from_config_reads_the_real_schema() -> None:
    from aivtube.config.schema import ChatConfig

    cfg = WindowConfig.from_config(ChatConfig())
    assert cfg == WindowConfig()


# --- the §4.3 score --------------------------------------------------------------------------
def test_score_matches_the_formula_exactly(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    m = msg(
        fake_clock,
        "ไพลินชอบกินอะไรครับ",
        "fan",
        age=15.0,
        first_msg=True,
        is_sub=True,
        is_mod=True,
        is_vip=True,
    )
    w.add(m)
    expected = 2 * math.exp(-1.0) + 3 + 0.8 + 0.7 + 0.5 + 0.3 + 0.5
    assert w.score(m.id) == pytest.approx(expected)
    [(snap_msg, snap_score)] = w.snapshot()
    assert snap_msg is m and snap_score == pytest.approx(expected)


def test_score_components_individually(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    plain = msg(fake_clock, "สวัสดีทุกคน", "a")
    question = msg(fake_clock, "เล่นเกมอะไรอยู่", "b")
    mention = msg(fake_clock, "น้องไพลินจ๋า", "c")
    sub = msg(fake_clock, "สวัสดีค่ะ", "d", is_sub=True)
    for m in (plain, question, mention, sub):
        w.add(m)
    assert w.score(plain.id) == pytest.approx(2.0)
    assert w.score(question.id) == pytest.approx(2.8)
    assert w.score(mention.id) == pytest.approx(5.0)
    assert w.score(sub.id) == pytest.approx(2.7)
    fake_clock.advance(30.0)
    assert w.score(plain.id) == pytest.approx(2 * math.exp(-2.0))
    assert w.score("not-there") is None


def test_age_uses_the_earlier_of_ts_and_received(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    future = ChatMessage(
        Platform.TWITCH,
        "f",
        ChatUser(Platform.TWITCH, "u", "u"),
        "hi",
        ts=fake_clock.now() + 50,
        received=fake_clock.now(),
    )
    w.add(future)
    assert w.score("f") == pytest.approx(2.0)  # a platform clock ahead of ours is not "fresh+"


def test_duplicates_are_penalised_by_ln_dup(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    spam = [msg(fake_clock, text, f"s{i}") for i, text in enumerate(["5555", "555555!!", "55 5"])]
    unique = msg(fake_clock, "มาแล้วค่ะ", "u")
    for m in (*spam, unique):
        w.add(m)
    for m in spam:
        assert w.score(m.id) == pytest.approx(2.0 - 1.5 * math.log(3))
    assert w.score(unique.id) == pytest.approx(2.0)


def test_recently_picked_user_is_penalised_for_60_s(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    first = msg(fake_clock, "สวัสดีค่ะ", "alice")
    w.add(first)
    sel = w.select(fake_clock.now(), k=1)
    assert ids(sel.candidates) == [first.id]
    fake_clock.advance(10.0)
    again = msg(fake_clock, "มาอีกแล้ว", "alice")
    w.add(again)
    assert w.score(again.id) == pytest.approx(2.0 - 5.0)
    fake_clock.advance(51.0)  # 61 s after the pick
    assert w.score(again.id) == pytest.approx(2 * math.exp(-51.0 / 15.0))


# --- routing (add) ---------------------------------------------------------------------------
def test_support_goes_to_must_ack_sorted_by_value(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    small = msg(fake_clock, "", "a", kind=MsgKind.DONATION, value_usd=1.0, amount=100)
    sub = msg(fake_clock, "", "b", kind=MsgKind.SUB, value_usd=4.99)
    raid = msg(fake_clock, "", "c", kind=MsgKind.RAID, amount=42)
    big = msg(fake_clock, "ขอเพลงหน่อย", "d", kind=MsgKind.DONATION, value_usd=20.0)
    gift = msg(fake_clock, "", "e", kind=MsgKind.GIFT_SUB, value_usd=24.95)
    redeem = msg(fake_clock, "ร้องเพลง", "f", kind=MsgKind.REDEEM)
    for m in (small, sub, raid, big, gift, redeem):
        assert w.add(m) == "priority"
    w.add(msg(fake_clock, "สวัสดี", "g"))
    assert w.pending() == (7, False)
    sel = w.select(fake_clock.now())
    assert ids(sel.must_ack) == [gift.id, big.id, sub.id, small.id, raid.id, redeem.id]
    assert all(m.kind is MsgKind.TEXT for m in sel.candidates)
    assert w.pending() == (0, False)
    assert w.select(fake_clock.now()).must_ack == ()


def test_drops_shared_chat_commands_empty_long_system_and_duplicates(
    fake_clock: FakeClock,
) -> None:
    w = window(fake_clock, max_msg_chars=20)
    cases = {
        "shared_chat": msg(fake_clock, "hi", "a", source_channel="99999"),
        "command": msg(fake_clock, "!discord", "b"),
        "empty": msg(fake_clock, "   ", "c"),
        "too_long": msg(fake_clock, "ก" * 21, "d"),
        "system": msg(fake_clock, "announcement", "e", kind=MsgKind.SYSTEM),
    }
    for reason, m in cases.items():
        assert w.add(m) == "dropped", reason
        assert w.last_reason == reason
    shared_donation = msg(
        fake_clock, "", "f", kind=MsgKind.DONATION, value_usd=5, source_channel="99999"
    )
    assert w.add(shared_donation) == "dropped"
    ok = msg(fake_clock, "ok", "g")
    assert w.add(ok) == "window" and w.last_reason == "window"
    assert w.add(ok) == "dropped" and w.last_reason == "duplicate"
    same_id_other_platform = msg(fake_clock, "ok", "g", platform=Platform.YOUTUBE, msg_id=ok.id)
    assert w.add(same_id_other_platform) == "window"
    assert w.drops["shared_chat"] == 2 and w.drops["duplicate"] == 1
    assert w.pending() == (2, False)


def test_id_dedupe_is_an_lru(fake_clock: FakeClock) -> None:
    w = window(fake_clock, dedupe_lru=3, user_burst=100)
    first = msg(fake_clock, "a", "u0")
    w.add(first)
    for i in range(3):
        w.add(msg(fake_clock, f"x{i}", f"u{i + 1}"))
    w.select(fake_clock.now())
    assert w.add(first) == "window"  # evicted from the 3-entry LRU, so accepted again


def test_per_user_rate_limit(fake_clock: FakeClock) -> None:
    w = window(fake_clock)  # 3 messages per 10 s per user
    results = []
    for i in range(5):
        results.append(w.add(msg(fake_clock, f"ข้อความ {i}", "spammer")))
        fake_clock.advance(0.4)
    assert results == ["window"] * 3 + ["dropped"] * 2
    assert w.drops["rate_limited"] == 2
    assert w.add(msg(fake_clock, "คนอื่น", "other")) == "window"
    fake_clock.advance(10.0)
    assert w.add(msg(fake_clock, "กลับมาแล้ว", "spammer")) == "window"


def test_rate_limit_is_per_platform_user(fake_clock: FakeClock) -> None:
    w = window(fake_clock, user_burst=1)
    assert w.add(msg(fake_clock, "a", "same")) == "window"
    assert w.add(msg(fake_clock, "b", "same", platform=Platform.YOUTUBE)) == "window"
    assert w.add(msg(fake_clock, "c", "same")) == "dropped"


def test_max_window_evicts_the_oldest(fake_clock: FakeClock) -> None:
    w = window(fake_clock, max_window=3)
    added = [msg(fake_clock, f"ข้อความ {i}", f"u{i}") for i in range(5)]
    for m in added:
        w.add(m)
    assert {m.id for m, _ in w.snapshot()} == {m.id for m in added[2:]}
    assert w.pending()[0] == 3


# --- selection -------------------------------------------------------------------------------
def _twelve_in_three_seconds(clock: FakeClock, w: ScoredChatWindow) -> dict[str, ChatMessage]:
    users = ["a", "b", "c", "d", "e", "f", "a", "b", "c", "g", "h", "i"]
    texts = {10: "@pailin ชอบสีอะไร"}
    out: dict[str, ChatMessage] = {}
    for i, user in enumerate(users):
        m = msg(clock, texts.get(i, f"ข้อความที่ {i} จาก {user}"), user)
        out[m.id] = m
        assert w.add(m) == "window"
        clock.advance(0.25)
    return out


def test_twelve_messages_select_three_one_per_user_mention_preferred(
    fake_clock: FakeClock,
) -> None:
    w = window(fake_clock, seed=7)
    added = _twelve_in_three_seconds(fake_clock, w)
    assert w.pending() == (12, True)
    sel = w.select(fake_clock.now(), k=3)
    assert isinstance(sel, ChatSelection)
    assert len(sel.candidates) == 3
    assert len({m.user.id for m in sel.candidates}) == 3
    assert any("pailin" in m.text for m in sel.candidates)
    assert [m.ts for m in sel.candidates] == sorted(m.ts for m in sel.candidates)
    assert set(ids(sel.ambient)) <= set(added) - set(ids(sel.candidates))
    assert w.pending() == (0, False)
    assert w.snapshot() == []
    assert w.select(fake_clock.now()).candidates == ()


def test_mention_is_preferred_across_seeds(fake_clock: FakeClock) -> None:
    hits = 0
    for seed in range(200):
        clock = FakeClock()
        w = window(clock, seed=seed)
        _twelve_in_three_seconds(clock, w)
        sel = w.select(clock.now(), k=3)
        hits += any("pailin" in m.text for m in sel.candidates)
    assert hits >= 195


def test_selection_is_deterministic_for_a_seed(fake_clock: FakeClock) -> None:
    def run(seed: int) -> list[str]:
        clock = FakeClock()
        w = window(clock, seed=seed)
        for i in range(20):
            w.add(msg(clock, f"ข้อความ {i}", f"user{i}", msg_id=f"d{i}"))
            clock.advance(0.1)
        return [m.text for m in w.select(clock.now(), k=3).candidates]

    assert run(42) == run(42)
    assert len({tuple(run(s)) for s in range(10)}) > 1


def test_softmax_probabilities_follow_the_temperature(fake_clock: FakeClock) -> None:
    # two messages whose scores differ by 0.8 (a question): P(question) = 1 / (1 + e^(-0.8/0.6))
    rng = random.Random(1234)
    wins = 0
    trials = 3000
    for _ in range(trials):
        clock = FakeClock()
        w = ScoredChatWindow(clock, WindowConfig(), ALIASES, rng=rng)
        w.add(msg(clock, "สวัสดีค่ะ", "plain"))
        q = msg(clock, "เล่นเกมอะไรครับ", "asker")
        w.add(q)
        wins += w.select(clock.now(), k=1).candidates[0].id == q.id
    expected = 1.0 / (1.0 + math.exp(-0.8 / 0.6))
    assert wins / trials == pytest.approx(expected, abs=0.03)


def test_one_candidate_per_near_duplicate_group(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    for i in range(6):
        w.add(msg(fake_clock, "5555555" if i % 2 else "555", f"laugher{i}"))
    real = msg(fake_clock, "วันนี้เล่นเกมอะไร", "asker")
    w.add(real)
    sel = w.select(fake_clock.now(), k=3)
    assert len(sel.candidates) == 2  # one "555" group + the real question
    assert real in sel.candidates


def test_one_candidate_per_user_picks_their_best_message(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    w.add(msg(fake_clock, "ก่อนหน้า", "solo", age=30))
    best = msg(fake_clock, "ไพลินชอบเกมอะไร", "solo")
    w.add(best)
    sel = w.select(fake_clock.now(), k=3)
    assert sel.candidates == (best,)
    assert len(sel.ambient) == 1


def test_expired_messages_are_not_selected(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    old = msg(fake_clock, "เก่าแล้ว", "old")
    w.add(old)
    fake_clock.advance(41.0)
    fresh = msg(fake_clock, "ใหม่", "new")
    w.add(fresh)
    assert w.pending() == (1, False)
    sel = w.select(fake_clock.now(), k=3)
    assert sel.candidates == (fresh,) and sel.ambient == ()


def test_must_ack_expires_after_its_ttl(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    w.add(msg(fake_clock, "", "d", kind=MsgKind.DONATION, value_usd=5))
    fake_clock.advance(601.0)
    assert w.pending() == (0, False)
    assert w.select(fake_clock.now()).must_ack == ()
    assert w.drops["must_ack_expired"] == 1


def test_k_zero_still_consumes_and_ambient_is_bounded(fake_clock: FakeClock) -> None:
    w = window(fake_clock, max_ambient=2)
    for i in range(5):
        w.add(msg(fake_clock, f"ข้อความ {i}", f"u{i}"))
        fake_clock.advance(0.1)
    sel = w.select(fake_clock.now(), k=0)
    assert sel.candidates == ()
    assert [m.text for m in sel.ambient] == ["ข้อความ 3", "ข้อความ 4"]
    assert w.pending() == (0, False)
    w0 = window(fake_clock, max_ambient=0)
    w0.add(msg(fake_clock, "x", "x"))
    w0.add(msg(fake_clock, "y", "y"))
    assert w0.select(fake_clock.now(), k=1).ambient == ()


# --- consume / recent / snapshot -------------------------------------------------------------
def test_consume_and_recent_for_read_aloud_dedupe(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    old = msg(fake_clock, "ข้อความเก่ามาก", "old")
    w.add(old)
    fake_clock.advance(61.0)
    read = msg(fake_clock, "ไพลินชอบแมวไหม", "reader")
    live = msg(fake_clock, "สวัสดีครับ", "live")
    donation = msg(fake_clock, "", "donor", kind=MsgKind.DONATION, value_usd=2)
    for m in (read, live, donation):
        w.add(m)
    assert w.consume(read.id) is read
    assert w.consume(read.id) is None
    assert w.consume(donation.id) is donation
    assert w.consume("never-added") is None
    recent = w.recent(60.0)
    assert read in recent and live in recent and donation in recent
    assert old not in recent
    assert old in w.recent(120.0)
    sel = w.select(fake_clock.now(), k=3)
    assert sel.candidates == (live,) and sel.must_ack == ()


def test_snapshot_is_sorted_by_score(fake_clock: FakeClock) -> None:
    w = window(fake_clock)
    w.add(msg(fake_clock, "เก่านิดนึง", "a", age=20))
    w.add(msg(fake_clock, "ไพลินจ๋า", "b"))
    w.add(msg(fake_clock, "สวัสดี", "c"))
    snap = w.snapshot()
    scores = [s for _, s in snap]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) for s in scores)
    assert snap[0][0].text == "ไพลินจ๋า"
