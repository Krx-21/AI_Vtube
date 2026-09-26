"""The §4.5 state machine and the §4.13 idle scheduler."""

from __future__ import annotations

import random

import pytest

from aivtube.brain.idle import IdleScheduler
from aivtube.brain.state import BrainStateMachine, avatar_state
from aivtube.contracts.events import StateChanged
from aivtube.contracts.types import Rank, StimulusKind
from aivtube.testing.fakes import FakeAvatarDriver, FakeClock, FakeEventBus

# --- state machine ------------------------------------------------------------------------


def test_transitions_publish_state_changed_and_drive_the_avatar(fake_bus: FakeEventBus) -> None:
    avatar = FakeAvatarDriver()
    sm = BrainStateMachine(character="pailin", bus=fake_bus, avatar=avatar)
    assert sm.set("pre_show")
    assert sm.set("idle")
    assert sm.set("deciding")
    assert sm.set("speaking")
    assert sm.set("idle")
    changes = [(e.old, e.new) for e in fake_bus.of_type(StateChanged)]
    assert changes == [
        ("booting", "pre_show"),
        ("pre_show", "idle"),
        ("idle", "deciding"),
        ("deciding", "speaking"),
        ("speaking", "idle"),
    ]
    assert avatar.states[-3:] == ["thinking", "speaking", "idle"]


def test_paused_only_leaves_through_resume_states(fake_bus: FakeEventBus) -> None:
    sm = BrainStateMachine(character="pailin", bus=fake_bus, initial="speaking")
    assert sm.set("paused")
    assert not sm.set("deciding")
    assert not sm.set("speaking")
    assert sm.state == "paused"
    assert sm.set("idle")


def test_listening_pose_while_the_streamer_speaks(fake_bus: FakeEventBus) -> None:
    avatar = FakeAvatarDriver()
    sm = BrainStateMachine(character="pailin", bus=fake_bus, avatar=avatar, initial="idle")
    sm.set_user_speaking(True)
    assert avatar.states[-1] == "listening"
    sm.set("speaking")
    assert avatar.states[-1] == "speaking"
    assert avatar_state("deciding", user_speaking=False) == "thinking"
    assert avatar_state("paused", user_speaking=True) == "paused"


# --- idle ---------------------------------------------------------------------------------


def test_deadline_is_25_plus_minus_5_after_playback(fake_clock: FakeClock) -> None:
    for seed in range(50):
        idle = IdleScheduler(fake_clock, rng=random.Random(seed))
        assert idle.deadline() is None
        idle.on_playback_end(100.0)
        d = idle.deadline()
        assert d is not None and 120.0 <= d <= 130.0


def test_activity_cancels(fake_clock: FakeClock) -> None:
    idle = IdleScheduler(fake_clock, rng=random.Random(1))
    idle.on_playback_end(0.0)
    idle.on_activity(1.0)
    assert idle.deadline() is None


def test_chat_activity_doubles_up_to_max(fake_clock: FakeClock) -> None:
    idle = IdleScheduler(fake_clock, jitter_s=0.0)
    idle.on_playback_end(0.0)
    idle.on_chat_activity(1.0)
    assert idle.deadline() == pytest.approx(50.0)
    for t in range(2, 6):
        idle.on_chat_activity(float(t))
    assert idle.interval == 120.0
    assert idle.deadline() == pytest.approx(120.0)
    # quiet chat for max_s resets the interval at the next arm
    idle.on_playback_end(200.0)
    assert idle.deadline() == pytest.approx(225.0)


def test_stimulus_has_rank_idle_and_the_last_5_topics(fake_clock: FakeClock) -> None:
    idle = IdleScheduler(fake_clock, jitter_s=0.0)
    idle.on_chat_activity(0.0)
    idle.on_playback_end(0.0)
    s = idle.make_stimulus("pailin", ["a", "b", "c", "d", "e", "f"])
    assert s.kind is StimulusKind.IDLE and s.rank is Rank.IDLE and s.ttl_s is None
    assert s.payload["topics"] == ["b", "c", "d", "e", "f"]
    assert idle.deadline() is None and idle.interval == 25.0
