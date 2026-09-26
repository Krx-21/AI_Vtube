"""The §4.4 priority/interrupt table and the §4.3 abort-restart exceptions."""

from __future__ import annotations

import pytest

from aivtube.brain.preempt import BrainSnapshot, PreemptAction, decide_preemption
from aivtube.contracts.types import Priority, Rank, Stimulus, StimulusKind


def stim(priority: Priority, rank: Rank = Rank.CHAT) -> Stimulus:
    return Stimulus(
        id="s", kind=StimulusKind.CHAT, character="pailin", text="x", created=0.0,
        priority=priority, rank=rank,
    )  # fmt: skip


SPEAKING = BrainSnapshot("speaking", False, "t1", Rank.CHAT, "t1/u1", True, False)
SPEAKING_TAIL = BrainSnapshot("speaking", False, None, None, "t1/u1", True, False)
QUEUED_NOT_AUDIBLE = BrainSnapshot("speaking", False, None, None, "t1/u1", False, False)
DECIDING_QUIET = BrainSnapshot("deciding", False, "t1", Rank.CHAT, "t1/u1", False, False)
IDLE = BrainSnapshot("idle", False, None, None, None, False, False)
PAUSED = BrainSnapshot("paused", False, None, None, None, False, True)


@pytest.mark.parametrize(
    ("priority", "expected"),
    [
        (Priority.LOW, PreemptAction(False, None, False, "queue")),
        (Priority.MEDIUM, PreemptAction(True, "after_segment", False, "preempt_medium")),
        (Priority.HIGH, PreemptAction(True, "after_segment", True, "preempt_high")),
        (Priority.CRITICAL, PreemptAction(True, "now", True, "preempt_critical")),
    ],
)
def test_while_speaking_with_decision_in_flight(
    priority: Priority, expected: PreemptAction
) -> None:
    assert decide_preemption(stim(priority), SPEAKING) == expected


@pytest.mark.parametrize(
    ("priority", "stop", "start_now"),
    [
        (Priority.LOW, None, False),
        (Priority.MEDIUM, "after_segment", False),
        (Priority.HIGH, "after_segment", True),
        (Priority.CRITICAL, "now", True),
    ],
)
@pytest.mark.parametrize("snap", [SPEAKING_TAIL, QUEUED_NOT_AUDIBLE])
def test_while_speaking_after_the_decision_ended(
    priority: Priority, stop: str | None, start_now: bool, snap: BrainSnapshot
) -> None:
    act = decide_preemption(stim(priority), snap)
    assert act.cancel_decision is False  # nothing to cancel
    assert act.speech_stop == stop
    assert act.start_now is start_now


@pytest.mark.parametrize(
    ("priority", "rank", "expected"),
    [
        (Priority.LOW, Rank.OPERATOR, PreemptAction(False, None, False, "merge")),
        (Priority.MEDIUM, Rank.OPERATOR, PreemptAction(False, None, False, "merge")),
        (Priority.HIGH, Rank.VOICE, PreemptAction(True, "now", True, "restart_merged")),
        (Priority.CRITICAL, Rank.IDLE, PreemptAction(True, "now", True, "restart_critical")),
    ],
)
def test_while_deciding_nothing_audible(
    priority: Priority, rank: Rank, expected: PreemptAction
) -> None:
    assert decide_preemption(stim(priority, rank), DECIDING_QUIET) == expected


def test_high_needs_a_strictly_better_rank_to_abort() -> None:
    snap = BrainSnapshot("deciding", False, "t1", Rank.VOICE, "t1/u1", False, False)
    assert decide_preemption(stim(Priority.HIGH, Rank.OPERATOR), snap).cancel_decision
    same = decide_preemption(stim(Priority.HIGH, Rank.VOICE), snap)
    assert same == PreemptAction(False, None, False, "merge")
    worse = decide_preemption(stim(Priority.HIGH, Rank.MENTION), snap)
    assert worse == PreemptAction(False, None, False, "merge")


def test_abort_before_begin_sends_no_stop() -> None:
    snap = BrainSnapshot("deciding", False, "t1", Rank.CHAT, None, False, False)
    act = decide_preemption(stim(Priority.CRITICAL), snap)
    assert act.cancel_decision and act.speech_stop is None


@pytest.mark.parametrize("priority", list(Priority))
def test_not_speaking_all_levels_behave_the_same(priority: Priority) -> None:
    assert decide_preemption(stim(priority), IDLE) == PreemptAction(False, None, False, "queue")


@pytest.mark.parametrize("priority", list(Priority))
def test_paused_never_preempts(priority: Priority) -> None:
    assert decide_preemption(stim(priority), PAUSED).cancel_decision is False
    assert decide_preemption(stim(priority), PAUSED).speech_stop is None
