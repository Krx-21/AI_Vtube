"""The pure priority/interrupt table (ARCHITECTURE.md §4.4, §4.3 exceptions a and b).

No I/O and no clock. ``Brain.submit`` calls :func:`decide_preemption` for every new stimulus
and applies the returned :class:`PreemptAction`.

| Priority | While SPEAKING | While DECIDING, nothing audible yet |
|---|---|---|
| LOW | queue; decide after ``UtteranceDone`` | merge into the next decision |
| MEDIUM | cancel the LLM, ``stop(after_segment)``, then decide | merge |
| HIGH | as MEDIUM, but start the next decision now | abort and restart merged, only with a strictly better rank |
| CRITICAL | ``stop(now)``, cancel the LLM, decide at once | abort and restart |

"Speaking" means a segment is audible, or an utterance is still open after its decision ended
(queued speech that has not started yet counts: ``after_segment`` drops it). When she is
neither speaking nor deciding, all four levels behave the same: the stimulus is just queued.
"""

from __future__ import annotations

from dataclasses import dataclass

from aivtube.contracts.speech import StopMode
from aivtube.contracts.types import Priority, Rank, Stimulus

__all__ = ["BrainSnapshot", "PreemptAction", "decide_preemption"]


@dataclass(frozen=True, slots=True)
class BrainSnapshot:
    """What the preemption table needs to know about the brain right now."""

    state: str
    user_speaking: bool
    decision_turn: str | None  # the decision in flight, if any
    decision_rank: Rank | None  # rank of that decision's primary stimulus
    utterance: str | None  # the open utterance (begun, no UtteranceDone yet)
    audible: bool  # a segment of the open utterance has started playing
    paused: bool


@dataclass(frozen=True, slots=True)
class PreemptAction:
    """``cancel_decision``: cancel the decision in flight. ``speech_stop``: stop the open
    utterance with this mode. ``start_now``: the stimulus may start a decision while audio of
    the previous utterance still plays. ``reason`` is logged and becomes the abort reason."""

    cancel_decision: bool
    speech_stop: StopMode | None
    start_now: bool
    reason: str


def decide_preemption(incoming: Stimulus, snap: BrainSnapshot) -> PreemptAction:
    """Apply the §4.4 table (plus §4.3 a/b) to ``incoming`` given the brain snapshot."""
    if snap.paused or snap.state in ("paused", "booting"):
        return PreemptAction(False, None, False, "paused")
    p = incoming.priority
    deciding = snap.decision_turn is not None
    speaking = snap.audible or (snap.utterance is not None and not deciding)
    if speaking:
        if p is Priority.CRITICAL:
            return PreemptAction(deciding, "now", True, "preempt_critical")
        if p is Priority.HIGH:
            return PreemptAction(deciding, "after_segment", True, "preempt_high")
        if p is Priority.MEDIUM:
            return PreemptAction(deciding, "after_segment", False, "preempt_medium")
        return PreemptAction(False, None, False, "queue")
    if deciding:
        quiet_stop: StopMode | None = "now" if snap.utterance is not None else None
        if p is Priority.CRITICAL:
            return PreemptAction(True, quiet_stop, True, "restart_critical")
        better = snap.decision_rank is not None and incoming.rank < snap.decision_rank
        if p is Priority.HIGH and better:
            return PreemptAction(True, quiet_stop, True, "restart_merged")
        return PreemptAction(False, None, False, "merge")
    return PreemptAction(False, None, False, "queue")
