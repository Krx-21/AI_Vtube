"""Core-side bookkeeping shared by the ``SpeechOutput`` implementations.

``UtteranceBook`` remembers, per utterance, the character, the turn, the segments the core
queued (their captions and emotions are not sent back by the voice worker), the heard text
so far and whether it ended in "Filtered.". ``TurnDirectory`` learns ``utt_id → turn_id``
from ``UtteranceStarted`` events (or ``bind``), so result events carry the turn id without a
change to the ``SpeechOutput`` contract.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aivtube.contracts.events import UtteranceStarted
from aivtube.contracts.infra import EventBus
from aivtube.contracts.speech import TTSConstraints
from aivtube.contracts.types import Segment

__all__ = [
    "DEFAULT_CONSTRAINTS",
    "SegmentCheck",
    "TurnDirectory",
    "UttRecord",
    "UtteranceBook",
    "check_segment",
    "truncate_at_space",
]

log = logging.getLogger("aivtube.speech")

DEFAULT_CONSTRAINTS = TTSConstraints(
    first_min_chars=8, min_chars=40, max_chars=160, backend="unknown", identity="unknown"
)

SegmentCheck = Callable[[Segment], None]
"""Invariant I7 hook: called with every segment before it is sent; tests make it assert that
the segment is gate output (e.g. that its text was returned by ``SafetyGate.check_output``)."""


def check_segment(seg: object, hook: SegmentCheck | None) -> Segment:
    """Invariant I7: only ``Segment`` objects (built from gate output) reach the voice."""
    if not isinstance(seg, Segment):
        raise TypeError(
            f"SpeechOutput.segment() takes a Segment built from gate output (I7), "
            f"not {type(seg).__name__}"
        )
    if hook is not None:
        hook(seg)
    return seg


def truncate_at_space(text: str, n: int) -> str:
    """The first ``n`` characters of ``text``, cut back to the last complete word."""
    if n >= len(text):
        return text
    if n <= 0:
        return ""
    prefix = text[:n]
    if not text[n].isspace() and " " in prefix.strip():
        prefix = prefix[: prefix.rstrip().rfind(" ")]
    return prefix.rstrip()


@dataclass(eq=False)
class UttRecord:
    utt_id: str
    character: str
    began: float
    turn_id: str | None = None
    segments: dict[int, Segment] = field(default_factory=dict)
    heard: dict[int, str] = field(default_factory=dict)
    filtered: bool = False
    stopped: bool = False
    done: bool = False
    link: object | None = None  # the connection it was begun on (BusSpeechOutput)

    def heard_text(self) -> str:
        """Heard texts of the finished segments in order (the chunker keeps the spacing)."""
        return "".join(self.heard[seq] for seq in sorted(self.heard))


class UtteranceBook:
    """Open utterances plus a short memory of finished ones (late or duplicate messages)."""

    def __init__(self, *, keep_done: int = 64) -> None:
        self._open: dict[str, UttRecord] = {}
        self._done: collections.OrderedDict[str, UttRecord] = collections.OrderedDict()
        self._keep = keep_done

    def begin(self, utt_id: str, character: str, now: float) -> UttRecord:
        if utt_id in self._open or utt_id in self._done:
            raise ValueError(f"utterance {utt_id!r} already begun")
        rec = UttRecord(utt_id, character, now)
        self._open[utt_id] = rec
        return rec

    def get(self, utt_id: str) -> UttRecord | None:
        return self._open.get(utt_id) or self._done.get(utt_id)

    def open_records(self, character: str | None = None) -> list[UttRecord]:
        return [r for r in self._open.values() if character is None or r.character == character]

    def close(self, rec: UttRecord) -> bool:
        """Mark ``rec`` finished; ``False`` if it already was."""
        if rec.done:
            return False
        rec.done = True
        self._open.pop(rec.utt_id, None)
        self._done[rec.utt_id] = rec
        while len(self._done) > self._keep:
            self._done.popitem(last=False)
        return True

    def mark_filtered(self, character: str) -> None:
        """``play_canned("filtered")``: the newest open (or just stopped) utterance of the
        character ended in "Filtered."."""
        for rec in reversed(list(self._open.values())):
            if rec.character == character:
                rec.filtered = True
                return
        for rec in reversed(list(self._done.values())):
            if rec.character == character:
                rec.filtered = True
                return


class TurnDirectory:
    """``utt_id → turn_id`` from ``UtteranceStarted`` events on the bus, or ``bind``."""

    def __init__(self, bus: EventBus | None, *, keep: int = 512) -> None:
        self._map: collections.OrderedDict[str, str | None] = collections.OrderedDict()
        self._keep = keep
        self._sub: Any = None
        self._drain: Callable[[], list[Any]] | None = None
        if bus is not None:
            try:
                self._sub = bus.subscribe(UtteranceStarted, name="speech.turns", maxsize=256)
            except Exception:
                log.exception("cannot subscribe to UtteranceStarted; turn ids stay empty")
            else:
                drain = getattr(self._sub, "drain", None)
                self._drain = drain if callable(drain) else None

    def bind(self, utt_id: str, turn_id: str | None) -> None:
        self._map[utt_id] = turn_id
        self._map.move_to_end(utt_id)
        while len(self._map) > self._keep:
            self._map.popitem(last=False)

    def lookup(self, utt_id: str) -> str | None:
        if self._drain is not None:
            for ev in self._drain():
                if isinstance(ev, UtteranceStarted):
                    self.bind(ev.utt_id, ev.turn_id)
        return self._map.get(utt_id)

    def close(self) -> None:
        if self._sub is not None:
            self._sub.close()
            self._sub = None
            self._drain = None
