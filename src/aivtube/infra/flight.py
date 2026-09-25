"""``FlightRecorder`` (§2.10): the last 2000 events and 50 LLM request summaries, dumped on
any error, crash or watchdog trip. ``aivtube replay`` reads a dump back with ``load_dump``."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aivtube.contracts.events import Event, event_from_json, event_to_json

__all__ = ["FlightRecorder", "load_dump"]

log = logging.getLogger("aivtube.flight")

DUMP_VERSION = 1


class FlightRecorder:
    """Ring buffers of recent events and LLM summaries. ``record`` is safe from any thread."""

    def __init__(self, maxlen: int = 2000, *, llm_maxlen: int = 50) -> None:
        self._events: collections.deque[Event] = collections.deque(maxlen=maxlen)
        self._llm: collections.deque[dict[str, Any]] = collections.deque(maxlen=llm_maxlen)
        self.recorded = 0

    def record(self, ev: Event) -> None:
        self._events.append(ev)
        self.recorded += 1

    def record_llm(self, summary: Mapping[str, Any]) -> None:
        self._llm.append(dict(summary))

    def events(self) -> list[Event]:
        return list(self._events)

    def snapshot(self) -> dict[str, Any]:
        """The dump content as JSON-safe data."""
        return _render(list(self._events), list(self._llm), self.recorded)

    def dump(self, path: Path) -> Path:
        """Write a JSON dump to ``path`` (a file, or a folder for ``flight-<time>.json``).

        Blocking file I/O: from the event loop prefer ``await adump(path)``.
        """
        return _write(Path(path), self.snapshot())

    async def adump(self, path: Path) -> Path:
        """``dump`` with serialisation and file I/O on a worker thread."""
        events, llm, recorded = list(self._events), list(self._llm), self.recorded
        return await asyncio.to_thread(lambda: _write(Path(path), _render(events, llm, recorded)))


def _render(events_in: list[Event], llm: list[dict[str, Any]], recorded: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for ev in events_in:
        try:
            events.append(event_to_json(ev))
        except Exception as exc:  # never let one odd payload lose the whole dump
            events.append({"type": type(ev).__name__, "unserializable": repr(exc)})
    return {
        "version": DUMP_VERSION,
        "dumped_at": time.time(),
        "dumped_perf": time.perf_counter(),
        "pid": os.getpid(),
        "recorded": recorded,
        "events": events,
        "llm": llm,
    }


def _write(path: Path, data: dict[str, Any]) -> Path:
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = path / f"flight-{stamp}-{os.getpid()}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    log.warning("flight recorder dumped %d events to %s", len(data["events"]), path)
    return path


def load_dump(path: Path) -> tuple[list[Event], list[dict[str, Any]]]:
    """Read a dump: the events that decode (others are skipped) and the LLM summaries."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events: list[Event] = []
    for raw in data.get("events", []):
        try:
            events.append(event_from_json(raw))
        except ValueError:
            log.debug("skipping undecodable event %r", raw.get("type"))
    return events, list(data.get("llm", []))
