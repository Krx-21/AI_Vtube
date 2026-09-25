"""TurnTraceRecorder, FlightRecorder, MetricsRegistry and PluginRegistry."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts.events import LatencyMark, TurnTraceReady, UserSpeechStarted
from aivtube.infra import (
    STAGES,
    AsyncEventBus,
    FlightRecorder,
    MetricsRegistry,
    PluginRegistry,
    TurnTraceRecorder,
    load_dump,
)


class ManualClock:
    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        return self.t

    def wall(self) -> float:
        return 1.7e9

    async def sleep(self, seconds: float) -> None:
        self.t += seconds
        await asyncio.sleep(0)


# --- TurnTraceRecorder ------------------------------------------------------------------


async def test_voice_turn_trace() -> None:
    clock = ManualClock()
    bus = AsyncEventBus(clock)
    sub = bus.subscribe(TurnTraceReady, name="panel")
    finished: list[Mapping[str, Any]] = []
    rec = TurnTraceRecorder(clock, bus=bus, on_finish=finished.append)
    rec.begin("t1", "voice", "pailin")
    rec.mark("t1", "vad_end", 10.0)
    rec.mark("t1", "stt_final", 10.1)
    rec.mark("t1", "decision_start", 10.11)
    rec.mark("t1", "first_chunk", 10.7)
    rec.mark("t1", "first_chunk", 10.9)  # first mark wins
    rec.mark("t1", "first_audible", 11.4)
    rec.mark("t1", "last_audible", 12.0)
    rec.mark("t1", "last_audible", 13.0)  # latest wins
    rec.set("t1", provider="local-30b", prompt_n=2000, cache_n=1800, fallback="edge->azure")
    rec.set("t1", opener=TurnTraceRecorder.opener("  ว้าว มาแล้ว "))
    clock.t = 13.5
    trace = rec.finish("t1")
    assert trace["ttfa_ms"] == pytest.approx(1400.0)
    assert trace["stages_ms"]["vad_end"] == 0.0
    assert trace["stages_ms"]["first_chunk"] == pytest.approx(700.0)
    assert trace["stages"]["last_audible"] == 13.0 and trace["stages"]["done"] == 13.5
    assert trace["provider"] == "local-30b" and trace["fallbacks"] == ["edge->azure"]
    assert trace["opener"] == "ว้าวมา"  # 6 code points; the tone mark counts as one
    assert trace["outcome"] == "ok" and trace["character"] == "pailin"
    assert finished == [trace]
    (event,) = sub.drain()
    assert isinstance(event, TurnTraceReady) and event.turn_id == "t1"
    assert event.trace["ttfa_ms"] == trace["ttfa_ms"]
    assert rec.recent() == [trace]


def test_chat_turn_measures_from_decision_start() -> None:
    clock = ManualClock()
    rec = TurnTraceRecorder(clock)
    rec.begin("c1", "chat", "pailin")
    rec.mark("c1", "stimulus_in", 1.0)
    rec.mark("c1", "decision_start", 3.0)
    rec.mark("c1", "first_audible", 3.8)
    trace = rec.finish("c1")
    assert trace["ttfa_ms"] == pytest.approx(800.0)


def test_unknown_turns_are_ignored_and_open_traces_are_bounded() -> None:
    rec = TurnTraceRecorder(ManualClock(), max_open=2)
    rec.mark("nope", "done")
    rec.set("nope", provider="x")
    assert rec.finish("nope") == {}
    for i in range(3):
        rec.begin(f"t{i}", "idle", "pailin")
    assert rec.get("t0") is None and rec.get("t2") is not None
    trace = rec.finish("t2")
    assert trace["ttfa_ms"] is None
    assert STAGES[0] == "stimulus_in" and STAGES[-1] == "done"


# --- FlightRecorder ---------------------------------------------------------------------


def test_flight_recorder_is_bounded_and_round_trips(tmp_path: Path) -> None:
    flight = FlightRecorder(maxlen=5, llm_maxlen=2)
    for i in range(8):
        flight.record(LatencyMark(stage=f"s{i}", ts=float(i)))
    flight.record(UserSpeechStarted(barge=True, ts=9.0, character="pailin"))
    for i in range(3):
        flight.record_llm({"provider": "local-30b", "ttft_ms": 400 + i})
    assert len(flight.events()) == 5 and flight.recorded == 9

    path = flight.dump(tmp_path / "dump.json")
    events, llm = load_dump(path)
    assert events == flight.events()
    assert [s["ttft_ms"] for s in llm] == [401, 402]

    folder_dump = flight.dump(tmp_path / "flight")
    assert folder_dump.parent == tmp_path / "flight" and folder_dump.name.startswith("flight-")


async def test_async_dump(tmp_path: Path) -> None:
    flight = FlightRecorder()
    flight.record(LatencyMark(stage="x", ts=1.0))
    path = await flight.adump(tmp_path / "a.json")
    assert load_dump(path)[0] == flight.events()


# --- MetricsRegistry --------------------------------------------------------------------


def test_metrics_counters_and_gauges() -> None:
    m = MetricsRegistry()
    m.inc("underflows")
    m.inc("underflows", 2)
    m.set("tok_s", 41.5)
    assert m.snapshot() == {"underflows": 3.0, "tok_s": 41.5}
    assert m.get("missing", -1.0) == -1.0
    m.reset("tok")
    assert m.snapshot() == {"underflows": 3.0}


def test_metrics_are_thread_safe() -> None:
    m = MetricsRegistry()

    def work() -> None:
        for _ in range(10_000):
            m.inc("n")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.get("n") == 40_000


# --- PluginRegistry ---------------------------------------------------------------------


def test_plugin_registry() -> None:
    reg = PluginRegistry()
    reg.register("tts", "fake", lambda cfg, services: ("fake", cfg["x"], services))
    assert reg.create("tts", "fake", {"x": 1}, "svc") == ("fake", 1, "svc")
    assert reg.names("tts") == ["fake"] and reg.kinds() == ["tts"]
    assert ("tts", "fake") in reg and ("tts", "edge") not in reg
    with pytest.raises(ValueError):
        reg.register("tts", "fake", lambda c, s: None)
    reg.register("tts", "fake", lambda c, s: "replaced", replace=True)
    assert reg.create("tts", "fake", {}, None) == "replaced"
    with pytest.raises(KeyError, match="registered: fake"):
        reg.create("tts", "edge", {}, None)
    with pytest.raises(TypeError):
        reg.register("tts", "x", "not callable")  # type: ignore[arg-type]


class _EntryPoint:
    def __init__(self, name: str, target: Any) -> None:
        self.name = name
        self.value = f"pkg:{name}"
        self._target = target

    def load(self) -> Any:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_entry_points_load_and_failures_are_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def builtin(reg: PluginRegistry) -> None:
        order.append("builtin")
        reg.register("llm", "canned", lambda c, s: "canned")

    def extra(reg: PluginRegistry) -> None:
        order.append("extra")

    points = [
        _EntryPoint("zzz", extra),
        _EntryPoint("broken", ImportError("missing dep")),
        _EntryPoint("builtin", builtin),
    ]

    def fake_entry_points(group: str) -> list[_EntryPoint]:
        assert group == "aivtube.plugins"
        return points

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    reg = PluginRegistry()
    reg.load_entry_points()
    assert order == ["builtin", "extra"]
    assert reg.names("llm") == ["canned"]
    assert reg.loaded == ["builtin", "zzz"]
    assert reg.load_errors == [("broken", "ImportError: missing dep")]
    reg.load_entry_points()  # already-loaded plugins are not registered twice
    assert order == ["builtin", "extra"]
