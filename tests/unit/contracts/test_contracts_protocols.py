"""Protocols and domain value types (§3.3–§3.13): runtime checks, defaults, exceptions."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import pickle
import typing
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

import aivtube.contracts as contracts
from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.events import Event
from aivtube.contracts.games import GameAction, GameForce
from aivtube.contracts.infra import Clock, EventBus, Overflow, Subscription, TaskSupervisor
from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    ProviderFailed,
    TextDelta,
    ToolCall,
)
from aivtube.contracts.memory import MemoryItem, SlotsFull, Turn
from aivtube.contracts.safety import FilterContext, FilterResult, TextFilter, Verdict
from aivtube.contracts.speech import SpeechOutput, TTSConstraints, VoicePolicy
from aivtube.contracts.tools import ToolPolicy, ToolResult
from aivtube.contracts.types import Health, HealthState, Priority, Segment
from aivtube.contracts.voice import (
    AudioChunk,
    EndpointerConfig,
    TTSUnavailable,
    VadEnd,
    VadEvent,
    VadPartial,
    VadStart,
)

MODULES = [
    "infra",
    "voice",
    "speech",
    "avatar",
    "llm",
    "chat",
    "memory",
    "safety",
    "tools",
    "control",
    "games",
]


def _protocols() -> list[type]:
    found = []
    for name in MODULES:
        mod = importlib.import_module(f"aivtube.contracts.{name}")
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == mod.__name__
                and getattr(obj, "_is_protocol", False)
            ):
                found.append(obj)
    return found


def test_every_protocol_is_runtime_checkable() -> None:
    protos = _protocols()
    assert len(protos) >= 30
    for proto in protos:
        assert getattr(proto, "_is_runtime_protocol", False), proto


class _Clock:
    def now(self) -> float:
        return 1.0

    def wall(self) -> float:
        return 2.0

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)


class _Sub:
    name = "s"
    dropped = 0

    def __aiter__(self) -> AsyncIterator[Event]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _Bus:
    def publish(self, event: Event) -> None:
        pass

    def publish_threadsafe(self, event: Event) -> None:
        pass

    def subscribe(
        self,
        *types: type[Event],
        name: str,
        maxsize: int = 1024,
        overflow: Overflow = Overflow.DROP_OLDEST,
    ) -> Subscription:
        return _Sub()


class _Supervisor:
    def spawn(self, name: str, factory: Callable[[], Awaitable[None]], **kw: Any) -> None:
        pass

    def track(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
        raise NotImplementedError

    async def restart(self, name: str) -> None:
        pass

    def status(self) -> list[Health]:
        return []

    async def aclose(self, timeout: float = 5.0) -> None:
        pass


class _Filter:
    name = "t0"

    def check(self, text: str, ctx: FilterContext) -> FilterResult:
        return FilterResult(Verdict.PASS, text, "t0")

    def reload(self) -> None:
        pass


def test_structural_fakes_satisfy_protocols() -> None:
    clock: Clock = _Clock()
    bus: EventBus = _Bus()
    filt: TextFilter = _Filter()
    assert isinstance(clock, Clock)
    assert isinstance(bus, EventBus)
    assert isinstance(_Sub(), Subscription)
    assert isinstance(_Supervisor(), TaskSupervisor)
    assert isinstance(filt, TextFilter)
    assert not isinstance(_Filter(), EventBus)
    assert not isinstance(object(), Clock)
    assert not isinstance(clock, SpeechOutput)


def test_provider_failed_carries_emitted_and_pickles() -> None:
    exc = ProviderFailed("stall", emitted=True)
    assert exc.emitted is True
    assert str(exc) == "stall"
    back = pickle.loads(pickle.dumps(exc))
    assert isinstance(back, ProviderFailed)
    assert (str(back), back.emitted) == ("stall", True)
    with pytest.raises(TypeError):
        ProviderFailed("x", True)  # type: ignore[misc]


def test_slots_full_carries_slots_and_pickles() -> None:
    items = [MemoryItem(id=i, kind="core", text=f"fact {i}", slot=i) for i in range(3)]
    exc = SlotsFull(items)
    assert exc.slots == items
    back = pickle.loads(pickle.dumps(exc))
    assert isinstance(back, SlotsFull)
    assert back.slots == items


def test_tts_unavailable_is_an_exception() -> None:
    assert issubclass(TTSUnavailable, Exception)


def test_endpointer_defaults_match_the_reference_config() -> None:
    cfg = EndpointerConfig()
    assert dataclasses.asdict(cfg) == {
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "barge_threshold": 0.6,
        "min_speech_ms": 250,
        "end_silence_ms": 600,
        "preroll_ms": 300,
        "max_segment_s": 15.0,
        "max_turn_s": 60.0,
        "particle_endpointing": False,
        "final_particle_ms": 420,
        "continuation_ms": 900,
    }


def test_array_carrying_dataclasses_use_identity_equality() -> None:
    audio = np.zeros(512, dtype=np.float32)
    a, b = VadEnd(1.0, audio), VadEnd(1.0, audio)
    assert a == a
    assert a != b  # element-wise array comparison would raise inside a generated __eq__
    part = VadPartial(1.0, audio)
    assert part in [VadEnd(1.0, audio), part]  # membership tests must not raise
    chunk = AudioChunk(np.zeros(10, dtype=np.int16), 24000)
    assert {chunk: 1}[chunk] == 1  # hashable by identity
    assert VadStart(1.0, True) == VadStart(1.0, True)
    assert set(get_args(VadEvent)) == {VadStart, VadPartial, VadEnd}


def test_llm_event_union_and_defaults() -> None:
    assert set(get_args(LLMEvent)) == {TextDelta, ToolCall, Done}
    req = ChatRequest(messages=({"role": "user", "content": "สวัสดี"},))
    assert (req.purpose, req.tools, req.tool_choice, req.response_schema) == (
        "speak",
        (),
        "auto",
        None,
    )
    assert (req.max_tokens, req.temperature, req.character, req.turn_id) == (256, 0.6, "", "")
    assert (req.slot, req.first_token_timeout_s) == (None, None)


def test_speech_defaults() -> None:
    p = VoicePolicy()
    assert (p.mic_mode, p.ptt_active, p.barge_in, p.echo_mode, p.listening) == (
        "open",
        False,
        "interrupt",
        "auto",
        True,
    )
    c = TTSConstraints(8, 40, 160, "edge", "premwadee")
    assert hash(c) == hash(dataclasses.replace(c))


def test_domain_defaults() -> None:
    m = MemoryItem(id=None, kind="fact", text="x")
    assert (m.slot, m.subject, m.platform, m.user_id, m.importance) == (None, None, None, None, 3)
    assert (m.source, m.origin, m.status, m.pinned, m.locked) == (
        "model",
        "",
        "active",
        False,
        False,
    )
    t = Turn(role="assistant", text="ค่ะ", source="voice")
    assert (t.speaker, t.heard_text, t.interrupted, t.filtered, t.provider) == (
        None,
        None,
        False,
        False,
        None,
    )
    assert (t.turn_ref, t.tool_calls, t.provider_extra, t.ts, t.id) == ("", None, None, 0.0, None)
    pol = ToolPolicy()
    assert (pol.risk, pol.side_effect, pol.follow_up, pol.requires_approval) == (
        "safe",
        False,
        False,
        False,
    )
    assert (pol.rate_limit, pol.timeout_s, pol.requires) == (None, 5.0, frozenset())
    assert ToolResult(True, "ok").note is None
    cmd = OpCommand(OpKind.FREEZE)
    assert (cmd.args, cmd.character, cmd.operator, cmd.id) == ({}, None, "local", "")
    assert OpResult(True) == OpResult(True, "", 0.0)
    ctx = FilterContext(direction="out", character="pailin")
    assert (ctx.platform, ctx.user_id, ctx.prev_tail) == (None, None, "")
    res = FilterResult(Verdict.BLOCK, "", "t0")
    assert (res.rule, res.category, res.score, res.fail_closed) == (None, None, None, False)
    force = GameForce("f1", "g", "your turn", None, ("play",), Priority.HIGH, False)
    assert force.attempt == 0
    assert hash(force)
    assert GameAction("g", "play", "Play a cell", None).schema is None


def test_segment_kind_literal_is_shared() -> None:
    hints = typing.get_type_hints(Segment)
    assert get_args(hints["kind"]) == ("speech", "filtered", "filler", "operator", "canned")


def test_memory_backup_signature_uses_path() -> None:
    from aivtube.contracts.memory import MemoryStore

    hints = typing.get_type_hints(MemoryStore.backup)
    assert hints["dest_dir"] is Path
    assert hints["return"] is Path


def test_health_state_values_round_trip() -> None:
    assert Health("x", HealthState("ok")).state is HealthState.OK
    assert contracts.HealthState is HealthState
