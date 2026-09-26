"""``CoreControl``: routing, the FREEZE/SKIP/MUTE fast paths, deadlines and op_audit."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from panel_testkit import EditableMemory, FakeOps, RecordingBrain

from aivtube.contracts.control import ControlSurface, OpCommand, OpKind, OpResult
from aivtube.contracts.events import OperatorAction
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.types import ChatMessage, MsgKind
from aivtube.panel.control import BrainControl, CoreControl
from aivtube.testing.fakes import (
    FakeClock,
    FakeEventBus,
    FakeLLM,
    FakeLLMRouter,
    FakeMemoryStore,
    FakeSafetyGate,
    FakeSpeechOutput,
    FakeTaskSupervisor,
    FakeTool,
    FakeToolRegistry,
)


@dataclass
class Rig:
    clock: FakeClock
    bus: FakeEventBus
    speech: FakeSpeechOutput
    brains: dict[str, RecordingBrain]
    router: FakeLLMRouter
    registry: FakeToolRegistry
    memory: EditableMemory
    safety: FakeSafetyGate
    tasks: FakeTaskSupervisor
    ops: FakeOps
    control: CoreControl
    restarts: list[str] = field(default_factory=list)
    ingested: list[ChatMessage] = field(default_factory=list)

    async def run(self, kind: OpKind, args: Mapping[str, Any] | None = None, **kw: Any) -> OpResult:
        result = await self.control.execute(OpCommand(kind, dict(args or {}), **kw))
        await self.clock.run_until_idle()
        return result


def make_rig(
    brains: Mapping[str, RecordingBrain] | None = None,
    *,
    ingest: bool = True,
    handlers: Mapping[OpKind, Any] | None = None,
) -> Rig:
    clock = FakeClock()
    bus = FakeEventBus(clock)
    speech = FakeSpeechOutput(bus, clock)
    router = FakeLLMRouter(
        [FakeLLM([], name="local-30b", clock=clock), FakeLLM([], name="local-4b", clock=clock)],
        clock=clock,
    )
    registry = FakeToolRegistry([FakeTool()])
    memory = EditableMemory("pailin", clock)
    safety = FakeSafetyGate(strict=["pailin"])
    tasks = FakeTaskSupervisor(clock)
    ops = FakeOps()
    brain_map = dict(brains if brains is not None else {"pailin": RecordingBrain()})
    restarts: list[str] = []
    ingested: list[ChatMessage] = []

    async def restart(name: str) -> None:
        restarts.append(name)

    control = CoreControl(
        brain_map,
        router=router,
        registry=registry,
        speech=speech,
        memory_by_char={"pailin": memory},
        safety=safety,
        ops=ops,
        bus=bus,
        clock=clock,
        restart=restart,
        tasks=tasks,
        ingest=ingested.append if ingest else None,
        handlers=handlers,
    )
    return Rig(
        clock, bus, speech, brain_map, router, registry, memory, safety, tasks, ops, control,
        restarts, ingested,
    )  # fmt: skip


def test_core_control_is_a_control_surface() -> None:
    rig = make_rig()
    surface: ControlSurface = rig.control
    assert isinstance(surface, ControlSurface)
    brain: BrainControl = rig.brains["pailin"]
    assert isinstance(brain, BrainControl)


async def test_freeze_reaches_every_brain_and_stops_speech() -> None:
    rig = make_rig({"pailin": RecordingBrain(), "twin": RecordingBrain()})
    result = await rig.run(OpKind.FREEZE, operator="panel")
    assert result.ok, result.detail
    assert all(b.kinds() == [OpKind.FREEZE] for b in rig.brains.values())
    assert ("stop", (None, "now", "operator_freeze", 30)) in rig.speech.calls
    snap = rig.control.snapshot()
    assert snap["frozen"] is True
    assert snap["characters"]["twin"] == {"state": "paused"}
    actions = [e for e in rig.bus.history if isinstance(e, OperatorAction)]
    assert [(a.kind, a.ok) for a in actions] == [("freeze", True)]
    assert rig.ops.rows and rig.ops.rows[0]["command"] == "freeze"
    assert rig.ops.rows[0]["operator"] == "panel" and rig.ops.rows[0]["result"] == "ok"
    resumed = await rig.run(OpKind.RESUME)
    assert resumed.ok and rig.control.snapshot()["frozen"] is False


async def test_freeze_silences_even_when_a_brain_hangs() -> None:
    rig = make_rig({"pailin": RecordingBrain(hang=frozenset({OpKind.FREEZE}))})
    task = asyncio.ensure_future(rig.control.execute(OpCommand(OpKind.FREEZE)))
    await rig.clock.run_until_idle()
    assert ("stop", (None, "now", "operator_freeze", 30)) in rig.speech.calls  # at once
    await rig.clock.run_for(0.6)
    result = await task
    assert not result.ok and "timeout" in result.detail
    assert result.latency_ms == pytest.approx(500.0, abs=1.0)  # the fast-path deadline


async def test_freeze_for_one_character_does_not_stop_everyone() -> None:
    rig = make_rig({"pailin": RecordingBrain(), "twin": RecordingBrain()})
    assert (await rig.run(OpKind.FREEZE, character="twin")).ok
    assert rig.brains["pailin"].commands == []
    assert not any(name == "stop" for name, _ in rig.speech.calls)
    assert rig.control.snapshot()["frozen"] is False


async def test_fast_path_is_not_blocked_by_a_slow_command() -> None:
    rig = make_rig({"pailin": RecordingBrain(hang=frozenset({OpKind.GO_LIVE}))})
    slow = asyncio.ensure_future(rig.control.execute(OpCommand(OpKind.GO_LIVE)))
    await rig.clock.run_until_idle()
    freeze = await rig.control.execute(OpCommand(OpKind.FREEZE))
    assert freeze.ok  # did not wait for GO_LIVE (which holds the command lock)
    queued = asyncio.ensure_future(rig.control.execute(OpCommand(OpKind.CHAT_INTAKE)))
    await rig.clock.run_until_idle()
    assert rig.brains["pailin"].kinds() == [OpKind.GO_LIVE, OpKind.FREEZE]  # serial order
    await rig.clock.run_for(10.5)
    assert not (await slow).ok
    assert (await queued).ok
    assert rig.brains["pailin"].kinds()[-1] is OpKind.CHAT_INTAKE


async def test_approve_does_not_wait_behind_a_slow_command() -> None:
    rig = make_rig({"pailin": RecordingBrain(hang=frozenset({OpKind.GO_LIVE}))})
    slow = asyncio.ensure_future(rig.control.execute(OpCommand(OpKind.GO_LIVE)))
    await rig.clock.run_until_idle()
    approve = await rig.control.execute(OpCommand(OpKind.APPROVE, {"utt_id": "u1"}))
    assert approve.ok and rig.brains["pailin"].kinds() == [OpKind.GO_LIVE, OpKind.APPROVE]
    await rig.clock.run_for(10.5)
    assert not (await slow).ok


async def test_a_cancelled_command_is_still_audited() -> None:
    rig = make_rig({"pailin": RecordingBrain(hang=frozenset({OpKind.GO_LIVE}))})
    task = asyncio.ensure_future(rig.control.execute(OpCommand(OpKind.GO_LIVE, operator="panel")))
    await rig.clock.run_until_idle()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await rig.clock.run_until_idle()
    [row] = rig.ops.rows
    assert (row["command"], row["result"]) == ("go_live", "error: cancelled")
    [action] = rig.bus.of_type(OperatorAction)
    assert action.kind == "go_live" and not action.ok


async def test_skip_goes_to_the_brain_and_falls_back_to_a_direct_stop() -> None:
    rig = make_rig()
    assert (await rig.run(OpKind.SKIP)).ok
    assert rig.brains["pailin"].kinds() == [OpKind.SKIP]
    assert not any(name == "stop" for name, _ in rig.speech.calls)
    lonely = make_rig({})
    assert (await lonely.run(OpKind.SKIP)).ok
    assert ("stop", (None, "now", "operator_skip", 30)) in lonely.speech.calls
    broken = make_rig({"pailin": RecordingBrain(fail=True)})
    assert (await broken.run(OpKind.SKIP)).ok  # the direct stop worked
    assert ("stop", (None, "now", "operator_skip", 30)) in broken.speech.calls


async def test_mute_and_mic_policy_act_on_speech_and_notify_brains() -> None:
    rig = make_rig({"pailin": RecordingBrain(fail=True)})  # brain notifications may fail
    assert (await rig.run(OpKind.MUTE)).ok
    assert rig.speech.muted and rig.control.snapshot()["muted"] is True
    assert (await rig.run(OpKind.UNMUTE)).ok and not rig.speech.muted
    assert (await rig.run(OpKind.MIC_MODE, {"mode": "ptt"})).ok
    assert rig.speech.policy.mic_mode == "ptt"
    assert (await rig.run(OpKind.PTT, {"active": True})).ok
    assert rig.speech.policy.ptt_active and rig.speech.policy.mic_mode == "ptt"
    assert (await rig.run(OpKind.MIC_MODE, {"mode": "deafened"})).ok
    snap = rig.control.snapshot()
    assert snap["mic_mode"] == "deafened" and snap["mic_mode_before_deafen"] == "ptt"
    assert rig.brains["pailin"].kinds() == [
        OpKind.MUTE,
        OpKind.UNMUTE,
        OpKind.MIC_MODE,
        OpKind.PTT,
        OpKind.MIC_MODE,
    ]
    bad = await rig.run(OpKind.MIC_MODE, {"mode": "loud"})
    assert not bad.ok and "mode" in bad.detail


async def test_llm_tools_filters_and_restart_route_to_their_services() -> None:
    rig = make_rig()
    assert (await rig.run(OpKind.LLM_USE, {"name": "local-4b"})).ok
    assert rig.router.active() == "local-4b"
    assert not (await rig.run(OpKind.LLM_USE, {"name": "nope"})).ok
    assert (await rig.run(OpKind.LLM_ROLLBACK)).ok and rig.router.active() == "local-30b"
    assert (await rig.run(OpKind.TOOLS_MODE, {"mode": "dry_run"})).ok
    assert rig.registry.mode == "dry_run"
    assert rig.control.snapshot()["tools"] == {"mode": "dry_run"}
    tool_name = next(iter(rig.registry.enabled))
    assert (await rig.run(OpKind.TOOL_ENABLE, {"name": tool_name, "enabled": False})).ok
    assert rig.registry.enabled[tool_name] is False
    assert not (await rig.run(OpKind.TOOL_ENABLE, {"name": "ghost"})).ok
    assert (await rig.run(OpKind.FILTER_RELOAD)).ok and rig.safety.reloads == 1
    assert (await rig.run(OpKind.RESTART, {"component": "voice"})).ok
    assert rig.restarts == ["voice"]


async def test_memory_commands_round_trip() -> None:
    rig = make_rig()
    item = await rig.memory.remember(
        MemoryItem(id=None, kind="fact", text="ต้นกล้าชอบแมว", status="quarantined")
    )
    assert item.id is not None
    ok = await rig.run(OpKind.MEMORY_STATUS, {"id": item.id, "status": "active"})
    assert ok.ok, ok.detail
    assert (await rig.memory.list_memories(status="active"))[0].id == item.id
    edited = await rig.run(OpKind.MEMORY_EDIT, {"id": item.id, "text": "ต้นกล้าชอบหมา"})
    assert edited.ok
    assert (await rig.memory.list_memories())[0].text == "ต้นกล้าชอบหมา"
    assert (await rig.run(OpKind.MEMORY_EDIT, {"id": item.id, "locked": True})).ok
    locked = await rig.run(OpKind.MEMORY_STATUS, {"id": item.id, "status": "deleted"})
    assert not locked.ok and "locked" in locked.detail
    unsupported = await rig.run(OpKind.MEMORY_EDIT, {"id": item.id, "pinned": True})
    assert not unsupported.ok and "pinned" in unsupported.detail
    missing = await rig.run(OpKind.MEMORY_STATUS, {"id": 999, "status": "active"})
    assert not missing.ok
    nobody = await rig.run(OpKind.MEMORY_STATUS, {"id": 1, "status": "active"}, character="x")
    assert not nobody.ok and "no memory store" in nobody.detail
    audit = [r["command"] for r in rig.ops.rows]
    assert audit.count("memory_status") == 4 and audit.count("memory_edit") == 3


async def test_memory_edit_needs_an_editable_store() -> None:
    rig = make_rig()
    plain = FakeMemoryStore("pailin", rig.clock)
    control = CoreControl(
        rig.brains,
        router=rig.router,
        registry=rig.registry,
        speech=rig.speech,
        memory_by_char={"pailin": plain},
        safety=rig.safety,
        ops=None,
        bus=rig.bus,
        clock=rig.clock,
        restart=lambda name: asyncio.sleep(0),
        tasks=rig.tasks,
    )
    result = await control.execute(OpCommand(OpKind.MEMORY_EDIT, {"id": 1, "text": "x"}))
    assert not result.ok and "edit" in result.detail


async def test_say_goes_to_the_default_character_only() -> None:
    rig = make_rig({"pailin": RecordingBrain(), "twin": RecordingBrain()})
    assert (await rig.run(OpKind.SAY, {"text": "สวัสดีค่ะทุกคน"})).ok
    assert rig.brains["pailin"].kinds() == [OpKind.SAY]
    assert rig.brains["twin"].commands == []
    assert (await rig.run(OpKind.DIRECT, {"text": "พูดเรื่องแมว"}, character="twin")).ok
    assert rig.brains["twin"].kinds() == [OpKind.DIRECT]
    unknown = await rig.run(OpKind.SAY, {"text": "hi"}, character="ghost")
    assert not unknown.ok and "unknown character" in unknown.detail
    empty = await rig.run(OpKind.SAY, {"text": ""})
    assert not empty.ok


async def test_fake_chat_and_inject_event_use_ingest() -> None:
    rig = make_rig()
    assert (await rig.run(OpKind.FAKE_CHAT, {"user": "tom", "text": "ไพลินกินข้าวยัง"})).ok
    donation = {"user": "amy", "amount": 50, "currency": "THB", "text": "ขอบคุณค่ะ"}
    assert (await rig.run(OpKind.INJECT_EVENT, donation)).ok
    assert [m.kind for m in rig.ingested] == [MsgKind.TEXT, MsgKind.DONATION]
    assert rig.brains["pailin"].commands == []
    bad = await rig.run(OpKind.INJECT_EVENT, {"user": ""})
    assert not bad.ok
    without = make_rig(ingest=False)
    assert (await without.run(OpKind.FAKE_CHAT, {"user": "tom", "text": "hi"})).ok
    assert without.brains["pailin"].kinds() == [OpKind.FAKE_CHAT]


async def test_handlers_override_routing_and_unknown_kinds_fail() -> None:
    seen: list[OpCommand] = []

    async def tts_identity(cmd: OpCommand) -> OpResult:
        seen.append(cmd)
        return OpResult(True, "azure first")

    rig = make_rig(handlers={OpKind.TTS_IDENTITY: tts_identity})
    result = await rig.run(OpKind.TTS_IDENTITY, {"identity": "premwadee-azure"})
    assert result.ok and result.detail == "azure first" and len(seen) == 1
    assert rig.brains["pailin"].commands == []
    no_brain = make_rig({})
    assert not (await no_brain.run(OpKind.GO_LIVE)).ok


async def test_execute_never_raises_and_reports_brain_errors() -> None:
    rig = make_rig({"pailin": RecordingBrain(fail=True)})
    result = await rig.run(OpKind.GO_LIVE)
    assert not result.ok and "brain exploded" in result.detail
    assert rig.ops.rows[-1]["result"].startswith("error:")
    assert [e.ok for e in rig.bus.history if isinstance(e, OperatorAction)] == [False]


async def test_snapshot_reports_services_and_survives_failures() -> None:
    rig = make_rig()
    snap = rig.control.snapshot()
    assert snap["default_character"] == "pailin"
    assert snap["llm"]["active"] == "local-30b"
    assert [p["name"] for p in snap["llm"]["providers"]] == ["local-30b", "local-4b"]
    assert snap["tts"]["pailin"]["identity"] == "fake"
    assert snap["strict"] == {"pailin": True}
    assert snap["voice_policy"]["mic_mode"] == "open"

    class BadBrain(RecordingBrain):
        def snapshot(self) -> Mapping[str, Any]:
            raise RuntimeError("no snapshot")

    broken = make_rig({"pailin": BadBrain()})
    assert broken.control.snapshot()["characters"]["pailin"] == {}


async def test_audit_rows_carry_operator_args_and_latency() -> None:
    rig = make_rig()
    await rig.run(OpKind.SAY, {"text": "ทดสอบ"}, character="pailin", operator="console")
    row = rig.ops.rows[-1]
    assert row["operator"] == "console" and row["command"] == "say"
    assert row["args"] == {"text": "ทดสอบ", "character": "pailin"}
    assert row["ts"] == rig.clock.wall() and row["latency_ms"] >= 0
    assert not rig.tasks.errors


async def test_freeze_reaches_speech_stop_within_150_ms_with_real_time(real_clock: Any) -> None:
    """Acceptance: FREEZE -> speech.stop well inside the 150 ms target (in-process)."""
    rig = make_rig()
    control = CoreControl(
        rig.brains,
        router=rig.router,
        registry=rig.registry,
        speech=rig.speech,
        memory_by_char={},
        safety=rig.safety,
        ops=rig.ops,
        bus=rig.bus,
        clock=real_clock,
        restart=lambda name: asyncio.sleep(0),
        tasks=FakeTaskSupervisor(real_clock),
    )
    t0 = real_clock.now()
    result = await control.execute(OpCommand(OpKind.FREEZE))
    elapsed_ms = (real_clock.now() - t0) * 1000.0
    assert result.ok and elapsed_ms < 150.0
    assert any(name == "stop" for name, _ in rig.speech.calls)


def test_voice_policy_is_seeded_from_config() -> None:
    rig = make_rig()
    from aivtube.contracts.speech import VoicePolicy

    control = CoreControl(
        rig.brains,
        router=rig.router,
        registry=rig.registry,
        speech=rig.speech,
        memory_by_char={},
        safety=rig.safety,
        ops=None,
        bus=rig.bus,
        clock=rig.clock,
        restart=lambda name: asyncio.sleep(0),
        tasks=rig.tasks,
        voice_policy=dataclasses.replace(VoicePolicy(), mic_mode="deafened"),
    )
    snap = control.snapshot()
    assert snap["mic_mode"] == "deafened" and snap["mic_mode_before_deafen"] == "open"
