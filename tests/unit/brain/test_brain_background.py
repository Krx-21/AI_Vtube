"""BackgroundJobs: compaction + epoch flip on the inactive slot, slot restore, episodes (§4.8, §6)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from brain_testkit import character, run_real, stim

from aivtube.brain.background import EPISODE_JOB, BackgroundJobs, parse_summary
from aivtube.brain.history import History
from aivtube.brain.prompt import PromptBuilder
from aivtube.contracts.llm import ChatRequest, LLMEvent, ProviderStatus
from aivtube.contracts.types import StimulusKind
from aivtube.memory import OpsDb, SqliteMemory
from aivtube.testing.fakes import (
    FakeClock,
    FakeLauncher,
    FakeLLM,
    FakeLLMRouter,
    FakeMemoryStore,
    FakeReply,
    FakeTool,
    FakeToolRegistry,
)
from aivtube.text import estimate_tokens

SUMMARY = '{"summary": "สตรีมเมอร์เล่าเรื่องแมวและเกม"}'


class SlowPrefillRouter:
    """``LLMRouter`` over a FakeLLMRouter whose ``prefill`` takes ``prefill_s`` clock seconds."""

    def __init__(self, inner: FakeLLMRouter, clock: FakeClock, prefill_s: float = 0.0) -> None:
        self.inner = inner
        self.clock = clock
        self.prefill_s = prefill_s
        self.prefills: list[ChatRequest] = []
        self.completed: list[ChatRequest] = []
        self.fail_prefill = False

    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        return self.inner.stream(req)

    async def prefill(self, req: ChatRequest) -> None:
        self.prefills.append(req)
        if self.fail_prefill:
            raise ConnectionError("llama-server is down")
        if self.prefill_s:
            await self.clock.sleep(self.prefill_s)
        self.completed.append(req)

    def promote(self, name: str) -> None:
        self.inner.promote(name)

    def rollback(self) -> None:
        self.inner.rollback()

    def active(self) -> str:
        return self.inner.active()

    def status(self) -> list[ProviderStatus]:
        return self.inner.status()


class Rig:
    def __init__(
        self,
        clock: FakeClock,
        *,
        llm: FakeLLM | None = None,
        prefill_s: float = 0.0,
        budget: int = 5000,
        memory: Any = None,
        ops: Any = None,
        launcher: FakeLauncher | None = None,
    ) -> None:
        self.clock = clock
        self.busy = False
        self.memory = memory or FakeMemoryStore(clock=clock)
        self.llm = llm or FakeLLM([FakeReply("", SUMMARY, repeat=True)], clock=clock, ttft_s=0.1)
        self.router = SlowPrefillRouter(FakeLLMRouter([self.llm]), clock, prefill_s)
        self.prompt = PromptBuilder(character(), "เธอคือไพลิน", FakeToolRegistry([FakeTool()]))
        self.history = History(
            self.memory, budget_tokens=budget, estimate=estimate_tokens, clock=clock
        )
        self.launcher = launcher
        self.jobs = BackgroundJobs(
            character="pailin",
            router=self.router,
            memory=self.memory,
            ops=ops,
            prompt=self.prompt,
            history=self.history,
            servers=launcher,
            clock=clock,
            busy=lambda: self.busy,
            server="local30b" if launcher is not None else None,
            retry_s=5.0,
        )

    async def talk(self, n: int) -> list[int]:
        ids: list[int] = []
        for i in range(n):
            ids.append(
                await self.history.append_user(
                    f"[สตรีมเมอร์] เรื่องที่ {i} " + "ก" * 40, stim(StimulusKind.VOICE)
                )
            )
            ids.append(
                await self.history.append_assistant(
                    turn_ref=f"u{i}",
                    emitted=f"คำตอบที่ {i} " + "ข" * 40,
                    heard=None,
                    interrupted=False,
                    filtered=False,
                    provider="fake",
                    tool_calls=None,
                    provider_extra=None,
                )
            )
        return ids


def test_parse_summary() -> None:
    assert parse_summary('{"summary": "  สรุป  สั้นๆ "}') == "สรุป สั้นๆ"
    assert parse_summary('นี่คือ {"summary": "ก"} ค่ะ') == "ก"
    assert parse_summary("ข้อความธรรมดา") == "ข้อความธรรมดา"
    with pytest.raises(ValueError):
        parse_summary('{"other": 1}')
    with pytest.raises(ValueError):
        parse_summary("   ")


async def test_flip_only_after_prewarm_and_decisions_cancel_it(fake_clock: FakeClock) -> None:
    rig = Rig(fake_clock, prefill_s=2.0, budget=100)
    await rig.memory.start_session()
    ids = await rig.talk(6)
    epoch0 = await rig.jobs.load()
    assert epoch0.slot == 0
    rig.jobs.warm = True  # skip the start-up warm-up in this test
    rig.jobs.request_epoch_rebuild("history")
    task = asyncio.ensure_future(rig.jobs.run())
    try:
        # the compaction summary runs on the background slot, then the prewarm starts
        await fake_clock.run_until(lambda: len(rig.router.prefills) == 1)
        assert rig.llm.requests[0].purpose == "background"
        assert rig.llm.requests[0].slot == 2 and rig.llm.requests[0].response_schema is not None
        prewarm = rig.router.prefills[0]
        assert prewarm.slot == 1 and prewarm.max_tokens == 1  # the INACTIVE speak slot
        # a decision starts mid-prewarm: cancelled, the old epoch stays
        rig.busy = True
        rig.jobs.cancel_active()
        await fake_clock.run_for(10.0)
        assert rig.jobs.current_epoch() == epoch0 and rig.router.completed == []
        assert len(rig.router.prefills) == 1  # nothing runs while DECIDING
        rig.busy = False
        await fake_clock.run_until(lambda: rig.jobs.flips == 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    flipped = rig.jobs.current_epoch()
    assert flipped.slot == 1 and flipped.id != epoch0.id
    assert len(rig.llm.requests) == 1  # the summary was kept across the interruption
    assert "สตรีมเมอร์เล่าเรื่องแมวและเกม" in flipped.messages[1]["content"]
    warmed = rig.router.completed[-1]  # the flip follows a successful prewarm of this prefix
    assert warmed.messages == flipped.messages and warmed.slot == flipped.slot
    prefix = await rig.memory.prefix_block()
    assert prefix.rolling_summary == "สตรีมเมอร์เล่าเรื่องแมวและเกม"
    kept = [t.id for t in rig.history.turns()]
    assert kept and kept[0] > ids[0] and kept == ids[len(ids) - len(kept) :]
    assert kept[0] - 1 == rig.memory.state.epochs[-1].upto_turn_id
    # the flip opened a new epoch in the store: recent_turns matches the rebased history
    assert [t.id for t in await rig.memory.recent_turns(flipped.id)] == kept


async def test_memory_rebuild_without_compaction(fake_clock: FakeClock) -> None:
    rig = Rig(fake_clock)
    await rig.memory.start_session()
    ids = await rig.talk(2)
    epoch0 = await rig.jobs.load()
    rig.jobs.warm = True
    from aivtube.contracts.memory import MemoryItem

    await rig.memory.remember(MemoryItem(id=None, kind="core", text="สตรีมเมอร์ชื่อบอส"))
    rig.jobs.request_epoch_rebuild("memory")
    task = asyncio.ensure_future(rig.jobs.run())
    try:
        await fake_clock.run_until(lambda: rig.jobs.flips == 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert rig.llm.requests == []  # no compaction needed
    epoch = rig.jobs.current_epoch()
    assert epoch.slot == 1 and "สตรีมเมอร์ชื่อบอส" in epoch.messages[1]["content"]
    assert [t.id for t in rig.history.turns()] == ids  # nothing dropped
    assert await rig.memory.pending_since_epoch() == []  # <new_memories> resets at the flip
    assert epoch0.prefix_hash != epoch.prefix_hash


async def test_restore_slot_when_the_file_exists(fake_clock: FakeClock) -> None:
    launcher = FakeLauncher(["local30b"], clock=fake_clock)
    await launcher.ensure_running("local30b", 5.0)
    memory = FakeMemoryStore(clock=fake_clock)
    await memory.start_session()
    first = Rig(fake_clock, memory=memory, launcher=launcher)
    await first.jobs.load()
    await first.jobs.restore_or_prewarm()  # no file yet: prewarm with max_tokens=1, then save
    assert [r.max_tokens for r in first.router.prefills] == [1]
    epoch = first.jobs.current_epoch()
    name = f"pailin-{epoch.prefix_hash}.bin"
    assert ("save_slot", ("local30b", 0, name)) in launcher.calls
    # the next start finds the file: restore, no prewarm
    second = Rig(fake_clock, memory=memory, launcher=launcher)
    await second.jobs.load()
    await second.jobs.restore_or_prewarm()
    assert second.router.prefills == [] and second.jobs.warm
    assert ("restore_slot", ("local30b", 0, name)) in launcher.calls


async def test_episode_job_waits_for_the_llm_and_runs_at_next_start(
    fake_clock: FakeClock, tmp_path: Any
) -> None:
    ops = OpsDb(tmp_path / "ops.db", clock=fake_clock, job_backoff_s=(1.0, 10.0))
    memory = SqliteMemory(tmp_path / "pailin.sqlite", "pailin", fake_clock, fts=False)
    try:
        await ops.start()
        await memory.start_session()
        down = Rig(
            fake_clock, memory=memory, ops=ops, llm=FakeLLM([], fail="connect", clock=fake_clock)
        )
        await down.talk(2)
        await down.jobs.end_session()  # the LLM is down: the job stays queued
        jobs = await ops.jobs()
        assert [(j["kind"], j["state"]) for j in jobs] == [(EPISODE_JOB, "pending")]
        assert await memory.list_memories(kind="episode") == []
        # next start: a working LLM; jobs run only while not DECIDING
        await memory.start_session()
        fake_clock.advance(5.0)  # past the retry backoff (wall clock follows the fake clock)
        nxt = Rig(fake_clock, memory=memory, ops=ops)
        nxt.busy = True
        run = asyncio.ensure_future(nxt.jobs.run_due_jobs())
        with pytest.raises(TimeoutError):  # DECIDING: nothing runs
            await run_real(fake_clock, run.done, within=3.0)
        assert nxt.llm.requests == []
        nxt.busy = False
        await run_real(fake_clock, run.done)
        assert run.result() == 1
        episodes = await memory.list_memories(kind="episode")
        assert [m.text for m in episodes] == ["สตรีมเมอร์เล่าเรื่องแมวและเกม"]
        assert episodes[0].source == "consolidation" and episodes[0].status == "active"
        assert "เรื่องที่ 1" in nxt.llm.requests[0].messages[-1]["content"]  # the old session
        assert [j["state"] for j in await ops.jobs()] == ["done"]
    finally:
        await ops.aclose()
        await memory.aclose()


async def test_end_session_stores_the_episode_now(fake_clock: FakeClock) -> None:
    memory = FakeMemoryStore(clock=fake_clock)
    await memory.start_session()
    rig = Rig(fake_clock, memory=memory)
    await rig.talk(1)
    task = asyncio.ensure_future(rig.jobs.end_session())
    await fake_clock.run_until(task.done)
    task.result()
    assert rig.llm.requests[0].max_tokens <= 300 and rig.llm.requests[0].purpose == "background"
    assert any(m.kind == "episode" for m in memory.state.memories.values())
