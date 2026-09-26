"""``OpsDb``: traces, moderation, tool/op audit and jobs in ``data/ops.db`` (§6)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aivtube.contracts.events import HealthChanged
from aivtube.contracts.types import HealthState
from aivtube.infra import TurnTraceRecorder
from aivtube.memory import OpsDb, list_backups
from aivtube.testing.fakes import FakeClock, FakeEventBus, FakeTaskSupervisor


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def ops(tmp_path: Path, clock: FakeClock) -> AsyncIterator[OpsDb]:
    db = OpsDb(tmp_path / "ops.db", clock=clock)
    yield db
    await db.aclose()


async def test_schema_matches_architecture(ops: OpsDb, tmp_path: Path) -> None:
    await ops.start()
    conn = sqlite3.connect(tmp_path / "ops.db")
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tool_audit)")]
    finally:
        conn.close()
    assert tables == {"turn_trace", "moderation_log", "tool_audit", "op_audit", "job"}
    assert cols == [
        "id",
        "ts",
        "character",
        "turn_id",
        "tool",
        "args",
        "verdict",
        "result",
        "approved_by",
        "dry_run",
    ]
    assert ops.health().state is HealthState.OK


async def test_traces_from_the_recorder(ops: OpsDb, clock: FakeClock) -> None:
    rec = TurnTraceRecorder(clock)
    for i in range(3):
        rec.begin(f"t{i}", "voice", "pailin")
        rec.mark(f"t{i}", "vad_end")
        clock.advance(0.4)
        rec.mark(f"t{i}", "first_audible")
        rec.set(f"t{i}", provider="local-30b", prompt_n=900, cache_n=850, speculative=False)
        await ops.add_trace(rec.finish(f"t{i}"))
    await ops.add_trace({"no": "turn id"})  # ignored, never raises
    traces = await ops.recent_traces(2)
    assert [t["turn_id"] for t in traces] == ["t1", "t2"]
    t = traces[-1]
    assert t["ttfa_ms"] == pytest.approx(400.0)
    assert t["stages"]["first_audible"] == pytest.approx(400.0)
    assert (t["provider"], t["cache_n"], t["speculative"], t["character"]) == (
        "local-30b",
        850,
        0,
        "pailin",
    )
    await ops.add_trace({"turn_id": "t2", "stages": {"done": 1.0}, "outcome": "aborted"})
    latest = (await ops.recent_traces(1))[0]
    assert (latest["turn_id"], latest["outcome"]) == ("t2", "aborted")
    assert len(await ops.recent_traces(10)) == 3


async def test_audit_rows(ops: OpsDb, clock: FakeClock) -> None:
    await ops.log_tool(
        character="pailin",
        turn_id="t1",
        tool="remember",
        args={"text": "แมว"},
        verdict="ok",
        result='{"ok": true}',
        approved_by=None,
        dry_run=False,
        bogus=1,
    )
    await ops.log_op(operator="local", command="freeze", args={}, result="ok", latency_ms=3.5)
    await ops.log_moderation(
        character="pailin",
        direction="out",
        source="llm",
        tier="tier0",
        category="slur",
        rule="r1",
        verdict="block",
        text_masked="[masked]",
        text_sha256="ab" * 32,
        author=None,
    )
    tools = await ops.audit_rows("tool_audit")
    assert len(tools) == 1
    row = tools[0]
    assert json.loads(row["args"]) == {"text": "แมว"} and row["dry_run"] == 0
    assert row["ts"] == pytest.approx(clock.wall())
    ops_rows = await ops.audit_rows("op_audit")
    assert ops_rows[0]["command"] == "freeze" and ops_rows[0]["latency_ms"] == 3.5
    assert await ops.audit_rows("tool_audit", since_id=row["id"]) == []
    with pytest.raises(ValueError):
        await ops.audit_rows("turn_trace")  # type: ignore[arg-type]
    clock.advance(31 * 86400)
    await ops.log_moderation(character="pailin", verdict="drop", text_masked="x")
    assert await ops.purge_moderation() == 1
    assert len(await ops.audit_rows("moderation_log")) == 1


async def test_audit_after_close_is_dropped_not_raised(tmp_path: Path) -> None:
    db = OpsDb(tmp_path / "ops.db")
    await db.start()
    await db.aclose()
    await db.log_tool(tool="x", verdict="ok")
    await db.add_trace({"turn_id": "x", "stages": {}})
    await db.aclose()  # idempotent


async def test_jobs_claim_retry_and_requeue(tmp_path: Path, clock: FakeClock) -> None:
    db = OpsDb(tmp_path / "ops.db", clock=clock, max_job_attempts=3, job_backoff_s=(10.0, 15.0))
    try:
        now = clock.wall()
        a = await db.enqueue_job("episode", {"session_id": 1})
        b = await db.enqueue_job("dedupe", {"x": [1, 2]}, run_at=now + 100)
        due = await db.due_jobs(now)
        assert [(j["id"], j["kind"], j["payload"]) for j in due] == [
            (a, "episode", {"session_id": 1})
        ]
        assert await db.due_jobs(now) == []  # claimed
        await db.finish_job(a, ok=False, error="llm down")
        job = (await db.jobs(state="pending"))[0]
        assert (job["id"], job["attempts"], job["last_error"]) == (a, 1, "llm down")
        assert job["next_run_at"] == pytest.approx(now + 10.0)
        clock.advance(11)
        assert [j["id"] for j in await db.due_jobs(clock.wall())] == [a]
        await db.finish_job(a, ok=False)
        clock.advance(16)
        assert [j["id"] for j in await db.due_jobs(clock.wall())] == [a]
        await db.finish_job(a, ok=False)
        assert [j["id"] for j in await db.jobs(state="failed")] == [a]
        clock.advance(100)
        assert [j["id"] for j in await db.due_jobs(clock.wall())] == [b]
        with pytest.raises(KeyError):
            await db.finish_job(999, ok=True)
    finally:
        await db.aclose()
    # b was claimed when the process "died": the next open re-queues it
    again = OpsDb(tmp_path / "ops.db", clock=clock)
    try:
        assert [j["id"] for j in await again.due_jobs(clock.wall())] == [b]
        await again.finish_job(b, ok=True)
        assert [j["state"] for j in await again.jobs()] == ["failed", "done"]
    finally:
        await again.aclose()


async def test_corrupt_ops_db_degrades(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ops.db"
    path.write_bytes(b"garbage" * 1000)
    bus = FakeEventBus(clock)
    db = OpsDb(path, clock=clock, bus=bus)
    try:
        await db.log_tool(tool="remember", verdict="ok")
        assert db.degraded and db.health().state is HealthState.DEGRADED
        assert [e.health.component for e in bus.of_type(HealthChanged)] == ["ops_db"]
        assert len(await db.audit_rows("tool_audit")) == 1
    finally:
        await db.aclose()


async def test_ops_backup(ops: OpsDb, tmp_path: Path) -> None:
    await ops.log_op(command="x")
    out = await ops.backup(tmp_path / "bk", keep=1)
    await ops.backup(tmp_path / "bk", keep=1)
    assert not out.exists() or len(list_backups(tmp_path / "bk", "ops")) == 1
    assert len(list_backups(tmp_path / "bk", "ops")) == 1


async def test_trace_sink_writes_in_the_background(ops: OpsDb, clock: FakeClock) -> None:
    tasks = FakeTaskSupervisor(clock)
    rec = TurnTraceRecorder(clock, on_finish=ops.trace_sink(tasks))
    rec.begin("t9", "chat", "pailin")
    rec.mark("t9", "decision_start")
    rec.finish("t9")
    assert len(tasks.tracked) == 1
    await asyncio.gather(*tasks.tracked)
    assert [t["turn_id"] for t in await ops.recent_traces(5)] == ["t9"]
