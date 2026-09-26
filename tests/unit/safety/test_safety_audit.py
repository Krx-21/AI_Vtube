"""ModerationAudit: masked rows, background writes through the sink, failure handling."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivtube.contracts.safety import FilterResult, Verdict
from aivtube.infra import SystemClock
from aivtube.safety.audit import ModerationAudit, ModerationRecord, ops_sink, sha256_text
from aivtube.testing.fakes import FakeClock, FakeTaskSupervisor

BLOCKED = FilterResult(Verdict.BLOCK, "", "tier0", rule="ควย", category="sexual")
COLUMNS = {
    "ts",
    "character",
    "direction",
    "source",
    "tier",
    "category",
    "rule",
    "verdict",
    "text_masked",
    "text_sha256",
    "author",
    "turn_id",
}


class Sink:
    def __init__(self, *, fail: int = 0, hang: bool = False) -> None:
        self.rows: list[ModerationRecord] = []
        self.fail = fail
        self.hang = hang

    async def __call__(self, rec: ModerationRecord) -> None:
        if self.hang:
            await asyncio.Event().wait()
        if self.fail:
            self.fail -= 1
            raise OSError("disk full")
        self.rows.append(rec)


def record(audit: ModerationAudit, text: str = "ควย โทร 0812345678") -> ModerationRecord:
    return audit.record(
        character="pailin", direction="in", source="twitch", result=BLOCKED, text=text, author="u1"
    )


def test_rows_are_masked_and_hashed() -> None:
    clock = FakeClock()
    audit = ModerationAudit(None, clock=clock)
    rec = record(audit)
    assert rec.text_masked == "ควย โทร [เบอร์โทร]"
    assert "0812345678" not in rec.text_masked
    assert rec.text_sha256 == sha256_text("ควย โทร 0812345678")
    assert (rec.verdict, rec.category, rec.rule, rec.tier) == ("block", "sexual", "ควย", "tier0")
    assert rec.ts == clock.wall()
    assert set(rec.as_row()) == COLUMNS
    assert list(audit.recent) == [rec]
    assert audit.pending == 0


def test_long_text_is_truncated() -> None:
    audit = ModerationAudit(None, clock=FakeClock(), max_chars=10)
    assert len(record(audit, "ก" * 100).text_masked) == 10


def test_a_sink_needs_a_supervisor() -> None:
    with pytest.raises(ValueError):
        ModerationAudit(Sink(), clock=FakeClock())


async def test_rows_are_written_in_order_in_the_background() -> None:
    sink = Sink()
    tasks = FakeTaskSupervisor()
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=tasks)
    first = record(audit, "หนึ่ง")
    second = record(audit, "สอง")
    assert audit.pending == 2  # record() never awaits
    await audit.flush()
    assert sink.rows == [first, second]
    assert (audit.written, audit.pending, audit.errors) == (2, 0, 0)
    assert all(t.get_name() == "safety.audit" for t in tasks.tracked)


async def test_a_failing_sink_skips_the_row_and_continues() -> None:
    sink = Sink(fail=1)
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=FakeTaskSupervisor())
    record(audit, "หนึ่ง")
    ok = record(audit, "สอง")
    await audit.flush()
    assert sink.rows == [ok]
    assert (audit.errors, audit.written) == (1, 1)


async def test_a_hanging_sink_hits_the_deadline() -> None:
    sink = Sink(hang=True)
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=FakeTaskSupervisor(), timeout_s=0.05)
    record(audit)
    await asyncio.wait_for(audit.flush(), timeout=5)
    assert (audit.errors, audit.pending) == (1, 0)


async def test_backlog_is_bounded() -> None:
    sink = Sink()
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=FakeTaskSupervisor(), max_pending=2)
    for i in range(5):
        record(audit, f"ข้อความ {i}")
    assert audit.dropped == 3
    await audit.flush()
    assert [r.text_masked for r in sink.rows] == ["ข้อความ 3", "ข้อความ 4"]


def test_rows_recorded_without_a_loop_are_written_later() -> None:
    sink = Sink()
    tasks = FakeTaskSupervisor()
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=tasks)
    rec = record(audit)
    assert audit.pending == 1 and not tasks.tracked

    async def later() -> None:
        await audit.flush()

    asyncio.run(later())
    assert sink.rows == [rec]


async def test_cancellation_keeps_the_row_queued() -> None:
    sink = Sink(hang=True)
    tasks = FakeTaskSupervisor()
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=tasks, timeout_s=30)
    record(audit)
    await asyncio.sleep(0)
    (drainer,) = tasks.tracked
    drainer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drainer
    assert audit.pending == 1 and audit.errors == 0


async def test_aclose_gives_up_after_its_timeout() -> None:
    audit = ModerationAudit(
        Sink(hang=True), clock=SystemClock(), tasks=FakeTaskSupervisor(), timeout_s=30
    )
    record(audit)
    await asyncio.wait_for(audit.aclose(timeout_s=0.05), timeout=5)
    assert audit.pending == 1


async def test_ops_sink_adapts_an_ops_db() -> None:
    class OpsDb:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        async def log_moderation(self, **row: Any) -> None:
            self.rows.append(row)

    ops = OpsDb()
    audit = ModerationAudit(ops_sink(ops), clock=SystemClock(), tasks=FakeTaskSupervisor())
    rec = record(audit)
    await audit.flush()
    assert ops.rows == [rec.as_row()]
    assert set(ops.rows[0]) == COLUMNS


async def test_flush_survives_a_drainer_cancelled_by_someone_else() -> None:
    gate = asyncio.Event()

    async def sink(rec: ModerationRecord) -> None:
        await gate.wait()

    tasks = FakeTaskSupervisor()
    audit = ModerationAudit(sink, clock=SystemClock(), tasks=tasks, timeout_s=30)
    record(audit)
    flusher = asyncio.create_task(audit.flush())
    await asyncio.sleep(0.01)
    first = next(iter(tasks.tracked))
    first.cancel()  # e.g. the supervisor shutting down tracked tasks
    await asyncio.sleep(0.01)
    assert not flusher.done()
    gate.set()
    await asyncio.wait_for(flusher, timeout=5)
    assert audit.pending == 0 and audit.written == 1
