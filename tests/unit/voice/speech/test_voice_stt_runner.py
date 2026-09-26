"""SttRunner (priority threads, timeout chain, health) and UtteranceAssembler (voice.stt)."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from aivtube.contracts.types import Health, HealthState, Transcript
from aivtube.infra.clock import SystemClock
from aivtube.testing.fakes import FakeClock, FakePostProcessor, FakeRecognizer
from aivtube.voice.stt import NamePostProcessor, SttRunner, UtteranceAssembler


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), np.float32)


async def wait_for(predicate, timeout: float = 3.0) -> None:  # type: ignore[no-untyped-def]
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > end:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.005)


async def test_transcribes_on_the_primary_and_post_processes() -> None:
    rec = FakeRecognizer(["สวัสดีครับ ไทลิน"], name="rt")
    post = NamePostProcessor({"ไพลิน": ["ไทลิน"]})
    runner = SttRunner([rec], post)
    try:
        t = await runner.transcribe(audio(1.0))
        assert t is not None and t.text == "สวัสดีครับ ไพลิน" and t.engine == "rt"
        assert runner.health().state == HealthState.OK
        assert rec.calls == [(16000, False)]
    finally:
        await runner.aclose()


async def test_quick_request_jumps_ahead_of_queued_normal_requests() -> None:
    rec = FakeRecognizer(["a", "b", "c", "q"], delay_s=0.15, name="rt")
    runner = SttRunner([rec], None, timeout_s=5.0)
    try:
        first = asyncio.create_task(runner.transcribe(audio(1.0)))
        await wait_for(lambda: len(rec.calls) == 1)  # the first decode is running
        normal_b = asyncio.create_task(runner.transcribe(audio(2.0)))
        normal_c = asyncio.create_task(runner.transcribe(audio(3.0)))
        await asyncio.sleep(0)
        quick = asyncio.create_task(runner.transcribe(audio(0.8), quick=True))
        await asyncio.gather(first, normal_b, normal_c, quick)
        assert rec.calls == [(16000, False), (12800, True), (32000, False), (48000, False)]
    finally:
        await runner.aclose()


async def test_timeout_moves_to_the_next_backend_and_marks_degraded() -> None:
    slow = FakeRecognizer(["slow"], delay_s=0.6, name="slow")
    fast = FakeRecognizer(["fast"], name="fast")
    seen: list[Health] = []
    runner = SttRunner([slow, fast], None, timeout_s=0.15, on_health=seen.append)
    try:
        t0 = time.perf_counter()
        t = await runner.transcribe(audio(1.0))
        assert time.perf_counter() - t0 < 0.5
        assert t is not None and t.engine == "fast" and t.text == "fast"
        assert runner.health().state == HealthState.DEGRADED
        assert "slow" in runner.health().detail
        assert [h.state for h in seen] == [HealthState.DEGRADED]
        # while the slow backend is stuck in its decode it is skipped at once
        t0 = time.perf_counter()
        t2 = await runner.transcribe(audio(1.0))
        assert t2 is not None and t2.engine == "fast"
        assert time.perf_counter() - t0 < 0.1
        assert len(slow.calls) == 1
        # once the stuck decode returns, the primary serves again and health recovers
        await asyncio.sleep(0.7)
        slow.delay_s = 0.0
        t3 = await runner.transcribe(audio(1.0))
        assert t3 is not None and t3.engine == "slow"
        assert runner.health().state == HealthState.OK
        assert seen[-1].state == HealthState.OK
    finally:
        await runner.aclose()


async def test_timeout_is_driven_by_the_injected_clock() -> None:
    clock = FakeClock()
    slow = FakeRecognizer(["slow"], delay_s=0.4, name="slow")
    fast = FakeRecognizer(["fast"], name="fast")
    runner = SttRunner([slow, fast], None, timeout_s=3.0, clock=clock)
    try:
        task = asyncio.create_task(runner.transcribe(audio(1.0)))
        await wait_for(lambda: len(slow.calls) == 1)
        await clock.run_for(2.9)
        assert not task.done()
        await clock.run_for(0.2)  # 3 s of fake time: the deadline fires
        t = await asyncio.wait_for(task, 2.0)
        assert t is not None and t.engine == "fast"
    finally:
        await runner.aclose()


async def test_errors_fall_back_and_all_failing_returns_none() -> None:
    bad = FakeRecognizer(["x"], fail_times=1, name="bad")
    good = FakeRecognizer(["ok"], name="good")
    runner = SttRunner([bad, good], None)
    try:
        t = await runner.transcribe(audio(1.0))
        assert t is not None and t.engine == "good"
        assert runner.stats["fallbacks"] == 1 and runner.stats["bad.error"] == 1
    finally:
        await runner.aclose()

    worse = FakeRecognizer(["x"], fail_times=5, name="worse")
    runner2 = SttRunner([worse], None)
    try:
        assert await runner2.transcribe(audio(1.0)) is None
        assert runner2.health().state == HealthState.DEGRADED
        assert runner2.stats["lost"] == 1
    finally:
        await runner2.aclose()


async def test_post_processor_drop_does_not_move_down_the_chain() -> None:
    primary = FakeRecognizer(["สวัสดี"], name="p")
    secondary = FakeRecognizer(["other"], name="s")
    post = FakePostProcessor(min_audio_s=0.3)
    runner = SttRunner([primary, secondary], post)
    try:
        assert await runner.transcribe(audio(0.2)) is None  # too short: dropped
        assert secondary.calls == []
        assert runner.health().state == HealthState.OK
        assert post.calls and post.calls[0][1] == ""
        await runner.transcribe(audio(1.0), recent_tts_text="เมื่อกี้พูดว่า")
        assert post.calls[-1][1] == "เมื่อกี้พูดว่า"
    finally:
        await runner.aclose()


async def test_breaker_takes_a_failing_backend_out_for_the_cooldown() -> None:
    clock = FakeClock()
    bad = FakeRecognizer(["x"] * 10, fail_times=10, name="bad")
    good = FakeRecognizer(["ok"] * 10, name="good")
    runner = SttRunner(
        [bad, good],
        None,
        clock=clock,
        breaker_failures=3,
        breaker_window_s=60,
        breaker_cooldown_s=30,
    )
    try:
        for _ in range(3):
            assert (await runner.transcribe(audio(1.0))) is not None
        assert len(bad.calls) == 3
        await runner.transcribe(audio(1.0))
        assert len(bad.calls) == 3  # skipped while open
        clock.advance(31.0)
        await runner.transcribe(audio(1.0))
        assert len(bad.calls) == 4
    finally:
        await runner.aclose()


async def test_cancelled_request_is_skipped_by_the_decode_thread() -> None:
    rec = FakeRecognizer(["a", "b"], delay_s=0.15, name="rt")
    runner = SttRunner([rec], None)
    try:
        first = asyncio.create_task(runner.transcribe(audio(1.0)))
        await wait_for(lambda: len(rec.calls) == 1)
        second = asyncio.create_task(runner.transcribe(audio(2.0)))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert (await first) is not None
        await asyncio.sleep(0.1)
        assert rec.calls == [(16000, False)]
    finally:
        await runner.aclose()


async def test_warmup_runs_on_the_decode_thread_and_close_stops_everything() -> None:
    rec = FakeRecognizer(["x"], name="rt")
    runner = SttRunner([rec], None)
    health = await runner.warmup()
    assert rec.warmed and health.state == HealthState.OK
    await runner.aclose()
    assert rec.closed
    assert await runner.transcribe(audio(1.0)) is None


async def test_warmup_failure_of_the_primary_marks_degraded() -> None:
    class Broken(FakeRecognizer):
        def warmup(self) -> None:
            raise RuntimeError("model missing")

    runner = SttRunner([Broken(["x"], name="broken")], None)
    try:
        health = await runner.warmup()
        assert health.state == HealthState.DEGRADED and "broken" in health.detail
    finally:
        await runner.aclose()


async def test_empty_chain_is_down_and_returns_none() -> None:
    runner = SttRunner([], None, clock=SystemClock())
    assert runner.health().state == HealthState.DOWN
    assert await runner.transcribe(audio(1.0)) is None
    runner.close()


def tr(text: str, audio_s: float, engine: str = "rt", latency: float = 50.0) -> Transcript:
    return Transcript(text=text, is_final=True, audio_s=audio_s, latency_ms=latency, engine=engine)


def test_assembler_joins_partials_with_spaces_and_counts_parts() -> None:
    asm = UtteranceAssembler()
    asm.add_partial(tr("วันนี้เราจะเล่นเกม", 15.0, latency=120.0))
    asm.add_partial(tr("แล้วก็คุยกับแชท", 15.0, latency=110.0))
    assert asm.pending == 2
    out = asm.finish(tr("กันนะ", 3.0, latency=40.0))
    assert out is not None
    assert out.text == "วันนี้เราจะเล่นเกม แล้วก็คุยกับแชท กันนะ"
    assert asm.parts == 3
    assert out.audio_s == pytest.approx(33.0)
    assert out.latency_ms == 40.0 and out.is_final and out.engine == "rt"
    assert asm.pending == 0


def test_assembler_handles_empty_and_missing_parts() -> None:
    asm = UtteranceAssembler()
    asm.add_partial(tr("", 15.0))  # noise-only split
    asm.add_partial(tr("สวัสดี", 15.0, engine="rt"))
    out = asm.finish(None)  # the final decode was dropped
    assert out is not None and out.text == "สวัสดี" and asm.parts == 1
    assert out.audio_s == pytest.approx(30.0)
    assert asm.finish(None) is None and asm.parts == 0
    asm.add_partial(tr("a", 1.0, engine="rt"))
    out2 = asm.finish(tr("b", 1.0, engine="pythaiasr"))
    assert out2 is not None and out2.engine == "rt+pythaiasr" and asm.parts == 2
    asm.add_partial(tr("x", 1.0))
    asm.reset()
    out3 = asm.finish(tr("y", 1.0))
    assert out3 is not None and out3.text == "y" and asm.parts == 1
