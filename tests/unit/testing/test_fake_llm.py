"""FakeLLM / FakeLLMRouter / FakeLauncher / FakeEmergencyServer behaviour."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx2
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aivtube.contracts.llm import ChatRequest, Done, ProviderFailed, TextDelta, ToolCall
from aivtube.contracts.types import HealthState
from aivtube.testing.fakes import (
    THAI_COMBINING,
    FakeClock,
    FakeEmergencyServer,
    FakeLauncher,
    FakeLLM,
    FakeLLMRouter,
    FakeReply,
    split_deltas,
    tool_call,
)


def req(text: str, slot: int | None = 0, system: str = "persona") -> ChatRequest:
    return ChatRequest(
        messages=({"role": "system", "content": system}, {"role": "user", "content": text}),
        slot=slot,
    )


async def collect(llm: Any, r: ChatRequest) -> list[Any]:
    return [e async for e in llm.stream(r)]


@given(st.text(alphabet="กขคโอเไ้่๊๋ัิีึืุู็์ abc", max_size=60), st.integers(1, 5))
def test_split_deltas_is_lossless_and_splits_marks(text: str, n: int) -> None:
    parts = split_deltas(text, n)
    assert "".join(parts) == text
    assert all(0 < len(p) <= n or (p[0] in THAI_COMBINING and len(p) <= n) for p in parts)
    for p in parts[1:]:
        assert len(p) <= n
    # every combining mark that follows another character starts its own delta
    joined = 0
    for p in parts:
        for i, ch in enumerate(p):
            if ch in THAI_COMBINING and joined + i > 0:
                assert i == 0
        joined += len(p)


async def test_scripted_order_text_then_calls_then_done() -> None:
    call = tool_call("set_stream_title", {"title": "ไพลิน"})
    llm = FakeLLM([FakeReply("เกม", "ได้เลยค่ะ", tool_calls=[call])], ttft_s=0, tok_s=0)
    events = await collect(llm, req("เปลี่ยนชื่อเกม"))
    kinds = [type(e) for e in events]
    assert kinds[-2:] == [ToolCall, Done] and set(kinds[:-2]) == {TextDelta}
    done = events[-1]
    assert (
        done.finish_reason == "tool_calls"
        and done.assistant_message["tool_calls"][0]["id"] == call.id
    )
    with pytest.raises(ProviderFailed) as info:  # consumed: nothing left to match
        await collect(llm, req("เปลี่ยนชื่อเกม"))
    assert info.value.emitted is False


async def test_fake_clock_paces_tokens() -> None:
    clock = FakeClock(start=0.0)
    llm = FakeLLM([FakeReply("", "abcdef")], ttft_s=0.3, tok_s=10.0, clock=clock)
    task = asyncio.ensure_future(collect(llm, req("x")))
    await clock.run_for(0.29)
    assert not task.done()
    await clock.run_for(1.0)
    events = await task
    assert events[-1].ttft_ms == pytest.approx(300.0)
    assert clock.sleeps[:2] == [0.3, 0.1]


@pytest.mark.parametrize(
    ("fail", "emitted"), [("connect", False), ("timeout", False), ("mid_stream", True)]
)
async def test_fail_modes(fail: str, emitted: bool) -> None:
    llm = FakeLLM([FakeReply("", "หนึ่งสองสามสี่ห้าหกเจ็ด", repeat=True)], ttft_s=0, tok_s=0, fail=fail)  # type: ignore[arg-type]
    r = ChatRequest(messages=({"role": "user", "content": "x"},), first_token_timeout_s=0.01)
    seen: list[Any] = []
    with pytest.raises(ProviderFailed) as info:
        async for ev in llm.stream(r):
            seen.append(ev)
    assert info.value.emitted is emitted and bool(seen) is emitted
    assert (await llm.probe()).state is HealthState.OK


async def test_stall_until_closed_and_instrumentation() -> None:
    llm = FakeLLM([FakeReply("", "abcd", repeat=True)], ttft_s=0, tok_s=0, fail="stall")
    stream: Any = llm.stream(req("x"))
    await stream.__anext__()
    assert llm.in_flight == 1
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(stream.__anext__(), 0.1)
    await stream.aclose()
    assert llm.in_flight == 0 and llm.closed_early == 1 and llm.max_in_flight == 1


async def test_prompt_cache_accounting() -> None:
    llm = FakeLLM([FakeReply("", "ค่ะ", repeat=True)], ttft_s=0, tok_s=0)
    first = (await collect(llm, req("คำถาม")))[-1]
    second = (await collect(llm, req("คำถามใหม่")))[-1]
    other_slot = (await collect(llm, req("คำถามใหม่", slot=1)))[-1]
    assert first.cache_n == 0 and second.cache_n > 0 and other_slot.cache_n == 0
    assert second.prompt_n < first.prompt_n + 5


async def test_router_fallback_and_promotion() -> None:
    a = FakeLLM([FakeReply("", "A", repeat=True)], ttft_s=0, tok_s=0, name="a", fail="connect")
    b = FakeLLM([FakeReply("", "B", repeat=True)], ttft_s=0, tok_s=0, name="b")
    router = FakeLLMRouter([a, b])
    assert (await collect(router, req("x")))[-1].provider == "b"
    assert next(s for s in router.status() if s.name == "a").fails == 1
    router.promote("b")
    assert router.active() == "b" and router.switches[-1] == ("a", "b", "promote")
    router.rollback()
    assert router.active() == "a"


async def test_launcher_start_fail_and_slots() -> None:
    clock = FakeClock()
    launcher = FakeLauncher(fail_start=["local4b"], start_delay_s=2.0, clock=clock)
    task = asyncio.ensure_future(launcher.ensure_running("local30b", 10.0))
    await clock.run_for(2.0)
    assert await task is True
    fail = asyncio.ensure_future(launcher.ensure_running("local4b", 10.0))
    await clock.run_for(2.0)
    assert await fail is False
    assert await launcher.restore_slot("local30b", 0, "x.bin") is False
    assert await launcher.save_slot("local30b", 0, "x.bin") is True
    assert await launcher.restore_slot("local30b", 0, "x.bin") is True
    launcher.crash("local30b")
    with pytest.raises(ConnectionError):
        await launcher.props("local30b")


async def test_emergency_endpoint_requires_the_token() -> None:
    srv = FakeEmergencyServer(token="secret")
    url = await srv.start()
    try:
        async with httpx2.AsyncClient(base_url=url, trust_env=False) as http:
            assert (await http.post("/hardkill")).status_code == 403
            ok = await http.post("/hardkill", headers={"Authorization": "Bearer secret"})
            assert ok.json()["voice"] == "down" and srv.voice_down
            assert (await http.post("/rearm?token=secret")).json()["voice"] == "up"
            ensure = await http.post("/llm/ensure/local30b", headers={"X-Aivtube-Token": "secret"})
            assert ensure.json()["ok"] is True
            status = (await http.get("/status?token=secret")).json()
            assert status["llm"]["local30b"] == "up" and status["voice"] == "up"
            await http.post("/restart/voice?token=secret")
            assert srv.restarts == ["voice"]
    finally:
        await srv.stop()
