"""``OpenAICompatProvider`` and ``FallbackRouter`` against ``SseFixtureServer`` over real HTTP."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    LLMProvider,
    ProviderFailed,
    TextDelta,
    ToolCall,
)
from aivtube.contracts.types import HealthState
from aivtube.llm.openai_stream import Flavor, ProviderError
from aivtube.llm.providers import OpenAICompatProvider, ProviderCfg
from aivtube.llm.router import FallbackRouter
from aivtube.testing.contracts import SET_TITLE_SPEC, case_id, llm_provider_suite, set_title_call
from aivtube.testing.fakes import THAI_COMBINING, FakeReply, SseFixtureServer, tool_call

REPLY = "สวัสดีค่ะ ไพลินเองนะ วันนี้โอ้โห สนุกมาก"
TOOL_PROMPT = "ช่วยเปลี่ยนชื่อไลฟ์หน่อย"


def provider(
    base_url: str, *, flavor: Flavor = "llamacpp", name: str = "local", **cfg: Any
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        ProviderCfg(
            name=name,
            kind="llamacpp" if flavor == "llamacpp" else "openai_compat",
            flavor=flavor,
            base_url=base_url,
            model="pailin-30b",
            slot_map={"speak": 0} if flavor == "llamacpp" else {},
            **cfg,
        )
    )


def req(prompt: str = "สวัสดี", **kw: Any) -> ChatRequest:
    return ChatRequest(
        messages=({"role": "system", "content": "คุณคือไพลิน"}, {"role": "user", "content": prompt}),
        **kw,
    )


async def collect(stream: AsyncIterator[LLMEvent]) -> list[LLMEvent]:
    return [ev async for ev in stream]


@pytest.fixture
async def server() -> AsyncIterator[SseFixtureServer]:
    srv = SseFixtureServer(
        script=[
            FakeReply(TOOL_PROMPT, "ได้เลยค่ะ", [set_title_call("ไพลินเล่นเกม")], repeat=True),
        ],
        default=FakeReply("", REPLY, repeat=True),
        chunk_delay_s=0.02,
    )
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


# --- the provider contract suite over real HTTP ---------------------------------------------------


class Target:
    server: SseFixtureServer | None = None
    made: list[OpenAICompatProvider] = []  # noqa: RUF012 - test-local registry


def _factory() -> LLMProvider:
    assert Target.server is not None
    p = provider(Target.server.base_url)
    Target.made.append(p)
    return p


async def _closed_probe(_: LLMProvider) -> bool:
    assert Target.server is not None
    return await Target.server.wait_disconnects(1, timeout=1.0)


@pytest.mark.parametrize(
    "case",
    llm_provider_suite(
        _factory,
        expect_text=REPLY,
        tool_prompt=TOOL_PROMPT,
        tool=SET_TITLE_SPEC,
        closed_probe=_closed_probe,
    ),
    ids=case_id,
)
async def test_provider_contract(case: Callable[[], Any], server: SseFixtureServer) -> None:
    Target.server = server
    try:
        await case()
    finally:
        for p in Target.made:
            await p.aclose()
        Target.made.clear()
        Target.server = None


# --- streaming behaviour --------------------------------------------------------------------------


async def test_text_deltas_keep_split_combining_marks(server: SseFixtureServer) -> None:
    p = provider(server.base_url)
    events = await collect(p.stream(req()))
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(deltas) == REPLY and any(d[0] in THAI_COMBINING for d in deltas)
    done = events[-1]
    assert isinstance(done, Done) and done.prompt_n is not None and done.cache_n is not None
    body = server.requests[-1]
    assert body["id_slot"] == 0 and body["cache_prompt"] is True and body["stream"] is True
    # a repeated prefix is reported as cached
    again = await collect(p.stream(req()))
    last = again[-1]
    assert isinstance(last, Done) and (last.cache_n or 0) > 0
    await p.aclose()


async def test_recorded_llama_server_streams(fixtures_dir: Path) -> None:
    sse = fixtures_dir / "sse"
    srv = SseFixtureServer(
        script=[
            FakeReply("text", "", raw_sse=(sse / "llamacpp_text.sse").read_bytes()),
            FakeReply("tool", "", raw_sse=(sse / "llamacpp_tool_call.sse").read_bytes()),
        ]
    )
    p = provider(await srv.start())
    try:
        text = await collect(p.stream(req("text")))
        done = text[-1]
        assert isinstance(done, Done) and done.finish_reason == "length"
        assert (done.prompt_n, done.cache_n) == (1, 58)
        tool = await collect(p.stream(req("tool", tools=(SET_TITLE_SPEC,))))
        call = tool[0]
        assert isinstance(call, ToolCall) and call.arguments == {"title": "ไพลินเล่นเกม Minecraft"}
    finally:
        await p.aclose()
        await srv.stop()


async def test_gemini_style_tool_call_over_http() -> None:
    sig = {"google": {"thought_signature": "c2ln"}}
    call = tool_call("remember", {"text": "แมวชื่อส้ม"}, id="call_g", extra=sig)
    srv = SseFixtureServer(script=[FakeReply("", "", [call], no_index=True)])
    p = provider(await srv.start(), flavor="gemini", cloud=False, name="gemini-local")
    try:
        events = await collect(p.stream(req()))
        got = events[0]
        assert isinstance(got, ToolCall) and got.id == "call_g" and got.extra == sig
        assert got.arguments == {"text": "แมวชื่อส้ม"}
    finally:
        await p.aclose()
        await srv.stop()


async def test_stall_ends_the_stream_and_disconnects(server: SseFixtureServer) -> None:
    server.add(FakeReply("stall", "ก ข ค ง จ ฉ ช", stall_after=2, stall_s=10.0))
    p = provider(server.base_url, stall_timeout_s=0.3)
    seen: list[LLMEvent] = []
    with pytest.raises(ProviderError) as ei:
        async for ev in p.stream(req("stall")):
            seen.append(ev)
    assert ei.value.reason == "stall" and ei.value.emitted is True and len(seen) == 2
    assert await server.wait_disconnects(1, timeout=1.0)
    await p.aclose()


async def test_mid_stream_death(server: SseFixtureServer) -> None:
    server.add(FakeReply("die", "ก ข ค ง จ ฉ ช", die_after=2))
    p = provider(server.base_url)
    seen: list[LLMEvent] = []
    with pytest.raises(ProviderError) as ei:
        async for ev in p.stream(req("die")):
            seen.append(ev)
    assert ei.value.emitted is True and ei.value.reason == "stream_error" and len(seen) == 2
    assert server.killed == 1
    await p.aclose()


async def test_first_token_timeout_disconnects(server: SseFixtureServer) -> None:
    server.add(FakeReply("slow", "ช้า", ttft_s=5.0))
    p = provider(server.base_url, first_token_timeout_s=0.3)
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(req("slow")))
    assert ei.value.reason == "first_token_timeout" and ei.value.emitted is False
    assert await server.wait_disconnects(1, timeout=1.0)
    await p.aclose()


async def test_http_error_status(server: SseFixtureServer) -> None:
    server.add(FakeReply("err", "", status=503))
    p = provider(server.base_url)
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(req("err")))
    assert ei.value.status == 503 and ei.value.emitted is False
    await p.aclose()


async def test_connection_refused_is_connect_failure() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = provider(f"http://127.0.0.1:{port}/v1")
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(req()))
    assert ei.value.reason == "connect" and ei.value.emitted is False
    assert (await p.probe()).state is HealthState.DOWN
    await p.aclose()


async def test_probe_reports_loading(server: SseFixtureServer) -> None:
    p = provider(server.base_url)
    assert (await p.probe()).state is HealthState.OK
    server.loading = True
    assert (await p.probe()).state is HealthState.STARTING
    await p.aclose()


@pytest.mark.timing
async def test_cancel_closes_the_http_stream_within_200_ms(server: SseFixtureServer) -> None:
    server.add(FakeReply("long", "ไพลิน " * 60))
    p = provider(server.base_url)
    first = asyncio.Event()

    async def consume() -> None:
        async for _ev in p.stream(req("long")):
            first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), 5.0)
    t0 = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await server.wait_disconnects(1, timeout=1.0)
    assert time.perf_counter() - t0 <= 0.2
    await p.aclose()


# --- the router over real HTTP --------------------------------------------------------------------


async def test_router_falls_back_between_real_servers(server: SseFixtureServer) -> None:
    broken = SseFixtureServer(script=[FakeReply("", "", status=500, repeat=True)])
    a = provider(await broken.start(), name="local-30b")
    b = provider(server.base_url, name="local-4b")
    router = FallbackRouter([a, b])
    try:
        events = await collect(router.stream(req()))
        done = events[-1]
        assert isinstance(done, Done) and done.provider == "local-4b"
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == REPLY
        # mid-stream death after the first delta is not retried elsewhere
        server.add(FakeReply("die", "ก ข ค ง จ", die_after=2))
        with pytest.raises(ProviderFailed) as ei:
            await collect(router.stream(req("die")))
        assert ei.value.emitted is True
    finally:
        await router.aclose()
        await broken.stop()
