"""SseFixtureServer parsed by the real openai SDK (3.19.2, httpx2): text, fragmented tool calls,
Thai combining marks split across chunks, timings, stalls, mid-stream death and cancellation."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from aivtube.testing.fakes import THAI_COMBINING, FakeReply, SseFixtureServer, tool_call

TEXT = "โอ้โห ไม่เป็นไรค่ะ ไพลินสบายดี"


@pytest.fixture
async def server() -> AsyncIterator[SseFixtureServer]:
    srv = SseFixtureServer("127.0.0.1", 0, [])
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def client(server: SseFixtureServer) -> AsyncIterator[AsyncOpenAI]:
    http = httpx2.AsyncClient(trust_env=False, timeout=httpx2.Timeout(10.0, connect=2.0))
    c = AsyncOpenAI(base_url=server.base_url, api_key="sk-local", max_retries=0, http_client=http)
    yield c
    await c.close()


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": "คุณคือไพลิน"}, {"role": "user", "content": text}]


async def _chunks(client: AsyncOpenAI, prompt: str, **kw: Any) -> list[Any]:
    stream = await client.chat.completions.create(
        model="pailin-30b",
        messages=_messages(prompt),
        stream=True,
        **kw,  # type: ignore[arg-type]
    )
    return [chunk async for chunk in stream]


async def test_text_stream_splits_combining_marks(
    server: SseFixtureServer, client: AsyncOpenAI
) -> None:
    server.add(FakeReply("สวัสดี", TEXT))
    chunks = await _chunks(client, "สวัสดีไพลิน", extra_body={"id_slot": 0, "cache_prompt": True})
    deltas = [
        c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
    ]
    assert "".join(deltas) == TEXT
    assert any(d[0] in THAI_COMBINING for d in deltas), "no delta starts with a combining mark"
    assert chunks[0].choices[0].delta.role == "assistant"
    final = chunks[-1]
    assert final.choices[0].finish_reason == "stop"
    timings = (final.model_extra or {})["timings"]
    assert {"cache_n", "prompt_n", "predicted_n"} <= set(timings)
    assert server.requests[-1]["id_slot"] == 0 and server.completed == 1


async def test_fragmented_tool_calls_accumulate(
    server: SseFixtureServer, client: AsyncOpenAI
) -> None:
    calls = [
        tool_call("set_stream_title", {"title": "ไพลินเล่นเกม Minecraft"}, id="call_a"),
        tool_call(
            "create_poll", {"title": "เกมต่อไป", "choices": ["มายคราฟ", "โอเวอร์คุก"]}, id="call_b"
        ),
    ]
    server.add(FakeReply("title", "", tool_calls=calls))
    tools = [
        {"type": "function", "function": {"name": c.name, "parameters": {"type": "object"}}}
        for c in calls
    ]
    chunks = await _chunks(
        client, "เปลี่ยน title", tools=tools, extra_body={"parallel_tool_calls": True}
    )
    acc: dict[int, dict[str, str]] = {}
    fragments = 0
    for ch in chunks:
        for tc in (ch.choices[0].delta.tool_calls or []) if ch.choices else []:
            entry = acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if tc.id:
                entry["id"] = tc.id
            if tc.function and tc.function.name:
                entry["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                entry["args"] += tc.function.arguments
                fragments += 1
    assert fragments > 4, "arguments were not fragmented"
    assert [acc[i]["name"] for i in sorted(acc)] == ["set_stream_title", "create_poll"]
    assert [acc[i]["id"] for i in sorted(acc)] == ["call_a", "call_b"]
    assert json.loads(acc[0]["args"]) == {"title": "ไพลินเล่นเกม Minecraft"}
    assert json.loads(acc[1]["args"])["choices"] == ["มายคราฟ", "โอเวอร์คุก"]
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


async def test_gemini_style_tool_call_without_index(
    server: SseFixtureServer, client: AsyncOpenAI
) -> None:
    extra = {"google": {"thought_signature": "sig-123"}}
    server.add(
        FakeReply(
            "", "", tool_calls=[tool_call("x", {"a": 1}, id="c1", extra=extra)], no_index=True
        )
    )
    chunks = await _chunks(client, "anything")
    tc = next(t for c in chunks if c.choices for t in (c.choices[0].delta.tool_calls or []))
    assert tc.index is None and tc.id == "c1"
    assert json.loads(tc.function.arguments) == {"a": 1}
    assert (tc.model_extra or {})["extra_content"] == extra


async def test_prompt_cache_and_usage_chunk(server: SseFixtureServer, client: AsyncOpenAI) -> None:
    server.add(FakeReply("", "ค่ะ", repeat=True))
    first = await _chunks(client, "คำถามแรก", extra_body={"id_slot": 0})
    second = await _chunks(
        client, "คำถามแรก", extra_body={"id_slot": 0}, stream_options={"include_usage": True}
    )
    t1 = (first[-1].model_extra or {})["timings"]
    finish = next(c for c in second if c.choices and c.choices[0].finish_reason)
    t2 = (finish.model_extra or {})["timings"]
    assert t1["cache_n"] == 0 and t2["cache_n"] > 0 and t2["prompt_n"] <= t1["prompt_n"]
    assert second[-1].choices == [] and second[-1].usage is not None


async def test_stall_is_observable(server: SseFixtureServer, client: AsyncOpenAI) -> None:
    server.add(FakeReply("", "หนึ่ง สอง สาม สี่", stall_after=2, stall_s=0.4))
    stream = await client.chat.completions.create(
        model="m",
        messages=_messages("x"),
        stream=True,  # type: ignore[arg-type]
    )
    stamps: list[float] = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            stamps.append(time.perf_counter())
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    assert max(gaps) >= 0.35


async def test_mid_stream_death_raises(server: SseFixtureServer, client: AsyncOpenAI) -> None:
    server.add(FakeReply("", "ประโยคนี้จะขาดกลางทางแน่นอน", die_after=3))
    got: list[str] = []
    with pytest.raises((openai.APIError, httpx2.HTTPError)):
        stream = await client.chat.completions.create(
            model="m",
            messages=_messages("x"),
            stream=True,  # type: ignore[arg-type]
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                got.append(chunk.choices[0].delta.content)
    assert 0 < len(got) <= 3 and server.killed == 1 and server.completed == 0


async def test_client_close_reaches_the_server() -> None:
    srv = SseFixtureServer("127.0.0.1", 0, [FakeReply("", "ก" * 80)], chunk_delay_s=0.05)
    http = httpx2.AsyncClient(trust_env=False)
    c = AsyncOpenAI(base_url=await srv.start(), api_key="k", max_retries=0, http_client=http)
    try:
        stream = await c.chat.completions.create(
            model="m",
            messages=_messages("x"),
            stream=True,  # type: ignore[arg-type]
        )
        async for _chunk in stream:
            break
        await stream.close()
        assert await srv.wait_disconnects(1, timeout=3.0), "server never saw the disconnect"
        assert srv.completed == 0
    finally:
        await c.close()
        await srv.stop()


async def test_http_error_status(server: SseFixtureServer, client: AsyncOpenAI) -> None:
    server.add(FakeReply("", "", status=503))
    with pytest.raises(openai.APIStatusError) as info:
        await _chunks(client, "x")
    assert info.value.status_code == 503


async def test_non_streaming_completion(server: SseFixtureServer, client: AsyncOpenAI) -> None:
    server.add(FakeReply("", "ได้ค่ะ", tool_calls=[tool_call("set_stream_title", {"title": "t"})]))
    resp = await client.chat.completions.create(model="m", messages=_messages("x"))  # type: ignore[arg-type]
    msg = resp.choices[0].message
    assert (
        msg.content == "ได้ค่ะ"
        and msg.tool_calls
        and msg.tool_calls[0].function.name == "set_stream_title"
    )


async def test_admin_endpoints(server: SseFixtureServer) -> None:
    async with httpx2.AsyncClient(base_url=server.root_url, trust_env=False) as http:
        assert (await http.get("/health")).json() == {"status": "ok"}
        server.loading = True
        assert (await http.get("/health")).status_code == 503
        props = (await http.get("/props")).json()
        assert props["chat_template_caps"]["supports_tool_calls"] is True
        saved = await http.post("/slots/0?action=save", json={"filename": "pailin-abc.bin"})
        assert saved.json()["filename"] == "pailin-abc.bin"
        assert server.slot_actions == [(0, "save", "pailin-abc.bin")]
        models = (await http.get("/v1/models")).json()
        assert models["data"][0]["id"] == "pailin-30b"
