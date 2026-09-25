"""Recorded llama-server SSE (tests/fixtures/sse) replayed over HTTP and parsed by the openai SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
from openai import AsyncOpenAI

from aivtube.testing.fakes import THAI_COMBINING, FakeReply, SseFixtureServer


def _events(raw: bytes) -> list[dict[str, Any]]:
    out = []
    for block in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
        line = block.strip()
        if line.startswith(b"data: ") and line != b"data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


async def _replay(raw: bytes) -> list[Any]:
    srv = SseFixtureServer("127.0.0.1", 0, [FakeReply("", "", raw_sse=raw)])
    http = httpx2.AsyncClient(trust_env=False)
    client = AsyncOpenAI(base_url=await srv.start(), api_key="k", max_retries=0, http_client=http)
    try:
        stream = await client.chat.completions.create(
            model="pailin-4b",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        return [c async for c in stream]
    finally:
        await client.close()
        await srv.stop()


async def test_recorded_text_stream(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "sse" / "llamacpp_text.sse").read_bytes()
    chunks = await _replay(raw)
    assert len(chunks) == len(_events(raw))
    deltas = [
        c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
    ]
    assert any(d[0] in THAI_COMBINING for d in deltas), "the recording splits combining marks"
    timings = (chunks[-1].model_extra or {})["timings"]
    assert timings["cache_n"] > 0 and timings["prompt_n"] >= 1
    assert chunks[-1].choices[0].finish_reason in ("stop", "length")


async def test_recorded_tool_call_stream(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "sse" / "llamacpp_tool_call.sse").read_bytes()
    chunks = await _replay(raw)
    name, args, ids = "", "", set()
    calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
    for tc in calls:
        assert tc.index == 0
        if tc.id:
            ids.add(tc.id)
        if tc.function and tc.function.name:
            name = tc.function.name
        if tc.function and tc.function.arguments:
            args += tc.function.arguments
    assert name == "set_stream_title" and len(ids) == 1
    assert json.loads(args) == {"title": "ไพลินเล่นเกม Minecraft"}
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_recorded_props_and_requests(fixtures_dir: Path) -> None:
    sse = fixtures_dir / "sse"
    with_template = json.loads((sse / "llamacpp_props.json").read_text(encoding="utf-8"))
    without = json.loads((sse / "llamacpp_props_no_template.json").read_text(encoding="utf-8"))
    assert with_template["chat_template_caps"]["supports_tool_calls"] is True
    assert without["chat_template_caps"]["supports_tool_calls"] is False
    req = json.loads((sse / "llamacpp_tool_call.request.json").read_text(encoding="utf-8"))
    assert req["tools"][0]["function"]["name"] == "set_stream_title" and req["stream"] is True
