"""``OpenAICompatProvider`` against ``httpx2.MockTransport`` (no sockets, FakeClock timeouts)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from llm_testkit import (
    ROLE,
    Body,
    Fail,
    MockLLM,
    Sleep,
    chunk,
    collect,
    frame,
    make_provider,
    request,
    sse,
    text_chunks,
    timings,
)

from aivtube.contracts.llm import ChatRequest, Done, ProviderFailed, TextDelta, ToolCall, ToolSpec
from aivtube.contracts.types import HealthState
from aivtube.infra import SystemClock
from aivtube.llm.openai_stream import ProviderError
from aivtube.llm.providers import OpenAICompatProvider, ProviderCfg
from aivtube.llm.router import FallbackRouter
from aivtube.testing.fakes import THAI_COMBINING, FakeClock

REMEMBER = ToolSpec(
    "remember",
    "จำข้อเท็จจริง",
    {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)


async def test_streams_text_then_done_with_llama_timings(fake_clock: FakeClock) -> None:
    mock = MockLLM(
        Body(
            [
                sse(ROLE, done=False),
                Sleep(0.25),
                sse(
                    *text_chunks("สวัสดี", "ค่ะ"),
                    chunk({}, "stop", timings=timings(prompt_n=7, cache_n=301, predicted_n=2)),
                ),
            ],
            sleep=fake_clock.sleep,
        )
    )
    p = make_provider(mock, clock=fake_clock)
    task = asyncio.create_task(collect(p.stream(request())))
    await mock.wait_requests()
    await fake_clock.run_for(0.3)
    events = await task
    assert events[:2] == [TextDelta("สวัสดี"), TextDelta("ค่ะ")]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.provider == "test-llamacpp" and done.finish_reason == "stop"
    assert done.ttft_ms == pytest.approx(250.0)
    assert (done.prompt_n, done.cache_n, done.completion_tokens) == (7, 301, 2)
    assert done.assistant_message == {"role": "assistant", "content": "สวัสดีค่ะ"}
    assert mock.bodies[0].closed


async def test_request_body_for_llamacpp() -> None:
    mock = MockLLM(Body([sse(chunk({"content": "ok"}), chunk({}, "stop"))]))
    p = make_provider(mock)
    req = request(tools=(REMEMBER,), max_tokens=48, temperature=0.4, slot=1)
    await collect(p.stream(req))
    body = mock.body()
    assert mock.requests[0].url.path == "/v1/chat/completions"
    assert body["model"] == "pailin-30b" and body["stream"] is True
    assert body["max_tokens"] == 48 and body["temperature"] == 0.4
    assert body["id_slot"] == 1 and body["cache_prompt"] is True
    assert body["parallel_tool_calls"] is True
    assert body["tools"][0]["function"]["name"] == "remember"
    assert all(isinstance(m["content"], str) for m in body["messages"])


async def test_thai_combining_marks_split_across_events_and_bytes() -> None:
    pieces = ["โอ", "้", "โห ", "น่", "ารั", "ก", "จั", "ง"]
    assert any(p[0] in THAI_COMBINING for p in pieces)
    raw = sse(*text_chunks(*pieces), chunk({}, "stop"))
    # 5-byte HTTP chunks cut through the 3-byte UTF-8 sequences of Thai characters
    mock = MockLLM(Body([raw[i : i + 5] for i in range(0, len(raw), 5)]))
    events = await collect(make_provider(mock).stream(request()))
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == pieces
    done = events[-1]
    assert isinstance(done, Done) and done.assistant_message["content"] == "".join(pieces)


async def test_fragmented_tool_args_come_after_text(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "sse" / "llamacpp_tool_call.sse").read_bytes()
    mock = MockLLM(Body([sse(chunk({"content": "ได้เลย "}), done=False), raw]))
    events = await collect(make_provider(mock).stream(request(tools=(REMEMBER,))))
    assert isinstance(events[0], TextDelta)
    call = events[1]
    assert isinstance(call, ToolCall) and call.name == "set_stream_title"
    assert call.arguments == {"title": "ไพลินเล่นเกม Minecraft"}
    assert json.loads(call.raw_arguments) == call.arguments and call.extra is None
    done = events[2]
    assert isinstance(done, Done) and done.finish_reason == "tool_calls"
    assert (done.prompt_n, done.cache_n) == (1, 207)
    msg_call = done.assistant_message["tool_calls"][0]
    assert msg_call["id"] == call.id and msg_call["function"]["arguments"] == call.raw_arguments


async def test_recorded_text_stream_keeps_every_delta(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "sse" / "llamacpp_text.sse").read_bytes()
    events = await collect(make_provider(MockLLM(Body([raw]))).stream(request()))
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert any(d[0] in THAI_COMBINING for d in deltas)
    done = events[-1]
    assert isinstance(done, Done) and done.finish_reason == "length" and done.cache_n == 58
    assert done.assistant_message["content"] == "".join(deltas)


async def test_truncated_tool_arguments_are_repaired() -> None:
    first: dict[str, Any] = {
        "index": 0,
        "id": "c9",
        "type": "function",
        "function": {"name": "remember"},
    }
    frags = ['{"text": "แมว', "ชื่อส้ม"]
    mock = MockLLM(
        Body(
            [
                sse(
                    chunk(
                        {
                            "tool_calls": [
                                {**first, "function": {**first["function"], "arguments": ""}}
                            ]
                        }
                    ),
                    *[
                        chunk({"tool_calls": [{"index": 0, "function": {"arguments": f}}]})
                        for f in frags
                    ],
                    chunk({}, "length"),
                )
            ]
        )
    )
    events = await collect(make_provider(mock).stream(request(tools=(REMEMBER,))))
    call = events[0]
    assert isinstance(call, ToolCall)
    assert call.raw_arguments == '{"text": "แมวชื่อส้ม' and call.arguments == {"text": "แมวชื่อส้ม"}
    done = events[-1]
    assert isinstance(done, Done)
    sent_back = done.assistant_message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent_back) == {"text": "แมวชื่อส้ม"}


async def test_gemini_tool_call_without_index_keeps_thought_signature() -> None:
    sig = {"google": {"thought_signature": "U0lHTkFUVVJF"}}
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "remember", "arguments": '{"text":"x"}'},
        "extra_content": sig,
    }
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "", "tool_calls": [dict(tc)]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    body = [sse(chunk({"role": "assistant", "tool_calls": [tc]}), chunk({}, "stop"))]
    gemini = MockLLM(Body(body))
    events = await collect(
        make_provider(gemini, flavor="gemini", reasoning_effort="minimal").stream(
            request(tools=(REMEMBER,))
        )
    )
    call = events[0]
    assert isinstance(call, ToolCall) and call.extra == sig and call.arguments == {"text": "x"}
    done = events[-1]
    assert isinstance(done, Done) and done.finish_reason == "tool_calls"  # Gemini said "stop"
    assert done.assistant_message["tool_calls"][0]["extra_content"] == sig
    # outgoing history: Gemini keeps extra_content, the reasoning effort and no llama extras
    replay = ChatRequest(messages=tuple(history))
    again = MockLLM(Body(body))
    await collect(make_provider(again, flavor="gemini", reasoning_effort="minimal").stream(replay))
    sent = again.body()
    assert sent["messages"][1]["tool_calls"][0]["extra_content"] == sig
    assert sent["reasoning_effort"] == "minimal"
    assert "id_slot" not in sent and "cache_prompt" not in sent
    # the same history sent to llama.cpp or Typhoon loses extra_content
    for flavor in ("llamacpp", "typhoon"):
        other = MockLLM(Body(body))
        events = await collect(make_provider(other, flavor=flavor).stream(replay))
        assert "extra_content" not in other.body()["messages"][1]["tool_calls"][0]
        call = next(e for e in events if isinstance(e, ToolCall))
        assert call.extra is None


async def test_typhoon_request_and_error_detail() -> None:
    mock = MockLLM(httpx2.Response(401, json={"detail": "Invalid API Key"}))
    p = make_provider(mock, flavor="typhoon", extra_body={"repetition_penalty": 1.05})
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(request()))
    err = ei.value
    assert err.emitted is False and err.reason == "auth" and err.status == 401
    assert "Invalid API Key" in str(err)
    body = mock.body()
    assert body["repetition_penalty"] == 1.05 and "id_slot" not in body
    assert mock.requests[0].headers["authorization"] == "Bearer sk-test"


async def test_cloud_provider_without_key_never_sends() -> None:
    mock = MockLLM()
    p = make_provider(mock, flavor="typhoon", api_key="")
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(request()))
    assert ei.value.reason == "auth" and not ei.value.emitted and not mock.requests
    assert (await p.probe()).state is HealthState.DOWN and not mock.requests


async def test_connect_error_before_first_token() -> None:
    p = make_provider(MockLLM(Fail(httpx2.ConnectError)))
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(request()))
    assert ei.value.emitted is False and ei.value.reason == "connect"


@pytest.mark.parametrize(("status", "reason"), [(500, "status"), (429, "rate_limited")])
async def test_http_errors_before_first_token(status: int, reason: str) -> None:
    mock = MockLLM(httpx2.Response(status, json={"error": {"code": status, "message": "nope"}}))
    with pytest.raises(ProviderError) as ei:
        await collect(make_provider(mock).stream(request()))
    assert ei.value.emitted is False and ei.value.reason == reason and ei.value.status == status
    assert "nope" in str(ei.value)


async def test_first_token_timeout(fake_clock: FakeClock) -> None:
    body = Body([sse(ROLE, done=False), Sleep(60)], sleep=fake_clock.sleep)
    mock = MockLLM(body)
    p = make_provider(mock, clock=fake_clock, first_token_timeout_s=2.0)
    task = asyncio.create_task(collect(p.stream(request())))
    await mock.wait_requests()
    await fake_clock.run_for(1.9)
    assert not task.done()
    await fake_clock.run_for(0.2)
    with pytest.raises(ProviderError) as ei:
        await task
    assert ei.value.reason == "first_token_timeout" and ei.value.emitted is False
    assert body.closed


async def test_request_level_first_token_timeout_overrides(fake_clock: FakeClock) -> None:
    body = Body([Sleep(60)], sleep=fake_clock.sleep)
    mock = MockLLM(body)
    p = make_provider(mock, clock=fake_clock, first_token_timeout_s=10.0)
    task = asyncio.create_task(collect(p.stream(request(first_token_timeout_s=0.5))))
    await mock.wait_requests()
    await fake_clock.run_for(0.6)
    with pytest.raises(ProviderError) as ei:
        await task
    assert ei.value.reason == "first_token_timeout"


async def test_inter_token_stall_ends_with_emitted_failure(fake_clock: FakeClock) -> None:
    body = Body(
        [sse(ROLE, *text_chunks("สวัส", "ดี"), done=False), Sleep(60), sse(chunk({}, "stop"))],
        sleep=fake_clock.sleep,
    )
    mock = MockLLM(body)
    p = make_provider(mock, clock=fake_clock)
    seen: list[object] = []

    async def consume() -> None:
        async for ev in p.stream(request()):
            seen.append(ev)

    task = asyncio.create_task(consume())
    await mock.wait_requests()
    await fake_clock.run_for(2.9)
    assert not task.done() and seen == [TextDelta("สวัส"), TextDelta("ดี")]
    await fake_clock.run_for(0.2)
    with pytest.raises(ProviderFailed) as ei:
        await task
    assert ei.value.emitted is True and isinstance(ei.value, ProviderError)
    assert ei.value.reason == "stall" and body.closed


async def test_stall_before_any_text_is_not_emitted(fake_clock: FakeClock) -> None:
    tc = {
        "index": 0,
        "id": "c",
        "type": "function",
        "function": {"name": "remember", "arguments": "{"},
    }
    body = Body([sse(chunk({"tool_calls": [tc]}), done=False), Sleep(60)], sleep=fake_clock.sleep)
    mock = MockLLM(body)
    p = make_provider(mock, clock=fake_clock)
    task = asyncio.create_task(collect(p.stream(request(tools=(REMEMBER,)))))
    await mock.wait_requests()
    await fake_clock.run_for(3.1)
    with pytest.raises(ProviderError) as ei:
        await task
    assert ei.value.reason == "stall" and ei.value.emitted is False


async def test_mid_stream_death_is_emitted() -> None:
    body = Body([sse(ROLE, *text_chunks("สวัสดี"), done=False), httpx2.ReadError("reset")])
    events: list[object] = []
    with pytest.raises(ProviderError) as ei:
        async for ev in make_provider(MockLLM(body)).stream(request()):
            events.append(ev)
    assert events == [TextDelta("สวัสดี")]
    assert ei.value.emitted is True and ei.value.reason == "stream_error"


async def test_stream_error_event_mid_stream() -> None:
    body = Body([sse(chunk({"content": "ก"}), done=False), frame({"error": {"message": "OOM"}})])
    with pytest.raises(ProviderError) as ei:
        await collect(make_provider(MockLLM(body)).stream(request()))
    assert ei.value.emitted is True and "OOM" in str(ei.value)


async def test_chunks_with_missing_fields_are_tolerated() -> None:
    body = Body(
        [
            sse(
                {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0}]},
                {"id": "x", "object": "chat.completion.chunk"},
                chunk({"content": "ไพลิน"}),
                {"id": "x", "choices": [{"index": 0, "finish_reason": "stop"}]},
            )
        ]
    )
    events = await collect(make_provider(MockLLM(body)).stream(request()))
    assert events[0] == TextDelta("ไพลิน")
    done = events[-1]
    assert isinstance(done, Done) and done.finish_reason == "stop"


async def test_empty_stream_is_a_failure() -> None:
    with pytest.raises(ProviderError) as ei:
        await collect(make_provider(MockLLM(Body([b"data: [DONE]\n\n"]))).stream(request()))
    assert ei.value.reason == "protocol" and ei.value.emitted is False


async def test_aclose_mid_stream_closes_the_transport(fake_clock: FakeClock) -> None:
    body = Body(
        [sse(ROLE, *text_chunks("ไพ", "ลิน"), done=False), Sleep(60)], sleep=fake_clock.sleep
    )
    stream = make_provider(MockLLM(body), clock=fake_clock).stream(request())
    assert await stream.__anext__() == TextDelta("ไพ")
    await stream.aclose()
    assert body.closed


async def test_cancelling_the_consumer_closes_the_transport(fake_clock: FakeClock) -> None:
    body = Body([sse(ROLE, *text_chunks("ไพ"), done=False), Sleep(60)], sleep=fake_clock.sleep)
    mock = MockLLM(body)
    p = make_provider(mock, clock=fake_clock)
    first = asyncio.Event()

    async def consume() -> None:
        async for _ev in p.stream(request()):
            first.set()

    task = asyncio.create_task(consume())
    await mock.wait_requests()
    await fake_clock.run_until(first.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed


async def test_cloud_usage_maps_to_prompt_and_cache() -> None:
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 9,
        "total_tokens": 1209,
        "prompt_tokens_details": {"cached_tokens": 1000},
    }
    body = Body(
        [
            sse(
                chunk({"content": "hi"}),
                chunk({}, "stop"),
                {**chunk(), "choices": [], "usage": usage},
            )
        ]
    )
    events = await collect(make_provider(MockLLM(body), flavor="gemini").stream(request()))
    done = events[-1]
    assert isinstance(done, Done)
    assert (done.prompt_n, done.cache_n, done.completion_tokens) == (200, 1000, 9)


# --- probe / prefill ------------------------------------------------------------------------


async def test_llamacpp_probe_maps_health() -> None:
    mock = MockLLM(
        httpx2.Response(200, json={"status": "ok"}),
        httpx2.Response(503, json={"error": {"code": 503, "message": "Loading model"}}),
        Fail(httpx2.ConnectError),
    )
    p = make_provider(mock)
    states = [(await p.probe()).state for _ in range(3)]
    assert states == [HealthState.OK, HealthState.STARTING, HealthState.DOWN]
    assert [r.url.path for r in mock.requests] == ["/health"] * 3


async def test_typhoon_probe_uses_a_raw_get_and_accepts_a_bare_array() -> None:
    mock = MockLLM(
        httpx2.Response(200, json=[{"id": "typhoon-v2.5-30b-a3b-instruct"}]),
        httpx2.Response(401, json={"detail": "Invalid API Key"}),
    )
    p = make_provider(mock, flavor="typhoon")
    assert (await p.probe()).state is HealthState.OK
    bad = await p.probe()
    assert bad.state is HealthState.DOWN and "401" in bad.detail
    assert mock.requests[0].url.path == "/v1/models"


async def test_local_openai_flavor_probe_needs_no_key() -> None:
    mock = MockLLM(httpx2.Response(200, json={"object": "list", "data": []}))
    p = make_provider(mock, flavor="openai", api_key_env=None)
    assert not p.cfg.cloud
    assert (await p.probe()).state is HealthState.OK
    assert mock.requests[0].url.path == "/v1/models"
    assert "authorization" not in mock.requests[0].headers


async def test_probe_has_a_deadline(fake_clock: FakeClock) -> None:
    async def hang(request: httpx2.Request) -> httpx2.Response:
        await fake_clock.sleep(60)
        return httpx2.Response(200)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(hang))
    p = OpenAICompatProvider(
        ProviderCfg("x", kind="llamacpp", flavor="llamacpp", base_url="http://h/v1", model="m"),
        http_client=http,
        clock=fake_clock,
    )
    task = asyncio.create_task(p.probe())
    await fake_clock.run_for(2.1)
    health = await task
    assert health.state is HealthState.DOWN and "timed out" in health.detail


async def test_llamacpp_prefill_is_a_one_token_call_on_the_slot(fake_clock: FakeClock) -> None:
    completion = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "pailin-30b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ค"},
                "finish_reason": "length",
            }
        ],
    }
    mock = MockLLM(httpx2.Response(200, json=completion))
    took = await make_provider(mock, clock=fake_clock).prefill(request(slot=1, max_tokens=256))
    assert took == 0.0
    body = mock.body()
    assert body["max_tokens"] == 1 and body["stream"] is False
    assert body["id_slot"] == 1 and body["cache_prompt"] is True


async def test_prefill_failure_raises_provider_failed() -> None:
    with pytest.raises(ProviderError) as ei:
        await make_provider(MockLLM(Fail(httpx2.ConnectError))).prefill(request())
    assert ei.value.emitted is False and ei.value.reason == "connect"


async def test_cloud_prefill_is_a_no_op() -> None:
    mock = MockLLM()
    assert await make_provider(mock, flavor="gemini").prefill(request()) == 0.0
    assert not mock.requests


async def test_repr_never_shows_the_key() -> None:
    p = make_provider(MockLLM(), flavor="gemini", api_key="sk-secret-value")
    assert "sk-secret-value" not in repr(p)


# --- the router over real providers (modules.json acceptance) -------------------------------------


async def test_router_falls_back_on_connect_error_before_the_first_token() -> None:
    down = MockLLM(Fail(httpx2.ConnectError))
    up = MockLLM(
        Body([sse(ROLE, *text_chunks("สวัสดี", "ค่ะ"), chunk({}, "stop", timings=timings()))])
    )
    router = FallbackRouter(
        [make_provider(down, name="local-30b"), make_provider(up, name="local-4b")],
        clock=SystemClock(),
    )
    events = await collect(router.stream(request()))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("สวัสดี"), TextDelta("ค่ะ")]
    done = events[-1]
    assert isinstance(done, Done) and done.provider == "local-4b" and done.cache_n == 288
    assert len(down.requests) == 1 and len(up.requests) == 1
    st = {s.name: s for s in router.status()}
    assert st["local-30b"].fails == 1 and st["local-4b"].fails == 0
    await router.aclose()


async def test_router_never_falls_back_after_the_first_token() -> None:
    dying = Body([sse(ROLE, *text_chunks("สวัส"), done=False), httpx2.ReadError("reset")])
    spare = MockLLM(Body([sse(chunk({"content": "จากสำรอง"}), chunk({}, "stop"))]))
    router = FallbackRouter(
        [make_provider(MockLLM(dying), name="a"), make_provider(spare, name="b")],
        clock=SystemClock(),
    )
    seen: list[object] = []
    with pytest.raises(ProviderFailed) as ei:
        async for ev in router.stream(request()):
            seen.append(ev)
    assert ei.value.emitted is True and seen == [TextDelta("สวัส")]
    assert not spare.requests and dying.closed
    await router.aclose()
