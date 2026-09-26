"""Pure parts of ``aivtube.llm.openai_stream``: accumulation, parsing, sanitising, requests."""

from __future__ import annotations

import json
import pickle
from types import MappingProxyType
from typing import Any

import pytest

from aivtube.contracts.llm import ChatRequest, ProviderFailed, ToolSpec
from aivtube.llm.openai_stream import (
    ProviderError,
    ToolCallAccumulator,
    assistant_message,
    build_request,
    content_text,
    error_detail,
    history_arguments,
    parse_arguments,
    sanitize_messages,
    tool_payload,
)

REMEMBER = ToolSpec(
    name="remember",
    description="จำข้อเท็จจริงข้ามสตรีม",
    parameters=MappingProxyType(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ("text",),
        }
    ),
)


def _frag(
    index: int | None = None,
    id: str | None = None,
    name: str | None = None,
    args: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tc: dict[str, Any] = {"function": {}}
    if index is not None:
        tc["index"] = index
    if id is not None:
        tc["id"] = id
    if name is not None:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    if extra is not None:
        tc["extra_content"] = extra
    return tc


# --- tool-call accumulation -----------------------------------------------------------------


def test_llamacpp_fragments_merge_by_index_with_thai_split_mid_word() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "HC6M", "set_stream_title", "{")])
    for piece in ['"title": "ไพ', "ลิน", "เล่นเกม", " Minecraft", '"}']:
        acc.add([_frag(0, args=piece)])
    (call,) = acc.calls(keep_extra=False)
    assert call.id == "HC6M" and call.name == "set_stream_title"
    assert call.raw_arguments == '{"title": "ไพลินเล่นเกม Minecraft"}'
    assert call.arguments == {"title": "ไพลินเล่นเกม Minecraft"}


def test_parallel_calls_keep_arrival_order() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "a", "remember", '{"text":'), _frag(1, "b", "forget", '{"slot":')])
    acc.add([_frag(1, args=" 3}"), _frag(0, args=' "x"}')])
    calls = acc.calls(keep_extra=False)
    assert [c.name for c in calls] == ["remember", "forget"]
    assert calls[0].arguments == {"text": "x"} and calls[1].arguments == {"slot": 3}


def test_gemini_calls_without_index_key_on_id_and_keep_signature() -> None:
    sig = {"google": {"thought_signature": "U0lH"}}
    acc = ToolCallAccumulator()
    acc.add(
        [
            _frag(id="call_1", name="remember", args='{"text":"แมวชื่อส้ม"}', extra=sig),
            _frag(id="call_2", name="forget", args='{"slot":1}'),
        ]
    )
    kept = acc.calls(keep_extra=True)
    assert [c.id for c in kept] == ["call_1", "call_2"]
    assert kept[0].extra == sig and kept[1].extra is None
    assert all(c.extra is None for c in acc.calls(keep_extra=False))


def test_fragments_without_index_or_id_continue_the_last_call() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(name="remember", args='{"te')])
    acc.add([_frag(args='xt": "ok"}')])
    acc.add([_frag(name="forget", args="{}")])
    calls = acc.calls(keep_extra=False)
    assert [c.name for c in calls] == ["remember", "forget"]
    assert calls[0].arguments == {"text": "ok"} and calls[0].id == "call_0"
    assert calls[1].id == "call_1"


def test_index_then_id_only_fragment_joins_the_same_call() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "X1", "remember", '{"text": "a')])
    acc.add([_frag(id="X1", args='b"}')])
    (call,) = acc.calls(keep_extra=False)
    assert call.arguments == {"text": "ab"}


def test_nameless_calls_are_dropped() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "z", None, "{}")])
    assert acc.calls(keep_extra=False) == [] and len(acc) == 1


def test_sdk_objects_are_accepted() -> None:
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    tc = ChoiceDeltaToolCall.construct(
        index=None,
        id="g1",
        type="function",
        function=ChoiceDeltaToolCallFunction.construct(name="remember", arguments='{"text":"x"}'),
        extra_content={"google": {"thought_signature": "S"}},
    )
    acc = ToolCallAccumulator()
    acc.add([tc])
    (call,) = acc.calls(keep_extra=True)
    assert call.extra == {"google": {"thought_signature": "S"}}


# --- arguments ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"text": "ไพลิน"}', {"text": "ไพลิน"}),
        ("", {}),
        ("   ", {}),
        ('{"text": "ไพลิน', {"text": "ไพลิน"}),  # cut off by max_tokens: json_repair
        ('{"a": 1,}', {"a": 1}),
        ("[1, 2]", None),
        ("not json at all", None),
        ('"just a string"', None),
    ],
)
def test_parse_arguments(raw: str, expected: dict[str, Any] | None) -> None:
    assert parse_arguments(raw) == expected


# --- messages -------------------------------------------------------------------------------


def test_content_is_always_a_plain_string() -> None:
    assert content_text(None) == ""
    assert content_text("สวัสดี") == "สวัสดี"
    assert content_text([{"type": "text", "text": "สวั"}, {"type": "text", "text": "สดี"}]) == (
        "สวัสดี"
    )
    assert content_text([{"type": "image_url", "image_url": {}}, "x"]) == "x"


def test_sanitize_strips_extra_content_unless_gemini() -> None:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "p", "junk": 1},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "remember", "arguments": {"text": "x"}},
                    "extra_content": {"google": {"thought_signature": "S"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}]},
    ]
    other = sanitize_messages(history, keep_extra_content=False)
    assert other[0] == {"role": "system", "content": "p"}
    assert other[1]["content"] == ""
    call = other[1]["tool_calls"][0]
    assert "extra_content" not in call and call["function"]["arguments"] == '{"text": "x"}'
    assert other[2] == {"role": "tool", "content": "ok", "tool_call_id": "c1"}
    gemini = sanitize_messages(history, keep_extra_content=True)
    assert gemini[1]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "S"}}


def test_tool_payload_is_plain_json() -> None:
    (tool,) = tool_payload((REMEMBER,))
    assert tool == {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "จำข้อเท็จจริงข้ามสตรีม",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


# --- requests -------------------------------------------------------------------------------


def _req(**kw: Any) -> ChatRequest:
    return ChatRequest(messages=({"role": "user", "content": "hi"},), **kw)


def test_llamacpp_request_carries_slot_cache_and_parallel_tools() -> None:
    slots = {"speak": 0, "background": 2}
    kw = build_request(
        flavor="llamacpp",
        model="pailin-30b",
        req=_req(tools=(REMEMBER,), max_tokens=64),
        stream=True,
        slot_map=slots,
    )
    assert kw["extra_body"] == {"id_slot": 0, "cache_prompt": True, "parallel_tool_calls": True}
    assert kw["tools"][0]["function"]["name"] == "remember" and "tool_choice" not in kw
    assert kw["max_tokens"] == 64 and kw["stream"] is True
    bg = build_request(
        flavor="llamacpp", model="m", req=_req(purpose="background"), stream=True, slot_map=slots
    )
    assert bg["extra_body"] == {"id_slot": 2, "cache_prompt": True}
    explicit = build_request(flavor="llamacpp", model="m", req=_req(slot=1), stream=True)
    assert explicit["extra_body"]["id_slot"] == 1


def test_tools_are_sent_even_with_tool_choice_none() -> None:
    kw = build_request(
        flavor="llamacpp", model="m", req=_req(tools=(REMEMBER,), tool_choice="none"), stream=True
    )
    assert kw["tools"] and kw["tool_choice"] == "none"


def test_cloud_requests_have_no_llama_extras() -> None:
    typhoon = build_request(
        flavor="typhoon",
        model="typhoon-v2.5-30b-a3b-instruct",
        req=_req(tools=(REMEMBER,), slot=0),
        stream=True,
        extra_body={"repetition_penalty": 1.05},
    )
    assert typhoon["extra_body"] == {"repetition_penalty": 1.05}
    assert "reasoning_effort" not in typhoon
    gemini = build_request(
        flavor="gemini",
        model="gemini-3.5-flash-lite",
        req=_req(),
        stream=True,
        reasoning_effort="minimal",
    )
    assert gemini["reasoning_effort"] == "minimal" and "extra_body" not in gemini


def test_response_schema_becomes_json_schema_response_format() -> None:
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    kw = build_request(flavor="llamacpp", model="m", req=_req(response_schema=schema), stream=False)
    assert kw["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


# --- errors and messages --------------------------------------------------------------------


def test_error_detail_shapes() -> None:
    assert error_detail({"detail": "Invalid API Key"}) == "Invalid API Key"  # Typhoon
    assert error_detail({"error": {"code": 503, "message": "Loading model"}}) == "Loading model"
    assert error_detail({"code": 500, "message": "boom"}) == "boom"
    assert error_detail([{"error": {"message": "quota"}}]) == "quota"  # Gemini
    assert error_detail(None) == ""


def test_provider_error_is_a_provider_failed_and_pickles() -> None:
    err = ProviderError("x: stall", emitted=True, reason="stall", status=None, provider="x")
    assert isinstance(err, ProviderFailed)
    back = pickle.loads(pickle.dumps(err))
    assert (back.emitted, back.reason, back.provider, str(back)) == (True, "stall", "x", "x: stall")


def test_assistant_message_uses_raw_arguments() -> None:
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "c1", "remember", '{"text":"ok"}')])
    msg = assistant_message("ได้เลย", acc.calls(keep_extra=False))
    assert msg == {
        "role": "assistant",
        "content": "ได้เลย",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "remember", "arguments": '{"text":"ok"}'},
            }
        ],
    }
    assert assistant_message("", []) == {"role": "assistant", "content": ""}


def test_history_arguments_are_always_valid_json() -> None:
    # llama-server re-parses history arguments and answers HTTP 500 for broken JSON
    assert history_arguments('{"a": 1}') == '{"a": 1}'  # valid: byte-identical (cache)
    assert history_arguments('{"text": "แมวชื่อส') == '{"text": "แมวชื่อส"}'
    assert history_arguments("garbage") == "{}"
    assert history_arguments("[1]") == "{}"
    acc = ToolCallAccumulator()
    acc.add([_frag(0, "c1", "remember", '{"text": "ส้ม')])
    msg = assistant_message("", acc.calls(keep_extra=False))
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"text": "ส้ม"}'
    history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "remember", "arguments": '{"t'}}],
        }
    ]
    sent = sanitize_messages(history, keep_extra_content=False)
    # json_repair reads '{"t' as a list, which is not an object: fall back to an empty object
    args = sent[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(json.loads(args), dict)
    assert args == "{}"
    # every truncation of a real call still yields a JSON object
    whole = '{"text": "แมวชื่อส้ม", "importance": 3}'
    for cut in range(len(whole) + 1):
        assert isinstance(json.loads(history_arguments(whole[:cut])), dict), whole[:cut]


def _timeout_from(cause: BaseException) -> BaseException:
    import httpx2
    import openai

    try:
        try:
            raise cause
        except BaseException as inner:
            raise openai.APITimeoutError(
                request=httpx2.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
            ) from inner
    except openai.APITimeoutError as exc:
        return exc


def test_connect_timeout_maps_to_connect_failure() -> None:
    import httpx2

    from aivtube.llm.openai_stream import map_error

    exc = _timeout_from(httpx2.ConnectTimeout("connect timed out"))
    err = map_error(exc, provider="p", emitted=False, phase="connect")
    assert err.reason == "connect" and err.emitted is False


def test_read_timeout_stays_a_timeout() -> None:
    import httpx2

    from aivtube.llm.openai_stream import map_error

    exc = _timeout_from(httpx2.ReadTimeout("read timed out"))
    assert map_error(exc, provider="p", emitted=False, phase="connect").reason == "timeout"
    exc = _timeout_from(httpx2.ConnectTimeout("x"))
    assert map_error(exc, provider="p", emitted=True, phase="stream").reason == "timeout"
