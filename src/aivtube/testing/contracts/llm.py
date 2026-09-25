"""Contract suites for ``LLMProvider``, ``LLMRouter`` and ``LocalServerManager`` (§3.6)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    LLMProvider,
    LLMRouter,
    LocalServerManager,
    ProviderCaps,
    ProviderFailed,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from aivtube.contracts.types import Health
from aivtube.testing.contracts._base import AsyncCase, _Cases, check, maybe_await, raises
from aivtube.testing.fakes.llm import FakeLLM, FakeReply, tool_call

__all__ = [
    "SET_TITLE_SPEC",
    "llm_provider_suite",
    "llm_router_suite",
    "local_server_manager_suite",
    "set_title_call",
]

SET_TITLE_SPEC = ToolSpec(
    name="set_stream_title",
    description="Change the live stream title",
    parameters={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
)


def _request(prompt: str, tools: tuple[ToolSpec, ...] = ()) -> ChatRequest:
    return ChatRequest(
        messages=(
            {"role": "system", "content": "คุณคือไพลิน ตอบสั้นๆ"},
            {"role": "user", "content": prompt},
        ),
        tools=tools,
        max_tokens=48,
        temperature=0.0,
        character="pailin",
        turn_id="contract",
        first_token_timeout_s=30.0,
    )


async def _collect(stream: AsyncIterator[LLMEvent]) -> list[LLMEvent]:
    return [ev async for ev in stream]


def _check_shape(events: Sequence[LLMEvent], name: str) -> Done:
    check(events and isinstance(events[-1], Done), "the stream must end with Done")
    check(sum(isinstance(e, Done) for e in events) == 1, "exactly one Done")
    done = events[-1]
    assert isinstance(done, Done)
    check(isinstance(done.provider, str) and done.provider, "Done.provider")
    check(done.ttft_ms >= 0.0, "Done.ttft_ms")
    first_call = next((i for i, e in enumerate(events) if isinstance(e, ToolCall)), len(events))
    check(
        not any(isinstance(e, TextDelta) for e in events[first_call:]),
        "ToolCalls must come after every TextDelta",
    )
    check(done.assistant_message.get("role") == "assistant", "assistant_message.role")
    return done


def llm_provider_suite(
    factory: Callable[[], LLMProvider | Awaitable[LLMProvider]],
    *,
    prompt: str = "สวัสดี",
    expect_text: str | None = None,
    tool_prompt: str | None = None,
    tool: ToolSpec = SET_TITLE_SPEC,
    closed_probe: Callable[[LLMProvider], Awaitable[bool]] | None = None,
) -> list[AsyncCase]:
    """The provider must answer ``prompt`` with text (``expect_text`` exactly, if given) and,
    when ``tool_prompt`` is given, answer it with a call to ``tool``. ``closed_probe`` confirms
    that closing the stream early reached the transport (e.g. the server saw a disconnect)."""
    cases = _Cases("llm_provider")

    async def make() -> LLMProvider:
        return await maybe_await(factory())

    @cases
    async def attributes_and_probe() -> None:
        p = await make()
        check(isinstance(p.name, str) and p.name, "name")
        check(isinstance(p.caps, ProviderCaps), "caps")
        check(isinstance(await p.probe(), Health), "probe() must return Health")

    @cases
    async def streams_text_then_done() -> None:
        p = await make()
        events = await _collect(p.stream(_request(prompt)))
        done = _check_shape(events, p.name)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        check(text, "no text streamed")
        if expect_text is not None:
            check(text == expect_text, f"text {text!r} != {expect_text!r}")
        content = done.assistant_message.get("content")
        check(content in (text, None) or content == text, "assistant_message.content")

    if tool_prompt is not None:

        @cases
        async def tool_calls_after_text() -> None:
            p = await make()
            events = await _collect(p.stream(_request(tool_prompt, (tool,))))
            done = _check_shape(events, p.name)
            calls = [e for e in events if isinstance(e, ToolCall)]
            check(calls, "no ToolCall")
            call = calls[0]
            check(call.name == tool.name, f"called {call.name!r}")
            check(isinstance(call.raw_arguments, str), "raw_arguments")
            check(call.arguments is None or isinstance(call.arguments, Mapping), "arguments")
            check(isinstance(call.id, str) and call.id, "id")
            check(done.finish_reason == "tool_calls", f"finish_reason {done.finish_reason!r}")

    @cases
    async def aclose_mid_stream_is_clean() -> None:
        p = await make()
        stream: Any = p.stream(_request(prompt))
        first = await stream.__anext__()
        check(isinstance(first, TextDelta | ToolCall | Done), "unexpected event type")
        await stream.aclose()
        if closed_probe is not None:
            check(await closed_probe(p), "closing the stream did not reach the transport")

    @cases
    async def prefill_returns_seconds() -> None:
        p = await make()
        took = await p.prefill(_request(prompt))
        check(isinstance(took, float) and took >= 0.0, f"prefill returned {took!r}")

    return cases.items


def llm_router_suite(
    factory: Callable[[Sequence[LLMProvider]], LLMRouter | Awaitable[LLMRouter]],
) -> list[AsyncCase]:
    """``factory(providers)`` builds a router whose chain is ``providers`` in order (all local,
    enabled, no consent needed). The suite supplies ``FakeLLM`` providers named a and b."""
    cases = _Cases("llm_router")

    def fake(name: str, text: str, **kw: Any) -> FakeLLM:
        return FakeLLM([FakeReply("", text, repeat=True)], ttft_s=0.0, tok_s=0.0, name=name, **kw)

    async def run(router: LLMRouter) -> list[LLMEvent]:
        return await _collect(router.stream(_request("สวัสดี")))

    @cases
    async def uses_the_first_provider() -> None:
        a, b = fake("a", "จากเอ"), fake("b", "จากบี")
        events = await run(await maybe_await(factory([a, b])))
        done = events[-1]
        check(isinstance(done, Done) and done.provider == "a", f"served by {done!r}")
        check(not b.requests, "the second provider was called")

    @cases
    async def falls_back_before_the_first_event() -> None:
        a, b = fake("a", "จากเอ", fail="connect"), fake("b", "จากบี")
        events = await run(await maybe_await(factory([a, b])))
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        check(text == "จากบี", f"fallback text {text!r}")
        done = events[-1]
        check(isinstance(done, Done) and done.provider == "b", "fallback provider")

    @cases
    async def no_fallback_after_emission() -> None:
        a, b = fake("a", "สวัสดีค่ะทุกคนวันนี้", fail="mid_stream"), fake("b", "จากบี")
        router = await maybe_await(factory([a, b]))
        seen: list[LLMEvent] = []
        with raises(ProviderFailed, what="a mid-stream failure"):
            async for ev in router.stream(_request("สวัสดี")):
                seen.append(ev)
        check(seen, "nothing was emitted before the failure")
        check(not b.requests, "the router fell back after emitting")

    @cases
    async def promote_then_rollback() -> None:
        a, b = fake("a", "จากเอ"), fake("b", "จากบี")
        router = await maybe_await(factory([a, b]))
        router.promote("b")
        done = (await run(router))[-1]
        check(isinstance(done, Done) and done.provider == "b", "promote() ignored")
        check(router.active() == "b", f"active() == {router.active()!r}")
        router.rollback()
        done = (await run(router))[-1]
        check(isinstance(done, Done) and done.provider == "a", "rollback() ignored")

    @cases
    async def status_lists_every_provider() -> None:
        a, b = fake("a", "เอ"), fake("b", "บี")
        router = await maybe_await(factory([a, b]))
        st = router.status()
        check({s.name for s in st} >= {"a", "b"}, f"status names {[s.name for s in st]}")
        check(sum(s.active for s in st) == 1, "exactly one active provider")

    return cases.items


def local_server_manager_suite(
    factory: Callable[[], LocalServerManager | Awaitable[LocalServerManager]],
    *,
    server: str = "local30b",
    missing: str = "no-such-server",
) -> list[AsyncCase]:
    cases = _Cases("local_server_manager")

    @cases
    async def ensure_then_props() -> None:
        m = await maybe_await(factory())
        check(await m.ensure_running(server, 30.0) is True, "ensure_running failed")
        props = await m.props(server)
        caps = props.get("chat_template_caps") or {}
        check(caps.get("supports_tool_calls") is True, "props must report tool-call support")
        await m.stop(server)

    @cases
    async def unknown_server_is_false_not_error() -> None:
        m = await maybe_await(factory())
        check(await m.ensure_running(missing, 0.5) is False, "unknown server reported running")

    @cases
    async def save_then_restore_slot() -> None:
        m = await maybe_await(factory())
        await m.ensure_running(server, 30.0)
        check(await m.save_slot(server, 0, "pailin-contract.bin") is True, "save_slot")
        check(await m.restore_slot(server, 0, "pailin-contract.bin") is True, "restore_slot")
        await m.stop(server)

    @cases
    async def restart_after_stop() -> None:
        m = await maybe_await(factory())
        await m.ensure_running(server, 30.0)
        await m.stop(server)
        check(await m.ensure_running(server, 30.0) is True, "could not start again after stop()")
        await m.stop(server)

    return cases.items


def set_title_call(title: str) -> ToolCall:
    """A ``set_stream_title`` call, for harnesses that script providers for this suite."""
    return tool_call("set_stream_title", {"title": title})
