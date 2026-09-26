"""LLM fakes: ``FakeReply``, ``FakeLLM``, ``FakeLLMRouter`` and ``FakeLauncher`` (§3.6, §10)."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from aivtube.contracts.infra import Clock
from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    LLMProvider,
    ProviderCaps,
    ProviderFailed,
    ProviderStatus,
    TextDelta,
    ToolCall,
)
from aivtube.contracts.types import Health, HealthState

__all__ = [
    "THAI_COMBINING",
    "FailMode",
    "FakeLLM",
    "FakeLLMRouter",
    "FakeReply",
    "ReplyScript",
    "assistant_message",
    "last_user_text",
    "split_deltas",
    "tool_call",
]

THAI_COMBINING = frozenset(chr(c) for c in (0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)))
"""Thai marks that attach to the previous character (never cut before one)."""

FailMode = Literal["connect", "timeout", "mid_stream", "stall", "down"]


@dataclass(frozen=True)
class FakeReply:
    """One scripted completion, chosen when ``match`` occurs in the last user message.

    ``match=""`` matches anything. Replies are consumed in order unless ``repeat`` is set.
    Optional behaviour knobs (used by ``FakeLLM`` and ``SseFixtureServer``): ``ttft_s``,
    ``stall_after``/``stall_s`` (pause after N text chunks), ``die_after`` (the stream dies after
    N text chunks), ``status`` (HTTP error, SSE only), ``no_index`` (Gemini-style tool-call
    chunks: complete, without ``index``), ``prompt_n``/``cache_n`` (timings override) and
    ``raw_sse`` (SSE only: replay recorded bytes verbatim, e.g. ``tests/fixtures/sse/*.sse``).
    """

    match: str
    text: str
    tool_calls: Sequence[ToolCall] = ()
    ttft_s: float | None = None
    stall_after: int | None = None
    stall_s: float = 0.0
    die_after: int | None = None
    status: int = 200
    no_index: bool = False
    finish_reason: str | None = None
    prompt_n: int | None = None
    cache_n: int | None = None
    repeat: bool = False
    raw_sse: bytes | None = None

    @property
    def finish(self) -> str:
        if self.finish_reason is not None:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


def tool_call(
    name: str,
    arguments: Mapping[str, Any],
    *,
    id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> ToolCall:
    """Build a ``ToolCall`` whose ``raw_arguments`` is the compact JSON of ``arguments``."""
    raw = json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"))
    return ToolCall(
        id=id or f"call_{uuid.uuid4().hex[:12]}",
        name=name,
        arguments=dict(arguments),
        raw_arguments=raw,
        extra=extra,
    )


def split_deltas(text: str, max_chars: int = 3, *, split_marks: bool = True) -> list[str]:
    """Split ``text`` into streaming deltas of at most ``max_chars`` characters.

    With ``split_marks`` every Thai combining mark starts a new delta, reproducing llama.cpp
    splitting 'โอ' + '้' across SSE chunks.
    """
    out: list[str] = []
    cur = ""
    for ch in text:
        if cur and (len(cur) >= max_chars or (split_marks and ch in THAI_COMBINING)):
            out.append(cur)
            cur = ""
        cur += ch
    if cur:
        out.append(cur)
    return out


def last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Text of the last ``user`` message (or of the last message when there is none)."""
    chosen: Mapping[str, Any] | None = None
    for m in reversed(messages):
        if m.get("role") == "user":
            chosen = m
            break
    if chosen is None and messages:
        chosen = messages[-1]
    if chosen is None:
        return ""
    content = chosen.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, Mapping))
    return ""


def assistant_message(text: str, calls: Sequence[ToolCall]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.raw_arguments},
            }
            for c in calls
        ]
    return msg


class ReplyScript:
    """Thread-safe reply selection shared by ``FakeLLM`` and ``SseFixtureServer``."""

    def __init__(self, replies: Sequence[FakeReply], default: FakeReply | None = None) -> None:
        self._items = list(replies)
        self.default = default
        self._lock = threading.Lock()

    def take(self, messages: Sequence[Mapping[str, Any]]) -> FakeReply | None:
        text = last_user_text(messages)
        with self._lock:
            for i, r in enumerate(self._items):
                if r.match in text:
                    if not r.repeat:
                        del self._items[i]
                    return r
        return self.default

    def add(self, reply: FakeReply) -> None:
        with self._lock:
            self._items.append(reply)

    @property
    def remaining(self) -> int:
        return len(self._items)


class _PromptCache:
    """Estimates llama.cpp ``prompt_n``/``cache_n`` from the longest common prefix."""

    def __init__(self) -> None:
        self._last: dict[int | None, str] = {}

    def account(
        self, req_messages: Sequence[Mapping[str, Any]], slot: int | None
    ) -> tuple[int, int]:
        rendered = json.dumps(list(req_messages), ensure_ascii=False, sort_keys=True, default=str)
        prev = self._last.get(slot, "")
        common = 0
        for a, b in zip(prev, rendered, strict=False):
            if a != b:
                break
            common += 1
        self._last[slot] = rendered
        cache_n = common // 4
        prompt_n = max(1, (len(rendered) - common) // 4)
        return prompt_n, cache_n


class FakeLLM:
    """``LLMProvider`` streaming scripted replies.

    Text is streamed in ``split_deltas`` pieces (Thai combining marks split), one per
    ``1 / tok_s`` seconds after ``ttft_s``; tool calls follow the text, then ``Done``.
    ``fail`` simulates a provider failure for every request: ``connect`` and ``timeout`` fail
    before the first event (``emitted=False``), ``mid_stream`` dies halfway (``emitted=True``),
    ``stall`` hangs halfway until the stream is closed, and ``down`` also makes ``probe`` DOWN.
    Instrumentation: ``requests``, ``in_flight``/``max_in_flight``, ``closed_early`` (the
    consumer called ``aclose()`` or was cancelled before the end), ``completed``.
    """

    def __init__(
        self,
        script: Sequence[FakeReply],
        *,
        ttft_s: float = 0.3,
        tok_s: float = 45.0,
        fail: FailMode | None = None,
        name: str = "fake",
        clock: Clock | None = None,
        default: FakeReply | None = None,
        cloud: bool = False,
    ) -> None:
        self.name = name
        self.caps = ProviderCaps(
            tools=True,
            parallel_tools=True,
            json_schema=True,
            prompt_cache=not cloud,
            cloud=cloud,
            keeps_thought_signatures=False,
            reasoning="none",
        )
        self.script = ReplyScript(script, default)
        self.ttft_s = ttft_s
        self.tok_s = tok_s
        self.fail: FailMode | None = fail
        self._sleep: Callable[[float], Awaitable[None]] = clock.sleep if clock else asyncio.sleep
        self._cache = _PromptCache()
        self.requests: list[ChatRequest] = []
        self.prefills: list[ChatRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed_early = 0
        self.completed = 0

    async def probe(self) -> Health:
        state = HealthState.DOWN if self.fail == "down" else HealthState.OK
        return Health(self.name, state)

    async def prefill(self, req: ChatRequest) -> float:
        self.prefills.append(req)
        self._cache.account(req.messages, req.slot)
        return 0.0

    async def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(req)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        finished = False
        try:
            if self.fail in ("connect", "down"):
                raise ProviderFailed(f"{self.name}: connection refused", emitted=False)
            if self.fail == "timeout":
                await self._sleep(req.first_token_timeout_s or 4.0)
                raise ProviderFailed(f"{self.name}: first-token timeout", emitted=False)
            reply = self.script.take(req.messages)
            if reply is None:
                raise ProviderFailed(f"{self.name}: no scripted reply", emitted=False)
            prompt_n, cache_n = self._cache.account(req.messages, req.slot)
            ttft = self.ttft_s if reply.ttft_s is None else reply.ttft_s
            await self._sleep(ttft)
            chunks = split_deltas(reply.text)
            half = max(1, len(chunks) // 2)
            for i, chunk in enumerate(chunks):
                if self.fail == "mid_stream" and i >= half:
                    raise ProviderFailed(f"{self.name}: died mid-stream", emitted=True)
                if self.fail == "stall" and i >= half:
                    await self._sleep(1e9)
                if reply.die_after is not None and i >= reply.die_after:
                    raise ProviderFailed(f"{self.name}: stream died", emitted=i > 0)
                if reply.stall_after is not None and i == reply.stall_after:
                    await self._sleep(reply.stall_s)
                yield TextDelta(chunk)
                if self.tok_s > 0 and i < len(chunks) - 1:
                    await self._sleep(1.0 / self.tok_s)
            for call in reply.tool_calls:
                yield call
            yield Done(
                provider=self.name,
                finish_reason=reply.finish,
                ttft_ms=ttft * 1000.0,
                prompt_n=reply.prompt_n if reply.prompt_n is not None else prompt_n,
                cache_n=reply.cache_n if reply.cache_n is not None else cache_n,
                completion_tokens=len(chunks),
                assistant_message=assistant_message(reply.text, reply.tool_calls),
            )
            finished = True
            self.completed += 1
        finally:
            self.in_flight -= 1
            if not finished:
                self.closed_early += 1


@dataclass
class _RouterEntry:
    provider: LLMProvider
    enabled: bool = True
    fails: int = 0
    down_until: float = 0.0
    ttfts: list[float] = field(default_factory=list)


class FakeLLMRouter:
    """``LLMRouter`` over a list of providers: falls back only before the first event."""

    def __init__(self, providers: Sequence[LLMProvider], *, clock: Clock | None = None) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._entries = {p.name: _RouterEntry(p) for p in providers}
        self._order = [p.name for p in providers]
        self._active = self._order[0]
        self._previous: str | None = None
        self._clock = clock
        self.switches: list[tuple[str, str, str]] = []

    def active(self) -> str:
        return self._active

    def promote(self, name: str) -> None:
        if name not in self._entries:
            raise KeyError(name)
        if name != self._active:
            self._previous, self._active = self._active, name
            self.switches.append((self._previous, name, "promote"))

    def rollback(self) -> None:
        if self._previous is not None:
            self.switches.append((self._active, self._previous, "rollback"))
            self._active, self._previous = self._previous, None

    def status(self) -> list[ProviderStatus]:
        out = []
        for name in self._order:
            e = self._entries[name]
            p50 = sorted(e.ttfts)[len(e.ttfts) // 2] if e.ttfts else None
            out.append(
                ProviderStatus(
                    name=name,
                    healthy=e.fails == 0,
                    active=name == self._active,
                    enabled=e.enabled,
                    cloud=e.provider.caps.cloud,
                    down_until=e.down_until,
                    fails=e.fails,
                    ttft_p50_ms=p50,
                )
            )
        return out

    async def prefill(self, req: ChatRequest) -> None:
        await self._entries[self._active].provider.prefill(req)

    async def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]:
        chain = [self._active, *[n for n in self._order if n != self._active]]
        last: BaseException | None = None
        for name in chain:
            entry = self._entries[name]
            if not entry.enabled:
                continue
            emitted = False
            agen = entry.provider.stream(req)
            try:
                async for ev in agen:
                    emitted = True
                    if isinstance(ev, Done):
                        entry.ttfts.append(ev.ttft_ms)
                    yield ev
                entry.fails = 0
                return
            except ProviderFailed as exc:
                entry.fails += 1
                last = exc
                if emitted or exc.emitted:
                    raise ProviderFailed(str(exc), emitted=True) from exc
            except Exception as exc:
                entry.fails += 1
                last = exc
                if emitted:
                    raise ProviderFailed(
                        f"{name} failed mid-stream: {exc!r}", emitted=True
                    ) from exc
            finally:
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    await aclose()
        raise ProviderFailed("all providers failed", emitted=False) from last
