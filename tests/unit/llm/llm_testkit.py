"""Helpers for driving ``OpenAICompatProvider`` through ``httpx2.MockTransport``.

``MockLLM`` answers each request with the next scripted reply: a ``Body`` (a 200 SSE stream
whose parts can sleep on the test clock or raise), an ``httpx2.Response``, or an exception
(``Fail``: the transport raises, e.g. ``httpx2.ConnectError``). ``Body`` records ``aclose()``, which is
how a test sees that closing the provider's stream reached the transport.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2

from aivtube.contracts.infra import Clock
from aivtube.contracts.llm import ChatRequest, LLMEvent
from aivtube.llm.openai_stream import Flavor
from aivtube.llm.providers import OpenAICompatProvider, ProviderCfg

BASE_URL = "http://llama.test/v1"


def chunk(
    delta: Mapping[str, Any] | None = None,
    finish: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One ``chat.completion.chunk`` object."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "pailin-30b",
        "choices": [{"index": 0, "delta": dict(delta or {}), "finish_reason": finish}],
        **extra,
    }


def frame(obj: Mapping[str, Any] | str) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


def sse(*chunks: Mapping[str, Any], done: bool = True) -> bytes:
    return b"".join(frame(c) for c in chunks) + (b"data: [DONE]\n\n" if done else b"")


def text_chunks(*pieces: str) -> list[dict[str, Any]]:
    return [chunk({"content": p}) for p in pieces]


ROLE = chunk({"role": "assistant", "content": None})


def timings(prompt_n: int = 12, cache_n: int = 288, predicted_n: int = 5) -> dict[str, Any]:
    return {"prompt_n": prompt_n, "cache_n": cache_n, "predicted_n": predicted_n}


@dataclass(frozen=True)
class Sleep:
    seconds: float


class Body(httpx2.AsyncByteStream):
    """A scripted SSE response body. Parts: ``bytes`` (sent), ``Sleep`` (awaits the clock) or
    an exception (raised mid-stream, like a dropped connection)."""

    def __init__(
        self,
        parts: Iterable[bytes | Sleep | BaseException],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.parts = list(parts)
        self._sleep = sleep
        self.sent = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            if isinstance(part, Sleep):
                await self._sleep(part.seconds)
            elif isinstance(part, BaseException):
                raise part
            else:
                self.sent += 1
                yield part

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class Fail:
    """Make the transport raise ``error(message, request=…)`` (e.g. ``httpx2.ConnectError``)."""

    error: type[httpx2.RequestError] = httpx2.ConnectError
    message: str = "connection refused"


Reply = Body | httpx2.Response | Fail | Callable[[httpx2.Request], httpx2.Response]


class MockLLM:
    """A fake OpenAI-compatible server behind ``httpx2.MockTransport``."""

    def __init__(self, *replies: Reply) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.requests: list[httpx2.Request] = []
        self.bodies: list[Body] = []

    def add(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self.replies:
            return httpx2.Response(500, json={"error": {"message": "no scripted reply"}})
        reply = self.replies.popleft()
        if isinstance(reply, Fail):
            raise reply.error(reply.message, request=request)
        if isinstance(reply, Body):
            self.bodies.append(reply)
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=reply)
        if isinstance(reply, httpx2.Response):
            return reply
        return reply(request)

    async def wait_requests(self, n: int = 1, timeout_s: float = 5.0) -> None:
        """Wait in real time until ``n`` requests arrived. The openai SDK detects the platform
        in a worker thread on a client's first request, which a FakeClock cannot see; call this
        before driving the fake clock."""
        end = time.perf_counter() + timeout_s
        while len(self.requests) < n:
            if time.perf_counter() > end:
                raise TimeoutError(f"only {len(self.requests)} of {n} requests arrived")
            await asyncio.sleep(0.001)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))

    def body(self, i: int = -1) -> dict[str, Any]:
        """The JSON body of request ``i``."""
        data: dict[str, Any] = json.loads(self.requests[i].content)
        return data


def make_provider(
    mock: MockLLM,
    *,
    flavor: Flavor = "llamacpp",
    clock: Clock | None = None,
    name: str | None = None,
    api_key: str | None = None,
    **cfg: Any,
) -> OpenAICompatProvider:
    cloud = flavor in ("typhoon", "gemini")
    env = cfg.pop("api_key_env", "TEST_LLM_KEY" if cloud else None)
    pc = ProviderCfg(
        name=name or f"test-{flavor}",
        kind="llamacpp" if flavor == "llamacpp" else "openai_compat",
        flavor=flavor,
        base_url=BASE_URL,
        model="pailin-30b",
        api_key_env=env,
        cloud=cloud,
        slot_map={"speak": 0, "background": 2, "game": 3} if flavor == "llamacpp" else {},
        **cfg,
    )
    secrets = {env: api_key if api_key is not None else "sk-test"} if env else {}
    return OpenAICompatProvider(
        pc, secrets=_Secrets(secrets), http_client=mock.client(), clock=clock
    )


class _Secrets:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = dict(values)

    def get(self, name: str) -> str | None:
        return self.values.get(name)


def request(prompt: str = "สวัสดีไพลิน", **kw: Any) -> ChatRequest:
    return ChatRequest(
        messages=(
            {"role": "system", "content": "คุณคือไพลิน"},
            {"role": "user", "content": prompt},
        ),
        **kw,
    )


async def collect(stream: AsyncIterator[LLMEvent]) -> list[LLMEvent]:
    return [ev async for ev in stream]
