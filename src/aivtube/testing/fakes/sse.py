"""``SseFixtureServer``: an OpenAI-compatible llama-server double over real HTTP (§10).

It streams ``chat.completion.chunk`` SSE exactly like llama-server: a role chunk, content
deltas with Thai combining marks split across chunks, llama.cpp tool-call fragments (first
delta has index, id, type and name; later deltas carry only index and argument fragments), a
final chunk with ``finish_reason`` and a ``timings`` object, then ``data: [DONE]``. Per-reply
knobs (``FakeReply``) add stalls, mid-stream death, HTTP errors and Gemini-style tool calls.
It also serves ``/health``, ``/props``, ``/v1/models``, ``/slots/{id}`` and ``/completion``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from aiohttp import web

from aivtube.testing.fakes._http import HttpFixture
from aivtube.testing.fakes.launcher import DEFAULT_PROPS
from aivtube.testing.fakes.llm import FakeReply, ReplyScript, _PromptCache, split_deltas

__all__ = ["SseFixtureServer", "chunk_fragments", "sse_chunks"]


def chunk_fragments(raw: str, size: int = 6) -> list[str]:
    """Split tool-call argument JSON into fragments (Thai words get cut mid-word)."""
    return [raw[i : i + size] for i in range(0, len(raw), size)] or [""]


def sse_chunks(
    reply: FakeReply,
    *,
    model: str = "pailin-30b",
    completion_id: str | None = None,
    created: int | None = None,
    prompt_n: int = 1,
    cache_n: int = 0,
    fragment_size: int = 6,
) -> list[dict[str, Any]]:
    """The JSON chunk objects llama-server would stream for ``reply`` (without ``[DONE]``)."""
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    ts = int(time.time()) if created is None else created

    def chunk(delta: Mapping[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
        return {
            "choices": [{"finish_reason": finish, "index": 0, "delta": dict(delta)}],
            "created": ts,
            "id": cid,
            "model": model,
            "system_fingerprint": "b11177-fake",
            "object": "chat.completion.chunk",
            **extra,
        }

    out = [chunk({"role": "assistant", "content": None})]
    pieces = split_deltas(reply.text)
    out += [chunk({"content": p}) for p in pieces]
    for i, call in enumerate(reply.tool_calls):
        if reply.no_index:  # Gemini: the whole call in one chunk, no index, extra_content
            tc: dict[str, Any] = {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            if call.extra:
                tc["extra_content"] = dict(call.extra)
            out.append(chunk({"tool_calls": [tc]}))
            continue
        frags = chunk_fragments(call.raw_arguments, fragment_size)
        first = {
            "index": i,
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": frags[0]},
        }
        out.append(chunk({"tool_calls": [first]}))
        out += [
            chunk({"tool_calls": [{"index": i, "function": {"arguments": f}}]}) for f in frags[1:]
        ]
    n_pred = len(pieces) + sum(len(chunk_fragments(c.raw_arguments)) for c in reply.tool_calls)
    timings = {
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "prompt_ms": 12.5,
        "prompt_per_token_ms": 12.5 / max(1, prompt_n),
        "prompt_per_second": 1000.0 * max(1, prompt_n) / 12.5,
        "predicted_n": n_pred,
        "predicted_ms": 22.0 * n_pred,
        "predicted_per_token_ms": 22.0,
        "predicted_per_second": 45.45,
    }
    out.append(chunk({}, reply.finish, timings=timings))
    return out


class SseFixtureServer(HttpFixture):
    """Fake llama-server on ``host:port`` (``port=0`` = any free port).

    ``start()`` returns the OpenAI base URL (``http://127.0.0.1:<port>/v1``). Instrumentation:
    ``requests`` (JSON bodies), ``completed``, ``disconnects`` (the client went away mid-stream),
    ``killed`` (streams dropped on purpose by ``die_after``), ``slot_actions``. ``loading=True`` makes ``/health`` answer 503 like a model still loading.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        script: Sequence[FakeReply] = (),
        *,
        ttft_s: float = 0.0,
        chunk_delay_s: float = 0.0,
        model_alias: str = "pailin-30b",
        props: Mapping[str, Any] | None = None,
        default: FakeReply | None = None,
    ) -> None:
        super().__init__(host, port)
        self.script = ReplyScript(script, default)
        self.ttft_s = ttft_s
        self.chunk_delay_s = chunk_delay_s
        self.model_alias = model_alias
        self.props: dict[str, Any] = {**DEFAULT_PROPS, "model_alias": model_alias, **(props or {})}
        self.loading = False
        self.requests: list[dict[str, Any]] = []
        self.completed = 0
        self.disconnects = 0
        self.slot_actions: list[tuple[int, str, str]] = []
        self.killed = 0
        self._cache = _PromptCache()

    @property
    def base_url(self) -> str:
        return f"{self.root_url}/v1"

    async def start(self) -> str:
        await self._start_http()
        return self.base_url

    async def stop(self) -> None:
        await self._stop_http()

    def add(self, reply: FakeReply) -> None:
        self.script.add(reply)

    async def wait_disconnects(self, n: int = 1, timeout: float = 2.0) -> bool:  # noqa: ASYNC109
        """Wait until at least ``n`` clients went away mid-stream (real-time ``timeout``)."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if self.disconnects >= n:
                return True
            await asyncio.sleep(0.01)
        return self.disconnects >= n

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/chat/completions", self._chat)
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/health", self._health)
        app.router.add_get("/props", self._props)
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/slots/{slot}", self._slots)
        app.router.add_post("/completion", self._completion)
        return app

    # --- handlers ---------------------------------------------------------------------------
    async def _health(self, request: web.Request) -> web.Response:
        if self.loading:
            return web.json_response(
                {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
                status=503,
            )
        return web.json_response({"status": "ok"})

    async def _props(self, request: web.Request) -> web.Response:
        return web.json_response(self.props)

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": self.model_alias, "object": "model", "owned_by": "llamacpp"}],
            }
        )

    async def _slots(self, request: web.Request) -> web.Response:
        slot = int(request.match_info["slot"])
        action = request.query.get("action", "")
        body = await request.json() if request.can_read_body else {}
        filename = str(body.get("filename", ""))
        self.slot_actions.append((slot, action, filename))
        if action not in ("save", "restore", "erase"):
            return web.json_response({"error": {"code": 400, "message": "bad action"}}, status=400)
        key = "n_saved" if action == "save" else "n_restored"
        return web.json_response({"id_slot": slot, "filename": filename, key: 123})

    async def _completion(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        return web.json_response(
            {"content": "", "stop": True, "timings": {"prompt_n": 1, "cache_n": 0}}
        )

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body: dict[str, Any] = await request.json()
        self.requests.append(body)
        messages = body.get("messages") or []
        reply = self.script.take(messages)
        if reply is None:
            return web.json_response(
                {"error": {"code": 500, "message": "no scripted reply", "type": "server_error"}},
                status=500,
            )
        if reply.status != 200:
            return web.json_response(
                {"error": {"code": reply.status, "message": "scripted error", "type": "error"}},
                status=reply.status,
            )
        prompt_n, cache_n = self._cache.account(messages, body.get("id_slot"))
        prompt_n = reply.prompt_n if reply.prompt_n is not None else prompt_n
        cache_n = reply.cache_n if reply.cache_n is not None else cache_n
        chunks = sse_chunks(reply, model=self.model_alias, prompt_n=prompt_n, cache_n=cache_n)
        if reply.raw_sse is not None:
            return await self._replay(request, reply.raw_sse)
        if not body.get("stream"):
            return web.json_response(self._non_stream(reply, chunks))
        return await self._stream(request, body, reply, chunks)

    async def _replay(self, request: web.Request, raw: bytes) -> web.StreamResponse:
        """Send recorded SSE bytes event by event (``chunk_delay_s`` apart)."""
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for event in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
                if event.strip():
                    await resp.write(event + b"\n\n")
                    if self.chunk_delay_s:
                        await asyncio.sleep(self.chunk_delay_s)
            await resp.write_eof()
            self.completed += 1
        except (ConnectionError, asyncio.CancelledError):
            self.disconnects += 1
            raise
        return resp

    def _non_stream(self, reply: FakeReply, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments},
                }
                for c in reply.tool_calls
            ]
        last = chunks[-1]
        self.completed += 1
        return {
            "id": last["id"],
            "object": "chat.completion",
            "created": last["created"],
            "model": last["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": reply.finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "timings": last["timings"],
        }

    async def _stream(
        self,
        request: web.Request,
        body: Mapping[str, Any],
        reply: FakeReply,
        chunks: list[dict[str, Any]],
    ) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        ttft = self.ttft_s if reply.ttft_s is None else reply.ttft_s
        text_chunks = len(split_deltas(reply.text))
        try:
            await resp.write(self._frame(chunks[0]))
            if ttft:
                await asyncio.sleep(ttft)
            for i, ch in enumerate(chunks[1:]):
                is_text = i < text_chunks
                if is_text and reply.die_after is not None and i >= reply.die_after:
                    self.killed += 1  # the server drops the connection on purpose
                    if request.transport is not None:
                        request.transport.abort()
                    return resp
                if is_text and reply.stall_after is not None and i == reply.stall_after:
                    await asyncio.sleep(reply.stall_s)
                await resp.write(self._frame(ch))
                if self.chunk_delay_s:
                    await asyncio.sleep(self.chunk_delay_s)
            options = body.get("stream_options") or {}
            if options.get("include_usage"):
                usage = {
                    **{k: v for k, v in chunks[-1].items() if k != "timings"},
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                await resp.write(self._frame(usage))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            self.completed += 1
        except (ConnectionError, asyncio.CancelledError):
            self.disconnects += 1
            raise
        return resp

    @staticmethod
    def _frame(obj: Mapping[str, Any]) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
