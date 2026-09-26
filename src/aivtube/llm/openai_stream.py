"""One streamed OpenAI-compatible chat completion as ``LLMEvent``s (§3.6, §4.6, §4.9, §4.11).

``stream_turn`` sends one ``/v1/chat/completions`` request with the openai SDK (httpx2) and
yields ``TextDelta``s as they arrive, then every ``ToolCall`` (the stream is over by then), then
one ``Done``. Rules it enforces:

- **String content only.** The Typhoon template renders content-part arrays as ``""``.
- **Tool calls** accumulate keyed by ``index``, else ``id``, else position (Gemini sends whole
  calls without an index; llama.cpp sends an index and argument fragments). Arguments are parsed
  after the stream: ``json.loads``, then ``json_repair``. ``raw_arguments`` keeps the text.
- **Gemini ``extra_content``** (the ``thought_signature``) is kept on outgoing assistant tool
  calls and on ``ToolCall.extra``/``Done.assistant_message`` for Gemini, and stripped for every
  other provider.
- **Deadlines** (I2): the first token must arrive within the provider's first-token timeout,
  then every chunk within the stall timeout (3 s). Failures raise ``ProviderError`` (a
  ``ProviderFailed``) whose ``emitted`` says whether a ``TextDelta`` was already yielded.
- **Closing** the generator (``aclose()`` or cancelling the consuming task) closes the HTTP
  response in ``finally``, so llama-server cancels generation and frees the slot (≤ 0.2 s).

``Done.finish_reason`` is ``"tool_calls"`` whenever calls were made and the provider said
``"stop"`` (Gemini does); a truncated ``"length"`` is kept.

``Done.prompt_n``/``cache_n`` use llama.cpp semantics (tokens processed now / reused from the KV
cache) from the final chunk's ``timings``; for cloud providers they come from ``usage``
(``prompt_tokens`` minus cached, and ``prompt_tokens_details.cached_tokens``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import json_repair
import openai

from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    ProviderFailed,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from aivtube.infra.clock import DeadlineExceeded, deadline

if TYPE_CHECKING:
    from aivtube.llm.providers import OpenAICompatProvider

__all__ = [
    "FailReason",
    "Flavor",
    "ProviderError",
    "ToolCallAccumulator",
    "assistant_message",
    "build_request",
    "content_text",
    "error_detail",
    "history_arguments",
    "map_error",
    "parse_arguments",
    "sanitize_messages",
    "stream_turn",
    "tool_payload",
]

log = logging.getLogger(__name__)

Flavor: TypeAlias = Literal["llamacpp", "typhoon", "gemini", "openai"]
FailReason: TypeAlias = Literal[
    "auth",
    "connect",
    "timeout",
    "status",
    "rate_limited",
    "first_token_timeout",
    "stall",
    "stream_error",
    "protocol",
]

CLOSE_TIMEOUT_S = 1.0
"""Safety net for closing the HTTP response; the close itself is immediate."""


class ProviderError(ProviderFailed):
    """``ProviderFailed`` with a machine-readable ``reason`` and the HTTP ``status`` if any.

    ``reason == "connect"`` means the server could not be reached at all (likely down), which
    the router treats differently from a slow or erroring server.
    """

    reason: FailReason
    status: int | None
    provider: str

    def __init__(
        self,
        msg: str,
        *,
        emitted: bool,
        reason: FailReason,
        status: int | None = None,
        provider: str = "",
    ) -> None:
        super().__init__(msg, emitted=emitted)
        self.reason = reason
        self.status = status
        self.provider = provider

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild_provider_error,
            (str(self), self.emitted, self.reason, self.status, self.provider),
        )


def _rebuild_provider_error(
    msg: str, emitted: bool, reason: FailReason, status: int | None, provider: str
) -> ProviderError:
    return ProviderError(msg, emitted=emitted, reason=reason, status=status, provider=provider)


# --- request building -------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Deep-copy Mappings/Sequences into dicts/lists so the JSON body never sees proxies."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def content_text(content: Any) -> str:
    """Message content as one plain string (content-part arrays are joined, ``None`` is "")."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, bytes | bytearray):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def history_arguments(raw: str, parsed: Mapping[str, Any] | None = None) -> str:
    """Tool-call arguments safe to send back in history: ``raw`` when it is a JSON object
    (byte-stable for the prompt cache), else the repaired object, else ``"{}"``.

    llama-server parses these arguments while rendering the template and answers HTTP 500 for
    invalid JSON, e.g. a call cut off by ``max_tokens``.
    """
    try:
        if isinstance(json.loads(raw), dict):
            return raw
    except ValueError:
        pass
    value = parsed if parsed is not None else parse_arguments(raw)
    return json.dumps(_plain(value or {}), ensure_ascii=False)


def _sanitize_call(call: Mapping[str, Any], keep_extra_content: bool) -> dict[str, Any]:
    fn = call.get("function")
    fn = fn if isinstance(fn, Mapping) else {}
    args = fn.get("arguments")
    if isinstance(args, str):
        args = history_arguments(args)
    else:
        args = "{}" if args is None else json.dumps(_plain(args), ensure_ascii=False)
    out: dict[str, Any] = {
        "id": str(call.get("id") or ""),
        "type": "function",
        "function": {"name": str(fn.get("name") or ""), "arguments": args},
    }
    extra = call.get("extra_content")
    if keep_extra_content and isinstance(extra, Mapping) and extra:
        out["extra_content"] = _plain(extra)
    return out


def sanitize_messages(
    messages: Sequence[Mapping[str, Any]], *, keep_extra_content: bool
) -> list[dict[str, Any]]:
    """OpenAI messages with plain-string content and only the standard keys.

    ``extra_content`` on assistant tool calls survives only when ``keep_extra_content`` (Gemini).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        msg: dict[str, Any] = {"role": role, "content": content_text(m.get("content"))}
        name = m.get("name")
        if isinstance(name, str) and name:
            msg["name"] = name
        if role == "tool" and m.get("tool_call_id") is not None:
            msg["tool_call_id"] = str(m["tool_call_id"])
        calls = m.get("tool_calls")
        if role == "assistant" and isinstance(calls, Sequence) and calls:
            msg["tool_calls"] = [
                _sanitize_call(c, keep_extra_content) for c in calls if isinstance(c, Mapping)
            ]
        out.append(msg)
    return out


def tool_payload(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """``ToolSpec``s in the OpenAI ``tools`` format (order kept: it is part of the prompt)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _plain(t.parameters),
            },
        }
        for t in tools
    ]


def build_request(
    *,
    flavor: Flavor,
    model: str,
    req: ChatRequest,
    stream: bool,
    extra_body: Mapping[str, Any] | None = None,
    reasoning_effort: str | None = None,
    slot_map: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``client.chat.completions.create`` for one provider flavor.

    llama.cpp gets ``id_slot`` (``req.slot`` or the purpose's slot), ``cache_prompt`` and, with
    tools, ``parallel_tool_calls`` through ``extra_body``. Tools are always sent when present,
    even with ``tool_choice="none"``: the template renders them into the cached prefix.
    """
    kw: dict[str, Any] = {
        "model": model,
        "messages": sanitize_messages(req.messages, keep_extra_content=flavor == "gemini"),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": stream,
    }
    body: dict[str, Any] = _plain(dict(extra_body or {}))
    if req.tools:
        kw["tools"] = tool_payload(req.tools)
        if req.tool_choice != "auto":
            kw["tool_choice"] = req.tool_choice
    if req.response_schema is not None:
        kw["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _plain(req.response_schema)},
        }
    if flavor == "llamacpp":
        slot = req.slot if req.slot is not None else (slot_map or {}).get(req.purpose)
        if slot is not None:
            body["id_slot"] = int(slot)
        body["cache_prompt"] = True
        if req.tools:
            body["parallel_tool_calls"] = True
    elif reasoning_effort is not None:
        kw["reasoning_effort"] = reasoning_effort
    if body:
        kw["extra_body"] = body
    return kw


# --- tool-call accumulation ---------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _extra_content(tc: Any) -> Mapping[str, Any] | None:
    if isinstance(tc, Mapping):
        value = tc.get("extra_content")
    else:
        value = (getattr(tc, "model_extra", None) or {}).get("extra_content")
    return value if isinstance(value, Mapping) and value else None


@dataclass
class _CallRec:
    id: str | None = None
    name: str = ""
    args: list[str] = field(default_factory=list)
    extra: Mapping[str, Any] | None = None


class ToolCallAccumulator:
    """Merges streamed tool-call deltas into whole calls.

    Key order: ``index`` (llama.cpp, OpenAI), else ``id`` (Gemini: one complete call per delta,
    no index), else position: a fragment with neither index, id nor name continues the last
    call; one that carries a name starts a new call.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, Any], _CallRec] = {}
        self._order: list[_CallRec] = []

    def __len__(self) -> int:
        return len(self._order)

    def add(self, deltas: Sequence[Any]) -> None:
        for tc in deltas:
            index = _get(tc, "index")
            call_id = _get(tc, "id")
            fn = _get(tc, "function")
            name = _get(fn, "name")
            fragment = _get(fn, "arguments")
            rec: _CallRec | None = None
            if isinstance(index, int):
                rec = self._by_key.get(("index", index))
            if rec is None and isinstance(call_id, str) and call_id:
                rec = self._by_key.get(("id", call_id))
            if rec is None and index is None and not call_id and not name and self._order:
                rec = self._order[-1]
            if rec is None:
                rec = _CallRec()
                self._order.append(rec)
            if isinstance(index, int):
                self._by_key.setdefault(("index", index), rec)
            if isinstance(call_id, str) and call_id:
                self._by_key.setdefault(("id", call_id), rec)
                rec.id = rec.id or call_id
            if isinstance(name, str) and name and not rec.name:
                rec.name = name
            if isinstance(fragment, str) and fragment:
                rec.args.append(fragment)
            extra = _extra_content(tc)
            if extra is not None:
                rec.extra = extra

    def calls(self, *, keep_extra: bool) -> list[ToolCall]:
        """The accumulated calls in arrival order; calls without a name are dropped."""
        out: list[ToolCall] = []
        for i, rec in enumerate(self._order):
            raw = "".join(rec.args)
            if not rec.name:
                log.warning("dropping a tool call without a name (arguments %r)", raw[:80])
                continue
            out.append(
                ToolCall(
                    id=rec.id or f"call_{i}",
                    name=rec.name,
                    arguments=parse_arguments(raw),
                    raw_arguments=raw,
                    extra=dict(rec.extra) if keep_extra and rec.extra else None,
                )
            )
        return out


def parse_arguments(raw: str) -> Mapping[str, Any] | None:
    """Tool arguments as an object: ``json.loads``, then ``json_repair`` (e.g. a call cut off by
    ``max_tokens``). Empty text is ``{}``; anything that is not an object is ``None``."""
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = json_repair.loads(text)
        except Exception:  # json_repair is best effort; never let it break a turn
            return None
    return value if isinstance(value, dict) else None


def assistant_message(text: str, calls: Sequence[ToolCall]) -> dict[str, Any]:
    """The assistant turn to append to history: string content, tool-call arguments as valid
    JSON (``history_arguments``) and ``extra_content`` only when the call carries it (Gemini)."""
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        out: list[dict[str, Any]] = []
        for c in calls:
            args = history_arguments(c.raw_arguments, c.arguments)
            item: dict[str, Any] = {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": args},
            }
            if c.extra:
                item["extra_content"] = _plain(c.extra)
            out.append(item)
        msg["tool_calls"] = out
    return msg


# --- errors -----------------------------------------------------------------------------------


def error_detail(body: Any) -> str:
    """A short message from an error body: OpenAI ``{error:{message}}``, llama.cpp
    ``{code,message}``, Typhoon ``{detail}``, or Gemini's ``[{error:{…}}]``."""
    if isinstance(body, Mapping):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        err = body.get("error")
        if isinstance(err, Mapping):
            return error_detail(err)
        if isinstance(err, str):
            return err
        message = body.get("message")
        if isinstance(message, str):
            return message
        return ""
    if isinstance(body, list) and body:
        return error_detail(body[0])
    if isinstance(body, str):
        return body[:200]
    return ""


def map_error(
    exc: BaseException,
    *,
    provider: str,
    emitted: bool,
    phase: Literal["connect", "stream"],
) -> ProviderError:
    """Translate an SDK/transport exception into a ``ProviderError``."""
    status: int | None = None
    reason: FailReason
    if isinstance(exc, ProviderError):
        if exc.emitted == emitted:
            return exc
        reason, status, detail = exc.reason, exc.status, str(exc)
    elif isinstance(exc, openai.APITimeoutError):
        reason, detail = "timeout", "request timed out"
    elif isinstance(exc, openai.APIConnectionError):
        reason = "connect" if phase == "connect" else "stream_error"
        cause = exc.__cause__ or exc.__context__
        detail = f"{exc.message} ({type(cause).__name__})" if cause else exc.message
    elif isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        detail = error_detail(exc.body) or exc.message
        if status == 429:
            reason = "rate_limited"
        elif status in (401, 403):
            reason = "auth"
        else:
            reason = "status"
    elif isinstance(exc, openai.APIError):
        reason, detail = "stream_error", error_detail(exc.body) or exc.message
    else:
        reason, detail = "protocol", f"{type(exc).__name__}: {exc}"
    where = f" (HTTP {status})" if status is not None else ""
    return ProviderError(
        f"{provider}: {reason}{where}: {detail}",
        emitted=emitted,
        reason=reason,
        status=status,
        provider=provider,
    )


# --- streaming --------------------------------------------------------------------------------


@dataclass
class _Usage:
    prompt_n: int | None = None
    cache_n: int | None = None
    completion_tokens: int | None = None
    from_timings: bool = False


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _read_usage(chunk: Any, usage: _Usage) -> None:
    timings = (getattr(chunk, "model_extra", None) or {}).get("timings")
    if isinstance(timings, Mapping):  # llama.cpp: the final chunk
        usage.prompt_n = _as_int(timings.get("prompt_n"))
        usage.cache_n = _as_int(timings.get("cache_n"))
        usage.completion_tokens = _as_int(timings.get("predicted_n"))
        usage.from_timings = True
        return
    u = getattr(chunk, "usage", None)
    if u is None or usage.from_timings:
        return
    total = _as_int(_get(u, "prompt_tokens"))
    cached = _as_int(_get(_get(u, "prompt_tokens_details"), "cached_tokens")) or 0
    if total is not None:
        usage.prompt_n = max(0, total - cached)
        usage.cache_n = cached
    usage.completion_tokens = _as_int(_get(u, "completion_tokens"))


_END: Any = object()


async def _next(it: AsyncIterator[Any]) -> Any:
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return _END


async def close_stream(stream: Any) -> None:
    """Close an SDK ``AsyncStream`` (the HTTP response) without ever raising."""
    try:
        async with deadline(CLOSE_TIMEOUT_S, what="close LLM stream"):
            await stream.close()
    except Exception as exc:  # closing is best effort; the socket dies with the client
        log.debug("closing the LLM stream failed: %r", exc)


async def stream_turn(
    provider: OpenAICompatProvider, req: ChatRequest
) -> AsyncGenerator[LLMEvent, None]:
    """Stream one completion from ``provider`` (see the module docstring for the rules)."""
    cfg = provider.cfg
    clock = provider.clock
    name = provider.name
    if cfg.cloud and not provider.api_key:
        raise ProviderError(
            f"{name}: no API key (set {cfg.api_key_env} in .env)",
            emitted=False,
            reason="auth",
            provider=name,
        )
    kw = build_request(
        flavor=cfg.flavor,
        model=cfg.model,
        req=req,
        stream=True,
        extra_body=cfg.extra_body,
        reasoning_effort=cfg.reasoning_effort,
        slot_map=cfg.slot_map,
    )
    keep_extra = cfg.flavor == "gemini"
    first_timeout = req.first_token_timeout_s or cfg.first_token_timeout_s
    t0 = clock.now()
    first_deadline = t0 + first_timeout
    t_first: float | None = None
    emitted = False
    finish: str | None = None
    usage = _Usage()
    acc = ToolCallAccumulator()
    text_parts: list[str] = []
    stream: Any = None
    try:
        try:
            async with deadline(first_timeout, what=f"{name} first token", clock=clock):
                stream = await provider.client.chat.completions.create(**kw)
        except DeadlineExceeded as exc:
            raise ProviderError(
                f"{name}: no first token within {first_timeout:g} s",
                emitted=False,
                reason="first_token_timeout",
                provider=name,
            ) from exc
        except Exception as exc:
            raise map_error(exc, provider=name, emitted=False, phase="connect") from exc
        while True:
            if t_first is None:
                budget, what = max(0.0, first_deadline - clock.now()), "first token"
            else:
                budget, what = cfg.stall_timeout_s, "next token"
            try:
                async with deadline(budget, what=f"{name} {what}", clock=clock):
                    chunk = await _next(stream)
            except DeadlineExceeded as exc:
                if t_first is None:
                    raise ProviderError(
                        f"{name}: no first token within {first_timeout:g} s",
                        emitted=emitted,
                        reason="first_token_timeout",
                        provider=name,
                    ) from exc
                raise ProviderError(
                    f"{name}: stream stalled for {cfg.stall_timeout_s:g} s",
                    emitted=emitted,
                    reason="stall",
                    provider=name,
                ) from exc
            except Exception as exc:
                raise map_error(exc, provider=name, emitted=emitted, phase="stream") from exc
            if chunk is _END:
                break
            _read_usage(chunk, usage)
            for choice in getattr(chunk, "choices", None) or ():
                if getattr(choice, "index", 0) not in (0, None):
                    continue
                # the SDK does not validate chunks: any field may be missing
                reason = getattr(choice, "finish_reason", None)
                if isinstance(reason, str) and reason:
                    finish = reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                calls = getattr(delta, "tool_calls", None)
                reasoning = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                if t_first is None and (text or calls or reasoning):
                    t_first = clock.now()
                if calls:
                    acc.add(calls)
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    emitted = True
                    yield TextDelta(text)
    finally:
        if stream is not None:
            await close_stream(stream)
    if t_first is None and finish is None:
        raise ProviderError(
            f"{name}: the stream ended without any output",
            emitted=emitted,
            reason="protocol",
            provider=name,
        )
    if finish is None:
        log.warning("%s: stream ended without finish_reason", name)
    calls_out = acc.calls(keep_extra=keep_extra)
    if calls_out and finish in (None, "stop"):
        finish = "tool_calls"  # Gemini says "stop" even when it calls tools; llama.cpp does not
    for call in calls_out:
        yield call
    text = "".join(text_parts)
    yield Done(
        provider=name,
        finish_reason=finish,
        ttft_ms=((t_first if t_first is not None else clock.now()) - t0) * 1000.0,
        prompt_n=usage.prompt_n,
        cache_n=usage.cache_n,
        completion_tokens=usage.completion_tokens,
        assistant_message=assistant_message(text, calls_out),
    )
