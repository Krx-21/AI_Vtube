"""Nightly: the real CPU llama-server with Typhoon 2.5 4B Q4_K_M (§10 layer 6).

Needs ``$AIVTUBE_TEST_ASSETS`` with ``llama-bin/llama-server`` and
``typhoon2.5-qwen3-4b-q4_k_m.gguf``. The server is started and stopped by
``LlamaServerManager`` (spawn mode) with the command from ``build_llama_argv``: port 18080,
``-t 2 -c 4096 -np 2 --jinja --slot-save-path <tmp>``. Replies are capped at 24 tokens.

The official typhoon-ai GGUF has no tool-capable template (llama.cpp falls back to plain ChatML
and silently drops tools), which the startup assertion must reject; the server under test
therefore gets ``--chat-template-file typhoon25_chat_template.jinja``: the unmodified
``chat_template.jinja`` of typhoon-ai/typhoon2.5-qwen3-4b (Apache-2.0).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from aivtube.config.schema import LlamaServerConfig
from aivtube.contracts.llm import ChatRequest, Done, LLMEvent, TextDelta, ToolCall, ToolSpec
from aivtube.llm.llama_args import build_llama_argv, server_root_url
from aivtube.llm.llamacpp import (
    LlamaCppAdmin,
    LlamaServerManager,
    LlamaServerSpec,
    TemplateCapsError,
)
from aivtube.llm.providers import OpenAICompatProvider, ProviderCfg

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.timeout(900),
]

LOAD_TIMEOUT_S = 600.0
"""A cold 2.4 GB load from a slow disk took over 180 s in the dev container."""

TEMPLATE = Path(__file__).with_name("typhoon25_chat_template.jinja")
GGUF = "typhoon2.5-qwen3-4b-q4_k_m.gguf"
PORT = 18080
SYSTEM = "คุณคือไพลิน สาวไทยร่าเริง ตอบสั้นๆ เป็นภาษาไทย"
REMEMBER = ToolSpec(
    name="remember",
    description="Save a fact about the viewer or the stream to long-term memory.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "the fact, one sentence"}},
        "required": ["text"],
    },
)


def _spec(assets: Path, port: int, slot_dir: Path, extra: list[str]) -> LlamaServerSpec:
    exe = assets / "llama-bin" / "llama-server"
    model = assets / GGUF
    if not exe.exists() or not model.exists():
        pytest.skip(f"{exe} or {model} is missing")
    cfg = LlamaServerConfig(
        exe=str(exe),
        model=str(model),
        alias="pailin-4b",
        port=port,
        ctx=4096,
        parallel=2,
        threads=2,
        slot_save_dir=str(slot_dir),
        extra_args=extra,
    )
    return LlamaServerSpec(
        name="local4b",
        base_url=server_root_url(cfg),
        alias="pailin-4b",
        model=str(model),
        argv=tuple(build_llama_argv(cfg, root=assets)),
        cwd=assets,
        log_path=slot_dir / "llama-local4b.log",
    )


@dataclass
class Llama:
    manager: LlamaServerManager
    provider: OpenAICompatProvider
    admin: LlamaCppAdmin


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def llama(
    test_assets: Path, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[Llama]:
    slot_dir = tmp_path_factory.mktemp("kv")
    spec = _spec(test_assets, PORT, slot_dir, ["--chat-template-file", str(TEMPLATE)])
    manager = LlamaServerManager([spec], mode="spawn", graceful_timeout_s=10.0)
    try:
        assert await manager.ensure_running("local4b", LOAD_TIMEOUT_S), "llama-server did not start"
        assert not manager.adopted("local4b"), f"port {PORT} is taken by another server"
        provider = OpenAICompatProvider(
            ProviderCfg(
                name="local-4b",
                kind="llamacpp",
                flavor="llamacpp",
                base_url=f"{spec.base_url}/v1",
                model="pailin-4b",
                first_token_timeout_s=90.0,
                stall_timeout_s=30.0,
                prefill_timeout_s=120.0,
                slot_map={"speak": 0, "background": 1},
                server="local4b",
            )
        )
        admin = manager.admin("local4b")
        assert admin is not None
        yield Llama(manager, provider, admin)
        await provider.aclose()
    finally:
        await manager.aclose()


def _req(prompt: str, **kw: Any) -> ChatRequest:
    kw.setdefault("max_tokens", 24)
    kw.setdefault("temperature", 0.0)
    return ChatRequest(
        messages=({"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}),
        character="pailin",
        **kw,
    )


async def _collect(stream: AsyncIterator[LLMEvent]) -> list[LLMEvent]:
    return [ev async for ev in stream]


def _done(events: list[LLMEvent]) -> Done:
    done = events[-1]
    assert isinstance(done, Done), events
    return done


def _is_thai(text: str) -> bool:
    return any("฀" <= ch <= "๿" for ch in text)


async def test_real_props_support_tool_calls(llama: Llama) -> None:
    props = await llama.manager.props("local4b")
    assert props["model_alias"] == "pailin-4b"
    assert props["chat_template_caps"]["supports_tool_calls"] is True
    await llama.admin.assert_tool_caps()


async def test_real_short_thai_streamed_reply(llama: Llama) -> None:
    events = await _collect(llama.provider.stream(_req("สวัสดีไพลิน", max_tokens=16)))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert _is_thai(text), text
    done = _done(events)
    assert done.finish_reason in ("stop", "length") and done.ttft_ms > 0
    assert done.prompt_n is not None and done.completion_tokens is not None
    assert done.assistant_message == {"role": "assistant", "content": text}


async def test_real_tool_call_round_trip(llama: Llama) -> None:
    ask = _req("ไพลิน จำไว้นะว่าแมวของฉันชื่อส้ม", tools=(REMEMBER,), tool_choice="required")
    events = await _collect(llama.provider.stream(ask))
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert calls, events
    call = calls[0]
    assert call.name == "remember" and call.id
    assert call.arguments is not None and isinstance(call.arguments.get("text"), str)
    done = _done(events)
    assert done.assistant_message["tool_calls"][0]["id"] == call.id
    follow = ChatRequest(
        messages=(
            *ask.messages,
            done.assistant_message,
            {"role": "tool", "tool_call_id": call.id, "content": '{"ok": true, "slot": 1}'},
        ),
        tools=(REMEMBER,),
        tool_choice="none",
        max_tokens=24,
        temperature=0.0,
    )
    reply = await _collect(llama.provider.stream(follow))
    text = "".join(e.text for e in reply if isinstance(e, TextDelta))
    assert text.strip(), reply
    assert not any(isinstance(e, ToolCall) for e in reply)


async def test_real_cache_n_on_a_repeated_prefix(llama: Llama) -> None:
    req = _req("วันนี้อากาศเป็นยังไงบ้าง", max_tokens=4, slot=1)
    took = await llama.provider.prefill(req)
    assert took > 0.0
    done = _done(await _collect(llama.provider.stream(req)))
    assert done.cache_n is not None and done.cache_n > 0, done
    assert done.prompt_n is not None and done.cache_n > done.prompt_n


@pytest.mark.timing
async def test_real_cancel_frees_the_slot(llama: Llama) -> None:
    """Closing the stream frees the slot within 0.2 s plus a few decode steps: llama-server
    notices the disconnect between tokens (~0.2 s per token on 2 contended CPU threads here,
    a few ms on the RTX 4070)."""
    stream = llama.provider.stream(_req("เล่านิทานเรื่องแมวกับหมาให้ฟังหน่อย", slot=0))
    first = await stream.__anext__()
    t_first = time.perf_counter()
    second = await stream.__anext__()
    token_s = time.perf_counter() - t_first
    assert isinstance(first, TextDelta) and isinstance(second, TextDelta)
    t0 = time.perf_counter()
    await stream.aclose()
    close_s = time.perf_counter() - t0
    assert close_s < 0.05, f"closing the stream took {close_s:.3f} s"
    busy = True
    while time.perf_counter() - t0 < 5.0:
        slots = await llama.admin.slots()
        busy = any(s.get("id") == 0 and s.get("is_processing") for s in slots)
        if not busy:
            break
        await asyncio.sleep(0.005)
    freed_s = time.perf_counter() - t0
    assert not busy, "slot 0 still processing 5 s after the stream was closed"
    budget = 0.2 + 3 * token_s
    assert freed_s <= budget, f"slot 0 freed after {freed_s:.3f} s (token {token_s:.3f} s)"
    # the same slot answers the next request at once
    t1 = time.perf_counter()
    done = _done(await _collect(llama.provider.stream(_req("สวัสดี", max_tokens=2, slot=0))))
    assert done.completion_tokens is not None and time.perf_counter() - t1 < 10.0


async def test_real_slot_save_and_restore(llama: Llama) -> None:
    await _collect(llama.provider.stream(_req("สวัสดี", max_tokens=2, slot=0)))
    assert await llama.manager.save_slot("local4b", 0, "pailin-nightly.bin") is True
    assert await llama.manager.restore_slot("local4b", 0, "pailin-nightly.bin") is True
    assert await llama.manager.restore_slot("local4b", 0, "pailin-missing.bin") is False


async def test_real_gguf_without_a_tool_template_is_a_hard_error(
    test_assets: Path, tmp_path: Path
) -> None:
    spec = _spec(test_assets, PORT + 1, tmp_path, [])
    manager = LlamaServerManager([spec], mode="spawn", graceful_timeout_s=10.0)
    try:
        props_caps: dict[str, Any] = {}
        try:
            assert await manager.ensure_running("local4b", LOAD_TIMEOUT_S)
        except TemplateCapsError:
            props_caps = dict((await manager.props("local4b"))["chat_template_caps"])
        else:
            pytest.skip("this GGUF embeds a tool-capable template")
        assert props_caps.get("supports_tool_calls") is False
    finally:
        await manager.aclose()
