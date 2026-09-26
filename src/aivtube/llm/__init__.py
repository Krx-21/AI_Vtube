"""LLM providers, the fallback router and llama-server management (ARCHITECTURE.md §4.8–§4.12).

- ``openai_stream``: one streamed OpenAI-compatible chat completion as ``LLMEvent``s (tool-call
  accumulation, message sanitising, error mapping, deadlines).
- ``providers``: ``OpenAICompatProvider`` (llama.cpp, Typhoon API, Gemini) and ``CannedProvider``.
- ``router``: ``FallbackRouter`` (chain, breakers, consent gating, promote/rollback, auto-rollback).
- ``llamacpp``: ``LlamaCppAdmin`` (health, ``/props``, slots) and the ``LocalServerManager``
  implementations (standalone spawn/adopt, or through the launcher).
- ``llama_args``: the llama-server command builder. Standard library only, so the launcher can
  import it.

Names load lazily: importing ``aivtube.llm.llama_args`` never imports the openai SDK.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aivtube.llm.llama_args import build_llama_argv, server_root_url, slot_save_path
    from aivtube.llm.llamacpp import (
        LauncherServerManager,
        LlamaAdminError,
        LlamaCppAdmin,
        LlamaServerManager,
        LlamaServerSpec,
        TemplateCapsError,
        check_tool_caps,
        props_match,
        slot_filename,
    )
    from aivtube.llm.openai_stream import (
        ProviderError,
        ToolCallAccumulator,
        build_request,
        parse_arguments,
        sanitize_messages,
        stream_turn,
    )
    from aivtube.llm.providers import (
        CannedProvider,
        OpenAICompatProvider,
        OpenAIStreamProvider,
        ProviderCfg,
        build_providers,
        caps_for,
        provider_cfgs,
    )
    from aivtube.llm.router import ConsentRequired, FallbackRouter, build_router

_LAZY: dict[str, str] = {
    "build_llama_argv": "llama_args",
    "server_root_url": "llama_args",
    "slot_save_path": "llama_args",
    "LauncherServerManager": "llamacpp",
    "LlamaAdminError": "llamacpp",
    "LlamaCppAdmin": "llamacpp",
    "LlamaServerManager": "llamacpp",
    "LlamaServerSpec": "llamacpp",
    "TemplateCapsError": "llamacpp",
    "check_tool_caps": "llamacpp",
    "props_match": "llamacpp",
    "slot_filename": "llamacpp",
    "ProviderError": "openai_stream",
    "ToolCallAccumulator": "openai_stream",
    "build_request": "openai_stream",
    "parse_arguments": "openai_stream",
    "sanitize_messages": "openai_stream",
    "stream_turn": "openai_stream",
    "CannedProvider": "providers",
    "OpenAICompatProvider": "providers",
    "OpenAIStreamProvider": "providers",
    "ProviderCfg": "providers",
    "build_providers": "providers",
    "caps_for": "providers",
    "provider_cfgs": "providers",
    "ConsentRequired": "router",
    "FallbackRouter": "router",
    "build_router": "router",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CannedProvider",
    "ConsentRequired",
    "FallbackRouter",
    "LauncherServerManager",
    "LlamaAdminError",
    "LlamaCppAdmin",
    "LlamaServerManager",
    "LlamaServerSpec",
    "OpenAICompatProvider",
    "OpenAIStreamProvider",
    "ProviderCfg",
    "ProviderError",
    "TemplateCapsError",
    "ToolCallAccumulator",
    "build_llama_argv",
    "build_providers",
    "build_request",
    "build_router",
    "caps_for",
    "check_tool_caps",
    "parse_arguments",
    "props_match",
    "provider_cfgs",
    "sanitize_messages",
    "server_root_url",
    "slot_filename",
    "slot_save_path",
    "stream_turn",
]
