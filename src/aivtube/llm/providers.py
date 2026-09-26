"""LLM providers (§4.11): ``OpenAICompatProvider`` and ``CannedProvider``.

``OpenAICompatProvider`` talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint with
the openai SDK (``AsyncOpenAI``, ``max_retries=0``, httpx2). Flavors:

- ``llamacpp``: local llama-server; ``id_slot``/``cache_prompt``/``parallel_tool_calls`` via
  ``extra_body``; ``probe()`` is ``GET /health``; ``prefill()`` is a ``max_tokens=1`` call.
- ``typhoon``: the Typhoon API. The SDK's ``models.list()`` is never called (the endpoint
  returns a bare array and the SDK crashes); error bodies are ``{detail}``.
- ``gemini``: Gemini's OpenAI endpoint with ``reasoning_effort`` (``"minimal"``) and
  ``extra_content``/``thought_signature`` kept on tool calls.
- ``openai``: any other compatible endpoint.

Local providers use their own httpx2 client with ``trust_env=False`` (no proxy for 127.0.0.1,
§2.2); cloud providers honour the environment (proxy, ``SSL_CERT_FILE``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx2
from openai import AsyncOpenAI

from aivtube.contracts.infra import Clock
from aivtube.contracts.llm import ChatRequest, Done, LLMEvent, LLMProvider, ProviderCaps, TextDelta
from aivtube.contracts.types import Health, HealthState
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.llm.llama_args import root_url
from aivtube.llm.openai_stream import (
    Flavor,
    ProviderError,
    build_request,
    map_error,
    stream_turn,
)

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig

__all__ = [
    "CannedProvider",
    "EnvSecrets",
    "OpenAICompatProvider",
    "OpenAIStreamProvider",
    "ProviderCfg",
    "SecretSource",
    "build_providers",
    "caps_for",
    "provider_cfgs",
    "root_url",
]

log = logging.getLogger(__name__)

_FLAVORS: frozenset[str] = frozenset({"llamacpp", "typhoon", "gemini", "openai"})


class SecretSource(Protocol):
    """Where API keys come from (``config.Secrets`` implements it)."""

    def get(self, name: str) -> str | None: ...


class EnvSecrets:
    """``SecretSource`` over ``os.environ`` (tests and tools without a ``.env``)."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name) or None


@dataclass(frozen=True, slots=True)
class ProviderCfg:
    """One ``[llm.providers.<name>]`` entry plus the router-wide timeouts it needs."""

    name: str
    kind: str = "openai_compat"
    flavor: Flavor = "openai"
    base_url: str = ""
    model: str = ""
    api_key_env: str | None = None
    first_token_timeout_s: float = 4.0
    cloud: bool = False
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None
    slot_map: Mapping[str, int] = field(default_factory=dict)
    server: str | None = None
    connect_timeout_s: float = 2.0
    stall_timeout_s: float = 3.0
    request_timeout_s: float = 60.0
    prefill_timeout_s: float = 30.0
    probe_timeout_s: float = 2.0

    @property
    def root_url(self) -> str:
        return root_url(self.base_url)


def caps_for(cfg: ProviderCfg) -> ProviderCaps:
    """What each flavor supports (Typhoon 2.5 is non-thinking; Gemini 3 always thinks)."""
    if cfg.flavor == "llamacpp":
        return ProviderCaps(
            tools=True,
            parallel_tools=True,
            json_schema=True,
            prompt_cache=True,
            cloud=False,
            keeps_thought_signatures=False,
            reasoning="none",
        )
    if cfg.flavor == "typhoon":
        return ProviderCaps(
            tools=True,
            parallel_tools=False,
            json_schema=False,
            prompt_cache=False,
            cloud=cfg.cloud,
            keeps_thought_signatures=False,
            reasoning="none",
        )
    if cfg.flavor == "gemini":
        return ProviderCaps(
            tools=True,
            parallel_tools=True,
            json_schema=True,
            prompt_cache=False,
            cloud=cfg.cloud,
            keeps_thought_signatures=True,
            reasoning="effort",
        )
    return ProviderCaps(
        tools=True,
        parallel_tools=True,
        json_schema=True,
        prompt_cache=False,
        cloud=cfg.cloud,
        keeps_thought_signatures=False,
        reasoning="effort" if cfg.reasoning_effort else "none",
    )


class OpenAICompatProvider:
    """``LLMProvider`` over one OpenAI-compatible endpoint (see the module docstring).

    ``http_client`` (an ``httpx2.AsyncClient``, e.g. with a ``MockTransport``) is used for the
    SDK and for the raw probes; when omitted the provider owns one and ``aclose()`` closes it.
    """

    name: str
    caps: ProviderCaps

    def __init__(
        self,
        cfg: ProviderCfg,
        *,
        secrets: SecretSource | None = None,
        http_client: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.caps = caps_for(cfg)
        self.clock: Clock = clock if clock is not None else SystemClock()
        source: SecretSource = secrets if secrets is not None else EnvSecrets()
        self.api_key: str | None = source.get(cfg.api_key_env) if cfg.api_key_env else None
        timeout = httpx2.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s)
        self._owns_http = http_client is None
        self.http: httpx2.AsyncClient = (
            http_client
            if http_client is not None
            else httpx2.AsyncClient(timeout=timeout, trust_env=cfg.cloud)
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key or ("sk-local" if not cfg.cloud else "missing-key"),
            base_url=cfg.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=self.http,
        )

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(name={self.name!r}, flavor={self.cfg.flavor!r})"

    def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
        """Stream one completion; ``aclose()`` closes the HTTP response (see ``stream_turn``)."""
        return stream_turn(self, req)

    async def probe(self) -> Health:
        """llama.cpp: ``GET /health`` (200 OK, 503 loading). Cloud: a raw ``GET …/models`` that
        checks reachability and the key (never the SDK's ``models.list``)."""
        component = f"llm:{self.name}"
        headers: dict[str, str] = {}
        if self.cfg.flavor == "llamacpp":
            url = f"{self.cfg.root_url}/health"
        else:
            if self.cfg.cloud and not self.api_key:
                return self._health(HealthState.DOWN, f"no API key ({self.cfg.api_key_env})")
            url = f"{self.cfg.base_url.rstrip('/')}/models"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with deadline(
                self.cfg.probe_timeout_s, what=f"{component} probe", clock=self.clock
            ):
                resp = await self.http.get(url, headers=headers)
                await resp.aclose()
        except DeadlineExceeded:
            return self._health(HealthState.DOWN, "probe timed out")
        except Exception as exc:
            return self._health(HealthState.DOWN, f"unreachable: {type(exc).__name__}")
        code = resp.status_code
        if code == 200:
            return self._health(HealthState.OK)
        if self.cfg.flavor == "llamacpp":
            if code == 503:
                return self._health(HealthState.STARTING, "loading model")
            return self._health(HealthState.DOWN, f"/health returned {code}")
        if code in (401, 403):
            return self._health(HealthState.DOWN, f"API key rejected ({code})")
        return self._health(HealthState.DEGRADED, f"HTTP {code}")

    def _health(self, state: HealthState, detail: str = "") -> Health:
        return Health(f"llm:{self.name}", state, detail, self.clock.now())

    async def prefill(self, req: ChatRequest) -> float:
        """Warm the slot's KV cache with a ``max_tokens=1`` completion of ``req`` (llama.cpp
        only; cloud providers return 0.0 without a request). Returns the seconds it took."""
        if self.cfg.flavor != "llamacpp":
            return 0.0
        kw = build_request(
            flavor=self.cfg.flavor,
            model=self.cfg.model,
            req=replace(req, max_tokens=1),
            stream=False,
            extra_body=self.cfg.extra_body,
            slot_map=self.cfg.slot_map,
        )
        t0 = self.clock.now()
        try:
            async with deadline(
                self.cfg.prefill_timeout_s, what=f"{self.name} prefill", clock=self.clock
            ):
                await self.client.chat.completions.create(**kw)
        except DeadlineExceeded as exc:
            raise ProviderError(
                f"{self.name}: prefill took longer than {self.cfg.prefill_timeout_s:g} s",
                emitted=False,
                reason="timeout",
                provider=self.name,
            ) from exc
        except Exception as exc:
            raise map_error(exc, provider=self.name, emitted=False, phase="connect") from exc
        return max(0.0, self.clock.now() - t0)

    async def aclose(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._owns_http:
            await self.client.close()


OpenAIStreamProvider = OpenAICompatProvider
"""Alias used by the work-package brief."""


class CannedProvider:
    """The chain's last resort (§2.8): says the fixed "brain freeze" line.

    Its ``Done.provider`` is its name (``"canned"``), so the brain can tell it apart.
    """

    name: str
    caps: ProviderCaps

    def __init__(self, line: str, *, name: str = "canned") -> None:
        if not line.strip():
            raise ValueError("the canned line must not be empty")
        self.line = line
        self.name = name
        self.caps = ProviderCaps(
            tools=False,
            parallel_tools=False,
            json_schema=False,
            prompt_cache=False,
            cloud=False,
            keeps_thought_signatures=False,
            reasoning="none",
        )

    async def probe(self) -> Health:
        return Health(f"llm:{self.name}", HealthState.OK)

    async def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
        yield TextDelta(self.line)
        yield Done(
            provider=self.name,
            finish_reason="stop",
            ttft_ms=0.0,
            prompt_n=None,
            cache_n=None,
            completion_tokens=None,
            assistant_message={"role": "assistant", "content": self.line},
        )

    async def prefill(self, req: ChatRequest) -> float:
        return 0.0


# --- from config ----------------------------------------------------------------------------


def _flavor(value: str) -> Flavor:
    if value not in _FLAVORS:
        raise ValueError(f"unknown LLM flavor {value!r}")
    return cast(Flavor, value)


def provider_cfgs(cfg: AppConfig, *, include_disabled: bool = False) -> list[ProviderCfg]:
    """``ProviderCfg`` for every ``llamacpp``/``openai_compat`` provider in ``cfg.llm``.

    Disabled providers (including cloud ones switched off for lack of consent) are left out
    unless ``include_disabled``.
    """
    llm = cfg.llm
    slot_map = {
        "speak": llm.slots.speak[0],
        "background": llm.slots.background,
        "game": llm.slots.game,
    }
    out: list[ProviderCfg] = []
    for name, p in llm.providers.items():
        if p.kind not in ("llamacpp", "openai_compat"):
            continue
        if not p.enabled and not include_disabled:
            continue
        flavor = _flavor(p.effective_flavor)
        out.append(
            ProviderCfg(
                name=name,
                kind=p.kind,
                flavor=flavor,
                base_url=p.base_url,
                model=p.model,
                api_key_env=p.api_key_env,
                first_token_timeout_s=p.first_token_timeout_s,
                cloud=p.cloud,
                extra_body=dict(p.extra_body),
                reasoning_effort=p.reasoning_effort,
                slot_map=dict(slot_map) if flavor == "llamacpp" else {},
                server=p.server,
                connect_timeout_s=llm.connect_timeout_s,
                stall_timeout_s=llm.stall_timeout_s,
            )
        )
    return out


def build_providers(
    cfg: AppConfig,
    *,
    secrets: SecretSource | None,
    clock: Clock | None = None,
    http_client: httpx2.AsyncClient | None = None,
) -> list[LLMProvider]:
    """Every enabled provider in ``cfg.llm.providers`` (``fake`` kinds are left to the app).

    Cloud providers are constructed only with ``privacy.cloud_llm_consent`` (config disables
    them otherwise). ``kind = "canned"`` entries say ``llm.canned_line``.
    """
    out: list[LLMProvider] = [
        OpenAICompatProvider(pc, secrets=secrets, clock=clock, http_client=http_client)
        for pc in provider_cfgs(cfg)
    ]
    for name, p in cfg.llm.providers.items():
        if p.kind == "canned" and p.enabled:
            out.append(CannedProvider(cfg.llm.canned_line, name=name))
        elif p.kind not in ("llamacpp", "openai_compat", "canned"):
            log.debug("provider %s has kind %r; the app builds it", name, p.kind)
    return out
