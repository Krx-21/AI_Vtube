"""Build the ``TTSRouter`` from ``[tts]`` config (``TtsConfig.model_dump()`` / ``voice.configure``).

Backend kinds: ``edge`` (optional ``ca_bundle``), ``azure`` (key and region from the secrets
named by ``key_env``/``region_env``; F0 adds ``min_chars_when_primary_f0`` to the
constraints), ``fake`` (CI) and ``captions`` (implicit: the router's fallback). ``piper`` is
not built in M1 (logged); identities that only use it fall through to captions. A backend
that cannot be built is logged and left out.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from aivtube.contracts.infra import Clock
from aivtube.contracts.speech import TTSConstraints
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import PhraseCache, TTSBackend
from aivtube.voice.tts.cache import DiskPhraseCache
from aivtube.voice.tts.router import ChunkLimits, IdentityCfg, TTSRouter

__all__ = ["build_tts_backend", "build_tts_router", "identities_from_config"]

log = logging.getLogger("aivtube.voice.tts")

SecretLookup = Callable[[str], str | None]


def _path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def build_tts_backend(
    name: str,
    spec: Mapping[str, Any],
    *,
    root: Path,
    secrets: SecretLookup,
    clock: Clock | None = None,
) -> TTSBackend | None:
    """One backend from its ``[tts.backends.<name>]`` spec, or ``None``."""
    if not spec.get("enabled", True):
        return None
    kind = str(spec.get("kind", ""))
    if kind == "edge":
        from aivtube.voice.tts.edge import EdgeTTSBackend

        ca = spec.get("ca_bundle") or None
        return EdgeTTSBackend(
            name=name, ca_bundle=_path(root, str(ca)) if ca else None, clock=clock
        )
    if kind == "azure":
        from aivtube.voice.tts.azure import AzureTTSBackend

        key = secrets(str(spec.get("key_env") or "")) if spec.get("key_env") else None
        region = secrets(str(spec.get("region_env") or "")) if spec.get("region_env") else None
        if not key or not region:
            log.info("TTS backend %s skipped: Azure key/region not set", name)
            return None
        tier: Literal["F0", "S0"] = "S0" if str(spec.get("tier", "F0")).upper() == "S0" else "F0"
        return AzureTTSBackend(
            key,
            region,
            tier=tier,
            requests_per_min=int(spec.get("requests_per_min", 18)),
            name=name,
            clock=clock,
        )
    if kind == "fake":
        from aivtube.testing.fakes.tts import FakeTTS

        return FakeTTS(name=name, clock=clock, ttfa_s=float(spec.get("ttfa_s", 0.05)))
    if kind == "captions":
        return None
    log.warning("TTS backend %s skipped: kind %r is not available in this build", name, kind)
    return None


def identities_from_config(identities: Mapping[str, Mapping[str, Any]]) -> dict[str, IdentityCfg]:
    out: dict[str, IdentityCfg] = {}
    for name, spec in identities.items():
        voice = VoiceSpec(
            identity=name,
            voice=str(spec["voice"]),
            rate=str(spec.get("rate", "+0%")),
            pitch=str(spec.get("pitch", "+0Hz")),
            volume=str(spec.get("volume", "+0%")),
        )
        out[name] = IdentityCfg(name, voice, tuple(str(b) for b in spec.get("backends", ())))
    return out


def build_tts_router(
    tts: Mapping[str, Any],
    chains: Mapping[str, Sequence[str]],
    *,
    root: Path,
    secrets: SecretLookup,
    clock: Clock | None = None,
    cache: PhraseCache | None = None,
    backends: Mapping[str, TTSBackend] | None = None,
    on_fallback: Callable[[str, str | None, str], None] | None = None,
    on_constraints: Callable[[str, TTSConstraints], None] | None = None,
) -> TTSRouter:
    """``chains`` maps each character to its identity chain (``cfg.tts_chain_for(char)``).

    ``backends`` replaces the built backends (tests); otherwise each ``tts.backends`` entry
    that some identity uses is built.
    """
    identities = identities_from_config(tts.get("identities", {}))
    used = {b for ident in identities.values() for b in ident.backends}
    specs: Mapping[str, Mapping[str, Any]] = tts.get("backends", {})
    min_chars: dict[str, int] = {}
    if backends is None:
        built: dict[str, TTSBackend] = {}
        for name in sorted(used):
            spec = specs.get(name)
            if spec is None:
                log.warning("TTS backend %s has no [tts.backends.%s] section", name, name)
                continue
            try:
                backend = build_tts_backend(name, spec, root=root, secrets=secrets, clock=clock)
            except Exception as exc:
                log.warning("TTS backend %s could not be built: %s", name, exc)
                continue
            if backend is not None:
                built[name] = backend
        backends = built
    for name, spec in specs.items():
        if spec.get("kind") == "azure" and str(spec.get("tier", "F0")).upper() == "F0":
            min_chars[name] = int(spec.get("min_chars_when_primary_f0", 60))
    chunk_spec: Mapping[str, Any] = tts.get("chunk") or tts.get("chunker") or {}
    chunk = ChunkLimits(
        first_min_chars=int(chunk_spec.get("first_min_chars", 8)),
        min_chars=int(chunk_spec.get("min_chars", 40)),
        max_chars=int(chunk_spec.get("max_chars", 160)),
    )
    if cache is None and tts.get("cache_dir"):
        cache = DiskPhraseCache(_path(root, str(tts["cache_dir"])))
    default_chain = list(tts.get("identity_chain", [])) or None
    return TTSRouter(
        identities,
        backends,
        {c: list(chain) for c, chain in chains.items()},
        first_timeout_s=float(tts.get("first_audio_timeout_s", 2.0)),
        later_timeout_s=float(tts.get("later_timeout_s", 4.0)),
        first_retries=int(tts.get("first_retries", 0)),
        later_retries=int(tts.get("later_retries", 1)),
        captions_fallback=bool(tts.get("captions_fallback", True)),
        breaker=(
            int(tts.get("breaker_failures", 3)),
            float(tts.get("breaker_window_s", 60.0)),
            float(tts.get("breaker_cooldown_s", 120.0)),
        ),
        chunk=chunk,
        min_chars_by_backend=min_chars,
        default_chain=default_chain,
        cache=cache,
        clock=clock,
        on_fallback=on_fallback,
        on_constraints=on_constraints,
    )
