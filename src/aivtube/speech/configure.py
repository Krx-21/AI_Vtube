"""What the core tells the voice worker at every (re)connect (Appendix A ``voice.configure``).

``voice_configure`` turns the validated config into the ``voice.configure`` payload: audio,
VAD, barge-in, the STT chain as inline backend specs (cloud entries only with consent, I8),
the TTS section (identities, backends, ``chunk`` limits, timeouts, cache) and per character
the identity chain, the phrases to pre-synthesise (``Filtered.`` first) and the STT alias map.
Paths stay relative: the worker resolves them against its own project root.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aivtube.contracts.speech import TTSConstraints, VoicePolicy

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig, CharacterConfig

__all__ = ["FILTERED_PHRASE", "default_constraints", "voice_configure", "voice_policy"]

FILTERED_PHRASE = "Filtered."


def voice_configure(cfg: AppConfig, characters: Mapping[str, CharacterConfig]) -> dict[str, Any]:
    """The ``voice.configure`` data for ``cfg`` and the loaded characters."""
    chain: list[dict[str, Any]] = []
    for name in cfg.stt.effective_chain():
        spec = cfg.stt.backends[name].model_dump(exclude_none=True)
        chain.append({"name": name, **spec})
    tts = cfg.tts.model_dump(mode="json")
    tts["chunk"] = dict(tts.get("chunker") or {})
    chars: dict[str, Any] = {}
    for cid, char in characters.items():
        phrases = list(dict.fromkeys([FILTERED_PHRASE, *char.cached_phrases]))
        chars[cid] = {
            "identity_chain": cfg.tts_chain_for(char),
            "cached_phrases": phrases,
            "stt_aliases": {k: list(v) for k, v in char.stt_aliases.items()},
        }
    return {
        "audio": cfg.audio.model_dump(mode="json"),
        "vad": cfg.vad.model_dump(mode="json"),
        "barge_in": cfg.barge_in.model_dump(mode="json"),
        "stt_chain": chain,
        "tts": tts,
        "characters": chars,
    }


def voice_policy(cfg: AppConfig) -> VoicePolicy:
    """The start-of-show ``VoicePolicy`` from ``[mic]``, ``[barge_in]`` and ``[audio]``."""
    return VoicePolicy(
        mic_mode=cfg.mic.mode,
        ptt_active=False,
        barge_in=cfg.barge_in.policy,
        echo_mode=cfg.audio.echo_mode,
        listening=True,
    )


def default_constraints(cfg: AppConfig, character: CharacterConfig) -> TTSConstraints:
    """Chunk limits until the worker reports ``tts.constraints`` (primary identity/backend)."""
    chunk = cfg.tts.chunker
    chain = cfg.tts_chain_for(character)
    identity = chain[0] if chain else "captions"
    ident = cfg.tts.identities.get(identity)
    backend = ident.backends[0] if ident is not None and ident.backends else "captions"
    min_chars = chunk.min_chars
    spec = cfg.tts.backends.get(backend)
    if spec is not None and spec.kind == "azure" and spec.tier == "F0":
        min_chars = max(min_chars, spec.min_chars_when_primary_f0)
    return TTSConstraints(
        first_min_chars=chunk.first_min_chars,
        min_chars=min(min_chars, chunk.max_chars),
        max_chars=chunk.max_chars,
        backend=backend,
        identity=identity,
    )
