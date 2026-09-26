"""Text-to-speech inside the voice worker (§2.8, §3.4, §4.6; TTS brief).

- ``EdgeTTSBackend``: edge-tts through a stateful PyAV MP3 decoder (default, free).
- ``AzureTTSBackend``: Azure Speech with the same Premwadee voice (raw PCM, token-bucketed).
- ``TTSRouter`` / ``UtteranceTTS``: identities, per-segment substitution, breakers, captions.
- ``DiskPhraseCache``: pre-synthesised phrases (``Filtered.``, fillers) on disk.
- ``TokenBucket``: quota for Azure F0.
- ``Mp3StreamDecoder``: sample-exact incremental MP3 decoding.

edge-tts, aiohttp, PyAV and the Azure SDK are imported only when a backend first needs them.
"""

from aivtube.voice.tts.azure import AzureTTS, AzureTTSBackend, azure_ssml
from aivtube.voice.tts.cache import DiskPhraseCache, normalize_phrase, phrase_key
from aivtube.voice.tts.decoder import Mp3StreamDecoder
from aivtube.voice.tts.edge import (
    CA_BUNDLE_ENV,
    EdgeTTS,
    EdgeTTSBackend,
    edge_ssl_context,
    install_edge_ssl_context,
)
from aivtube.voice.tts.factory import build_tts_backend, build_tts_router, identities_from_config
from aivtube.voice.tts.quota import TokenBucket
from aivtube.voice.tts.router import (
    CAPTIONS,
    BackendBreaker,
    ChunkLimits,
    IdentityCfg,
    SynthStream,
    TTSRouter,
    UtteranceTTS,
    shift_rate,
)

__all__ = [
    "CAPTIONS",
    "CA_BUNDLE_ENV",
    "AzureTTS",
    "AzureTTSBackend",
    "BackendBreaker",
    "ChunkLimits",
    "DiskPhraseCache",
    "EdgeTTS",
    "EdgeTTSBackend",
    "IdentityCfg",
    "Mp3StreamDecoder",
    "SynthStream",
    "TTSRouter",
    "TokenBucket",
    "UtteranceTTS",
    "azure_ssml",
    "build_tts_backend",
    "build_tts_router",
    "edge_ssl_context",
    "identities_from_config",
    "install_edge_ssl_context",
    "normalize_phrase",
    "phrase_key",
    "shift_rate",
]
