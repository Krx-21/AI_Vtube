"""Core-side speech output (ARCHITECTURE.md §3.5, Appendix A).

- ``BusSpeechOutput``: the ``SpeechOutput`` the core uses when the voice worker runs; every
  call becomes an IPC message and the worker's messages come back as bus events.
- ``ConsoleSpeechOutput``: text mode; prints captions with simulated speaking time.
- ``voice_configure`` / ``voice_policy`` / ``default_constraints``: what the core sends the
  worker at every (re)connect, derived from the config.

The core never touches audio; segments reach the worker only after the output gate (I7).
"""

from aivtube.speech.configure import (
    FILTERED_PHRASE,
    default_constraints,
    voice_configure,
    voice_policy,
)
from aivtube.speech.console import CONSOLE_CANNED, ConsoleSpeechOutput
from aivtube.speech.output import BusSpeechOutput

__all__ = [
    "CONSOLE_CANNED",
    "FILTERED_PHRASE",
    "BusSpeechOutput",
    "ConsoleSpeechOutput",
    "default_constraints",
    "voice_configure",
    "voice_policy",
]
