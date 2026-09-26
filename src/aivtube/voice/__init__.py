"""Voice worker package (ARCHITECTURE.md §2.1, §2.4, §3.4).

Everything that touches audio lives here: device I/O, echo handling, VAD, endpointing, the
barge-in reflex, STT, TTS and lip-sync. The core never imports this package; it talks to the
voice worker over IPC (Appendix A).

The front-end names below are re-exported lazily (PEP 562), so ``import aivtube.voice.stt``
does not import the audio front-end, and nothing here imports a native audio stack
(sounddevice, soxr, onnxruntime, sherpa-onnx, livekit) until it is actually used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aivtube.voice.aec import (
        EnergyDTD,
        HalfDuplexGate,
        LiveKitAEC,
        NullAEC,
        WebRtcAEC,
        make_echo_canceller,
    )
    from aivtube.voice.audio_io import (
        AudioDeviceError,
        DeviceInfo,
        MicCapture,
        StreamingPlayer,
        list_devices,
        reinit_portaudio,
        resolve_device,
    )
    from aivtube.voice.barge import (
        BargeConfig,
        BargeInController,
        barge_threshold_for,
        is_backchannel,
    )
    from aivtube.voice.endpointer import SileroEndpointer
    from aivtube.voice.frontend import VoiceFrontEnd, post_to_loop
    from aivtube.voice.vad import (
        EnergyVAD,
        ModelChecksumError,
        SherpaSileroVAD,
        SileroOrtVAD,
        make_vad,
    )

_LAZY: dict[str, str] = {
    # audio_io
    "AudioDeviceError": "aivtube.voice.audio_io",
    "DeviceInfo": "aivtube.voice.audio_io",
    "MicCapture": "aivtube.voice.audio_io",
    "StreamingPlayer": "aivtube.voice.audio_io",
    "list_devices": "aivtube.voice.audio_io",
    "reinit_portaudio": "aivtube.voice.audio_io",
    "resolve_device": "aivtube.voice.audio_io",
    # aec
    "EnergyDTD": "aivtube.voice.aec",
    "HalfDuplexGate": "aivtube.voice.aec",
    "LiveKitAEC": "aivtube.voice.aec",
    "NullAEC": "aivtube.voice.aec",
    "WebRtcAEC": "aivtube.voice.aec",
    "make_echo_canceller": "aivtube.voice.aec",
    # vad
    "EnergyVAD": "aivtube.voice.vad",
    "ModelChecksumError": "aivtube.voice.vad",
    "SherpaSileroVAD": "aivtube.voice.vad",
    "SileroOrtVAD": "aivtube.voice.vad",
    "make_vad": "aivtube.voice.vad",
    # endpointer / front-end / barge-in
    "SileroEndpointer": "aivtube.voice.endpointer",
    "VoiceFrontEnd": "aivtube.voice.frontend",
    "post_to_loop": "aivtube.voice.frontend",
    "BargeConfig": "aivtube.voice.barge",
    "BargeInController": "aivtube.voice.barge",
    "barge_threshold_for": "aivtube.voice.barge",
    "is_backchannel": "aivtube.voice.barge",
}

__all__ = [
    "AudioDeviceError",
    "BargeConfig",
    "BargeInController",
    "DeviceInfo",
    "EnergyDTD",
    "EnergyVAD",
    "HalfDuplexGate",
    "LiveKitAEC",
    "MicCapture",
    "ModelChecksumError",
    "NullAEC",
    "SherpaSileroVAD",
    "SileroEndpointer",
    "SileroOrtVAD",
    "StreamingPlayer",
    "VoiceFrontEnd",
    "WebRtcAEC",
    "barge_threshold_for",
    "is_backchannel",
    "list_devices",
    "make_echo_canceller",
    "make_vad",
    "post_to_loop",
    "reinit_portaudio",
    "resolve_device",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'aivtube.voice' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
