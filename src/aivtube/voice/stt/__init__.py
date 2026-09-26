"""Speech recognition inside the voice worker (§3.4, §2.8; STT brief).

- ``SherpaTyphoonRT``: Typhoon ASR Realtime int8 in sherpa-onnx (default, CPU, 0 VRAM).
- ``PyThaiAsrRecognizer``: PyThaiASR (Typhoon RT fp32 ONNX) fallback.
- ``TyphoonApiRecognizer``: the hosted Typhoon API, only with ``cloud_stt_consent``.
- ``NamePostProcessor``: name alias map, short-audio drop, echo drop.
- ``SttRunner``: decode threads with priority queues and the 3 s timeout chain.
- ``UtteranceAssembler``: joins forced-split partial decodes with the final decode.

Native dependencies (sherpa_onnx, pythaiasr, httpx2) are imported only when a recognizer is
built, so ``import aivtube.voice.stt`` stays light.
"""

from aivtube.voice.stt.factory import build_recognizer, build_stt_chain
from aivtube.voice.stt.post import NamePostProcessor, echo_similarity
from aivtube.voice.stt.pythaiasr_backend import PyThaiAsrRecognizer
from aivtube.voice.stt.runner import SttRunner, UtteranceAssembler
from aivtube.voice.stt.sherpa import SherpaTyphoonRT, find_transducer_files
from aivtube.voice.stt.typhoon_api import TyphoonApiRecognizer, wav_bytes

__all__ = [
    "NamePostProcessor",
    "PyThaiAsrRecognizer",
    "SherpaTyphoonRT",
    "SttRunner",
    "TyphoonApiRecognizer",
    "UtteranceAssembler",
    "build_recognizer",
    "build_stt_chain",
    "echo_similarity",
    "find_transducer_files",
    "wav_bytes",
]
