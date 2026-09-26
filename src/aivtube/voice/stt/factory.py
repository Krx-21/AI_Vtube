"""Build the STT chain from config (``[stt]``) or the ``voice.configure`` IPC payload.

``stt_chain`` entries are backend names (resolved in ``backends``) or inline objects with a
``kind``. Kinds: ``sherpa_nemo_transducer`` (Typhoon RT), ``pythaiasr``, ``openai_audio``
(Typhoon API, cloud) and ``fake`` (CI). ``faster_whisper_worker`` is M2 and skipped.

Cloud backends are skipped unless ``cloud_consent`` is true (I8), whatever the payload says.
Constructing ``SherpaTyphoonRT`` loads the model (about 1 s): call ``build_stt_chain`` through
``asyncio.to_thread``. A backend that cannot be built (missing files, missing key) is logged
and left out, so the rest of the chain still works.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aivtube.contracts.voice import SpeechRecognizer

__all__ = ["build_recognizer", "build_stt_chain"]

log = logging.getLogger("aivtube.voice.stt")

SecretLookup = Callable[[str], str | None]


def _resolve(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def build_recognizer(
    name: str,
    spec: Mapping[str, Any],
    *,
    root: Path,
    secrets: SecretLookup,
    cloud_consent: bool = False,
    timeout_s: float = 3.0,
) -> SpeechRecognizer | None:
    """One recognizer from a backend spec, or ``None`` when it is disabled or not allowed."""
    kind = str(spec.get("kind", ""))
    if not spec.get("enabled", True):
        return None
    cloud = bool(spec.get("cloud", False)) or kind == "openai_audio"
    if cloud and not cloud_consent:
        log.info("STT backend %s skipped: cloud STT needs privacy.cloud_stt_consent", name)
        return None
    if kind == "sherpa_nemo_transducer":
        from aivtube.voice.stt.sherpa import SherpaTyphoonRT

        return SherpaTyphoonRT(
            _resolve(root, str(spec.get("model_dir", ""))),
            num_threads=int(spec.get("num_threads", 2)),
            name=name,
        )
    if kind == "pythaiasr":
        from aivtube.voice.stt.pythaiasr_backend import PyThaiAsrRecognizer

        model_dir = spec.get("model_dir") or None
        return PyThaiAsrRecognizer(
            model_dir=_resolve(root, str(model_dir)) if model_dir else None, name=name
        )
    if kind == "openai_audio":
        from aivtube.voice.stt.typhoon_api import TyphoonApiRecognizer

        env = str(spec.get("api_key_env") or "")
        key = secrets(env) if env else None
        if not key:
            log.warning("STT backend %s skipped: %s is not set", name, env or "api_key_env")
            return None
        return TyphoonApiRecognizer(
            str(spec.get("base_url", "")),
            str(spec.get("model", "")),
            key,
            timeout_s=timeout_s,
            name=name,
        )
    if kind == "fake":
        from aivtube.testing.fakes.stt import FakeRecognizer

        script = spec.get("script", [])
        return FakeRecognizer(script, float(spec.get("delay_s", 0.0)), name=name)
    log.warning("STT backend %s skipped: kind %r is not supported here", name, kind)
    return None


def build_stt_chain(
    chain: Sequence[str | Mapping[str, Any]],
    backends: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
    secrets: SecretLookup,
    cloud_consent: bool = False,
    timeout_s: float = 3.0,
) -> list[SpeechRecognizer]:
    """Recognizers in chain order; entries that fail to build are logged and skipped."""
    out: list[SpeechRecognizer] = []
    for entry in chain:
        if isinstance(entry, str):
            name, spec = entry, backends.get(entry)
            if spec is None:
                log.warning("STT chain entry %s has no backend section", entry)
                continue
        else:
            spec = entry
            name = str(entry.get("name") or entry.get("kind") or "stt")
        try:
            rec = build_recognizer(
                name,
                spec,
                root=root,
                secrets=secrets,
                cloud_consent=cloud_consent,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            log.warning("STT backend %s could not be built: %s", name, exc)
            continue
        if rec is not None:
            out.append(rec)
    return out
