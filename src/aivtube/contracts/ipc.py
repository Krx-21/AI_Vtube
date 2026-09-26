"""Core <-> voice-worker IPC protocol v1: envelope, codec and message schemas (Appendix A).

Transport: WebSocket ``ws://127.0.0.1:8771/bus``, JSON text frames, the core is the server.
Envelope: ``{"v":1,"type":…,"id":…,"corr":…,"ts":<perf_counter>,"data":{…}}``. Every ``ts``
and every time field in ``data`` (``t``, ``t0``, ``t_end``, ``t_audible``) is the sender's
``time.perf_counter()``.

Handshake: the worker sends ``hello`` with its token; the core answers with a ``reply``
(``corr`` = the hello's id) or closes the socket (bad token, different major version).
Then 5 x ``ping``/``pong`` measure RTT and clock offset; ``heartbeat`` runs at 1 Hz and a peer
is dead after 3 misses. ``speak.segment`` is answered with a ``reply`` whose status is
``ok`` or ``busy`` (backpressure: at most 8 queued segments per utterance).

``decode`` accepts unknown message types (the receiver ignores and logs them). ``validate``
checks ``data`` against :data:`MESSAGE_SCHEMAS` with jsonschema; it runs in tests and when
``AIVTUBE_IPC_DEBUG=1``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "BARGE_CANDIDATE",
    "BARGE_CONFIRMED",
    "BARGE_REJECTED",
    "CORE_TO_VOICE",
    "HEALTH",
    "HEARTBEAT",
    "HELLO",
    "IPC_VERSION",
    "LIP_TRACK",
    "MESSAGE_SCHEMAS",
    "PING",
    "PONG",
    "REPLY",
    "SEG_DONE",
    "SEG_STARTED",
    "SESSION",
    "SPEAK_BEGIN",
    "SPEAK_CANNED",
    "SPEAK_DUCK",
    "SPEAK_GATE",
    "SPEAK_SEGMENT",
    "SPEAK_STOP",
    "STT_FINAL",
    "STT_SPECULATIVE",
    "TTS_CONSTRAINTS",
    "TTS_FALLBACK",
    "UTT_DONE",
    "VAD_END",
    "VAD_START",
    "VOICE_CONFIGURE",
    "VOICE_MUTE",
    "VOICE_POLICY",
    "VOICE_RATE",
    "VOICE_TO_CORE",
    "Envelope",
    "IpcError",
    "decode",
    "encode",
    "validate",
]

IPC_VERSION: Final = 1

# --- message types ----------------------------------------------------------------------
# session (both directions)
HELLO: Final = "hello"
PING: Final = "ping"
PONG: Final = "pong"
HEARTBEAT: Final = "heartbeat"
REPLY: Final = "reply"
# core -> voice
VOICE_CONFIGURE: Final = "voice.configure"
VOICE_POLICY: Final = "voice.policy"
SPEAK_BEGIN: Final = "speak.begin"
SPEAK_SEGMENT: Final = "speak.segment"
SPEAK_GATE: Final = "speak.gate"
SPEAK_STOP: Final = "speak.stop"
SPEAK_DUCK: Final = "speak.duck"
SPEAK_CANNED: Final = "speak.canned"
VOICE_MUTE: Final = "voice.mute"
VOICE_RATE: Final = "voice.rate"
# voice -> core
HEALTH: Final = "health"
VAD_START: Final = "vad.start"
VAD_END: Final = "vad.end"
STT_FINAL: Final = "stt.final"
STT_SPECULATIVE: Final = "stt.speculative"  # M2
BARGE_CANDIDATE: Final = "barge.candidate"
BARGE_CONFIRMED: Final = "barge.confirmed"
BARGE_REJECTED: Final = "barge.rejected"
SEG_STARTED: Final = "speech.segment_started"
LIP_TRACK: Final = "lip.track"
SEG_DONE: Final = "speech.segment_done"
UTT_DONE: Final = "speech.utterance_done"
TTS_FALLBACK: Final = "tts.fallback"
TTS_CONSTRAINTS: Final = "tts.constraints"

SESSION: Final[frozenset[str]] = frozenset({HELLO, PING, PONG, HEARTBEAT, REPLY})
CORE_TO_VOICE: Final[frozenset[str]] = frozenset(
    {
        VOICE_CONFIGURE,
        VOICE_POLICY,
        SPEAK_BEGIN,
        SPEAK_SEGMENT,
        SPEAK_GATE,
        SPEAK_STOP,
        SPEAK_DUCK,
        SPEAK_CANNED,
        VOICE_MUTE,
        VOICE_RATE,
    }
)
VOICE_TO_CORE: Final[frozenset[str]] = frozenset(
    {
        HEALTH,
        VAD_START,
        VAD_END,
        STT_FINAL,
        STT_SPECULATIVE,
        BARGE_CANDIDATE,
        BARGE_CONFIRMED,
        BARGE_REJECTED,
        SEG_STARTED,
        LIP_TRACK,
        SEG_DONE,
        UTT_DONE,
        TTS_FALLBACK,
        TTS_CONSTRAINTS,
    }
)


@dataclass(frozen=True, slots=True)
class Envelope:
    v: int
    type: str
    id: str
    ts: float  # sender perf_counter
    data: Mapping[str, Any]
    corr: str | None = None  # id of the message this one answers


class IpcError(Exception):
    """Malformed, unencodable or invalid IPC message."""


# --- codec ------------------------------------------------------------------------------


def encode(env: Envelope) -> str:
    """Serialise to a compact JSON text frame. Raises ``IpcError`` if it cannot be encoded."""
    _check_envelope(env.v, env.type, env.id, env.ts, env.corr)
    if not isinstance(env.data, Mapping):
        raise IpcError("envelope data must be a mapping")
    obj = {
        "v": env.v,
        "type": env.type,
        "id": env.id,
        "corr": env.corr,
        "ts": env.ts,
        "data": env.data,
    }
    try:
        return json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_default
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise IpcError(f"cannot encode {env.type!r}: {exc}") from exc


def decode(raw: str | bytes) -> Envelope:
    """Parse a frame. Raises ``IpcError`` if malformed; an unknown ``type`` is accepted."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IpcError(f"frame is not UTF-8: {exc}") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise IpcError(f"frame must be str or bytes, got {type(raw).__name__}")
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise IpcError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise IpcError("frame must be a JSON object")
    v, mtype, mid, ts, corr = _check_envelope(
        obj.get("v"), obj.get("type"), obj.get("id"), obj.get("ts"), obj.get("corr")
    )
    data = obj.get("data")
    if not isinstance(data, dict):
        raise IpcError("'data' must be a JSON object")
    return Envelope(v=v, type=mtype, id=mid, ts=ts, data=data, corr=corr)


def _check_envelope(
    v: object, mtype: object, mid: object, ts: object, corr: object
) -> tuple[int, str, str, float, str | None]:
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise IpcError(f"'v' must be a positive integer, got {v!r}")
    if not isinstance(mtype, str) or not mtype:
        raise IpcError("'type' must be a non-empty string")
    if not isinstance(mid, str) or not mid:
        raise IpcError("'id' must be a non-empty string")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        raise IpcError(f"'ts' must be a finite number, got {ts!r}")
    if corr is not None and (not isinstance(corr, str) or not corr):
        raise IpcError("'corr' must be a non-empty string or null")
    return v, mtype, mid, ts, corr


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def _default(value: Any) -> Any:
    """``json.dumps`` fallback: mappings, tuples/sets, dataclasses, numpy scalars/arrays."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


# --- schemas ----------------------------------------------------------------------------

_STR: Final[dict[str, Any]] = {"type": "string"}
_NONEMPTY: Final[dict[str, Any]] = {"type": "string", "minLength": 1}
_OPT_STR: Final[dict[str, Any]] = {"type": ["string", "null"]}
_NUM: Final[dict[str, Any]] = {"type": "number"}
_NONNEG: Final[dict[str, Any]] = {"type": "number", "minimum": 0}
_SEQ: Final[dict[str, Any]] = {"type": "integer", "minimum": 0}
_BOOL: Final[dict[str, Any]] = {"type": "boolean"}
_OBJ: Final[dict[str, Any]] = {"type": "object"}
_STR_LIST: Final[dict[str, Any]] = {"type": "array", "items": {"type": "string"}}
_UNIT_LIST: Final[dict[str, Any]] = {
    "type": "array",
    "items": {"type": "number", "minimum": 0, "maximum": 1},
}


def _msg(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] | None = None,
    extra: bool = False,
    title: str,
) -> dict[str, Any]:
    """An object schema; every property is required unless ``required`` says otherwise."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": properties,
        "required": list(properties if required is None else required),
        "additionalProperties": extra,
    }


_SCHEMAS: dict[str, dict[str, Any]] = {
    # session ------------------------------------------------------------------------
    HELLO: _msg(
        {
            "role": _NONEMPTY,  # "voice" for the worker, "core" if the core introduces itself
            "pid": {"type": "integer", "minimum": 0},
            # IPC protocol version: an int major (IPC_VERSION) or a "major.minor" string
            "version": {
                "anyOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)*$"},
                ]
            },
            "token": _NONEMPTY,
            "caps": _OBJ,
        },
        title="hello",
    ),
    PING: _msg({"seq": _SEQ}, required=(), title="ping"),
    PONG: _msg({"ping_ts": _NUM, "seq": _SEQ}, required=("ping_ts",), title="pong"),
    HEARTBEAT: _msg({"seq": _SEQ}, required=(), title="heartbeat"),
    REPLY: _msg(
        {"status": {"enum": ["ok", "busy", "error"]}, "detail": _STR},
        required=("status",),
        extra=True,
        title="reply",
    ),
    # core -> voice ------------------------------------------------------------------
    VOICE_CONFIGURE: _msg(
        {
            "audio": _OBJ,
            "vad": _OBJ,
            "barge_in": _OBJ,
            "stt_chain": {"type": "array", "items": {"type": ["string", "object"]}},
            "tts": {
                "type": "object",
                "properties": {"identities": _OBJ, "backends": _OBJ, "chunk": _OBJ},
                "required": ["identities", "backends", "chunk"],
            },
            "characters": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "identity_chain": {**_STR_LIST, "minItems": 1},
                        "cached_phrases": _STR_LIST,
                    },
                    "required": ["identity_chain", "cached_phrases"],
                },
            },
        },
        title="voice.configure",
    ),
    VOICE_POLICY: _msg(
        {
            "mic_mode": {"enum": ["open", "ptt", "deafened"]},
            "ptt_active": _BOOL,
            "barge_in": {"enum": ["interrupt", "duck_only", "off"]},
            "echo_mode": {"enum": ["auto", "aec", "energy_dtd", "half_duplex", "none"]},
            "listening": _BOOL,
        },
        title="voice.policy",
    ),
    SPEAK_BEGIN: _msg(
        {
            "utt": _NONEMPTY,
            "character": _NONEMPTY,
            "filler_after_s": {"type": ["number", "null"], "minimum": 0},
            "gate_open": _BOOL,
        },
        title="speak.begin",
    ),
    SPEAK_SEGMENT: _msg(
        {
            "utt": _NONEMPTY,
            "seq": _SEQ,
            "text": _STR,
            "caption": _STR,
            "emotion": _OPT_STR,
            "last": _BOOL,
            "kind": {"enum": ["speech", "filtered", "filler", "operator", "canned"]},
        },
        title="speak.segment",
    ),
    SPEAK_GATE: _msg({"utt": _NONEMPTY}, title="speak.gate"),
    SPEAK_STOP: _msg(
        {
            "utt": {"type": ["string", "null"], "minLength": 1},
            "mode": {"enum": ["now", "after_segment"]},
            "reason": _STR,
            "fade_ms": {"type": "integer", "minimum": 0},
        },
        title="speak.stop",
    ),
    SPEAK_DUCK: _msg(
        {
            "gain": {"type": "number", "minimum": 0, "maximum": 1},
            "ramp_ms": {"type": "integer", "minimum": 0},
        },
        title="speak.duck",
    ),
    SPEAK_CANNED: _msg({"key": _NONEMPTY, "character": _NONEMPTY}, title="speak.canned"),
    VOICE_MUTE: _msg({"on": _BOOL}, title="voice.mute"),
    VOICE_RATE: _msg({"character": _NONEMPTY, "percent": {"type": "integer"}}, title="voice.rate"),
    # voice -> core ------------------------------------------------------------------
    HEALTH: _msg(
        {
            "state": {"enum": ["starting", "ok", "degraded", "down", "disabled", "failed"]},
            "detail": _STR,
            "stats": _OBJ,
        },
        title="health",
    ),
    VAD_START: _msg({"t": _NUM, "barge": _BOOL}, title="vad.start"),
    VAD_END: _msg({"t": _NUM, "audio_s": _NONNEG}, title="vad.end"),
    STT_FINAL: _msg(
        {
            "text": _STR,
            "engine": _NONEMPTY,
            "latency_ms": _NONNEG,
            "t_end": _NUM,
            "audio_s": _NONNEG,
            "parts": {"type": "integer", "minimum": 1},
        },
        title="stt.final",
    ),
    STT_SPECULATIVE: _msg(
        {"text": _STR, "t": _NUM, "silence_ms": _NONNEG}, title="stt.speculative"
    ),
    BARGE_CANDIDATE: _msg({"t": _NUM}, title="barge.candidate"),
    BARGE_CONFIRMED: _msg({"t": _NUM, "text": _STR, "cut_local": _BOOL}, title="barge.confirmed"),
    BARGE_REJECTED: _msg({"t": _NUM}, title="barge.rejected"),
    SEG_STARTED: _msg(
        {
            "utt": _NONEMPTY,
            "seq": _SEQ,
            "t_audible": _NUM,
            "duration_s": {"type": ["number", "null"], "minimum": 0},
            "backend": _STR,
            "silent": _BOOL,
        },
        title="speech.segment_started",
    ),
    LIP_TRACK: _msg(
        {
            "utt": _NONEMPTY,
            "seq": _SEQ,
            "t0": _NUM,
            "fps": {"type": "integer", "minimum": 1},
            "mouth": _UNIT_LIST,
            "form": _UNIT_LIST,
            "final": _BOOL,
        },
        title="lip.track",
    ),
    SEG_DONE: _msg(
        {"utt": _NONEMPTY, "seq": _SEQ, "heard": _BOOL, "heard_text": _STR},
        title="speech.segment_done",
    ),
    UTT_DONE: _msg(
        {"utt": _NONEMPTY, "heard_text": _STR, "cancelled": _BOOL, "reason": _OPT_STR},
        title="speech.utterance_done",
    ),
    TTS_FALLBACK: _msg(
        {"from": _STR, "to": _OPT_STR, "reason": _STR},  # to = null: captions only
        title="tts.fallback",
    ),
    TTS_CONSTRAINTS: _msg(
        {
            "character": _NONEMPTY,
            "first_min_chars": _SEQ,
            "min_chars": _SEQ,
            "max_chars": {"type": "integer", "minimum": 1},
            "backend": _STR,
            "identity": _STR,
        },
        title="tts.constraints",
    ),
}

MESSAGE_SCHEMAS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(_SCHEMAS)
"""JSON Schema (Draft 2020-12) of ``data`` for every message type in Appendix A."""

_NEEDS_CORR: Final[frozenset[str]] = frozenset({REPLY, PONG})
_VALIDATORS: dict[str, Any] = {}


def validate(env: Envelope) -> None:
    """Check the envelope and its ``data`` against the schema of its type.

    Raises ``IpcError`` for a wrong version, an unknown type, a ``reply``/``pong`` without
    ``corr``, or data that does not match. ``data`` is checked as it would be sent (after a
    JSON round trip), so tuples and numpy scalars are fine.
    """
    if env.v != IPC_VERSION:
        raise IpcError(f"unsupported IPC version {env.v} (expected {IPC_VERSION})")
    if env.type not in _SCHEMAS:
        raise IpcError(f"unknown message type {env.type!r}")
    if env.type in _NEEDS_CORR and env.corr is None:
        raise IpcError(f"{env.type!r} must carry 'corr'")
    wire = decode(encode(env))
    validator = _VALIDATORS.get(env.type)
    if validator is None:
        from jsonschema import Draft202012Validator

        validator = Draft202012Validator(_SCHEMAS[env.type])
        _VALIDATORS[env.type] = validator
    errors = sorted(validator.iter_errors(wire.data), key=lambda e: str(e.json_path))
    if errors:
        detail = "; ".join(_describe(e) for e in errors)
        raise IpcError(f"invalid {env.type!r} data: {detail}")


def _describe(error: Any) -> str:
    path = "/".join(str(p) for p in error.absolute_path) or "<data>"
    return f"{path}: {error.message}"
