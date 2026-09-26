"""contracts/ipc.py (§3.14, Appendix A): envelope codec and per-message schemas."""

from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType
from typing import Any, get_args

import pytest
from jsonschema import Draft202012Validator

from aivtube.contracts import ipc
from aivtube.contracts.ipc import (
    IPC_VERSION,
    MESSAGE_SCHEMAS,
    Envelope,
    IpcError,
    decode,
    encode,
    validate,
)
from aivtube.contracts.speech import BargePolicy, EchoMode, MicMode, StopMode, VoicePolicy
from aivtube.contracts.types import HealthState, Segment, SegmentKind

# Field names exactly as listed in Appendix A (plus the session messages we pin down).
APPENDIX_A: dict[str, set[str]] = {
    "hello": {"role", "pid", "version", "token", "caps"},
    "voice.configure": {"audio", "vad", "barge_in", "stt_chain", "tts", "characters"},
    "voice.policy": {"mic_mode", "ptt_active", "barge_in", "echo_mode", "listening"},
    "speak.begin": {"utt", "character", "filler_after_s", "gate_open"},
    "speak.segment": {"utt", "seq", "text", "caption", "emotion", "last", "kind"},
    "speak.gate": {"utt"},
    "speak.stop": {"utt", "mode", "reason", "fade_ms"},
    "speak.duck": {"gain", "ramp_ms"},
    "speak.canned": {"key", "character"},
    "voice.mute": {"on"},
    "voice.rate": {"character", "percent"},
    "health": {"state", "detail", "stats"},
    "vad.start": {"t", "barge"},
    "vad.end": {"t", "audio_s"},
    "stt.final": {"text", "engine", "latency_ms", "t_end", "audio_s", "parts"},
    "stt.speculative": {"text", "t", "silence_ms"},
    "barge.candidate": {"t"},
    "barge.confirmed": {"t", "text", "cut_local"},
    "barge.rejected": {"t"},
    "speech.segment_started": {"utt", "seq", "t_audible", "duration_s", "backend", "silent"},
    "lip.track": {"utt", "seq", "t0", "fps", "mouth", "form", "final"},
    "speech.segment_done": {"utt", "seq", "heard", "heard_text"},
    "speech.utterance_done": {"utt", "heard_text", "cancelled", "reason"},
    "tts.fallback": {"from", "to", "reason"},
    "tts.constraints": {
        "character",
        "first_min_chars",
        "min_chars",
        "max_chars",
        "backend",
        "identity",
    },
}

VALID: dict[str, dict[str, Any]] = {
    "hello": {
        "role": "voice",
        "pid": 4242,
        "version": 1,
        "token": "s3cret",
        "caps": {"aec": True, "stt": ["typhoon_rt", "pythaiasr"]},
    },
    "ping": {"seq": 0},
    "pong": {"ping_ts": 1234.5678, "seq": 0},
    "heartbeat": {"seq": 17},
    "reply": {"status": "busy", "detail": "8 segments queued"},
    "voice.configure": {
        "audio": {"output_device": "", "samplerate": 48000, "block_ms": 10},
        "vad": {"threshold": 0.5, "end_silence_ms": 600},
        "barge_in": {"policy": "interrupt", "duck_db": -12.0},
        "stt_chain": ["typhoon_rt", {"name": "pythaiasr", "kind": "pythaiasr"}],
        "tts": {
            "identities": {"premwadee": {"voice": "th-TH-PremwadeeNeural", "backends": ["edge"]}},
            "backends": {"edge": {"kind": "edge"}},
            "chunk": {"first_min_chars": 8, "min_chars": 40, "max_chars": 160},
        },
        "characters": {
            "pailin": {
                "identity_chain": ["premwadee"],
                "cached_phrases": ["Filtered.", "อืม…", "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ"],
            }
        },
    },
    "voice.policy": {
        "mic_mode": "ptt",
        "ptt_active": True,
        "barge_in": "duck_only",
        "echo_mode": "energy_dtd",
        "listening": True,
    },
    "speak.begin": {"utt": "u1", "character": "pailin", "filler_after_s": 1.2, "gate_open": True},
    "speak.segment": {
        "utt": "u1",
        "seq": 0,
        "text": "สวัสดีค่ะ ทุกคน",
        "caption": "สวัสดีค่ะ ทุกคน",
        "emotion": None,
        "last": False,
        "kind": "speech",
    },
    "speak.gate": {"utt": "u1"},
    "speak.stop": {"utt": None, "mode": "now", "reason": "freeze", "fade_ms": 30},
    "speak.duck": {"gain": 0.25, "ramp_ms": 30},
    "speak.canned": {"key": "filtered", "character": "pailin"},
    "voice.mute": {"on": True},
    "voice.rate": {"character": "pailin", "percent": -20},
    "health": {"state": "degraded", "detail": "stt fallback", "stats": {"underflows": 0}},
    "vad.start": {"t": 100.25, "barge": False},
    "vad.end": {"t": 102.5, "audio_s": 2.25},
    "stt.final": {
        "text": "ไพลินวันนี้เล่นเกมอะไร",
        "engine": "typhoon_rt",
        "latency_ms": 48.2,
        "t_end": 102.55,
        "audio_s": 2.25,
        "parts": 1,
    },
    "stt.speculative": {"text": "ไพลิน", "t": 102.3, "silence_ms": 300},
    "barge.candidate": {"t": 50.0},
    "barge.confirmed": {"t": 50.6, "text": "หยุดก่อน", "cut_local": True},
    "barge.rejected": {"t": 52.0},
    "speech.segment_started": {
        "utt": "u1",
        "seq": 0,
        "t_audible": 103.1,
        "duration_s": None,
        "backend": "edge",
        "silent": False,
    },
    "lip.track": {
        "utt": "u1",
        "seq": 0,
        "t0": 103.1,
        "fps": 60,
        "mouth": [0.0, 0.4, 1.0],
        "form": [0.5, 0.5, 0.7],
        "final": True,
    },
    "speech.segment_done": {"utt": "u1", "seq": 0, "heard": True, "heard_text": "สวัสดีค่ะ"},
    "speech.utterance_done": {
        "utt": "u1",
        "heard_text": "สวัสดีค่ะ",
        "cancelled": True,
        "reason": "barge_in",
    },
    "tts.fallback": {"from": "edge", "to": None, "reason": "first_audio_timeout"},
    "tts.constraints": {
        "character": "pailin",
        "first_min_chars": 8,
        "min_chars": 60,
        "max_chars": 160,
        "backend": "azure",
        "identity": "premwadee",
    },
}


def _without(d: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != key}


def _with(d: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {**d, **kw}


INVALID: dict[str, dict[str, Any]] = {
    "hello": _with(VALID["hello"], version="one"),
    "ping": {"seq": -1},
    "pong": {"seq": 0},
    "heartbeat": {"seq": "1"},
    "reply": {"status": "maybe"},
    "voice.configure": _with(
        VALID["voice.configure"], characters={"pailin": {"identity_chain": []}}
    ),
    "voice.policy": _with(VALID["voice.policy"], mic_mode="whisper"),
    "speak.begin": _without(VALID["speak.begin"], "gate_open"),
    "speak.segment": _with(VALID["speak.segment"], kind="shout"),
    "speak.gate": {"utt": ""},
    "speak.stop": _with(VALID["speak.stop"], mode="later"),
    "speak.duck": _with(VALID["speak.duck"], gain=1.5),
    "speak.canned": {"key": "filtered"},
    "voice.mute": {"on": "yes"},
    "voice.rate": _with(VALID["voice.rate"], percent=1.5),
    "health": _with(VALID["health"], state="sleepy"),
    "vad.start": {"t": "now", "barge": False},
    "vad.end": _with(VALID["vad.end"], audio_s=-1.0),
    "stt.final": _with(VALID["stt.final"], parts=0),
    "stt.speculative": _without(VALID["stt.speculative"], "silence_ms"),
    "barge.candidate": {},
    "barge.confirmed": _with(VALID["barge.confirmed"], cut_local=None),
    "barge.rejected": {"t": 1.0, "extra": True},
    "speech.segment_started": _with(VALID["speech.segment_started"], seq=-1),
    "lip.track": _with(VALID["lip.track"], mouth=[0.1, "open"]),
    "speech.segment_done": _without(VALID["speech.segment_done"], "heard_text"),
    "speech.utterance_done": _with(VALID["speech.utterance_done"], cancelled="true"),
    "tts.fallback": _with(VALID["tts.fallback"], to=3),
    "tts.constraints": _with(VALID["tts.constraints"], max_chars=0),
}


def _env(mtype: str, data: dict[str, Any], *, corr: str | None = None) -> Envelope:
    if corr is None and mtype in ("reply", "pong"):
        corr = "req-1"
    return Envelope(v=IPC_VERSION, type=mtype, id=f"{mtype}-1", ts=1234.5678, data=data, corr=corr)


def test_every_message_type_has_a_schema_and_samples() -> None:
    constants = ipc.SESSION | ipc.CORE_TO_VOICE | ipc.VOICE_TO_CORE
    assert set(MESSAGE_SCHEMAS) == constants
    assert set(VALID) == set(MESSAGE_SCHEMAS) == set(INVALID)
    assert set(APPENDIX_A) <= set(MESSAGE_SCHEMAS)
    assert {"hello", "ping", "pong", "heartbeat"} <= set(MESSAGE_SCHEMAS)
    assert not (ipc.CORE_TO_VOICE & ipc.VOICE_TO_CORE)
    assert not (ipc.SESSION & (ipc.CORE_TO_VOICE | ipc.VOICE_TO_CORE))


def test_message_constants() -> None:
    assert (ipc.SEG_STARTED, ipc.SEG_DONE, ipc.UTT_DONE) == (
        "speech.segment_started",
        "speech.segment_done",
        "speech.utterance_done",
    )
    for name in ipc.__all__:
        value = getattr(ipc, name)
        if name.isupper() and isinstance(value, str):
            assert value in MESSAGE_SCHEMAS, name


@pytest.mark.parametrize("mtype", sorted(APPENDIX_A))
def test_schema_fields_match_appendix_a(mtype: str) -> None:
    schema = MESSAGE_SCHEMAS[mtype]
    assert set(schema["properties"]) == APPENDIX_A[mtype]
    assert set(schema["required"]) == APPENDIX_A[mtype]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("mtype", sorted(MESSAGE_SCHEMAS))
def test_schema_is_valid_draft_2020_12(mtype: str) -> None:
    Draft202012Validator.check_schema(dict(MESSAGE_SCHEMAS[mtype]))


@pytest.mark.parametrize("mtype", sorted(MESSAGE_SCHEMAS))
def test_valid_sample_validates_and_round_trips(mtype: str) -> None:
    env = _env(mtype, VALID[mtype])
    validate(env)
    raw = encode(env)
    back = decode(raw)
    assert back == env
    assert encode(back) == raw
    assert decode(raw.encode("utf-8")) == env
    validate(back)


@pytest.mark.parametrize("mtype", sorted(MESSAGE_SCHEMAS))
def test_invalid_sample_is_rejected(mtype: str) -> None:
    env = _env(mtype, INVALID[mtype])
    raw = encode(env)  # the codec does not look at schemas
    assert decode(raw) == env
    with pytest.raises(IpcError, match="invalid"):
        validate(env)


def test_schema_enums_match_contract_literals() -> None:
    policy = MESSAGE_SCHEMAS["voice.policy"]["properties"]
    assert policy["mic_mode"]["enum"] == list(get_args(MicMode))
    assert policy["barge_in"]["enum"] == list(get_args(BargePolicy))
    assert policy["echo_mode"]["enum"] == list(get_args(EchoMode))
    seg = MESSAGE_SCHEMAS["speak.segment"]["properties"]
    assert seg["kind"]["enum"] == list(get_args(SegmentKind))
    assert MESSAGE_SCHEMAS["speak.stop"]["properties"]["mode"]["enum"] == list(get_args(StopMode))
    health = MESSAGE_SCHEMAS["health"]["properties"]["state"]["enum"]
    assert health == [s.value for s in HealthState]


def test_contract_dataclasses_fit_their_messages() -> None:
    validate(_env("voice.policy", dataclasses.asdict(VoicePolicy())))
    seg = Segment("u9", 3, "ข้อความ", "ข้อความ", emotion="happy", last=True, kind="filtered")
    data = {"utt": seg.utt_id, **_without(dataclasses.asdict(seg), "utt_id")}
    validate(_env("speak.segment", data))


def test_reply_and_pong_need_corr() -> None:
    for mtype in ("reply", "pong"):
        env = Envelope(v=1, type=mtype, id="x", ts=1.0, data=VALID[mtype])
        with pytest.raises(IpcError, match="corr"):
            validate(env)


def test_validate_rejects_unknown_type_and_other_version() -> None:
    with pytest.raises(IpcError, match="unknown message type"):
        validate(_env("speak.sing", {}))
    env = dataclasses.replace(_env("voice.mute", {"on": True}), v=IPC_VERSION + 1)
    with pytest.raises(IpcError, match="version"):
        validate(env)


def test_unknown_type_decodes() -> None:
    raw = '{"v":1,"type":"future.thing","id":"a1","corr":null,"ts":1.5,"data":{"x":[1,2]}}'
    env = decode(raw)
    assert env.type == "future.thing"
    assert env.data == {"x": [1, 2]}
    assert encode(env) == raw


def test_encode_format_is_compact_ordered_and_keeps_thai() -> None:
    env = Envelope(v=1, type="speak.canned", id="c1", ts=2.0, data={"key": "อืม", "character": "p"})
    raw = encode(env)
    assert raw == (
        '{"v":1,"type":"speak.canned","id":"c1","corr":null,"ts":2.0,'
        '"data":{"key":"อืม","character":"p"}}'
    )
    assert json.loads(raw)["data"]["key"] == "อืม"


def test_decode_without_corr_key_and_int_ts() -> None:
    env = decode('{"v":1,"type":"voice.mute","id":"m","ts":7,"data":{"on":false}}')
    assert env.corr is None
    assert env.ts == 7
    validate(env)


def test_encode_normalises_python_values() -> None:
    np = pytest.importorskip("numpy")
    data = {
        "utt": "u",
        "seq": np.int64(0),
        "t0": np.float64(1.5),
        "fps": 60,
        "mouth": np.array([0.0, 0.5], dtype=np.float32),
        "form": (0.5, 0.5),
        "final": False,
    }
    env = _env("lip.track", MappingProxyType(data))  # type: ignore[arg-type]
    validate(env)
    wire = json.loads(encode(env))["data"]
    assert wire == {
        "utt": "u",
        "seq": 0,
        "t0": 1.5,
        "fps": 60,
        "mouth": [0.0, 0.5],
        "form": [0.5, 0.5],
        "final": False,
    }
    policy = json.loads(encode(_env("voice.policy", {"p": VoicePolicy()})))
    assert policy["data"]["p"]["mic_mode"] == "open"


@pytest.mark.parametrize(
    "env",
    [
        Envelope(v=1, type="x", id="a", ts=1.0, data={"bad": float("nan")}),
        Envelope(v=1, type="x", id="a", ts=1.0, data={"bad": object()}),
        Envelope(v=1, type="", id="a", ts=1.0, data={}),
        Envelope(v=1, type="x", id="", ts=1.0, data={}),
        Envelope(v=0, type="x", id="a", ts=1.0, data={}),
        Envelope(v=1, type="x", id="a", ts=float("inf"), data={}),
        Envelope(v=1, type="x", id="a", ts=1.0, data={}, corr=""),
        Envelope(v=1, type="x", id="a", ts=1.0, data=[1]),  # type: ignore[arg-type]
    ],
)
def test_encode_rejects_bad_envelopes(env: Envelope) -> None:
    with pytest.raises(IpcError):
        encode(env)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[1, 2]",
        '"just a string"',
        '{"type":"voice.mute","id":"a","ts":1,"data":{}}',
        '{"v":true,"type":"voice.mute","id":"a","ts":1,"data":{}}',
        '{"v":"1","type":"voice.mute","id":"a","ts":1,"data":{}}',
        '{"v":1,"id":"a","ts":1,"data":{}}',
        '{"v":1,"type":"","id":"a","ts":1,"data":{}}',
        '{"v":1,"type":"voice.mute","ts":1,"data":{}}',
        '{"v":1,"type":"voice.mute","id":5,"ts":1,"data":{}}',
        '{"v":1,"type":"voice.mute","id":"a","ts":"1","data":{}}',
        '{"v":1,"type":"voice.mute","id":"a","ts":NaN,"data":{}}',
        '{"v":1,"type":"voice.mute","id":"a","ts":1,"data":{"x":Infinity}}',
        '{"v":1,"type":"voice.mute","id":"a","ts":1}',
        '{"v":1,"type":"voice.mute","id":"a","ts":1,"data":[]}',
        '{"v":1,"type":"voice.mute","id":"a","ts":1,"data":{},"corr":5}',
        b"\xff\xfe{}",
    ],
)
def test_decode_rejects_malformed(raw: str | bytes) -> None:
    with pytest.raises(IpcError):
        decode(raw)


def test_decode_rejects_non_text_input() -> None:
    with pytest.raises(IpcError):
        decode(123)  # type: ignore[arg-type]
