"""contracts/events.py (§3.2): catalogue, registry and JSON round trip for every event."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts import events as ev
from aivtube.contracts.events import EVENT_TYPES, Event, event_from_json, event_to_json
from aivtube.contracts.types import (
    ChatMessage,
    ChatUser,
    Health,
    HealthState,
    MsgKind,
    Platform,
    Priority,
    Rank,
    Stimulus,
    StimulusKind,
)

BASE = {"ts": 123.456789, "character": "pailin", "turn_id": "t-42"}

_USER = ChatUser(Platform.TWITCH, "u1", "ต้นกล้า", is_mod=True, is_sub=True, sub_months=7)
_MSG = ChatMessage(
    Platform.TWITCH,
    "m1",
    _USER,
    "ไพลินร้องเพลงหน่อย 555",
    10.25,
    10.5,
    kind=MsgKind.DONATION,
    amount=100.0,
    currency="THB",
    value_usd=2.8,
    first_msg=True,
    reply_to="m0",
    source_channel="other",
    raw={"tags": {"bits": "100"}, "badges": ["sub", "mod"]},
)
_STIM = Stimulus(
    id="s1",
    kind=StimulusKind.SUPPORT,
    character="pailin",
    text="ขอบคุณค่ะ",
    created=11.0,
    priority=Priority.MEDIUM,
    rank=Rank.SUPPORT,
    ttl_s=None,
    source="twitch",
    speaker="ต้นกล้า",
    addressed=True,
    payload={"message_id": "m1", "read_aloud_of": None, "nested": {"n": [1, 2.5, "x"]}},
    trace_id="tr-1",
)

SAMPLES: dict[str, Event] = {
    e.__class__.__name__: e
    for e in [
        ev.UserSpeechStarted(barge=True, **BASE),
        ev.UserSpeechEnded(audio_s=2.5, **BASE),
        ev.UserTranscript(
            text="ไพลินคะ", engine="typhoon_rt", latency_ms=41.5, audio_s=2.5, parts=2
        ),
        ev.BargeInCandidate(**BASE),
        ev.BargeInConfirmed(text="หยุดก่อน", cut_local=True, **BASE),
        ev.BargeInRejected(),
        ev.ChatReceived(message=_MSG, **BASE),
        ev.ChatDropped(message_id="m2", reason="duplicate", **BASE),
        ev.SupportReceived(message=_MSG, **BASE),
        ev.StimulusQueued(stimulus=_STIM, **BASE),
        ev.StimulusExpired(stimulus_id="s1", **BASE),
        ev.StateChanged(old="IDLE", new="DECIDING", **BASE),
        ev.DecisionStarted(
            stimulus_id="s1", merged_ids=("s2", "s3"), provider="local-30b", speculative=True
        ),
        ev.DecisionAborted(reason="critical", **BASE),
        ev.LLMFirstToken(provider="local-30b", ttft_ms=412.0, prompt_n=5120, cache_n=None, **BASE),
        ev.UtteranceStarted(utt_id="utt1", stimulus_id="s1", **BASE),
        ev.SegmentQueued(utt_id="utt1", seq=0, caption="สวัสดีค่ะ", **BASE),
        ev.SegmentStarted(
            utt_id="utt1",
            seq=0,
            t_audible=124.0,
            duration_s=None,
            backend="edge",
            silent=False,
            caption="สวัสดีค่ะ",
            emotion="happy",
            **BASE,
        ),
        ev.SegmentDone(utt_id="utt1", seq=0, heard=True, heard_text="สวัสดีค่ะ", **BASE),
        ev.UtteranceDone(
            utt_id="utt1",
            heard_text="สวัสดี",
            cancelled=True,
            reason="barge_in",
            filtered=False,
            **BASE,
        ),
        ev.Filtered(direction="out", tier="t0", category="monarchy", rule=None, ref="mod-7"),
        ev.ToolRequested(
            tool="remember", args={"text": "ต้นกล้าชอบแมว", "importance": 4, "tags": ["cat"]}
        ),
        ev.ToolExecuted(tool="remember", ok=True, content="saved slot 3", dry_run=True, **BASE),
        ev.ToolRejected(tool="timeout_user", reason="rate_limited", **BASE),
        ev.MemoryWritten(memory_id=17, kind="core", status="quarantined", **BASE),
        ev.EmotionChanged(emotion="smug", **BASE),
        ev.HealthChanged(health=Health("voice", HealthState.DEGRADED, "stt fallback", 99.5)),
        ev.ComponentRestarted(name="chat.twitch", count=3, **BASE),
        ev.ProviderSwitched(
            kind="llm", old="local-30b", new="local-4b", reason="auto_rollback", **BASE
        ),
        ev.OperatorAction(kind="freeze", args={"by": "panel"}, ok=True, latency_ms=12.5, **BASE),
        ev.LatencyMark(stage="first_audible", **BASE),
        ev.TurnTraceReady(
            trace={"turn_id": "t-42", "stages": {"stimulus_in": 1.0, "done": 3.2}, "cache_n": None}
        ),
        ev.Alert(level="warn", message="VRAM ต่ำกว่า 500 MiB", **BASE),
        ev.GameConnected(game="tic-tac-toe", **BASE),
        ev.GameDisconnected(game="tic-tac-toe", **BASE),
        ev.GameContextReceived(game="tic-tac-toe", text="Your turn", silent=False, **BASE),
        ev.GameForceReceived(game="tic-tac-toe", force_id="f1", priority=Priority.CRITICAL, **BASE),
        ev.GameActionSent(game="tic-tac-toe", name="play", force_id=None, **BASE),
        ev.GameActionResult(game="tic-tac-toe", name="play", ok=False, message="bad cell"),
    ]
}


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            assert type(k) is str
            _assert_json_safe(v)
    elif isinstance(value, list):
        for v in value:
            _assert_json_safe(v)
    else:
        assert value is None or type(value) in (str, int, float, bool), repr(value)


def test_registry_contains_every_event_subclass() -> None:
    defined = {
        name: obj
        for name, obj in vars(ev).items()
        if isinstance(obj, type) and issubclass(obj, Event) and obj is not Event
    }
    assert dict(EVENT_TYPES) == defined
    assert "Event" not in EVENT_TYPES
    assert len(EVENT_TYPES) == 39
    assert set(ev.__all__) >= set(EVENT_TYPES)


def test_samples_cover_every_event_type() -> None:
    assert set(SAMPLES) == set(EVENT_TYPES)


@pytest.mark.parametrize("name", sorted(EVENT_TYPES))
def test_event_class_shape(name: str) -> None:
    cls = EVENT_TYPES[name]
    params = cls.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    assert "__slots__" in cls.__dict__
    fields = dataclasses.fields(cls)
    assert all(f.kw_only for f in fields)
    assert [f.name for f in fields[:3]] == ["ts", "character", "turn_id"]
    assert "type" not in {f.name for f in fields}
    sample = SAMPLES[name]
    assert not hasattr(sample, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.ts = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("name", sorted(EVENT_TYPES))
def test_json_round_trip(name: str) -> None:
    event = SAMPLES[name]
    data = event_to_json(event)
    assert next(iter(data)) == "type"
    assert data["type"] == name
    assert list(data)[1:] == [f.name for f in dataclasses.fields(event)]
    _assert_json_safe(data)
    wire = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
    back = event_from_json(wire)
    assert type(back) is type(event)
    assert back == event
    assert event_to_json(back) == data


def test_nested_types_are_restored() -> None:
    chat = event_from_json(json.loads(json.dumps(event_to_json(SAMPLES["ChatReceived"]))))
    assert isinstance(chat, ev.ChatReceived)
    assert type(chat.message) is ChatMessage
    assert type(chat.message.user) is ChatUser
    assert chat.message.platform is Platform.TWITCH
    assert chat.message.kind is MsgKind.DONATION
    assert chat.message.raw == _MSG.raw  # raw is compare=False, so check it explicitly

    stim = event_from_json(event_to_json(SAMPLES["StimulusQueued"]))
    assert isinstance(stim, ev.StimulusQueued)
    assert stim.stimulus.kind is StimulusKind.SUPPORT
    assert stim.stimulus.priority is Priority.MEDIUM
    assert stim.stimulus.rank is Rank.SUPPORT
    assert stim.stimulus.ttl_s is None
    assert stim.stimulus.payload == _STIM.payload

    health = event_from_json(event_to_json(SAMPLES["HealthChanged"]))
    assert isinstance(health, ev.HealthChanged)
    assert health.health.state is HealthState.DEGRADED

    force = event_from_json(event_to_json(SAMPLES["GameForceReceived"]))
    assert isinstance(force, ev.GameForceReceived)
    assert force.priority is Priority.CRITICAL

    started = event_from_json(json.loads(json.dumps(event_to_json(SAMPLES["DecisionStarted"]))))
    assert isinstance(started, ev.DecisionStarted)
    assert started.merged_ids == ("s2", "s3")
    assert type(started.merged_ids) is tuple


def test_enums_serialise_as_values() -> None:
    data = event_to_json(SAMPLES["StimulusQueued"])
    stim = data["stimulus"]
    assert stim["kind"] == "support" and type(stim["kind"]) is str
    assert stim["priority"] == 1 and type(stim["priority"]) is int
    assert stim["rank"] == 30 and type(stim["rank"]) is int
    assert event_to_json(SAMPLES["ChatReceived"])["message"]["user"]["platform"] == "twitch"


def test_defaults_are_filled_and_unknown_keys_ignored() -> None:
    e = event_from_json(
        {
            "type": "UserTranscript",
            "text": "ฮัลโหล",
            "engine": "fake",
            "latency_ms": 5,
            "audio_s": 1,
            "future_field": "ignored",
        }
    )
    assert e == ev.UserTranscript(text="ฮัลโหล", engine="fake", latency_ms=5.0, audio_s=1.0)
    assert isinstance(e, ev.UserTranscript)
    assert (e.parts, e.ts, e.character, e.turn_id) == (1, 0.0, None, None)
    assert type(e.latency_ms) is float  # ints are accepted for float fields


def test_mapping_payload_values_are_plain_json() -> None:
    e = ev.ToolRequested(tool="t", args={"pair": (1, 2), "path": Path("a/b"), 3: "int key"})
    back = event_from_json(json.loads(json.dumps(event_to_json(e))))
    assert isinstance(back, ev.ToolRequested)
    assert back.args == {"pair": [1, 2], "path": str(Path("a/b")), "3": "int key"}


def test_numpy_values_in_any_payloads_become_json_numbers() -> None:
    np = pytest.importorskip("numpy")
    e = ev.TurnTraceReady(trace={"tok_s": np.float32(41.5), "n": np.int64(3), "arr": np.arange(3)})
    data = event_to_json(e)
    _assert_json_safe(data)
    assert data["trace"] == {"tok_s": 41.5, "n": 3, "arr": [0, 1, 2]}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"type": "NoSuchEvent"}, "unknown event type"),
        ({"ts": 1.0}, "no string 'type'"),
        ({"type": "StimulusExpired"}, "missing field 'stimulus_id'"),
        ({"type": "SegmentQueued", "utt_id": "u", "seq": "0", "caption": ""}, "integer"),
        ({"type": "SegmentQueued", "utt_id": "u", "seq": True, "caption": ""}, "integer"),
        ({"type": "SegmentQueued", "utt_id": "u", "seq": 1.5, "caption": ""}, "integer"),
        ({"type": "UserSpeechStarted", "barge": 1}, "boolean"),
        ({"type": "UserSpeechEnded", "audio_s": "1"}, "number"),
        ({"type": "Alert", "level": "fatal", "message": "x"}, "not one of"),
        ({"type": "GameForceReceived", "game": "g", "force_id": "f", "priority": 9}, "Priority"),
        (
            {"type": "DecisionStarted", "stimulus_id": "s", "merged_ids": "ab", "provider": "p"},
            "expected an array",
        ),
        ({"type": "ChatReceived", "message": {"platform": "twitch"}}, "missing field"),
        ({"type": "ChatReceived", "message": []}, "expected an object"),
        ({"type": "EmotionChanged", "emotion": "happy", "character": 5}, "expected a string"),
        ({"type": "ToolRequested", "tool": "t", "args": [1]}, "expected an object"),
    ],
)
def test_invalid_json_raises_value_error(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        event_from_json(payload)


def test_non_mapping_input_raises() -> None:
    with pytest.raises(ValueError):
        event_from_json([("type", "BargeInRejected")])  # type: ignore[arg-type]


def test_event_to_json_rejects_non_events() -> None:
    with pytest.raises(TypeError):
        event_to_json(_MSG)  # type: ignore[arg-type]


def test_events_with_hashable_fields_are_hashable() -> None:
    assert hash(SAMPLES["ChatReceived"]) == hash(dataclasses.replace(SAMPLES["ChatReceived"]))
    assert hash(SAMPLES["DecisionStarted"])
    with pytest.raises(TypeError):
        hash(SAMPLES["ToolRequested"])  # a dict payload is unhashable by design


def test_base_event_defaults() -> None:
    e = ev.BargeInRejected()
    assert (e.ts, e.character, e.turn_id) == (0.0, None, None)
    assert dataclasses.replace(e, ts=5.0).ts == 5.0
