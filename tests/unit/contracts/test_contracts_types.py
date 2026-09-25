"""contracts/types.py (§3.1): enums, ordering, frozen/slots dataclasses, defaults."""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from enum import Enum, StrEnum
from typing import Any

import pytest

import aivtube.contracts as contracts
from aivtube.contracts.control import OpKind
from aivtube.contracts.infra import Overflow
from aivtube.contracts.safety import Verdict
from aivtube.contracts.types import (
    CONTRACTS_VERSION,
    ChatMessage,
    ChatUser,
    Health,
    HealthState,
    MsgKind,
    Platform,
    Priority,
    Rank,
    Segment,
    Stimulus,
    StimulusKind,
    Transcript,
    VoiceSpec,
)


def _user() -> ChatUser:
    return ChatUser(Platform.TWITCH, "u1", "ต้นกล้า", is_sub=True, sub_months=3)


def _msg(**kw: Any) -> ChatMessage:
    return ChatMessage(Platform.TWITCH, "m1", _user(), "สวัสดีไพลิน", 10.0, 10.5, **kw)


def _all_contract_modules() -> list[Any]:
    return [
        importlib.import_module(f"aivtube.contracts.{m.name}")
        for m in pkgutil.iter_modules(contracts.__path__)
    ]


def _all_contract_dataclasses() -> list[type]:
    seen: dict[str, type] = {}
    for mod in _all_contract_modules():
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and dataclasses.is_dataclass(obj)
                and obj.__module__ == mod.__name__
            ):
                seen[f"{mod.__name__}.{obj.__name__}"] = obj
    return list(seen.values())


def test_contracts_version() -> None:
    assert CONTRACTS_VERSION == "1.0"
    assert contracts.CONTRACTS_VERSION == CONTRACTS_VERSION


def test_rank_ordering_lower_wins() -> None:
    assert Rank.OPERATOR < Rank.VOICE < Rank.MENTION < Rank.CHAT < Rank.GAME_CONTEXT < Rank.IDLE
    expected = [
        ("OPERATOR", 0),
        ("VOICE", 10),
        ("FORCE_URGENT", 20),
        ("SUPPORT", 30),
        ("MENTION", 40),
        ("FORCE", 50),
        ("CHAT", 60),
        ("GAME_CONTEXT", 70),
        ("CHARACTER", 75),
        ("VISION", 80),
        ("IDLE", 90),
    ]
    assert [(r.name, r.value) for r in Rank] == expected
    assert sorted(Rank, reverse=True)[0] is Rank.IDLE
    assert min(Rank) is Rank.OPERATOR


def test_priority_ordering() -> None:
    assert [(p.name, p.value) for p in Priority] == [
        ("LOW", 0),
        ("MEDIUM", 1),
        ("HIGH", 2),
        ("CRITICAL", 3),
    ]
    assert Priority.CRITICAL > Priority.HIGH > Priority.MEDIUM > Priority.LOW


@pytest.mark.parametrize(
    ("enum_cls", "names"),
    [
        (
            StimulusKind,
            "OPERATOR VOICE GAME_FORCE SUPPORT MENTION CHAT GAME_CONTEXT CHARACTER VISION IDLE",
        ),
        (Platform, "TWITCH YOUTUBE TIKTOK CONSOLE ALERT"),
        (MsgKind, "TEXT DONATION SUB GIFT_SUB RAID REDEEM SYSTEM"),
        (HealthState, "STARTING OK DEGRADED DOWN DISABLED FAILED"),
        (Overflow, "DROP_OLDEST DROP_NEWEST"),
        (Verdict, "PASS MASK REPLACE DROP BLOCK REVIEW"),
    ],
)
def test_str_enums_have_lowercase_values(enum_cls: type[StrEnum], names: str) -> None:
    assert [m.name for m in enum_cls] == names.split()
    for member in enum_cls:
        assert member.value == member.name.lower()
        assert member == member.name.lower()  # StrEnum compares equal to its value


def test_opkind_members() -> None:
    names = [
        "SKIP",
        "MUTE",
        "UNMUTE",
        "FREEZE",
        "RESUME",
        "GO_LIVE",
        "CHAT_INTAKE",
        "MIC_MODE",
        "PTT",
        "SAY",
        "DIRECT",
        "FAKE_CHAT",
        "INJECT_EVENT",
        "LLM_USE",
        "LLM_ROLLBACK",
        "TTS_IDENTITY",
        "TOOLS_MODE",
        "TOOL_ENABLE",
        "APPROVE",
        "MEMORY_EDIT",
        "MEMORY_STATUS",
        "MUTE_USER",
        "STRICT",
        "FILTER_RELOAD",
        "RESTART",
        "END_STREAM",
    ]
    assert [m.name for m in OpKind] == names
    assert all(m.value == m.name.lower() for m in OpKind)


def test_every_contract_dataclass_is_frozen_and_slotted() -> None:
    classes = _all_contract_dataclasses()
    assert len(classes) > 40
    for cls in classes:
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, cls
        assert "__slots__" in cls.__dict__, cls


def test_enums_are_all_int_or_str_enums() -> None:
    for mod in _all_contract_modules():
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, Enum) and obj.__module__ == mod.__name__:
                assert issubclass(obj, (StrEnum, int)), obj


SAMPLES: list[Any] = [
    _user(),
    _msg(kind=MsgKind.DONATION, amount=100.0, currency="THB", raw={"tags": {"bits": "100"}}),
    Health("voice", HealthState.OK, "ready", 1.5),
    VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz"),
    Segment("utt1", 0, "สวัสดีค่ะ", "สวัสดีค่ะ", emotion="happy", last=True),
    Transcript("ไพลินคะ", True, 1.2, 45.0, "typhoon_rt"),
]


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_frozen_slots_hashable(obj: Any) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, dataclasses.fields(obj)[0].name, "x")
    assert not hasattr(obj, "__dict__")
    assert hash(obj) == hash(dataclasses.replace(obj))
    assert obj == dataclasses.replace(obj)


def test_chat_message_raw_is_excluded_from_eq_hash_repr() -> None:
    a = _msg(raw={"secret": "oauth:abc"})
    b = _msg(raw={"other": 1})
    assert a == b
    assert hash(a) == hash(b)
    assert "oauth" not in repr(a)
    assert _msg().raw == {}
    assert _msg().raw is not _msg().raw  # default_factory, not a shared dict


def test_chat_defaults() -> None:
    u = ChatUser(Platform.YOUTUBE, "c", "n")
    assert (u.is_broadcaster, u.is_mod, u.is_vip, u.is_sub, u.sub_months, u.is_verified) == (
        False,
        False,
        False,
        False,
        0,
        False,
    )
    m = ChatMessage(Platform.YOUTUBE, "i", u, "t", 1.0, 2.0)
    assert m.kind is MsgKind.TEXT
    assert (m.amount, m.currency, m.value_usd, m.first_msg) == (0.0, "", 0.0, False)
    assert (m.reply_to, m.source_channel) == (None, None)


def test_stimulus_is_keyword_only_with_spec_defaults() -> None:
    with pytest.raises(TypeError):
        Stimulus("s1", StimulusKind.CHAT, "pailin", "hi", 1.0)  # type: ignore[misc]
    s = Stimulus(id="s1", kind=StimulusKind.CHAT, character="pailin", text="hi", created=1.0)
    assert s.priority is Priority.LOW
    assert s.rank is Rank.CHAT
    assert s.ttl_s == 30.0
    assert (s.source, s.speaker, s.addressed, s.trace_id) == ("", None, False, "")
    assert s.payload == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "x"  # type: ignore[misc]


def test_other_defaults() -> None:
    assert Health("x", HealthState.STARTING).detail == ""
    assert Health("x", HealthState.STARTING).since == 0.0
    v = VoiceSpec("id", "voice")
    assert (v.rate, v.pitch, v.volume) == ("+0%", "+0Hz", "+0%")
    seg = Segment("u", 1, "t", "c")
    assert (seg.emotion, seg.last, seg.kind) == (None, False, "speech")


def test_field_order_matches_spec() -> None:
    assert [f.name for f in dataclasses.fields(ChatMessage)] == [
        "platform",
        "id",
        "user",
        "text",
        "ts",
        "received",
        "kind",
        "amount",
        "currency",
        "value_usd",
        "first_msg",
        "reply_to",
        "source_channel",
        "raw",
    ]
    assert [f.name for f in dataclasses.fields(Stimulus)] == [
        "id",
        "kind",
        "character",
        "text",
        "created",
        "priority",
        "rank",
        "ttl_s",
        "source",
        "speaker",
        "addressed",
        "payload",
        "trace_id",
    ]
    assert [f.name for f in dataclasses.fields(Segment)] == [
        "utt_id",
        "seq",
        "text",
        "caption",
        "emotion",
        "last",
        "kind",
    ]
