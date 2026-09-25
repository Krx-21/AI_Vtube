"""Shared value types used by every module (ARCHITECTURE.md §3.1).

Frozen at M0: any change needs an ADR in ``docs/adr`` and a ``CONTRACTS_VERSION`` bump.

Time rule: every timestamp in these types (``ts``, ``received``, ``created``, ``since``) is
``time.perf_counter()`` seconds, obtained through ``Clock.now()``. Wall-clock time
(``Clock.wall()``) is used only when something is written to storage.

This module imports only the standard library.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal, TypeAlias

__all__ = [
    "CONTRACTS_VERSION",
    "ChatMessage",
    "ChatUser",
    "Health",
    "HealthState",
    "MsgKind",
    "Platform",
    "Priority",
    "Rank",
    "Segment",
    "SegmentKind",
    "Stimulus",
    "StimulusKind",
    "Transcript",
    "VoiceSpec",
]

CONTRACTS_VERSION = "1.0"


class Priority(IntEnum):
    """Speech priority with Neuro SDK semantics (T1); see §4.4."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class Rank(IntEnum):
    """Arbitration rank of a stimulus (§4.3). Lower wins."""

    OPERATOR = 0
    VOICE = 10
    FORCE_URGENT = 20
    SUPPORT = 30
    MENTION = 40
    FORCE = 50
    CHAT = 60
    GAME_CONTEXT = 70
    CHARACTER = 75
    VISION = 80
    IDLE = 90


class StimulusKind(StrEnum):
    OPERATOR = "operator"
    VOICE = "voice"
    GAME_FORCE = "game_force"
    SUPPORT = "support"
    MENTION = "mention"
    CHAT = "chat"
    GAME_CONTEXT = "game_context"
    CHARACTER = "character"
    VISION = "vision"
    IDLE = "idle"


class Platform(StrEnum):
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    CONSOLE = "console"
    ALERT = "alert"


class MsgKind(StrEnum):
    TEXT = "text"
    DONATION = "donation"
    SUB = "sub"
    GIFT_SUB = "gift_sub"
    RAID = "raid"
    REDEEM = "redeem"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ChatUser:
    platform: Platform
    id: str
    name: str
    is_broadcaster: bool = False
    is_mod: bool = False
    is_vip: bool = False
    is_sub: bool = False
    sub_months: int = 0
    is_verified: bool = False


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat line or alert. ``ts`` is the platform time mapped to perf_counter;
    ``received`` is our perf_counter at receipt. ``raw`` is excluded from eq/hash/repr."""

    platform: Platform
    id: str
    user: ChatUser
    text: str
    ts: float
    received: float
    kind: MsgKind = MsgKind.TEXT
    amount: float = 0.0
    currency: str = ""
    value_usd: float = 0.0
    first_msg: bool = False
    reply_to: str | None = None
    source_channel: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stimulus:
    """Something the brain may decide to respond to. ``text`` is ALREADY input-filtered."""

    id: str
    kind: StimulusKind
    character: str
    text: str
    created: float
    priority: Priority = Priority.LOW
    rank: Rank = Rank.CHAT
    ttl_s: float | None = 30.0
    source: str = ""
    speaker: str | None = None
    addressed: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str = ""


class HealthState(StrEnum):
    STARTING = "starting"
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Health:
    component: str
    state: HealthState
    detail: str = ""
    since: float = 0.0


@dataclass(frozen=True, slots=True)
class VoiceSpec:
    """A TTS voice. ``rate``/``pitch``/``volume`` use edge-tts/SSML prosody strings."""

    identity: str
    voice: str
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"


SegmentKind: TypeAlias = Literal["speech", "filtered", "filler", "operator", "canned"]


@dataclass(frozen=True, slots=True)
class Segment:
    """One speakable, already output-filtered piece of an utterance (invariant I7)."""

    utt_id: str
    seq: int
    text: str
    caption: str
    emotion: str | None = None
    last: bool = False
    kind: SegmentKind = "speech"


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    is_final: bool
    audio_s: float
    latency_ms: float
    engine: str
