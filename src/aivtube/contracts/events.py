"""Event base class, the event catalogue and its JSON registry (ARCHITECTURE.md §3.2).

Every event is a ``@dataclass(frozen=True, slots=True, kw_only=True)`` subclass of
:class:`Event`. ``ts`` is ``time.perf_counter()`` seconds; the bus stamps it when it is 0.

JSON form (panel, flight recorder, replay): ``{"type": ClassName, <fields in declaration
order>}``. Enums serialise as their values, tuples as lists, nested dataclasses
(``ChatMessage``, ``ChatUser``, ``Health``, ``Stimulus``) as objects without a ``type`` key.
:func:`event_from_json` restores enums, tuples and nested dataclasses from the field
annotations, so ``event_from_json(event_to_json(e)) == e``. ``Mapping[str, Any]`` payloads
come back as plain dicts of JSON values. Unknown keys are ignored for forward-compatible
replay; a missing required field or a value of the wrong type raises ``ValueError``.

``LipTrack`` is deliberately not an event: it travels on a direct channel to the avatar driver.
"""

import dataclasses
import numbers
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal

from aivtube.contracts.types import ChatMessage, Health, Priority, Stimulus

__all__ = [
    "EVENT_TYPES",
    "Alert",
    "BargeInCandidate",
    "BargeInConfirmed",
    "BargeInRejected",
    "ChatDropped",
    "ChatReceived",
    "ComponentRestarted",
    "DecisionAborted",
    "DecisionStarted",
    "EmotionChanged",
    "Event",
    "Filtered",
    "GameActionResult",
    "GameActionSent",
    "GameConnected",
    "GameContextReceived",
    "GameDisconnected",
    "GameForceReceived",
    "HealthChanged",
    "LLMFirstToken",
    "LatencyMark",
    "MemoryWritten",
    "OperatorAction",
    "ProviderSwitched",
    "SegmentDone",
    "SegmentQueued",
    "SegmentStarted",
    "StateChanged",
    "StimulusExpired",
    "StimulusQueued",
    "SupportReceived",
    "ToolExecuted",
    "ToolRejected",
    "ToolRequested",
    "TurnTraceReady",
    "UserSpeechEnded",
    "UserSpeechStarted",
    "UserTranscript",
    "UtteranceDone",
    "UtteranceStarted",
    "event_from_json",
    "event_to_json",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    ts: float = 0.0  # perf_counter seconds; 0 means "stamp on publish"
    character: str | None = None
    turn_id: str | None = None


# --- input ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class UserSpeechStarted(Event):
    barge: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class UserSpeechEnded(Event):
    audio_s: float


@dataclass(frozen=True, slots=True, kw_only=True)
class UserTranscript(Event):
    text: str
    engine: str
    latency_ms: float
    audio_s: float
    parts: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeInCandidate(Event):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeInConfirmed(Event):
    text: str
    cut_local: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeInRejected(Event):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatReceived(Event):
    message: ChatMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatDropped(Event):
    message_id: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SupportReceived(Event):
    message: ChatMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class StimulusQueued(Event):
    stimulus: Stimulus


@dataclass(frozen=True, slots=True, kw_only=True)
class StimulusExpired(Event):
    stimulus_id: str


# --- brain / speech ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class StateChanged(Event):
    old: str
    new: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionStarted(Event):
    stimulus_id: str
    merged_ids: tuple[str, ...]
    provider: str
    speculative: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionAborted(Event):
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMFirstToken(Event):
    provider: str
    ttft_ms: float
    prompt_n: int | None
    cache_n: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class UtteranceStarted(Event):
    utt_id: str
    stimulus_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentQueued(Event):
    utt_id: str
    seq: int
    caption: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentStarted(Event):
    utt_id: str
    seq: int
    t_audible: float
    duration_s: float | None
    backend: str
    silent: bool
    caption: str
    emotion: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentDone(Event):
    utt_id: str
    seq: int
    heard: bool
    heard_text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class UtteranceDone(Event):
    utt_id: str
    heard_text: str
    cancelled: bool
    reason: str | None
    filtered: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Filtered(Event):
    direction: str
    tier: str
    category: str | None
    rule: str | None
    ref: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRequested(Event):
    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecuted(Event):
    tool: str
    ok: bool
    content: str
    dry_run: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRejected(Event):
    tool: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryWritten(Event):
    memory_id: int
    kind: str
    status: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EmotionChanged(Event):
    emotion: str


# --- infra ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthChanged(Event):
    health: Health


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentRestarted(Event):
    name: str
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderSwitched(Event):
    kind: str
    old: str
    new: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorAction(Event):
    kind: str
    args: Mapping[str, Any]
    ok: bool
    latency_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class LatencyMark(Event):
    stage: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnTraceReady(Event):
    trace: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class Alert(Event):
    level: Literal["info", "warn", "error"]
    message: str


# --- games (M4) -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GameConnected(Event):
    game: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GameDisconnected(Event):
    game: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GameContextReceived(Event):
    game: str
    text: str
    silent: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class GameForceReceived(Event):
    game: str
    force_id: str
    priority: Priority


@dataclass(frozen=True, slots=True, kw_only=True)
class GameActionSent(Event):
    game: str
    name: str
    force_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class GameActionResult(Event):
    game: str
    name: str
    ok: bool
    message: str


# --- registry ---------------------------------------------------------------------------

_ALL_EVENTS: Final[tuple[type[Event], ...]] = (
    UserSpeechStarted,
    UserSpeechEnded,
    UserTranscript,
    BargeInCandidate,
    BargeInConfirmed,
    BargeInRejected,
    ChatReceived,
    ChatDropped,
    SupportReceived,
    StimulusQueued,
    StimulusExpired,
    StateChanged,
    DecisionStarted,
    DecisionAborted,
    LLMFirstToken,
    UtteranceStarted,
    SegmentQueued,
    SegmentStarted,
    SegmentDone,
    UtteranceDone,
    Filtered,
    ToolRequested,
    ToolExecuted,
    ToolRejected,
    MemoryWritten,
    EmotionChanged,
    HealthChanged,
    ComponentRestarted,
    ProviderSwitched,
    OperatorAction,
    LatencyMark,
    TurnTraceReady,
    Alert,
    GameConnected,
    GameDisconnected,
    GameContextReceived,
    GameForceReceived,
    GameActionSent,
    GameActionResult,
)

EVENT_TYPES: Final[Mapping[str, type[Event]]] = types.MappingProxyType(
    {cls.__name__: cls for cls in _ALL_EVENTS}
)
"""Class name -> event class, for every concrete event (the ``Event`` base is excluded)."""


# --- JSON codec -------------------------------------------------------------------------


def event_to_json(e: Event) -> dict[str, Any]:
    """Serialise an event to a JSON-safe dict: ``{"type": ClassName, **fields}``."""
    if not isinstance(e, Event):
        raise TypeError(f"not an Event: {type(e).__name__}")
    out: dict[str, Any] = {"type": type(e).__name__}
    for f in dataclasses.fields(e):
        out[f.name] = _to_json(getattr(e, f.name))
    return out


def event_from_json(d: Mapping[str, Any]) -> Event:
    """Rebuild an event from :func:`event_to_json` output. Raises ``ValueError`` if invalid."""
    if not isinstance(d, Mapping):
        raise ValueError(f"event JSON must be an object, got {type(d).__name__}")
    name = d.get("type")
    if not isinstance(name, str):
        raise ValueError("event JSON has no string 'type'")
    cls = EVENT_TYPES.get(name)
    if cls is None:
        raise ValueError(f"unknown event type {name!r}")
    event = _decode_dataclass(cls, d, name)
    assert isinstance(event, Event)
    return event


def _to_json(value: Any) -> Any:
    """Convert a value to plain JSON types (dict/list/str/int/float/bool/None)."""
    if isinstance(value, Enum):
        return _to_json(value.value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {_key_to_json(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json(v) for v in value]
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    tolist = getattr(value, "tolist", None)  # numpy arrays, should one slip into a payload
    if callable(tolist):
        return _to_json(tolist())
    return str(value)


def _key_to_json(key: Any) -> str:
    if isinstance(key, Enum):
        return str(key.value)
    return key if isinstance(key, str) else str(key)


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    hints = _HINTS.get(cls)
    if hints is None:
        hints = typing.get_type_hints(cls)
        _HINTS[cls] = hints
    return hints


def _decode_dataclass(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected an object for {cls.__name__}")
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ValueError(f"{where}: missing field {f.name!r}")
            continue
        kwargs[f.name] = _decode(hints[f.name], data[f.name], f"{where}.{f.name}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _decode(tp: Any, value: Any, where: str) -> Any:
    if tp is Any:
        return value
    if tp is type(None):
        if value is not None:
            raise ValueError(f"{where}: expected null")
        return None
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        errors: list[str] = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _decode(arg, value, where)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError("; ".join(errors) or f"{where}: no union member matched")
    if origin is Literal:
        allowed = typing.get_args(tp)
        for option in allowed:
            if type(value) is type(option) and value == option:
                return option
        raise ValueError(f"{where}: {value!r} is not one of {allowed!r}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected an array")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ValueError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _decode(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin in (list, Sequence):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected an array")
        (arg,) = typing.get_args(tp) or (Any,)
        return [_decode(arg, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: expected an object")
        args = typing.get_args(tp)
        vt = args[1] if len(args) == 2 else Any
        return {str(k): _decode(vt, v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(tp, type):
        return _decode_class(tp, value, where)
    raise ValueError(f"{where}: unsupported annotation {tp!r}")


def _decode_class(tp: type, value: Any, where: str) -> Any:
    if issubclass(tp, Enum):
        try:
            return tp(value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    if dataclasses.is_dataclass(tp):
        return _decode_dataclass(tp, value, where)
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected a boolean")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string")
        return value
    if not isinstance(value, tp):
        raise ValueError(f"{where}: expected {tp.__name__}")
    return value
