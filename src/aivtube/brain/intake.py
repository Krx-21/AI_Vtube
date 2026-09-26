"""Intake: transcripts, chat, support events and operator input become stimuli (§4.2).

**Voice.** ``mic.mode``: ``open`` (every utterance counts), ``ptt`` (the voice worker only
listens while the key is held, so what arrives counts) and ``deafened`` (ignored: "talking to
chat"). ``mic.addressing``: ``always`` (co-host default) or ``name_or_question``: addressed only
when the transcript contains a name alias (after the STT alias map), is shaped like a question,
or arrives within the 8 s follow-up window after she finished speaking to the streamer.
Unaddressed utterances become a history ``note`` and trigger no decision. VOICE stimuli are
rank VOICE, priority HIGH, TTL 20 s.

**Read-aloud dedupe.** A transcript is compared with chat from the last 60 s (normalised;
containment of at least 6 characters, or a ``SequenceMatcher`` ratio ≥ 0.75). A match is
consumed from the chat window (or its pending MENTION is withdrawn) and the VOICE stimulus
carries ``payload["read_aloud_of"]``, so the message is answered once. Support messages are
never deduped: they are always acknowledged.

**Chat.** Tier-0 input check (``SafetyGate.check_input``, which also returns the display-safe
name); the text is capped at 300 characters with role tokens stripped. A message that mentions
her becomes a MENTION stimulus (rank 40, LOW, TTL 40 s); donations, subs, gift subs, raids and
redeems become SUPPORT (rank 30, MEDIUM, TTL 600 s); everything else goes to the ``ChatWindow``
and pokes one coalesced CHAT stimulus. A donation whose text is blocked is still acknowledged by
name and amount, with the text read as ``Filtered.``. ``ChatReceived`` / ``SupportReceived`` are
published for accepted messages (with the cleaned text) and ``ChatDropped`` for dropped ones.

**Operator.** SAY is CRITICAL and bypasses the LLM; DIRECT is HIGH and never read aloud. Both
are rank OPERATOR with a 60 s TTL.

Everything here is synchronous and cheap (no I/O on the loop); the history note is written by a
tracked task.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import difflib
import logging
import re
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Final, Literal

from aivtube.brain.prompt import CHAT_CAP, sanitize_untrusted
from aivtube.contracts.chat import ChatWindow
from aivtube.contracts.events import ChatDropped, ChatReceived, SupportReceived, UserTranscript
from aivtube.contracts.infra import Clock, EventBus, TaskSupervisor
from aivtube.contracts.safety import FilterResult, SafetyGate, Verdict
from aivtube.contracts.speech import MicMode
from aivtube.contracts.types import ChatMessage, MsgKind, Priority, Rank, Stimulus, StimulusKind
from aivtube.text.normalize import FILTERED, NameMatcher, is_question

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig, CharacterConfig

__all__ = ["SUPPORT_KINDS", "Addressing", "Intake", "IntakeConfig"]

log = logging.getLogger("aivtube.brain.intake")

Addressing = Literal["always", "name_or_question"]

SUPPORT_KINDS: Final = frozenset(
    {MsgKind.DONATION, MsgKind.SUB, MsgKind.GIFT_SUB, MsgKind.RAID, MsgKind.REDEEM}
)
_STOPPING: Final = frozenset({Verdict.DROP, Verdict.BLOCK, Verdict.REVIEW})
_REWRITTEN: Final = frozenset({Verdict.MASK, Verdict.REPLACE})
_ZERO_WIDTH: Final = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
_MAX_RECENT: Final = 200
_MAX_SCAN: Final = 80  # newest chat messages compared per transcript (bounds loop time)
ANON_NAME: Final = "ใครบางคน"  # a viewer whose name cannot be shown ("somebody")


@dataclass(frozen=True, slots=True)
class IntakeConfig:
    """Intake knobs; the first seven mirror ``[mic]``, ``max_msg_chars`` mirrors ``[chat]``."""

    mic_mode: MicMode = "open"
    addressing: Addressing = "always"
    followup_window_s: float = 8.0
    read_aloud_dedupe: bool = True
    read_aloud_ratio: float = 0.75
    read_aloud_window_s: float = 60.0
    read_aloud_min_chars: int = 6
    max_msg_chars: int = CHAT_CAP
    voice_ttl_s: float = 20.0
    mention_ttl_s: float = 40.0
    support_ttl_s: float = 600.0
    operator_ttl_s: float = 60.0
    chat_intake: bool = True
    mute_user_s: float = 600.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> IntakeConfig:
        """Pick the known keys of a flat mapping (unknown keys are ignored)."""
        if not cfg:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})

    @classmethod
    def from_app(cls, app: AppConfig, **overrides: Any) -> IntakeConfig:
        mic = app.mic
        values: dict[str, Any] = {
            "mic_mode": mic.mode,
            "addressing": mic.addressing,
            "followup_window_s": mic.followup_window_s,
            "read_aloud_dedupe": mic.read_aloud_dedupe,
            "read_aloud_ratio": mic.read_aloud_ratio,
            "read_aloud_window_s": mic.read_aloud_window_s,
            "read_aloud_min_chars": mic.read_aloud_min_chars,
            "max_msg_chars": min(CHAT_CAP, app.chat.max_msg_chars),
            "chat_intake": app.chat.enabled,
            "mute_user_s": app.safety.auto_strict_mute_s,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class _Recent:
    """A chat message kept for read-aloud dedupe."""

    t: float
    msg: ChatMessage
    key: str
    stimulus_id: str | None  # a pending MENTION to withdraw instead of consuming the window


def _squash(text: str) -> str:
    """Comparison form: NFKC, casefolded, letters/marks/digits only (Thai tone marks and
    above-vowels are marks, so they are kept; spaces, punctuation and zero-widths go)."""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LMN")


class _AliasFixer:
    """Replaces STT mishearings of the name with the canonical spelling (single pass, longest
    first, case-insensitive). A mishearing that occurs inside its canonical form is skipped, so
    fixing twice (the voice worker already does) never changes the result."""

    def __init__(self, aliases: Mapping[str, list[str]]) -> None:
        self._map: dict[str, str] = {}
        for canonical, wrongs in aliases.items():
            canon = unicodedata.normalize("NFC", canonical.strip())
            if not canon:
                continue
            self._map.setdefault(canon.casefold(), canon)
            for wrong in wrongs:
                w = unicodedata.normalize("NFC", str(wrong).strip())
                if w and w.casefold() not in canon.casefold():
                    self._map.setdefault(w.casefold(), canon)
        keys = sorted(self._map, key=len, reverse=True)
        self._pattern = (
            re.compile("|".join(re.escape(k) for k in keys), re.IGNORECASE) if keys else None
        )

    def __call__(self, text: str) -> str:
        if self._pattern is None:
            return text
        text = unicodedata.normalize("NFC", text)
        return self._pattern.sub(lambda m: self._map.get(m.group(0).casefold(), m.group(0)), text)


class Intake:
    """See the module docstring. Use from the event-loop thread only.

    Extra keyword arguments beyond modules.json: ``withdraw`` removes a pending stimulus by id
    (``Arbiter.remove``; read-aloud dedupe of a mention) and ``tasks`` tracks the history-note
    task (``TaskSupervisor.track``; without it the intake keeps its own strong references).
    """

    def __init__(
        self,
        character: CharacterConfig,
        *,
        gate: SafetyGate,
        window: ChatWindow,
        submit: Callable[[Stimulus], None],
        note: Callable[[str], Awaitable[Any]],
        clock: Clock,
        bus: EventBus,
        cfg: Mapping[str, Any] | IntakeConfig | None = None,
        withdraw: Callable[[str], bool] | None = None,
        tasks: TaskSupervisor | None = None,
    ) -> None:
        self.character = character
        self._id = str(character.id)
        self._gate = gate
        self._window = window
        self._submit = submit
        self._note = note
        self._clock = clock
        self._bus = bus
        self.cfg = cfg if isinstance(cfg, IntakeConfig) else IntakeConfig.from_mapping(cfg)
        self._withdraw = withdraw
        self._tasks = tasks
        self._own_tasks: set[asyncio.Task[Any]] = set()
        self.mic_mode: MicMode = self.cfg.mic_mode
        self.addressing: Addressing = self.cfg.addressing
        self.ptt_active = False
        self.chat_intake = self.cfg.chat_intake
        self._fix_names = _AliasFixer(dict(character.stt_aliases))
        names = [*character.aliases, character.display_name]
        if character.name_th:
            names.append(character.name_th)
        names.extend(character.stt_aliases)
        self._is_named = NameMatcher(names)
        self._last_spoke: float | None = None
        self._recent: collections.deque[_Recent] = collections.deque(maxlen=_MAX_RECENT)
        self._muted: dict[tuple[str, str], float] = {}
        self._muted_names: dict[str, float] = {}

    # --- settings ---------------------------------------------------------------------------
    def set_addressing(self, mode: Addressing) -> None:
        if mode not in ("always", "name_or_question"):
            raise ValueError(f"unknown addressing mode {mode!r}")
        self.addressing = mode

    def set_mic_mode(self, mode: MicMode) -> None:
        if mode not in ("open", "ptt", "deafened"):
            raise ValueError(f"unknown mic mode {mode!r}")
        self.mic_mode = mode

    def set_ptt(self, active: bool) -> None:
        self.ptt_active = bool(active)

    def set_chat_intake(self, on: bool) -> None:
        self.chat_intake = bool(on)

    def mark_spoke_to_streamer(self, t: float) -> None:
        """She finished an utterance to the streamer at ``t``: the follow-up window opens."""
        self._last_spoke = t

    def mute_user(
        self,
        *,
        platform: str | None = None,
        user_id: str | None = None,
        name: str | None = None,
        seconds: float | None = None,
    ) -> bool:
        """Hide a chatter from her view for ``seconds`` (default ``mute_user_s``)."""
        until = self._clock.now() + (self.cfg.mute_user_s if seconds is None else seconds)
        if user_id:
            self._muted[(platform or "", user_id)] = until
            return True
        if name:
            self._muted_names[name.strip().casefold()] = until
            return True
        return False

    def is_muted(self, m: ChatMessage) -> bool:
        now = self._clock.now()
        for key in ((m.platform.value, m.user.id), ("", m.user.id)):
            until = self._muted.get(key)
            if until is not None:
                if now < until:
                    return True
                del self._muted[key]
        until = self._muted_names.get(m.user.name.strip().casefold())
        return until is not None and now < until

    # --- voice ------------------------------------------------------------------------------
    def fix_names(self, text: str) -> str:
        """Apply the character's STT alias map."""
        return self._fix_names(text)

    def addressed(self, text: str, now: float) -> bool:
        """Whether an utterance counts as addressed to her (``name_or_question`` rules)."""
        if self.addressing == "always":
            return True
        if self._is_named(text) or is_question(text):
            return True
        return self._last_spoke is not None and 0 <= now - self._last_spoke <= (
            self.cfg.followup_window_s
        )

    def on_transcript(self, ev: UserTranscript) -> Stimulus | None:
        """A final transcript; returns the VOICE stimulus it submitted, if any."""
        if self.mic_mode == "deafened":
            log.debug("transcript ignored: mic deafened")
            return None
        text = " ".join(self.fix_names(ev.text).split())
        if not text:
            return None
        now = self._clock.now()
        if not self.addressed(text, now):
            self._spawn(self._note(f"[สตรีมเมอร์พูดกับคนอื่น] {text}"), "intake-note")
            return None
        payload: dict[str, Any] = {
            "engine": ev.engine,
            "latency_ms": ev.latency_ms,
            "audio_s": ev.audio_s,
            "parts": ev.parts,
        }
        if ev.ts:
            payload["t_stt_final"] = ev.ts
            payload["t_vad_end"] = ev.ts - max(0.0, ev.latency_ms) / 1000.0
        if self.cfg.read_aloud_dedupe:
            match = self._read_aloud(text, now)
            if match is not None:
                payload["read_aloud_of"] = match.id
                payload["read_aloud_name"] = match.user.name
                payload["read_aloud_text"] = match.text
        stim = Stimulus(
            id=f"voice-{uuid.uuid4().hex[:8]}",
            kind=StimulusKind.VOICE,
            character=self._id,
            text=text,
            created=ev.ts or now,
            priority=Priority.HIGH,
            rank=Rank.VOICE,
            ttl_s=self.cfg.voice_ttl_s,
            source="voice",
            speaker="streamer",
            addressed=True,
            payload=payload,
            trace_id=ev.turn_id or "",
        )
        self._submit(stim)
        return stim

    def _read_aloud(self, text: str, now: float) -> ChatMessage | None:
        """Find (and consume) the chat message the streamer just read aloud."""
        spoken = _squash(text)
        n = self.cfg.read_aloud_min_chars
        if len(spoken) < n:
            return None
        horizon = now - self.cfg.read_aloud_window_s
        while self._recent and self._recent[0].t < horizon:
            self._recent.popleft()
        best: _Recent | None = None
        best_score = 0.0
        threshold = self.cfg.read_aloud_ratio
        matcher = difflib.SequenceMatcher(None, autojunk=False)
        matcher.set_seq2(spoken)
        for scanned, rec in enumerate(reversed(self._recent)):  # newest first: ties go to it
            if scanned >= _MAX_SCAN:
                break
            key = rec.key
            if len(key) < n:
                continue
            if key in spoken or spoken in key:
                score = 1.0
            else:
                short, long_ = sorted((len(key), len(spoken)))
                if 2.0 * short / (short + long_) < threshold:
                    continue  # the ratio cannot reach the threshold
                matcher.set_seq1(key)
                if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                    continue
                score = matcher.ratio()
                if score < threshold:
                    continue
            if score > best_score:
                best, best_score = rec, score
                if score >= 1.0:
                    break
        if best is None:
            return None
        self._recent.remove(best)
        taken = False
        if best.stimulus_id is not None and self._withdraw is not None:
            taken = self._withdraw(best.stimulus_id)
        if not taken:
            self._window.consume(best.msg.id)
        log.debug("read-aloud dedupe: transcript matches chat %s", best.msg.id)
        return best.msg

    # --- chat -------------------------------------------------------------------------------
    def on_chat(self, m: ChatMessage) -> None:
        """One chat line (support kinds are routed to :meth:`on_support`)."""
        if m.kind in SUPPORT_KINDS:
            self.on_support(m)
            return
        if m.kind is MsgKind.SYSTEM:
            self._dropped(m, "system")
            return
        if not self.chat_intake:
            self._dropped(m, "intake_off")
            return
        if self.is_muted(m):
            self._dropped(m, "muted")
            return
        checked = self._check(m)
        if checked is None:
            self._dropped(m, "filter_error")
            return
        res, name = checked
        if res.verdict in _STOPPING:
            self._dropped(m, f"filtered:{res.category or res.verdict.value}")
            return
        raw = res.text if res.verdict in _REWRITTEN else m.text
        text = sanitize_untrusted(raw, self.cfg.max_msg_chars)
        if not text:
            self._dropped(m, "empty")
            return
        clean = dataclasses.replace(m, text=text, user=dataclasses.replace(m.user, name=name))
        now = self._clock.now()
        if self._is_named(text):
            stim = self._chat_stimulus(clean, StimulusKind.MENTION, now)
            self._remember(clean, now, stim.id)
            self._bus.publish(ChatReceived(character=self._id, message=clean))
            self._submit(stim)
            return
        routed = self._window.add(clean)
        if routed == "dropped":
            reason = str(getattr(self._window, "last_reason", "") or "window")
            self._dropped(m, reason)
            return
        self._remember(clean, now, None)
        self._bus.publish(ChatReceived(character=self._id, message=clean))
        self._submit(
            Stimulus(
                id=f"chat-{uuid.uuid4().hex[:8]}",
                kind=StimulusKind.CHAT,
                character=self._id,
                text="",
                created=now,
                priority=Priority.LOW,
                rank=Rank.CHAT,
                ttl_s=None,
                source=m.platform.value,
            )
        )

    def on_support(self, m: ChatMessage) -> None:
        """A donation, sub, gift sub, raid or redeem: always acknowledged (by name, amount)."""
        checked = self._check(m)
        if checked is None:  # fail closed: acknowledge anonymously, text read as Filtered.
            res, name = FilterResult(Verdict.BLOCK, "", "error"), ANON_NAME
        else:
            res, name = checked
        blocked = res.verdict in _STOPPING and bool(m.text.strip())
        if blocked:
            text = FILTERED
        else:
            raw = res.text if res.verdict in _REWRITTEN else m.text
            text = sanitize_untrusted(raw, self.cfg.max_msg_chars)
        clean = dataclasses.replace(m, text=text, user=dataclasses.replace(m.user, name=name))
        now = self._clock.now()
        stim = self._chat_stimulus(clean, StimulusKind.SUPPORT, now)
        if blocked:
            stim = dataclasses.replace(stim, payload={**stim.payload, "filtered": True})
        self._bus.publish(SupportReceived(character=self._id, message=clean))
        self._submit(stim)

    def _chat_stimulus(self, m: ChatMessage, kind: StimulusKind, now: float) -> Stimulus:
        support = kind is StimulusKind.SUPPORT
        u = m.user
        badges = [
            b
            for b, on in (
                ("streamer", u.is_broadcaster),
                ("mod", u.is_mod),
                ("vip", u.is_vip),
                ("sub", u.is_sub),
                ("first", m.first_msg),
            )
            if on
        ]
        payload: dict[str, Any] = {
            "message": m,
            "msg_id": m.id,
            "platform": m.platform.value,
            "user_id": u.id,
            "badges": badges,
        }
        if support:
            payload.update(
                msg_kind=m.kind.value,
                amount=m.amount,
                currency=m.currency,
                value_usd=m.value_usd,
            )
        return Stimulus(
            id=f"{kind.value}-{uuid.uuid4().hex[:8]}",
            kind=kind,
            character=self._id,
            text=m.text,
            created=now,
            priority=Priority.MEDIUM if support else Priority.LOW,
            rank=Rank.SUPPORT if support else Rank.MENTION,
            ttl_s=self.cfg.support_ttl_s if support else self.cfg.mention_ttl_s,
            source=f"{m.platform.value}:{u.id}",
            speaker=u.name,
            addressed=not support,
            payload=payload,
        )

    def _check(self, m: ChatMessage) -> tuple[FilterResult, str] | None:
        """Tier-0 input check; ``None`` when the gate itself failed (the caller fails closed)."""
        try:
            return self._gate.check_input(m, character=self._id)
        except Exception:
            log.exception("input filter failed on chat %s", m.id)
            return None

    def _remember(self, m: ChatMessage, now: float, stimulus_id: str | None) -> None:
        if self.cfg.read_aloud_dedupe:
            self._recent.append(_Recent(now, m, _squash(m.text), stimulus_id))

    def _dropped(self, m: ChatMessage, reason: str) -> None:
        log.debug("chat %s dropped: %s", m.id, reason)
        self._bus.publish(ChatDropped(character=self._id, message_id=m.id, reason=reason))

    # --- operator ---------------------------------------------------------------------------
    def on_operator(self, kind: Literal["say", "direct"], text: str) -> Stimulus:
        """SAY (CRITICAL, spoken verbatim) or DIRECT (HIGH, an instruction never read aloud)."""
        if kind not in ("say", "direct"):
            raise ValueError(f"unknown operator input {kind!r}")
        body = " ".join(text.split())
        if not body:
            raise ValueError("operator text is empty")
        stim = Stimulus(
            id=f"op-{uuid.uuid4().hex[:8]}",
            kind=StimulusKind.OPERATOR,
            character=self._id,
            text=body,
            created=self._clock.now(),
            priority=Priority.CRITICAL if kind == "say" else Priority.HIGH,
            rank=Rank.OPERATOR,
            ttl_s=self.cfg.operator_ttl_s,
            source="operator",
            speaker="operator",
            addressed=True,
            payload={"op": kind},
        )
        self._submit(stim)
        return stim

    # --- internals --------------------------------------------------------------------------
    def _spawn(self, aw: Awaitable[Any], name: str) -> None:
        if self._tasks is not None:
            self._tasks.track(aw, name=name)
            return
        task = asyncio.ensure_future(aw)
        self._own_tasks.add(task)
        task.add_done_callback(self._own_done)

    def _own_done(self, task: asyncio.Task[Any]) -> None:
        self._own_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("intake task failed", exc_info=task.exception())
