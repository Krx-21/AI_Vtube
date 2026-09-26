"""The local barge-in reflex (ARCHITECTURE.md §4.7): duck, confirm, cut, recover.

The reflex runs in the voice worker so it does not wait for the core; the policy stays in the
core (a confirmed barge-in becomes a CRITICAL event there).

1. **Candidate.** A ``VadStart`` while she is audible (the endpointer already used its barge
   threshold) ducks the player by ``duck_db`` within ~80 ms and sends ``barge.candidate``.
2. **Confirm.** From ``confirm_min_ms`` after the onset, a quick decode of the last
   ``quick_decode_s`` runs on the STT thread (high priority). It confirms when the text has at
   least ``confirm_min_chars`` non-space characters and is not a backchannel. A backchannel
   heard within ``boundary_ignore_s`` of her utterance start ends the candidate at once;
   otherwise the decode repeats every ``redecode_s``.
3. **Act.** ``interrupt``: cut locally with a ``cut_fade_ms`` fade and send
   ``barge.confirmed{cut_local: true}``. ``duck_only``: stay ducked and send
   ``cut_local: false``. ``off``: never react (song mode).
4. **False alarm.** No confirmation within ``false_timeout_s``, the VAD ending first, or her
   playback ending: un-duck and send ``barge.rejected``.
5. **Echo.** ``half_duplex`` never barges; with ``aec`` barge-in is ignored for ``aec_warmup_s``
   after she first speaks (AEC3 converging); quick decodes get her recent TTS text so the STT
   post-processor drops echoes. ``barge_threshold_for`` gives the endpointer threshold per mode.

The controller owns the player gain: route mute/volume through ``set_base_gain`` so an
un-duck restores the right level. Call every method on the event-loop thread, and run
``run()`` as a supervised task.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Final, Literal, Protocol, runtime_checkable

from aivtube.contracts.infra import Clock
from aivtube.contracts.ipc import BARGE_CANDIDATE, BARGE_CONFIRMED, BARGE_REJECTED
from aivtube.contracts.speech import BargePolicy, EchoMode
from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import F32, AudioOut
from aivtube.infra.clock import DeadlineExceeded, deadline

__all__ = [
    "DEFAULT_BACKCHANNELS",
    "BargeConfig",
    "BargeInController",
    "BargeState",
    "QuickTranscriber",
    "RecentAudio",
    "barge_threshold_for",
    "is_backchannel",
]

log = logging.getLogger("aivtube.voice.barge")

DEFAULT_BACKCHANNELS: Final[frozenset[str]] = frozenset(
    {
        "อืม",
        "อือ",
        "อ๋อ",
        "เออ",
        "ครับ",
        "ค่ะ",
        "คะ",
        "จ้ะ",
        "จ้า",
        "ฮะ",
        "โอเค",
        "เหรอ",
        "หรอ",
        "555",
        "ฮ่า",
        "ฮ่าๆ",
    }
)

BargeState = Literal["idle", "candidate", "confirmed"]

_ECHO_BARGE_THRESHOLD: Final[Mapping[str, float]] = {"aec": 0.65, "energy_dtd": 0.7}
_PLAYING_TAIL_S: Final = 0.1  # "she is audible" for a new candidate (bridges segment gaps)
_ENDED_TAIL_S: Final = 0.25  # "her utterance ended" for an open candidate
_RESTORE_AFTER_CUT_S: Final = 0.15  # after the fade and the device buffer have played out
_STRIP = re.compile(r"[\s​.,!?…~\-–—'\"“”‘’()\[\]{}:;/\\ๆ]+")
_RUNS = re.compile(r"(.)\1+")


@dataclass(frozen=True, slots=True)
class BargeConfig:
    duck_db: float = -12.0
    confirm_min_ms: int = 500
    confirm_min_chars: int = 3
    false_timeout_s: float = 2.0
    cut_fade_ms: int = 60
    aec_warmup_s: float = 3.0
    boundary_ignore_s: float = 1.0
    backchannels: frozenset[str] = field(default=DEFAULT_BACKCHANNELS)
    quick_decode_s: float = 0.8
    redecode_s: float = 0.25
    decode_timeout_s: float = 1.0
    duck_ramp_ms: float = 30.0
    unduck_ramp_ms: float = 100.0

    def __post_init__(self) -> None:
        if self.duck_db > 0:
            raise ValueError("duck_db must be <= 0")
        if self.false_timeout_s <= 0 or self.quick_decode_s <= 0 or self.redecode_s <= 0:
            raise ValueError("false_timeout_s, quick_decode_s and redecode_s must be positive")
        if self.confirm_min_ms < 0 or self.confirm_min_chars < 1 or self.cut_fade_ms < 0:
            raise ValueError("confirm_min_ms/cut_fade_ms must be >= 0, confirm_min_chars >= 1")

    @property
    def duck_gain(self) -> float:
        return float(10.0 ** (self.duck_db / 20.0))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BargeConfig:
        """Build from ``voice.configure.barge_in`` / ``[barge_in]`` (unknown keys ignored)."""
        names = {f.name for f in fields(cls)}
        kw: dict[str, Any] = {k: v for k, v in data.items() if k in names}
        if "backchannels" in kw:
            kw["backchannels"] = frozenset(str(b) for b in kw["backchannels"])
        return cls(**kw)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).casefold()
    t = _STRIP.sub("", t)
    return _RUNS.sub(r"\1", t)  # อืมมม → อืม, 5555 → 5, ครับบ → ครับ


@functools.lru_cache(maxsize=16)
def _tokens(backchannels: frozenset[str]) -> tuple[str, ...]:
    toks = {_normalize(b) for b in backchannels}
    return tuple(sorted((t for t in toks if t), key=len, reverse=True))


def is_backchannel(text: str, backchannels: frozenset[str] = DEFAULT_BACKCHANNELS) -> bool:
    """Whether ``text`` is only backchannel tokens (``อืม``, ``ครับ ครับ``, ``5555``, ``ฮ่าๆๆ``).

    Case, spaces, punctuation, ``ๆ`` and stretched letters are ignored.
    """
    t = _normalize(text)
    toks = _tokens(frozenset(backchannels))
    if not t or not toks:
        return False
    ok = [False] * (len(t) + 1)  # ok[i]: t[:i] splits into tokens
    ok[0] = True
    for i in range(len(t)):
        if ok[i]:
            for tok in toks:
                if t.startswith(tok, i):
                    ok[i + len(tok)] = True
    return ok[len(t)]


def barge_threshold_for(echo_mode: EchoMode, base: float) -> float:
    """The endpointer's ``barge_threshold`` for an echo mode (§4.7: 0.65–0.7 on speakers)."""
    return max(base, _ECHO_BARGE_THRESHOLD.get(echo_mode, base))


@runtime_checkable
class QuickTranscriber(Protocol):
    """What the controller needs from ``voice.stt.SttRunner``."""

    async def transcribe(
        self, pcm16k: F32, *, quick: bool = False, recent_tts_text: str = ""
    ) -> Transcript | None: ...


@runtime_checkable
class RecentAudio(Protocol):
    """What the controller needs from ``VoiceFrontEnd``."""

    def recent_audio(self, seconds: float) -> F32: ...


class BargeInController:
    """The local barge-in state machine (see the module docstring)."""

    def __init__(
        self,
        *,
        player: AudioOut,
        stt: QuickTranscriber,
        frontend: RecentAudio,
        cfg: BargeConfig | None = None,
        send: Callable[[str, Mapping[str, Any]], None],
        clock: Callable[[], float] | Clock = time.perf_counter,
        policy: BargePolicy = "interrupt",
        recent_tts_text: Callable[[], str] | None = None,
        on_cut: Callable[[], None] | None = None,
        tick_s: float = 0.02,
    ) -> None:
        self._player = player
        self._stt = stt
        self._frontend = frontend
        self._cfg = cfg or BargeConfig()
        self._send_fn = send
        self._clock: Clock | None = clock if isinstance(clock, Clock) else None
        self._now: Callable[[], float] = clock.now if isinstance(clock, Clock) else clock
        self._sleep: Callable[[float], Awaitable[None]] = (
            self._clock.sleep if self._clock is not None else asyncio.sleep
        )
        self._policy: BargePolicy = policy
        self._recent_tts = recent_tts_text
        self._on_cut = on_cut
        self._tick = tick_s
        self._echo_mode: EchoMode = "none"
        self._warmup_pending = False
        self._warmup_until = float("-inf")
        self._state: BargeState = "idle"
        self._gen = 0
        self._onset = 0.0
        self._next_decode = 0.0
        self._deadline = 0.0
        self._cut = False
        self._restore_at: float | None = None
        self._base_gain = 1.0
        self._playing = False
        self._her_start: float | None = None
        self._her_end: float | None = None
        self.last_text = ""
        self.stats: Counter[str] = Counter()

    # --- configuration --------------------------------------------------------------------------
    @property
    def state(self) -> BargeState:
        return self._state

    @property
    def policy(self) -> BargePolicy:
        return self._policy

    @property
    def config(self) -> BargeConfig:
        return self._cfg

    def set_config(self, cfg: BargeConfig) -> None:
        self._cfg = cfg

    def set_policy(self, policy: BargePolicy) -> None:
        self._policy = policy
        if policy == "off" and self._state == "candidate":
            self._reject("policy")

    def set_echo_mode(self, mode: EchoMode) -> None:
        """Tell the controller the echo mode in effect; ``aec`` (re)starts the warm-up."""
        self._echo_mode = mode
        self._warmup_until = float("-inf")
        self._warmup_pending = mode == "aec"
        if self._warmup_pending and self._playing:
            self._warmup_pending = False
            self._warmup_until = self._now() + self._cfg.aec_warmup_s

    def set_base_gain(self, gain: float) -> None:
        """The level to restore after a duck (mute = 0.0); applied at once."""
        self._base_gain = max(0.0, float(gain))
        ducked = self._state == "candidate" or (self._state == "confirmed" and not self._cut)
        self._set_gain(self._base_gain * (self._cfg.duck_gain if ducked else 1.0), 20.0)

    def reset(self) -> None:
        """Drop any candidate without a message and restore the gain (e.g. link lost)."""
        self._gen += 1
        self._state = "idle"
        self._restore_at = None
        self._set_gain(self._base_gain, self._cfg.unduck_ramp_ms)

    # --- VAD events (loop thread) -----------------------------------------------------------------
    def on_vad_start(self, t: float, during_playback: bool) -> None:
        """A ``VadStart`` from the front-end; ``during_playback`` is its ``barge`` flag."""
        now = self._now()
        self._poll_playback(now)
        if self._policy == "off" or self._echo_mode == "half_duplex":
            self.stats["ignored_policy"] += 1
            return
        if not during_playback or not self._player.is_speaking(_PLAYING_TAIL_S):
            self.stats["ignored_idle"] += 1
            return
        if self._in_warmup(now):
            self.stats["ignored_warmup"] += 1
            return
        if self._state == "candidate":
            return
        cfg = self._cfg
        self._gen += 1
        self._state = "candidate"
        self._cut = False
        self._onset = t
        self._next_decode = max(now, t + cfg.confirm_min_ms / 1000.0)
        self._deadline = now + cfg.false_timeout_s
        self._restore_at = None
        self.last_text = ""
        self._set_gain(self._base_gain * cfg.duck_gain, cfg.duck_ramp_ms)
        self.stats["candidates"] += 1
        self._send(BARGE_CANDIDATE, {"t": t})

    def on_vad_end(self, t: float) -> None:
        """A ``VadEnd`` (the streamer stopped): a pending candidate is a false alarm."""
        if self._state == "candidate":
            self._reject("vad_end")
        elif self._state == "confirmed":
            self._state = "idle"
            if not self._cut:  # duck_only: she kept playing ducked; bring her back
                self._set_gain(self._base_gain, self._cfg.unduck_ramp_ms)

    def suppress_backchannel(self, text: str, t: float) -> bool:
        """True for a backchannel spoken over her, or within ``boundary_ignore_s`` of her
        utterance start or end: the worker should drop it instead of sending ``stt.final``."""
        if not is_backchannel(text, self._cfg.backchannels):
            return False
        self._poll_playback(self._now())
        start, end, b = self._her_start, self._her_end, self._cfg.boundary_ignore_s
        if start is None or t < start - b:
            return False
        return self._playing or end is None or end < start or t <= end + b

    # --- the task ---------------------------------------------------------------------------------
    async def run(self) -> None:
        """Poll timers and run confirm decodes until cancelled."""
        while True:
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats["step_error"] += 1
                if self.stats["step_error"] <= 3 or self.stats["step_error"] % 500 == 0:
                    log.exception("barge-in step failed (%d so far)", self.stats["step_error"])
            await self._sleep(self._tick)

    async def _step(self) -> None:
        now = self._now()
        self._poll_playback(now)
        if self._restore_at is not None and now >= self._restore_at:
            self._restore_at = None
            if self._state != "candidate":
                self._set_gain(self._base_gain, 5.0)
        if self._state != "candidate":
            return
        if not self._player.is_speaking(_ENDED_TAIL_S):
            self._reject("playback_ended")
        elif now >= self._deadline:
            self._reject("timeout")
        elif now >= self._next_decode:
            await self._try_confirm()

    async def _try_confirm(self) -> None:
        cfg = self._cfg
        gen = self._gen
        pcm = self._frontend.recent_audio(cfg.quick_decode_s)
        recent = ""
        if self._recent_tts is not None and self._echo_mode in ("aec", "energy_dtd"):
            try:
                recent = self._recent_tts()
            except Exception:
                log.exception("recent_tts_text failed")
        budget = max(0.05, min(cfg.decode_timeout_s, self._deadline - self._now()))
        self.stats["decodes"] += 1
        tr: Transcript | None = None
        try:
            async with deadline(budget, what="barge-in quick decode", clock=self._clock):
                tr = await self._stt.transcribe(pcm, quick=True, recent_tts_text=recent)
        except DeadlineExceeded:
            self.stats["decode_timeout"] += 1
        except Exception as exc:
            self.stats["decode_error"] += 1
            log.warning("barge-in quick decode failed: %s", exc)
        if gen != self._gen or self._state != "candidate":
            self.stats["stale_decode"] += 1
            return
        text = tr.text.strip() if tr is not None else ""
        self.last_text = text
        now = self._now()
        if self._confirms(text):
            self._confirm(text, now)
        elif text and is_backchannel(text, cfg.backchannels) and self._near_her_start():
            self._reject("backchannel")
        else:
            self._next_decode = now + cfg.redecode_s

    # --- helpers ----------------------------------------------------------------------------------
    def _confirms(self, text: str) -> bool:
        chars = len("".join(text.split()))
        return chars >= self._cfg.confirm_min_chars and not is_backchannel(
            text, self._cfg.backchannels
        )

    def _confirm(self, text: str, now: float) -> None:
        self._state = "confirmed"
        self.stats["confirmed"] += 1
        cut = self._policy == "interrupt"
        self._cut = cut
        if cut:
            try:
                self._player.cancel(fade_ms=float(self._cfg.cut_fade_ms))
            except Exception:
                log.exception("player.cancel failed")
            if self._on_cut is not None:
                try:
                    self._on_cut()
                except Exception:
                    log.exception("on_cut failed")
            latency = float(getattr(self._player, "output_latency_s", 0.05))
            self._restore_at = now + self._cfg.cut_fade_ms / 1000.0 + latency + _RESTORE_AFTER_CUT_S
            self.stats["cuts"] += 1
        self._send(BARGE_CONFIRMED, {"t": now, "text": text, "cut_local": cut})

    def _reject(self, reason: str) -> None:
        self._gen += 1
        self._state = "idle"
        self.stats["rejected"] += 1
        self.stats[f"rejected_{reason}"] += 1
        self._set_gain(self._base_gain, self._cfg.unduck_ramp_ms)
        self._send(BARGE_REJECTED, {"t": self._now()})

    def _near_her_start(self) -> bool:
        start = self._her_start
        return start is not None and self._onset - start < self._cfg.boundary_ignore_s

    def _in_warmup(self, now: float) -> bool:
        return self._echo_mode == "aec" and (self._warmup_pending or now < self._warmup_until)

    def _poll_playback(self, now: float) -> None:
        playing = self._player.is_speaking(_ENDED_TAIL_S)
        if playing and not self._playing:
            self._her_start = now
            if self._warmup_pending:
                self._warmup_pending = False
                self._warmup_until = now + self._cfg.aec_warmup_s
        elif self._playing and not playing:
            self._her_end = now
        self._playing = playing

    def _set_gain(self, gain: float, ramp_ms: float) -> None:
        try:
            self._player.set_gain(gain, ramp_ms=ramp_ms)
        except Exception:
            log.exception("set_gain failed")

    def _send(self, kind: str, data: Mapping[str, Any]) -> None:
        try:
            self._send_fn(kind, data)
        except Exception:
            self.stats["send_error"] += 1
            log.exception("sending %s failed", kind)
