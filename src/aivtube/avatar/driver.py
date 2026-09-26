"""``LiveAvatarDriver``: the 60 Hz avatar loop (§2.5, §3.7, §5 "mouth vs audio ±40 ms").

Every tick (paced by ``infra.PrecisionTicker``, a thread that posts to the loop) the driver
samples the lip tracks at ``perf_counter() + lead`` (the tracks' ``t0`` is the audible time of
frame 0; the 40 ms lead covers render and capture latency and is applied here, not in the voice
worker), smooths the mouth, blends the emotion baseline (faded over ``emotion_fade_s``), adds
idle motion and the state pose, and hands one frame to ``AvatarSink.set_params``. Unchanged
frames are skipped, but parameters are re-sent at least every ``keepalive_s`` (VTS reverts an
input it has not heard for 1 s). ``on_cut`` closes the mouth at once with a fast release.

If the ticker's p95 wake-up jitter exceeds ``jitter_limit_ms`` (8 ms) the driver drops to
``jitter_fallback_fps`` (30 Hz) for the rest of the run.
"""

from __future__ import annotations

import asyncio
import bisect
import collections
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from aivtube.avatar.emotion import DEFAULT_BASELINE, EmotionController
from aivtube.avatar.idle_motion import HEAD_PARAMS, IdleMotion
from aivtube.contracts.avatar import AvatarSink, AvatarState, LipTrack
from aivtube.contracts.infra import Clock
from aivtube.infra.clock import deadline
from aivtube.infra.ticker import PrecisionTicker

__all__ = ["POSES", "LiveAvatarDriver", "TickerLike"]

log = logging.getLogger("aivtube.avatar.driver")


class TickerLike(Protocol):
    """What the driver needs from ``PrecisionTicker``."""

    def start(self) -> None: ...

    def stop(self, timeout: float = 1.0) -> None: ...

    def set_hz(self, hz: float) -> None: ...

    def jitter_stats(self) -> Mapping[str, float]: ...


#: Offsets added on top of idle motion per state (VTS input units).
POSES: Mapping[str, Mapping[str, float]] = {
    "idle": {},
    "listening": {"FaceAngleZ": 4.0, "FaceAngleY": 2.0, "Brows": 0.08},
    "thinking": {
        "FaceAngleX": -5.0,
        "FaceAngleY": 6.0,
        "FaceAngleZ": -3.0,
        "EyeLeftX": -0.3,
        "EyeRightX": -0.3,
        "EyeLeftY": 0.45,
        "EyeRightY": 0.45,
        "Brows": -0.05,
    },
    "speaking": {},
    "paused": {},
}
#: Idle-motion amplitude per state (speaking is suppressed inside ``IdleMotion``).
MOTION_SCALE: Mapping[str, float] = {
    "idle": 1.0,
    "listening": 0.6,
    "thinking": 0.5,
    "speaking": 1.0,
    "paused": 0.5,
}
_POSE_KEYS = tuple(sorted({k for pose in POSES.values() for k in pose}))
_MAX_TRACKS = 256
_MAX_CUTS = 64
_EPS = 1e-4


def _alpha(dt: float, tau: float) -> float:
    """One-pole smoothing coefficient for a step of ``dt`` seconds."""
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def _unit(v: float) -> float:
    return min(1.0, max(0.0, v))


def _smoothstep(p: float) -> float:
    p = _unit(p)
    return p * p * (3.0 - 2.0 * p)


class LiveAvatarDriver:
    """The real ``AvatarDriver``. Call its methods from the event-loop thread."""

    def __init__(
        self,
        sink: AvatarSink,
        clock: Clock,
        *,
        fps: int = 60,
        lead_ms: float = 40.0,
        jitter_fallback_fps: int = 30,
        ticker_factory: Callable[..., TickerLike] = PrecisionTicker,
        jitter_limit_ms: float = 8.0,
        emotion_map: Mapping[str, Mapping[str, Any]] | None = None,
        emotion_fade_s: float = 0.3,
        emotion_hold_s: float = 4.0,
        idle_motion: bool | IdleMotion = True,
        seed: int = 0,
        keepalive_s: float = 0.5,
        attack_ms: float = 15.0,
        release_ms: float = 60.0,
        cut_release_ms: float = 25.0,
        pose_tau_s: float = 0.3,
        housekeeping_s: float = 0.5,
        jitter_check_s: float = 2.0,
        jitter_min_ticks: int = 120,
        emotion_timeout_s: float = 3.0,
    ) -> None:
        if fps <= 0 or jitter_fallback_fps <= 0:
            raise ValueError("fps must be > 0")
        self._sink = sink
        self._clock = clock
        self._base_fps = fps
        self._fps = fps
        self._fallback_fps = min(jitter_fallback_fps, fps)
        self.lead_s = lead_ms / 1000.0
        self._ticker_factory = ticker_factory
        self._jitter_limit_ms = jitter_limit_ms
        self._emotions = EmotionController(emotion_map or {})
        self.emotion_fade_s = emotion_fade_s
        self._emotion_hold_s = emotion_hold_s
        if isinstance(idle_motion, IdleMotion):
            self._idle: IdleMotion | None = idle_motion
        else:
            self._idle = IdleMotion(seed) if idle_motion else None
        self._keepalive_s = keepalive_s
        self._attack_s = attack_ms / 1000.0
        self._release_s = release_ms / 1000.0
        self._cut_release_s = cut_release_ms / 1000.0
        self._pose_tau_s = pose_tau_s
        self._housekeeping_s = max(0.05, housekeeping_s)
        self._jitter_check_s = jitter_check_s
        self._jitter_min_ticks = jitter_min_ticks
        self._emotion_timeout_s = emotion_timeout_s
        # lip tracks
        self._tracks: list[LipTrack] = []
        self._cuts: collections.OrderedDict[str, float] = collections.OrderedDict()
        self._fast_release_until = -math.inf
        # smoothed state
        self._t_origin = clock.now()
        self._last_tick: float | None = None
        self._mouth = 0.0
        self._form = 0.5
        self._speaking_w = 0.0
        self._motion = 1.0
        self._pose: dict[str, float] = dict.fromkeys(_POSE_KEYS, 0.0)
        self._state: AvatarState = "idle"
        self._last_speaking = -math.inf
        # emotion
        self._emotion = self._emotions.default
        neutral = self._emotions.baseline()
        self._emo_from: dict[str, float] = dict(neutral)
        self._emo_to: dict[str, float] = dict(neutral)
        self._emo_t0 = -math.inf
        self._pending_emotion: str | None = None
        self._emotion_event = asyncio.Event()
        # output
        self._last_sent: dict[str, float] | None = None
        self._last_sent_t = -math.inf
        self.frames_sent = 0
        self.frames_skipped = 0
        self._ticker: TickerLike | None = None
        self._running = False
        self._last_jitter_check = -math.inf
        self.jitter_fallback_active = False

    # --- AvatarDriver -------------------------------------------------------------------------
    @property
    def fps(self) -> int:
        return self._fps

    @property
    def state(self) -> AvatarState:
        return self._state

    @property
    def emotion(self) -> str:
        return self._emotion

    @property
    def mouth(self) -> float:
        """The smoothed mouth opening of the last frame."""
        return self._mouth

    def on_lip_track(self, track: LipTrack) -> None:
        if self._state == "paused" or track.fps <= 0 or not track.mouth:
            return
        cut = self._cuts.get(track.utt_id)
        if cut is not None and track.t0 >= cut:
            return  # decoded after the cut: never shown
        bisect.insort(self._tracks, track, key=lambda tr: tr.t0)
        if len(self._tracks) > _MAX_TRACKS:
            del self._tracks[: len(self._tracks) - _MAX_TRACKS]

    def on_cut(self, utt_id: str, t: float) -> None:
        self._cuts[utt_id] = t
        self._cuts.move_to_end(utt_id)
        while len(self._cuts) > _MAX_CUTS:
            self._cuts.popitem(last=False)
        self._tracks = [tr for tr in self._tracks if tr.utt_id != utt_id or tr.t0 < t]
        self._fast_release_until = max(self._clock.now(), t) + 0.3

    def set_state(self, state: AvatarState) -> None:
        if state == self._state:
            return
        self._state = state
        if state == "paused":  # FREEZE: mouth shut, neutral face (§2.11)
            self._tracks.clear()
            self._fast_release_until = self._clock.now() + 0.3
            self.set_emotion(None)

    def set_emotion(self, emotion: str | None) -> None:
        """Fade to ``emotion`` (``None`` = neutral) and switch the sink's expressions.

        Unknown emotions become neutral; without an ``emotion_map`` the name is passed through
        to the sink unchanged (with the neutral baseline).
        """
        if len(self._emotions.known) > 1 or not emotion:
            name = self._emotions.resolve(emotion)
        else:
            name = emotion.strip().casefold()
        if name == self._emotion:
            return
        now = self._clock.now()
        self._emo_from = self._baseline(now)
        self._emo_to = self._emotions.baseline(name)
        self._emo_t0 = now
        self._emotion = name
        self._pending_emotion = name
        self._emotion_event.set()

    async def run(self) -> None:
        """Start the ticker and run until cancelled."""
        if self._running:
            raise RuntimeError("LiveAvatarDriver.run() is already running")
        loop = asyncio.get_running_loop()
        self._fps = self._base_fps
        self.jitter_fallback_active = False
        self._last_jitter_check = self._clock.now()
        ticker = self._ticker_factory(self._fps, self._on_tick, loop, name="avatar-driver")
        self._ticker = ticker
        self._running = True
        if self._pending_emotion is not None:
            self._emotion_event.set()  # re-sync the sink with the current emotion
        try:
            ticker.start()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._push_emotions(), name="avatar-emotion")
                tg.create_task(self._housekeeping(), name="avatar-housekeeping")
        finally:
            self._running = False
            self._ticker = None
            ticker.stop(0.0)  # the daemon thread exits on its next wake-up

    # --- frame computation --------------------------------------------------------------------
    def sample(self, t: float) -> tuple[float, float]:
        """Raw (mouth, form) of the lip tracks at perf_counter time ``t`` (no lead, no smoothing)."""
        for tr in reversed(self._tracks):
            if tr.t0 > t:
                continue
            k = int((t - tr.t0) * tr.fps)
            if k >= len(tr.mouth):
                continue
            cut = self._cuts.get(tr.utt_id)
            if cut is not None and t >= cut:
                return 0.0, 0.5
            form = tr.form[k] if k < len(tr.form) else 0.5
            return _unit(tr.mouth[k]), _unit(form)
        return 0.0, 0.5

    def tick(self, now: float | None = None) -> dict[str, float]:
        """Compute one frame at ``now`` and send it (unless unchanged); returns the frame."""
        now = self._clock.now() if now is None else now
        last = self._last_tick
        dt = 1.0 / self._fps if last is None else min(0.25, max(0.0, now - last))
        self._last_tick = now
        raw_mouth, raw_form = self.sample(now + self.lead_s)
        if self._state == "paused":
            raw_mouth, raw_form = 0.0, 0.5
        if raw_mouth > self._mouth:
            tau = self._attack_s
        elif now < self._fast_release_until:
            tau = self._cut_release_s
        else:
            tau = self._release_s
        self._mouth += _alpha(dt, tau) * (raw_mouth - self._mouth)
        if self._mouth < 1e-3:
            self._mouth = 0.0
        self._form += _alpha(dt, self._attack_s) * (raw_form - self._form)
        speaking = self._state == "speaking" or raw_mouth > 0.02
        if speaking:
            self._last_speaking = now
        self._speaking_w += _alpha(dt, 0.4) * ((1.0 if speaking else 0.0) - self._speaking_w)
        mouth = self._mouth
        base = self._baseline(now)
        values: dict[str, float] = {k: v for k, v in base.items() if k not in DEFAULT_BASELINE}
        values["MouthOpen"] = mouth
        smile = base["MouthSmile"] + 0.6 * (self._form - 0.5) * min(1.0, 3.0 * mouth)
        values["MouthSmile"] = _unit(smile)
        if self._idle is not None:
            pose = POSES.get(self._state, {})
            a = _alpha(dt, self._pose_tau_s)
            for key in _POSE_KEYS:
                self._pose[key] += a * (pose.get(key, 0.0) - self._pose[key])
            self._motion += a * (MOTION_SCALE.get(self._state, 1.0) - self._motion)
            idle = self._idle.sample(now - self._t_origin, mouth, speaking=self._speaking_w)
            for key in HEAD_PARAMS:
                values[key] = idle.get(key, 0.0) * self._motion + self._pose.get(key, 0.0)
            values["EyeOpenLeft"] = 1.0
            values["EyeOpenRight"] = 1.0
            values["Brows"] = _unit(base["Brows"] + self._pose.get("Brows", 0.0))
        else:
            values["Brows"] = _unit(base["Brows"])
        frame = {k: round(v, 4) for k, v in values.items()}
        self._prune(now)
        if self._should_send(frame, now):
            self._sink.set_params(frame)
            self._last_sent = frame
            self._last_sent_t = now
            self.frames_sent += 1
        else:
            self.frames_skipped += 1
        return frame

    def _baseline(self, now: float) -> dict[str, float]:
        fade = self.emotion_fade_s
        if fade <= 0.0 or now >= self._emo_t0 + fade:
            return dict(self._emo_to)
        p = _smoothstep((now - self._emo_t0) / fade)
        out: dict[str, float] = {}
        for key in self._emo_from.keys() | self._emo_to.keys():
            default = DEFAULT_BASELINE.get(key, 0.0)
            a = self._emo_from.get(key, default)
            b = self._emo_to.get(key, default)
            out[key] = a + (b - a) * p
        return out

    def _should_send(self, frame: Mapping[str, float], now: float) -> bool:
        last = self._last_sent
        if last is None or now - self._last_sent_t >= self._keepalive_s:
            return True
        if last.keys() != frame.keys():
            return True
        return any(abs(frame[k] - last[k]) > _EPS for k in frame)

    def _prune(self, now: float) -> None:
        horizon = now - 0.5

        def ended(tr: LipTrack) -> bool:
            return tr.t0 + len(tr.mouth) / tr.fps < horizon

        if self._tracks and ended(self._tracks[0]):
            self._tracks = [tr for tr in self._tracks if not ended(tr)]

    def _on_tick(self, t: float) -> None:
        if self._running:
            self.tick()

    # --- background ---------------------------------------------------------------------------
    async def _push_emotions(self) -> None:
        while True:
            await self._emotion_event.wait()
            self._emotion_event.clear()
            emotion = self._pending_emotion
            if emotion is None:
                continue
            try:
                async with deadline(
                    self._emotion_timeout_s, what="avatar set_emotion", clock=self._clock
                ):
                    await self._sink.set_emotion(emotion, self.emotion_fade_s)
            except Exception as exc:
                log.info("avatar set_emotion(%s) failed: %r", emotion, exc)

    async def _housekeeping(self) -> None:
        while True:
            await self._clock.sleep(self._housekeeping_s)
            now = self._clock.now()
            self._check_jitter(now)
            if (
                self._emotion != self._emotions.default
                and self._state != "speaking"
                and now - max(self._last_speaking, self._emo_t0) >= self._emotion_hold_s
            ):
                self.set_emotion(None)

    def _check_jitter(self, now: float) -> None:
        ticker = self._ticker
        if ticker is None or self.jitter_fallback_active or self._fps <= self._fallback_fps:
            return
        if now - self._last_jitter_check < self._jitter_check_s:
            return
        self._last_jitter_check = now
        stats = ticker.jitter_stats()
        if stats.get("ticks", 0.0) < self._jitter_min_ticks:
            return
        p95 = float(stats.get("p95_ms", 0.0))
        if p95 > self._jitter_limit_ms:
            log.warning(
                "avatar ticker p95 jitter %.1f ms > %.1f ms; dropping to %d Hz",
                p95,
                self._jitter_limit_ms,
                self._fallback_fps,
            )
            ticker.set_hz(self._fallback_fps)
            self._fps = self._fallback_fps
            self.jitter_fallback_active = True
