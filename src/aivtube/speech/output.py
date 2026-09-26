"""Core-side ``SpeechOutput`` implementations (§3.5, Appendix A).

``BusSpeechOutput`` proxies to the voice worker over IPC. Each method sends its Appendix A
message; ``segment`` waits for the worker's ``reply`` (``ok`` → ``True``, ``busy`` → ``False``,
backpressure). The worker's messages come back as events on the bus: ``SegmentStarted`` (with
the caption and emotion of the segment the core queued), ``SegmentDone``, ``UtteranceDone``
(``filtered`` when the core stopped it for "Filtered."), ``UserSpeechStarted`` /
``UserSpeechEnded`` / ``UserTranscript``, ``BargeIn*``, ``HealthChanged("voice")`` and
``ProviderSwitched(kind="tts")``. Lip tracks go straight to ``lip_sink`` (the avatar driver's
``on_lip_track``), never onto the bus. Worker times are mapped onto the core clock with the
link's clock-offset correction.

Invariant I7: ``segment`` takes only ``Segment`` objects, and callers build them from the
output gate's result (``SafetyGate.check_output`` runs in the core before this). ``i7_check``
is an assertion hook for tests. ``constraints`` returns the worker's latest
``tts.constraints`` per character, or ``defaults`` until the first one arrives.

On (re)connect the core re-sends ``voice.configure``, the voice policy, mute, duck and talking
speeds, and ends every utterance that was open with ``reason="voice_restart"`` (keeping the
text heard so far); the same happens when the link is lost (§2.8). ``ready()`` is true once the
worker has reported ``health`` ok or degraded after the (re)connect.

``ConsoleSpeechOutput`` (text mode) is re-exported from ``aivtube.speech.console``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

from aivtube.contracts import ipc
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.events import (
    BargeInCandidate,
    BargeInConfirmed,
    BargeInRejected,
    Event,
    HealthChanged,
    ProviderSwitched,
    SegmentDone,
    SegmentStarted,
    UserSpeechEnded,
    UserSpeechStarted,
    UserTranscript,
    UtteranceDone,
)
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.speech import StopMode, TTSConstraints, VoicePolicy
from aivtube.contracts.types import Health, HealthState, Segment
from aivtube.infra.clock import DeadlineExceeded
from aivtube.ipc import IpcLinkError, IpcPeer, IpcServer
from aivtube.speech._common import (
    DEFAULT_CONSTRAINTS,
    SegmentCheck,
    TurnDirectory,
    UtteranceBook,
    UttRecord,
    check_segment,
)
from aivtube.speech.console import ConsoleSpeechOutput

__all__ = ["BusSpeechOutput", "ConsoleSpeechOutput"]

log = logging.getLogger("aivtube.speech.output")

_READY_STATES = frozenset({"ok", "degraded"})


class BusSpeechOutput:
    """``SpeechOutput`` backed by the voice worker (see the module docstring)."""

    def __init__(
        self,
        server: IpcServer,
        bus: EventBus,
        clock: Clock,
        *,
        lip_sink: Callable[[LipTrack], None],
        on_cut: Callable[[str, float], None],
        role: str = "voice",
        configure: Mapping[str, Any] | None = None,
        policy: VoicePolicy | None = None,
        defaults: TTSConstraints | Mapping[str, TTSConstraints] | None = None,
        request_timeout_s: float = 1.0,
        i7_check: SegmentCheck | None = None,
        component: str = "voice",
    ) -> None:
        self._server = server
        self._bus = bus
        self._clock = clock
        self._lip_sink = lip_sink
        self._on_cut = on_cut
        self.role = role
        self._configure: dict[str, Any] | None = dict(configure) if configure else None
        self._policy = policy or VoicePolicy()
        if isinstance(defaults, TTSConstraints) or defaults is None:
            self._default = defaults or DEFAULT_CONSTRAINTS
            self._defaults: dict[str, TTSConstraints] = {}
        else:
            self._default = DEFAULT_CONSTRAINTS
            self._defaults = dict(defaults)
        self._timeout = request_timeout_s
        self._i7 = i7_check
        self.component = component
        self._book = UtteranceBook()
        self._turns = TurnDirectory(bus)
        self._constraints: dict[str, TTSConstraints] = {}
        self._muted = False
        self._duck: tuple[float, int] = (1.0, 30)
        self._rates: dict[str, int] = {}
        self._worker_ready = False
        self._health: tuple[str, str] | None = None
        self.voice_stats: dict[str, Any] = {}
        self.stats: dict[str, int] = {}
        handlers: dict[str, Callable[[ipc.Envelope], None]] = {
            ipc.HEALTH: self._on_health,
            ipc.VAD_START: self._on_vad_start,
            ipc.VAD_END: self._on_vad_end,
            ipc.STT_FINAL: self._on_stt_final,
            ipc.STT_SPECULATIVE: self._on_ignored,
            ipc.BARGE_CANDIDATE: self._on_barge_candidate,
            ipc.BARGE_CONFIRMED: self._on_barge_confirmed,
            ipc.BARGE_REJECTED: self._on_barge_rejected,
            ipc.SEG_STARTED: self._on_seg_started,
            ipc.LIP_TRACK: self._on_lip_track,
            ipc.SEG_DONE: self._on_seg_done,
            ipc.UTT_DONE: self._on_utt_done,
            ipc.TTS_FALLBACK: self._on_tts_fallback,
            ipc.TTS_CONSTRAINTS: self._on_tts_constraints,
        }
        for mtype, handler in handlers.items():
            server.on(mtype, self._guard(handler))
        server.on_peer_ready(self._on_peer_ready)
        server.on_peer_lost(self._on_peer_lost)

    # --- SpeechOutput ---------------------------------------------------------------------
    def ready(self) -> bool:
        return self._peer() is not None and self._worker_ready

    def constraints(self, character: str) -> TTSConstraints:
        got = self._constraints.get(character) or self._defaults.get(character)
        return got or self._default

    async def begin(
        self,
        utt_id: str,
        character: str,
        *,
        filler_after_s: float | None = None,
        gate_open: bool = True,
    ) -> None:
        rec = self._book.begin(utt_id, character, self._clock.now())
        peer = self._peer()
        rec.link = peer.link if peer is not None else None
        data = {
            "utt": utt_id,
            "character": character,
            "filler_after_s": None if filler_after_s is None else max(0.0, filler_after_s),
            "gate_open": gate_open,
        }
        if not self._post(ipc.SPEAK_BEGIN, data):
            self._finish(rec, cancelled=True, reason="voice_down")

    async def segment(self, seg: Segment) -> bool:
        """Queue a gate-approved segment (I7); ``False`` = backpressure, retry later."""
        check_segment(seg, self._i7)
        rec = self._book.get(seg.utt_id)
        if rec is None or rec.done:
            self._count("dropped_segments")
            return True  # accepted and dropped, like the worker for a stopped utterance
        peer = self._peer()
        if peer is None:
            self._finish(rec, cancelled=True, reason="voice_down")
            return True
        rec.segments[seg.seq] = seg
        if seg.kind == "filtered":
            rec.filtered = True
        data = {
            "utt": seg.utt_id,
            "seq": seg.seq,
            "text": seg.text,
            "caption": seg.caption,
            "emotion": seg.emotion,
            "last": seg.last,
            "kind": seg.kind,
        }
        try:
            reply = await peer.request(ipc.SPEAK_SEGMENT, data, timeout=self._timeout)
        except DeadlineExceeded:
            self._count("segment_timeout")
            log.warning("speak.segment %s/%s: no reply in time", seg.utt_id, seg.seq)
            return False
        except IpcLinkError:
            return True  # the link dropped: the utterance ends as voice_restart
        except ipc.IpcError as exc:
            log.error("speak.segment %s/%s not sent: %s", seg.utt_id, seg.seq, exc)
            return True
        status = reply.data.get("status")
        if status == "ok":
            return True
        if status == "busy":
            self._count("busy")
            return False
        self._count("segment_error")
        log.warning(
            "voice refused segment %s/%s: %s", seg.utt_id, seg.seq, reply.data.get("detail", "")
        )
        return True

    async def open_gate(self, utt_id: str) -> None:
        self._post(ipc.SPEAK_GATE, {"utt": utt_id})

    async def stop(
        self, utt_id: str | None, mode: StopMode, reason: str, fade_ms: int = 30
    ) -> None:
        if utt_id is None:
            targets = self._book.open_records()
        else:
            rec = self._book.get(utt_id)
            targets = [rec] if rec is not None and not rec.done else []
        for rec in targets:
            rec.stopped = True
            if reason == "filtered":
                rec.filtered = True
        data = {"utt": utt_id, "mode": mode, "reason": reason, "fade_ms": max(0, int(fade_ms))}
        sent = self._post(ipc.SPEAK_STOP, data)
        if mode == "now":
            now = self._clock.now()
            for rec in targets:
                self._cut(rec.utt_id, now)
        if not sent:
            for rec in targets:
                self._finish(rec, cancelled=True, reason=reason)

    async def duck(self, gain: float, ramp_ms: int = 30) -> None:
        self._duck = (min(1.0, max(0.0, float(gain))), max(0, int(ramp_ms)))
        self._post(ipc.SPEAK_DUCK, {"gain": self._duck[0], "ramp_ms": self._duck[1]})

    async def mute(self, on: bool) -> None:
        self._muted = bool(on)
        self._post(ipc.VOICE_MUTE, {"on": self._muted})

    async def set_policy(self, policy: VoicePolicy) -> None:
        self._policy = policy
        self._post(ipc.VOICE_POLICY, dataclasses.asdict(policy))

    async def set_voice_rate(self, character: str, percent: int) -> None:
        self._rates[character] = int(percent)
        self._post(ipc.VOICE_RATE, {"character": character, "percent": int(percent)})

    async def play_canned(self, key: str, character: str) -> None:
        if key == "filtered":
            self._book.mark_filtered(character)
        self._post(ipc.SPEAK_CANNED, {"key": key, "character": character})

    # --- extras ---------------------------------------------------------------------------
    def bind_turn(self, utt_id: str, turn_id: str | None) -> None:
        """Say which turn an utterance belongs to (``UtteranceStarted`` does this too)."""
        self._turns.bind(utt_id, turn_id)

    def set_configure(self, configure: Mapping[str, Any]) -> None:
        """Replace the ``voice.configure`` payload; sent now and after every reconnect."""
        self._configure = dict(configure)
        self._post(ipc.VOICE_CONFIGURE, self._configure)

    @property
    def policy(self) -> VoicePolicy:
        return self._policy

    @property
    def muted(self) -> bool:
        return self._muted

    async def aclose(self) -> None:
        self._turns.close()

    # --- connection events -----------------------------------------------------------------
    def _peer(self) -> IpcPeer | None:
        return self._server.peer(self.role)

    def _post(self, mtype: str, data: Mapping[str, Any]) -> bool:
        peer = self._peer()
        if peer is None:
            self._count("dropped_down")
            return False
        return peer.post(mtype, data)

    def _on_peer_ready(self, peer: IpcPeer) -> None:
        if peer.role != self.role:
            return
        self._worker_ready = False
        self._health = None
        for rec in self._book.open_records():
            if rec.link is not peer.link:  # opened on a previous connection: already dropped
                self._cut(rec.utt_id, self._clock.now())
                self._finish(rec, cancelled=True, reason="voice_restart")
        if self._configure is not None:
            peer.post(ipc.VOICE_CONFIGURE, self._configure)
        peer.post(ipc.VOICE_POLICY, dataclasses.asdict(self._policy))
        if self._muted:
            peer.post(ipc.VOICE_MUTE, {"on": True})
        if self._duck[0] != 1.0:
            peer.post(ipc.SPEAK_DUCK, {"gain": self._duck[0], "ramp_ms": self._duck[1]})
        for character, percent in self._rates.items():
            peer.post(ipc.VOICE_RATE, {"character": character, "percent": percent})

    def _on_peer_lost(self, role: str, reason: str) -> None:
        if role != self.role:
            return
        self._worker_ready = False
        self._health = None
        self._publish(
            HealthChanged(
                health=Health(
                    self.component, HealthState.DOWN, f"link lost: {reason}", self._clock.now()
                )
            )
        )
        now = self._clock.now()
        for rec in self._book.open_records():
            self._cut(rec.utt_id, now)
            self._finish(rec, cancelled=True, reason="voice_restart")

    # --- voice -> core ----------------------------------------------------------------------
    def _guard(self, handler: Callable[[ipc.Envelope], None]) -> Callable[[ipc.Envelope], None]:
        def run(env: ipc.Envelope) -> None:
            try:
                handler(env)
            except Exception:
                self._count("handler_error")
                log.exception("voice message %s could not be handled", env.type)

        return run

    def _local(self, t: float) -> float:
        peer = self._peer()
        return peer.to_local(t) if peer is not None else t

    def _on_health(self, env: ipc.Envelope) -> None:
        d = env.data
        state, detail = str(d["state"]), str(d.get("detail", ""))
        self.voice_stats = dict(d.get("stats") or {})
        self._worker_ready = state in _READY_STATES
        if self._health != (state, detail):
            self._health = (state, detail)
            health = Health(self.component, HealthState(state), detail, self._local(env.ts))
            self._publish(HealthChanged(health=health))

    def _on_vad_start(self, env: ipc.Envelope) -> None:
        t = self._local(float(env.data["t"]))
        self._publish(UserSpeechStarted(ts=t, barge=bool(env.data["barge"])))

    def _on_vad_end(self, env: ipc.Envelope) -> None:
        t = self._local(float(env.data["t"]))
        self._publish(UserSpeechEnded(ts=t, audio_s=float(env.data["audio_s"])))

    def _on_stt_final(self, env: ipc.Envelope) -> None:
        d = env.data
        self._publish(
            UserTranscript(
                ts=self._local(env.ts),
                text=str(d["text"]),
                engine=str(d["engine"]),
                latency_ms=float(d["latency_ms"]),
                audio_s=float(d["audio_s"]),
                parts=int(d["parts"]),
            )
        )

    def _on_barge_candidate(self, env: ipc.Envelope) -> None:
        self._publish(BargeInCandidate(ts=self._local(float(env.data["t"]))))

    def _on_barge_confirmed(self, env: ipc.Envelope) -> None:
        t = self._local(float(env.data["t"]))
        cut = bool(env.data["cut_local"])
        self._publish(BargeInConfirmed(ts=t, text=str(env.data["text"]), cut_local=cut))
        if cut:  # the worker already cut the audio: close the mouth now
            for rec in self._book.open_records():
                if rec.segments:
                    self._cut(rec.utt_id, t)

    def _on_barge_rejected(self, env: ipc.Envelope) -> None:
        self._publish(BargeInRejected(ts=self._local(float(env.data["t"]))))

    def _on_seg_started(self, env: ipc.Envelope) -> None:
        d = env.data
        utt, seq = str(d["utt"]), int(d["seq"])
        rec = self._book.get(utt)
        seg = rec.segments.get(seq) if rec is not None else None
        t = self._local(float(d["t_audible"]))
        duration = d.get("duration_s")
        self._publish(
            SegmentStarted(
                ts=t,
                character=rec.character if rec is not None else None,
                turn_id=self._turn(rec, utt),
                utt_id=utt,
                seq=seq,
                t_audible=t,
                duration_s=None if duration is None else float(duration),
                backend=str(d["backend"]),
                silent=bool(d["silent"]),
                caption=seg.caption if seg is not None else "",
                emotion=seg.emotion if seg is not None else None,
            )
        )

    def _on_lip_track(self, env: ipc.Envelope) -> None:
        d = env.data
        track = LipTrack(
            utt_id=str(d["utt"]),
            seq=int(d["seq"]),
            t0=self._local(float(d["t0"])),
            fps=int(d["fps"]),
            mouth=tuple(float(v) for v in d["mouth"]),
            form=tuple(float(v) for v in d["form"]),
            final=bool(d["final"]),
        )
        try:
            self._lip_sink(track)
        except Exception:
            self._count("lip_sink_error")
            log.exception("lip sink failed")

    def _on_seg_done(self, env: ipc.Envelope) -> None:
        d = env.data
        utt, seq = str(d["utt"]), int(d["seq"])
        rec = self._book.get(utt)
        if rec is not None:
            rec.heard[seq] = str(d["heard_text"])
        self._publish(
            SegmentDone(
                ts=self._local(env.ts),
                character=rec.character if rec is not None else None,
                turn_id=self._turn(rec, utt),
                utt_id=utt,
                seq=seq,
                heard=bool(d["heard"]),
                heard_text=str(d["heard_text"]),
            )
        )

    def _on_utt_done(self, env: ipc.Envelope) -> None:
        d = env.data
        utt = str(d["utt"])
        rec = self._book.get(utt)
        if rec is None or rec.done:
            self._count("late_utterance_done")
            log.debug("utterance_done for %s ignored (unknown or already ended)", utt)
            return
        reason = d.get("reason")
        self._finish(
            rec,
            cancelled=bool(d["cancelled"]),
            reason=None if reason is None else str(reason),
            heard_text=str(d["heard_text"]),
            ts=self._local(env.ts),
        )

    def _on_tts_fallback(self, env: ipc.Envelope) -> None:
        d = env.data
        new = d.get("to")
        self._publish(
            ProviderSwitched(
                ts=self._local(env.ts),
                kind="tts",
                old=str(d["from"]),
                new="captions" if new is None else str(new),
                reason=str(d["reason"]),
            )
        )

    def _on_tts_constraints(self, env: ipc.Envelope) -> None:
        d = env.data
        self._constraints[str(d["character"])] = TTSConstraints(
            first_min_chars=max(1, int(d["first_min_chars"])),
            min_chars=max(1, int(d["min_chars"])),
            max_chars=int(d["max_chars"]),
            backend=str(d["backend"]),
            identity=str(d["identity"]),
        )

    def _on_ignored(self, env: ipc.Envelope) -> None:
        self._count(f"ignored:{env.type}")

    # --- helpers ----------------------------------------------------------------------------
    def _turn(self, rec: UttRecord | None, utt: str) -> str | None:
        if rec is not None and rec.turn_id is not None:
            return rec.turn_id
        turn = self._turns.lookup(utt)
        if rec is not None and turn is not None:
            rec.turn_id = turn
        return turn

    def _finish(
        self,
        rec: UttRecord,
        *,
        cancelled: bool,
        reason: str | None,
        heard_text: str | None = None,
        ts: float = 0.0,
    ) -> None:
        if not self._book.close(rec):
            return
        self._publish(
            UtteranceDone(
                ts=ts,
                character=rec.character,
                turn_id=self._turn(rec, rec.utt_id),
                utt_id=rec.utt_id,
                heard_text=rec.heard_text() if heard_text is None else heard_text,
                cancelled=cancelled,
                reason=reason if cancelled else None,
                filtered=rec.filtered,
            )
        )

    def _cut(self, utt_id: str, t: float) -> None:
        try:
            self._on_cut(utt_id, t)
        except Exception:
            log.exception("on_cut failed")

    def _publish(self, event: Event) -> None:
        try:
            self._bus.publish(event)
        except Exception:
            log.exception("publishing %s failed", type(event).__name__)

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1
