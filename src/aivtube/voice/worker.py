"""The voice worker process: audio, echo handling, VAD, STT, barge-in, TTS and playback (§2.1).

``python -m aivtube _voice`` runs :func:`main`. The worker raises its own priority on Windows
(ABOVE_NORMAL) and sets ``sys.setswitchinterval(0.001)`` so the audio threads get the GIL
quickly (§2.3, §2.4), loads the config, connects to the core's IPC bus with the launcher's
token and waits for ``voice.configure``. From that payload it builds (blocking steps in
worker threads):

``StreamingPlayer`` (always open) · ``MicCapture`` → ``VoiceFrontEnd`` ([AEC] → soxr → VAD →
``SileroEndpointer``) · ``SttRunner`` + ``NamePostProcessor`` + ``UtteranceAssembler`` ·
``TTSRouter`` + phrase cache · ``SpeechQueue`` · ``LipSyncAnalyzer`` · ``BargeInController``.

It then warms STT and TTS up, pre-synthesises every character's cached phrases (checking that
"Filtered." is there), reports ``tts.constraints`` and ``health{state: ok}`` (READY). A second
``voice.configure`` with the same payload (a restarted core) only re-reports READY.

Invariants:

- **I1.** Only segments received over the authenticated link are spoken. When the link drops
  the worker finishes the segment that is playing (it was already filtered), clears its queue,
  stops listening and waits; listening resumes after the next accepted hello.
- **I7** is the core's: segments arrive already gated.
- The barge-in controller owns the player gain; MUTE and duck go through it.

``--fake-audio`` drives a ``FakeSD`` from a real-time thread (``FakeAudioDevice``: speakers
pumped every 10 ms, microphone fed from ``inject``); ``--fake-models`` swaps the STT chain and
every TTS backend for fakes (the VAD stays Silero when its model file exists). Neither needs
PortAudio, a GPU or the network.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from aivtube.contracts import ipc
from aivtube.contracts.infra import Clock
from aivtube.contracts.speech import EchoMode, VoicePolicy
from aivtube.contracts.types import HealthState, Segment
from aivtube.contracts.voice import (
    F32,
    AudioBackend,
    EndpointerConfig,
    SpeechRecognizer,
    TTSBackend,
    VadEnd,
    VadEvent,
    VadPartial,
    VadStart,
)
from aivtube.infra.bus import AsyncEventBus
from aivtube.infra.clock import SystemClock, deadline
from aivtube.infra.tasks import SupervisedTasks
from aivtube.ipc import IpcClient

__all__ = [
    "DEFAULT_FAKE_STT_SCRIPT",
    "EXIT_CONFIG",
    "EXIT_CRITICAL",
    "FakeAudioDevice",
    "VoiceWorker",
    "build_parser",
    "main",
]

log = logging.getLogger("aivtube.voice.worker")

EXIT_CONFIG: Final = 2
EXIT_CRITICAL: Final = 70
BUS_TOKEN_ENV: Final = "AIVTUBE_BUS_TOKEN"
FAKE_AUDIO_ENV: Final = "AIVTUBE_FAKE_AUDIO"
VAD_RATE: Final = 16000
ABOVE_NORMAL_PRIORITY_CLASS: Final = 0x00008000

DEFAULT_FAKE_STT_SCRIPT: Final[Mapping[str, str]] = {
    "440": "สวัสดีครับ ไพลิน",
    "550": "วันนี้เล่นเกมอะไรดี",
    "660": "หยุดก่อนนะ ฟังผมหน่อย",
    "770": "ขอบคุณมากครับ",
}
_FAKE_TTS: Final[Mapping[str, Any]] = {"kind": "fake", "ttfa_s": 0.05}
_READY_STATES: Final = frozenset({"ok", "degraded"})


# --- process setup ---------------------------------------------------------------------------


def raise_priority() -> bool:
    """ABOVE_NORMAL priority on Windows (§2.1); ``True`` if it was applied."""
    if sys.platform == "win32":
        import ctypes

        try:
            kernel32 = ctypes.windll.kernel32
            ok = kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS
            )
        except (AttributeError, OSError) as exc:
            log.warning("cannot raise the process priority: %s", exc)
            return False
        return bool(ok)
    return False


class _FnClock:
    """A ``Clock`` around a ``perf_counter``-like function (``sleep`` is ``asyncio.sleep``)."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[], float]) -> None:
        self._fn = fn

    def now(self) -> float:
        return self._fn()

    def wall(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


def _as_clock(clock: Callable[[], float] | Clock) -> Clock:
    if isinstance(clock, Clock):
        return clock
    if clock is time.perf_counter:
        return SystemClock()
    return _FnClock(clock)


# --- fake sound card -------------------------------------------------------------------------


class FakeAudioDevice:
    """Runs a ``FakeSD`` like a sound card, in real time, on its own thread.

    Every ``block_s`` it pumps one block through each active output stream (recording the
    first one's samples with their ``perf_counter`` time) and pushes one block of microphone
    audio into each active input stream: queued ``inject``-ed audio, else silence. Audio is
    never fed faster than real time: the endpointer measures turns in capture time.
    """

    def __init__(
        self,
        sd: Any,
        *,
        block_s: float = 0.01,
        mic_rate: int = 48000,
        keep_output_s: float = 20.0,
    ) -> None:
        self.sd = sd
        self.block_s = block_s
        self.mic_rate = mic_rate
        self._lock = threading.Lock()
        self._mic: collections.deque[F32] = collections.deque()
        self._mic_pos = 0
        self.output: collections.deque[tuple[float, F32]] = collections.deque(
            maxlen=max(1, round(keep_output_s / block_s))
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="fake-audio", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def inject(self, pcm: F32, sample_rate: int = 48000) -> None:
        """Queue microphone audio (mono float32), resampled to the device rate."""
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sample_rate != self.mic_rate:
            x = _resample(x, sample_rate, self.mic_rate)
        with self._lock:
            self._mic.append(x)

    @property
    def mic_pending_s(self) -> float:
        with self._lock:
            n = sum(a.size for a in self._mic) - self._mic_pos
        return max(0, n) / self.mic_rate

    def output_since(self, t: float) -> F32:
        """Speaker samples pumped at or after ``perf_counter`` time ``t``."""
        blocks = [b for ts, b in list(self.output) if ts >= t]
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def output_between(self, t0: float, t1: float) -> F32:
        blocks = [b for ts, b in list(self.output) if t0 <= ts < t1]
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def _mic_block(self, n: int) -> F32:
        out = np.zeros(n, np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._mic:
                head = self._mic[0]
                take = min(n - filled, head.size - self._mic_pos)
                out[filled : filled + take] = head[self._mic_pos : self._mic_pos + take]
                filled += take
                self._mic_pos += take
                if self._mic_pos >= head.size:
                    self._mic.popleft()
                    self._mic_pos = 0
        return out

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                self.errors += 1
                if self.errors <= 3:
                    log.exception("fake audio device tick failed")
            next_t += self.block_s
            now = time.perf_counter()
            if next_t < now - 0.1:  # fell far behind (loaded CI runner): do not burst
                next_t = now
            time.sleep(max(0.0, next_t - now))

    def _tick(self) -> None:
        now = time.perf_counter()
        recorded = False
        for stream in list(self.sd.output_streams):
            if stream.active:
                block = stream.pump(1)
                if not recorded:
                    self.output.append((now, block))
                    recorded = True
        inputs = [s for s in list(self.sd.input_streams) if s.active]
        if inputs:
            block = self._mic_block(int(inputs[0].blocksize))
            for stream in inputs:
                stream.push(block)


def _resample(x: F32, src: int, dst: int) -> F32:
    try:
        import soxr

        return np.asarray(soxr.resample(x, src, dst), dtype=np.float32)
    except ImportError:
        from aivtube.testing.fakes import resample_linear

        return resample_linear(x, src, dst)


# --- the pipeline ----------------------------------------------------------------------------


@dataclass(eq=False)
class _Pipeline:
    """Everything one ``voice.configure`` built; closed as a unit."""

    payload: Mapping[str, Any]
    player: Any  # StreamingPlayer
    mic: Any  # MicCapture | None
    frontend: Any  # VoiceFrontEnd
    endpointer: Any  # SileroEndpointer
    base_ep_cfg: EndpointerConfig
    stt: Any  # SttRunner
    router: Any  # TTSRouter
    queue: Any  # SpeechQueue
    barge: Any  # BargeInController
    audio: Mapping[str, Any]
    characters: dict[str, Mapping[str, Any]]
    echo_mode: EchoMode = "none"  # in effect
    echo_request: EchoMode | None = None  # the policy mode it was built for
    notes: list[str] = field(default_factory=list)
    stt_q: asyncio.Queue[tuple[str, F32, float]] = field(default_factory=asyncio.Queue)
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    assembler: Any = None  # UtteranceAssembler
    turn_audio_s: float = 0.0
    ready: bool = False
    filtered_cached: bool = False

    def clear_stt(self) -> None:
        while not self.stt_q.empty():
            self.stt_q.get_nowait()
        if self.assembler is not None:
            self.assembler.reset()
        self.turn_audio_s = 0.0


class VoiceWorker:
    """The voice worker (see the module docstring). Run it with ``await worker.run()``."""

    def __init__(
        self,
        bus_url: str,
        token: str,
        *,
        backend: AudioBackend | None = None,
        clock: Callable[[], float] | Clock = time.perf_counter,
        root: Path | None = None,
        secrets: Callable[[str], str | None] | None = None,
        cloud_stt_consent: bool = False,
        stt_timeout_s: float = 3.0,
        fake_audio: bool = False,
        fake_models: bool = False,
        stt_chain: Sequence[SpeechRecognizer] | None = None,
        tts_backends: Mapping[str, TTSBackend] | None = None,
        heartbeat_s: float = 1.0,
        misses: int = 3,
        health_interval_s: float = 5.0,
        policy: VoicePolicy | None = None,
    ) -> None:
        self.bus_url = bus_url
        self._token = token
        self._clock = _as_clock(clock)
        self._now: Callable[[], float] = self._clock.now
        self.root = Path(root) if root is not None else Path.cwd()
        self._secrets: Callable[[str], str | None] = secrets or (lambda _name: None)
        self.cloud_stt_consent = cloud_stt_consent
        self.stt_timeout_s = stt_timeout_s
        self.fake_audio = fake_audio
        self.fake_models = fake_models
        self._stt_override = list(stt_chain) if stt_chain is not None else None
        self._tts_override = dict(tts_backends) if tts_backends is not None else None
        self.heartbeat_s = heartbeat_s
        self.misses = misses
        self.health_interval_s = health_interval_s
        self._backend: Any = backend
        if fake_audio and backend is None:
            from aivtube.testing.fakes import FakeSD

            self._backend = FakeSD()
        self.fake_device: FakeAudioDevice | None = (
            FakeAudioDevice(self._backend) if fake_audio else None
        )
        self._policy = policy or VoicePolicy()
        self._muted = False
        self._duck: tuple[float, int] = (1.0, 30)
        self._rates: dict[str, int] = {}
        self._pipeline: _Pipeline | None = None
        self._config_lock: asyncio.Lock | None = None
        self._configuring: asyncio.Task[None] | None = None
        self._build_error: str | None = None
        self._state: tuple[str, str] = ("starting", "waiting for voice.configure")
        self._last_sent: tuple[str, str] | None = None
        self._linked = False
        self._client: IpcClient | None = None
        self._tasks: SupervisedTasks | None = None
        self._bus: AsyncEventBus | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt: asyncio.Event | None = None
        self._health_evt: asyncio.Event | None = None
        self._lost_since: float | None = None
        self._last_reinit = float("-inf")
        self._stopping = False
        self.exit_code = 0
        self.sent: collections.deque[tuple[str, Mapping[str, Any]]] = collections.deque(
            maxlen=4096
        )  # everything the worker tried to send (tests, flight dumps)
        self.stats: collections.Counter[str] = collections.Counter()

    # --- public ---------------------------------------------------------------------------
    @property
    def pipeline(self) -> _Pipeline | None:
        return self._pipeline

    @property
    def ready(self) -> bool:
        pipe = self._pipeline
        return pipe is not None and pipe.ready

    @property
    def linked(self) -> bool:
        return self._linked

    @property
    def client(self) -> IpcClient | None:
        return self._client

    @property
    def policy(self) -> VoicePolicy:
        return self._policy

    def stop(self) -> None:
        """Ask ``run()`` to shut down (call on the loop thread)."""
        if self._stop_evt is not None:
            self._stop_evt.set()

    async def run(self) -> None:
        """Connect, serve the core and keep the pipeline alive until ``stop()``/cancel."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._stop_evt = asyncio.Event()
        self._health_evt = asyncio.Event()
        self._config_lock = asyncio.Lock()
        self._bus = AsyncEventBus(self._clock, loop=loop)
        self._tasks = SupervisedTasks(self._clock, self._bus, on_critical_failure=self._critical)
        self._client = IpcClient(
            self.bus_url,
            self._token,
            "voice",
            self._clock,
            caps={
                "worker": "aivtube-voice",
                "fake_audio": self.fake_audio,
                "fake_models": self.fake_models,
            },
            heartbeat_s=self.heartbeat_s,
            misses=self.misses,
        )
        self._register(self._client)
        self._client.on_link_up = self._on_link_up
        self._client.on_link_lost = self._on_link_lost
        if self.fake_device is not None:
            self.fake_device.start()
        self._tasks.spawn("ipc-client", self._client.run, critical=True)
        self._tasks.spawn("voice-health", self._health_loop)
        try:
            await self._stop_evt.wait()
        finally:
            await self._shutdown()

    async def apply_configure(self, data: Mapping[str, Any]) -> None:
        """Build (or keep) the pipeline for a ``voice.configure`` payload, then report READY."""
        assert self._config_lock is not None, "call run() first"
        async with self._config_lock:
            payload = dict(data)
            pipe = self._pipeline
            if pipe is not None and pipe.ready and pipe.payload == payload:
                log.info("voice.configure unchanged: keeping the pipeline")
                self._send_constraints(pipe)
                self._update_health(force=True)
                return
            if pipe is not None:
                self._pipeline = None
                await self._close_pipeline(pipe)
            self._build_error = None
            self._set_state("starting", "loading audio, VAD, STT and TTS")
            try:
                pipe = await self._build(payload)
            except Exception as exc:
                log.exception("voice pipeline could not be built")
                self._build_error = f"{type(exc).__name__}: {exc}"[:300]
                self._set_state("failed", self._build_error)
                return
            self._pipeline = pipe
            await self._warmup(pipe)
            await self._apply_state(pipe)  # what arrived while it was loading
            pipe.ready = True
            self._send_constraints(pipe)
            self._update_health(force=True)
            state, detail = self._compute_health()
            log.info("voice worker READY: %s %s", state, detail)

    async def apply_policy(self, policy: VoicePolicy) -> None:
        """Mic mode, push-to-talk, barge-in policy, echo mode and listening."""
        self._policy = policy
        pipe = self._pipeline
        if pipe is None:
            return
        if self._linked:
            pipe.frontend.apply_policy(policy)
        pipe.barge.set_policy(policy.barge_in)
        if not self._listening_allowed():
            pipe.barge.reset()
            pipe.clear_stt()
        if policy.echo_mode != pipe.echo_request:
            await self._apply_echo(pipe)

    # --- IPC handlers -----------------------------------------------------------------------
    def _register(self, client: IpcClient) -> None:
        client.on(ipc.VOICE_CONFIGURE, self._h_configure)
        client.on(ipc.VOICE_POLICY, self._h_policy)
        client.on(ipc.SPEAK_BEGIN, self._h_begin)
        client.on(ipc.SPEAK_SEGMENT, self._h_segment)
        client.on(ipc.SPEAK_GATE, self._h_gate)
        client.on(ipc.SPEAK_STOP, self._h_stop)
        client.on(ipc.SPEAK_DUCK, self._h_duck)
        client.on(ipc.SPEAK_CANNED, self._h_canned)
        client.on(ipc.VOICE_MUTE, self._h_mute)
        client.on(ipc.VOICE_RATE, self._h_rate)

    async def _h_configure(self, env: ipc.Envelope) -> None:
        # shielded: a link drop must not cancel a half-built pipeline (it would leak devices)
        assert self._tasks is not None
        prev = self._configuring
        if prev is not None and not prev.done():
            await asyncio.shield(prev)
        task = self._tasks.track(self.apply_configure(env.data), name="voice-configure")
        self._configuring = task
        await asyncio.shield(task)

    async def _h_policy(self, env: ipc.Envelope) -> None:
        await self.apply_policy(VoicePolicy(**env.data))

    async def _h_begin(self, env: ipc.Envelope) -> None:
        pipe = self._ready_pipeline()
        if pipe is None:
            self.stats["begin_not_ready"] += 1
            return
        d = env.data
        try:
            await pipe.queue.begin(
                str(d["utt"]),
                str(d["character"]),
                filler_after_s=d.get("filler_after_s"),
                gate_open=bool(d["gate_open"]),
            )
        except ValueError as exc:
            log.warning("speak.begin ignored: %s", exc)

    async def _h_segment(self, env: ipc.Envelope) -> None:
        client = self._client
        assert client is not None
        pipe = self._ready_pipeline()
        if pipe is None:
            client.reply(env, "error", detail="voice worker not ready")
            return
        d = env.data
        seg = Segment(
            utt_id=str(d["utt"]),
            seq=int(d["seq"]),
            text=str(d["text"]),
            caption=str(d["caption"]),
            emotion=d.get("emotion"),
            last=bool(d["last"]),
            kind=d["kind"],
        )
        try:
            ok = await pipe.queue.segment(seg)
        except Exception as exc:
            log.exception("speak.segment failed")
            client.reply(env, "error", detail=f"{type(exc).__name__}: {exc}"[:200])
            return
        client.reply(env, "ok" if ok else "busy")

    async def _h_gate(self, env: ipc.Envelope) -> None:
        pipe = self._ready_pipeline()
        if pipe is not None:
            await pipe.queue.open_gate(str(env.data["utt"]))

    async def _h_stop(self, env: ipc.Envelope) -> None:
        pipe = self._ready_pipeline()
        if pipe is None:
            return
        d = env.data
        utt = d.get("utt")
        await pipe.queue.stop(
            None if utt is None else str(utt), d["mode"], str(d["reason"]), int(d["fade_ms"])
        )

    async def _h_duck(self, env: ipc.Envelope) -> None:
        self._duck = (float(env.data["gain"]), int(env.data["ramp_ms"]))
        pipe = self._ready_pipeline()
        if pipe is not None:
            await pipe.queue.duck(*self._duck)

    async def _h_canned(self, env: ipc.Envelope) -> None:
        pipe = self._ready_pipeline()
        if pipe is not None:
            await pipe.queue.play_canned(str(env.data["key"]), str(env.data["character"]))

    async def _h_mute(self, env: ipc.Envelope) -> None:
        self._muted = bool(env.data["on"])
        pipe = self._ready_pipeline()
        if pipe is not None:
            pipe.queue.set_muted(self._muted)

    async def _h_rate(self, env: ipc.Envelope) -> None:
        character, percent = str(env.data["character"]), int(env.data["percent"])
        self._rates[character] = percent
        pipe = self._ready_pipeline()
        if pipe is not None:
            await pipe.queue.set_voice_rate(character, percent)

    def _ready_pipeline(self) -> _Pipeline | None:
        pipe = self._pipeline
        return pipe if pipe is not None and pipe.ready else None

    # --- link events (invariant I1) ---------------------------------------------------------
    def _on_link_up(self) -> None:
        self._linked = True
        self._last_sent = None
        self._update_health(force=True)
        pipe = self._pipeline
        if pipe is not None:
            pipe.frontend.apply_policy(self._policy)  # listen again (per policy)
            if pipe.ready:
                self._send_constraints(pipe)

    def _on_link_lost(self) -> None:
        """I1: finish the current segment, clear the queue, stop listening, wait."""
        self._linked = False
        self.stats["link_lost"] += 1
        pipe = self._pipeline
        if pipe is None or self._stopping:
            return
        pipe.frontend.set_enabled(False)
        pipe.barge.reset()
        pipe.clear_stt()
        assert self._tasks is not None
        self._tasks.track(pipe.queue.drop_all(), name="i1-drop-all")

    def _listening_allowed(self) -> bool:
        p = self._policy
        return self._linked and p.listening and p.mic_mode != "deafened"

    # --- sending ----------------------------------------------------------------------------
    def _send(self, mtype: str, data: Mapping[str, Any]) -> None:
        self.sent.append((mtype, data))
        client = self._client
        if client is not None:
            client.post(mtype, data)

    def _send_constraints(self, pipe: _Pipeline) -> None:
        for character in pipe.characters:
            self._on_constraints(character, pipe.queue.constraints(character))

    def _on_constraints(self, character: str, c: Any) -> None:
        self._send(
            ipc.TTS_CONSTRAINTS,
            {
                "character": character,
                "first_min_chars": int(c.first_min_chars),
                "min_chars": int(c.min_chars),
                "max_chars": int(c.max_chars),
                "backend": str(c.backend),
                "identity": str(c.identity),
            },
        )

    def _on_fallback(self, frm: str, to: str | None, reason: str) -> None:
        self._send(ipc.TTS_FALLBACK, {"from": frm, "to": to, "reason": reason})
        self._health_changed()

    # --- health -----------------------------------------------------------------------------
    def _set_state(self, state: str, detail: str) -> None:
        self._state = (state, detail)
        self._update_health()

    def _health_changed(self) -> None:
        """Any thread: re-evaluate health soon."""
        loop = self._loop
        evt = self._health_evt
        if loop is None or evt is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(evt.set)

    def _compute_health(self) -> tuple[str, str]:
        if self._build_error is not None:
            return "failed", self._build_error
        pipe = self._pipeline
        if pipe is None or not pipe.ready:
            return self._state
        problems = list(pipe.notes)
        stt = pipe.stt.health()
        if stt.state is not HealthState.OK:
            problems.append(f"stt {stt.state.value}: {stt.detail}".strip())
        if not pipe.player.device_ok:
            problems.append("output device lost")
        if pipe.mic is not None and not pipe.mic.device_ok:
            problems.append("input device lost")
        for b in pipe.router.status():
            if b["open"]:
                problems.append(f"tts {b['backend']} out for {b['down_for_s']:.0f} s")
        if problems:
            return "degraded", "; ".join(problems)[:500]
        return "ok", ""

    def _stats(self) -> dict[str, Any]:
        pipe = self._pipeline
        out: dict[str, Any] = {"worker": dict(self.stats), "linked": self._linked}
        if pipe is not None:
            out["player"] = dict(pipe.player.stats)
            if pipe.mic is not None:
                out["mic"] = dict(pipe.mic.stats)
            out["speech"] = dict(pipe.queue.stats)
            out["stt"] = dict(pipe.stt.stats)
            out["barge"] = dict(pipe.barge.stats)
            out["frontend"] = dict(pipe.frontend.stats)
            out["echo_mode"] = pipe.echo_mode
        if self.fake_device is not None:
            out["fake_audio_errors"] = self.fake_device.errors
        return out

    def _update_health(self, *, force: bool = False) -> None:
        state, detail = self._compute_health()
        if force or self._last_sent != (state, detail):
            self._last_sent = (state, detail)
            self._send(ipc.HEALTH, {"state": state, "detail": detail, "stats": self._stats()})

    async def _health_loop(self) -> None:
        assert self._health_evt is not None
        next_full = self._clock.now() + self.health_interval_s
        while True:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(1.0):
                    await self._health_evt.wait()
            self._health_evt.clear()
            now = self._clock.now()
            force = now >= next_full
            if force:
                next_full = now + self.health_interval_s
            self._update_health(force=force)
            await self._check_devices(now)

    async def _check_devices(self, now: float) -> None:
        """§2.8: PortAudio only sees a re-plugged device after re-initialisation."""
        pipe = self._pipeline
        if pipe is None or not pipe.ready:
            return
        lost = not pipe.player.device_ok or (pipe.mic is not None and not pipe.mic.device_ok)
        if not lost:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = now
            return
        if now - self._lost_since >= 6.0 and now - self._last_reinit >= 10.0:
            self._last_reinit = now
            self.stats["portaudio_reinit"] += 1
            log.warning("audio device still lost; re-initialising PortAudio")
            from aivtube.voice.audio_io import reinit_portaudio

            try:
                async with deadline(5.0, what="PortAudio re-init"):
                    await asyncio.to_thread(reinit_portaudio, self._backend)
            except Exception as exc:
                log.error("PortAudio re-initialisation failed: %s", exc)

    # --- building ---------------------------------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p

    def _fake_payload(self, data: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(data)
        chain = list(out.get("stt_chain") or [])
        if not chain or not all(isinstance(e, Mapping) and e.get("kind") == "fake" for e in chain):
            out["stt_chain"] = [
                {"name": "fake", "kind": "fake", "script": dict(DEFAULT_FAKE_STT_SCRIPT)}
            ]
        tts = dict(out.get("tts") or {})
        backends = {}
        for name, spec in dict(tts.get("backends") or {}).items():
            kind = spec.get("kind") if isinstance(spec, Mapping) else None
            backends[name] = spec if kind in ("fake", "captions") else dict(_FAKE_TTS)
        tts["backends"] = backends
        out["tts"] = tts
        return out

    async def _build(self, payload: Mapping[str, Any]) -> _Pipeline:
        """Everything that touches devices or model files runs in worker threads."""
        from aivtube.voice.aec import EnergyDTD, make_echo_canceller
        from aivtube.voice.audio_io import MicCapture, StreamingPlayer
        from aivtube.voice.barge import BargeConfig, BargeInController
        from aivtube.voice.endpointer import SileroEndpointer
        from aivtube.voice.frontend import VoiceFrontEnd, post_to_loop
        from aivtube.voice.speech_queue import SpeechQueue
        from aivtube.voice.stt import NamePostProcessor, SttRunner, UtteranceAssembler
        from aivtube.voice.stt.factory import build_stt_chain
        from aivtube.voice.tts.factory import build_tts_router

        loop = asyncio.get_running_loop()
        spec = self._fake_payload(payload) if self.fake_models else dict(payload)
        audio: Mapping[str, Any] = spec.get("audio") or {}
        vad_spec: Mapping[str, Any] = spec.get("vad") or {}
        chars: dict[str, Mapping[str, Any]] = dict(spec.get("characters") or {})
        notes: list[str] = []
        rate = int(audio.get("samplerate", 48000))
        block = max(1, rate * int(audio.get("block_ms", 10)) // 1000)
        closers: list[Callable[[], Any]] = []
        try:
            # output first: she must be able to speak even without a microphone
            player = StreamingPlayer(
                str(audio.get("output_device") or "") or None,
                samplerate=rate,
                channels=2,
                blocksize=block,
                latency=float(audio.get("output_latency_s", 0.04)),
                mirror_device=str(audio.get("mirror_output_device") or "") or None,
                backend=self._backend,
                clock=self._now,
                on_device_lost=self._health_changed,
            )
            await _offload(player.start, what="open the output device", limit_s=15.0)
            closers.append(player.close)

            vad, vad_note = await _offload(self._make_vad, vad_spec, what="load the VAD")
            if vad_note:
                notes.append(vad_note)
            ep_cfg = _endpointer_config(vad_spec)
            endpointer = SileroEndpointer(vad, ep_cfg)

            post = NamePostProcessor(_aliases(chars))
            if self._stt_override is not None:
                recognizers: list[SpeechRecognizer] = list(self._stt_override)
            else:
                recognizers = await _offload(
                    build_stt_chain,
                    list(spec.get("stt_chain") or []),
                    {},
                    root=self.root,
                    secrets=self._secrets,
                    cloud_consent=self.cloud_stt_consent,
                    timeout_s=self.stt_timeout_s,
                    what="load the STT models",
                    limit_s=120.0,
                )
            stt = SttRunner(
                recognizers,
                post,
                timeout_s=self.stt_timeout_s,
                clock=self._clock,
                on_health=lambda _h: self._health_changed(),
            )
            closers.append(stt.close)
            if not recognizers:
                notes.append("no STT backend could be built")

            chains = {cid: list(c.get("identity_chain") or []) for cid, c in chars.items()}
            tts_spec = dict(spec.get("tts") or {})
            router = await _offload(
                build_tts_router,
                tts_spec,
                chains,
                root=self.root,
                secrets=self._secrets,
                clock=self._clock,
                backends=self._tts_override,
                on_fallback=self._on_fallback,
                on_constraints=self._on_constraints,
                what="build the TTS router",
            )

            pipe_ref: list[_Pipeline] = []

            def set_gain(gain: float, _ramp_ms: float) -> None:
                if pipe_ref:
                    pipe_ref[0].barge.set_base_gain(gain)

            queue = SpeechQueue(
                player=player,
                tts=router,
                send=self._send,
                clock=self._clock,
                supervisor=self._tasks,
                set_gain=set_gain,
                max_in_flight=int(tts_spec.get("max_in_flight", 2)),
                max_queued=int(tts_spec.get("max_queued_segments", 8)),
                filler_min_interval_s=float(tts_spec.get("filler_min_interval_s", 30.0)),
            )
            frontend = VoiceFrontEnd(
                endpointer,
                player=player,
                aec=None,
                dtd=None,
                in_rate=rate,
                on_event=post_to_loop(loop, self._vad_handler(pipe_ref)),
            )
            barge = BargeInController(
                player=player,
                stt=stt,
                frontend=frontend,
                cfg=BargeConfig.from_mapping(spec.get("barge_in") or {}),
                send=self._send,
                clock=self._clock,
                policy=self._policy.barge_in,
                recent_tts_text=queue.recent_tts_text,
                on_cut=self._barge_cut(pipe_ref),
            )
            pipe = _Pipeline(
                payload=dict(payload),
                player=player,
                mic=None,
                frontend=frontend,
                endpointer=endpointer,
                base_ep_cfg=ep_cfg,
                stt=stt,
                router=router,
                queue=queue,
                barge=barge,
                audio=audio,
                characters=chars,
                notes=notes,
                assembler=UtteranceAssembler(),
            )
            pipe_ref.append(pipe)
            await self._apply_echo(pipe, make=make_echo_canceller, dtd_cls=EnergyDTD)
            await self._apply_state(pipe)

            mic = MicCapture(
                str(audio.get("input_device") or "") or None,
                samplerate=rate,
                blocksize=block,
                backend=self._backend,
                clock=self._now,
                on_device_lost=self._health_changed,
            )
            try:
                await _offload(mic.start, frontend.feed, what="open the microphone", limit_s=15.0)
            except Exception as exc:
                log.error("microphone unavailable: %s", exc)
                notes.append(f"no microphone: {exc}"[:200])
                mic.close()
            else:
                pipe.mic = mic
            assert self._tasks is not None
            pipe.tasks.append(self._tasks.track(barge.run(), name="barge-in"))
            pipe.tasks.append(self._tasks.track(self._stt_loop(pipe), name="stt-turns"))
            return pipe
        except BaseException:
            await asyncio.to_thread(_close_all, closers)  # joins threads: not on the loop
            raise

    def _make_vad(self, vad_spec: Mapping[str, Any]) -> tuple[Any, str]:
        """The configured VAD, or the energy VAD with a note when it cannot be loaded."""
        from aivtube.voice.vad import EnergyVAD, make_vad

        backend = str(vad_spec.get("backend", "silero_ort"))
        energy = float(vad_spec.get("energy_threshold_dbfs", -42.0))
        try:
            vad = make_vad(
                backend,  # type: ignore[arg-type]
                self._resolve(str(vad_spec.get("model", "models/vad/silero_vad.onnx"))),
                str(vad_spec.get("model_sha256", "")),
                threshold=float(vad_spec.get("threshold", 0.5)),
                energy_threshold_dbfs=energy,
            )
        except Exception as exc:
            log.error("VAD %s unavailable (%s); using the energy VAD", backend, exc)
            return EnergyVAD(energy), f"VAD {backend} unavailable: energy VAD in use"
        vad.prob(np.zeros(vad.frame_samples, np.float32))  # warm-up
        vad.reset()
        return vad, ""

    async def _apply_echo(
        self,
        pipe: _Pipeline,
        *,
        make: Callable[..., Any] | None = None,
        dtd_cls: type[Any] | None = None,
    ) -> None:
        """Echo handling for the policy's (or the config's) mode; §4.7 step 6."""
        from aivtube.voice.barge import barge_threshold_for

        if make is None or dtd_cls is None:
            from aivtube.voice.aec import EnergyDTD, make_echo_canceller

            make, dtd_cls = make_echo_canceller, EnergyDTD
        pipe.echo_request = self._policy.echo_mode
        wanted = self._policy.echo_mode
        if wanted == "auto":
            wanted = pipe.audio.get("echo_mode", "auto")
        headphones = bool(pipe.audio.get("headphones", True))
        rate = int(pipe.audio.get("samplerate", 48000))
        aec, effective = await _offload(
            make, wanted, headphones, rate=rate, what="start echo cancellation"
        )
        pipe.echo_mode = effective
        pipe.frontend.configure_echo(
            aec=aec,
            dtd=dtd_cls() if effective == "energy_dtd" else None,
            half_duplex=effective == "half_duplex",
        )
        pipe.barge.set_echo_mode(effective)
        base = pipe.base_ep_cfg
        pipe.endpointer.set_config(
            replace(base, barge_threshold=barge_threshold_for(effective, base.barge_threshold))
        )
        note = "echo mode aec unavailable: energy_dtd in use"
        if wanted == "aec" and effective != "aec":
            if note not in pipe.notes:
                pipe.notes.append(note)
        elif note in pipe.notes:
            pipe.notes.remove(note)

    async def _apply_state(self, pipe: _Pipeline) -> None:
        """Policy, echo mode, MUTE, duck and talking speeds as last received from the core."""
        if pipe.echo_request != self._policy.echo_mode:
            await self._apply_echo(pipe)
        if self._linked:
            pipe.frontend.apply_policy(self._policy)
        else:
            pipe.frontend.set_enabled(False)
        pipe.barge.set_policy(self._policy.barge_in)
        pipe.queue.set_muted(self._muted)
        await pipe.queue.duck(*self._duck)
        for character, percent in self._rates.items():
            await pipe.queue.set_voice_rate(character, percent)

    async def _warmup(self, pipe: _Pipeline) -> None:
        self._set_state("starting", "warming up STT and TTS")
        await pipe.stt.warmup(timeout_s=60.0)
        warm = await pipe.router.warmup(timeout_s=10.0)
        for name, ok in warm.items():
            if not ok:
                pipe.notes.append(f"tts {name} warm-up failed")
        filtered = True
        for character, spec in pipe.characters.items():
            phrases = [str(p) for p in spec.get("cached_phrases") or []]
            if not phrases:
                continue
            jobs = [pipe.router.presynthesize(character, [p], timeout_s=15.0) for p in phrases]
            results: dict[str, bool] = {}
            try:
                async with deadline(30.0, what="phrase pre-synthesis", clock=self._clock):
                    # an outer cancellation still propagates out of gather()
                    for part in await asyncio.gather(*jobs, return_exceptions=True):
                        if isinstance(part, BaseException):
                            log.warning("pre-synthesis failed: %r", part)
                        else:
                            results.update(part)
            except TimeoutError:
                log.warning("pre-synthesis for %s timed out", character)
            missing = [p for p in phrases if not results.get(p)]
            if missing:
                log.warning("%s: %d cached phrase(s) unavailable", character, len(missing))
            if "Filtered." in phrases and not results.get("Filtered."):
                filtered = False
                pipe.notes.append(f"{character}: 'Filtered.' is not cached")
        pipe.filtered_cached = filtered

    async def _close_pipeline(self, pipe: _Pipeline) -> None:
        pipe.ready = False
        with contextlib.suppress(Exception):
            pipe.frontend.set_enabled(False)
        for task in pipe.tasks:
            task.cancel()
        await asyncio.gather(*pipe.tasks, return_exceptions=True)
        if pipe.mic is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(pipe.mic.close)
        with contextlib.suppress(Exception):
            await pipe.queue.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(pipe.player.close)
        with contextlib.suppress(Exception):
            await pipe.stt.aclose()
        with contextlib.suppress(Exception):
            await pipe.router.aclose()

    # --- mic → STT ----------------------------------------------------------------------------
    def _vad_handler(self, pipe_ref: list[_Pipeline]) -> Callable[[VadEvent], None]:
        def handle(ev: VadEvent) -> None:
            if not pipe_ref or pipe_ref[0] is not self._pipeline:
                return  # an event from a pipeline that has been replaced
            try:
                self._on_vad_event(pipe_ref[0], ev)
            except Exception:
                log.exception("VAD event handling failed")

        return handle

    def _on_vad_event(self, pipe: _Pipeline, ev: VadEvent) -> None:
        if not pipe.ready or not self._listening_allowed():
            self.stats["vad_ignored"] += 1
            return
        if isinstance(ev, VadStart):
            pipe.turn_audio_s = 0.0
            pipe.barge.on_vad_start(ev.t, ev.barge)
            self._send(ipc.VAD_START, {"t": ev.t, "barge": ev.barge})
        elif isinstance(ev, VadPartial):
            pipe.turn_audio_s += ev.audio.size / VAD_RATE
            pipe.stt_q.put_nowait(("partial", ev.audio, ev.t))
        elif isinstance(ev, VadEnd):
            audio_s = pipe.turn_audio_s + ev.audio.size / VAD_RATE
            pipe.turn_audio_s = 0.0
            pipe.barge.on_vad_end(ev.t)
            self._send(ipc.VAD_END, {"t": ev.t, "audio_s": audio_s})
            pipe.stt_q.put_nowait(("end", ev.audio, ev.t))

    async def _stt_loop(self, pipe: _Pipeline) -> None:
        """Decodes one turn's parts in order and sends one ``stt.final`` per turn."""
        while True:
            kind, audio, t = await pipe.stt_q.get()
            try:
                tr = await pipe.stt.transcribe(audio, recent_tts_text=pipe.queue.recent_tts_text())
                if kind == "partial":
                    if tr is not None:
                        pipe.assembler.add_partial(tr)
                    continue
                turn = pipe.assembler.finish(tr)
                if turn is None:
                    self.stats["stt_empty"] += 1
                    continue
                if not self._listening_allowed():
                    self.stats["stt_dropped_deaf"] += 1
                    continue
                if pipe.barge.suppress_backchannel(turn.text, t):
                    self.stats["stt_backchannel"] += 1
                    continue
                self._send(
                    ipc.STT_FINAL,
                    {
                        "text": turn.text,
                        "engine": turn.engine,
                        "latency_ms": max(0.0, float(turn.latency_ms)),
                        "t_end": t,
                        "audio_s": max(0.0, float(turn.audio_s)),
                        "parts": max(1, int(pipe.assembler.parts)),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats["stt_error"] += 1
                log.exception("STT turn failed")

    def _barge_cut(self, pipe_ref: list[_Pipeline]) -> Callable[[], None]:
        def cut() -> None:
            if pipe_ref and self._tasks is not None:
                self._tasks.track(
                    pipe_ref[0].queue.stop(None, "now", "barge_in", 60), name="barge-cut"
                )

        return cut

    # --- lifecycle ----------------------------------------------------------------------------
    def _critical(self, name: str, exc: BaseException) -> None:
        if self._stopping:
            return  # the IPC client returns when we close it
        log.critical("critical task %s failed: %s", name, exc)
        self.exit_code = EXIT_CRITICAL
        self.stop()

    async def _shutdown(self) -> None:
        self._stopping = True
        client = self._client
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()
        configuring = self._configuring
        if configuring is not None and not configuring.done():
            with contextlib.suppress(Exception):
                async with asyncio.timeout(10.0):
                    await asyncio.shield(configuring)
        pipe, self._pipeline = self._pipeline, None
        if pipe is not None:
            await self._close_pipeline(pipe)
        if self._tasks is not None:
            await self._tasks.aclose(3.0)
        if self.fake_device is not None:
            await asyncio.to_thread(self.fake_device.stop)
        if self._bus is not None:
            self._bus.close()


async def _offload(
    fn: Callable[..., Any], *args: Any, what: str, limit_s: float = 60.0, **kwargs: Any
) -> Any:
    """Run blocking ``fn(*args, **kwargs)`` in a worker thread within ``limit_s`` (I2): a hung
    driver or model load fails the configure instead of wedging it (the thread lives on)."""
    async with deadline(limit_s, what=what):
        return await asyncio.to_thread(fn, *args, **kwargs)


def _close_all(closers: Sequence[Callable[[], Any]]) -> None:
    for close in reversed(closers):
        try:
            close()
        except Exception:
            log.exception("closing %r failed", close)


def _endpointer_config(vad: Mapping[str, Any]) -> EndpointerConfig:
    names = {f.name for f in dataclasses.fields(EndpointerConfig)}
    return EndpointerConfig(**{k: v for k, v in vad.items() if k in names})


def _aliases(chars: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in chars.values():
        for canon, wrongs in dict(spec.get("stt_aliases") or {}).items():
            out.setdefault(str(canon), []).extend(str(w) for w in wrongs)
    return out


# --- process entry ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aivtube _voice", description="The aivtube voice worker.")
    p.add_argument("--root", type=Path, default=None, help="the AI_Vtube folder")
    p.add_argument("--profile", default=None)
    p.add_argument(
        "--url", default=None, help="IPC bus URL (default ws://127.0.0.1:<ports.bus>/bus)"
    )
    p.add_argument("--fake-audio", action="store_true", help="FakeSD instead of PortAudio")
    p.add_argument("--fake-models", action="store_true", help="fake STT and TTS")
    p.add_argument("--fake-llm", action="store_true", help="(launcher flag) implies --fake-models")
    p.add_argument("--text", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--speak", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--safe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--log-level", default=None)
    return p


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None) -> int:
    """Run the voice worker until the launcher stops it; returns the exit code."""
    args = build_parser().parse_args(argv)
    sys.setswitchinterval(0.001)
    raise_priority()
    from aivtube.config import ConfigError, find_root, load_config, load_secrets
    from aivtube.infra.logging import add_secrets, fault_file, setup_logging, shutdown_logging

    try:
        root = find_root(args.root)
        overrides: dict[str, Any] = {"app.fakes": True} if args.fake_llm else {}
        cfg = load_config(root, profile=args.profile, cli_overrides=overrides)
        secrets = load_secrets(root)
    except ConfigError as exc:
        print(f"voice worker: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    level = args.log_level or cfg.logging.level
    setup_logging(
        "voice",
        cfg.resolve_path(cfg.logging.dir),
        secrets=secrets.redaction_values(),
        level=level,
        console=cfg.logging.console,
        jsonl=cfg.logging.jsonl,
        max_bytes=cfg.logging.max_mb * 1024 * 1024,
        backups=cfg.logging.backups,
        keep_days=cfg.logging.keep_days,
    )
    try:
        token = os.environ.get(BUS_TOKEN_ENV, "")
        if not token:
            log.error("%s is not set; the launcher passes the IPC token", BUS_TOKEN_ENV)
            return EXIT_CONFIG
        add_secrets(token)
        with contextlib.suppress(ImportError):
            from aivtube.launcher.childside import start_dump_request_watcher

            start_dump_request_watcher(file=fault_file())
        fake_audio = (
            args.fake_audio or _truthy(os.environ.get(FAKE_AUDIO_ENV)) or cfg.active_profile == "ci"
        )
        worker = VoiceWorker(
            args.url or f"ws://127.0.0.1:{cfg.ports.bus}/bus",
            token,
            root=root,
            secrets=secrets.get,
            cloud_stt_consent=cfg.privacy.cloud_stt_consent,
            stt_timeout_s=cfg.stt.timeout_s,
            fake_audio=fake_audio,
            fake_models=args.fake_models or args.fake_llm or cfg.app.fakes,
            policy=VoicePolicy(  # until the core sends voice.policy
                mic_mode=cfg.mic.mode, barge_in=cfg.barge_in.policy, echo_mode=cfg.audio.echo_mode
            ),
        )
        asyncio.run(_amain(worker))
        return worker.exit_code
    except KeyboardInterrupt:
        return 0
    finally:
        shutdown_logging()


async def _amain(worker: VoiceWorker) -> None:
    loop = asyncio.get_running_loop()

    def on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d: stopping", signum)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(worker.stop)

    names = ("SIGINT", "SIGTERM", "SIGBREAK")  # SIGBREAK: CTRL_BREAK_EVENT from the launcher
    for name in names:
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(ValueError, OSError):  # not the main thread
                signal.signal(sig, on_signal)
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
