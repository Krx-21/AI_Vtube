"""STT runner: dedicated decode threads, priority queues, a timeout chain and health (§2.4, §2.8).

Each recognizer in the chain gets one dedicated daemon thread with a priority queue, so a decode
never runs on the event loop and the native model is only ever used from one thread. Quick
requests (barge-in confirmation) jump ahead of queued normal requests.

``SttRunner.transcribe`` tries the chain in order. A backend that raises, or that has not
answered within ``timeout_s`` (3 s, including queueing), is skipped for this request and the
next backend gets the audio. A backend whose thread is stuck in a timed-out decode is skipped
until that decode returns (its queued jobs are rerouted at once instead of timing out one by
one), and ``breaker_failures`` failures within ``breaker_window_s`` take it out for
``breaker_cooldown_s``. Any fallback marks the runner DEGRADED; a successful primary decode
makes it OK again. The post-processor runs on the decode thread too.

``UtteranceAssembler`` joins the decodes of forced splits (``VadPartial``) with the final
decode (``VadEnd``) into one transcript per turn.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aivtube.contracts.infra import Clock
from aivtube.contracts.types import Health, HealthState, Transcript
from aivtube.contracts.voice import F32, SpeechRecognizer, TranscriptPostProcessor
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.voice.stt.sherpa import to_mono_f32

__all__ = ["SttRunner", "UtteranceAssembler"]

log = logging.getLogger("aivtube.voice.stt")

_QUICK, _NORMAL, _STOP = 1, 2, 0  # PriorityQueue: lower first


class _Rerouted(Exception):
    """The backend became unavailable while the job waited in its queue."""


@dataclass(eq=False)
class _Job:
    pcm: F32
    quick: bool
    recent: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[tuple[Transcript, Transcript | None]]
    cancelled: bool = False


@dataclass(eq=False)
class _Backend:
    rec: SpeechRecognizer
    post: TranscriptPostProcessor | None
    queue: queue.PriorityQueue[tuple[int, int, _Job | None]] = field(
        default_factory=queue.PriorityQueue
    )
    thread: threading.Thread | None = None
    current: _Job | None = None
    stuck: bool = False
    closed: bool = False
    failures: collections.deque[float] = field(default_factory=collections.deque)
    open_until: float = float("-inf")
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def name(self) -> str:
        return str(getattr(self.rec, "name", "stt"))


def _resolve(
    fut: asyncio.Future[tuple[Transcript, Transcript | None]],
    result: tuple[Transcript, Transcript | None] | None,
    exc: BaseException | None,
) -> None:
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        assert result is not None
        fut.set_result(result)


class SttRunner:
    """Async front of the STT chain. Create and use it on one event loop."""

    def __init__(
        self,
        chain: Sequence[SpeechRecognizer],
        post: TranscriptPostProcessor | None,
        *,
        timeout_s: float = 3.0,
        clock: Clock | None = None,
        breaker_failures: int = 3,
        breaker_window_s: float = 60.0,
        breaker_cooldown_s: float = 30.0,
        on_health: Callable[[Health], None] | None = None,
        name: str = "stt",
    ) -> None:
        if not chain:
            log.error("STT chain is empty: speech input is disabled")
        self.timeout_s = timeout_s
        self.name = name
        self._clock: Clock = clock or SystemClock()
        self._backends = [_Backend(rec, post) for rec in chain]
        self._seq = itertools.count()
        self._breaker = (breaker_failures, breaker_window_s, breaker_cooldown_s)
        self._on_health = on_health
        self._closed = False
        state = HealthState.OK if chain else HealthState.DOWN
        self._health = Health(name, state, "" if chain else "no STT backend", self._clock.now())
        self.stats: collections.Counter[str] = collections.Counter()

    # --- public API -----------------------------------------------------------------------
    @property
    def chain(self) -> list[str]:
        return [b.name for b in self._backends]

    def health(self) -> Health:
        return self._health

    async def warmup(self, timeout_s: float = 60.0) -> Health:
        """Warm every recognizer on its own thread; a failing primary marks DEGRADED."""
        for i, b in enumerate(self._backends):
            try:
                async with deadline(timeout_s, what=f"stt warmup {b.name}", clock=self._clock):
                    await self._run_on(b, b.rec.warmup)
            except Exception as exc:  # incl. DeadlineExceeded
                log.warning("STT backend %s failed to warm up: %s", b.name, exc)
                self._record_failure(b)
                if i == 0:
                    self._set_health(HealthState.DEGRADED, f"{b.name} warm-up failed: {exc}")
        return self._health

    async def transcribe(
        self, pcm16k: F32, *, quick: bool = False, recent_tts_text: str = ""
    ) -> Transcript | None:
        """Decode on the first backend that answers in time; ``None`` if dropped or all failed.

        ``None`` also means the post-processor dropped the transcript (too short, empty or an
        echo); that is not a failure and never moves down the chain.
        """
        if self._closed or not self._backends:
            return None
        loop = asyncio.get_running_loop()
        pcm = to_mono_f32(pcm16k)
        reasons: list[str] = []
        for i, b in enumerate(self._backends):
            now = self._clock.now()
            if b.closed:
                continue
            if b.stuck or now < b.open_until:
                reasons.append(f"{b.name}: unavailable")
                continue
            job = _Job(pcm, quick, recent_tts_text, loop, loop.create_future())
            self._submit(b, job)
            try:
                async with deadline(self.timeout_s, what=f"stt {b.name}", clock=self._clock):
                    _raw, out = await job.future
            except DeadlineExceeded:
                job.cancelled = True
                self.stats[f"{b.name}.timeout"] += 1
                self._on_timeout(b)
                reasons.append(f"{b.name}: no result within {self.timeout_s:g} s")
                continue
            except asyncio.CancelledError:
                job.cancelled = True
                raise
            except _Rerouted:
                reasons.append(f"{b.name}: stuck")
                continue
            except Exception as exc:
                if self._closed:
                    return None
                self.stats[f"{b.name}.error"] += 1
                self._record_failure(b)
                log.warning("STT backend %s failed: %s", b.name, exc)
                reasons.append(f"{b.name}: {type(exc).__name__}")
                continue
            self.stats[f"{b.name}.ok"] += 1
            if i == 0:
                if self._health.state != HealthState.OK:
                    self._set_health(HealthState.OK, "")
            else:
                self.stats["fallbacks"] += 1
                self._set_health(HealthState.DEGRADED, "; ".join(reasons))
            if out is None:
                self.stats["dropped"] += 1
            return out
        self.stats["lost"] += 1
        self._set_health(HealthState.DEGRADED, "; ".join(reasons) or "no STT backend available")
        return None

    def close(self) -> None:
        """Stop the decode threads without blocking; each closes its recognizer when idle."""
        if self._closed:
            return
        self._closed = True
        for b in self._backends:
            b.closed = True
            self._drain(b, RuntimeError("SttRunner closed"))
            with b.lock:
                thread = b.thread
            if thread is None:
                with contextlib.suppress(Exception):
                    b.rec.close()
            else:
                b.queue.put((_STOP, -1, None))

    def join(self, timeout: float = 2.0) -> None:
        """Wait for the decode threads to exit (blocking; call via ``asyncio.to_thread``)."""
        for b in self._backends:
            if b.thread is not None:
                b.thread.join(timeout)

    async def aclose(self, join_s: float = 2.0) -> None:
        self.close()
        await asyncio.to_thread(self.join, join_s)

    # --- internals ------------------------------------------------------------------------
    def _ensure_thread(self, b: _Backend) -> None:
        with b.lock:
            if b.thread is None:
                b.thread = threading.Thread(
                    target=self._worker, args=(b,), name=f"stt-{b.name}", daemon=True
                )
                b.thread.start()

    def _submit(self, b: _Backend, job: _Job) -> None:
        self._ensure_thread(b)
        b.queue.put((_QUICK if job.quick else _NORMAL, next(self._seq), job))

    async def _run_on(self, b: _Backend, fn: Callable[[], Any]) -> None:
        """Run ``fn`` on the backend's thread (warm-up) and wait for it."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[Transcript, Transcript | None]] = loop.create_future()
        job = _Job(to_mono_f32([]), False, "", loop, fut)
        self._ensure_thread(b)
        b.queue.put((_NORMAL, next(self._seq), _WarmupJob(job, fn)))
        await fut

    def _worker(self, b: _Backend) -> None:
        while True:
            _prio, _seq, job = b.queue.get()
            if job is None:
                break
            if job.cancelled or job.future.done():
                continue
            with b.lock:
                b.current = job
            result: tuple[Transcript, Transcript | None] | None = None
            error: BaseException | None = None
            try:
                if isinstance(job, _WarmupJob):
                    job.fn()
                    result = (_EMPTY, None)
                else:
                    raw = b.rec.transcribe(job.pcm, quick=job.quick)
                    out = b.post(raw, job.recent) if b.post is not None else raw
                    result = (raw, out)
            except Exception as exc:
                error = exc
            finally:
                with b.lock:
                    b.current = None
                    b.stuck = False
            with contextlib.suppress(RuntimeError):  # the loop may already be closed
                job.loop.call_soon_threadsafe(_resolve, job.future, result, error)
        with contextlib.suppress(Exception):
            b.rec.close()

    def _on_timeout(self, b: _Backend) -> None:
        self._record_failure(b)
        with b.lock:
            running = b.current is not None
            if running:
                b.stuck = True
        if running:
            log.warning("STT backend %s is stuck in a decode; rerouting its queue", b.name)
            self._drain(b, _Rerouted(b.name))

    def _drain(self, b: _Backend, exc: BaseException) -> None:
        pending: list[_Job] = []
        while True:
            try:
                _p, _s, job = b.queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                pending.append(job)
        for job in pending:
            job.cancelled = True
            with contextlib.suppress(RuntimeError):
                job.loop.call_soon_threadsafe(_resolve, job.future, None, exc)

    def _record_failure(self, b: _Backend) -> None:
        count, window, cooldown = self._breaker
        now = self._clock.now()
        b.failures.append(now)
        while b.failures and b.failures[0] < now - window:
            b.failures.popleft()
        if len(b.failures) >= count:
            b.open_until = now + cooldown
            b.failures.clear()
            log.warning("STT backend %s out for %.0f s after %d failures", b.name, cooldown, count)

    def _set_health(self, state: HealthState, detail: str) -> None:
        if state == self._health.state and detail == self._health.detail:
            return
        changed = state != self._health.state
        self._health = Health(self.name, state, detail, self._clock.now())
        if changed and self._on_health is not None:
            try:
                self._on_health(self._health)
            except Exception:
                log.exception("STT health callback failed")


_EMPTY = Transcript(text="", is_final=True, audio_s=0.0, latency_ms=0.0, engine="warmup")


class _WarmupJob(_Job):
    """A job that runs an arbitrary callable on the backend thread."""

    def __init__(self, job: _Job, fn: Callable[[], Any]) -> None:
        super().__init__(job.pcm, False, "", job.loop, job.future)
        self.fn = fn


class UtteranceAssembler:
    """Joins the decodes of one turn: forced-split partials plus the final decode.

    Texts are joined with single spaces; ``audio_s`` is the sum of every part; ``latency_ms``
    is the final decode's (what the user waits for); ``engine`` lists the engines used.
    ``parts`` is the number of non-empty decodes in the last ``finish()``.
    """

    def __init__(self) -> None:
        self._parts: list[Transcript] = []
        self.parts = 0

    @property
    def pending(self) -> int:
        return len(self._parts)

    def add_partial(self, t: Transcript) -> None:
        self._parts.append(t)

    def reset(self) -> None:
        self._parts.clear()

    def finish(self, last: Transcript | None) -> Transcript | None:
        items = [*self._parts, *([last] if last is not None else [])]
        self._parts = []
        spoken = [p for p in items if p.text.strip()]
        self.parts = len(spoken)
        if not spoken:
            return None
        engines = list(dict.fromkeys(p.engine for p in spoken))
        return Transcript(
            text=" ".join(p.text.strip() for p in spoken),
            is_final=True,
            audio_s=sum(p.audio_s for p in items),
            latency_ms=(last or spoken[-1]).latency_ms,
            engine="+".join(engines),
        )
