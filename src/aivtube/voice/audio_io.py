"""Audio device I/O: WASAPI device resolution, the always-open player and mic capture.

ARCHITECTURE.md §2.4 (voice threads), §2.5 (timing), §2.8 (device loss), §3.4 (``AudioOut`` /
``AudioIn``); ported from the audio brief's ``audio_io.py`` (docs/research/components/audio.md).

Rules this module keeps:
- Every PortAudio call goes through an injected ``AudioBackend`` (``FakeSD`` in tests). The
  real ``sounddevice`` is imported only when no backend is given, at ``start()``.
- Devices are resolved **by name through the WASAPI host API**, never through ``sd.default``
  (PortAudio's default host API on Windows is MME: ~90 ms latency, 31-char names). Only WASAPI
  devices get ``WasapiSettings(auto_convert=True)``; exclusive mode is never used.
- Every timestamp comes from the injected ``clock`` (``time.perf_counter`` by default), never
  ``time.monotonic()`` (15.6 ms resolution on Windows).
- PortAudio callbacks only copy data. They never raise (an exception would stop PortAudio from
  ever calling them again): errors zero-fill the block and are counted in ``stats``.
- Marks and levels are delivered on a notifier thread when the audio actually reaches the DAC
  (``outputBufferDacTime``), not from the audio callback.
- A lost device (``finished_callback`` or 1 s without callbacks) is reopened by name every 2 s.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Literal, cast

import numpy as np

from aivtube.contracts.voice import F32, AudioBackend, MarkCallback

__all__ = [
    "HOSTAPI_PREF",
    "WASAPI",
    "AudioDeviceError",
    "DeviceInfo",
    "MicCapture",
    "StreamingPlayer",
    "default_backend",
    "list_devices",
    "reinit_portaudio",
    "resolve_device",
]

log = logging.getLogger("aivtube.voice.audio")

WASAPI: Final = "Windows WASAPI"
HOSTAPI_PREF: Final[tuple[str, ...]] = (WASAPI, "MME", "Windows DirectSound")

_POLL_S: Final = 0.005  # notifier/consumer poll period
_SILENT_RMS: Final = 1e-4


class AudioDeviceError(RuntimeError):
    """The audio device could not be resolved or opened."""


def default_backend() -> AudioBackend:
    """The real ``sounddevice`` module (lazy: it raises ``OSError`` without PortAudio)."""
    import sounddevice

    return cast(AudioBackend, sounddevice)


def _safe(cb: Callable[..., object] | None, *args: object) -> None:
    """Run a user callback; never let it kill an audio or notifier thread."""
    if cb is None:
        return
    try:
        cb(*args)
    except Exception:
        log.exception("audio callback %r failed", cb)


# --- devices ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_in: int
    max_out: int
    default_sr: float


def list_devices(backend: AudioBackend | None = None) -> list[DeviceInfo]:
    """Every PortAudio device with its host-API name."""
    sd = backend or default_backend()
    apis = list(sd.query_hostapis())
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            DeviceInfo(
                index=i,
                name=str(d["name"]),
                hostapi=str(apis[int(d["hostapi"])]["name"]),
                max_in=int(d["max_input_channels"]),
                max_out=int(d["max_output_channels"]),
                default_sr=float(d["default_samplerate"]),
            )
        )
    return out


def resolve_device(
    query: str | int | None,
    kind: Literal["input", "output"],
    backend: AudioBackend | None = None,
    hostapis: tuple[str, ...] = HOSTAPI_PREF,
) -> int:
    """Resolve a device to a PortAudio index.

    - ``None`` or ``""``: the default endpoint of the first preferred host API that has one
      (the WASAPI default on Windows, never PortAudio's MME default).
    - ``str``: a case-insensitive name substring (``"CABLE Input"``, ``"Realtek"``), searched
      host API by host API in preference order; an exact name wins over a substring.
    - ``int``: used as-is (only for tools; config never stores numbers).

    When none of ``hostapis`` exists (Linux/macOS development), every host API is searched.
    Raises ``LookupError`` when nothing matches.
    """
    if isinstance(query, bool):
        raise TypeError("device must be a name, an index or None")
    if isinstance(query, int):
        return query
    q = (query or "").strip().casefold()
    sd = backend or default_backend()
    apis = list(sd.query_hostapis())
    devs = list_devices(sd)
    api_names = [str(a["name"]) for a in apis]
    order = [n for n in hostapis if n in api_names] or api_names
    for api_name in order:
        api = apis[api_names.index(api_name)]

        def usable(d: DeviceInfo, api_name: str = api_name) -> bool:
            chans = d.max_out if kind == "output" else d.max_in
            return d.hostapi == api_name and chans > 0

        if not q:
            key = "default_output_device" if kind == "output" else "default_input_device"
            idx = api.get(key)
            if isinstance(idx, int) and 0 <= idx < len(devs) and usable(devs[idx]):
                return idx
            continue
        hits = [d for d in devs if usable(d) and q in d.name.casefold()]
        if hits:
            exact = [d for d in hits if d.name.casefold() == q]
            if len(hits) > 1 and not exact:
                log.info(
                    "%s device %r matches %s; using %r",
                    kind,
                    query,
                    [d.name for d in hits],
                    hits[0].name,
                )
            return (exact or hits)[0].index
    what = "the default" if not q else f"a name containing {query!r}"
    raise LookupError(f"no {kind} device: {what} in host APIs {tuple(order)}")


def _extra_settings(sd: AudioBackend, device_index: int) -> Any:
    """``WasapiSettings(auto_convert=True)`` for WASAPI devices; ``None`` for the others.

    auto_convert lets Windows resample/remix when our rate or channel count differs from the
    endpoint mix format (shared mode). Other host APIs reject WASAPI settings.
    """
    info = sd.query_devices(device_index)
    api = sd.query_hostapis(int(info["hostapi"]))
    if str(api["name"]) != WASAPI:
        return None
    return sd.WasapiSettings(auto_convert=True)


def reinit_portaudio(backend: AudioBackend | None = None) -> None:
    """Refresh PortAudio's frozen device list (hot-plug).

    Closes and invalidates **every** stream in the process, so the owner must reopen the player
    and the mic afterwards. Uses private ``sounddevice`` API (issue #516: terminate, reload the
    DLL, initialise).
    """
    sd: Any = backend or default_backend()
    sd._terminate()
    try:
        sd._ffi.dlclose(sd._lib)
        sd._lib = sd._ffi.dlopen(sd._libname)
    except Exception:
        log.warning("PortAudio library reload failed; plain re-initialisation only", exc_info=True)
    sd._initialize()


def _close_stream(stream: Any) -> None:
    if stream is None:
        return
    with contextlib.suppress(Exception):
        stream.abort()
    with contextlib.suppress(Exception):
        stream.close()


def _dac_delay(time_info: Any, fallback: float) -> float:
    """Seconds from now until the block reaches the DAC (``outputBufferDacTime``).

    MME/DirectSound may report 0; fall back to the stream latency then.
    """
    try:
        dac = float(time_info.outputBufferDacTime)
        cur = float(time_info.currentTime)
    except (AttributeError, TypeError, ValueError):
        return fallback
    if dac > 0.0 and dac >= cur and dac - cur < 1.0:
        return dac - cur
    return fallback


# --- player -------------------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class _Chunk:
    pcm: F32  # mono float32 at the device rate
    pos: int = 0


@dataclass(slots=True, eq=False)
class _Mark:
    cb: MarkCallback


_Event = tuple[float, MarkCallback, bool]  # (t_audible, callback, heard)


class _Mirror:
    """A second sink (e.g. VB-CABLE) that plays the main player's post-gain blocks.

    It has its own clock, so blocks go through a small jitter buffer: consumption starts once
    ``prime`` blocks are queued, a backlog above ``cap`` blocks is trimmed (drift), and an empty
    buffer plays silence and re-primes. Never used as the AEC reference.
    """

    def __init__(self, owner: StreamingPlayer, device: str, *, prime: int = 2, cap: int = 10):
        self.owner = owner
        self.device = device
        self.stream: Any = None
        self.gen = 0
        self.lost = False
        self.next_try = 0.0
        self.ok = False
        self._lock = threading.Lock()
        self._fifo: collections.deque[F32] = collections.deque()
        self._head_pos = 0
        self._queued = 0
        self._primed = False
        self._prime = prime * owner._blocksize
        self._cap = cap * owner._blocksize

    def open(self) -> None:
        o = self.owner
        sd = o._backend()
        idx = resolve_device(self.device, "output", sd)
        self.gen += 1
        stream = sd.OutputStream(
            device=idx,
            samplerate=o._samplerate,
            channels=o._channels,
            dtype="float32",
            blocksize=o._blocksize,
            latency=o._latency,
            extra_settings=_extra_settings(sd, idx),
            callback=self._callback,
            finished_callback=partial(self._on_finished, self.gen),
        )
        try:
            stream.start()
        except Exception:
            _close_stream(stream)
            raise
        if o._closing:  # the player was closed while we were opening
            _close_stream(stream)
            raise AudioDeviceError("player closed")
        self.stream = stream
        self.lost = False
        self.ok = True
        with self._lock:
            self._fifo.clear()
            self._head_pos = self._queued = 0
            self._primed = False
        log.info("mirror output %r opened (device %d)", self.device, idx)

    def close(self) -> None:
        stream, self.stream = self.stream, None
        self.gen += 1
        _close_stream(stream)

    def _on_finished(self, gen: int) -> None:
        if gen == self.gen and not self.owner._closing:
            self.lost = True

    def push(self, block: F32) -> None:
        """Main audio thread: queue one post-gain block."""
        if self.stream is None:
            return
        with self._lock:
            self._fifo.append(block)
            self._queued += block.size
            if self._queued > self._cap:
                while self._queued - self._head_pos > self._prime and len(self._fifo) > 1:
                    old = self._fifo.popleft()
                    self._queued -= old.size
                    self._head_pos = 0
                self.owner._stats["mirror_drop"] += 1

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        try:
            outdata.fill(0)
            with self._lock:
                avail = self._queued - self._head_pos
                if not self._primed:
                    if avail < self._prime:
                        return
                    self._primed = True
                n = 0
                while n < frames and self._fifo:
                    head = self._fifo[0]
                    take = min(frames - n, head.size - self._head_pos)
                    outdata[n : n + take] = head[self._head_pos : self._head_pos + take, None]
                    n += take
                    self._head_pos += take
                    if self._head_pos >= head.size:
                        self._fifo.popleft()
                        self._queued -= head.size
                        self._head_pos = 0
                if n < frames:
                    self._primed = False
                    self.owner._stats["mirror_underflow"] += 1
        except Exception:
            with contextlib.suppress(Exception):
                outdata.fill(0)
            self.owner._stats["mirror_cb_error"] += 1


class StreamingPlayer:
    """Always-open callback ``OutputStream`` fed from a lock-protected queue (``AudioOut``).

    - ``play(pcm, sr)``: float32 mono at any rate; resampled to the device rate with a
      streaming soxr resampler (flushed at every ``mark`` and whenever the queue runs dry).
    - ``mark(cb)``: ``cb(True, t)`` when everything queued before it is audible (``t`` =
      perf_counter at the DAC); ``cb(False, t)`` if it is cancelled first.
    - ``cancel(fade_ms)``: the next block starts a fade to silence; the queue is dropped.
    - ``set_gain(gain, ramp_ms)``: linear per-sample ramp (ducking, mute).
    - ``reference``: ``(t_audible, post-gain block)`` pairs for AEC and double-talk detection.
    - ``stats``: ``underflow``, ``cb_error``, ``cancel``, ``dropped``, ``device_lost``,
      ``reopened``, ``reopen_failed`` and the ``mirror_*`` counters.

    ``fade_ms`` (constructor) is the minimum fade any ``cancel`` uses (de-click). A
    ``mirror_device`` gets the same post-gain blocks and is never the AEC reference.
    """

    def __init__(
        self,
        device: str | int | None = None,
        *,
        samplerate: int = 48000,
        channels: int = 2,
        blocksize: int = 480,
        latency: float = 0.04,
        fade_ms: float = 8.0,
        on_level: Callable[[float], None] | None = None,
        on_device_lost: Callable[[], None] | None = None,
        mirror_device: str | None = None,
        backend: AudioBackend | None = None,
        clock: Callable[[], float] = time.perf_counter,
        level_hz: float = 60.0,
        reference_s: float = 3.0,
        stall_timeout_s: float = 1.0,
        reopen_interval_s: float = 2.0,
    ) -> None:
        if samplerate <= 0 or blocksize <= 0 or channels <= 0:
            raise ValueError("samplerate, blocksize and channels must be positive")
        self._device = device
        self._samplerate = int(samplerate)
        self._channels = int(channels)
        self._blocksize = int(blocksize)
        self._latency = float(latency)
        self._min_fade = max(1, round(samplerate * max(0.0, fade_ms) / 1000.0))
        self._on_level = on_level
        self._on_device_lost = on_device_lost
        self._sd = backend
        self._clock = clock
        self._level_period = 1.0 / level_hz if level_hz > 0 else 0.0
        self._stall_timeout = stall_timeout_s
        self._reopen_interval = reopen_interval_s

        self._reference: collections.deque[tuple[float, F32]] = collections.deque(
            maxlen=max(1, round(reference_s * samplerate / blocksize))
        )
        self._stats: collections.Counter[str] = collections.Counter()
        self._lock = threading.Lock()  # queue, fade, gain, events
        self._rs_lock = threading.Lock()  # resampler; order: _rs_lock, then _lock
        self._q: collections.deque[_Chunk | _Mark] = collections.deque()
        self._queued = 0  # samples in _q
        self._fade: F32 | None = None
        self._rs: Any = None
        self._rs_sr = 0
        self._rs_pending = False
        self._gain = 1.0
        self._gain_target = 1.0
        self._gain_step = 0.0
        cap = self._blocksize * 8
        self._mix = np.zeros(cap, np.float32)
        self._gbuf = np.zeros(cap, np.float32)
        self._ramp = np.arange(1, cap + 1, dtype=np.float32)
        self._events: collections.deque[_Event] = collections.deque()
        self._last_event_t = float("-inf")
        self._levels: collections.deque[tuple[float, float]] = collections.deque()
        self._speaking_until = float("-inf")
        self._last_ref_t = float("-inf")
        self._out_latency = self._latency
        self._stream: Any = None
        self._gen = 0
        self._device_index: int | None = None
        self._last_cb = 0.0
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._closing = False
        self._closed = False
        self._device_ok = False
        self._next_try = 0.0
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._mirror = _Mirror(self, mirror_device) if mirror_device else None

    # --- AudioOut data members --------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._samplerate

    @property
    def output_latency_s(self) -> float:
        return self._out_latency

    @property
    def reference(self) -> collections.deque[tuple[float, F32]]:
        return self._reference

    @property
    def stats(self) -> collections.Counter[str]:
        return self._stats

    # --- extra state for the worker's health report -------------------------------------------
    @property
    def device_ok(self) -> bool:
        """False while the output device is lost and being reopened."""
        return self._device_ok

    @property
    def device_index(self) -> int | None:
        return self._device_index

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def gain(self) -> float:
        """The gain the ramp is heading to."""
        return self._gain_target

    # --- lifecycle ----------------------------------------------------------------------------
    def start(self) -> None:
        """Open the device and start the notifier thread. Raises ``AudioDeviceError``."""
        if self._closed:
            raise AudioDeviceError("player is closed")
        if self._started:
            return
        try:
            self._open()
        except Exception as exc:
            self._last_error = str(exc)
            raise AudioDeviceError(f"cannot open output device {self._device!r}: {exc}") from exc
        if self._mirror is not None:
            try:
                self._mirror.open()
            except Exception as exc:
                log.warning("mirror output %r unavailable: %s", self._mirror.device, exc)
                self._mirror.next_try = self._clock() + self._reopen_interval
        self._started = True
        self._thread = threading.Thread(target=self._notifier, name="player-notify", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the stream(s) and the notifier; pending marks fire ``False``. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._closing = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        stream, self._stream = self._stream, None
        _close_stream(stream)
        if self._mirror is not None:
            self._mirror.close()
        now = self._clock()
        with self._lock:
            marks = [m.cb for m in self._q if isinstance(m, _Mark)]
            self._q.clear()
            self._queued = 0
            self._fade = None
            events = list(self._events)
            self._events.clear()
        for cb in [e[1] for e in events] + marks:
            _safe(cb, False, now)

    def __enter__(self) -> StreamingPlayer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _backend(self) -> AudioBackend:
        if self._sd is None:
            self._sd = default_backend()
        return self._sd

    def _open(self) -> None:
        sd = self._backend()
        idx = resolve_device(self._device, "output", sd)
        self._gen += 1
        stream = sd.OutputStream(
            device=idx,
            samplerate=self._samplerate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            latency=self._latency,
            extra_settings=_extra_settings(sd, idx),
            callback=self._callback,
            finished_callback=partial(self._on_finished, self._gen),
        )
        try:
            stream.start()
        except Exception:
            _close_stream(stream)
            raise
        if self._closing:  # close() ran while we were (re)opening: never leak the stream
            _close_stream(stream)
            raise AudioDeviceError("player closed")
        try:
            self._out_latency = float(stream.latency)
        except (AttributeError, TypeError, ValueError):
            self._out_latency = self._latency
        self._stream = stream
        self._device_index = idx
        self._last_cb = self._clock()
        self._lost.clear()
        self._device_ok = True
        self._last_error = None
        log.info("output device %d opened, latency %.1f ms", idx, self._out_latency * 1e3)

    def _on_finished(self, gen: int) -> None:
        # PortAudio thread: never call PortAudio from here; just flag it.
        if gen == self._gen and not self._closing:
            self._lost.set()

    # --- producer API (any thread) --------------------------------------------------------------
    def play(self, pcm: F32, sample_rate: int) -> None:
        """Queue mono float32 audio (a 2-D array is mixed down); resampled to the device rate."""
        x = np.asarray(pcm, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1, dtype=np.float32) if x.shape[1] > 1 else x[:, 0]
        x = np.array(x.reshape(-1), dtype=np.float32)  # private copy: callers reuse buffers
        if x.size and not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if self._closed:
            self._stats["dropped"] += 1
            return
        if sample_rate != self._samplerate:
            import soxr

            with self._rs_lock:
                if self._rs is None or self._rs_sr != sample_rate:
                    self._flush_resampler_locked()
                    self._rs = soxr.ResampleStream(
                        sample_rate, self._samplerate, 1, dtype="float32", quality="HQ"
                    )
                    self._rs_sr = sample_rate
                y = np.asarray(self._rs.resample_chunk(x), dtype=np.float32).reshape(-1)
                self._rs_pending = True
                self._append(y)
            return
        with self._rs_lock:  # keep order with audio still inside a resampler
            self._flush_resampler_locked()
            self._append(x)

    def _append(self, x: F32) -> None:
        if not x.size:
            return
        with self._lock:
            self._q.append(_Chunk(x))
            self._queued += int(x.size)

    def _flush_resampler_locked(self) -> None:
        """Push the resampler's held-back tail into the queue (caller holds ``_rs_lock``)."""
        rs, self._rs = self._rs, None
        if rs is None or not self._rs_pending:
            self._rs_pending = False
            return
        self._rs_pending = False
        tail = np.asarray(rs.resample_chunk(np.zeros(0, np.float32), last=True), np.float32)
        self._append(np.ascontiguousarray(tail.reshape(-1)))

    def mark(self, cb: MarkCallback) -> None:
        with self._rs_lock:
            self._flush_resampler_locked()
            with self._lock:
                if not self._closed:
                    self._q.append(_Mark(cb))
                    return
        _safe(cb, False, self._clock())

    def cancel(self, fade_ms: float = 30.0) -> float:
        """Fade out from the next block and drop the queue; pending marks fire ``False``.

        Returns the seconds of queued audio that were dropped (the faded part included).
        """
        fade_len = max(self._min_fade, round(self._samplerate * max(0.0, fade_ms) / 1000.0))
        with self._rs_lock:
            self._rs = None
            self._rs_pending = False
        now = self._clock()
        with self._lock:
            parts: list[F32] = []
            need = fade_len
            marks: list[MarkCallback] = []
            for item in self._q:
                if isinstance(item, _Mark):
                    marks.append(item.cb)
                elif need > 0:
                    rem = item.pcm[item.pos : item.pos + need]
                    parts.append(rem)
                    need -= rem.size
            dropped = self._queued
            self._q.clear()
            self._queued = 0
            if parts and self._fade is None:
                head = np.concatenate(parts)
                ramp = np.linspace(1.0, 0.0, head.size, endpoint=False, dtype=np.float32)
                self._fade = (head * ramp).astype(np.float32)
            t_false = max(now, self._last_event_t)
            for cb in marks:
                self._events.append((t_false, cb, False))
            self._last_event_t = t_false
            self._speaking_until = min(
                self._speaking_until, now + self._out_latency + fade_len / self._samplerate
            )
        self._stats["cancel"] += 1
        return dropped / self._samplerate

    def set_gain(self, gain: float, ramp_ms: float = 20.0) -> None:
        """Ramp linearly to ``gain`` (0..4) over ``ramp_ms``, starting at the next block."""
        g = float(gain)
        if not np.isfinite(g):
            raise ValueError("gain must be finite")
        g = min(4.0, max(0.0, g))
        with self._lock:
            self._gain_target = g
            steps = max(1.0, self._samplerate * max(0.0, ramp_ms) / 1000.0)
            self._gain_step = (g - self._gain) / steps

    def is_speaking(self, tail_s: float = 0.25) -> bool:
        with self._lock:
            pending = self._queued > 0 or self._fade is not None
        return pending or self._clock() < self._speaking_until + tail_s

    def queued_s(self) -> float:
        """Seconds of audio waiting in the queue (not yet handed to the device)."""
        return self._queued / self._samplerate

    def ref_rms_max(self, t0: float, t1: float) -> float:
        """Max block RMS of the reference audible within ``[t0, t1]`` (perf_counter)."""
        best = 0.0
        block_s = self._blocksize / self._samplerate
        for t, blk in reversed(tuple(self._reference)):
            if t + block_s < t0:
                break
            if t <= t1 and blk.size:
                best = max(best, float(np.sqrt(np.dot(blk, blk) / blk.size)))
        return best

    # --- audio thread ---------------------------------------------------------------------------
    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        try:
            now = self._clock()
            if getattr(status, "output_underflow", False):
                self._stats["underflow"] += 1
            if self._rs_pending and self._queued < frames and self._rs_lock.acquire(False):
                try:  # the queue is running dry: play the resampler's held-back tail now
                    self._flush_resampler_locked()
                finally:
                    self._rs_lock.release()
            if frames > self._mix.size:
                self._mix = np.zeros(frames, np.float32)
                self._gbuf = np.zeros(frames, np.float32)
                self._ramp = np.arange(1, frames + 1, dtype=np.float32)
            buf = self._mix[:frames]
            buf.fill(0.0)
            t0 = now + _dac_delay(time_info, self._out_latency)
            sr = self._samplerate
            gain_vec: F32 | None = None
            with self._lock:
                n = 0
                if self._fade is not None:
                    k = min(frames, self._fade.size)
                    buf[:k] = self._fade[:k]
                    self._fade = self._fade[k:] if k < self._fade.size else None
                    n = frames  # the rest of this block stays silent
                q = self._q
                while n < frames and q:
                    item = q[0]
                    if isinstance(item, _Mark):
                        q.popleft()
                        self._push_event(t0 + n / sr, item.cb)
                        continue
                    take = min(frames - n, item.pcm.size - item.pos)
                    buf[n : n + take] = item.pcm[item.pos : item.pos + take]
                    item.pos += take
                    n += take
                    self._queued -= take
                    if item.pos >= item.pcm.size:
                        q.popleft()
                while q and isinstance(q[0], _Mark):  # marks right at the end of this block
                    mark = q.popleft()
                    assert isinstance(mark, _Mark)
                    self._push_event(t0 + n / sr, mark.cb)
                gain = self._gain
                if gain != self._gain_target:
                    g = self._gbuf[:frames]
                    np.multiply(self._ramp[:frames], self._gain_step, out=g)
                    g += gain
                    if self._gain_step > 0:
                        np.minimum(g, self._gain_target, out=g)
                    else:
                        np.maximum(g, self._gain_target, out=g)
                    self._gain = float(g[-1])
                    gain_vec = g
            if gain_vec is not None:
                buf *= gain_vec
            elif gain != 1.0:
                buf *= gain
            outdata[:] = buf[:, None]
            rms = float(np.sqrt(np.dot(buf, buf) / frames)) if frames else 0.0
            if rms > _SILENT_RMS:
                self._speaking_until = t0 + frames / sr
            if self._on_level is not None:
                self._levels.append((t0, rms))
            block = buf.copy()
            t_ref = t0 if t0 > self._last_ref_t else self._last_ref_t + 1e-6
            self._last_ref_t = t_ref
            self._reference.append((t_ref, block))
            if self._mirror is not None:
                self._mirror.push(block)
            self._last_cb = now
        except Exception:
            with contextlib.suppress(Exception):
                outdata.fill(0)
            self._stats["cb_error"] += 1

    def _push_event(self, t: float, cb: MarkCallback) -> None:
        """Queue a heard mark (caller holds ``_lock``); keeps event times monotonic."""
        t = max(t, self._last_event_t)
        self._last_event_t = t
        self._events.append((t, cb, True))

    # --- notifier / watchdog thread -------------------------------------------------------------
    def _notifier(self) -> None:
        peak, have_level, next_level = 0.0, False, 0.0
        while not self._stop.is_set():
            now = self._clock()
            events = self._events
            while events and events[0][0] <= now:
                t, cb, heard = events.popleft()
                _safe(cb, heard, t)
            levels = self._levels
            while levels and levels[0][0] <= now:
                peak = max(peak, levels.popleft()[1])
                have_level = True
            if have_level and now >= next_level:
                _safe(self._on_level, peak)
                peak, have_level, next_level = 0.0, False, now + self._level_period
            try:
                self._watchdog(now)
            except Exception:
                log.exception("player watchdog failed")
            wait = _POLL_S
            if events:
                wait = min(wait, max(0.0005, events[0][0] - now))
            time.sleep(wait)  # high-resolution on Windows (3.11+), unlike Event.wait

    def _watchdog(self, now: float) -> None:
        stream = self._stream
        lost = self._lost.is_set()
        stalled = stream is not None and now - self._last_cb > self._stall_timeout
        if (lost or stalled or stream is None) and now >= self._next_try and not self._closing:
            if self._device_ok:
                self._device_ok = False
                self._stats["device_lost"] += 1
                why = "finished" if lost else ("stalled" if stalled else "closed")
                log.warning("output device lost (%s); reopening %r", why, self._device)
                _safe(self._on_device_lost)
            self._stream = None
            self._gen += 1  # late finished_callbacks of the old stream are ignored
            _close_stream(stream)
            try:
                self._open()
            except Exception as exc:
                self._last_error = str(exc)
                self._stats["reopen_failed"] += 1
                self._next_try = now + self._reopen_interval
                log.debug("output reopen failed: %s", exc)
            else:
                self._stats["reopened"] += 1
        m = self._mirror
        if m is not None and (m.lost or m.stream is None) and now >= m.next_try:
            if m.ok:
                m.ok = False
                self._stats["mirror_lost"] += 1
                log.warning("mirror output %r lost; reopening", m.device)
            m.close()
            try:
                m.open()
            except Exception as exc:
                m.next_try = now + self._reopen_interval
                log.debug("mirror reopen failed: %s", exc)


# --- mic capture ----------------------------------------------------------------------------------


class MicCapture:
    """Callback ``InputStream`` → bounded queue → mic thread → ``on_frame(block, t_capture)``.

    48 kHz / 10 ms blocks by default, so mic frames line up 1:1 with the 48 kHz player for AEC.
    The PortAudio callback only copies the block into a queue; the mic thread (the consumer)
    runs ``on_frame`` (the front-end: AEC, resampling, VAD). ``t_capture`` is the perf_counter
    time of the block's first sample at the ADC.

    ``stats``: ``frames``, ``overflow``, ``dropped`` (queue full), ``frame_error``,
    ``device_lost``, ``reopened``, ``reopen_failed``.
    """

    def __init__(
        self,
        device: str | int | None = None,
        *,
        samplerate: int = 48000,
        blocksize: int = 480,
        latency: float | str = "low",
        backend: AudioBackend | None = None,
        max_queue_s: float = 2.0,
        clock: Callable[[], float] = time.perf_counter,
        on_device_lost: Callable[[], None] | None = None,
        stall_timeout_s: float = 1.0,
        reopen_interval_s: float = 2.0,
    ) -> None:
        if samplerate <= 0 or blocksize <= 0:
            raise ValueError("samplerate and blocksize must be positive")
        self._device = device
        self._samplerate = int(samplerate)
        self._blocksize = int(blocksize)
        self._latency = latency
        self._sd = backend
        self._clock = clock
        self._on_device_lost = on_device_lost
        self._stall_timeout = stall_timeout_s
        self._reopen_interval = reopen_interval_s
        self._max_blocks = max(1, round(max_queue_s * samplerate / blocksize))
        self._q: queue.SimpleQueue[tuple[float, F32]] = queue.SimpleQueue()
        self._stats: collections.Counter[str] = collections.Counter()
        self._on_frame: Callable[[F32, float], None] | None = None
        self._stream: Any = None
        self._gen = 0
        self._channels = 1
        self._device_index: int | None = None
        self._last_cb = 0.0
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._closing = False
        self._closed = False
        self._device_ok = False
        self._next_try = 0.0
        self._last_error: str | None = None
        self._errors_logged = 0
        self._thread: threading.Thread | None = None

    @property
    def sample_rate(self) -> int:
        return self._samplerate

    @property
    def block_samples(self) -> int:
        return self._blocksize

    @property
    def stats(self) -> collections.Counter[str]:
        return self._stats

    @property
    def device_ok(self) -> bool:
        return self._device_ok

    @property
    def device_index(self) -> int | None:
        return self._device_index

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self, on_frame: Callable[[F32, float], None]) -> None:
        """Open the device and start the mic thread. Raises ``AudioDeviceError``."""
        if self._closed:
            raise AudioDeviceError("mic is closed")
        if self._thread is not None:
            raise RuntimeError("MicCapture already started")
        self._on_frame = on_frame
        try:
            self._open()
        except Exception as exc:
            self._last_error = str(exc)
            raise AudioDeviceError(f"cannot open input device {self._device!r}: {exc}") from exc
        self._thread = threading.Thread(target=self._consumer, name="mic", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closing = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        stream, self._stream = self._stream, None
        _close_stream(stream)

    def __enter__(self) -> MicCapture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _backend(self) -> AudioBackend:
        if self._sd is None:
            self._sd = default_backend()
        return self._sd

    def _open(self) -> None:
        sd = self._backend()
        idx = resolve_device(self._device, "input", sd)
        extra = _extra_settings(sd, idx)
        self._gen += 1
        last_err: Exception | None = None
        stream: Any = None
        for ch in (1, 2):  # some endpoints refuse mono without auto-convert
            try:
                stream = sd.InputStream(
                    device=idx,
                    samplerate=self._samplerate,
                    channels=ch,
                    dtype="float32",
                    blocksize=self._blocksize,
                    latency=self._latency,
                    extra_settings=extra,
                    callback=self._callback,
                    finished_callback=partial(self._on_finished, self._gen),
                )
            except Exception as exc:
                last_err = exc
                continue
            self._channels = ch
            break
        if stream is None:
            assert last_err is not None
            raise last_err
        try:
            stream.start()
        except Exception:
            _close_stream(stream)
            raise
        if self._closing:  # close() ran while we were (re)opening: never leak the stream
            _close_stream(stream)
            raise AudioDeviceError("mic closed")
        self._stream = stream
        self._device_index = idx
        self._last_cb = self._clock()
        self._lost.clear()
        self._device_ok = True
        self._last_error = None
        log.info("input device %d opened (%d ch)", idx, self._channels)

    def _on_finished(self, gen: int) -> None:
        if gen == self._gen and not self._closing:
            self._lost.set()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        try:
            now = self._clock()
            if getattr(status, "input_overflow", False):
                self._stats["overflow"] += 1
            try:
                adc = float(time_info.inputBufferAdcTime)
                cur = float(time_info.currentTime)
            except (AttributeError, TypeError, ValueError):
                adc = cur = 0.0
            if adc > 0.0 and cur >= adc and cur - adc < 1.0:
                t_cap = now - (cur - adc)
            else:
                t_cap = now - frames / self._samplerate
            if self._channels == 1:
                x = np.array(indata[:, 0], dtype=np.float32)
            else:
                x = indata.mean(axis=1, dtype=np.float32)
            if self._q.qsize() >= self._max_blocks:
                self._stats["dropped"] += 1
            else:
                self._q.put_nowait((t_cap, x))
            self._last_cb = now
        except Exception:
            self._stats["cb_error"] += 1

    def _consumer(self) -> None:
        while not self._stop.is_set():
            try:
                t, x = self._q.get(timeout=0.05)
            except queue.Empty:
                pass
            else:
                self._deliver(x, t)
            try:
                self._watchdog(self._clock())
            except Exception:
                log.exception("mic watchdog failed")

    def _deliver(self, x: F32, t: float) -> None:
        cb = self._on_frame
        if cb is None:
            return
        self._stats["frames"] += 1
        try:
            cb(x, t)
        except Exception:
            self._stats["frame_error"] += 1
            if self._errors_logged < 3 or self._stats["frame_error"] % 1000 == 0:
                self._errors_logged += 1
                log.exception("mic frame handler failed (%d so far)", self._stats["frame_error"])

    def _watchdog(self, now: float) -> None:
        stream = self._stream
        lost = self._lost.is_set()
        stalled = stream is not None and now - self._last_cb > self._stall_timeout
        if not (lost or stalled or stream is None) or now < self._next_try or self._closing:
            return
        if self._device_ok:
            self._device_ok = False
            self._stats["device_lost"] += 1
            log.warning("input device lost; reopening %r", self._device)
            _safe(self._on_device_lost)
        self._stream = None
        self._gen += 1
        _close_stream(stream)
        try:
            self._open()
        except Exception as exc:
            self._last_error = str(exc)
            self._stats["reopen_failed"] += 1
            self._next_try = now + self._reopen_interval
            log.debug("input reopen failed: %s", exc)
        else:
            self._stats["reopened"] += 1
