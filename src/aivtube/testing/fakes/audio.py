"""Audio fakes: a PortAudio/``sounddevice`` stand-in plus AudioOut/AudioIn/AEC/VAD/Endpointer fakes.

``FakeSD`` is ported from the audio brief's ``test_audio_io.py``: the Windows host-API layout
(MME + WASAPI duplicates), ``pump(n)`` to drive output callbacks, ``push(x)`` to drive input
callbacks and ``die()`` to fire ``finished_callback``. It also models PortAudio's frozen device
list (``unplug``/``replug`` only show up after ``_terminate``/``_initialize``) and the WASAPI
``extra_settings`` host-API check, so device-resolution and reopen logic can be tested on Linux.
"""

from __future__ import annotations

import collections
import copy
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from aivtube.contracts.voice import F32, MarkCallback, VadEvent

__all__ = [
    "MME",
    "WASAPI",
    "FakeAudioIn",
    "FakeAudioOut",
    "FakeEchoCanceller",
    "FakeEndpointer",
    "FakePortAudioError",
    "FakeSD",
    "FakeStream",
    "FakeVAD",
    "FakeWasapiSettings",
    "dominant_frequency",
    "marker_tone",
    "resample_linear",
]

WASAPI = "Windows WASAPI"
MME = "MME"


class FakePortAudioError(Exception):
    """Mirrors ``sounddevice.PortAudioError``."""


class CallbackStop(Exception):
    """Mirrors ``sounddevice.CallbackStop``: finish after this block."""


class CallbackAbort(Exception):
    """Mirrors ``sounddevice.CallbackAbort``: stop at once."""


@dataclass
class FakeWasapiSettings:
    """What ``sd.WasapiSettings(...)`` returns; ``kw`` holds exactly the arguments passed."""

    kw: dict[str, Any]
    _streaminfo: SimpleNamespace = field(default_factory=SimpleNamespace)

    @property
    def auto_convert(self) -> bool:
        return bool(self.kw.get("auto_convert", False))

    @property
    def exclusive(self) -> bool:
        return bool(self.kw.get("exclusive", False))


def _status(**flags: bool) -> SimpleNamespace:
    base = dict(
        input_underflow=False,
        input_overflow=False,
        output_underflow=False,
        output_overflow=False,
        priming_output=False,
    )
    base.update(flags)
    return SimpleNamespace(**base)


class FakeStream:
    """A callback stream. ``pump``/``push`` run the user callback on the caller's thread."""

    def __init__(self, sd: FakeSD, kind: Literal["out", "in"], **kw: Any) -> None:
        self.sd, self.kind, self.kw = sd, kind, kw
        self.callback: Callable[..., None] = kw["callback"]
        self.finished: Callable[[], None] | None = kw.get("finished_callback")
        self.device: int = kw["device"]
        self.blocksize: int = int(kw.get("blocksize") or 480)
        self.channels: int = int(kw.get("channels") or 1)
        self.samplerate: float = float(kw.get("samplerate") or 48000)
        self.dtype: str = str(kw.get("dtype") or "float32")
        self.latency: float = sd._stream_latency(self.device, kind, kw.get("latency"))
        self.extra_settings = kw.get("extra_settings")
        self.active = False
        self.stopped = True
        self.closed = False
        self.t = 0.0  # stream clock (Pa_GetStreamTime), advanced per block
        self.blocks = 0
        self.last_output: npt.NDArray[Any] | None = None
        self._pending_in = np.zeros((0, self.channels), np.float32)

    # --- sounddevice stream API -------------------------------------------------------------
    def start(self) -> None:
        if self.closed:
            raise FakePortAudioError("Stream is closed")
        self.active, self.stopped = True, False

    def stop(self) -> None:
        self.active, self.stopped = False, True

    def abort(self) -> None:
        self.active, self.stopped = False, True

    def close(self) -> None:
        self.active, self.stopped, self.closed = False, True, True

    @property
    def time(self) -> float:
        return self.t

    @property
    def cpu_load(self) -> float:
        return 0.0

    def __enter__(self) -> FakeStream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- test drivers -----------------------------------------------------------------------
    def _time_info(self) -> SimpleNamespace:
        self.t += self.blocksize / self.samplerate
        return SimpleNamespace(
            currentTime=self.t,
            outputBufferDacTime=self.t + self.latency,
            inputBufferAdcTime=self.t - 0.01,
        )

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError(f"{self.kind} stream on device {self.device} is not active")

    def pump(self, n: int = 1, *, underflow: bool = False, all_channels: bool = False) -> F32:
        """Drive an output stream ``n`` blocks; returns channel 0 (or all channels) concatenated."""
        if self.kind != "out":
            raise TypeError("pump() drives output streams; use push() for input")
        self._require_active()
        outs: list[npt.NDArray[Any]] = []
        for _ in range(n):
            if not self.active:  # CallbackStop/Abort or die() inside the loop
                outs.append(np.zeros((self.blocksize, self.channels), np.float32))
                continue
            if self.dtype == "float32":
                out = np.full((self.blocksize, self.channels), np.nan, np.float32)
            else:
                out = np.zeros((self.blocksize, self.channels), np.dtype(self.dtype))
            stop = self._invoke(out, _status(output_underflow=underflow))
            if self.sd.strict and self.dtype == "float32" and np.isnan(out).any():
                raise AssertionError("output callback left part of outdata unwritten")
            self.blocks += 1
            self.last_output = out
            outs.append(out.astype(np.float32, copy=False))
            if stop is not None:
                self._finish(stop)
        full = np.concatenate(outs) if outs else np.zeros((0, self.channels), np.float32)
        return (full if all_channels else full[:, 0]).astype(np.float32, copy=True)

    def push(self, x: npt.ArrayLike, *, overflow: bool = False) -> int:
        """Feed input samples (mono or ``(frames, channels)``); whole blocks go to the callback.

        A trailing partial block is kept for the next call. Returns the number of callbacks.
        """
        if self.kind != "in":
            raise TypeError("push() drives input streams; use pump() for output")
        self._require_active()
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], self.channels, axis=1)
        elif arr.shape[1] != self.channels:
            arr = np.repeat(arr[:, :1], self.channels, axis=1)
        data = np.concatenate([self._pending_in, arr])
        n = 0
        bs = self.blocksize
        while data.shape[0] - n * bs >= bs and self.active:
            blk = np.ascontiguousarray(data[n * bs : (n + 1) * bs])
            stop = self._invoke(blk, _status(input_overflow=overflow))
            self.blocks += 1
            n += 1
            if stop is not None:
                self._finish(stop)
        self._pending_in = data[n * bs :].copy()
        return n

    def die(self) -> None:
        """Simulate device loss: the stream stops and ``finished_callback`` fires."""
        was_running = self.active
        self.active, self.stopped = False, True
        if was_running and self.finished is not None:
            self.finished()

    def _invoke(self, buf: npt.NDArray[Any], status: SimpleNamespace) -> type[Exception] | None:
        try:
            self.callback(buf, self.blocksize, self._time_info(), status)
        except CallbackStop:
            return CallbackStop
        except CallbackAbort:
            return CallbackAbort
        return None

    def _finish(self, _how: type[Exception]) -> None:
        self.active, self.stopped = False, True
        if self.finished is not None:
            self.finished()


@dataclass
class _Device:
    name: str
    hostapi: str
    max_in: int
    max_out: int
    default_samplerate: float = 48000.0
    refuse_mono: bool = False
    plugged: bool = True


class FakeSD:
    """``AudioBackend`` fake with the audio brief's Windows layout (MME and WASAPI duplicates).

    Default layout (index: name, host API):
    0 Microphone (USB Mic), MME · 1 Speakers (Realtek(R) Audio), MME · 2 CABLE Input (VB-Audio
    Virtual C, MME (31-char truncation) · 3 Microphone (USB Mic), WASAPI · 4 Speakers
    (Realtek(R) Audio), WASAPI · 5 CABLE Input (VB-Audio Virtual Cable), WASAPI (8 channels).
    MME defaults are 0/1 and WASAPI defaults 3/4.
    """

    PortAudioError = FakePortAudioError
    CallbackStop = CallbackStop
    CallbackAbort = CallbackAbort

    def __init__(self, layout: Literal["windows", "empty"] = "windows", *, strict: bool = True):
        self.strict = strict
        self.streams: list[FakeStream] = []
        self.stream_latency: float | None = None  # force every stream's .latency when set
        self.reinit_count = 0
        self.open_errors: collections.deque[Exception] = collections.deque()
        self.default = SimpleNamespace(device=[None, None], samplerate=None, channels=[None, None])
        self._registry: list[_Device] = []
        self._api_names: list[str] = []
        self._explicit_defaults: dict[tuple[str, str], str] = {}
        self._lib = SimpleNamespace(eAudioCategoryCommunications=3)
        self._libname = "fake-portaudio"
        self._ffi = SimpleNamespace(dlclose=lambda lib: None, dlopen=self._dlopen)
        if layout == "windows":
            for name, api, i, o in (
                ("Microphone (USB Mic)", MME, 1, 0),
                ("Speakers (Realtek(R) Audio)", MME, 0, 2),
                ("CABLE Input (VB-Audio Virtual C", MME, 0, 2),
                ("Microphone (USB Mic)", WASAPI, 1, 0),
                ("Speakers (Realtek(R) Audio)", WASAPI, 0, 2),
                ("CABLE Input (VB-Audio Virtual Cable)", WASAPI, 0, 8),
            ):
                self._registry.append(_Device(name, api, i, o))
        self._snapshot()

    # --- setup helpers ----------------------------------------------------------------------
    def add_device(
        self,
        name: str,
        hostapi: str,
        max_in: int,
        max_out: int,
        *,
        default_samplerate: float = 48000.0,
        default: bool = False,
        refuse_mono: bool = False,
    ) -> int:
        """Add a device (visible at once) and return its index. The first input/output device of
        a host API becomes that API's default, as does any device added with ``default=True``."""
        self._registry.append(
            _Device(name, hostapi, max_in, max_out, default_samplerate, refuse_mono)
        )
        if default:
            if max_in > 0:
                self._explicit_defaults[(hostapi, "in")] = name
            if max_out > 0:
                self._explicit_defaults[(hostapi, "out")] = name
        self._snapshot()
        return len(self._devices) - 1

    def unplug(self, name_substring: str) -> int:
        """Unplug matching devices: their streams die; the frozen list keeps them until re-init."""
        q = name_substring.lower()
        hit = 0
        for dev in self._registry:
            if q in dev.name.lower() and dev.plugged:
                dev.plugged = False
                hit += 1
        for st in self.streams:
            if st.active and q in self._devices[st.device].name.lower():
                st.die()
        return hit

    def replug(self, name_substring: str) -> int:
        q = name_substring.lower()
        hit = 0
        for dev in self._registry:
            if q in dev.name.lower() and not dev.plugged:
                dev.plugged = True
                hit += 1
        return hit

    def fail_next_open(self, exc: Exception | None = None, times: int = 1) -> None:
        for _ in range(times):
            self.open_errors.append(
                exc or FakePortAudioError("Error opening stream [PaErrorCode -9996]")
            )

    # --- sounddevice API --------------------------------------------------------------------
    def query_hostapis(self, index: int | None = None) -> Any:
        apis = tuple(copy.deepcopy(a) for a in self._apis)
        return apis if index is None else apis[index]

    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> Any:
        if kind not in (None, "input", "output"):
            raise ValueError(f"Invalid kind: {kind!r}")
        if device is None and kind is None:
            return tuple(self._info(i) for i in range(len(self._devices)))
        if device is None:
            api = self._apis[0] if self._apis else None
            idx = (
                api["default_input_device" if kind == "input" else "default_output_device"]
                if api
                else -1
            )
            if idx < 0:
                raise FakePortAudioError(f"Error querying device -1 (no default {kind} device)")
            return self._info(idx)
        idx = device if isinstance(device, int) else self._find(device, kind)
        if not 0 <= idx < len(self._devices):
            raise FakePortAudioError(f"Error querying device {idx}")
        info = self._info(idx)
        if kind == "input" and info["max_input_channels"] < 1:
            raise ValueError(f"Not an input device: {info['name']!r}")
        if kind == "output" and info["max_output_channels"] < 1:
            raise ValueError(f"Not an output device: {info['name']!r}")
        return info

    def WasapiSettings(self, **kw: Any) -> FakeWasapiSettings:
        allowed = {"exclusive", "auto_convert", "explicit_sample_format"}
        unknown = set(kw) - allowed
        if unknown:
            raise TypeError(f"unexpected WasapiSettings arguments: {sorted(unknown)}")
        return FakeWasapiSettings(kw=dict(kw))

    def OutputStream(self, **kw: Any) -> FakeStream:
        return self._open("out", kw)

    def InputStream(self, **kw: Any) -> FakeStream:
        return self._open("in", kw)

    def get_portaudio_version(self) -> tuple[int, str]:
        return 1246976, "PortAudio V19.7.0-devel (fake)"

    def _terminate(self) -> None:
        for st in self.streams:
            st.close()

    def _initialize(self) -> None:
        self.reinit_count += 1
        self._snapshot()

    # --- drivers across all streams ---------------------------------------------------------
    @property
    def output_streams(self) -> list[FakeStream]:
        return [s for s in self.streams if s.kind == "out"]

    @property
    def input_streams(self) -> list[FakeStream]:
        return [s for s in self.streams if s.kind == "in"]

    def pump(self, n_blocks: int = 1) -> list[F32]:
        """Drive every active output stream ``n_blocks``; one mono array per stream, in order."""
        return [s.pump(n_blocks) for s in self.output_streams if s.active]

    def push(self, x: npt.ArrayLike) -> None:
        """Feed ``x`` to every active input stream."""
        for s in self.input_streams:
            if s.active:
                s.push(x)

    def die(self) -> None:
        """Every active stream loses its device (``finished_callback`` fires)."""
        for s in self.streams:
            if s.active:
                s.die()

    # --- internals --------------------------------------------------------------------------
    def _dlopen(self, _name: str) -> SimpleNamespace:
        return SimpleNamespace(eAudioCategoryCommunications=3)

    def _snapshot(self) -> None:
        """Freeze the visible device list (PortAudio only rescans on re-initialisation)."""
        self._devices = [copy.copy(d) for d in self._registry if d.plugged]
        for d in self._registry:  # host APIs keep their first-seen order; MME stays first
            if d.hostapi not in self._api_names:
                self._api_names.append(d.hostapi)
        self._api_names.sort(key=lambda n: n != MME)
        names = self._api_names
        apis: list[dict[str, Any]] = []
        for api_name in names:
            idxs = [i for i, d in enumerate(self._devices) if d.hostapi == api_name]
            apis.append(
                dict(
                    name=api_name,
                    devices=idxs,
                    default_input_device=self._default_for(api_name, idxs, "in"),
                    default_output_device=self._default_for(api_name, idxs, "out"),
                )
            )
        self._apis = apis

    def _default_for(self, api_name: str, idxs: list[int], kind: str) -> int:
        wanted = self._explicit_defaults.get((api_name, kind))
        for i in idxs:
            d = self._devices[i]
            chans = d.max_in if kind == "in" else d.max_out
            if chans > 0 and (wanted is None or d.name == wanted):
                return i
        return -1

    def _info(self, idx: int) -> dict[str, Any]:
        d = self._devices[idx]
        return dict(
            name=d.name,
            index=idx,
            hostapi=self._api_names.index(d.hostapi),
            max_input_channels=d.max_in,
            max_output_channels=d.max_out,
            default_low_input_latency=0.01 if d.max_in else -1.0,
            default_low_output_latency=0.01 if d.max_out else -1.0,
            default_high_input_latency=0.04 if d.max_in else -1.0,
            default_high_output_latency=0.04 if d.max_out else -1.0,
            default_samplerate=d.default_samplerate,
        )

    def _find(self, query: str, kind: str | None) -> int:
        """sounddevice-style lookup: every whitespace-separated word must match name or API."""
        words = query.lower().split()
        hits = []
        for i, d in enumerate(self._devices):
            if kind == "input" and d.max_in < 1:
                continue
            if kind == "output" and d.max_out < 1:
                continue
            hay = f"{d.name}, {d.hostapi}".lower()
            if all(w in hay for w in words):
                hits.append(i)
        if not hits:
            raise ValueError(f"No {kind or ''} device matching {query!r}".replace("  ", " "))
        if len(hits) > 1:
            names = "\n".join(
                f"[{i}] {self._devices[i].name}, {self._devices[i].hostapi}" for i in hits
            )
            raise ValueError(f"Multiple {kind or ''} devices found for {query!r}:\n{names}")
        return hits[0]

    def _stream_latency(self, device: int, kind: str, requested: Any) -> float:
        if self.stream_latency is not None:
            return self.stream_latency
        if isinstance(requested, int | float):
            return float(requested)
        return 0.04 if requested == "high" else 0.03

    def _open(self, kind: Literal["out", "in"], kw: dict[str, Any]) -> FakeStream:
        if self.open_errors:
            raise self.open_errors.popleft()
        if "callback" not in kw:
            raise TypeError("FakeSD only supports callback streams")
        device = kw.get("device")
        if device is None:
            api = self._apis[0] if self._apis else None
            device = (
                api["default_output_device" if kind == "out" else "default_input_device"]
                if api
                else -1
            )
        elif isinstance(device, str):
            device = self._find(device, "output" if kind == "out" else "input")
        if not isinstance(device, int) or not 0 <= device < len(self._devices):
            raise FakePortAudioError(f"Error querying device {device}")
        dev = self._devices[device]
        registry_dev = next(
            (r for r in self._registry if r.name == dev.name and r.hostapi == dev.hostapi), None
        )
        if registry_dev is None or not registry_dev.plugged:
            raise FakePortAudioError(
                f"Error opening {kind}put stream: Device unavailable [PaErrorCode -9985]"
            )
        max_ch = dev.max_out if kind == "out" else dev.max_in
        channels = int(kw.get("channels") or 1)
        if max_ch < 1 or channels > max_ch:
            raise FakePortAudioError("Invalid number of channels [PaErrorCode -9998]")
        extra = kw.get("extra_settings")
        if isinstance(extra, FakeWasapiSettings) and dev.hostapi != WASAPI:
            raise FakePortAudioError(
                "Incompatible host API specific stream info [PaErrorCode -9984]"
            )
        auto = isinstance(extra, FakeWasapiSettings) and extra.auto_convert
        if dev.refuse_mono and channels == 1 and not auto:
            raise FakePortAudioError("Invalid number of channels [PaErrorCode -9998]")
        kw = dict(kw, device=device, channels=channels)
        st = FakeStream(self, kind, **kw)
        self.streams.append(st)
        return st


# --- AudioOut / AudioIn / AEC -------------------------------------------------------------------


def resample_linear(pcm: F32, src_rate: int, dst_rate: int) -> F32:
    """Cheap linear-interpolation resampler (tests only; the voice worker uses soxr)."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or x.size == 0:
        return x.copy()
    n_out = round(x.size * dst_rate / src_rate)
    t_out = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(t_out, np.arange(x.size, dtype=np.float64), x).astype(np.float32)


@dataclass
class _Mark:
    cb: MarkCallback


class FakeAudioOut:
    """``AudioOut`` simulated in memory; ``pump(n)`` plays ``n`` blocks like a device callback.

    Marks fire synchronously inside ``pump`` (``True`` at the audible time) or inside
    ``cancel``/``close`` (``False``), rather than on a notifier thread.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        blocksize: int = 480,
        output_latency_s: float = 0.04,
        clock: Callable[[], float] = time.perf_counter,
        reference_s: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.output_latency_s = output_latency_s
        self.reference: collections.deque[tuple[float, F32]] = collections.deque(
            maxlen=max(1, int(reference_s * sample_rate / blocksize))
        )
        self.stats: dict[str, int] = collections.Counter()
        self._clock = clock
        self._lock = threading.Lock()
        self._queue: collections.deque[F32 | _Mark] = collections.deque()
        self._gain = 1.0
        self._target_gain = 1.0
        self._gain_step = 0.0
        self._speaking_until = 0.0
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def play(self, pcm: F32, sample_rate: int) -> None:
        x = resample_linear(np.asarray(pcm, np.float32), sample_rate, self.sample_rate)
        with self._lock:
            if self.closed:
                return
            self._queue.append(x)
            self.stats["played_samples"] += int(x.size)

    def mark(self, cb: MarkCallback) -> None:
        with self._lock:
            if self.closed:
                cb(False, self._clock())
                return
            self._queue.append(_Mark(cb))

    def cancel(self, fade_ms: float = 30.0) -> float:
        fade_len = max(1, int(self.sample_rate * fade_ms / 1000.0))
        with self._lock:
            audio = [x for x in self._queue if not isinstance(x, _Mark)]
            marks = [x for x in self._queue if isinstance(x, _Mark)]
            total = sum(int(a.size) for a in audio)
            self._queue.clear()
            if total:
                head = np.concatenate(audio)[:fade_len]
                ramp = np.linspace(1.0, 0.0, head.size, endpoint=False, dtype=np.float32)
                self._queue.append((head * ramp).astype(np.float32))
            self.stats["cancels"] += 1
        now = self._clock()
        for m in marks:
            m.cb(False, now)
        return max(0, total - fade_len) / self.sample_rate

    def set_gain(self, gain: float, ramp_ms: float = 20.0) -> None:
        with self._lock:
            self._target_gain = float(gain)
            steps = max(1.0, self.sample_rate * ramp_ms / 1000.0)
            self._gain_step = (self._target_gain - self._gain) / steps

    def is_speaking(self, tail_s: float = 0.25) -> bool:
        with self._lock:
            queued = any(not isinstance(x, _Mark) and x.size for x in self._queue)
        return queued or self._clock() < self._speaking_until + tail_s

    def close(self) -> None:
        with self._lock:
            marks = [x for x in self._queue if isinstance(x, _Mark)]
            self._queue.clear()
            self.closed = True
        now = self._clock()
        for m in marks:
            m.cb(False, now)

    # --- test driver ------------------------------------------------------------------------
    def pump(self, n_blocks: int = 1) -> F32:
        """Render ``n_blocks`` as the device callback would; returns the mono output."""
        out = np.zeros(n_blocks * self.blocksize, np.float32)
        fired: list[tuple[MarkCallback, float]] = []
        with self._lock:
            now = self._clock()
            pos = 0
            while pos < out.size and self._queue:
                item = self._queue[0]
                if isinstance(item, _Mark):
                    self._queue.popleft()
                    fired.append((item.cb, now + self.output_latency_s + pos / self.sample_rate))
                    continue
                take = min(item.size, out.size - pos)
                out[pos : pos + take] = item[:take]
                if take == item.size:
                    self._queue.popleft()
                else:
                    self._queue[0] = item[take:]
                pos += take
            if pos:
                self._speaking_until = now + self.output_latency_s + pos / self.sample_rate
            # a mark at the very end of the queue fires when its preceding audio has played
            while self._queue and isinstance(self._queue[0], _Mark):
                m = self._queue.popleft()
                assert isinstance(m, _Mark)
                fired.append((m.cb, now + self.output_latency_s + pos / self.sample_rate))
            out *= self._gain_ramp(out.size)
            for b in range(n_blocks):
                blk = out[b * self.blocksize : (b + 1) * self.blocksize].copy()
                t_blk = now + self.output_latency_s + b * self.blocksize / self.sample_rate
                self.reference.append((t_blk, blk))
            self.stats["blocks"] += n_blocks
        for cb, t in fired:
            cb(True, t)
        return out

    def _gain_ramp(self, n: int) -> npt.NDArray[np.float32]:
        if self._gain == self._target_gain:
            return np.full(n, self._gain, np.float32)
        g = self._gain + self._gain_step * np.arange(1, n + 1, dtype=np.float64)
        g = (
            np.minimum(g, self._target_gain)
            if self._gain_step > 0
            else np.maximum(g, self._target_gain)
        )
        self._gain = float(g[-1])
        return g.astype(np.float32)


class FakeAudioIn:
    """``AudioIn`` fed by ``push``; frames are delivered synchronously with a capture time."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        block_samples: int = 480,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.stats: dict[str, int] = collections.Counter()
        self._clock = clock
        self._on_frame: Callable[[F32, float], None] | None = None
        self._pending = np.zeros(0, np.float32)
        self.closed = False

    def start(self, on_frame: Callable[[F32, float], None]) -> None:
        self._on_frame = on_frame

    def push(self, x: npt.ArrayLike, t0: float | None = None) -> int:
        """Deliver whole blocks of ``x``; ``t0`` is the capture time of the first new sample."""
        if self._on_frame is None or self.closed:
            raise RuntimeError("FakeAudioIn is not started")
        data = np.concatenate([self._pending, np.asarray(x, np.float32).reshape(-1)])
        start = self._clock() if t0 is None else t0
        start -= self._pending.size / self.sample_rate
        bs = self.block_samples
        n = data.size // bs
        for i in range(n):
            self._on_frame(data[i * bs : (i + 1) * bs].copy(), start + i * bs / self.sample_rate)
        self._pending = data[n * bs :].copy()
        self.stats["frames"] += n
        return n

    def close(self) -> None:
        self.closed = True


class FakeEchoCanceller:
    """``EchoCanceller``: subtracts ``leak`` times the oldest reference block (0 = pass-through)."""

    def __init__(self, leak: float = 0.0, *, max_blocks: int = 64) -> None:
        self.leak = leak
        self.refs: collections.deque[F32] = collections.deque(maxlen=max_blocks)
        self.processed = 0

    def feed_reference(self, block: F32) -> None:
        self.refs.append(np.asarray(block, np.float32).copy())

    def process(self, mic_block: F32) -> F32:
        self.processed += 1
        x = np.asarray(mic_block, np.float32)
        if self.leak and self.refs and self.refs[0].shape == x.shape:
            return (x - self.leak * self.refs.popleft()).astype(np.float32)
        return x.copy()


# --- VAD / endpointer ----------------------------------------------------------------------------


class FakeVAD:
    """``VoiceActivityDetector`` returning scripted probabilities, then an energy fallback.

    After the script runs out, a frame counts as speech (1.0) when its RMS is above
    ``energy_threshold_dbfs``, so marker tones in fake mic audio are detected as speech.
    """

    def __init__(
        self,
        probs: Sequence[float] = (),
        *,
        energy_threshold_dbfs: float = -40.0,
        sample_rate: int = 16000,
        frame_samples: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.script = list(probs)
        self.energy_threshold_dbfs = energy_threshold_dbfs
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def prob(self, frame: F32) -> float:
        x = np.asarray(frame, np.float32).reshape(-1)
        if x.size != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {x.size}")
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return float(self.script[i])
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
        dbfs = 20.0 * math.log10(rms + 1e-12)
        return 1.0 if dbfs > self.energy_threshold_dbfs else 0.0


class FakeEndpointer:
    """``Endpointer`` that returns scripted event lists, one list per ``push`` call."""

    def __init__(self, script: Sequence[Sequence[VadEvent]] = ()) -> None:
        self.script = [list(s) for s in script]
        self.pushes: list[tuple[int, float, bool]] = []
        self.tail_hints: list[str] = []
        self.resets = 0

    def push(self, frame16k: F32, t: float, ai_speaking: bool) -> list[VadEvent]:
        i = len(self.pushes)
        self.pushes.append((int(np.asarray(frame16k).size), t, ai_speaking))
        return self.script[i] if i < len(self.script) else []

    def set_tail_hint(self, text: str) -> None:
        self.tail_hints.append(text)

    def reset(self) -> None:
        self.resets += 1


# --- marker tones ---------------------------------------------------------------------------------


def marker_tone(
    freq_hz: float, seconds: float, *, sample_rate: int = 16000, amplitude: float = 0.3
) -> F32:
    """A pure tone; ``FakeRecognizer`` maps its frequency to a scripted sentence."""
    t = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def dominant_frequency(
    pcm: F32, sample_rate: int = 16000, *, min_rms: float = 1e-3
) -> float | None:
    """Frequency (Hz) of the strongest FFT bin, or ``None`` for (near) silence."""
    x = np.asarray(pcm, np.float64).reshape(-1)
    if x.size < 64 or float(np.sqrt(np.mean(x * x))) < min_rms:
        return None
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if 0 < k < spec.size - 1:  # parabolic interpolation for sub-bin accuracy
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        k_f = k + (0.5 * (a - c) / denom if denom else 0.0)
    else:
        k_f = float(k)
    return float(k_f * sample_rate / x.size)
