"""FakeSD reproduces the audio brief's tested PortAudio behaviours (pump/push/die, WASAPI
host-API listing, WasapiSettings, device loss and re-initialisation)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from aivtube.testing.fakes import MME, WASAPI, FakePortAudioError, FakeSD, FakeWasapiSettings


def resolve(sd: FakeSD, query: str | None, kind: str) -> int:
    """The audio brief's resolve_device(): WASAPI first, by default endpoint or name substring."""
    apis = list(sd.query_hostapis())
    key = "max_output_channels" if kind == "output" else "max_input_channels"
    for api_name in (WASAPI, MME):
        api_idx = next(i for i, a in enumerate(apis) if a["name"] == api_name)
        if query is None:
            idx = apis[api_idx][
                "default_output_device" if kind == "output" else "default_input_device"
            ]
            if idx >= 0:
                return int(idx)
            continue
        for d in sd.query_devices():
            if d["hostapi"] == api_idx and d[key] > 0 and query.lower() in d["name"].lower():
                return int(d["index"])
    raise LookupError(query)


class Player:
    """Minimal always-open player: queue -> callback, zero-fill when idle."""

    def __init__(self, sd: FakeSD, device: int) -> None:
        self.queue = np.zeros(0, np.float32)
        self.finished = 0
        self.times: list[SimpleNamespace] = []
        extra = sd.WasapiSettings(auto_convert=True)
        self.stream = sd.OutputStream(
            device=device,
            samplerate=48000,
            channels=2,
            dtype="float32",
            blocksize=480,
            latency=0.04,
            extra_settings=extra,
            callback=self.callback,
            finished_callback=self.on_finished,
        )
        self.stream.start()

    def callback(self, out: Any, frames: int, time_info: Any, status: Any) -> None:
        n = min(frames, self.queue.size)
        out[:] = 0.0
        out[:n, 0] = self.queue[:n]
        out[:n, 1] = self.queue[:n]
        self.queue = self.queue[n:]
        self.times.append(time_info)

    def on_finished(self) -> None:
        self.finished += 1


def test_windows_layout_prefers_wasapi() -> None:
    sd = FakeSD()
    names = [a["name"] for a in sd.query_hostapis()]
    assert names == [MME, WASAPI]
    assert resolve(sd, None, "output") == 4  # WASAPI default, not MME's 1
    assert resolve(sd, None, "input") == 3
    assert resolve(sd, "cable input", "output") == 5  # full, untruncated WASAPI name
    assert resolve(sd, "usb mic", "input") == 3
    with pytest.raises(LookupError):
        resolve(sd, "nonexistent", "output")
    assert sd.query_devices(2)["name"] == "CABLE Input (VB-Audio Virtual C"  # MME 31-char cut


def test_query_devices_like_sounddevice() -> None:
    sd = FakeSD()
    assert sd.query_devices(kind="output")["index"] == 1  # sd.default is MME
    assert sd.query_devices("Realtek WASAPI", kind="output")["index"] == 4
    with pytest.raises(ValueError, match="Multiple"):
        sd.query_devices("Realtek", kind="output")
    with pytest.raises(ValueError, match="No"):
        sd.query_devices("Nope")
    with pytest.raises(ValueError, match="Not an input"):
        sd.query_devices(4, kind="input")
    info = sd.query_devices(4)
    assert {"name", "index", "hostapi", "max_input_channels", "default_samplerate"} <= set(info)


def test_pump_drives_output_callback_with_dac_times() -> None:
    sd = FakeSD()
    p = Player(sd, 4)
    st = sd.streams[-1]
    assert isinstance(st.kw["extra_settings"], FakeWasapiSettings)
    assert st.kw["extra_settings"].kw == {"auto_convert": True}
    p.queue = np.full(700, 0.25, np.float32)
    out = st.pump(3)
    assert out.shape == (1440,)
    assert np.all(out[:700] == 0.25) and np.all(out[700:] == 0.0)
    assert sd.pump(1)[0].shape == (480,)
    ti = p.times
    assert all(t.outputBufferDacTime - t.currentTime == pytest.approx(0.04) for t in ti)
    assert [t.currentTime for t in ti] == sorted(t.currentTime for t in ti)


def test_unwritten_outdata_is_caught() -> None:
    sd = FakeSD()

    def lazy(out: Any, frames: int, t: Any, status: Any) -> None:
        out[:10] = 0.0  # forgets the rest of the buffer

    st = sd.OutputStream(device=4, channels=2, blocksize=480, callback=lazy)
    st.start()
    with pytest.raises(AssertionError, match="unwritten"):
        st.pump(1)


def test_push_feeds_whole_blocks_and_keeps_the_remainder() -> None:
    sd = FakeSD()
    got: list[np.ndarray] = []
    flags: list[bool] = []

    def cb(indata: Any, frames: int, t: Any, status: Any) -> None:
        got.append(indata[:, 0].copy())
        flags.append(status.input_overflow)

    st = sd.InputStream(device=3, channels=1, blocksize=480, samplerate=48000, callback=cb)
    st.start()
    assert st.push(np.arange(1000, dtype=np.float32)) == 2
    assert st.push(np.arange(1000, 1440, dtype=np.float32), overflow=True) == 1
    assert np.array_equal(np.concatenate(got), np.arange(1440, dtype=np.float32))
    assert flags == [False, False, True]


def test_die_fires_finished_and_reopen_gets_a_new_stream() -> None:
    sd = FakeSD()
    p = Player(sd, 4)
    first = sd.streams[-1]
    first.die()
    assert p.finished == 1 and not first.active
    with pytest.raises(RuntimeError):
        first.pump(1)
    Player(sd, 4)
    assert sd.streams[-1] is not first and sd.streams[-1].active


def test_wasapi_settings_only_on_wasapi_devices() -> None:
    sd = FakeSD()
    with pytest.raises(FakePortAudioError, match="Incompatible host API"):
        sd.OutputStream(
            device=1, channels=2, extra_settings=sd.WasapiSettings(auto_convert=True),
            callback=lambda *a: None,
        )  # fmt: skip
    with pytest.raises(TypeError):
        sd.WasapiSettings(bogus=True)
    s = sd.WasapiSettings(auto_convert=True)
    s._streaminfo.streamCategory = sd._lib.eAudioCategoryCommunications  # private API used
    assert s.auto_convert and not s.exclusive


def test_channel_limits_and_refuse_mono() -> None:
    sd = FakeSD()
    with pytest.raises(FakePortAudioError, match="channels"):
        sd.OutputStream(device=4, channels=8, callback=lambda *a: None)
    idx = sd.add_device("Headset Mic (Bluetooth)", WASAPI, 2, 0, refuse_mono=True)
    with pytest.raises(FakePortAudioError):
        sd.InputStream(device=idx, channels=1, callback=lambda *a: None)
    st = sd.InputStream(
        device=idx, channels=1, extra_settings=sd.WasapiSettings(auto_convert=True),
        callback=lambda *a: None,
    )  # fmt: skip
    assert st.channels == 1


def test_unplug_then_reinit_rescans_devices() -> None:
    sd = FakeSD()
    lost: list[int] = []
    st = sd.OutputStream(
        device=5,
        channels=2,
        callback=lambda o, *a: o.fill(0),
        finished_callback=lambda: lost.append(1),
    )
    st.start()
    assert sd.unplug("CABLE Input") == 2
    assert lost == [1] and not st.active
    assert len(sd.query_devices()) == 6  # PortAudio's list is frozen until re-init
    with pytest.raises(FakePortAudioError, match="unavailable"):
        sd.OutputStream(device=5, channels=2, callback=lambda *a: None)
    sd._terminate()
    sd._ffi.dlclose(sd._lib)
    sd._lib = sd._ffi.dlopen(sd._libname)
    sd._initialize()
    assert sd.reinit_count == 1
    assert [d["name"] for d in sd.query_devices()].count(
        "CABLE Input (VB-Audio Virtual Cable)"
    ) == 0
    sd.replug("CABLE")
    sd._initialize()
    assert resolve(sd, "cable input", "output") >= 0


def test_callback_stop_finishes_the_stream() -> None:
    sd = FakeSD()
    fin: list[int] = []

    def cb(out: Any, frames: int, t: Any, status: Any) -> None:
        out.fill(0.1)
        raise sd.CallbackStop

    st = sd.OutputStream(device=4, channels=2, callback=cb, finished_callback=lambda: fin.append(1))
    st.start()
    y = st.pump(3)
    assert fin == [1] and not st.active
    assert np.all(y[:480] == np.float32(0.1)) and np.all(y[480:] == 0.0)


def test_fail_next_open_and_add_device_defaults() -> None:
    sd = FakeSD(layout="empty")
    assert sd.query_devices() == ()
    out = sd.add_device("Speakers", WASAPI, 0, 2)
    mic = sd.add_device("Mic", WASAPI, 1, 0)
    api = sd.query_hostapis(0)
    assert api["name"] == WASAPI
    assert (api["default_output_device"], api["default_input_device"]) == (out, mic)
    sd.fail_next_open()
    with pytest.raises(FakePortAudioError):
        sd.OutputStream(device=out, channels=2, callback=lambda *a: None)
    assert sd.OutputStream(device=out, channels=2, callback=lambda *a: None).device == out
