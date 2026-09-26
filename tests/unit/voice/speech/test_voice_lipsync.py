"""``LipSyncAnalyzer`` shape tests (voice.lipsync acceptance)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from aivtube.voice.lipsync import LipSyncAnalyzer, LipSyncConfig, pcm_to_f32

SR = 24000


def tone(seconds: float, freq: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def whole(
    x: np.ndarray, sr: int = SR, cfg: LipSyncConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    a = LipSyncAnalyzer(cfg)
    m1, f1 = a.feed(x, sr)
    m2, f2 = a.flush()
    return np.concatenate([m1, m2]), np.concatenate([f1, f2])


def test_silence_keeps_the_mouth_closed_and_the_form_neutral() -> None:
    mouth, form = whole(np.zeros(SR, np.float32))
    assert mouth.size == 60 and form.size == 60
    assert float(mouth.max()) == 0.0
    assert np.allclose(form, 0.5)


def test_a_loud_vowel_opens_the_mouth_above_0_7() -> None:
    mouth, _ = whole(tone(0.5, 220.0, 0.5))
    assert mouth.size == 30
    assert float(mouth[5:].min()) > 0.7  # after the 30 ms attack
    assert mouth.dtype == np.float32 and float(mouth.max()) <= 1.0


def test_quiet_audio_below_the_noise_floor_stays_closed_and_gain_normalises() -> None:
    quiet = tone(0.5, 220.0, amp=0.004)  # about -51 dBFS RMS
    assert float(whole(quiet)[0].max()) == 0.0
    boosted = whole(quiet, cfg=LipSyncConfig(gain_db=30.0))[0]
    assert float(boosted[10:].min()) > 0.5


def test_attack_is_faster_than_release() -> None:
    x = np.concatenate([tone(0.3, 220.0), np.zeros(int(0.3 * SR), np.float32)])
    mouth, _ = whole(x)
    rise = int(np.argmax(mouth > 0.9))  # frames to open
    fall = int(np.argmax(mouth[18:] < 0.1))  # frames to close after the tone stops
    assert 0 < rise < fall


def test_form_follows_spectral_tilt() -> None:
    bright = tone(0.5, 2500.0) + tone(0.5, 300.0, 0.05)  # "i/e": energy high
    dark = tone(0.5, 400.0) + tone(0.5, 2500.0, 0.01)  # "o/u": energy low
    fb = whole(bright)[1][10:].mean()
    fd = whole(dark)[1][10:].mean()
    assert fb > 0.55 > 0.45 > fd
    assert fd >= 0.0 and fb <= 1.0


@pytest.mark.parametrize("sr", [24000, 22050, 48000, 16000])
def test_chunked_feeding_equals_whole_buffer_analysis(sr: int) -> None:
    rng = np.random.default_rng(sr)
    x = (
        0.3 * rng.standard_normal(int(1.3 * sr)) * np.sin(np.linspace(0, 9, int(1.3 * sr)))
    ).astype(np.float32)
    ref_m, ref_f = whole(x, sr)
    a = LipSyncAnalyzer()
    parts_m, parts_f = [], []
    i = 0
    while i < x.size:
        n = int(rng.integers(1, 3000))
        m, f = a.feed(x[i : i + n], sr)
        parts_m.append(m)
        parts_f.append(f)
        i += n
    m, f = a.flush()
    got_m = np.concatenate([*parts_m, m])
    got_f = np.concatenate([*parts_f, f])
    assert got_m.shape == ref_m.shape and got_m.size == int(np.ceil(1.3 * 60 - 1e-9))
    assert np.max(np.abs(got_m - ref_m)) < 1e-3
    assert np.max(np.abs(got_f - ref_f)) < 1e-3
    assert a.frames == got_m.size


def test_int16_input_reset_and_rate_change() -> None:
    x = tone(0.2, 220.0)
    a = LipSyncAnalyzer()
    m16, _ = a.feed((x * 32767).astype(np.int16), SR)
    a.reset()
    mf, _ = a.feed(x, SR)
    assert np.allclose(m16, mf, atol=1e-3)
    with pytest.raises(ValueError):
        a.feed(x, 16000)
    a.reset()
    assert a.feed(x[:10], 16000)[0].size == 0  # less than a frame: nothing yet
    assert a.flush()[0].size == 1
    assert LipSyncAnalyzer().flush()[0].size == 0
    assert pcm_to_f32(np.zeros((10, 2), np.float32)).shape == (10,)
    with pytest.raises(ValueError):
        LipSyncConfig(fps=0)
    with pytest.raises(ValueError):
        LipSyncConfig(floor_db=-10, ceil_db=-20)


@pytest.mark.timing
def test_cost_is_under_10_ms_per_second_of_audio() -> None:
    x = (0.2 * np.random.default_rng(0).standard_normal(SR * 5)).astype(np.float32)
    a = LipSyncAnalyzer()
    a.feed(x[:SR], SR)  # warm numpy's FFT
    best = float("inf")
    for _ in range(3):
        a.reset()
        t0 = time.perf_counter()
        for i in range(0, x.size, 2880):  # edge-sized chunks (120 ms)
            a.feed(x[i : i + 2880], SR)
        best = min(best, (time.perf_counter() - t0) / 5.0)
    assert best < 0.010, f"{best * 1000:.2f} ms per second of audio"
