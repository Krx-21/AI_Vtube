"""LiveAvatarDriver with FakeClock: lead scheduling, cuts, keepalive, poses, emotion, jitter."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from aivtube.avatar import LiveAvatarDriver, VTSClient, VTSSink
from aivtube.contracts.avatar import AvatarDriver, LipTrack
from aivtube.infra import SystemClock
from aivtube.testing.fakes import FakeAvatarSink, FakeClock, FakeVTSServer

Spawn = Callable[[Coroutine[Any, Any, Any]], asyncio.Task[Any]]
FPS = 60


def track(utt: str, t0: float, mouth: list[float] | float, n: int = 60, seq: int = 0) -> LipTrack:
    values = mouth if isinstance(mouth, list) else [mouth] * n
    return LipTrack(utt, seq, t0, FPS, mouth=tuple(values), form=(0.5,) * len(values))


def make(sink: Any, clock: FakeClock, factory: Any, **kw: Any) -> LiveAvatarDriver:
    kw.setdefault("idle_motion", False)
    return LiveAvatarDriver(sink, clock, ticker_factory=factory, **kw)


async def test_is_an_avatar_driver(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any
) -> None:
    assert isinstance(make(rec_sink, fake_clock, ticker_factory), AvatarDriver)


async def test_tracks_are_sampled_40ms_ahead(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, attack_ms=0.0, release_ms=0.0)
    spawn(drv.run())
    await fake_clock.run_for(0.1)
    t0 = fake_clock.now() + 0.5
    drv.on_lip_track(track("u1", t0, 0.8, n=60))  # 1 s of audio from t0
    assert drv.sample(t0 - 0.001) == (0.0, 0.5)
    assert drv.sample(t0 + 0.5) == (0.8, 0.5)
    await fake_clock.run_for(2.0)
    opened = [t for t, v in rec_sink.frames if v["MouthOpen"] > 0.5]
    lead = drv.lead_s
    assert lead == pytest.approx(0.040)
    # the mouth opens (and closes) 40 ms before the audio, give or take one frame
    assert t0 - lead - 1e-9 <= opened[0] <= t0 - lead + 1 / FPS + 1e-9
    closed = next(t for t, v in rec_sink.frames if t > opened[0] and v["MouthOpen"] < 0.5)
    assert t0 + 1.0 - lead - 1e-9 <= closed <= t0 + 1.0 - lead + 1 / FPS + 1e-9
    assert ticker_factory.tickers[0].hz == FPS


async def test_chunked_tracks_play_back_to_back(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, lead_ms=0.0)
    t0 = fake_clock.now()
    drv.on_lip_track(track("u", t0 + 0.5, 0.6, n=30, seq=0))  # arrives out of order
    drv.on_lip_track(track("u", t0, 0.2, n=30, seq=0))
    assert drv.sample(t0 + 0.25) == (0.2, 0.5)
    assert drv.sample(t0 + 0.75) == (0.6, 0.5)
    assert drv.sample(t0 + 1.01) == (0.0, 0.5)


async def test_on_cut_closes_the_mouth_within_release_time(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory)
    spawn(drv.run())
    drv.on_lip_track(track("u", fake_clock.now(), 1.0, n=300))
    await fake_clock.run_for(0.5)
    assert rec_sink.current["MouthOpen"] > 0.9
    drv.on_cut("u", fake_clock.now())
    await fake_clock.run_for(0.1)
    assert rec_sink.current["MouthOpen"] <= 0.05  # 25 ms release after a cut
    drv.on_lip_track(track("u", fake_clock.now(), 1.0, seq=1))  # decoded after the cut
    await fake_clock.run_for(0.3)
    assert rec_sink.current["MouthOpen"] == 0.0
    drv.on_lip_track(track("u2", fake_clock.now(), 0.7))  # the next utterance still works
    await fake_clock.run_for(0.3)
    assert rec_sink.current["MouthOpen"] == pytest.approx(0.7, abs=0.01)


async def test_normal_release_is_slower_than_a_cut(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, lead_ms=0.0)
    spawn(drv.run())
    drv.on_lip_track(track("u", fake_clock.now(), 1.0, n=30))
    await fake_clock.run_for(0.5 + 0.05)  # 50 ms after the track ended
    assert 0.2 < rec_sink.current["MouthOpen"] < 0.7  # 60 ms release: still closing
    await fake_clock.run_for(0.5)
    assert rec_sink.current["MouthOpen"] == 0.0


async def test_params_are_resent_at_least_every_second_while_idle(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory)
    start = fake_clock.now()
    spawn(drv.run())
    await fake_clock.run_for(5.0)
    times = [start] + [t for t, _ in rec_sink.frames]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert len(rec_sink.frames) >= 9 and max(gaps) <= 0.5 + 1 / FPS + 1e-6
    assert drv.frames_skipped > 200  # unchanged frames are not re-sent every tick
    assert set(rec_sink.frames[0][1]) == {"MouthOpen", "MouthSmile", "Brows"}


async def test_idle_motion_streams_every_tick(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, idle_motion=True)
    spawn(drv.run())
    await fake_clock.run_for(2.0)
    assert 118 <= len(rec_sink.frames) <= 121
    frame = rec_sink.frames[-1][1]
    for key in ("MouthOpen", "MouthSmile", "Brows", "FaceAngleX", "FaceAngleY", "FaceAngleZ",
                "FacePositionY", "EyeLeftX", "EyeRightY", "EyeOpenLeft", "EyeOpenRight"):  # fmt: skip
        assert key in frame
    assert frame["EyeOpenLeft"] == 1.0
    xs = {round(v["FaceAngleX"], 3) for _, v in rec_sink.frames}
    assert len(xs) > 50


async def test_state_poses(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, idle_motion=True)
    spawn(drv.run())
    await fake_clock.run_for(1.0)
    drv.set_state("thinking")
    assert drv.state == "thinking"
    await fake_clock.run_for(1.5)
    recent = [v for t, v in rec_sink.frames if t > fake_clock.now() - 0.5]
    assert min(v["EyeLeftY"] for v in recent) > 0.25  # eyes up while thinking
    drv.set_state("listening")
    await fake_clock.run_for(1.5)
    recent = [v for t, v in rec_sink.frames if t > fake_clock.now() - 0.5]
    assert max(v["EyeLeftY"] for v in recent) < 0.25


async def test_paused_means_neutral_and_closed_mouth(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn, pailin_map: Any
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, emotion_map=pailin_map)
    spawn(drv.run())
    drv.set_emotion("happy")
    drv.on_lip_track(track("u", fake_clock.now(), 1.0, n=600))
    await fake_clock.run_for(0.5)
    assert rec_sink.current["MouthOpen"] > 0.9
    drv.set_state("paused")  # FREEZE
    drv.on_lip_track(track("u", fake_clock.now(), 1.0, seq=2))  # ignored while paused
    await fake_clock.run_for(0.5)
    assert rec_sink.current["MouthOpen"] == 0.0
    assert drv.emotion == "neutral" and rec_sink.emotions[-1] == ("neutral", 0.3)
    assert rec_sink.current["MouthSmile"] == pytest.approx(0.5)


async def test_emotion_fades_and_is_pushed_once(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn, pailin_map: Any
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, emotion_map=pailin_map, emotion_hold_s=4.0)
    spawn(drv.run())
    await fake_clock.run_for(0.1)
    assert rec_sink.current["MouthSmile"] == 0.5
    drv.set_emotion("happy")
    drv.set_emotion("happy")  # idempotent
    await fake_clock.run_for(0.15)
    assert 0.55 < rec_sink.current["MouthSmile"] < 0.85  # mid-fade
    await fake_clock.run_for(0.3)
    assert rec_sink.current["MouthSmile"] == pytest.approx(0.9)
    assert rec_sink.current["Brows"] == pytest.approx(0.7)
    assert rec_sink.emotions == [("happy", 0.3)]
    # no speech for emotion_hold_s: back to neutral
    await fake_clock.run_for(4.5)
    assert drv.emotion == "neutral" and rec_sink.emotions[-1] == ("neutral", 0.3)
    assert rec_sink.current["MouthSmile"] == pytest.approx(0.5)


async def test_emotion_holds_while_speaking(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn, pailin_map: Any
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, emotion_map=pailin_map, emotion_hold_s=1.0)
    spawn(drv.run())
    drv.set_state("speaking")
    drv.set_emotion("sad")
    await fake_clock.run_for(3.0)
    assert drv.emotion == "sad"
    drv.set_state("idle")
    await fake_clock.run_for(1.6)
    assert drv.emotion == "neutral"


async def test_smile_follows_mouth_form(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory)
    spawn(drv.run())
    now = fake_clock.now()
    drv.on_lip_track(LipTrack("u", 0, now, FPS, mouth=(0.8,) * 120, form=(1.0,) * 120))
    await fake_clock.run_for(0.5)
    assert rec_sink.current["MouthSmile"] == pytest.approx(0.5 + 0.6 * 0.5, abs=0.01)


async def test_jitter_above_limit_drops_to_30hz(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    ticker_factory.p95_ms = 12.0
    drv = make(
        rec_sink, fake_clock, ticker_factory, idle_motion=True, jitter_check_s=1.0,
        jitter_min_ticks=30,
    )  # fmt: skip
    spawn(drv.run())
    await fake_clock.run_for(3.0)
    ticker = ticker_factory.tickers[0]
    assert drv.jitter_fallback_active and drv.fps == 30 and ticker.hz_changes == [30]
    n = len(rec_sink.frames)
    await fake_clock.run_for(1.0)
    assert 29 <= len(rec_sink.frames) - n <= 31


async def test_low_jitter_keeps_60hz(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    ticker_factory.p95_ms = 3.0
    drv = make(rec_sink, fake_clock, ticker_factory, jitter_check_s=1.0, jitter_min_ticks=30)
    spawn(drv.run())
    await fake_clock.run_for(5.0)
    assert drv.fps == 60 and not drv.jitter_fallback_active
    assert ticker_factory.tickers[0].hz_changes == []


async def test_run_stops_the_ticker_and_can_restart(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory, idle_motion=True)
    task = spawn(drv.run())
    await fake_clock.run_for(0.2)
    with pytest.raises(RuntimeError):
        await drv.run()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ticker_factory.tickers[0].stopped
    n = len(rec_sink.frames)
    await fake_clock.run_for(0.5)
    assert len(rec_sink.frames) == n  # nothing is sent after run() ends
    spawn(drv.run())
    await fake_clock.run_for(0.5)
    assert len(rec_sink.frames) > n and len(ticker_factory.tickers) == 2


async def test_sink_failures_never_break_the_driver(
    fake_clock: FakeClock, ticker_factory: Any, spawn: Spawn, pailin_map: Any
) -> None:
    class Broken(FakeAvatarSink):
        async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None:
            raise RuntimeError("renderer gone")

    sink = Broken()
    sink.connect()
    drv = make(sink, fake_clock, ticker_factory, emotion_map=pailin_map)
    task = spawn(drv.run())
    drv.set_emotion("happy")
    await fake_clock.run_for(1.0)
    assert not task.done() and sink.params


@pytest.mark.timing
async def test_real_ticker_drives_vts_at_60hz(token_path: Any, spawn: Spawn) -> None:
    """End to end: PrecisionTicker → driver → VTSSink → FakeVTS, 2 s at 60 Hz, 0 dropped."""
    async with FakeVTSServer(frame_hz=60.0) as server:
        clock = SystemClock()
        client = VTSClient(server.url, "AI_Vtube Brain", "AI_Vtube", token_path)
        sink = VTSSink(client, {}, clock, poll_s=0.05)
        spawn(sink.run())
        loop = asyncio.get_running_loop()
        end = loop.time() + 5.0
        while not sink.connected and loop.time() < end:
            await asyncio.sleep(0.01)
        assert sink.connected
        drv = LiveAvatarDriver(sink, clock, idle_motion=True)
        before = len(server.injected)
        spawn(drv.run())
        await asyncio.sleep(2.0)
        sent = drv.frames_sent
        assert 110 <= sent <= 125
        await asyncio.sleep(0.2)
        assert client.dropped_frames == 0
        assert len(server.injected) - before >= sent - 1


async def test_without_an_emotion_map_names_pass_through(
    fake_clock: FakeClock, rec_sink: Any, ticker_factory: Any, spawn: Spawn
) -> None:
    drv = make(rec_sink, fake_clock, ticker_factory)
    spawn(drv.run())
    drv.set_emotion("Happy")
    await fake_clock.run_for(0.1)
    assert drv.emotion == "happy" and rec_sink.emotions == [("happy", 0.3)]
    assert rec_sink.current["MouthSmile"] == 0.5  # neutral baseline: the sink owns the look
