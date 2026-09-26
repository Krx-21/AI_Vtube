"""Spike S5 on the streaming PC (``AIVTUBE_HARDWARE=1``, VTube Studio running, API on).

Checks token auth against the real VTS, the 60 Hz ticker's jitter, and which mouth inputs are
injectable (VoiceA/I/U/E/O are unverified). The first run shows the "Allow" popup in VTS.
Set ``AIVTUBE_VTS_URL`` for a non-default port.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from aivtube.avatar import LiveAvatarDriver, VTSAPIError, VTSClient, VTSSink
from aivtube.contracts.avatar import LipTrack
from aivtube.infra import PrecisionTicker, SystemClock

pytestmark = pytest.mark.hardware


async def test_s5_real_vts(tmp_path: Path) -> None:
    url = os.environ.get("AIVTUBE_VTS_URL", "ws://127.0.0.1:8001")
    token = Path(os.environ.get("AIVTUBE_VTS_TOKEN", str(tmp_path / "vts_s5.txt")))
    clock = SystemClock()
    client = VTSClient(url, "AI_Vtube Brain", "AI_Vtube", token, clock=clock)
    sink = VTSSink(client, {}, clock)
    run = asyncio.ensure_future(sink.run())
    try:
        for _ in range(12_000):  # up to 120 s for the streamer to click Allow
            if sink.connected:
                break
            await asyncio.sleep(0.01)
        assert sink.connected, sink.health().detail

        injectable: dict[str, bool] = {}
        for name in ("MouthOpen", "MouthSmile", "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO"):
            try:
                await client.request(
                    "InjectParameterDataRequest",
                    {
                        "faceFound": True,
                        "mode": "set",
                        "parameterValues": [{"id": name, "value": 0}],
                    },
                )
                injectable[name] = True
            except VTSAPIError as exc:
                injectable[name] = False
                print(f"{name}: error {exc.error_id} {exc.message}")
        print("injectable:", injectable)

        tickers: list[PrecisionTicker] = []

        def ticker_factory(*args: Any, **kw: Any) -> PrecisionTicker:
            tickers.append(PrecisionTicker(*args, **kw))
            return tickers[-1]

        drv = LiveAvatarDriver(sink, clock, ticker_factory=ticker_factory)
        mouth = tuple(0.5 + 0.5 * ((i // 6) % 2) for i in range(180))
        drv.on_lip_track(LipTrack("s5", 0, clock.now() + 0.2, 60, mouth, (0.5,) * 180))
        task = asyncio.ensure_future(drv.run())
        await asyncio.sleep(3.0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        stats = await client.request("StatisticsRequest")
        print(
            f"frames {drv.frames_sent}, dropped {client.dropped_frames}, VTS {stats.get('framerate')} fps"
        )
        print("ticker:", tickers[0].jitter_stats(), "fallback:", drv.jitter_fallback_active)
        assert client.dropped_frames <= 3
        if not injectable["MouthOpen"]:  # held by another plugin: the sink must have remapped
            assert "MouthOpen" in sink.remapped, sink.health().detail
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
