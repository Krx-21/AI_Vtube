"""``SpeechQueue`` driving the real ``StreamingPlayer`` over ``FakeSD`` in real time.

A thread pumps the fake PortAudio device every 10 ms like a WASAPI callback; the player's
notifier thread fires the DAC-timed markers; the queue moves them onto the loop. Wall-clock
based, so it is marked ``timing``. Skipped when the audio front-end is not importable.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest
from voice_integration_kit import CPS, ScriptedEdge

from aivtube.contracts import ipc
from aivtube.contracts.types import Segment, VoiceSpec
from aivtube.infra.clock import SystemClock
from aivtube.testing.fakes import FakeSD
from aivtube.voice.speech_queue import SpeechQueue
from aivtube.voice.tts import EdgeTTSBackend, IdentityCfg, TTSRouter

pytestmark = pytest.mark.timing

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz")


async def _wait(pred: Any, timeout: float) -> None:
    end = time.perf_counter() + timeout
    while not pred():
        if time.perf_counter() > end:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.01)


@pytest.fixture
def audio_io() -> Any:
    pytest.importorskip("av")
    pytest.importorskip("soxr")
    return pytest.importorskip("aivtube.voice.audio_io")


async def test_markers_heard_text_and_cut_with_the_real_player(audio_io: Any) -> None:
    sd = FakeSD()
    player = audio_io.StreamingPlayer(backend=sd, clock=time.perf_counter)
    player.start()
    stop = threading.Event()

    def pump() -> None:
        next_t = time.perf_counter()
        while not stop.is_set():
            sd.pump(1)
            next_t += 0.01
            time.sleep(max(0.0, next_t - time.perf_counter()))

    pumper = threading.Thread(target=pump, name="fake-device", daemon=True)
    pumper.start()
    sent: list[tuple[str, Mapping[str, Any]]] = []
    router = TTSRouter(
        {"premwadee": IdentityCfg("premwadee", VOICE, ("edge",))},
        {"edge": EdgeTTSBackend(communicate_factory=ScriptedEdge(ttfa_s=0.05))},
        {"pailin": ["premwadee"]},
    )
    q = SpeechQueue(
        player=player, tts=router, send=lambda t, d: sent.append((t, d)), clock=SystemClock()
    )
    try:
        await q.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await q.segment(Segment("u1", 0, "สวัสดี ค่ะ ", "สวัสดี ค่ะ "))
        await q.segment(Segment("u1", 1, "หนึ่ง สอง สาม สี่ ห้า หก", "หนึ่ง สอง สาม สี่ ห้า หก", last=True))
        await _wait(lambda: any(t == ipc.SEG_STARTED and d["seq"] == 1 for t, d in sent), 5.0)
        started = [d for t, d in sent if t == ipc.SEG_STARTED]
        gap = started[1]["t_audible"] - started[0]["t_audible"] - len("สวัสดี ค่ะ") / CPS
        assert -0.02 < gap < 0.15  # back to back, DAC-timed
        await asyncio.sleep(1.0)  # into "สี่"/"ห้า" of segment 1
        await q.stop("u1", "now", "barge_in", fade_ms=30)
        done = [d for t, d in sent if t == ipc.SEG_DONE]
        assert done[0]["heard"] and not done[1]["heard"]
        heard = done[1]["heard_text"]
        assert heard.startswith("หนึ่ง สอง") and heard != "หนึ่ง สอง สาม สี่ ห้า หก"
        utt = next(d for t, d in sent if t == ipc.UTT_DONE)
        assert utt["cancelled"] and utt["heard_text"] == "สวัสดี ค่ะ " + heard
        await asyncio.sleep(0.2)
        assert not player.is_speaking(tail_s=0.0)
        for i, (t, d) in enumerate(sent):
            ipc.validate(ipc.Envelope(v=1, type=t, id=f"m{i}", ts=0.0, data=d))
    finally:
        await q.aclose()
        stop.set()
        pumper.join(1.0)
        player.close()
