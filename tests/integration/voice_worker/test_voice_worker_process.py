"""``aivtube.voice.worker.main`` as a real child process, and the worker's small helpers."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from worker_kit import ROOT, SILERO, TOKEN, wait_for

from aivtube.config import load_characters, load_config
from aivtube.contracts import ipc
from aivtube.contracts.events import SegmentStarted, UtteranceDone
from aivtube.contracts.types import Segment
from aivtube.contracts.voice import EndpointerConfig
from aivtube.infra import AsyncEventBus, SystemClock
from aivtube.ipc import IpcServer
from aivtube.speech import BusSpeechOutput, voice_configure, voice_policy
from aivtube.testing.fakes import FakeSD
from aivtube.voice import worker as W


async def test_main_runs_a_worker_process_that_speaks_and_stops_on_a_signal(
    tmp_path: Path,
) -> None:
    clock = SystemClock()
    bus = AsyncEventBus(clock)
    sub = bus.subscribe(SegmentStarted, UtteranceDone, name="t")
    server = IpcServer("127.0.0.1", 0, TOKEN, clock, bus)
    cfg = load_config(ROOT, profile="stream", env={})
    payload = voice_configure(cfg, load_characters(cfg))
    payload["vad"]["model"] = str(SILERO)
    payload["tts"]["cache_dir"] = str(tmp_path / "phrases")
    out = BusSpeechOutput(
        server,
        bus,
        clock,
        lip_sink=lambda t: None,
        on_cut=lambda u, t: None,
        configure=payload,
        policy=voice_policy(cfg),
    )
    serving = asyncio.create_task(server.serve())
    await server.wait_started()
    env = {
        **os.environ,
        "AIVTUBE_BUS_TOKEN": TOKEN,
        "AIVTUBE__LOGGING__DIR": str(tmp_path / "logs"),
        "AIVTUBE__LOGGING__CONSOLE": "false",
        "PYTHONUNBUFFERED": "1",
    }
    code = "import sys; from aivtube.voice.worker import main; sys.exit(main(sys.argv[1:]))"
    args = ["--root", str(ROOT), "--url", server.url, "--fake-audio", "--fake-models"]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        *args,
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await wait_for(out.ready, 60.0, "worker process READY")
        peer = server.peer("voice")
        assert peer is not None and peer.pid == proc.pid
        assert peer.caps["fake_audio"] and peer.caps["fake_models"]
        await out.begin("u1", "pailin")
        assert await out.segment(Segment("u1", 0, "สวัสดีค่ะ", "สวัสดีค่ะ", last=True))
        events: list[object] = []

        def done() -> bool:
            while (ev := sub.get_nowait()) is not None:  # type: ignore[attr-defined]
                events.append(ev)
            return any(isinstance(e, UtteranceDone) for e in events)

        await wait_for(done, 20.0)
        finished = next(e for e in events if isinstance(e, UtteranceDone))
        assert not finished.cancelled and finished.heard_text == "สวัสดีค่ะ"
        if sys.platform == "win32":
            # CTRL_BREAK_EVENT needs a shared console, which CI runners may not have; the
            # launcher's graceful stop is covered by its own tests
            proc.terminate()
            async with asyncio.timeout(15):
                await proc.wait()
            return
        proc.send_signal(signal.SIGINT)  # graceful stop (SIGBREAK on Windows)
        async with asyncio.timeout(15):
            rc = await proc.wait()
        assert rc == 0, (await proc.stderr.read()).decode(errors="replace")[-2000:]  # type: ignore[union-attr]
        logs = list((tmp_path / "logs").rglob("voice.log"))
        text = logs[0].read_text(encoding="utf-8") if logs else ""
        assert "READY" in text
        assert TOKEN not in text  # the token is redacted
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        await out.aclose()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        sub.close()


def test_main_without_a_token_exits_with_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AIVTUBE_BUS_TOKEN", raising=False)
    monkeypatch.setenv("AIVTUBE__LOGGING__DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AIVTUBE__LOGGING__CONSOLE", "false")
    interval = sys.getswitchinterval()
    try:
        assert W.main(["--root", str(ROOT), "--fake-audio"]) == W.EXIT_CONFIG
        assert sys.getswitchinterval() == pytest.approx(0.001)
    finally:
        sys.setswitchinterval(interval)


def test_raise_priority_is_a_no_op_off_windows() -> None:
    if sys.platform != "win32":
        assert W.raise_priority() is False


def test_fake_models_payload_swaps_stt_and_tts_for_fakes() -> None:
    cfg = load_config(ROOT, profile="stream", env={})
    payload = voice_configure(cfg, load_characters(cfg))
    worker = W.VoiceWorker("ws://127.0.0.1:9/bus", TOKEN, fake_models=True)
    spec = worker._fake_payload(payload)
    assert [e["kind"] for e in spec["stt_chain"]] == ["fake"]
    assert all(b["kind"] == "fake" for b in spec["tts"]["backends"].values())
    assert spec["tts"]["identities"] == payload["tts"]["identities"]  # same voices
    custom = {**payload, "stt_chain": [{"name": "mine", "kind": "fake", "script": ["x"]}]}
    assert worker._fake_payload(custom)["stt_chain"][0]["name"] == "mine"  # kept
    assert payload["stt_chain"][0]["kind"] == "sherpa_nemo_transducer"  # input untouched


def test_endpointer_config_and_alias_helpers() -> None:
    cfg = W._endpointer_config({"threshold": 0.55, "max_segment_s": 2.0, "backend": "energy"})
    assert cfg == EndpointerConfig(threshold=0.55, max_segment_s=2.0)
    aliases = W._aliases(
        {"a": {"stt_aliases": {"ไพลิน": ["ไทลิน"]}}, "b": {"stt_aliases": {"ไพลิน": ["ไทยลิน"]}}}
    )
    assert aliases == {"ไพลิน": ["ไทลิน", "ไทยลิน"]}


def test_fake_audio_device_pumps_speakers_and_feeds_the_mic_in_real_time() -> None:
    sd = FakeSD()
    out = sd.OutputStream(device=4, channels=2, blocksize=480, callback=_tone_out)
    got: list[np.ndarray] = []
    inp = sd.InputStream(
        device=3, channels=1, blocksize=480, callback=lambda d, f, t, s: got.append(d[:, 0].copy())
    )
    out.start()
    inp.start()
    dev = W.FakeAudioDevice(sd)
    dev.inject(np.full(1600, 0.5, np.float32), 16000)  # 0.1 s, resampled to 48 kHz
    assert dev.mic_pending_s == pytest.approx(0.1, abs=0.01)
    t0 = time.perf_counter()
    dev.start()
    try:
        deadline = time.perf_counter() + 5.0
        while (len(got) < 20 or len(dev.output) < 20) and time.perf_counter() < deadline:
            time.sleep(0.01)
    finally:
        dev.stop()
    elapsed = time.perf_counter() - t0
    assert len(dev.output) <= elapsed / 0.01 + 5  # paced, never a burst
    mic = np.concatenate(got)
    assert float(mic[:4000].max()) > 0.4 and float(np.abs(mic[6000:]).max()) == 0.0
    assert float(np.abs(dev.output_since(t0)).max()) == pytest.approx(0.25, abs=0.01)
    assert dev.mic_pending_s == 0.0 and dev.errors == 0


def _tone_out(outdata: Any, frames: int, time_info: Any, status: Any) -> None:
    outdata[:] = 0.25


def test_voice_configure_payload_carries_ipc_valid_worker_messages() -> None:
    """The worker's own messages (health, constraints) validate against Appendix A."""
    worker = W.VoiceWorker("ws://127.0.0.1:9/bus", TOKEN)
    worker._update_health(force=True)
    (mtype, data), *_ = list(worker.sent)
    assert mtype == ipc.HEALTH and data["state"] == "starting"
    ipc.validate(ipc.Envelope(v=1, type=mtype, id="h", ts=0.0, data=data))
