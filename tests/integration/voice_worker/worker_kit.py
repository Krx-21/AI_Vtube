"""The voice-worker integration rig: real IPC over localhost, real worker, fake devices/models.

``Stack`` runs, in one process and one event loop:

- core side: ``IpcServer`` (port 0) + ``BusSpeechOutput`` + the real ``AsyncEventBus``;
- worker side: ``VoiceWorker`` with ``--fake-audio`` (``FakeSD`` pumped in real time by
  ``FakeAudioDevice``), the committed Silero model, a ``FakeRecognizer`` and ``FakeTTS``.

``speech_like`` makes audio the real Silero model scores as speech (pure tones score ~0).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aivtube.config import load_characters, load_config
from aivtube.contracts.avatar import LipTrack
from aivtube.contracts.events import Event
from aivtube.contracts.types import Segment
from aivtube.contracts.voice import SpeechRecognizer
from aivtube.infra import AsyncEventBus, SystemClock
from aivtube.ipc import IpcServer
from aivtube.speech import BusSpeechOutput, default_constraints, voice_configure, voice_policy
from aivtube.testing.fakes import FakeRecognizer, FakeTTS
from aivtube.voice.worker import VoiceWorker

ROOT = Path(__file__).resolve().parents[3]
SILERO = ROOT / "tests" / "fixtures" / "silero_vad.onnx"
TOKEN = "integration-token-abcdef"

_VOWELS = np.array(
    [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480)],
    dtype=np.float64,
)
_BW = np.array([90.0, 110.0, 170.0])


def speech_like(seconds: float, *, seed: int = 0, f0: float = 110.0, amp: float = 0.3) -> Any:
    """16 kHz voiced syllables with moving formants (Silero scores them as speech)."""
    sr = 16000
    rng = np.random.default_rng(seed)
    total = round(seconds * sr)
    out = np.zeros(total)
    pos, phase = 0, 0.0
    harmonics = np.arange(1, int(7000 / f0))
    while pos < total:
        syl = int(rng.uniform(0.17, 0.27) * sr)
        n = min(syl, total - pos)
        tt = np.arange(n) / sr
        ph = phase + 2 * np.pi * np.cumsum(f0 * (1.15 - 0.3 * tt / (syl / sr))) / sr
        phase = float(ph[-1])
        fa = _VOWELS[rng.integers(len(_VOWELS))]
        fb = _VOWELS[rng.integers(len(_VOWELS))]
        mix = tt / (syl / sr)
        y = np.zeros(n)
        for h in harmonics:
            fh = h * f0
            ga = np.sum(1 / (1 + ((fh - fa) / _BW) ** 2))
            gb = np.sum(1 / (1 + ((fh - fb) / _BW) ** 2))
            y += (ga * (1 - mix) + gb * mix + 0.01) / np.sqrt(h) * np.sin(h * ph)
        y *= np.clip(np.minimum(tt / 0.025, (syl / sr - tt) / 0.04), 0.0, 1.0)
        out[pos : pos + n] = y
        pos += syl
    peak = float(np.max(np.abs(out))) or 1.0
    return (amp * out / peak).astype(np.float32)


def silence(seconds: float) -> Any:
    return np.zeros(round(seconds * 16000), np.float32)


async def wait_for(pred: Callable[[], bool], timeout: float = 10.0, what: str = "") -> None:
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(0.01)


class Stack:
    """Core + worker over a real localhost websocket (see the module docstring)."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        stt_script: Sequence[str] | dict[str, str] = ("สวัสดีครับ ไพลิน",),
        tts_cps: float = 25.0,
        tone_hz: float | None = 220.0,
        vad: dict[str, Any] | None = None,
        heartbeat_s: float = 1.0,
    ) -> None:
        self.clock = SystemClock()
        self.bus = AsyncEventBus(self.clock)
        self.events: list[Event] = []
        self._sub = self.bus.subscribe(name="stack")
        cfg = load_config(ROOT, profile="stream", env={})
        chars = load_characters(cfg)
        payload = voice_configure(cfg, chars)
        payload["vad"] = {**payload["vad"], "model": str(SILERO), **(vad or {})}
        payload["tts"]["cache_dir"] = str(tmp_path / "phrases")
        self.payload = payload
        self.policy = voice_policy(cfg)
        self.defaults = default_constraints(cfg, chars["pailin"])
        self.tmp_path = tmp_path
        self.server = IpcServer(
            "127.0.0.1", 0, TOKEN, self.clock, self.bus, heartbeat_s=heartbeat_s
        )
        self.lips: list[LipTrack] = []
        self.cuts: list[tuple[str, float]] = []
        self.out = BusSpeechOutput(
            self.server,
            self.bus,
            self.clock,
            lip_sink=self.lips.append,
            on_cut=lambda utt, t: self.cuts.append((utt, t)),
            configure=payload,
            policy=self.policy,
            defaults=self.defaults,
        )
        self.recognizer: SpeechRecognizer = FakeRecognizer(stt_script, name="fake_stt")
        self.tts = FakeTTS(
            name="edge", clock=self.clock, ttfa_s=0.02, chars_per_s=tts_cps, tone_hz=tone_hz
        )
        self.worker: VoiceWorker | None = None
        self._heartbeat_s = heartbeat_s
        self._server_task: asyncio.Task[None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._pump: asyncio.Task[None] | None = None

    async def _collect(self) -> None:
        async for ev in self._sub:
            self.events.append(ev)

    async def start_server(self) -> None:
        self._server_task = asyncio.create_task(self.server.serve())
        async with asyncio.timeout(5):
            await self.server.wait_started()

    async def stop_server(self) -> None:
        task, self._server_task = self._server_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def start(self) -> Stack:
        self._pump = asyncio.create_task(self._collect())
        await self.start_server()
        self.worker = VoiceWorker(
            self.server.url,
            TOKEN,
            root=ROOT,
            fake_audio=True,
            fake_models=True,
            stt_chain=[self.recognizer],
            tts_backends={"edge": self.tts},
            heartbeat_s=self._heartbeat_s,
        )
        self._worker_task = asyncio.create_task(self.worker.run())
        await wait_for(self.out.ready, 20.0, "worker READY")
        return self

    async def aclose(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self._worker_task is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(15):
                    await asyncio.gather(self._worker_task, return_exceptions=True)
            if not self._worker_task.done():
                self._worker_task.cancel()
                await asyncio.gather(self._worker_task, return_exceptions=True)
        await self.stop_server()
        await self.out.aclose()
        self._sub.close()
        if self._pump is not None:
            self._pump.cancel()
            await asyncio.gather(self._pump, return_exceptions=True)

    # --- helpers ------------------------------------------------------------------------------
    @property
    def device(self) -> Any:
        assert self.worker is not None and self.worker.fake_device is not None
        return self.worker.fake_device

    def of(self, *types: type[Event]) -> list[Any]:
        return [e for e in self.events if isinstance(e, types)]

    def sent(self, mtype: str) -> list[dict[str, Any]]:
        assert self.worker is not None
        return [dict(d) for t, d in list(self.worker.sent) if t == mtype]

    def speak(self, utt: str, texts: Sequence[str]) -> list[Segment]:
        return [
            Segment(utt, i, t, t, emotion="happy" if i == 0 else None, last=i == len(texts) - 1)
            for i, t in enumerate(texts)
        ]

    def say_into_mic(self, seconds: float, *, seed: int = 0, tail_s: float = 1.0) -> None:
        self.device.inject(
            np.concatenate([speech_like(seconds, seed=seed), silence(tail_s)]), 16000
        )


@contextlib.asynccontextmanager
async def stack(tmp_path: Path, **kw: Any) -> AsyncIterator[Stack]:
    s = Stack(tmp_path, **kw)
    try:
        yield await s.start()
    finally:
        await s.aclose()


def perf() -> float:
    return time.perf_counter()
