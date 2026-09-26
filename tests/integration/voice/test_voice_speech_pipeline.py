"""Voice speech pipeline, real code with fake I/O (§10 layer 3).

Config → ``build_tts_router`` → ``EdgeTTSBackend`` (offline scripted stream of real MP3) →
stateful PyAV decode → ``SpeechQueue`` → ``FakeAudioOut``, with real lip-sync, a real
on-disk phrase cache and IPC messages validated against Appendix A. Plus the STT side:
``SttRunner`` + the character's alias map + ``UtteranceAssembler`` + echo rejection fed by
the queue's ``recent_tts_text``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("av")

from voice_integration_kit import CPS, ScriptedEdge

from aivtube.config import load_characters, load_config
from aivtube.contracts import ipc
from aivtube.contracts.types import Segment
from aivtube.testing.fakes import FakeAudioOut, FakeClock, FakeRecognizer, marker_tone
from aivtube.voice.speech_queue import SpeechQueue
from aivtube.voice.stt import NamePostProcessor, SttRunner, UtteranceAssembler
from aivtube.voice.tts import DiskPhraseCache, EdgeTTSBackend, build_tts_router

ROOT = Path(__file__).resolve().parents[3]


class Pipeline:
    def __init__(self, tmp_path: Path) -> None:
        self.clock = FakeClock()
        cfg = load_config(ROOT, profile="stream", env={})
        self.chars = load_characters(cfg)
        self.pailin = self.chars["pailin"]
        self.edge_script = ScriptedEdge(sleep=self.clock.sleep)
        edge = EdgeTTSBackend(communicate_factory=self.edge_script, clock=self.clock)
        self.cache = DiskPhraseCache(tmp_path / "phrases")
        self.sent: list[tuple[str, Mapping[str, Any]]] = []
        self.router = build_tts_router(
            cfg.tts.model_dump(),
            {cid: cfg.tts_chain_for(ch) for cid, ch in self.chars.items()},
            root=tmp_path,
            secrets=lambda _k: None,
            clock=self.clock,
            cache=self.cache,
            backends={"edge": edge},
            on_fallback=lambda a, b, r: self.sent.append(
                (ipc.TTS_FALLBACK, {"from": a, "to": b, "reason": r})
            ),
        )
        self.player = FakeAudioOut(clock=self.clock.now, output_latency_s=0.04)
        self.queue = SpeechQueue(
            player=self.player,
            tts=self.router,
            send=lambda t, d: self.sent.append((t, d)),
            clock=self.clock,
            max_in_flight=cfg.tts.max_in_flight,
            max_queued=cfg.tts.max_queued_segments,
            filler_min_interval_s=cfg.tts.filler_min_interval_s,
        )
        self.audio: list[np.ndarray] = []
        self._pump: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Pipeline:
        async def pump() -> None:
            while True:
                await self.clock.sleep(0.01)
                self.audio.append(self.player.pump(1))

        self._pump = asyncio.create_task(pump())
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.queue.aclose()
        assert self._pump is not None
        self._pump.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self._pump

    def of(self, mtype: str) -> list[Mapping[str, Any]]:
        return [d for t, d in self.sent if t == mtype]

    def validate_all(self) -> None:
        for i, (t, d) in enumerate(self.sent):
            ipc.validate(ipc.Envelope(v=ipc.IPC_VERSION, type=t, id=f"m{i}", ts=0.0, data=d))


async def test_voice_turn_reply_plays_through_edge_decoder_player_and_lipsync(
    tmp_path: Path,
) -> None:
    async with Pipeline(tmp_path) as p:
        # startup: pre-synthesise the character's cached phrases (incl. "Filtered.")
        warm = asyncio.create_task(p.router.presynthesize("pailin", p.pailin.cached_phrases))
        await p.clock.run_until(warm.done, within=30)
        assert all((await warm).values()) and (await warm)["Filtered."]
        assert len(list((tmp_path / "phrases" / "premwadee").glob("*.pcm"))) == len(
            p.pailin.cached_phrases
        )
        n_warm = len(p.edge_script.requests)

        chunks = ["สวัสดีค่ะ ทุกคน ", "วันนี้ ไพลิน จะมา เล่นเกม Minecraft กันนะคะ ", "สนุกแน่นอน ค่ะ"]
        await p.queue.begin("u1", "pailin", filler_after_s=1.2, gate_open=True)
        for i, text in enumerate(chunks):
            assert await p.queue.segment(
                Segment("u1", i, text, text, emotion="happy" if i == 0 else None, last=i == 2)
            )
        await p.clock.run_until(lambda: bool(p.of(ipc.UTT_DONE)), within=60)
        p.validate_all()

        # edge was asked with the identity's prosody, one Communicate per segment
        reqs = p.edge_script.requests[n_warm:]
        assert [r[0] for r in reqs] == [c.strip() for c in chunks]
        assert all(r[1] == "th-TH-PremwadeeNeural" for r in reqs)
        assert all((r[2]["rate"], r[2]["pitch"]) == ("+8%", "+20Hz") for r in reqs)
        started = p.of(ipc.SEG_STARTED)
        assert [s["seq"] for s in started] == [0, 1, 2]
        assert all(s["backend"] == "edge" and not s["silent"] for s in started)
        # prefetched segments report their exact duration (stateful decode, sample exact)
        for s, text in zip(started[1:], chunks[1:], strict=True):
            assert s["duration_s"] == pytest.approx(len(text.strip()) / CPS, abs=0.08)
        done = p.of(ipc.SEG_DONE)
        assert all(d["heard"] for d in done)
        utt = p.of(ipc.UTT_DONE)[0]
        assert utt == {
            "utt": "u1",
            "heard_text": "".join(chunks),
            "cancelled": False,
            "reason": None,
        }
        # lip tracks: t0 at the audible start, the mouth moves with the voiced audio
        tracks = [t for t in p.of(ipc.LIP_TRACK) if t["seq"] == 1]
        assert tracks[0]["t0"] == pytest.approx(started[1]["t_audible"])
        assert tracks[-1]["final"] and max(max(t["mouth"]) for t in tracks) > 0.7
        # audio actually reached the device, and no filler was needed (first audio in time)
        assert float(np.abs(np.concatenate(p.audio)).max()) > 0.1
        assert p.queue.stats["fillers"] == 0
        assert "ไพลิน" in p.queue.recent_tts_text()


async def test_filtered_path_stops_after_segment_and_plays_the_cached_clip(tmp_path: Path) -> None:
    async with Pipeline(tmp_path) as p:
        warm = asyncio.create_task(p.router.presynthesize("pailin", ["Filtered."]))
        await p.clock.run_until(warm.done, within=10)
        n_warm = len(p.edge_script.requests)
        await p.queue.begin("u1", "pailin", filler_after_s=None, gate_open=True)
        await p.queue.segment(Segment("u1", 0, "ประโยค แรก ปกติ ดี ", "ประโยค แรก ปกติ ดี "))
        await p.queue.segment(Segment("u1", 1, "ประโยค ที่สอง ", "ประโยค ที่สอง "))
        await p.clock.run_until(lambda: len(p.of(ipc.SEG_STARTED)) == 1, within=10)
        # chunk 3 was BLOCKed by the gate: the reply pipeline stops after the segment ...
        await p.queue.stop("u1", "after_segment", "filtered")
        await p.queue.play_canned("filtered", "pailin")  # ... and plays "Filtered."
        await p.clock.run_until(lambda: bool(p.of(ipc.UTT_DONE)) and p.queue.idle, within=20)
        p.validate_all()
        assert [d["seq"] for d in p.of(ipc.SEG_DONE)] == [0]
        assert p.of(ipc.UTT_DONE)[0]["cancelled"] and p.of(ipc.UTT_DONE)[0]["reason"] == "filtered"
        # segment 1 was being prefetched; "Filtered." itself came from the cache, not edge
        assert all(r[0] != "Filtered." for r in p.edge_script.requests[n_warm:])
        canned = [t for t in p.of(ipc.LIP_TRACK) if t["utt"].startswith("canned-filtered")]
        assert canned and canned[0]["final"]


async def test_stt_turn_with_forced_split_alias_map_and_echo_rejection(tmp_path: Path) -> None:
    async with Pipeline(tmp_path) as p:
        # she speaks first, so the echo filter has something to compare against
        said = "วันนี้ ไพลิน จะมา เล่นเกม กันนะคะ"
        await p.queue.begin("u0", "pailin", filler_after_s=None, gate_open=True)
        await p.queue.segment(Segment("u0", 0, said, said, last=True))
        await p.clock.run_until(lambda: bool(p.of(ipc.UTT_DONE)), within=20)

        rec = FakeRecognizer(
            {
                "440": "สวัสดีครับ ไทลิน วันนี้",  # forced split at 15 s (VadPartial)
                "660": "เล่นเกมอะไรดีครับ",  # the end of the turn (VadEnd)
                "880": "วันนี้ไทลินจะมาเล่นเกมกันนะคะ",  # her own voice through speakers
            },
            name="typhoon_rt",
        )
        post = NamePostProcessor(p.pailin.stt_aliases)
        runner = SttRunner([rec], post)
        try:
            asm = UtteranceAssembler()
            recent = p.queue.recent_tts_text(3600.0)
            part = await runner.transcribe(marker_tone(440, 1.0), recent_tts_text=recent)
            assert part is not None
            asm.add_partial(part)
            final = await runner.transcribe(marker_tone(660, 1.0), recent_tts_text=recent)
            turn = asm.finish(final)
            assert turn is not None and asm.parts == 2
            assert turn.text == "สวัสดีครับ ไพลิน วันนี้ เล่นเกมอะไรดีครับ"
            assert turn.audio_s == pytest.approx(2.0, abs=0.05)
            echo = await runner.transcribe(marker_tone(880, 1.0), recent_tts_text=recent)
            assert echo is None  # dropped as an echo of her TTS
            stt_final = {
                "text": turn.text,
                "engine": turn.engine,
                "latency_ms": turn.latency_ms,
                "t_end": 1.0,
                "audio_s": turn.audio_s,
                "parts": asm.parts,
            }
            ipc.validate(ipc.Envelope(v=1, type=ipc.STT_FINAL, id="s", ts=0.0, data=stt_final))
        finally:
            await runner.aclose()
