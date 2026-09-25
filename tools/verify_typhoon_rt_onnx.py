#!/usr/bin/env python3
"""Check a Typhoon RT int8 ONNX export with the project's own sherpa-onnx (1.13.8).

Loads the files the way ``voice.stt.SherpaTyphoonRT`` does, transcribes a Thai sample and
checks the character error rate against the reference text. By default the sample is
generated with edge-tts (``th-TH-PremwadeeNeural``) and decoded with PyAV, both from the
``voice`` extra; ``--wav`` uses a 16 kHz mono WAV instead. With ``--allow-synthetic`` a
failure to generate speech (no network) degrades to a load-and-decode smoke test on a tone.

Usage: uv run --frozen python tools/verify_typhoon_rt_onnx.py dist/typhoon-rt
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

REFERENCE = "สวัสดีค่ะ วันนี้อากาศดีมาก เราไปเที่ยวทะเลกันเถอะ"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def cer(ref: str, hyp: str) -> float:
    """Character error rate, ignoring whitespace."""
    r = [c for c in unicodedata.normalize("NFC", ref) if not c.isspace()]
    h = [c for c in unicodedata.normalize("NFC", hyp) if not c.isspace()]
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(r))


async def _edge_mp3(text: str, voice: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for item in edge_tts.Communicate(text, voice).stream():
        if item.get("type") == "audio":
            buf += item["data"]
    if not buf:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(buf)


def _decode_16k(mp3: bytes) -> np.ndarray:
    import av

    out: list[np.ndarray] = []
    with av.open(io.BytesIO(mp3)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                out.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            out.append(rf.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)


def generated_sample(text: str, voice: str, attempts: int = 3) -> np.ndarray:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _decode_16k(asyncio.run(_edge_mp3(text, voice)))
        except Exception as exc:  # network hiccups: retry with backoff
            last = exc
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"could not generate a sample with edge-tts: {last!r}")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected a 16 kHz mono 16-bit WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def transcribe(model_dir: Path, audio: np.ndarray, threads: int) -> tuple[str, float]:
    import sherpa_onnx

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider="cpu",
    )
    stream = rec.create_stream()
    stream.accept_waveform(16000, audio)
    t0 = time.perf_counter()
    rec.decode_stream(stream)
    return str(stream.result.text), (time.perf_counter() - t0) * 1000.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--wav", type=Path, help="16 kHz mono WAV to use instead of edge-tts")
    ap.add_argument("--text", default=REFERENCE, help="reference text (also what edge-tts says)")
    ap.add_argument("--voice", default="th-TH-PremwadeeNeural")
    ap.add_argument("--max-cer", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args(argv)

    missing = [f for f in FILES if not (args.model_dir / f).is_file()]
    if missing:
        print(f"::error::missing files: {missing}")
        return 1
    tokens = (args.model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines()
    if len(tokens) != 2049 or not tokens[-1].startswith("<blk> "):
        print(f"::error::unexpected tokens.txt ({len(tokens)} lines)")
        return 1

    synthetic = False
    try:
        audio = read_wav(args.wav) if args.wav else generated_sample(args.text, args.voice)
    except Exception as exc:
        if not args.allow_synthetic:
            print(f"::error::{exc}")
            return 1
        print(f"::warning::{exc}; falling back to a synthetic smoke test")
        t = np.arange(32000) / 16000
        audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        synthetic = True

    text, ms = transcribe(args.model_dir, audio, args.threads)
    seconds = audio.size / 16000
    print(f"audio {seconds:.2f} s, decode {ms:.0f} ms, text: {text!r}")
    if synthetic:
        print("::warning::model loads and decodes; accuracy NOT verified (synthetic input)")
        return 0
    score = cer(args.text, text)
    print(f"CER {score:.3f} (limit {args.max_cer})")
    if score > args.max_cer:
        print(f"::error::CER {score:.3f} above {args.max_cer}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
