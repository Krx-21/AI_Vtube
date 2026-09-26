"""``bench`` functions against fakes: TTS (G1), LLM (G2, --tune), STT, audio (S4)."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.config import load_config
from aivtube.config.layers import read_toml
from aivtube.launcher.llama import TuningStore
from aivtube.ops import bench as B
from aivtube.testing.fakes import FakeClock, FakeRecognizer, FakeSD, FakeTTS, SseFixtureServer
from aivtube.testing.fakes.llm import FakeReply

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    return tmp_path


def test_percentile_and_sentences() -> None:
    assert B.percentile([], 50) is None
    assert B.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert B.percentile([5.0], 95) == 5.0
    s = B.novel_sentences(40, seed=1)
    assert len(set(s)) == 40 and all("ไพลิน" in x for x in s)


async def _run_tts(cfg: Any, backends: dict[str, FakeTTS], clock: FakeClock, **kw: Any) -> Any:
    task = asyncio.ensure_future(B.bench_tts(cfg, backends=backends, clock=clock, pause_s=0,
                                             seed=3, **kw))
    await clock.run_until(task.done, within=600, step=0.05)
    return await task


async def test_bench_tts_g1_switches_to_azure_when_edge_is_slow(root: Path) -> None:
    clock = FakeClock()
    cfg = load_config(root, env={})
    backends = {"edge": FakeTTS(name="edge", ttfa_s=1.5, clock=clock),
                "azure": FakeTTS(name="azure", ttfa_s=0.3, clock=clock)}
    result = await _run_tts(cfg, backends, clock, n=10)
    edge, azure = result["backends"]["edge"], result["backends"]["azure"]
    assert edge["n"] == 10 and edge["ok"] == 10 and edge["fail_rate"] == 0.0
    assert edge["ttfa_p50_s"] == pytest.approx(1.5, abs=0.06)
    assert azure["ttfa_p95_s"] == pytest.approx(0.3, abs=0.06)
    assert edge["slow_rate"] == 0.0  # 1.5 s < the 2 s first-audio timeout
    rec = result["recommendation"]
    assert rec["change"] is True and rec["backends"] == ["azure", "edge"]
    assert rec["min_chars"] == 60 and "G1" in rec["reason"]
    assert len({t for t, _ in backends["edge"].requests}) == 10  # novel text every time
    path = B.apply_tts_recommendation(root, "premwadee", rec)
    assert path is not None
    assert read_toml(path)["tts"]["identities"]["premwadee"]["backends"] == ["azure", "edge"]


async def test_bench_tts_keeps_edge_when_fast_and_counts_failures(root: Path) -> None:
    clock = FakeClock()
    cfg = load_config(root, env={})
    fast = {"edge": FakeTTS(name="edge", ttfa_s=0.3, clock=clock)}
    result = await _run_tts(cfg, fast, clock, n=8)
    assert result["recommendation"]["change"] is False
    flaky = {"edge": FakeTTS(name="edge", ttfa_s=0.3, fail_rate=0.5, seed=1, clock=clock),
             "azure": FakeTTS(name="azure", ttfa_s=0.4, clock=clock)}
    result = await _run_tts(cfg, flaky, clock, n=20)
    edge = result["backends"]["edge"]
    assert edge["failures"] > 1 and edge["fail_rate"] > 0.05 and edge["last_error"]
    assert result["recommendation"]["backends"][0] == "azure"
    no_azure = {"edge": FakeTTS(name="edge", ttfa_s=1.0, clock=clock)}
    rec = (await _run_tts(cfg, no_azure, clock, n=3))["recommendation"]
    assert rec["change"] is False and "Azure is not available" in rec["reason"]


async def test_bench_tts_repeat_mode_uses_one_sentence(root: Path) -> None:
    clock = FakeClock()
    cfg = load_config(root, env={})
    tts = {"edge": FakeTTS(name="edge", ttfa_s=0.2, clock=clock)}
    await _run_tts(cfg, tts, clock, n=4, novel=False)
    assert {t for t, _ in tts["edge"].requests} == {B.REPEAT_SENTENCE}


async def test_bench_llm_measures_ttft_and_cache(root: Path) -> None:
    srv = SseFixtureServer(default=FakeReply("", "สวัสดีค่ะ วันนี้ไพลินอารมณ์ดีมากเลย", repeat=True),
                           ttft_s=0.05)
    await srv.start()
    try:
        cfg = load_config(root, env={})
        result = await asyncio.to_thread(B.bench_llm, cfg, base_url=srv.root_url, runs=3, seed=1)
    finally:
        await srv.stop()
    assert "error" not in result, result
    assert result["runs"] == 3 and result["server"] == "local30b"
    assert result["ttft_p50_s"] is not None and result["ttft_p50_s"] >= 0.04
    assert result["tok_s_p50"] == 45.5  # llama-server's own timings
    assert result["cache_ratio_p50"] is not None and result["cache_ratio_p50"] > 0
    body = srv.requests[-1]
    assert body["stream"] is True and body["id_slot"] == 0 and body["cache_prompt"] is True
    assert result["recommendation"]["profile"] == "stream"  # fast enough for G2


def test_g2_recommends_light_when_slow() -> None:
    slow = {"alias": "pailin-30b", "ttft_p50_s": 0.9, "tok_s_p50": 42.0}
    assert B.g2_recommendation("local30b", slow)["profile"] == "light"
    few = {"alias": "pailin-30b", "ttft_p50_s": 0.4, "tok_s_p50": 22.0}
    assert B.g2_recommendation("local30b", few)["profile"] == "light"
    assert B.g2_recommendation("local4b", {"alias": "pailin-4b"})["profile"] is None


def test_bench_llm_unreachable(root: Path) -> None:
    cfg = load_config(root, env={})
    result = B.bench_llm(cfg, base_url="http://127.0.0.1:9", runs=1)
    assert "not reachable" in result["error"]


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in B.NCMOE_SWEEP:
        for t in B.THREAD_SWEEP:
            tg = 40.0 - abs(n - 34) * 1.5 - abs(t - 8) * 0.5
            if n == 36 and t == 8:
                tg = 39.2  # within 3 % of the best: the higher N wins (more VRAM headroom)
            rows.append({"n_cpu_moe": n, "n_threads": t, "n_prompt": 512, "n_gen": 0,
                         "avg_ts": 500.0 - n})
            rows.append({"n_cpu_moe": n, "n_threads": t, "n_prompt": 0, "n_gen": 128,
                         "avg_ts": tg})
    return rows


def test_pick_tuning() -> None:
    choice = B.pick_tuning(_rows())
    assert choice == {"n_cpu_moe": 36, "threads": 8, "tok_s": 39.2, "pp_tok_s": 464.0,
                      "best_tok_s": 40.0}
    assert B.pick_tuning([]) is None
    assert B.pick_tuning([{"n_threads": 8, "n_gen": 128, "avg_ts": 3.0}]) is None


def test_tune_runs_llama_bench_and_pins_user_toml(root: Path) -> None:
    cfg = load_config(root, env={})
    store = TuningStore(root / "data" / "state" / "llama_tuning.json")
    store.set("local30b", n_cpu_moe=40, base=0, model="x.gguf")
    seen: list[list[str]] = []

    def runner(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "build: b11177\n" + json.dumps(_rows()), "")

    out = B.tune_llm(cfg, "local30b", runner=runner)
    argv = seen[0]
    assert Path(argv[0]).name.startswith("llama-bench")
    assert argv[argv.index("-ncmoe") + 1] == "28,30,32,34,36,38,40,42,44"
    assert argv[argv.index("-t") + 1] == "6,8,12"
    assert argv[argv.index("-m") + 1].endswith("typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf")
    assert out["choice"]["n_cpu_moe"] == 36
    user = read_toml(root / "config" / "user.toml")
    server = user["llm"]["servers"]["local30b"]
    assert server == {"placement": "pinned", "pinned_n_cpu_moe": 36, "threads": 8}
    assert store.get("local30b") is None  # the runtime bump is superseded
    reloaded = load_config(root, env={})
    assert reloaded.llm.servers["local30b"].pinned_n_cpu_moe == 36


def test_tune_without_llama_bench(root: Path) -> None:
    cfg = load_config(root, env={})
    assert "not found" in B.tune_llm(cfg, "local30b")["error"]

    def garbage(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "error: out of memory", "cuda error")

    assert "no JSON" in B.tune_llm(cfg, "local30b", runner=garbage)["error"]


def _wav(path: Path, hz: float, seconds: float = 1.0, sr: int = 16000) -> None:
    t = np.arange(int(sr * seconds)) / sr
    pcm = (0.5 * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


async def test_bench_stt_over_wavs(root: Path, tmp_path: Path) -> None:
    folder = tmp_path / "clips"
    folder.mkdir()
    _wav(folder / "a.wav", 440)
    _wav(folder / "b.wav", 660, sr=48000)  # resampled to 16 kHz
    (folder / "texts.txt").write_text("a.wav\tสวัสดีค่ะ\nb.wav\tไพลินน่ารัก\n", encoding="utf-8")
    rec = FakeRecognizer({"440": "สวัสดีค่ะ", "660": "ไทลินน่ารัก"}, delay_s=0.01)
    cfg = load_config(root, env={})
    result = await B.bench_stt(cfg, wavs=sorted(folder.glob("*.wav")), recognizer=rec)
    assert result["files"] == 2 and result["engine"] == "fake"
    by = {r["file"]: r for r in result["results"]}
    assert by["a.wav"]["cer"] == 0.0 and by["b.wav"]["cer"] == pytest.approx(1 / 10, abs=0.01)
    assert result["latency_p50_s"] >= 0.01 and result["rtf_p50"] is not None
    assert B.cer("ไพลิน", "ไทลิน") == pytest.approx(0.2)


def test_bench_audio_with_fake_sd(root: Path) -> None:
    cfg = load_config(root, env={})
    sd = FakeSD()
    block = cfg.audio.samplerate * cfg.audio.block_ms // 1000

    def step(dt: float) -> None:
        sd.pump(int(dt * cfg.audio.samplerate / block))

    result = B.bench_audio(cfg, soak_s=3, backend=sd, step=step)
    assert result["duration_s"] == 3.0 and result["blocksize"] == 480
    assert result["stream_latency_s"] is not None
    assert 0.04 <= result["recommendation"]["output_latency_s"] <= 0.06
    assert result["underflows"] >= 0


async def test_bench_e2e_reports_missing_sim(root: Path) -> None:
    cfg = load_config(root, env={})
    result = await B.bench_e2e(cfg, clips=root, profile="ci")
    assert result["kind"] == "e2e"


def test_write_recommendations_merges(root: Path) -> None:
    p1 = B.write_recommendations(root, {"tts": {"x": 1}})
    p2 = B.write_recommendations(root, {"llm": {"y": 2}})
    assert p1 == p2
    data = json.loads(p2.read_text(encoding="utf-8"))
    assert data["tts"] == {"x": 1} and data["llm"] == {"y": 2} and "updated" in data


def test_bench_cli_prints_json_and_merges_results(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def fake_llm(cfg: Any, **kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"kind": "llm", "error": "llama-server is not reachable"}

    monkeypatch.setattr(B, "bench_llm", fake_llm)
    assert B.main(["--root", str(root), "--profile", "light", "llm", "--tune", "--runs", "2"]) == 1
    assert seen == {"tune": True, "server": None, "runs": 2}
    assert json.loads(capsys.readouterr().out)["kind"] == "llm"
    stored = json.loads((root / "data" / "state" / "bench.json").read_text(encoding="utf-8"))
    assert stored["llm"]["error"]
    with pytest.raises(SystemExit):
        B.main(["--root", str(root), "nonsense"])
