"""``setup`` wizard steps with every external dependency faked."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ops_testkit import FileServer, sha, write_manifest

from aivtube.config.layers import read_toml
from aivtube.ops.models import ModelManifest
from aivtube.ops.setup_wizard import (
    STEPS,
    SetupDeps,
    SetupOptions,
    SetupWizard,
    run_setup,
)
from aivtube.testing.fakes import FakeSD

REPO = Path(__file__).resolve().parents[3]
VAD = os.urandom(3000)
M4B = os.urandom(20000)
M30B = os.urandom(30000)


class Answers:
    """A prompter with scripted answers (matched by a substring of the question)."""

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[str] = []
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def ask(self, question: str, default: str) -> str:
        self.asked.append(question)
        for key, value in self.answers.items():
            if key in question:
                return value
        return default


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    return tmp_path


@pytest.fixture
def server() -> Iterator[FileServer]:
    srv = FileServer({"/vad.onnx": VAD, "/4b.gguf": M4B, "/30b.gguf": M30B})
    yield srv
    srv.close()


def manifest_for(root: Path, url: str) -> Path:
    def entry(path: str, data: bytes, dest: str, req: str) -> dict[str, Any]:
        return {"url": url + path, "sha256": sha(data), "size": len(data), "licence": "MIT",
                "required_by": [req], "dest": dest}

    return write_manifest(root, {
        "silero_vad": entry("/vad.onnx", VAD, "models/vad/silero_vad.onnx", "vad.model"),
        "typhoon_rt_tokens": {"url": url + "/missing.txt", "sha256": "", "size": 0,
                              "licence": "CC-BY-4.0", "required_by": ["stt.backends.typhoon_rt"],
                              "dest": "models/stt/typhoon-asr-rt-int8/tokens.txt"},
        "llm_4b": entry("/4b.gguf", M4B, "models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf",
                        "llm.servers.local4b"),
        "llm_30b": entry("/30b.gguf", M30B, "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
                         "llm.servers.local30b"),
    })


def deps(**over: Any) -> SetupDeps:
    async def bench(cfg: Any) -> dict[str, Any]:
        return {"identity": "premwadee", "backends": {"edge": {"ttfa_p50_s": 1.9, "fail_rate": 0.2}},
                "recommendation": {"change": True, "backends": ["azure", "edge"],
                                   "reason": "G1: edge p50 1.90 s", "min_chars": 60}}

    async def presynth(cfg: Any) -> dict[str, dict[str, bool]]:
        return {"pailin": {p: True for p in ("Filtered.", "อืม…")}}

    base: dict[str, Any] = {
        "gpus": lambda: [{"name": "NVIDIA GeForce RTX 4070", "driver": "581.42"}],
        "spawn_background": lambda argv, log: 4242,
        "audio_backend": lambda: FakeSD(),
        "find_spec": lambda name: True,
        "discover_vts": lambda t: [{"active": True, "port": 8001, "windowTitle": "VTube Studio"}],
        "bench_tts": bench,
        "presynth": presynth,
        "doctor": lambda cfg: "✔ everything\n\n0 error(s)",
        "play_tone": None,
        "mic_level": None,
        "platform": "linux",
    }
    base.update(over)
    return SetupDeps(**base)


def user(root: Path) -> dict[str, Any]:
    return read_toml(root / "config" / "user.toml", required=False)


class FakeManifest:
    def __init__(self) -> None:
        real = ModelManifest.load(REPO / "models" / "manifest.toml", root=REPO)
        self.entries = real.entries
        self.pulled: list[str] = []

    def pull_one(self, name: str, **kw: Any) -> None:
        self.pulled.append(name)


@pytest.mark.parametrize("driver, variant", [("581.42", "cuda-13.4"), ("560.94", "cuda-12.4")])
def test_driver_step_selects_the_cuda_build(root: Path, driver: str, variant: str) -> None:
    fake = FakeManifest()
    runs: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        runs.append(argv)
        return subprocess.CompletedProcess(argv, 0, "Available devices:\n  CUDA0: NVIDIA GeForce "
                                                    "RTX 4070 (12281 MiB, 11000 MiB free)\n", "")

    wiz = SetupWizard(root, SetupOptions(only=["driver"]), Answers(), deps(
        gpus=lambda: [{"name": "NVIDIA GeForce RTX 4070", "driver": driver}],
        manifest=lambda r: fake, run=run, platform="win32",
    ))
    assert wiz.run() == 0
    suffix = "13" if variant == "cuda-13.4" else "12"
    assert sorted(fake.pulled) == [f"cudart_cuda{suffix}", f"llama_cpp_cuda{suffix}"]
    assert runs[0][-1] == "--list-devices"
    assert variant in wiz.results["driver"].detail


def test_driver_step_fails_without_a_cuda_device(root: Path) -> None:
    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "Available devices:\n", "")

    wiz = SetupWizard(root, SetupOptions(only=["driver"]), Answers(), deps(
        gpus=lambda: None, manifest=lambda r: FakeManifest(), run=run, platform="win32",
    ))
    assert wiz.run() == 1
    assert "no CUDA" in wiz.results["driver"].detail


def test_models_light_until_the_30b_verifies_then_stream(root: Path, server: FileServer) -> None:
    manifest_for(root, server.url)
    spawned: list[list[str]] = []

    def spawn(argv: list[str], log: Path) -> int:
        spawned.append(argv)
        return 99

    d = deps(spawn_background=spawn)
    part = root / "models" / "llm" / "typhoon2.5-qwen3-4b.Q4_K_M.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(M4B[:5000])  # an interrupted earlier run
    assert run_setup(root, only=["models"], non_interactive=True, deps=d, prompter=Answers()) == 0
    assert (root / "models" / "vad" / "silero_vad.onnx").read_bytes() == VAD
    assert (root / "models" / "llm" / "typhoon2.5-qwen3-4b.Q4_K_M.gguf").read_bytes() == M4B
    assert ("/4b.gguf", "bytes=5000-", 206) in server.requests  # resumed, then verified
    assert not (root / "models" / "llm" / "typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf").exists()
    assert user(root)["active_profile"] == "light"
    argv = spawned[0]
    assert argv[argv.index("pull") + 1] == "llm_30b" and "--then-profile" in argv
    state = json.loads((root / "data" / "state" / "setup.json").read_text(encoding="utf-8"))
    assert state["profile_pending_30b"] is True and state["steps"]["models"]["ok"]
    # the background download finished: the next setup run switches back to stream
    big = root / "models" / "llm" / "typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf"
    big.write_bytes(M30B)
    assert run_setup(root, only=["models"], non_interactive=True, deps=d, prompter=Answers()) == 0
    assert user(root)["active_profile"] == "stream"
    assert len(spawned) == 1


def test_models_step_does_not_start_a_second_30b_download(root: Path, server: FileServer) -> None:
    from aivtube.ops.models import MANIFEST, ModelManifest, _FileLock

    manifest_for(root, server.url)
    spawned: list[list[str]] = []

    def spawn(argv: list[str], log: Path) -> int:
        spawned.append(argv)
        return 99

    m = ModelManifest.load(root / MANIFEST, root=root)
    running = _FileLock(m.lock_path(m.entry("llm_30b")))  # the earlier run's background pull
    assert running.acquire()
    try:
        wiz = SetupWizard(root, SetupOptions(only=["models"]), Answers(),
                          deps(spawn_background=spawn))
        assert wiz.run() == 0
    finally:
        running.release()
    assert spawned == []
    assert "already downloading" in wiz.results["models"].detail
    assert user(root)["active_profile"] == "light"


def test_audio_step_by_name_and_number(root: Path) -> None:
    answers = Answers({"output device": "2", "input device": "Microphone", "headphones": "n"})
    wiz = SetupWizard(root, SetupOptions(only=["audio"]), answers, deps())
    assert wiz.run() == 0
    audio = user(root)["audio"]
    assert audio["output_device"] == "Speakers (Realtek(R) Audio)"  # the 2nd WASAPI output
    assert audio["input_device"] == "Microphone" and audio["headphones"] is False
    assert any("CABLE Input (VB-Audio Virtual Cable)" in s for s in answers.said)


def test_audio_step_tone_and_mic(root: Path) -> None:
    played: list[Any] = []
    wiz = SetupWizard(root, SetupOptions(only=["audio"], output_device="Speakers",
                                         input_device="", headphones=True),
                      Answers({"test tone": "y", "mic": "y"}),
                      deps(play_tone=played.append, mic_level=lambda cfg: -60.0))
    assert wiz.run() == 0
    assert played and played[0].audio.output_device == "Speakers"
    assert any("too quiet" in s for s in wiz.io.said)  # type: ignore[attr-defined]


def test_chat_step(root: Path) -> None:
    assert run_setup(root, options=SetupOptions(only=["chat"], chat="twitch", channel="#PailinCh"),
                     deps=deps(), prompter=Answers()) == 0
    chat = user(root)["chat"]
    assert chat["sources"] == ["twitch_irc"] and chat["twitch_irc"]["channel"] == "pailinch"
    assert run_setup(root, options=SetupOptions(only=["chat"], chat="youtube", video="@pailin"),
                     deps=deps(), prompter=Answers()) == 0
    assert user(root)["chat"]["youtube_poll"]["handle"] == "@pailin"


def test_privacy_consent_defaults_to_no(root: Path) -> None:
    answers = Answers()
    assert run_setup(root, only=["privacy"], deps=deps(), prompter=answers) == 0
    assert user(root)["privacy"] == {"cloud_llm_consent": False, "cloud_stt_consent": False}
    assert any("cloud" in q.lower() for q in answers.asked)
    assert run_setup(root, only=["privacy"], deps=deps(),
                     prompter=Answers({"LLM": "y"})) == 0
    assert user(root)["privacy"]["cloud_llm_consent"] is True


def test_tts_step_applies_the_g1_recommendation(root: Path) -> None:
    assert run_setup(root, only=["tts"], deps=deps(), prompter=Answers()) == 0
    assert user(root)["tts"]["identities"]["premwadee"]["backends"] == ["azure", "edge"]
    bench = json.loads((root / "data" / "state" / "bench.json").read_text(encoding="utf-8"))
    assert bench["tts"]["recommendation"]["change"] is True
    no_voice = deps(find_spec=lambda n: False)
    wiz = SetupWizard(root, SetupOptions(only=["tts", "phrases"]), Answers(), no_voice)
    assert wiz.run() == 0
    assert "skipped" in wiz.results["tts"].detail and "skipped" in wiz.results["phrases"].detail


def test_phrases_step_reports_failures(root: Path) -> None:
    async def partial(cfg: Any) -> dict[str, dict[str, bool]]:
        return {"pailin": {"Filtered.": True, "อืม…": False}}

    wiz = SetupWizard(root, SetupOptions(only=["phrases"]), Answers(), deps(presynth=partial))
    assert wiz.run() == 1
    assert "pailin:อืม…" in wiz.results["phrases"].detail


def test_full_non_interactive_run(root: Path, server: FileServer) -> None:
    manifest_for(root, server.url)
    wiz = SetupWizard(root, SetupOptions(chat="twitch", channel="pailin"), Answers(), deps())
    assert wiz.run() == 0
    assert list(wiz.results) == list(STEPS)
    state = json.loads((root / "data" / "state" / "setup.json").read_text(encoding="utf-8"))
    assert set(state["steps"]) == set(STEPS)
    u = user(root)
    assert u["active_profile"] == "light" and u["privacy"]["cloud_llm_consent"] is False
    with pytest.raises(ValueError):
        SetupWizard(root, SetupOptions(only=["nope"]), Answers(), deps()).selected()
