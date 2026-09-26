"""Every adapter Protocol in aivtube.contracts has a fake that structurally satisfies it."""

from __future__ import annotations

import importlib

import pytest

from aivtube.contracts import avatar, chat, control, games, infra, llm, memory, safety, speech
from aivtube.contracts import tools as tools_mod
from aivtube.testing import fakes as F

VOICE = importlib.import_module("aivtube.contracts.voice")


def _speech_output() -> object:
    clock = F.FakeClock()
    return F.FakeSpeechOutput(F.FakeEventBus(clock), clock)


# Protocol -> factory of its fake
FAKES: dict[type, object] = {
    infra.Clock: F.FakeClock,
    infra.Component: F.FakeComponent,
    infra.TaskSupervisor: F.FakeTaskSupervisor,
    infra.EventBus: F.FakeEventBus,
    infra.Subscription: lambda: F.FakeEventBus().subscribe(name="x"),
    VOICE.AudioBackend: F.FakeSD,
    VOICE.AudioOut: F.FakeAudioOut,
    VOICE.AudioIn: F.FakeAudioIn,
    VOICE.EchoCanceller: F.FakeEchoCanceller,
    VOICE.VoiceActivityDetector: F.FakeVAD,
    VOICE.Endpointer: F.FakeEndpointer,
    VOICE.SpeechRecognizer: lambda: F.FakeRecognizer(["x"]),
    VOICE.TranscriptPostProcessor: F.FakePostProcessor,
    VOICE.TTSBackend: F.FakeTTS,
    VOICE.PhraseCache: F.FakePhraseCache,
    speech.SpeechOutput: _speech_output,
    llm.LLMProvider: lambda: F.FakeLLM([]),
    llm.LLMRouter: lambda: F.FakeLLMRouter([F.FakeLLM([])]),
    llm.LocalServerManager: F.FakeLauncher,
    avatar.AvatarSink: F.FakeAvatarSink,
    avatar.AvatarDriver: F.FakeAvatarDriver,
    chat.ChatSource: F.FakeChatSource,
    chat.ChannelActions: F.FakeChannelActions,
    chat.ChatWindow: F.FakeChatWindow,
    memory.MemoryStore: F.FakeMemoryStore,
    safety.TextFilter: F.FakeTextFilter,
    safety.Classifier: F.FakeClassifier,
    safety.SafetyGate: F.FakeSafetyGate,
    tools_mod.Tool: F.FakeTool,
    tools_mod.ToolRegistry: F.FakeToolRegistry,
    control.ControlSurface: F.FakeControlSurface,
    games.GameServer: F.FakeGameServer,
    games.GameBrain: F.FakeGameBrain,
}


def _all_protocols() -> set[type]:
    found: set[type] = set()
    for mod in (infra, VOICE, speech, llm, avatar, chat, memory, safety, tools_mod, control, games):
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == mod.__name__
                and getattr(obj, "_is_protocol", False)
            ):
                found.add(obj)
    return found


def test_every_protocol_has_a_fake() -> None:
    missing = {p.__name__ for p in _all_protocols()} - {p.__name__ for p in FAKES}
    assert not missing, f"protocols without a fake: {sorted(missing)}"


@pytest.mark.parametrize("proto", list(FAKES), ids=lambda p: p.__name__)
def test_fake_satisfies_protocol(proto: type) -> None:
    fake = FAKES[proto]()  # type: ignore[operator]
    assert isinstance(fake, proto), f"{type(fake).__name__} does not implement {proto.__name__}"


def test_fakes_import_no_native_stacks() -> None:
    import subprocess
    import sys

    code = (
        "import sys, aivtube.testing.fakes, aivtube.testing.contracts\n"
        "bad = [m for m in ('sounddevice','sherpa_onnx','onnxruntime','av','edge_tts','azure',"
        "'livekit','soxr','torch','pytest') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"imported: {out.stdout.strip()}"
