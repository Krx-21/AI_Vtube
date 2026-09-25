"""Test doubles for every adapter Protocol in ``aivtube.contracts`` (ARCHITECTURE.md §10).

They ship with the package so the simulator, CI and other agents can use them. None of them
imports PortAudio, onnxruntime, sherpa-onnx, PyAV, edge-tts or livekit.

| Protocol | Fake |
|---|---|
| Clock / Component | ``FakeClock`` (deterministic), ``RealClock`` / ``FakeComponent`` |
| EventBus / Subscription / TaskSupervisor | ``FakeEventBus`` / ``FakeSubscription`` / ``FakeTaskSupervisor`` |
| AudioBackend | ``FakeSD`` (+ ``FakeStream``) |
| AudioOut / AudioIn / EchoCanceller | ``FakeAudioOut`` / ``FakeAudioIn`` / ``FakeEchoCanceller`` |
| VoiceActivityDetector / Endpointer | ``FakeVAD`` / ``FakeEndpointer`` |
| SpeechRecognizer / TranscriptPostProcessor | ``FakeRecognizer`` / ``FakePostProcessor`` |
| TTSBackend / PhraseCache | ``FakeTTS`` / ``FakePhraseCache`` |
| SpeechOutput | ``FakeSpeechOutput`` |
| LLMProvider / LLMRouter / LocalServerManager | ``FakeLLM`` / ``FakeLLMRouter`` / ``FakeLauncher`` |
| (llama-server over HTTP) | ``SseFixtureServer``; launcher endpoint: ``FakeEmergencyServer`` |
| AvatarSink / AvatarDriver | ``FakeAvatarSink`` / ``FakeAvatarDriver``; VTS: ``FakeVTSServer`` |
| ChatSource / ChannelActions / ChatWindow | ``FakeChatSource`` / ``FakeChannelActions`` / ``FakeChatWindow``; IRC: ``FakeIrcServer``; YouTube: ``FakeYouTubeServer`` |
| MemoryStore | ``FakeMemoryStore`` |
| TextFilter / Classifier / SafetyGate | ``FakeTextFilter`` / ``FakeClassifier`` / ``FakeSafetyGate`` |
| Tool / ToolRegistry | ``FakeTool`` / ``FakeToolRegistry`` |
| ControlSurface | ``FakeControlSurface`` |
| GameServer / GameBrain | ``FakeGameServer`` / ``FakeGameBrain`` |
"""

from aivtube.testing.fakes.audio import (
    MME,
    WASAPI,
    FakeAudioIn,
    FakeAudioOut,
    FakeEchoCanceller,
    FakeEndpointer,
    FakePortAudioError,
    FakeSD,
    FakeStream,
    FakeVAD,
    FakeWasapiSettings,
    dominant_frequency,
    marker_tone,
    resample_linear,
)
from aivtube.testing.fakes.avatar import FakeAvatarDriver, FakeAvatarSink
from aivtube.testing.fakes.bus import (
    FakeComponent,
    FakeEventBus,
    FakeSubscription,
    FakeTaskSupervisor,
)
from aivtube.testing.fakes.chat import (
    ALL_CAPABILITIES,
    FakeChannelActions,
    FakeChatSource,
    FakeChatWindow,
    make_chat_message,
)
from aivtube.testing.fakes.clock import FakeClock, RealClock, run_until_idle
from aivtube.testing.fakes.control import FakeControlSurface, FakeGameBrain, FakeGameServer
from aivtube.testing.fakes.irc import SAMPLE_IRC_LINES, FakeIrcServer
from aivtube.testing.fakes.launcher import DEFAULT_PROPS, FakeEmergencyServer, FakeLauncher
from aivtube.testing.fakes.llm import (
    THAI_COMBINING,
    FakeLLM,
    FakeLLMRouter,
    FakeReply,
    ReplyScript,
    split_deltas,
    tool_call,
)
from aivtube.testing.fakes.memory import FakeMemoryStore
from aivtube.testing.fakes.safety import ANON_NAME, FakeClassifier, FakeSafetyGate, FakeTextFilter
from aivtube.testing.fakes.speech import DEFAULT_CANNED, FakeSpeechOutput
from aivtube.testing.fakes.sse import SseFixtureServer, sse_chunks
from aivtube.testing.fakes.stt import FakePostProcessor, FakeRecognizer
from aivtube.testing.fakes.tools import ECHO_SPEC, UNAVAILABLE, FakeTool, FakeToolRegistry
from aivtube.testing.fakes.tts import FakePhraseCache, FakeTTS
from aivtube.testing.fakes.vts import FakeVTSServer
from aivtube.testing.fakes.youtube import FakeYouTubeServer, yt_super_chat, yt_text_message

__all__ = [
    "ALL_CAPABILITIES",
    "ANON_NAME",
    "DEFAULT_CANNED",
    "DEFAULT_PROPS",
    "ECHO_SPEC",
    "MME",
    "SAMPLE_IRC_LINES",
    "THAI_COMBINING",
    "UNAVAILABLE",
    "WASAPI",
    "FakeAudioIn",
    "FakeAudioOut",
    "FakeAvatarDriver",
    "FakeAvatarSink",
    "FakeChannelActions",
    "FakeChatSource",
    "FakeChatWindow",
    "FakeClassifier",
    "FakeClock",
    "FakeComponent",
    "FakeControlSurface",
    "FakeEchoCanceller",
    "FakeEmergencyServer",
    "FakeEndpointer",
    "FakeEventBus",
    "FakeGameBrain",
    "FakeGameServer",
    "FakeIrcServer",
    "FakeLLM",
    "FakeLLMRouter",
    "FakeLauncher",
    "FakeMemoryStore",
    "FakePhraseCache",
    "FakePortAudioError",
    "FakePostProcessor",
    "FakeRecognizer",
    "FakeReply",
    "FakeSD",
    "FakeSafetyGate",
    "FakeSpeechOutput",
    "FakeStream",
    "FakeSubscription",
    "FakeTTS",
    "FakeTaskSupervisor",
    "FakeTextFilter",
    "FakeTool",
    "FakeToolRegistry",
    "FakeVAD",
    "FakeVTSServer",
    "FakeWasapiSettings",
    "FakeYouTubeServer",
    "RealClock",
    "ReplyScript",
    "SseFixtureServer",
    "dominant_frequency",
    "make_chat_message",
    "marker_tone",
    "resample_linear",
    "run_until_idle",
    "split_deltas",
    "sse_chunks",
    "tool_call",
    "yt_super_chat",
    "yt_text_message",
]
