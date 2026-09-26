"""Static Protocol conformance: each fake (and each real infra class) assigned to its Protocol.

Checked by mypy only (``files = ["src/aivtube"]`` in pyproject, and
``tests/contract/test_static_conformance.py``). Runtime ``isinstance`` on a
``runtime_checkable`` Protocol only checks that the members exist; mypy also checks
signatures, keyword names, defaults and attribute types. Never imported at runtime.

When you add a real implementation, add a line here so a signature drift fails CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aivtube.contracts import (
        avatar,
        chat,
        control,
        games,
        infra,
        llm,
        memory,
        safety,
        speech,
        tools,
        voice,
    )
    from aivtube.infra import AsyncEventBus, BusSubscription, SupervisedTasks, SystemClock
    from aivtube.testing import fakes as F

    # infra
    def _clock(x: F.FakeClock) -> infra.Clock:
        return x

    def _real_clock(x: F.RealClock) -> infra.Clock:
        return x

    def _component(x: F.FakeComponent) -> infra.Component:
        return x

    def _supervisor(x: F.FakeTaskSupervisor) -> infra.TaskSupervisor:
        return x

    def _bus(x: F.FakeEventBus) -> infra.EventBus:
        return x

    def _subscription(x: F.FakeSubscription) -> infra.Subscription:
        return x

    def _system_clock(x: SystemClock) -> infra.Clock:
        return x

    def _async_bus(x: AsyncEventBus) -> infra.EventBus:
        return x

    def _bus_subscription(x: BusSubscription) -> infra.Subscription:
        return x

    def _supervised_tasks(x: SupervisedTasks) -> infra.TaskSupervisor:
        return x

    # voice worker
    def _sd(x: F.FakeSD) -> voice.AudioBackend:
        return x

    def _audio_out(x: F.FakeAudioOut) -> voice.AudioOut:
        return x

    def _audio_in(x: F.FakeAudioIn) -> voice.AudioIn:
        return x

    def _aec(x: F.FakeEchoCanceller) -> voice.EchoCanceller:
        return x

    def _vad(x: F.FakeVAD) -> voice.VoiceActivityDetector:
        return x

    def _endpointer(x: F.FakeEndpointer) -> voice.Endpointer:
        return x

    def _recognizer(x: F.FakeRecognizer) -> voice.SpeechRecognizer:
        return x

    def _post(x: F.FakePostProcessor) -> voice.TranscriptPostProcessor:
        return x

    def _tts(x: F.FakeTTS) -> voice.TTSBackend:
        return x

    def _phrase_cache(x: F.FakePhraseCache) -> voice.PhraseCache:
        return x

    # core side
    def _speech_output(x: F.FakeSpeechOutput) -> speech.SpeechOutput:
        return x

    def _llm(x: F.FakeLLM) -> llm.LLMProvider:
        return x

    def _router(x: F.FakeLLMRouter) -> llm.LLMRouter:
        return x

    def _launcher(x: F.FakeLauncher) -> llm.LocalServerManager:
        return x

    def _avatar_sink(x: F.FakeAvatarSink) -> avatar.AvatarSink:
        return x

    def _avatar_driver(x: F.FakeAvatarDriver) -> avatar.AvatarDriver:
        return x

    def _chat_source(x: F.FakeChatSource) -> chat.ChatSource:
        return x

    def _channel_actions(x: F.FakeChannelActions) -> chat.ChannelActions:
        return x

    def _chat_window(x: F.FakeChatWindow) -> chat.ChatWindow:
        return x

    def _memory(x: F.FakeMemoryStore) -> memory.MemoryStore:
        return x

    def _text_filter(x: F.FakeTextFilter) -> safety.TextFilter:
        return x

    def _classifier(x: F.FakeClassifier) -> safety.Classifier:
        return x

    def _gate(x: F.FakeSafetyGate) -> safety.SafetyGate:
        return x

    def _tool(x: F.FakeTool) -> tools.Tool:
        return x

    def _registry(x: F.FakeToolRegistry) -> tools.ToolRegistry:
        return x

    def _control(x: F.FakeControlSurface) -> control.ControlSurface:
        return x

    def _game_server(x: F.FakeGameServer) -> games.GameServer:
        return x

    def _game_brain(x: F.FakeGameBrain) -> games.GameBrain:
        return x

    # llm (WP6)
    from aivtube.llm.llamacpp import LauncherServerManager, LlamaServerManager
    from aivtube.llm.providers import CannedProvider, OpenAICompatProvider
    from aivtube.llm.router import FallbackRouter

    def _openai_compat_provider(x: OpenAICompatProvider) -> llm.LLMProvider:
        return x

    def _canned_provider(x: CannedProvider) -> llm.LLMProvider:
        return x

    def _fallback_router(x: FallbackRouter) -> llm.LLMRouter:
        return x

    def _llama_server_manager(x: LlamaServerManager) -> llm.LocalServerManager:
        return x

    def _launcher_server_manager(x: LauncherServerManager) -> llm.LocalServerManager:
        return x

    # chat (WP10)
    from aivtube.chat import ScoredChatWindow, TwitchAnonIrc, YouTubeListPoller

    def _twitch_anon_irc(x: TwitchAnonIrc) -> chat.ChatSource:
        return x

    def _youtube_list_poller(x: YouTubeListPoller) -> chat.ChatSource:
        return x

    def _scored_chat_window(x: ScoredChatWindow) -> chat.ChatWindow:
        return x

    # avatar (WP11)
    from aivtube.avatar import LiveAvatarDriver, NullSink, VTSSink

    def _vts_sink(x: VTSSink) -> avatar.AvatarSink:
        return x

    def _null_sink(x: NullSink) -> avatar.AvatarSink:
        return x

    def _live_avatar_driver(x: LiveAvatarDriver) -> avatar.AvatarDriver:
        return x

    # voice front-end (WP1)
    from aivtube.voice.aec import NullAEC, WebRtcAEC
    from aivtube.voice.audio_io import MicCapture, StreamingPlayer
    from aivtube.voice.endpointer import SileroEndpointer
    from aivtube.voice.vad import EnergyVAD, SherpaSileroVAD, SileroOrtVAD

    def _streaming_player(x: StreamingPlayer) -> voice.AudioOut:
        return x

    def _mic_capture(x: MicCapture) -> voice.AudioIn:
        return x

    def _webrtc_aec(x: WebRtcAEC) -> voice.EchoCanceller:
        return x

    def _null_aec(x: NullAEC) -> voice.EchoCanceller:
        return x

    def _silero_ort_vad(x: SileroOrtVAD) -> voice.VoiceActivityDetector:
        return x

    def _sherpa_silero_vad(x: SherpaSileroVAD) -> voice.VoiceActivityDetector:
        return x

    def _energy_vad(x: EnergyVAD) -> voice.VoiceActivityDetector:
        return x

    def _silero_endpointer(x: SileroEndpointer) -> voice.Endpointer:
        return x

    # voice speech (WP2/WP3: STT, TTS)
    from aivtube.voice.stt import (
        NamePostProcessor,
        PyThaiAsrRecognizer,
        SherpaTyphoonRT,
        TyphoonApiRecognizer,
    )
    from aivtube.voice.tts import AzureTTSBackend, DiskPhraseCache, EdgeTTSBackend

    def _sherpa_typhoon_rt(x: SherpaTyphoonRT) -> voice.SpeechRecognizer:
        return x

    def _pythaiasr_recognizer(x: PyThaiAsrRecognizer) -> voice.SpeechRecognizer:
        return x

    def _typhoon_api_recognizer(x: TyphoonApiRecognizer) -> voice.SpeechRecognizer:
        return x

    def _name_post_processor(x: NamePostProcessor) -> voice.TranscriptPostProcessor:
        return x

    def _edge_tts_backend(x: EdgeTTSBackend) -> voice.TTSBackend:
        return x

    def _azure_tts_backend(x: AzureTTSBackend) -> voice.TTSBackend:
        return x

    def _disk_phrase_cache(x: DiskPhraseCache) -> voice.PhraseCache:
        return x

    # memory + tools (WP memory)
    from aivtube.memory import SqliteMemory
    from aivtube.tools import ForgetTool, PolicyToolRegistry, RememberTool

    def _sqlite_memory(x: SqliteMemory) -> memory.MemoryStore:
        return x

    def _policy_tool_registry(x: PolicyToolRegistry) -> tools.ToolRegistry:
        return x

    def _remember_tool(x: RememberTool) -> tools.Tool:
        return x

    def _forget_tool(x: ForgetTool) -> tools.Tool:
        return x

    # safety (WP8)
    from aivtube.safety import KeywordRegexFilter, LayeredSafetyGate

    def _keyword_regex_filter(x: KeywordRegexFilter) -> safety.TextFilter:
        return x

    def _layered_safety_gate(x: LayeredSafetyGate) -> safety.SafetyGate:
        return x
