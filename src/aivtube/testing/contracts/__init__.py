"""Reusable contract suites: what makes an implementation swappable (ARCHITECTURE.md §10).

Each ``*_suite(factory, ...)`` returns a list of zero-argument test cases (plain callables for
sync Protocols, coroutine functions for async ones). Run them with pytest parametrisation, so
real implementations written later are checked against exactly the same rules as the fakes::

    from aivtube.testing.contracts import case_id, memory_store_suite

    @pytest.mark.parametrize("case", memory_store_suite(make_store), ids=case_id)
    async def test_memory_contract(case):
        await case()

The suites need neither pytest, a GPU nor an audio device.
"""

from aivtube.testing.contracts._base import (
    AsyncCase,
    Case,
    ContractViolation,
    SyncCase,
    await_until,
    case_id,
    raises,
    wait_until,
)
from aivtube.testing.contracts.avatar import avatar_sink_suite
from aivtube.testing.contracts.chat import (
    channel_actions_suite,
    chat_source_suite,
    chat_window_suite,
)
from aivtube.testing.contracts.infra import clock_suite, event_bus_suite, task_supervisor_suite
from aivtube.testing.contracts.llm import (
    SET_TITLE_SPEC,
    llm_provider_suite,
    llm_router_suite,
    local_server_manager_suite,
    set_title_call,
)
from aivtube.testing.contracts.memory import memory_store_suite
from aivtube.testing.contracts.safety import safety_gate_suite, text_filter_suite
from aivtube.testing.contracts.speech import SpeechOutputHarness, speech_output_suite
from aivtube.testing.contracts.tools import BLOCKED_ARG, make_tool_context, tool_registry_suite
from aivtube.testing.contracts.voice import (
    audio_out_suite,
    phrase_cache_suite,
    speech_recognizer_suite,
    tts_backend_suite,
    vad_suite,
)

__all__ = [
    "BLOCKED_ARG",
    "SET_TITLE_SPEC",
    "AsyncCase",
    "Case",
    "ContractViolation",
    "SpeechOutputHarness",
    "SyncCase",
    "audio_out_suite",
    "avatar_sink_suite",
    "await_until",
    "case_id",
    "channel_actions_suite",
    "chat_source_suite",
    "chat_window_suite",
    "clock_suite",
    "event_bus_suite",
    "llm_provider_suite",
    "llm_router_suite",
    "local_server_manager_suite",
    "make_tool_context",
    "memory_store_suite",
    "phrase_cache_suite",
    "raises",
    "safety_gate_suite",
    "set_title_call",
    "speech_output_suite",
    "speech_recognizer_suite",
    "task_supervisor_suite",
    "text_filter_suite",
    "tool_registry_suite",
    "tts_backend_suite",
    "vad_suite",
    "wait_until",
]
