"""Shared helpers for the brain tests: stimuli, a character, and a wired-up Brain harness."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from aivtube.config.schema import CharacterConfig
from aivtube.contracts.types import Priority, Rank, Stimulus, StimulusKind

_ids = itertools.count(1)

DEFAULTS: Mapping[StimulusKind, tuple[Rank, Priority, float | None]] = {
    StimulusKind.OPERATOR: (Rank.OPERATOR, Priority.HIGH, 60.0),
    StimulusKind.VOICE: (Rank.VOICE, Priority.HIGH, 20.0),
    StimulusKind.SUPPORT: (Rank.SUPPORT, Priority.MEDIUM, 600.0),
    StimulusKind.MENTION: (Rank.MENTION, Priority.LOW, 40.0),
    StimulusKind.CHAT: (Rank.CHAT, Priority.LOW, None),
    StimulusKind.GAME_CONTEXT: (Rank.GAME_CONTEXT, Priority.LOW, 30.0),
    StimulusKind.IDLE: (Rank.IDLE, Priority.LOW, None),
}


def stim(
    kind: StimulusKind,
    text: str = "hello",
    *,
    created: float = 1000.0,
    priority: Priority | None = None,
    rank: Rank | None = None,
    ttl_s: float | str | None = "default",
    payload: Mapping[str, Any] | None = None,
    speaker: str | None = None,
    id: str | None = None,
) -> Stimulus:
    r, p, ttl = DEFAULTS.get(kind, (Rank.CHAT, Priority.LOW, 30.0))
    return Stimulus(
        id=id or f"{kind.value}-{next(_ids)}",
        kind=kind,
        character="pailin",
        text=text,
        created=created,
        priority=priority if priority is not None else p,
        rank=rank if rank is not None else r,
        ttl_s=ttl if ttl_s == "default" else ttl_s,  # type: ignore[arg-type]
        source=kind.value,
        speaker=speaker,
        payload=dict(payload or {}),
    )


def character(**overrides: Any) -> CharacterConfig:
    data: dict[str, Any] = {
        "id": "pailin",
        "display_name": "Pailin",
        "name_th": "ไพลิน",
        "aliases": ["ไพลิน", "pailin", "ไพ่ลิน", "น้องไพลิน"],
        "stt_aliases": {"ไพลิน": ["ไทลิน", "ไทยลิน"]},
    }
    data.update(overrides)
    return CharacterConfig(**data)


# --- a wired-up Brain on fakes ------------------------------------------------------------

PERSONA = "เธอคือไพลิน VTuber สาวร่าเริง"
EMOTIONS = frozenset({"neutral", "happy", "sad", "angry", "surprised", "shy", "smug"})


class BrainRig:
    """A ``Brain`` wired to fakes: FakeLLM → FakeLLMRouter, FakeSpeechOutput, FakeMemoryStore,
    FakeSafetyGate, FakeToolRegistry, the real Arbiter/ScoredChatWindow/History/PromptBuilder/
    ReplyPipeline/BackgroundJobs, all on one FakeClock."""

    def __init__(
        self,
        clock: Any,
        *,
        script: Sequence[Any] = (),
        llm: Any = None,
        router: Any = None,
        tools: Sequence[Any] = (),
        blocklist: Sequence[str] = (),
        cfg: Mapping[str, Any] | None = None,
        speech: Any = None,
        bus: Any = None,
        memory: Any = None,
        reply_kwargs: Mapping[str, Any] | None = None,
        dump_flight: Any = None,
        chars_per_s: float = 12.5,
    ) -> None:
        import random

        from aivtube.brain.arbiter import Arbiter
        from aivtube.brain.background import BackgroundJobs
        from aivtube.brain.history import History
        from aivtube.brain.idle import IdleScheduler
        from aivtube.brain.loop import Brain, BrainDeps
        from aivtube.brain.prompt import PromptBuilder
        from aivtube.brain.reply import ReplyPipeline
        from aivtube.brain.tool_flow import ToolFlow
        from aivtube.chat import ScoredChatWindow
        from aivtube.infra.trace import TurnTraceRecorder
        from aivtube.testing.fakes import (
            FakeAvatarDriver,
            FakeEventBus,
            FakeLLM,
            FakeLLMRouter,
            FakeMemoryStore,
            FakeSafetyGate,
            FakeSpeechOutput,
            FakeTaskSupervisor,
            FakeToolRegistry,
        )
        from aivtube.text import estimate_tokens, no_word_split

        self.clock = clock
        self.bus = bus or FakeEventBus(clock)
        self.tasks = FakeTaskSupervisor(clock)
        self.memory = memory or FakeMemoryStore(clock=clock)
        self.window = ScoredChatWindow(clock, rng=random.Random(7))
        settings: dict[str, Any] = {"chat_gather_s": 1.0}
        settings.update(cfg or {})
        self.cfg = settings
        self.arbiter = Arbiter(clock, self.window, settings, bus=self.bus, character="pailin")
        self.registry = FakeToolRegistry(tools)
        self.character = character()
        self.prompt = PromptBuilder(self.character, PERSONA, self.registry)
        self.llm = llm if llm is not None else FakeLLM(list(script), clock=clock, ttft_s=0.3)
        self.router = router if router is not None else FakeLLMRouter([self.llm], clock=clock)
        self.speech = speech or FakeSpeechOutput(self.bus, clock, chars_per_s=chars_per_s)
        self.gate = FakeSafetyGate(blocklist)
        self.trace = TurnTraceRecorder(clock, bus=self.bus)
        self.reply = ReplyPipeline(
            speech=self.speech,
            gate=self.gate,
            bus=self.bus,
            clock=clock,
            trace=self.trace,
            known_emotions=EMOTIONS,
            word_tokenize=no_word_split,
            **dict(reply_kwargs or {}),
        )
        self.history = History(
            self.memory, budget_tokens=5000, estimate=estimate_tokens, clock=clock
        )
        self.idle = IdleScheduler(clock, rng=random.Random(3))
        self.avatar = FakeAvatarDriver()
        self.tool_flow = ToolFlow(self.registry, clock=clock)
        self.background = BackgroundJobs(
            character="pailin",
            router=self.router,
            memory=self.memory,
            ops=None,
            prompt=self.prompt,
            history=self.history,
            servers=None,
            clock=clock,
            busy=lambda: self.brain.slot.busy,
        )
        self.brain = Brain(
            BrainDeps(
                character=self.character,
                clock=clock,
                bus=self.bus,
                tasks=self.tasks,
                arbiter=self.arbiter,
                prompt=self.prompt,
                router=self.router,
                reply=self.reply,
                tool_flow=self.tool_flow,
                registry=self.registry,
                history=self.history,
                background=self.background,
                memory=self.memory,
                speech=self.speech,
                safety=self.gate,
                idle=self.idle,
                avatar=self.avatar,
                trace=self.trace,
                cfg=settings,
                dump_flight=dump_flight,
            )
        )
        self.task: Any = None

    async def start(self, *, live: bool = True, session: bool = True) -> None:
        import asyncio

        from aivtube.contracts.control import OpCommand, OpKind

        if session:
            await self.memory.start_session()
        self.task = asyncio.ensure_future(self.brain.run())
        await self.clock.run_until_idle()
        for _ in range(5000):  # a real store loads on a thread: wait in real time
            if self.brain.ready or self.task.done():
                break
            await asyncio.sleep(0.001)
        assert self.brain.ready, "the brain did not become ready"
        if live:
            await self.brain.control(OpCommand(OpKind.GO_LIVE))
            await self.clock.run_until_idle()

    async def stop(self) -> None:
        import asyncio

        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.speech.aclose()
        for t in list(self.tasks.tracked):
            t.cancel()
        await asyncio.gather(*self.tasks.tracked, return_exceptions=True)

    def say(self, text: str) -> None:
        """Publish a final streamer transcript on the bus (as the voice worker would)."""
        from aivtube.contracts.events import UserTranscript

        self.bus.publish(
            UserTranscript(
                text=text, engine="fake", latency_ms=100.0, audio_s=1.0, ts=self.clock.now()
            )
        )

    def last_user(self, n: int = -1) -> str:
        """The tail (last user message) of LLM request ``n``."""
        msgs = self.llm.requests[n].messages
        return str(msgs[-1]["content"])

    async def idle_out(self, within: float = 120.0) -> None:
        """Run until no decision runs and nothing plays."""
        await self.clock.run_until(
            lambda: not self.brain.slot.busy and self.speech.idle and not self.brain._utts,
            within=within,
        )


async def run_real(clock: Any, pred: Any, within: float = 20.0, step: float = 0.02) -> None:
    """Like ``FakeClock.run_until`` but yields real time before every step, so a store working
    on a thread (SqliteMemory, OpsDb) finishes before fake deadlines expire (a 5 s fake
    deadline leaves such a thread about half a real second)."""
    import asyncio

    end = clock.now() + within
    while not pred():
        if clock.now() >= end:
            raise TimeoutError(f"condition not met within {within} fake seconds")
        for _ in range(2):  # let worker-thread results land on the loop
            await asyncio.sleep(0.001)
        await clock.run_for(step)
