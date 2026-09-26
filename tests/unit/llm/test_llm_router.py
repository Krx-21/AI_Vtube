"""``FallbackRouter``: fallback, breakers, consent, hot-swap and auto-rollback (FakeClock)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from aivtube.config import load_config
from aivtube.contracts.events import Alert, HealthChanged, ProviderSwitched
from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    LLMProvider,
    ProviderCaps,
    ProviderFailed,
    TextDelta,
)
from aivtube.contracts.types import Health, HealthState
from aivtube.infra import FlightRecorder, SystemClock
from aivtube.llm.llamacpp import TemplateCapsError
from aivtube.llm.openai_stream import ProviderError
from aivtube.llm.providers import CannedProvider, build_providers
from aivtube.llm.router import ConsentRequired, FallbackRouter, build_router
from aivtube.testing.contracts import case_id, llm_provider_suite, llm_router_suite
from aivtube.testing.fakes import FakeClock, FakeEventBus, FakeLauncher, FakeLLM, FakeReply

ROOT = Path(__file__).resolve().parents[3]
Mode = Literal["ok", "connect", "fail", "mid", "hang"]


class Scripted:
    """A provider whose behaviour and health a test flips between decisions."""

    def __init__(
        self, name: str, *, cloud: bool = False, text: str = "", ttft_ms: float = 100.0
    ) -> None:
        self.name = name
        self.caps = ProviderCaps(
            tools=True,
            parallel_tools=True,
            json_schema=True,
            prompt_cache=not cloud,
            cloud=cloud,
            keeps_thought_signatures=False,
            reasoning="none",
        )
        self.text = text or f"จาก{name}"
        self.ttft_ms = ttft_ms
        self.mode: Mode = "ok"
        self.health = HealthState.OK
        self.calls = 0
        self.probes = 0
        self.closed = 0
        self.prefills: list[ChatRequest] = []
        self.gate = asyncio.Event()

    async def probe(self) -> Health:
        self.probes += 1
        return Health(f"llm:{self.name}", self.health)

    async def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
        self.calls += 1
        try:
            if self.mode == "connect":
                raise ProviderError(f"{self.name}: refused", emitted=False, reason="connect")
            if self.mode == "fail":
                raise ProviderError(f"{self.name}: HTTP 500", emitted=False, reason="status")
            yield TextDelta(self.text)
            if self.mode == "mid":
                raise ProviderError(f"{self.name}: died", emitted=True, reason="stream_error")
            if self.mode == "hang":
                await self.gate.wait()
            yield Done(
                provider=self.name,
                finish_reason="stop",
                ttft_ms=self.ttft_ms,
                prompt_n=3,
                cache_n=100,
                completion_tokens=1,
                assistant_message={"role": "assistant", "content": self.text},
            )
        finally:
            self.closed += 1

    async def prefill(self, req: ChatRequest) -> float:
        self.prefills.append(req)
        if self.mode != "ok":
            raise ProviderError(f"{self.name}: prefill failed", emitted=False, reason="connect")
        return 0.0


def req(purpose: Literal["speak", "background", "game"] = "speak") -> ChatRequest:
    return ChatRequest(messages=({"role": "user", "content": "สวัสดี"},), purpose=purpose)


async def run(router: FallbackRouter, r: ChatRequest | None = None) -> list[LLMEvent]:
    return [ev async for ev in router.stream(r or req())]


def served_by(events: Sequence[LLMEvent]) -> str:
    done = events[-1]
    assert isinstance(done, Done)
    return done.provider


def make(
    providers: Sequence[LLMProvider],
    clock: FakeClock,
    bus: FakeEventBus | None = None,
    **kw: Any,
) -> FallbackRouter:
    return FallbackRouter(providers, bus=bus, clock=clock, **kw)


def switches(bus: FakeEventBus) -> list[tuple[str, str, str]]:
    return [(e.old, e.new, e.reason) for e in bus.of_type(ProviderSwitched)]


# --- the contract suite -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    llm_router_suite(lambda providers: FallbackRouter(providers, clock=SystemClock())),
    ids=case_id,
)
async def test_router_contract(case: Callable[[], Any]) -> None:
    await case()


@pytest.mark.parametrize(
    "case",
    llm_provider_suite(lambda: CannedProvider("เอ๊ะ สมองไพลินค้างแป๊บนึงนะ"), expect_text=None),
    ids=case_id,
)
async def test_canned_provider_contract(case: Callable[[], Any]) -> None:
    await case()


# --- fallback -----------------------------------------------------------------------------------


async def test_falls_back_before_first_event_and_reports_it(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a, b = Scripted("a"), Scripted("b")
    router = make([a, b], fake_clock, fake_bus)
    assert served_by(await run(router)) == "a"
    a.mode = "fail"
    events = await run(router)
    assert served_by(events) == "b" and events[0] == TextDelta("จากb")
    assert switches(fake_bus) == [("a", "b", "fallback")]
    st = {s.name: s for s in router.status()}
    assert st["a"].fails == 1 and not st["a"].healthy and st["a"].active
    assert st["a"].down_until == pytest.approx(fake_clock.now() + 2.0)


async def test_no_fallback_after_the_first_event(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "mid"
    router = make([a, b], fake_clock)
    seen: list[LLMEvent] = []
    with pytest.raises(ProviderFailed) as ei:
        async for ev in router.stream(req()):
            seen.append(ev)
    assert ei.value.emitted is True and seen == [TextDelta("จากa")]
    assert b.calls == 0 and a.closed == 1
    # the next decision goes to the next provider (a's breaker is open)
    assert served_by(await run(router)) == "b"


async def test_wraps_unexpected_errors(fake_clock: FakeClock) -> None:
    class Broken(Scripted):
        async def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
            yield TextDelta("x")
            raise RuntimeError("bug")

    router = make([Broken("a"), Scripted("b")], fake_clock)
    with pytest.raises(ProviderFailed) as ei:
        await run(router)
    assert ei.value.emitted is True


async def test_a_provider_that_raises_synchronously_is_a_normal_failure(
    fake_clock: FakeClock,
) -> None:
    class Sync(Scripted):
        def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
            self.calls += 1
            raise RuntimeError("not a generator")

    a, b = Sync("a"), Scripted("b")
    router = make([a, b], fake_clock)
    assert served_by(await run(router)) == "b"
    fake_clock.advance(2.01)
    assert served_by(await run(router)) == "b"
    assert a.calls == 2  # the half-open trial ran: the provider is not stuck in "trial"


async def test_breaker_backoff_doubles_up_to_60_s(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode, a.health = "connect", HealthState.DOWN
    router = make([a, b], fake_clock)
    waits = []
    for _ in range(8):
        assert served_by(await run(router)) == "b"
        st = next(s for s in router.status() if s.name == "a")
        waits.append(round(st.down_until - fake_clock.now()))
        fake_clock.advance(st.down_until - fake_clock.now() + 0.01)
    assert waits == [2, 4, 8, 16, 32, 60, 60, 60]
    assert a.calls == 1  # after the first failure only the half-open probes ran
    assert a.probes == 7


async def test_open_breaker_is_skipped_without_calls(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    router = make([a, b], fake_clock)
    await run(router)
    await run(router)
    fake_clock.advance(1.5)
    await run(router)
    assert a.calls == 1 and a.probes == 0 and b.calls == 3


async def test_half_open_probe_then_trial_closes_the_breaker(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    router = make([a, b], fake_clock, fake_bus)
    await run(router)
    a.mode = "ok"
    fake_clock.advance(2.01)
    assert served_by(await run(router)) == "a"
    assert a.probes == 1 and a.calls == 2
    st = next(s for s in router.status() if s.name == "a")
    assert st.fails == 0 and st.healthy and st.down_until == 0.0
    assert switches(fake_bus) == [("a", "b", "fallback"), ("b", "a", "recovered")]


async def test_failed_trial_reopens_with_a_longer_wait(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    router = make([a, b], fake_clock)
    await run(router)
    fake_clock.advance(2.01)
    await run(router)  # probe OK, trial request fails again
    st = next(s for s in router.status() if s.name == "a")
    assert st.fails == 2 and st.down_until == pytest.approx(fake_clock.now() + 4.0)


async def test_down_server_must_stay_healthy_before_it_is_preferred_again(
    fake_clock: FakeClock,
) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode, a.health = "connect", HealthState.DOWN
    router = make([a, b], fake_clock, auto_return_after_s=60.0)
    await run(router)  # a refused: server down
    a.mode, a.health = "ok", HealthState.OK  # the launcher restarted it
    fake_clock.advance(2.01)
    assert served_by(await run(router)) == "b"  # healthy for 0 s: deferred
    fake_clock.advance(30.0)
    assert served_by(await run(router)) == "b"
    fake_clock.advance(30.1)
    assert served_by(await run(router)) == "a"
    assert next(s for s in router.status() if s.name == "a").fails == 0


async def test_deferred_provider_serves_when_everything_else_fails(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "connect"
    router = make([a, b], fake_clock)
    await run(router)
    a.mode, b.mode = "ok", "fail"
    fake_clock.advance(2.01)
    assert served_by(await run(router)) == "a"


# --- consent ------------------------------------------------------------------------------------


async def test_cloud_providers_need_consent(fake_clock: FakeClock) -> None:
    local, cloud = Scripted("local"), Scripted("gemini", cloud=True)
    local.mode = "fail"
    router = make([local, cloud], fake_clock, canned_line="")
    with pytest.raises(ProviderFailed) as ei:
        await run(router)
    assert ei.value.emitted is False and cloud.calls == 0 and cloud.probes == 0
    st = {s.name: s for s in router.status()}
    assert st["gemini"].cloud and not st["gemini"].enabled
    with pytest.raises(ConsentRequired):
        router.promote("gemini")
    with pytest.raises(KeyError):
        router.promote("nope")
    consented = make([Scripted("local", text="x"), cloud], fake_clock, consent=True)
    consented._entries["local"].provider.mode = "fail"  # type: ignore[attr-defined]
    assert served_by(await run(consented)) == "gemini"


async def test_chain_order_and_unknown_chain_entries(fake_clock: FakeClock) -> None:
    a, b, c = Scripted("a"), Scripted("b"), Scripted("c")
    router = make([a, b, c], fake_clock, chain=["c", "missing", "a"])
    assert router.active() == "c"
    c.mode = "fail"
    assert served_by(await run(router)) == "a"
    assert [s.name for s in router.status()] == ["c", "a", "b"]
    router.promote("b")  # not in the chain but promotable
    assert served_by(await run(router)) == "b"


# --- canned line --------------------------------------------------------------------------------


async def test_canned_line_once_per_outage_then_silence(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a = Scripted("a")
    a.mode = "fail"
    router = make([a], fake_clock, fake_bus, canned_line="เอ๊ะ สมองไพลินค้างแป๊บนึงนะ")
    events = await run(router)
    assert events[0] == TextDelta("เอ๊ะ สมองไพลินค้างแป๊บนึงนะ") and served_by(events) == "canned"
    with pytest.raises(ProviderFailed) as ei:
        await run(router)
    assert ei.value.emitted is False
    downs = [e.health for e in fake_bus.of_type(HealthChanged) if e.health.component == "llm"]
    assert [h.state for h in downs] == [HealthState.DOWN]
    assert len([e for e in fake_bus.of_type(Alert) if e.level == "error"]) == 1
    # recovery re-arms the canned line and reports the LLM healthy
    a.mode = "ok"
    fake_clock.advance(10.0)
    assert served_by(await run(router)) == "a"
    downs = [e.health for e in fake_bus.of_type(HealthChanged) if e.health.component == "llm"]
    assert [h.state for h in downs] == [HealthState.DOWN, HealthState.OK]
    a.mode = "fail"
    assert served_by(await run(router)) == "canned"


async def test_canned_line_is_only_for_speech(fake_clock: FakeClock) -> None:
    a = Scripted("a")
    a.mode = "fail"
    router = make([a], fake_clock, canned_line="เอ๊ะ")
    with pytest.raises(ProviderFailed):
        await run(router, req("background"))
    assert served_by(await run(router, req("speak"))) == "canned"


async def test_configured_canned_entry_only_answers_speech(fake_clock: FakeClock) -> None:
    a = Scripted("a")
    a.mode = "fail"
    canned = CannedProvider("เอ๊ะ สมองไพลินค้างแป๊บนึงนะ", name="brain-freeze")
    router = make([a, canned], fake_clock, canned_line="")
    assert served_by(await run(router, req("speak"))) == "brain-freeze"
    fake_clock.advance(10.0)
    with pytest.raises(ProviderFailed) as ei:
        await run(router, req("background"))  # never a canned "summary"
    assert ei.value.emitted is False
    await router.prefill(req())  # nothing to warm: a no-op, not an error


# --- promote / rollback / auto-rollback -----------------------------------------------------


async def test_promote_takes_effect_at_the_next_decision(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "hang"
    router = make([a, b], fake_clock, fake_bus)
    stream = router.stream(req())
    assert await stream.__anext__() == TextDelta("จากa")
    router.promote("b")
    assert router.active() == "b"
    a.gate.set()
    rest = [ev async for ev in stream]
    assert isinstance(rest[-1], Done) and rest[-1].provider == "a"  # the decision in flight
    a.mode = "ok"
    assert served_by(await run(router)) == "b"
    router.rollback()
    assert router.active() == "a" and served_by(await run(router)) == "a"
    router.rollback()  # nothing to roll back to
    assert router.active() == "a"
    assert switches(fake_bus) == [("a", "b", "promote"), ("b", "a", "rollback")]


async def test_promote_resets_the_breaker(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    b.mode = "fail"
    router = make([a, b], fake_clock, chain=["b", "a"])
    await run(router)
    b.mode = "ok"
    router.promote("a")
    router.promote("b")  # the operator insists: b is tried at once despite its breaker
    assert served_by(await run(router)) == "b"


async def test_auto_rollback_after_three_failures(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a, b = Scripted("a"), Scripted("b")
    router = make([a, b], fake_clock, fake_bus, auto_rollback={"window_s": 600, "max_failures": 3})
    router.promote("b")
    b.mode = "fail"
    for _ in range(3):
        assert served_by(await run(router)) == "a"
        fake_clock.advance(65.0)
    assert router.active() == "a"
    assert switches(fake_bus)[-1] == ("b", "a", "auto_rollback")
    assert any(e.level == "warn" and "auto-rollback" in e.message for e in fake_bus.of_type(Alert))


async def test_no_auto_rollback_after_the_window(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    router = make([a, b], fake_clock)
    router.promote("b")
    fake_clock.advance(601.0)
    b.mode = "fail"
    for _ in range(3):
        await run(router)
        fake_clock.advance(65.0)
    assert router.active() == "b"


async def test_auto_rollback_on_ttft_p95(fake_clock: FakeClock, fake_bus: FakeEventBus) -> None:
    a, b = Scripted("a", ttft_ms=300.0), Scripted("b", ttft_ms=500.0)
    router = make([a, b], fake_clock, fake_bus, min_ttft_samples=5)
    for _ in range(5):
        await run(router)  # baseline p95 300 ms
    router.promote("b")
    for _ in range(5):
        await run(router)  # 500 ms <= 2 x 300 ms
    assert router.active() == "b"
    b.ttft_ms = 700.0  # nearest-rank p95 of 6 samples is the max: 700 > 2 x 300
    await run(router)
    assert router.active() == "a"
    assert switches(fake_bus)[-1] == ("b", "a", "auto_rollback")


# --- servers ------------------------------------------------------------------------------------


async def test_on_demand_server_is_started_before_use(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    launcher = FakeLauncher(["local4b"], clock=fake_clock)
    router = make([a, b], fake_clock, servers=launcher, on_demand={"b": "local4b"})
    assert served_by(await run(router)) == "b"
    assert launcher.calls[0] == ("ensure_running", ("local4b", 10.0))
    await run(router)
    assert [c for c in launcher.calls if c[0] == "ensure_running"] == [launcher.calls[0]]


async def test_on_demand_server_that_fails_to_start_is_skipped(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    launcher = FakeLauncher(["local4b"], fail_start=["local4b"], clock=fake_clock)
    router = make([a, b], fake_clock, servers=launcher, on_demand={"b": "local4b"})
    with pytest.raises(ProviderFailed):
        await run(router)
    assert b.calls == 0
    assert next(s for s in router.status() if s.name == "b").fails == 1
    # half-open for an on-demand server means: try to start it again
    launcher.fail_start.clear()
    fake_clock.advance(2.01)
    assert served_by(await run(router)) == "b"


async def test_template_caps_error_raises_an_alert(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    class NoTools(FakeLauncher):
        async def ensure_running(self, server: str, timeout_s: float) -> bool:
            raise TemplateCapsError("no tool calls")

    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    router = make([a, b], fake_clock, fake_bus, servers=NoTools(["x"]), on_demand={"b": "x"})
    with pytest.raises(ProviderFailed):
        await run(router)
    assert any("no tool calls" in e.message for e in fake_bus.of_type(Alert))


# --- health loop ----------------------------------------------------------------------------------


async def test_health_loop_opens_the_breaker_after_three_failed_probes(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    a, b = Scripted("a"), Scripted("b")
    cloud = Scripted("g", cloud=True)
    router = make([a, b, cloud], fake_clock, fake_bus, consent=True, probe_interval_s=2.0)
    task = asyncio.create_task(router.run())
    try:
        await fake_clock.run_for(0.1)
        assert a.probes == 1 and cloud.probes == 0  # cloud providers are not polled
        a.health = HealthState.DOWN
        await fake_clock.run_for(4.0)
        assert next(s for s in router.status() if s.name == "a").fails == 0
        await fake_clock.run_for(2.0)
        st = next(s for s in router.status() if s.name == "a")
        assert st.fails == 1 and not st.healthy
        assert served_by(await run(router)) == "b"
        a.health = HealthState.OK
        await fake_clock.run_for(59.0)
        assert served_by(await run(router)) == "b"  # healthy for < 60 s
        await fake_clock.run_for(4.0)
        assert served_by(await run(router)) == "a"
    finally:
        task.cancel()
    states = [
        e.health.state for e in fake_bus.of_type(HealthChanged) if e.health.component == "llm:a"
    ]
    assert states == [HealthState.OK, HealthState.DOWN, HealthState.OK]


# --- prefill, cancel, observability ---------------------------------------------------------------


async def test_prefill_uses_the_next_serving_local_provider(fake_clock: FakeClock) -> None:
    a, b, cloud = Scripted("a"), Scripted("b"), Scripted("g", cloud=True)
    router = make([a, b, cloud], fake_clock, consent=True)
    await router.prefill(req())
    assert len(a.prefills) == 1
    a.mode = "fail"
    await run(router)
    await router.prefill(req())
    assert len(a.prefills) == 1 and len(b.prefills) == 1
    router.promote("g")
    await router.prefill(req())  # cloud: no-op
    assert len(b.prefills) == 1 and not cloud.prefills
    router.rollback()
    b.mode = "connect"
    with pytest.raises(ProviderFailed):
        await router.prefill(req())
    assert next(s for s in router.status() if s.name == "b").fails == 0  # not counted


async def test_prefill_wraps_unexpected_errors(fake_clock: FakeClock) -> None:
    class Broken(Scripted):
        async def prefill(self, req: ChatRequest) -> float:
            raise RuntimeError("bug")

    router = make([Broken("a")], fake_clock)
    with pytest.raises(ProviderFailed) as ei:
        await router.prefill(req())
    assert ei.value.emitted is False and "bug" in str(ei.value)


async def test_closing_the_router_stream_closes_the_provider(fake_clock: FakeClock) -> None:
    a = Scripted("a")
    a.mode = "hang"
    router = make([a], fake_clock)
    stream = router.stream(req())
    await stream.__anext__()
    await stream.aclose()
    assert a.closed == 1
    assert next(s for s in router.status() if s.name == "a").fails == 0


async def test_cancelling_the_consumer_closes_the_fake_llm() -> None:
    clock = FakeClock()
    llm = FakeLLM([FakeReply("", "สวัสดีค่ะทุกคน", repeat=True)], ttft_s=0.0, tok_s=1.0, clock=clock)
    router = FallbackRouter([llm], clock=clock)
    got = asyncio.Event()

    async def consume() -> None:
        async for _ev in router.stream(req()):
            got.set()

    task = asyncio.create_task(consume())
    await clock.run_until(got.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.closed_early == 1 and llm.in_flight == 0


async def test_flight_recorder_gets_summaries(fake_clock: FakeClock) -> None:
    a, b = Scripted("a"), Scripted("b")
    a.mode = "fail"
    flight = FlightRecorder()
    router = make([a, b], fake_clock, flight=flight)
    await run(router)
    summaries = flight.snapshot()["llm"]
    assert [s["provider"] for s in summaries] == ["a", "b"]
    assert summaries[0]["ok"] is False and "HTTP 500" in summaries[0]["error"]
    assert summaries[1]["ok"] is True and summaries[1]["cache_n"] == 100


async def test_status_reports_ttft_p50(fake_clock: FakeClock) -> None:
    a = Scripted("a")
    router = make([a], fake_clock, canned_line="x")
    for ms in (100.0, 300.0, 200.0):
        a.ttft_ms = ms
        await run(router)
    a.ttft_ms = 9000.0
    await run(router, req("background"))  # a long compaction prompt is not a speech TTFT
    st = {s.name: s for s in router.status()}
    assert st["a"].ttft_p50_ms == 200.0 and st["a"].active and st["a"].healthy
    assert st["canned"].enabled and not st["canned"].active


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FallbackRouter([Scripted("a"), Scripted("a")])
    with pytest.raises(ValueError, match="at least one"):
        FallbackRouter([])
    assert FallbackRouter([], canned_line="x").active() == "canned"


# --- config -------------------------------------------------------------------------------------


def _config(**overrides: Mapping[str, Any]) -> Any:
    return load_config(ROOT, cli_overrides=dict(overrides), env={})


async def test_build_router_from_defaults(fake_clock: FakeClock) -> None:
    cfg = _config()
    providers = build_providers(cfg, secrets=None, clock=fake_clock)
    try:
        assert [p.name for p in providers] == ["local-30b", "local-4b"]  # no cloud consent
        router = build_router(cfg, providers, bus=None, clock=fake_clock)
        assert router.active() == "local-30b"
        assert router._entries["local-4b"].server == "local4b"  # on-demand cold standby
        assert router._entries["local-30b"].server is None
        assert router._canned is not None and router._canned.line == cfg.llm.canned_line
    finally:
        for p in providers:
            await p.aclose()  # type: ignore[attr-defined]


async def test_build_router_with_consent_keeps_cloud_providers(fake_clock: FakeClock) -> None:
    cfg = _config(privacy={"cloud_llm_consent": True})
    providers = build_providers(cfg, secrets=None, clock=fake_clock)
    try:
        names = [p.name for p in providers]
        assert names == ["local-30b", "local-4b", "typhoon-api", "gemini"]
        router = build_router(cfg, providers, bus=None, clock=fake_clock)
        assert all(s.enabled for s in router.status())
    finally:
        for p in providers:
            await p.aclose()  # type: ignore[attr-defined]
