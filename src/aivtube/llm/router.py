"""``FallbackRouter`` (§4.11, §2.8): the LLM chain with breakers, consent gating, hot-swap and
automatic rollback.

**Chain.** ``[active] + the configured chain`` (``local-30b → local-4b → typhoon-api → gemini``)
and then, for ``purpose="speak"`` only, the canned "brain freeze" line, at most once per
outage (after that the router raises and the brain stays silent until a provider is back).
A ``CannedProvider`` configured as a chain entry likewise answers only speech turns, never a
background summary or a game action. Cloud providers are skipped unless
``privacy.cloud_llm_consent`` (I8). TTFT statistics (status, auto-rollback) use speech turns
only.

**Fallback only before the first event.** A provider that fails before yielding anything is
skipped for this decision; once an event was yielded, the failure is re-raised as
``ProviderFailed(emitted=True)`` (the utterance ends with "…") and the next decision uses the
next provider. There are never two voices in one utterance.

**Breakers.** Every failure opens the provider's breaker for ``min(60, 2**fails)`` s. When it
expires the provider is half-open: the next decision first probes it (``probe()``, or
``ensure_running`` for an on-demand llama-server) and then sends one trial request; success
closes the breaker, failure re-opens it with a longer wait. A provider whose *server* was down
(connection refused, failed probe) must stay healthy for ``auto_return_after_s`` (60 s) before
it is preferred again, so a restarted llama-server can restore its KV first; until then it is
tried only if every other provider fails.

**Hot-swap.** ``promote(name)`` takes effect at the next decision and remembers the previous
provider; ``rollback()`` returns to it. Within ``window_s`` (600 s) of a promotion, 3 failures of
the promoted provider, or a TTFT p95 above ``ttft_p95_factor`` (2×) the previous provider's
p95, roll back automatically and publish ``ProviderSwitched(reason="auto_rollback")``.

``run()`` is an optional supervised health loop: it probes local providers every
``probe_interval_s`` (2 s) and opens a breaker after ``probe_failures`` (3) failed probes.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from aivtube.contracts.events import Alert, Event, HealthChanged, ProviderSwitched
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.llm import (
    ChatRequest,
    Done,
    LLMEvent,
    LLMProvider,
    LocalServerManager,
    ProviderFailed,
    ProviderStatus,
)
from aivtube.contracts.types import Health, HealthState
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.llm.llamacpp import TemplateCapsError
from aivtube.llm.openai_stream import ProviderError
from aivtube.llm.providers import CannedProvider

if TYPE_CHECKING:
    from aivtube.config.schema import AppConfig

__all__ = ["ConsentRequired", "FallbackRouter", "LLMFlightLog", "build_router"]

log = logging.getLogger(__name__)

KIND = "llm"
_Admit = Literal["go", "skip", "defer"]


class ConsentRequired(PermissionError):
    """``promote()`` of a cloud provider without ``privacy.cloud_llm_consent``."""


class LLMFlightLog(Protocol):
    """Where per-request summaries go (``infra.FlightRecorder`` implements it)."""

    def record_llm(self, summary: Mapping[str, Any]) -> None: ...


@dataclass
class _Entry:
    provider: LLMProvider
    name: str
    cloud: bool
    enabled: bool
    server: str | None = None  # on-demand llama-server started through the manager
    server_ready: bool = False
    fails: int = 0
    down_until: float = 0.0
    trial: bool = False
    recovering: bool = False  # the server was down; needs auto_return_after_s of health
    healthy_since: float | None = None
    probe_fails: int = 0
    last_health: HealthState | None = None
    reported: HealthState | None = None
    detail: str = ""
    ttfts: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class _Watch:
    name: str
    previous: str
    started: float
    baseline_p95: float | None
    failures: int = 0


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..1)."""
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[k]


async def _aclose(agen: Any) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:  # a provider's cleanup must never mask the real outcome
        log.debug("closing an LLM stream raised %r", exc)


class FallbackRouter:
    """``LLMRouter`` over ``providers`` (see the module docstring)."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        chain: Sequence[str] | None = None,
        consent: bool = False,
        canned_line: str = "",
        auto_rollback: Mapping[str, float] | None = None,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        servers: LocalServerManager | None = None,
        on_demand: Mapping[str, str] | None = None,
        auto_return_after_s: float = 60.0,
        breaker_max_s: float = 60.0,
        ensure_timeout_s: float = 10.0,
        probe_timeout_s: float = 2.0,
        probe_interval_s: float = 2.0,
        probe_failures: int = 3,
        min_ttft_samples: int = 5,
        flight: LLMFlightLog | None = None,
    ) -> None:
        by_name: dict[str, LLMProvider] = {}
        for p in providers:
            if p.name in by_name:
                raise ValueError(f"duplicate LLM provider name {p.name!r}")
            by_name[p.name] = p
        names = list(chain) if chain is not None else list(by_name)
        missing = [n for n in names if n not in by_name]
        if missing:
            log.info("LLM chain entries without a provider are skipped: %s", ", ".join(missing))
        self._chain = [n for n in names if n in by_name]
        ordered = self._chain + [n for n in by_name if n not in self._chain]
        demand = dict(on_demand or {})
        self._entries: dict[str, _Entry] = {}
        for n in ordered:
            p = by_name[n]
            cloud = bool(p.caps.cloud)
            self._entries[n] = _Entry(
                provider=p, name=n, cloud=cloud, enabled=consent or not cloud, server=demand.get(n)
            )
        self._canned = (
            CannedProvider(canned_line)
            if canned_line.strip() and not any(isinstance(p, CannedProvider) for p in providers)
            else None
        )
        if not self._entries and self._canned is None:
            raise ValueError("FallbackRouter needs at least one provider or a canned line")
        enabled = [n for n in ordered if self._entries[n].enabled]
        self._active = enabled[0] if enabled else (ordered[0] if ordered else self._canned_name())
        self._previous: str | None = None
        self._serving: str | None = self._active
        self._watch: _Watch | None = None
        self._consent = consent
        rb = dict(auto_rollback or {})
        self._rb_window = float(rb.get("window_s", 600.0))
        self._rb_max_failures = int(rb.get("max_failures", 3))
        self._rb_factor = float(rb.get("ttft_p95_factor", 2.0))
        self._bus = bus
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._servers = servers
        self._auto_return_after_s = auto_return_after_s
        self._breaker_max_s = breaker_max_s
        self._ensure_timeout_s = ensure_timeout_s
        self._probe_timeout_s = probe_timeout_s
        self._probe_interval_s = probe_interval_s
        self._probe_failures = probe_failures
        self._min_ttft_samples = min_ttft_samples
        self._flight = flight
        self._canned_served = False
        self._outage = False

    # --- LLMRouter ----------------------------------------------------------------------------

    def active(self) -> str:
        return self._active

    def promote(self, name: str) -> None:
        """Make ``name`` active from the next decision on; the current one is kept for
        ``rollback()``. Raises ``KeyError`` for an unknown name and ``ConsentRequired`` for a
        cloud provider without consent."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        if entry.cloud and not self._consent:
            raise ConsentRequired(
                f"{name} is a cloud provider; set privacy.cloud_llm_consent = true first "
                f"(ต้องเปิด privacy.cloud_llm_consent ก่อนใช้ {name})"
            )
        if name == self._active:
            return
        old = self._active
        old_entry = self._entries.get(old)
        baseline = None
        if old_entry is not None and len(old_entry.ttfts) >= self._min_ttft_samples:
            baseline = _percentile([ms for _, ms in old_entry.ttfts], 0.95)
        self._reset_breaker(entry)  # the operator asked for it explicitly
        self._previous, self._active, self._serving = old, name, name
        self._watch = _Watch(name, old, self._clock.now(), baseline)
        log.info("LLM promote %s -> %s (baseline TTFT p95 %s ms)", old, name, baseline)
        self._publish(ProviderSwitched(kind=KIND, old=old, new=name, reason="promote"))

    def rollback(self) -> None:
        """Return to the provider that was active before the last ``promote()``."""
        if self._previous is None:
            return
        old, new = self._active, self._previous
        self._active, self._previous, self._serving, self._watch = new, None, new, None
        log.info("LLM rollback %s -> %s", old, new)
        self._publish(ProviderSwitched(kind=KIND, old=old, new=new, reason="rollback"))

    def status(self) -> list[ProviderStatus]:
        out: list[ProviderStatus] = []
        for e in self._entries.values():
            p50 = _percentile([ms for _, ms in e.ttfts], 0.5)
            out.append(
                ProviderStatus(
                    name=e.name,
                    healthy=e.enabled and e.fails == 0 and e.last_health in (None, HealthState.OK),
                    active=e.name == self._active,
                    enabled=e.enabled,
                    cloud=e.cloud,
                    down_until=e.down_until,
                    fails=e.fails,
                    ttft_p50_ms=p50,
                )
            )
        if self._canned is not None:
            out.append(
                ProviderStatus(
                    name=self._canned.name,
                    healthy=True,
                    active=self._active == self._canned.name,
                    enabled=True,
                    cloud=False,
                    down_until=0.0,
                    fails=0,
                    ttft_p50_ms=None,
                )
            )
        return out

    async def prefill(self, req: ChatRequest) -> None:
        """Warm the KV cache of the provider that would serve the next decision (no-op when
        that is a cloud provider). Failures raise ``ProviderFailed`` and do not trip breakers."""
        for entry in self._decision_chain():
            if entry.fails or entry.recovering:
                continue
            if entry.server is not None and not entry.server_ready:
                continue
            if entry.cloud or isinstance(entry.provider, CannedProvider):
                return
            try:
                await entry.provider.prefill(req)
            except ProviderFailed:
                raise
            except Exception as exc:
                raise ProviderFailed(
                    f"{entry.name}: prefill failed: {exc!r}", emitted=False
                ) from exc
            return

    async def stream(self, req: ChatRequest) -> AsyncGenerator[LLMEvent, None]:
        """Stream from the first provider that answers (see the module docstring)."""
        queue = self._decision_chain()
        deferred: set[str] = set()
        last: BaseException | None = None
        i = 0
        while i < len(queue):
            entry = queue[i]
            i += 1
            if req.purpose != "speak" and isinstance(entry.provider, CannedProvider):
                continue  # a configured canned entry must never answer a summary or game turn
            admit = await self._admit(entry, deferred_pass=entry.name in deferred)
            if admit == "defer":
                deferred.add(entry.name)
                queue.append(entry)
                continue
            if admit == "skip":
                continue
            t0 = self._clock.now()
            emitted = False
            done: Done | None = None
            agen: Any = None
            try:
                agen = entry.provider.stream(req)
                async for ev in agen:
                    if not emitted:
                        emitted = True
                        self._on_serving(entry.name)
                    if isinstance(ev, Done):
                        done = ev
                        self._on_success(entry, ev, speak=req.purpose == "speak")
                    yield ev
                if done is None:
                    raise ProviderError(
                        f"{entry.name}: the stream ended without Done",
                        emitted=emitted,
                        reason="protocol",
                        provider=entry.name,
                    )
            except ProviderFailed as exc:
                last = exc
                was_emitted = emitted or exc.emitted
                self._on_failure(entry, exc)
                self._record(entry, req, t0, error=exc, emitted=was_emitted)
                if exc.emitted:
                    raise
                if was_emitted:
                    raise ProviderFailed(str(exc), emitted=True) from exc
                continue
            except Exception as exc:
                last = exc
                self._on_failure(entry, exc)
                self._record(entry, req, t0, error=exc, emitted=emitted)
                if emitted:
                    raise ProviderFailed(
                        f"{entry.name} failed mid-stream: {exc!r}", emitted=True
                    ) from exc
                continue
            finally:
                entry.trial = False
                await _aclose(agen)
            self._record(entry, req, t0, done=done)
            return
        self._on_all_failed(last)
        if self._canned is not None and req.purpose == "speak" and not self._canned_served:
            self._canned_served = True
            self._on_serving(self._canned.name)
            async for ev in self._canned.stream(req):
                yield ev
            return
        raise ProviderFailed(
            f"all LLM providers failed ({last})" if last else "no LLM provider is available",
            emitted=False,
        ) from last

    # --- health loop -------------------------------------------------------------------------

    async def run(self) -> None:
        """Probe local providers forever (run it under ``TaskSupervisor.spawn``)."""
        while True:
            await self.probe_all()
            await self._clock.sleep(self._probe_interval_s)

    async def probe_all(self) -> None:
        """One round of health probes of every enabled local provider."""
        for entry in list(self._entries.values()):
            if not entry.enabled or entry.cloud or entry.trial:
                continue
            if entry.server is not None and not entry.server_ready:
                continue  # an on-demand server that was never started: nothing to watch
            self._on_probe(entry, await self._probe(entry))

    async def aclose(self) -> None:
        """Close providers that own resources (HTTP clients)."""
        for entry in self._entries.values():
            closer = getattr(entry.provider, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception as exc:
                    log.debug("closing provider %s raised %r", entry.name, exc)

    # --- internals ---------------------------------------------------------------------------

    def _canned_name(self) -> str:
        return self._canned.name if self._canned is not None else "canned"

    def _decision_chain(self) -> list[_Entry]:
        first = self._entries.get(self._active)
        rest = [self._entries[n] for n in self._chain if n != self._active]
        return [e for e in ([first] if first is not None else []) + rest if e.enabled]

    async def _admit(self, entry: _Entry, *, deferred_pass: bool) -> _Admit:
        if not entry.enabled:
            return "skip"
        if entry.fails == 0:
            return await self._ensure_server(entry)
        now = self._clock.now()
        if entry.trial or now < entry.down_until:
            return "skip"
        entry.trial = True  # half-open: one probe, then one trial request
        verdict: _Admit = "skip"
        try:
            if entry.server is not None and self._servers is not None:
                verdict = await self._ensure_server(entry)
                return verdict
            health = await self._probe(entry)
            entry.last_health = health.state
            if health.state is not HealthState.OK:
                entry.healthy_since = None
                self._open(entry, f"probe: {health.detail or health.state}", server_down=True)
                return verdict
            if entry.recovering:
                if entry.healthy_since is None:
                    entry.healthy_since = now
                if now - entry.healthy_since < self._auto_return_after_s and not deferred_pass:
                    verdict = "defer"
                    return verdict
            verdict = "go"
            return verdict
        finally:
            if verdict != "go":
                entry.trial = False

    async def _ensure_server(self, entry: _Entry) -> _Admit:
        if entry.server is None or self._servers is None or entry.server_ready:
            return "go"
        detail = "did not become healthy"
        try:
            async with deadline(
                self._ensure_timeout_s + 1.0, what=f"start {entry.server}", clock=self._clock
            ):
                ok = await self._servers.ensure_running(entry.server, self._ensure_timeout_s)
        except TemplateCapsError as exc:
            ok, detail = False, str(exc)
            self._publish(Alert(level="error", message=str(exc)))
        except Exception as exc:  # incl. DeadlineExceeded; never let a start kill the turn
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            self._open(entry, f"server {entry.server}: {detail}", server_down=True)
            return "skip"
        entry.server_ready = True
        return "go"

    async def _probe(self, entry: _Entry) -> Health:
        component = f"llm:{entry.name}"
        try:
            async with deadline(
                self._probe_timeout_s, what=f"{component} probe", clock=self._clock
            ):
                return await entry.provider.probe()
        except DeadlineExceeded:
            return Health(component, HealthState.DOWN, "probe timed out", self._clock.now())
        except Exception as exc:
            return Health(component, HealthState.DOWN, f"probe failed: {exc!r}", self._clock.now())

    def _on_probe(self, entry: _Entry, health: Health) -> None:
        entry.last_health = health.state
        if health.state is HealthState.OK:
            entry.probe_fails = 0
            if entry.fails and entry.recovering and entry.healthy_since is None:
                entry.healthy_since = self._clock.now()
            if entry.fails == 0:
                self._report(entry, HealthState.OK, "")
            return
        entry.probe_fails += 1
        entry.healthy_since = None
        if entry.fails:
            entry.recovering = True
        elif entry.probe_fails >= self._probe_failures:
            self._open(entry, f"health: {health.detail or health.state}", server_down=True)

    def _open(self, entry: _Entry, reason: str, *, server_down: bool) -> None:
        entry.fails += 1
        wait = min(self._breaker_max_s, 2.0**entry.fails)
        entry.down_until = self._clock.now() + wait
        entry.server_ready = False
        entry.detail = reason
        if server_down:
            entry.recovering = True
            entry.healthy_since = None
        log.warning(
            "LLM %s breaker open for %.0f s (fail %d): %s", entry.name, wait, entry.fails, reason
        )
        self._report(entry, HealthState.DOWN if server_down else HealthState.DEGRADED, reason)
        self._watch_failure(entry)

    def _reset_breaker(self, entry: _Entry) -> None:
        entry.fails = 0
        entry.down_until = 0.0
        entry.recovering = False
        entry.healthy_since = None
        entry.probe_fails = 0
        entry.detail = ""

    def _on_failure(self, entry: _Entry, exc: BaseException) -> None:
        reason = exc.reason if isinstance(exc, ProviderError) else "error"
        self._open(entry, str(exc) or type(exc).__name__, server_down=reason == "connect")

    def _on_success(self, entry: _Entry, done: Done, *, speak: bool) -> None:
        if entry.fails:
            log.info("LLM %s recovered after %d failure(s)", entry.name, entry.fails)
        self._reset_breaker(entry)
        self._report(entry, HealthState.OK, "")
        if self._outage:
            self._outage = False
            self._publish(HealthChanged(health=Health(KIND, HealthState.OK, "", self._clock.now())))
        self._canned_served = False
        if speak and done.ttft_ms >= 0.0:
            # only speech turns: long background/game prompts would skew the TTFT baseline
            entry.ttfts.append((self._clock.now(), done.ttft_ms))
            self._watch_ttft(entry)

    def _on_serving(self, name: str) -> None:
        old = self._serving
        self._serving = name
        if old is None or old == name:
            return
        reason = "recovered" if name == self._active else "fallback"
        log.info("LLM now served by %s (%s, was %s)", name, reason, old)
        self._publish(ProviderSwitched(kind=KIND, old=old, new=name, reason=reason))

    def _on_all_failed(self, last: BaseException | None) -> None:
        if self._outage:
            return
        self._outage = True
        detail = str(last) if last is not None else "no provider available"
        self._publish(
            HealthChanged(health=Health(KIND, HealthState.DOWN, detail, self._clock.now()))
        )
        self._publish(
            Alert(
                level="error",
                message=f"LLM: ทุกผู้ให้บริการใช้งานไม่ได้ / every LLM provider failed: {detail}",
            )
        )

    def _report(self, entry: _Entry, state: HealthState, detail: str) -> None:
        if entry.reported is state:
            return
        entry.reported = state
        self._publish(
            HealthChanged(health=Health(f"llm:{entry.name}", state, detail, self._clock.now()))
        )

    # auto-rollback

    def _live_watch(self) -> _Watch | None:
        w = self._watch
        if w is not None and self._clock.now() - w.started > self._rb_window:
            self._watch = None
            return None
        return w

    def _watch_failure(self, entry: _Entry) -> None:
        w = self._live_watch()
        if w is None or w.name != entry.name:
            return
        w.failures += 1
        if w.failures >= self._rb_max_failures:
            self._auto_rollback(f"{w.failures} failures within {self._rb_window:g} s")

    def _watch_ttft(self, entry: _Entry) -> None:
        w = self._live_watch()
        if w is None or w.name != entry.name or w.baseline_p95 is None:
            return
        samples = [ms for t, ms in entry.ttfts if t >= w.started]
        if len(samples) < self._min_ttft_samples:
            return
        p95 = _percentile(samples, 0.95)
        if p95 is not None and p95 > self._rb_factor * w.baseline_p95:
            self._auto_rollback(
                f"TTFT p95 {p95:.0f} ms > {self._rb_factor:g}× baseline {w.baseline_p95:.0f} ms"
            )

    def _auto_rollback(self, why: str) -> None:
        w, self._watch = self._watch, None
        if w is None or self._active != w.name:
            return
        old, new = w.name, w.previous
        self._active, self._previous, self._serving = new, None, new
        log.warning("LLM auto-rollback %s -> %s: %s", old, new, why)
        self._publish(ProviderSwitched(kind=KIND, old=old, new=new, reason="auto_rollback"))
        self._publish(Alert(level="warn", message=f"LLM auto-rollback {old} → {new}: {why}"))

    # observability

    def _publish(self, event: Event) -> None:
        if self._bus is not None:
            self._bus.publish(event)

    def _record(
        self,
        entry: _Entry,
        req: ChatRequest,
        t0: float,
        *,
        done: Done | None = None,
        error: BaseException | None = None,
        emitted: bool = False,
    ) -> None:
        if self._flight is None:
            return
        summary: dict[str, Any] = {
            "provider": entry.name,
            "purpose": req.purpose,
            "character": req.character,
            "turn_id": req.turn_id,
            "messages": len(req.messages),
            "tools": len(req.tools),
            "ok": error is None,
            "elapsed_ms": round((self._clock.now() - t0) * 1000.0, 1),
        }
        if done is not None:
            summary |= {
                "ttft_ms": round(done.ttft_ms, 1),
                "prompt_n": done.prompt_n,
                "cache_n": done.cache_n,
                "completion_tokens": done.completion_tokens,
                "finish_reason": done.finish_reason,
            }
        if error is not None:
            summary |= {"error": str(error)[:300], "emitted": emitted}
        try:
            self._flight.record_llm(summary)
        except Exception as exc:
            log.debug("flight recorder rejected an LLM summary: %r", exc)


def build_router(
    cfg: AppConfig,
    providers: Sequence[LLMProvider],
    *,
    bus: EventBus | None,
    clock: Clock | None = None,
    servers: LocalServerManager | None = None,
    flight: LLMFlightLog | None = None,
) -> FallbackRouter:
    """A ``FallbackRouter`` configured from ``cfg.llm`` (chain, consent, canned line,
    auto-rollback, breaker cap, auto-return) and ``cfg.launcher`` (health cadence)."""
    llm = cfg.llm
    on_demand = {
        name: p.server
        for name, p in llm.providers.items()
        if p.server is not None and llm.servers[p.server].autostart == "on_demand"
    }
    return FallbackRouter(
        providers,
        chain=llm.chain,
        consent=cfg.privacy.cloud_llm_consent,
        canned_line=llm.canned_line,
        auto_rollback=llm.auto_rollback.model_dump(),
        bus=bus,
        clock=clock,
        servers=servers,
        on_demand=on_demand,
        auto_return_after_s=llm.auto_return_after_s,
        breaker_max_s=llm.breaker_max_s,
        probe_interval_s=cfg.launcher.llm_health_interval_s,
        probe_failures=cfg.launcher.llm_health_failures,
        flight=flight,
    )
