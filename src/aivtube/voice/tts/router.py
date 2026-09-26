"""TTS router: voice identities, per-segment substitution, breakers, quotas and captions (§2.8).

An **identity** is one voice (e.g. ``premwadee`` = ``th-TH-PremwadeeNeural`` +20Hz +8%) that
several backends can produce (``[edge, azure]``). Rules:

- Each utterance picks one identity at ``begin_utterance`` and keeps it: the voice may change
  **between** utterances, never within one.
- Per segment, backends of that identity are tried in order. A backend gets
  ``first_timeout_s`` (2 s, 0 retries) for the utterance's first segment and
  ``later_timeout_s`` (4 s, 1 retry) for later ones to deliver first audio. Word marks that
  arrive before the first audio are held back, so a failed attempt leaves nothing behind.
- A per-backend breaker takes a backend out for ``cooldown`` (120 s) after ``failures`` (3)
  failures within ``window`` (60 s); after the cooldown one trial decides (half-open).
- A quota-limited backend (Azure F0) is skipped for the segment when its token bucket is
  empty (never waited for), and is never used for speculative synthesis.
- A backend flagged ``whole_utterance`` (Piper) never substitutes mid-utterance.
- When every backend of the identity fails for a segment, the rest of the utterance is
  captions-only (silent segments) and the identity is demoted, so the next utterance uses the
  next identity in the character's chain; with no other identity it retries the same one.
- A segment whose text is in the phrase cache (for the utterance's exact voice) is served
  from the cache without any network request.

Mid-stream failures (audio already delivered) end the segment early (``truncated``): audio
cannot be spliced from another backend.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from aivtube.contracts.infra import Clock
from aivtube.contracts.speech import TTSConstraints
from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, PhraseCache, TTSBackend, TTSUnavailable, WordMark
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.voice.tts.cache import DiskPhraseCache, phrase_key
from aivtube.voice.tts.quota import TokenBucket

__all__ = [
    "CAPTIONS",
    "BackendBreaker",
    "ChunkLimits",
    "IdentityCfg",
    "SynthStream",
    "TTSRouter",
    "UtteranceTTS",
    "shift_rate",
]

log = logging.getLogger("aivtube.voice.tts.router")

CAPTIONS = "captions"
_RATE = re.compile(r"^([+-]?\d+)%$")
_GRACE_S = 0.5  # router-side guard on top of the backend's own deadlines


@dataclass(frozen=True, slots=True)
class IdentityCfg:
    """A voice identity and the backends that can speak it, in preference order."""

    name: str
    voice: VoiceSpec
    backends: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkLimits:
    """Chunk sizes advertised in ``TTSConstraints`` (``[tts.chunker]``)."""

    first_min_chars: int = 8
    min_chars: int = 40
    max_chars: int = 160


def shift_rate(rate: str, percent: int) -> str:
    """``shift_rate("+8%", 10) == "+18%"`` (prosody rate strings)."""
    m = _RATE.match(rate.strip())
    base = int(m.group(1)) if m else 0
    value = base + int(percent)
    return f"{value:+d}%"


class BackendBreaker:
    """``failures`` failures within ``window_s`` open the breaker for ``cooldown_s``."""

    def __init__(
        self,
        clock: Callable[[], float],
        failures: int = 3,
        window_s: float = 60.0,
        cooldown_s: float = 120.0,
    ) -> None:
        self._clock = clock
        self.failures = failures
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._times: collections.deque[float] = collections.deque()
        self.open_until = float("-inf")
        self._half_open = False
        self.opened = 0

    def is_open(self) -> bool:
        now = self._clock()
        if now < self.open_until:
            return True
        if self.open_until != float("-inf") and not self._half_open:
            self._half_open = True  # cooldown over: the next result decides
        return False

    def record_success(self) -> None:
        self._half_open = False
        self.open_until = float("-inf")
        self._times.clear()

    def record_failure(self) -> bool:
        """Count a failure; ``True`` if this opened the breaker."""
        now = self._clock()
        if self._half_open:
            self._half_open = False
            self._open(now)
            return True
        self._times.append(now)
        while self._times and self._times[0] < now - self.window_s:
            self._times.popleft()
        if len(self._times) >= self.failures:
            self._open(now)
            return True
        return False

    def _open(self, now: float) -> None:
        self.open_until = now + self.cooldown_s
        self._times.clear()
        self.opened += 1


class SynthStream:
    """One segment's synthesis: an async iterator of ``AudioChunk | WordMark`` with metadata.

    After iteration: ``backend`` (who produced the audio; ``"cache"`` for a phrase-cache hit,
    ``None`` if silent), ``silent`` (no audio at all), ``captions`` (silent because the
    identity is exhausted), ``truncated`` (the backend died mid-stream), ``deferred``
    (speculative synthesis was not allowed on the backends left; synthesise again once the
    utterance's gate opens) and ``error``.
    """

    def __init__(self) -> None:
        self.backend: str | None = None
        self.silent = False
        self.captions = False
        self.truncated = False
        self.cached = False
        self.deferred = False
        self.error: str | None = None
        self._gen: AsyncIterator[AudioChunk | WordMark] | None = None

    def __aiter__(self) -> SynthStream:
        return self

    async def __anext__(self) -> AudioChunk | WordMark:
        assert self._gen is not None
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        gen = self._gen
        if gen is not None:
            close = getattr(gen, "aclose", None)
            if close is not None:
                await close()


class UtteranceTTS:
    """Synthesis for one utterance with a fixed identity and voice."""

    def __init__(
        self,
        router: TTSRouter,
        character: str,
        identity: IdentityCfg | None,
        voice: VoiceSpec | None,
        *,
        demote_on_exhaust: bool = True,
    ) -> None:
        self._router = router
        self.character = character
        self._identity = identity
        self.identity = identity.name if identity is not None else CAPTIONS
        self.voice = voice
        self._captions = identity is None
        self._demote = demote_on_exhaust
        self.degraded = identity is None
        self.backends_used: list[str] = []
        self.fallbacks = 0

    @property
    def captions_only(self) -> bool:
        return self._captions

    def synth(self, text: str, *, first: bool, speculative: bool = False) -> SynthStream:
        """Stream one segment (see ``SynthStream`` for the metadata it carries)."""
        stream = SynthStream()
        stream._gen = self._run(stream, text, first, speculative)
        return stream

    open = synth

    async def _run(
        self, stream: SynthStream, text: str, first: bool, speculative: bool
    ) -> AsyncIterator[AudioChunk | WordMark]:
        if not text.strip():
            stream.silent = True
            return
        if self.voice is not None:
            hit = self._router.cached(self.voice, text)
            if hit is not None:
                audio, marks = hit
                stream.backend, stream.cached = "cache", True
                for mark in marks:
                    yield mark
                yield audio
                return
        if self._captions or self._identity is None or self.voice is None:
            stream.silent = stream.captions = True
            return
        router = self._router
        timeout = router.first_timeout_s if first else router.later_timeout_s
        attempts = 1 + (router.first_retries if first else router.later_retries)
        idle = router.idle_timeout_s
        reasons: list[str] = []
        previous: str | None = None
        spec_skipped = False
        for name in self._identity.backends:
            backend = router.backends.get(name)
            if backend is None:
                continue
            if router.breaker(name).is_open():
                reasons.append(f"{name}: breaker open")
                continue
            if getattr(backend, "whole_utterance", False) and any(
                b != name for b in self.backends_used
            ):
                reasons.append(f"{name}: whole utterances only")
                continue
            if backend.quota is not None and speculative:
                reasons.append(f"{name}: no speculative synthesis on a quota")
                spec_skipped = True
                continue
            for _attempt in range(attempts):
                bucket = router.bucket(name)
                if bucket is not None and not bucket.try_take():
                    reasons.append(f"{name}: quota exhausted")
                    break
                if previous is not None and previous != name:
                    self.fallbacks += 1
                    router.report_fallback(previous, name, reasons[-1] if reasons else "failed")
                previous = name
                started = False
                gen = backend.synth(
                    text, self.voice, first_audio_timeout=timeout, idle_timeout=idle
                )
                pending: list[WordMark] = []
                t0 = router.clock.now()
                try:
                    while True:
                        if started:
                            budget = idle + _GRACE_S
                        else:
                            budget = timeout + _GRACE_S - (router.clock.now() - t0)
                            if budget <= 0:
                                raise TTSUnavailable(f"{name}: no audio within {timeout:g} s")
                        try:
                            async with deadline(budget, what=f"tts {name}", clock=router.clock):
                                item = await anext(gen)
                        except StopAsyncIteration:
                            break
                        if isinstance(item, WordMark):
                            if started:
                                yield item
                            else:
                                pending.append(item)
                            continue
                        if not started:
                            started = True
                            stream.backend = name
                            if name not in self.backends_used:
                                self.backends_used.append(name)
                            for mark in pending:
                                yield mark
                            pending.clear()
                        yield item
                    if not started:
                        raise TTSUnavailable(f"{name}: no audio")
                    router.breaker(name).record_success()
                    router.check_constraints()
                    return
                except Exception as exc:  # TTSUnavailable, DeadlineExceeded, backend bugs
                    reason = f"{name}: {exc}"
                    if started:  # audio already delivered: end this segment early
                        stream.truncated = True
                        stream.error = reason
                        log.warning("TTS %s died mid-segment: %s", name, exc)
                        router.record_failure(name)
                        return
                    reasons.append(reason)
                    log.info("TTS attempt failed: %s", reason)
                    router.record_failure(name)
                finally:
                    await _aclose(gen)
                if router.breaker(name).is_open():
                    break
        if spec_skipped:  # not a failure of the voice: try again once the gate opens
            stream.silent = stream.deferred = True
            stream.error = "; ".join(reasons)
            return
        # every backend of the identity failed for this segment
        self._captions = True
        self.degraded = True
        stream.silent = stream.captions = True
        stream.error = "; ".join(reasons) or "no backend available"
        router.report_fallback(
            previous or self.identity, None, f"identity exhausted: {stream.error}"
        )
        if self._demote:
            router.demote(self.identity)


async def _aclose(gen: AsyncIterator[Any]) -> None:
    close = getattr(gen, "aclose", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        async with asyncio.timeout(2.0):
            await close()


class TTSRouter:
    """Picks identities per utterance and backends per segment (see the module docstring)."""

    def __init__(
        self,
        identities: Mapping[str, IdentityCfg],
        backends: Mapping[str, TTSBackend],
        chain_by_character: Mapping[str, Sequence[str]],
        *,
        first_timeout_s: float = 2.0,
        later_timeout_s: float = 4.0,
        first_retries: int = 0,
        later_retries: int = 1,
        idle_timeout_s: float | None = None,
        captions_fallback: bool = True,
        breaker: tuple[int, float, float] = (3, 60.0, 120.0),
        demote_s: float | None = None,
        chunk: ChunkLimits | None = None,
        min_chars_by_backend: Mapping[str, int] | None = None,
        default_chain: Sequence[str] | None = None,
        cache: PhraseCache | None = None,
        clock: Clock | None = None,
        on_fallback: Callable[[str, str | None, str], None] | None = None,
        on_constraints: Callable[[str, TTSConstraints], None] | None = None,
    ) -> None:
        self.identities = dict(identities)
        self.backends = dict(backends)
        self.chains = {c: list(chain) for c, chain in chain_by_character.items()}
        self.default_chain = list(default_chain) if default_chain else list(self.identities)
        self.first_timeout_s = first_timeout_s
        self.later_timeout_s = later_timeout_s
        self.first_retries = first_retries
        self.later_retries = later_retries
        self.idle_timeout_s = idle_timeout_s if idle_timeout_s is not None else later_timeout_s
        self.captions_fallback = captions_fallback
        self.chunk = chunk or ChunkLimits()
        self.min_chars_by_backend = dict(min_chars_by_backend or {})
        self.cache = cache
        self.clock: Clock = clock or SystemClock()
        self._breaker_cfg = breaker
        self._breakers: dict[str, BackendBreaker] = {}
        self._buckets: dict[str, TokenBucket | None] = {}
        self._demoted: dict[str, float] = {}
        self.demote_s = demote_s if demote_s is not None else breaker[2]
        self._rates: dict[str, int] = {}
        self._on_fallback = on_fallback
        self._on_constraints = on_constraints
        self._last_constraints: dict[str, TTSConstraints] = {}
        for ident in self.identities.values():
            missing = [b for b in ident.backends if b not in self.backends]
            if missing:
                log.warning("identity %s: backends %s are not available", ident.name, missing)

    # --- per-utterance --------------------------------------------------------------------
    def begin_utterance(self, character: str) -> UtteranceTTS:
        ident = self._pick(character)
        if ident is None:
            if not self.captions_fallback:
                log.error("no TTS identity available for %s and captions are off", character)
            return UtteranceTTS(self, character, None, None)
        return UtteranceTTS(self, character, ident, self.voice_for(character, ident.name))

    def voice_for(self, character: str, identity: str) -> VoiceSpec:
        """The identity's voice with the character's talking-speed offset applied."""
        voice = self.identities[identity].voice
        percent = self._rates.get(character, 0)
        return replace(voice, rate=shift_rate(voice.rate, percent)) if percent else voice

    def set_rate(self, character: str, percent: int) -> None:
        """Talking speed offset in percent (applies from the next utterance)."""
        self._rates[character] = int(percent)

    def chain_for(self, character: str) -> list[str]:
        return [i for i in self.chains.get(character, self.default_chain) if i in self.identities]

    # --- constraints ----------------------------------------------------------------------
    def constraints(self, character: str) -> TTSConstraints:
        ident = self._pick(character)
        backend = self._primary(ident) if ident is not None else None
        min_chars = self.chunk.min_chars
        if backend is not None:
            min_chars = max(min_chars, self.min_chars_by_backend.get(backend, 0))
        return TTSConstraints(
            first_min_chars=self.chunk.first_min_chars,
            min_chars=min(min_chars, self.chunk.max_chars),
            max_chars=self.chunk.max_chars,
            backend=backend or CAPTIONS,
            identity=ident.name if ident is not None and backend is not None else CAPTIONS,
        )

    def check_constraints(self) -> None:
        """Notify ``on_constraints`` for every character whose constraints changed."""
        if self._on_constraints is None:
            return
        for character in list(self.chains) or ["default"]:
            c = self.constraints(character)
            if self._last_constraints.get(character) != c:
                self._last_constraints[character] = c
                try:
                    self._on_constraints(character, c)
                except Exception:
                    log.exception("on_constraints callback failed")

    # --- cache ----------------------------------------------------------------------------
    def cached(self, voice: VoiceSpec, text: str) -> tuple[AudioChunk, tuple[WordMark, ...]] | None:
        """Phrase-cache lookup for ``text`` in exactly this voice."""
        if self.cache is None:
            return None
        key = phrase_key(voice, text)
        try:
            with_marks = getattr(self.cache, "get_with_marks", None)
            if with_marks is not None:
                got: tuple[AudioChunk, tuple[WordMark, ...]] | None = with_marks(key)
                return got
            audio = self.cache.get(key)
        except Exception:
            log.exception("phrase cache lookup failed")
            return None
        return (audio, ()) if audio is not None else None

    def cached_phrase(
        self, character: str, text: str
    ) -> tuple[AudioChunk, tuple[WordMark, ...]] | None:
        """The cached phrase in the voice the character's next utterance would use."""
        ident = self._pick(character, ignore_breakers=True)
        if ident is None:
            return None
        return self.cached(self.voice_for(character, ident.name), text)

    async def presynthesize(
        self, character: str, phrases: Sequence[str], *, timeout_s: float = 15.0
    ) -> dict[str, bool]:
        """Make sure each phrase is cached for the character's current identity.

        Returns ``{phrase: available}``. Synthesis failures here never demote identities.
        """
        result: dict[str, bool] = {}
        ident = self._pick(character, ignore_breakers=True)
        if ident is None or self.cache is None:
            return {p: False for p in phrases}
        voice = self.voice_for(character, ident.name)
        for phrase in phrases:
            if not phrase.strip():
                continue
            if self.cached(voice, phrase) is not None:
                result[phrase] = True
                continue
            utt = UtteranceTTS(self, character, ident, voice, demote_on_exhaust=False)
            chunks: list[np.ndarray] = []
            marks: list[WordMark] = []
            rate = 0
            stream = utt.synth(phrase, first=False)
            try:
                async with deadline(timeout_s, what="phrase pre-synthesis", clock=self.clock):
                    async for item in stream:
                        if isinstance(item, AudioChunk):
                            chunks.append(item.pcm)
                            rate = item.sample_rate
                        else:
                            marks.append(item)
            except DeadlineExceeded:
                log.warning("pre-synthesis of %r timed out", phrase)
            finally:
                await stream.aclose()
            ok = bool(chunks) and not stream.truncated and not stream.silent
            if ok:
                pcm = np.concatenate(chunks).astype(np.int16)
                key = phrase_key(voice, phrase)
                try:
                    if isinstance(self.cache, DiskPhraseCache):
                        self.cache.put(key, AudioChunk(pcm, rate), marks)
                    else:
                        self.cache.put(key, AudioChunk(pcm, rate))
                except Exception:
                    log.exception("phrase cache write failed")
                    ok = False
            result[phrase] = ok
        return result

    # --- lifecycle ------------------------------------------------------------------------
    async def warmup(self, timeout_s: float = 10.0) -> dict[str, bool]:
        """Warm every backend (bounded); returns ``{backend: ok}``."""
        out: dict[str, bool] = {}
        for name, backend in self.backends.items():
            try:
                async with deadline(timeout_s, what=f"{name} warm-up", clock=self.clock):
                    await backend.warmup()
                out[name] = True
            except Exception as exc:
                log.warning("TTS backend %s failed to warm up: %s", name, exc)
                out[name] = False
        return out

    async def aclose(self) -> None:
        for backend in self.backends.values():
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2.0):
                    await backend.aclose()

    def status(self) -> list[dict[str, Any]]:
        """Backend states for health reports and the panel."""
        now = self.clock.now()
        out = []
        for name in self.backends:
            br = self.breaker(name)
            bucket = self.bucket(name)
            out.append(
                {
                    "backend": name,
                    "open": br.is_open(),
                    "down_for_s": max(0.0, br.open_until - now),
                    "tokens": None if bucket is None else round(bucket.tokens, 2),
                }
            )
        return out

    # --- internals used by UtteranceTTS ---------------------------------------------------
    def breaker(self, name: str) -> BackendBreaker:
        br = self._breakers.get(name)
        if br is None:
            n, window, cooldown = self._breaker_cfg
            br = BackendBreaker(self.clock.now, n, window, cooldown)
            self._breakers[name] = br
        return br

    def bucket(self, name: str) -> TokenBucket | None:
        if name not in self._buckets:
            backend = self.backends.get(name)
            quota = getattr(backend, "quota", None)
            self._buckets[name] = (
                TokenBucket.from_quota(quota, self.clock.now) if quota is not None else None
            )
        return self._buckets[name]

    def record_failure(self, name: str) -> None:
        if self.breaker(name).record_failure():
            n, window, cooldown = self._breaker_cfg
            log.warning(
                "TTS backend %s out for %.0f s (%d failures in %.0f s)", name, cooldown, n, window
            )
        self.check_constraints()

    def report_fallback(self, frm: str, to: str | None, reason: str) -> None:
        log.info("TTS fallback %s -> %s: %s", frm, to or CAPTIONS, reason)
        if self._on_fallback is not None:
            try:
                self._on_fallback(frm, to, reason)
            except Exception:
                log.exception("on_fallback callback failed")

    def demote(self, identity: str) -> None:
        self._demoted[identity] = self.clock.now() + self.demote_s
        self.check_constraints()

    def _available(self, ident: IdentityCfg) -> bool:
        return any(b in self.backends and not self.breaker(b).is_open() for b in ident.backends)

    def _primary(self, ident: IdentityCfg) -> str | None:
        for b in ident.backends:
            if b in self.backends and not self.breaker(b).is_open():
                return b
        return None

    def _pick(self, character: str, *, ignore_breakers: bool = False) -> IdentityCfg | None:
        chain = [self.identities[i] for i in self.chain_for(character)]
        now = self.clock.now()
        usable = [i for i in chain if ignore_breakers or self._available(i)]
        for ident in usable:
            if self._demoted.get(ident.name, float("-inf")) <= now:
                return ident
        return usable[0] if usable else None
