"""The streaming reply pipeline (ARCHITECTURE.md §4.6, §4.10).

```
LLM TextDelta ─► EmotionTagExtractor ─► ThaiSpeechChunker (stall 500 ms → cut at the last space)
  ─► is_speakable? (no → merge forward) ─► SafetyGate.check_output(chunk, prev_tail=last 40)
       BLOCK/DROP → Filtered path · PASS/MASK/REPLACE/REVIEW → normalize → Segment → speech
ToolCalls accumulate; they are executed by ToolFlow after the stream ends.
```

- The chunker config is rebuilt per utterance from ``speech.constraints(character)``.
- The emotion goes only on the first segment after it changes.
- Consumption waits on one ``__anext__`` at a time, so backpressure (``segment()`` returning
  ``False``) pauses the LLM stream without dropping text.
- An inter-token stall of 3 s ends the reply with "…" (``stalled=True``); so does a provider
  that dies after emitting (``provider_failed=True``). A provider that fails before emitting
  leaves the utterance empty, and the brain decides what to say (the brain-freeze line).
- **Filtered.** On a blocked chunk the LLM stream is closed first, then
  ``speech.stop(utt, "after_segment")`` drops every segment not yet started (the one playing
  was clean) and ``play_canned("filtered")`` says "Filtered.". The blocked text is never part
  of the result, so it never reaches history or memory.
- The utterance always ends with a ``last=True`` segment (an empty marker when the final text
  was already sent), except on the Filtered path, where ``stop`` ends it.
- The LLM iterator is closed in ``finally`` so the server cancels generation (≤ 0.2 s).
- Before the first event the router owns the per-provider first-token deadlines and the
  fallback (§4.11/§4.12); only ``first_event_timeout_s`` (30 s) bounds that wait here.
- The pipeline publishes one ``Filtered`` per blocked reply, tied to the utterance
  (``ref=utt_id``). ``LayeredSafetyGate`` also publishes its own (``ref`` = text hash); pass
  ``publish_filtered=False`` if the panel should count each block once.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Final

from aivtube.brain._util import guarded, wait_future
from aivtube.contracts.events import Filtered, LatencyMark, LLMFirstToken, SegmentQueued
from aivtube.contracts.infra import Clock, EventBus
from aivtube.contracts.llm import Done, LLMEvent, ProviderFailed, TextDelta, ToolCall
from aivtube.contracts.safety import FilterResult, SafetyGate, Verdict
from aivtube.contracts.speech import SpeechOutput
from aivtube.contracts.types import Segment
from aivtube.infra.clock import DeadlineExceeded
from aivtube.infra.trace import TurnTraceRecorder
from aivtube.text.chunker import ChunkerConfig, ThaiSpeechChunker, WordTokenizer
from aivtube.text.normalize import is_speakable, normalize_cloud
from aivtube.text.tags import EmotionTagExtractor

__all__ = ["ELLIPSIS", "ReplyPipeline", "ReplyProgress", "ReplyResult"]

log = logging.getLogger("aivtube.brain.reply")

ELLIPSIS: Final = "…"
_STOPPING: Final = frozenset({Verdict.BLOCK, Verdict.DROP})


@dataclass(frozen=True, slots=True)
class ReplyResult:
    """``emitted_text`` is what passed the gate and went to speech (captions, joined)."""

    emitted_text: str
    segments: int
    tool_calls: tuple[ToolCall, ...]
    filtered: bool
    done: Done | None
    stalled: bool
    provider_failed: bool
    sent: tuple[Segment, ...] = ()
    blocked: FilterResult | None = None
    error: str | None = None

    @property
    def spoke(self) -> bool:
        """Some speakable text went to speech."""
        return any(s.text.strip() for s in self.sent)


@dataclass(slots=True)
class ReplyProgress:
    """A live view of a reply, readable after the decision was cancelled mid-stream."""

    utt_id: str = ""
    sent: list[Segment] = field(default_factory=list)
    emitted: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    filtered: bool = False
    ended: bool = False  # the last segment (or stop) closed the utterance

    @property
    def emitted_text(self) -> str:
        return "".join(self.emitted)

    def text_of(self, seq: int) -> str | None:
        for seg in self.sent:
            if seg.seq == seq:
                return seg.text
        return None


@dataclass(slots=True)
class _State:
    character: str
    utt_id: str
    turn_id: str
    progress: ReplyProgress
    it: AsyncIterator[LLMEvent]
    seq: int
    pending: asyncio.Future[LLMEvent] | None = None
    closed: bool = False
    carry: str = ""
    tail: str = ""
    emotion: str | None = None
    first_event: bool = True
    first_chunk: bool = True
    first_filter: bool = True
    done: Done | None = None
    stalled: bool = False
    provider_failed: bool = False
    blocked: FilterResult | None = None
    error: str | None = None
    speech_failed: bool = False  # the voice side stopped taking segments: send nothing more
    tool_calls: list[ToolCall] = field(default_factory=list)


async def _anext(it: AsyncIterator[LLMEvent]) -> LLMEvent:
    return await it.__anext__()


class ReplyPipeline:
    """Turns one LLM stream into speech segments; see the module docstring."""

    def __init__(
        self,
        *,
        speech: SpeechOutput,
        gate: SafetyGate,
        bus: EventBus,
        clock: Clock,
        trace: TurnTraceRecorder,
        known_emotions: frozenset[str],
        stall_ms: int = 500,
        prev_tail_chars: int = 40,
        normalizer: Callable[[str], str] = normalize_cloud,
        inter_token_timeout_s: float = 3.0,
        word_tokenize: WordTokenizer | None = None,
        chunker_base: ChunkerConfig | None = None,
        speech_timeout_s: float = 2.0,
        gate_timeout_s: float = 1.0,
        backpressure_timeout_s: float = 30.0,
        publish_filtered: bool = True,
        first_event_timeout_s: float = 30.0,
    ) -> None:
        self._speech = speech
        self._gate = gate
        self._bus = bus
        self._clock = clock
        self._trace = trace
        self._known = frozenset(known_emotions)
        self._prev_tail = max(0, prev_tail_chars)
        self._normalize = normalizer
        self._stall_s = inter_token_timeout_s
        self._first_s = max(first_event_timeout_s, inter_token_timeout_s)
        self._tokenize = word_tokenize
        base = chunker_base or ChunkerConfig()
        self._base = ChunkerConfig(
            first_min_chars=base.first_min_chars,
            first_max_chars=base.first_max_chars,
            min_chars=base.min_chars,
            max_chars=base.max_chars,
            strong_min_chars=base.strong_min_chars,
            stall_flush_s=max(0.05, stall_ms / 1000.0),
        )
        self._speech_timeout_s = speech_timeout_s
        self._gate_timeout_s = gate_timeout_s
        self._bp_timeout_s = backpressure_timeout_s
        self._publish_filtered = publish_filtered
        self._emotions: dict[str, str | None] = {}

    @property
    def normalizer(self) -> Callable[[str], str]:
        """The TTS normaliser segments are built with (``normalize_cloud`` or ``_local``)."""
        return self._normalize

    def reset_emotion(self, character: str, emotion: str | None = "neutral") -> None:
        """The avatar went back to ``emotion`` (e.g. neutral after "Filtered.")."""
        self._emotions[character] = emotion

    async def run(
        self,
        *,
        character: str,
        utt_id: str,
        turn_id: str,
        events: AsyncIterator[LLMEvent],
        progress: ReplyProgress | None = None,
    ) -> ReplyResult:
        prog = progress if progress is not None else ReplyProgress()
        prog.utt_id = utt_id
        st = _State(character, utt_id, turn_id, prog, events.__aiter__(), seq=len(prog.sent))
        cfg = ChunkerConfig.from_constraints(self._speech.constraints(character), base=self._base)
        chunker = ThaiSpeechChunker(cfg, word_tokenize=self._tokenize, clock=self._clock)
        tags = EmotionTagExtractor(self._known, current=self._emotions.get(character))
        last_event = self._clock.now()
        try:
            while not st.closed:
                if st.pending is None:
                    st.pending = asyncio.ensure_future(_anext(st.it))
                now = self._clock.now()
                # before the first event the router owns the (per-provider) first-token
                # deadlines and the fallback; only a generous safety bound applies here
                limit = self._first_s if st.first_event else self._stall_s
                wait = last_event + limit - now
                stall_at = chunker.stall_deadline()
                if stall_at is not None:
                    wait = min(wait, stall_at - now)
                if not await wait_future(st.pending, self._clock, max(0.0, wait)):
                    now = self._clock.now()
                    stall_at = chunker.stall_deadline()
                    if stall_at is not None and now >= stall_at:
                        if not await self._send(st, chunker.poll(now), final=False):
                            break
                        last_event = max(last_event, now - self._stall_s / 2)
                    if now - last_event >= limit:
                        st.stalled = True
                        log.warning("reply %s: LLM stalled for %.1f s", utt_id, limit)
                        break
                    continue
                task, st.pending = st.pending, None
                try:
                    ev = task.result()
                except StopAsyncIteration:
                    break
                except ProviderFailed as exc:
                    st.provider_failed, st.error = True, str(exc)
                    break
                except Exception as exc:  # an adapter bug or I/O error: end the reply
                    st.provider_failed, st.error = True, f"{type(exc).__name__}: {exc}"
                    log.warning("reply %s: LLM stream failed: %s", utt_id, st.error)
                    break
                last_event = self._clock.now()
                if st.first_event:
                    st.first_event = False
                    self._trace.mark(turn_id, "llm_first_token")
                if isinstance(ev, TextDelta):
                    if not await self._feed(st, tags, chunker, ev.text):
                        break
                    last_event = self._clock.now()  # backpressure time is not an LLM stall
                elif isinstance(ev, ToolCall):
                    st.tool_calls.append(ev)
                    prog.tool_calls.append(ev)
                elif isinstance(ev, Done):
                    st.done = ev
                    self._on_done(st, ev)
            if st.blocked is None:
                await self._finish(st, tags, chunker)
        finally:
            await self._close_stream(st)
            chunker.reset()
            if st.blocked is not None:
                self._emotions[character] = "neutral"
            else:
                self._emotions[character] = tags.last_emotion
        return ReplyResult(
            emitted_text=prog.emitted_text,
            segments=len(prog.sent),
            tool_calls=tuple(st.tool_calls),
            filtered=st.blocked is not None,
            done=st.done,
            stalled=st.stalled,
            provider_failed=st.provider_failed,
            sent=tuple(prog.sent),
            blocked=st.blocked,
            error=st.error,
        )

    # --- stream end -------------------------------------------------------------------------
    async def _feed(
        self, st: _State, tags: EmotionTagExtractor, chunker: ThaiSpeechChunker, delta: str
    ) -> bool:
        for text, emotion in tags.feed(delta):
            if emotion is not None:
                st.emotion = emotion
            if text:
                chunks = chunker.feed(text)
                if chunks and not await self._send(st, chunks, final=False):
                    return False
        return True

    async def _finish(
        self, st: _State, tags: EmotionTagExtractor, chunker: ThaiSpeechChunker
    ) -> None:
        if st.speech_failed:
            return
        for text, _ in tags.flush():
            if text:
                chunks = chunker.feed(text)
                if chunks and not await self._send(st, chunks, final=False):
                    return
        if st.stalled or (st.provider_failed and not st.first_event):
            chunker.feed(ELLIPSIS)  # the reply was cut short: end it audibly
        await self._send(st, chunker.flush(), final=True)

    # --- segments ---------------------------------------------------------------------------
    async def _send(self, st: _State, chunks: list[str], *, final: bool) -> bool:
        """Gate, normalise and queue ``chunks``; ``False`` once the reply must stop."""
        if st.speech_failed:
            return False
        prog = st.progress
        for i, chunk in enumerate(chunks):
            last = final and i == len(chunks) - 1
            if st.first_chunk:
                st.first_chunk = False
                self._trace.mark(st.turn_id, "first_chunk")
                self._bus.publish(
                    LatencyMark(character=st.character, turn_id=st.turn_id, stage="first_chunk")
                )
            text = st.carry + chunk
            if not is_speakable(text):
                st.carry = text  # merge forward (emoji, markdown, a bare list marker)
                continue
            st.carry = ""
            res = await self._check(st, text)
            if st.first_filter:
                st.first_filter = False
                self._trace.mark(st.turn_id, "filter_done")
                self._bus.publish(
                    LatencyMark(character=st.character, turn_id=st.turn_id, stage="filter_done")
                )
            if res.verdict in _STOPPING:
                await self._filtered(st, res)
                return False
            clean = text if res.verdict is Verdict.PASS else res.text
            if not clean.strip():
                continue
            seg = Segment(
                utt_id=st.utt_id,
                seq=st.seq,
                text=self._normalize(clean),
                caption=clean,  # keeps the chunker's spacing: heard text = joined captions
                emotion=st.emotion,
                last=last,
            )
            if not await self._queue(st, seg):
                return False
            st.seq += 1
            st.emotion = None
            prog.sent.append(seg)
            prog.emitted.append(clean)
            if self._prev_tail:
                st.tail = (st.tail + clean)[-self._prev_tail :]
            if last:
                prog.ended = True
        if final and not prog.ended:
            marker = Segment(utt_id=st.utt_id, seq=st.seq, text="", caption="", last=True)
            if await self._queue(st, marker):
                st.seq += 1
                prog.ended = True
        return True

    async def _check(self, st: _State, text: str) -> FilterResult:
        try:
            return await guarded(
                self._gate.check_output(text, character=st.character, prev_tail=st.tail),
                self._gate_timeout_s,
                what="output gate",
                clock=self._clock,
            )
        except DeadlineExceeded:
            log.error("reply %s: the output gate timed out; failing closed", st.utt_id)
            return FilterResult(Verdict.BLOCK, "", "gate", rule="timeout", fail_closed=True)

    async def _queue(self, st: _State, seg: Segment) -> bool:
        started = self._clock.now()
        while True:
            try:
                ok = await guarded(
                    self._speech.segment(seg),
                    self._speech_timeout_s,
                    what="speech.segment",
                    clock=self._clock,
                )
            except DeadlineExceeded:
                log.warning("reply %s: speech.segment timed out; ending the reply", st.utt_id)
                st.speech_failed, st.error = True, "speech timeout"
                return False
            if ok:
                self._bus.publish(
                    SegmentQueued(
                        character=st.character,
                        turn_id=st.turn_id,
                        utt_id=seg.utt_id,
                        seq=seg.seq,
                        caption=seg.caption,
                    )
                )
                return True
            if self._clock.now() - started >= self._bp_timeout_s:
                log.warning(
                    "reply %s: speech backpressure for %.0f s", st.utt_id, self._bp_timeout_s
                )
                st.stalled = st.speech_failed = True
                return False
            await self._clock.sleep(0.05)

    async def _filtered(self, st: _State, res: FilterResult) -> None:
        """§4.10: close the stream, drop unstarted segments, say "Filtered."."""
        st.blocked = res
        st.progress.filtered = True
        st.carry = ""
        await self._close_stream(st)
        try:
            await guarded(
                self._speech.stop(st.utt_id, "after_segment", "filtered"),
                self._speech_timeout_s,
                what="speech.stop",
                clock=self._clock,
            )
            st.progress.ended = True
            await guarded(
                self._speech.play_canned("filtered", st.character),
                self._speech_timeout_s,
                what="speech.play_canned",
                clock=self._clock,
            )
        except DeadlineExceeded:
            log.error("reply %s: speech did not take the Filtered stop in time", st.utt_id)
        if self._publish_filtered:
            self._bus.publish(
                Filtered(
                    character=st.character,
                    turn_id=st.turn_id,
                    direction="out",
                    tier=res.tier,
                    category=res.category,
                    rule=res.rule,
                    ref=st.utt_id,
                )
            )

    def _on_done(self, st: _State, done: Done) -> None:
        self._bus.publish(
            LLMFirstToken(
                character=st.character,
                turn_id=st.turn_id,
                provider=done.provider,
                ttft_ms=done.ttft_ms,
                prompt_n=done.prompt_n,
                cache_n=done.cache_n,
            )
        )
        facts: dict[str, Any] = {
            "provider": done.provider,
            "prompt_n": done.prompt_n,
            "cache_n": done.cache_n,
            "tokens_out": done.completion_tokens,
        }
        self._trace.set(st.turn_id, **facts)

    async def _close_stream(self, st: _State) -> None:
        """Close the LLM iterator (cancels a pending ``__anext__`` first)."""
        pending, st.pending = st.pending, None
        if pending is not None:
            if not pending.done():
                pending.cancel()
                await asyncio.wait({pending}, timeout=2.0)
            if pending.done() and not pending.cancelled():
                pending.exception()  # mark retrieved; the reply already ended
        if st.closed:
            return
        st.closed = True
        aclose = getattr(st.it, "aclose", None)
        if aclose is None:
            return
        try:
            await asyncio.wait_for(aclose(), timeout=2.0)
        except (RuntimeError, TimeoutError) as exc:
            log.warning("reply %s: closing the LLM stream failed: %r", st.utt_id, exc)
        except Exception as exc:  # a provider error surfacing on close; the reply is over
            log.debug("reply %s: LLM stream close raised %r", st.utt_id, exc)
