"""Contract suite for ``SpeechOutput`` (§3.5): results arrive as bus events."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aivtube.contracts.events import Event, SegmentDone, SegmentStarted, UtteranceDone
from aivtube.contracts.infra import EventBus
from aivtube.contracts.speech import SpeechOutput, TTSConstraints, VoicePolicy
from aivtube.contracts.types import Segment
from aivtube.testing.contracts._base import AsyncCase, _Cases, await_until, check, maybe_await

__all__ = ["SpeechOutputHarness", "speech_output_suite"]


@dataclass
class SpeechOutputHarness:
    """What a ``speech_output_suite`` factory returns.

    ``advance(dt)`` drives fake time (``FakeClock.run_for``); ``None`` means real time.
    ``aclose`` tears the implementation down after each case.
    """

    output: SpeechOutput
    bus: EventBus
    advance: Callable[[float], Awaitable[None]] | None = None
    aclose: Callable[[], Awaitable[None]] | None = None
    character: str = "pailin"


class _Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        self.sub = bus.subscribe(SegmentStarted, SegmentDone, UtteranceDone, name="contract")
        self.task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        async for ev in self.sub:
            self.events.append(ev)

    def of(self, utt: str) -> list[Event]:
        return [e for e in self.events if getattr(e, "utt_id", None) == utt]

    def done(self, utt: str) -> UtteranceDone | None:
        return next((e for e in self.of(utt) if isinstance(e, UtteranceDone)), None)

    async def close(self) -> None:
        self.sub.close()
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


def speech_output_suite(
    factory: Callable[[], SpeechOutputHarness | Awaitable[SpeechOutputHarness]],
    *,
    within: float = 20.0,
) -> list[AsyncCase]:
    cases = _Cases("speech_output")

    def seg(utt: str, seq: int, text: str, last: bool = False) -> Segment:
        return Segment(utt_id=utt, seq=seq, text=text, caption=text, last=last)

    async def setup() -> tuple[SpeechOutputHarness, _Recorder]:
        h = await maybe_await(factory())
        return h, _Recorder(h.bus)

    async def teardown(h: SpeechOutputHarness, rec: _Recorder) -> None:
        await rec.close()
        if h.aclose is not None:
            await h.aclose()

    async def until(h: SpeechOutputHarness, pred: Callable[[], bool], what: str) -> None:
        await await_until(pred, within=within, advance=h.advance, step=0.05, what=what)

    @cases
    async def ready_and_constraints() -> None:
        h, rec = await setup()
        try:
            check(isinstance(h.output.ready(), bool), "ready() must return bool")
            c = h.output.constraints(h.character)
            check(isinstance(c, TTSConstraints), "constraints() must return TTSConstraints")
            check(0 < c.first_min_chars <= c.max_chars, "first_min_chars")
            check(0 < c.min_chars <= c.max_chars, "min_chars")
        finally:
            await teardown(h, rec)

    @cases
    async def utterance_plays_segments_in_order() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-order", h.character)
            check(await h.output.segment(seg("u-order", 0, "สวัสดีค่ะ")), "segment 0 refused")
            check(await h.output.segment(seg("u-order", 1, "ยินดีที่ได้รู้จัก", True)), "seg 1 refused")
            await until(h, lambda: rec.done("u-order") is not None, "UtteranceDone")
            kinds = [(type(e).__name__, getattr(e, "seq", None)) for e in rec.of("u-order")]
            check(
                kinds
                == [
                    ("SegmentStarted", 0),
                    ("SegmentDone", 0),
                    ("SegmentStarted", 1),
                    ("SegmentDone", 1),
                    ("UtteranceDone", None),
                ],
                f"event order {kinds}",
            )
            done = rec.done("u-order")
            assert done is not None
            check(not done.cancelled and done.reason is None, "a complete utterance was cancelled")
            check("สวัสดีค่ะ" in done.heard_text and "ยินดี" in done.heard_text, done.heard_text)
            for e in rec.of("u-order"):
                check(e.character == h.character, f"{type(e).__name__} lacks the character")
                if isinstance(e, SegmentDone):
                    check(e.heard, "a fully played segment must be heard")
                if isinstance(e, SegmentStarted):
                    check(isinstance(e.t_audible, float), "t_audible")
        finally:
            await teardown(h, rec)

    @cases
    async def stop_now_cuts_and_cancels() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-cut", h.character)
            long_text = "นี่คือประโยคที่ยาวมากเพื่อทดสอบการหยุดพูดกลางคัน " * 3
            await h.output.segment(seg("u-cut", 0, long_text))
            await until(
                h, lambda: any(isinstance(e, SegmentStarted) for e in rec.of("u-cut")), "start"
            )
            await h.output.stop("u-cut", "now", "contract-skip")
            await until(h, lambda: rec.done("u-cut") is not None, "UtteranceDone after stop")
            done = rec.done("u-cut")
            assert done is not None
            check(done.cancelled and done.reason == "contract-skip", f"{done!r}")
            seg_done = [e for e in rec.of("u-cut") if isinstance(e, SegmentDone)]
            check(seg_done and not seg_done[0].heard, "a cut segment must not count as heard")
        finally:
            await teardown(h, rec)

    @cases
    async def stop_after_segment_finishes_current() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-after", h.character)
            await h.output.segment(seg("u-after", 0, "ประโยคแรกของไพลิน"))
            await h.output.segment(seg("u-after", 1, "ประโยคที่สองไม่ควรได้ยิน", True))
            await until(
                h, lambda: any(isinstance(e, SegmentStarted) for e in rec.of("u-after")), "start"
            )
            await h.output.stop("u-after", "after_segment", "filtered")
            await until(h, lambda: rec.done("u-after") is not None, "UtteranceDone")
            started = [e.seq for e in rec.of("u-after") if isinstance(e, SegmentStarted)]
            check(started == [0], f"segments started after stop(after_segment): {started}")
            first = next(e for e in rec.of("u-after") if isinstance(e, SegmentDone))
            check(first.heard, "the current segment must finish")
            done = rec.done("u-after")
            assert done is not None
            check(done.cancelled, "stop(after_segment) must report cancelled=True")
        finally:
            await teardown(h, rec)

    @cases
    async def stop_none_stops_what_is_playing() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-none", h.character)
            await h.output.segment(seg("u-none", 0, "กำลังพูดอยู่นะคะ ยาวหน่อย " * 3))
            await until(
                h, lambda: any(isinstance(e, SegmentStarted) for e in rec.of("u-none")), "start"
            )
            await h.output.stop(None, "now", "freeze")
            await until(h, lambda: rec.done("u-none") is not None, "UtteranceDone")
            done = rec.done("u-none")
            check(done is not None and done.cancelled, "stop(None) did not cancel")
        finally:
            await teardown(h, rec)

    @cases
    async def closed_gate_holds_playback() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-gate", h.character, gate_open=False)
            await h.output.segment(seg("u-gate", 0, "รอเปิดประตูก่อน", True))
            if h.advance is not None:
                await h.advance(1.0)
            else:
                await asyncio.sleep(0.3)
            check(not rec.of("u-gate"), "played before open_gate()")
            await h.output.open_gate("u-gate")
            await until(h, lambda: rec.done("u-gate") is not None, "UtteranceDone")
        finally:
            await teardown(h, rec)

    @cases
    async def backpressure_refuses_without_raising() -> None:
        h, rec = await setup()
        try:
            await h.output.begin("u-bp", h.character)
            results = [await h.output.segment(seg("u-bp", i, f"ท่อนที่ {i}")) for i in range(24)]
            check(all(isinstance(r, bool) for r in results), "segment() must return bool")
            check(False in results, "no backpressure after 24 queued segments")
            await h.output.stop("u-bp", "now", "cleanup")
        finally:
            await teardown(h, rec)

    @cases
    async def controls_do_not_raise() -> None:
        h, rec = await setup()
        try:
            out = h.output
            await out.duck(0.25, ramp_ms=30)
            await out.duck(1.0)
            await out.mute(True)
            await out.mute(False)
            await out.set_policy(VoicePolicy(mic_mode="ptt", ptt_active=True))
            await out.set_voice_rate(h.character, 10)
            await out.play_canned("filtered", h.character)
            await out.stop(None, "now", "idle stop")
        finally:
            await teardown(h, rec)

    return cases.items
