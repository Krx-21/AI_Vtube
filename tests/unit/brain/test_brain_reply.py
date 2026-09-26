"""ReplyPipeline: deltas → tags → chunker → gate → normaliser → segments (§4.6, §4.10)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from brain_testkit import EMOTIONS

from aivtube.brain.reply import ReplyPipeline, ReplyProgress, ReplyResult
from aivtube.contracts.events import Filtered, LatencyMark, LLMFirstToken
from aivtube.contracts.llm import ChatRequest, Done, LLMEvent, ProviderFailed, TextDelta
from aivtube.contracts.types import Segment
from aivtube.infra.trace import TurnTraceRecorder
from aivtube.testing.fakes import (
    THAI_COMBINING,
    FakeClock,
    FakeEventBus,
    FakeLLM,
    FakeReply,
    FakeSafetyGate,
    FakeSpeechOutput,
    tool_call,
)
from aivtube.text import is_speakable, no_word_split

REQ = ChatRequest(messages=({"role": "user", "content": "ทดสอบ"},))


class Rig:
    def __init__(
        self,
        clock: FakeClock,
        bus: FakeEventBus,
        *,
        blocklist: tuple[str, ...] = (),
        speech: Any = None,
        **kw: Any,
    ) -> None:
        self.clock = clock
        self.bus = bus
        self.speech = speech or FakeSpeechOutput(bus, clock)
        self.gate = FakeSafetyGate(blocklist)
        self.trace = TurnTraceRecorder(clock, bus=bus)
        self.trace.begin("t1", "voice", "pailin")
        self.pipe = ReplyPipeline(
            speech=self.speech,
            gate=self.gate,
            bus=bus,
            clock=clock,
            trace=self.trace,
            known_emotions=EMOTIONS,
            word_tokenize=no_word_split,
            **kw,
        )
        self.progress = ReplyProgress()

    async def run(
        self, events: AsyncIterator[LLMEvent], utt: str = "u1"
    ) -> asyncio.Task[ReplyResult]:
        await self.speech.begin(utt, "pailin")
        return asyncio.ensure_future(
            self.pipe.run(
                character="pailin", utt_id=utt, turn_id="t1", events=events, progress=self.progress
            )
        )

    def segments(self) -> list[Segment]:
        return [a[0] for n, a in self.speech.calls if n == "segment"]


class Scripted:
    """An async iterator over ``(event, sleep_before_s)`` that records ``aclose``."""

    def __init__(
        self, clock: FakeClock, items: list[tuple[LLMEvent | BaseException, float]]
    ) -> None:
        self.clock = clock
        self.items = list(items)
        self.closed = False
        self.consumed = 0

    def __aiter__(self) -> Scripted:
        return self

    async def __anext__(self) -> LLMEvent:
        if self.closed or not self.items:
            raise StopAsyncIteration
        item, pause = self.items.pop(0)
        if pause:
            await self.clock.sleep(pause)
        self.consumed += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def done(text: str = "") -> Done:
    return Done("fake", "stop", 120.0, 50, 40, 10, {"role": "assistant", "content": text})


async def test_split_combining_marks_make_well_formed_segments(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    text = "[happy] โอ้โห วันนี้สนุกมากเลยค่ะ ทุกคนที่ดูอยู่ตอนนี้เป็นยังไงกันบ้าง อยากให้เล่นเกมอะไรต่อดีคะ"
    llm = FakeLLM([FakeReply("", text)], clock=fake_clock, ttft_s=0.2, tok_s=20.0)
    rig = Rig(fake_clock, fake_bus)
    task = await rig.run(llm.stream(REQ))
    await fake_clock.run_until(lambda: bool(rig.segments()), within=10)
    first_at = fake_clock.now()
    assert not task.done()  # the first segment went out before the stream ended
    await fake_clock.run_until(task.done, within=30)
    result = task.result()
    assert first_at < fake_clock.now()
    segs = rig.segments()
    spoken = [s for s in segs if s.text]
    assert segs[-1].last and [s.seq for s in segs] == list(range(len(segs)))
    for s in spoken:
        assert s.text[0] not in THAI_COMBINING and is_speakable(s.text)
        assert len(s.caption) <= 160 and s.text == s.text.strip()
    assert "".join(s.caption for s in spoken) == text.replace("[happy] ", "")
    assert result.emitted_text == text.replace("[happy] ", "")
    # the emotion goes only on the first segment after it changes
    assert [s.emotion for s in spoken] == ["happy"] + [None] * (len(spoken) - 1)
    assert not result.filtered and not result.stalled and result.done is not None
    assert [m.stage for m in fake_bus.of_type(LatencyMark)] == ["first_chunk", "filter_done"]
    assert fake_bus.of_type(LLMFirstToken)[0].provider == "fake"
    assert {"llm_first_token", "first_chunk", "filter_done"} <= set(rig.trace.get("t1")["stages"])


async def test_blocked_chunk_takes_the_filtered_path(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    items: list[tuple[LLMEvent | BaseException, float]] = [
        (TextDelta("สวัสดีค่ะทุกคน วันนี้เรามาคุยกันเรื่อง"), 0.2),
        (TextDelta("คำต้องห้าม ที่ไม่ควรพูดนะคะ แล้วก็เรื่องอื่นอีกเยอะแยะเลยค่ะ ทุกคน"), 0.1),
        (TextDelta(" ต่อไปอีกยาวๆ"), 0.1),
        (done(), 0.0),
    ]
    stream = Scripted(fake_clock, items)
    rig = Rig(fake_clock, fake_bus, blocklist=("คำต้องห้าม",))
    task = await rig.run(stream)
    await fake_clock.run_until(task.done, within=10)
    result = task.result()
    assert result.filtered and result.blocked is not None
    assert stream.closed and stream.items  # closed early: the rest was never read
    names = rig.speech.call_names()
    i = names.index("stop")
    assert rig.speech.calls[i][1][1:3] == ("after_segment", "filtered")
    assert names[i + 1] == "play_canned" and rig.speech.calls[i + 1][1] == ("filtered", "pailin")
    assert "คำต้องห้าม" not in result.emitted_text
    assert all("คำต้องห้าม" not in s.caption for s in result.sent)
    [event] = fake_bus.of_type(Filtered)
    assert (event.direction, event.ref, event.turn_id) == ("out", "u1", "t1")
    assert result.emitted_text.startswith("สวัสดีค่ะทุกคน")  # the clean part was spoken


async def test_stall_flush_and_inter_token_stall(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    items: list[tuple[LLMEvent | BaseException, float]] = [
        (TextDelta("สวัสดีค่ะ วันนี้อากาศดีมาก ห้"), 0.2),
        (TextDelta("องฟ้าใส"), 1.0),  # 1 s gap: the 500 ms stall flush fires
        (TextDelta(" แล้วก็"), 0.1),
        (TextDelta(" ไม่มีวันมา"), 10.0),  # a 3 s inter-token stall ends the reply
    ]
    stream = Scripted(fake_clock, items)
    rig = Rig(fake_clock, fake_bus)
    task = await rig.run(stream)
    await fake_clock.run_for(0.2 + 0.6)
    flushed = [s.caption for s in rig.segments()]
    assert flushed == ["สวัสดีค่ะ ", "วันนี้อากาศดีมาก "]  # cut at the last space, never mid-word
    await fake_clock.run_until(task.done, within=10)
    result = task.result()
    assert result.stalled and stream.closed
    segs = rig.segments()
    assert segs[-1].last and result.emitted_text.rstrip().endswith("…")
    assert "ไม่มีวันมา" not in result.emitted_text


async def test_backpressure_pauses_without_dropping_text(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    speech = FakeSpeechOutput(fake_bus, fake_clock, max_queued=1, chars_per_s=40.0)
    text = "ประโยคที่หนึ่งค่ะ ประโยคที่สองค่ะ ประโยคที่สามค่ะ ประโยคที่สี่ค่ะ ประโยคที่ห้าค่ะ " * 3
    llm = FakeLLM([FakeReply("", text)], clock=fake_clock, ttft_s=0.1, tok_s=1000.0)
    rig = Rig(fake_clock, fake_bus, speech=speech)
    task = await rig.run(llm.stream(REQ))
    await fake_clock.run_until(task.done, within=120)
    result = task.result()
    refused = [a[0] for n, a in speech.calls if n == "segment"]
    assert len(refused) > len(result.sent)  # segment() said False and was retried
    assert result.emitted_text == text.strip()
    assert not result.stalled and speech.dropped == []
    await fake_clock.run_until(lambda: speech.idle, within=60)
    assert "".join(speech.heard_texts()).replace(" ", "") == text.replace(" ", "")


async def test_tool_calls_and_provider_failures(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    call = tool_call("echo", {"text": "hi"})
    llm = FakeLLM([FakeReply("", "", tool_calls=[call])], clock=fake_clock, ttft_s=0.1)
    rig = Rig(fake_clock, fake_bus)
    task = await rig.run(llm.stream(REQ))
    await fake_clock.run_until(task.done, within=5)
    result = task.result()
    assert result.tool_calls == (call,) and not result.spoke and result.done is not None
    assert rig.segments()[-1].last and rig.segments()[-1].text == ""  # an empty end marker

    # dies mid-utterance: the reply ends with "…"
    failing = Scripted(
        fake_clock,
        [
            (TextDelta("เรื่องนี้ยาวมากเลยค่ะ ทุกคน แล้วก็"), 0.1),
            (ProviderFailed("died", emitted=True), 0.1),
        ],
    )
    rig2 = Rig(fake_clock, fake_bus)
    task2 = await rig2.run(failing, utt="u2")
    await fake_clock.run_until(task2.done, within=5)
    result2 = task2.result()
    assert result2.provider_failed and result2.emitted_text.endswith("…")
    # fails before anything: nothing is sent except the end marker
    rig3 = Rig(fake_clock, fake_bus)
    task3 = await rig3.run(
        Scripted(fake_clock, [(ProviderFailed("down", emitted=False), 0.1)]), utt="u3"
    )
    await fake_clock.run_until(task3.done, within=5)
    result3 = task3.result()
    assert result3.provider_failed and result3.sent == () and result3.emitted_text == ""


class DeadSpeech(FakeSpeechOutput):
    """``segment()`` never returns (the voice worker is wedged)."""

    async def segment(self, seg: Segment) -> bool:
        await self.clock.sleep(1e9)
        return True


async def test_speech_timeout_ends_the_reply(fake_clock: FakeClock, fake_bus: FakeEventBus) -> None:
    llm = FakeLLM([FakeReply("", "สวัสดีค่ะ ทุกคน " * 10)], clock=fake_clock, ttft_s=0.1)
    rig = Rig(fake_clock, fake_bus, speech=DeadSpeech(fake_bus, fake_clock))
    task = await rig.run(llm.stream(REQ))
    await fake_clock.run_until(task.done, within=10)
    result = task.result()
    assert result.error == "speech timeout" and result.sent == ()
    assert fake_clock.now() - 1000.0 < 5.0  # one 2 s timeout, not one per chunk
    assert llm.closed_early == 1


async def test_a_slow_first_token_is_not_an_inter_token_stall(
    fake_clock: FakeClock, fake_bus: FakeEventBus
) -> None:
    """Before the first event the router owns the first-token deadlines and the fallback
    (§4.12: 4 s for the 30B, 6–8 s for cloud); the 3 s inter-token stall starts after it."""
    items: list[tuple[LLMEvent | BaseException, float]] = [
        (TextDelta("ตอบช้าหน่อยนะคะ วันนี้เน็ตช้า"), 5.0),
        (done(), 0.1),
    ]
    stream = Scripted(fake_clock, items)
    rig = Rig(fake_clock, fake_bus)
    task = await rig.run(stream)
    await fake_clock.run_until(task.done, within=10)
    result = task.result()
    assert not result.stalled and result.emitted_text == "ตอบช้าหน่อยนะคะ วันนี้เน็ตช้า"
    # a safety bound still exists before the first token
    hang = Scripted(fake_clock, [(done(), 100.0)])
    rig2 = Rig(fake_clock, fake_bus, first_event_timeout_s=30.0)
    task2 = await rig2.run(hang, utt="u2")
    await fake_clock.run_until(task2.done, within=40)
    assert task2.result().stalled and hang.closed
