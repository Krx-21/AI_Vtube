"""Text console (§9): parsing, dispatch, the stdin reader thread, event echo and /op freeze."""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from typing import Any

import pytest

from aivtube.console import TextConsole, parse_line, voice_submitter
from aivtube.console.textmode import (
    Bits,
    Chat,
    Empty,
    Help,
    Invalid,
    Mem,
    Op,
    Quit,
    State,
    Sub,
    Voice,
)
from aivtube.contracts.control import OpCommand, OpKind, OpResult
from aivtube.contracts.events import (
    Alert,
    Filtered,
    SegmentStarted,
    StateChanged,
    UserTranscript,
    UtteranceDone,
)
from aivtube.contracts.infra import EventBus
from aivtube.contracts.memory import MemoryItem
from aivtube.contracts.types import ChatMessage, MsgKind, Platform, Segment
from aivtube.testing.fakes import (
    FakeClock,
    FakeControlSurface,
    FakeEventBus,
    FakeLLM,
    FakeLLMRouter,
    FakeMemoryStore,
    FakeSafetyGate,
    FakeSpeechOutput,
    FakeTaskSupervisor,
    FakeToolRegistry,
)

# --- parsing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", Empty()),
        ("   ", Empty()),
        ("สวัสดีไพลิน วันนี้เป็นไงบ้าง", Voice("สวัสดีไพลิน วันนี้เป็นไงบ้าง")),
        ("﻿hello", Voice("hello")),
        ("/chat tom: ไพลินกินข้าวยัง", Chat("tom", "ไพลินกินข้าวยัง")),
        ("/chat ต้น กล้า : 5555", Chat("ต้น กล้า", "5555")),
        ("/bits amy 500", Bits("amy", 500, "")),
        ("/bits amy 1,000 สู้ๆ นะ", Bits("amy", 1000, "สู้ๆ นะ")),
        ("/sub bob 7", Sub("bob", 7, "")),
        ("/SUB bob 3 ครบสามเดือนแล้ว", Sub("bob", 3, "ครบสามเดือนแล้ว")),
        ("/op freeze", Op(OpKind.FREEZE, {})),
        ("/op RESUME", Op(OpKind.RESUME, {})),
        ("/op skip", Op(OpKind.SKIP, {})),
        ("/op live", Op(OpKind.GO_LIVE, {})),
        ("/op say สวัสดีค่ะทุกคน", Op(OpKind.SAY, {"text": "สวัสดีค่ะทุกคน"})),
        ("/op direct เปลี่ยนเรื่อง", Op(OpKind.DIRECT, {"text": "เปลี่ยนเรื่อง"})),
        ("/op mic PTT", Op(OpKind.MIC_MODE, {"mode": "ptt"})),
        ("/op ptt on", Op(OpKind.PTT, {"active": True})),
        ("/op chat ปิด", Op(OpKind.CHAT_INTAKE, {"on": False})),
        ("/op strict on", Op(OpKind.STRICT, {"on": True})),
        ("/op llm local-4b", Op(OpKind.LLM_USE, {"name": "local-4b"})),
        ("/op rollback", Op(OpKind.LLM_ROLLBACK, {})),
        ("/op tools dry_run", Op(OpKind.TOOLS_MODE, {"mode": "dry_run"})),
        ("/op reload", Op(OpKind.FILTER_RELOAD, {})),
        ("/op restart voice", Op(OpKind.RESTART, {"component": "voice"})),
        ("/op approve ap3", Op(OpKind.APPROVE, {"request": "ap3", "approved": True})),
        ("/op deny ap3", Op(OpKind.APPROVE, {"request": "ap3", "approved": False})),
        ("/mem", Mem(None)),
        ("/mem quarantined", Mem("quarantined")),
        ("/state", State()),
        ("/help", Help()),
        ("/quit", Quit()),
        ("/exit", Quit()),
    ],
)
def test_parse_line(line: str, expected: object) -> None:
    assert parse_line(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "/chat no colon here",
        "/chat : missing name",
        "/chat tom:   ",
        "/bits amy",
        "/bits amy lots",
        "/bits amy -5",
        "/sub bob zero",
        "/op",
        "/op say",
        "/op direct   ",
        "/op mic loud",
        "/op ptt maybe",
        "/op tools yolo",
        "/op restart",
        "/op dance",
        "/op approve",
        "/op deny ap1 ap2",
        "/mem everything",
        "/teleport",
    ],
)
def test_parse_line_rejects(line: str) -> None:
    result = parse_line(line)
    assert isinstance(result, Invalid) and result.message


# --- dispatch -------------------------------------------------------------------------------


class Rig:
    def __init__(self, clock: Any = None, *, control: Any = None, **kw: Any) -> None:
        self.clock = clock or FakeClock()
        self.control = control or FakeControlSurface()
        self.voice: list[str] = []
        self.chat: list[ChatMessage] = []
        self.out = io.StringIO()
        self.console = TextConsole(
            submit_voice=self.voice.append,
            ingest_chat=self.chat.append,
            control=self.control,
            out=self.out,
            clock=self.clock,
            **kw,
        )

    @property
    def text(self) -> str:
        return self.out.getvalue()


async def test_plain_lines_are_the_streamer_speaking() -> None:
    rig = Rig()
    assert await rig.console.handle("ไพลิน ช่วยอ่านแชทหน่อย") is True
    assert rig.voice == ["ไพลิน ช่วยอ่านแชทหน่อย"]
    assert await rig.console.handle("   ") is True and rig.voice == ["ไพลิน ช่วยอ่านแชทหน่อย"]


async def test_chat_bits_and_sub_become_messages() -> None:
    rig = Rig()
    await rig.console.handle("/chat tom: ไพลินกินข้าวยัง?")
    await rig.console.handle("/bits amy 250 เป็นกำลังใจให้")
    await rig.console.handle("/sub bob 6")
    chat, bits, sub = rig.chat
    assert (chat.platform, chat.kind, chat.user.name, chat.text) == (
        Platform.CONSOLE,
        MsgKind.TEXT,
        "tom",
        "ไพลินกินข้าวยัง?",
    )
    assert (bits.kind, bits.amount, bits.currency, bits.value_usd, bits.text) == (
        MsgKind.DONATION,
        250.0,
        "bits",
        2.5,
        "เป็นกำลังใจให้",
    )
    assert sub.kind is MsgKind.SUB and sub.user.sub_months == 6 and sub.user.is_sub
    assert len({m.id for m in rig.chat}) == 3
    assert "[bits] amy" in rig.text and "[sub] bob" in rig.text


async def test_op_commands_go_through_the_control_surface() -> None:
    rig = Rig()
    await rig.console.handle("/op freeze")
    await rig.console.handle("/op say สวัสดีค่ะ")
    freeze, say = rig.control.commands
    assert (freeze.kind, freeze.character, freeze.operator) == (OpKind.FREEZE, None, "console")
    assert (say.kind, say.args, say.character) == (OpKind.SAY, {"text": "สวัสดีค่ะ"}, "pailin")
    assert "[op] freeze: สำเร็จ" in rig.text
    failing = Rig(control=FakeControlSurface({OpKind.GO_LIVE: OpResult(False, "LLM ยังไม่พร้อม")}))
    await failing.console.handle("/op live")
    assert "[op] go_live: ไม่สำเร็จ — LLM ยังไม่พร้อม" in failing.text


async def test_op_timeout_is_reported() -> None:
    class Hanging(FakeControlSurface):
        async def execute(self, cmd: OpCommand) -> OpResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    rig = Rig(control=Hanging(), cmd_timeout_s=2.0)
    task = asyncio.ensure_future(rig.console.handle("/op resume"))
    await rig.clock.run_for(2.5)
    assert await task is True
    assert "หมดเวลา" in rig.text


async def test_mem_state_help_and_quit() -> None:
    clock = FakeClock()
    memory = FakeMemoryStore("pailin", clock)
    await memory.remember(MemoryItem(id=None, kind="core", text="ไพลินชอบแมว", source="operator"))
    await memory.remember(
        MemoryItem(id=None, kind="fact", text="ต้นกล้ามาจากเชียงใหม่", status="quarantined")
    )
    rig = Rig(clock, memory=memory)
    await rig.console.handle("/mem")
    assert "#1 s1 [core/active] ไพลินชอบแมว" in rig.text
    assert "[fact/quarantined] ต้นกล้ามาจากเชียงใหม่" in rig.text
    rig.out.truncate(0)
    rig.out.seek(0)
    await rig.console.handle("/mem active")
    assert "ไพลินชอบแมว" in rig.text and "เชียงใหม่" not in rig.text
    no_memory = Rig()
    await no_memory.console.handle("/mem")
    assert "ไม่มีหน่วยความจำ" in no_memory.text
    await rig.console.handle("/op freeze")
    await rig.console.handle("/state")
    assert "[state] paused: True" in rig.text
    await rig.console.handle("/help")
    assert "/chat ชื่อ: ข้อความ" in rig.text
    await rig.console.handle("/nonsense")
    assert "ไม่รู้จักคำสั่ง" in rig.text
    assert await rig.console.handle("/quit") is False


async def test_state_lists_pending_tool_approvals() -> None:
    class Control(FakeControlSurface):
        def snapshot(self) -> dict[str, Any]:
            pending = {"id": "ap1", "tool": "timeout_user", "args": {"user": "troll"}}
            return {"frozen": False, "approvals": [pending]}

    rig = Rig(control=Control())
    await rig.console.handle("/state")
    assert "[รออนุมัติ] ap1: timeout_user" in rig.text and "/op approve ap1" in rig.text
    await rig.console.handle("/op deny ap1")
    [cmd] = rig.control.commands
    assert (cmd.kind, dict(cmd.args), cmd.character) == (
        OpKind.APPROVE,
        {"request": "ap1", "approved": False},
        None,
    )


async def test_handle_never_raises() -> None:
    def broken(text: str) -> None:
        raise RuntimeError("intake down")

    rig = Rig()
    console = TextConsole(
        submit_voice=broken,
        ingest_chat=lambda m: None,
        control=rig.control,
        out=rig.out,
        clock=rig.clock,
    )
    assert await console.handle("hello") is True
    assert "[ผิดพลาด] RuntimeError: intake down" in rig.text


def test_print_survives_consoles_without_thai() -> None:
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="ascii")
    console = TextConsole(
        submit_voice=lambda t: None,
        ingest_chat=lambda m: None,
        control=FakeControlSurface(),
        out=out,
        clock=FakeClock(),
    )
    console.print("ไพลิน: hi")
    out.flush()
    assert raw.getvalue() == b"?????: hi\n"
    out.close()
    console.print("after close")  # no exception


# --- run(): stdin reader thread ---------------------------------------------------------------


async def test_run_reads_stdin_in_a_thread_until_quit(real_clock: Any) -> None:
    stdin = io.StringIO("สวัสดีไพลิน\r\n/chat tom: hi\n/quit\nหลัง quit\n")
    rig = Rig(real_clock, stdin=stdin)
    await asyncio.wait_for(rig.console.run(), 10)
    assert rig.voice == ["สวัสดีไพลิน"]
    assert [m.text for m in rig.chat] == ["hi"]
    assert "ลาก่อน" in rig.text
    # A restarted run() continues with the next line instead of losing it, then sees EOF.
    await asyncio.wait_for(rig.console.run(), 10)
    assert rig.voice == ["สวัสดีไพลิน", "หลัง quit"]
    await asyncio.wait_for(rig.console.run(), 10)  # input already ended: returns at once


async def test_a_line_in_flight_during_a_restart_reaches_the_new_run() -> None:
    """Regression: a delivery scheduled for the old run must not strand the line."""
    rig = Rig()
    loop = asyncio.get_running_loop()
    old: asyncio.Queue[str | None] = asyncio.Queue()
    new: asyncio.Queue[str | None] = asyncio.Queue()
    rig.console._reader = object()  # type: ignore[assignment]  # no real stdin thread here
    rig.console._attach(loop, old)
    rig.console._post("บรรทัดที่หนึ่ง")  # delivery is now scheduled on the loop
    rig.console._detach(old)  # the old run ends before the callback runs
    rig.console._post("บรรทัดที่สอง")  # no run attached: goes to the backlog
    rig.console._attach(loop, new)
    rig.console._post(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    got = [new.get_nowait() for _ in range(new.qsize())]
    assert got == ["บรรทัดที่หนึ่ง", "บรรทัดที่สอง", None]  # reading order is kept
    assert old.empty()


async def test_run_ends_at_end_of_input(real_clock: Any) -> None:
    rig = Rig(real_clock, stdin=io.StringIO("บรรทัดเดียว"))
    await asyncio.wait_for(rig.console.run(), 10)
    assert rig.voice == ["บรรทัดเดียว"]


async def test_run_can_be_cancelled(real_clock: Any) -> None:
    class Blocking(io.StringIO):
        def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
            import time

            time.sleep(0.2)
            return ""

    rig = Rig(real_clock, stdin=Blocking())
    task = asyncio.ensure_future(rig.console.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class _SlowStdin(io.StringIO):
    """Returns ``lines`` one by one after ``delay`` seconds each (a person typing)."""

    def __init__(self, lines: list[str], delay: float) -> None:
        super().__init__()
        self._lines = list(lines)
        self._delay = delay

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        import time

        if not self._lines:
            return ""
        time.sleep(self._delay)
        return self._lines.pop(0) + "\n"


# --- event echo -------------------------------------------------------------------------------


def _segment(caption: str, character: str) -> SegmentStarted:
    return SegmentStarted(
        utt_id="u1",
        seq=0,
        t_audible=1.0,
        duration_s=1.0,
        backend="console",
        silent=False,
        caption=caption,
        emotion=None,
        character=character,
    )


async def test_captions_and_alerts_are_echoed(bus: EventBus, real_clock: Any) -> None:
    typing = _SlowStdin(["/state", "/quit"], delay=0.3)
    rig = Rig(real_clock, bus=bus, stdin=typing, display_name="ไพลิน")

    async def publish_then_quit() -> None:
        await asyncio.sleep(0.05)
        bus.publish(_segment("สวัสดีค่ะทุกคน", "pailin"))
        bus.publish(_segment("ฉันคือแฝด", "twin"))
        bus.publish(Filtered(direction="out", tier="tier0", category="slur", rule=None, ref=None))
        bus.publish(
            UtteranceDone(
                utt_id="u1",
                heard_text="",
                cancelled=True,
                reason="operator_freeze",
                filtered=False,
                character="pailin",
            )
        )
        bus.publish(StateChanged(old="speaking", new="paused", character="pailin"))
        bus.publish(Alert(level="warn", message="TTS ช้า"))
        await asyncio.sleep(0.1)

    await asyncio.gather(rig.console.run(), publish_then_quit())
    text = rig.text
    assert "ไพลิน: สวัสดีค่ะทุกคน" in text
    assert "ฉันคือแฝด" not in text
    assert "ไพลิน: Filtered." in text
    assert "(ถูกตัด: operator_freeze)" in text
    assert "[สถานะ] speaking → paused" in text
    assert "[แจ้งเตือน/warn] TTS ช้า" in text


async def test_echo_captions_can_be_turned_off(bus: EventBus, real_clock: Any) -> None:
    rig = Rig(real_clock, bus=bus, echo_captions=False, stdin=_SlowStdin(["/quit"], delay=0.2))

    async def publish() -> None:
        await asyncio.sleep(0.05)
        bus.publish(_segment("ไม่ควรพิมพ์", "pailin"))

    await asyncio.gather(rig.console.run(), publish())
    assert "ไม่ควรพิมพ์" not in rig.text


# --- acceptance: /op freeze stops output --------------------------------------------------------


class _Brain:
    """Minimal ``BrainControl``: accepts every command."""

    def __init__(self) -> None:
        self.commands: list[OpCommand] = []

    async def control(self, cmd: OpCommand) -> OpResult:
        self.commands.append(cmd)
        return OpResult(True)

    def snapshot(self) -> dict[str, str]:
        return {"state": "paused" if self.commands else "speaking"}


async def test_op_freeze_stops_output() -> None:
    from aivtube.panel.control import CoreControl

    clock = FakeClock()
    bus = FakeEventBus(clock)
    speech = FakeSpeechOutput(bus, clock)
    control = CoreControl(
        {"pailin": _Brain()},
        router=FakeLLMRouter([FakeLLM([], name="local-4b", clock=clock)], clock=clock),
        registry=FakeToolRegistry(),
        speech=speech,
        memory_by_char={},
        safety=FakeSafetyGate(),
        ops=None,
        bus=bus,
        clock=clock,
        restart=lambda name: asyncio.sleep(0),
        tasks=FakeTaskSupervisor(clock),
    )
    rig = Rig(clock, control=control)
    long_text = "ไพลินกำลังเล่าเรื่องยาวมากๆ ที่ไม่ควรพูดต่อ " * 4
    await speech.begin("u1", "pailin")
    await speech.segment(Segment("u1", 0, long_text, long_text, last=True))
    await clock.run_for(1.0)
    assert speech.audible, "the segment should be playing"
    await rig.console.handle("/op freeze")
    await clock.run_for(0.2)
    done = [e for e in bus.history if isinstance(e, UtteranceDone)]
    assert done and done[-1].cancelled and done[-1].reason == "operator_freeze"
    heard = len(speech.audible)
    await clock.run_for(10.0)
    assert len(speech.audible) == heard  # nothing more is spoken
    assert "[op] freeze: สำเร็จ" in rig.text
    await speech.aclose()


async def test_typed_lines_become_console_transcripts() -> None:
    clock = FakeClock()
    bus = FakeEventBus(clock)
    console = TextConsole(
        submit_voice=voice_submitter(bus, character="pailin"),
        ingest_chat=lambda m: None,
        control=FakeControlSurface(),
        out=io.StringIO(),
        clock=clock,
    )
    await console.handle("ไพลิน วันนี้เล่นเกมอะไรดี")
    await console.handle("/chat tom: hi")  # chat is not a transcript
    [transcript] = bus.of_type(UserTranscript)
    assert (transcript.text, transcript.engine, transcript.character) == (
        "ไพลิน วันนี้เล่นเกมอะไรดี",
        "console",
        "pailin",
    )
    assert (transcript.audio_s, transcript.latency_ms, transcript.parts) == (0.0, 0.0, 1)


def test_console_import_is_light() -> None:
    code = "import sys, aivtube.console; print('aiohttp' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "False"
