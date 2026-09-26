"""Text console mode (ARCHITECTURE.md §9): the keyboard stands in for voice, chat and operator.

Input (one line at a time)::

    <text>                  the streamer says <text> (a VOICE stimulus via ``submit_voice``)
    /chat name: text        a chat message
    /bits name n [text]     a Bits donation (n bits)
    /sub name months [text] a (re)subscription
    /op freeze|resume|skip|mute|unmute|live|say <t>|direct <t>|mic open|ptt|deafened
        |ptt on|off|chat on|off|llm <name>|rollback|tools live|dry_run|off|reload
        |strict on|off|restart <component>|approve <id>|deny <id>
    /mem [active|quarantined|deleted]   list memories
    /state                  the control snapshot
    /help, /quit

stdin is read by a daemon thread (a blocking ``readline`` works on the Windows console and on
pipes) and handed to the event loop with ``call_soon_threadsafe``; everything else, including
all printing, happens on the loop thread. With a ``bus``, spoken captions (``SegmentStarted``),
"Filtered.", alerts and pauses are printed as they happen; turn that off with
``echo_captions=False`` when a ``ConsoleSpeechOutput`` already prints the speech.

:func:`voice_submitter` is the usual ``submit_voice``: a typed line becomes a
``UserTranscript(engine="console")`` on the bus, the same event the voice worker publishes
after STT, so the brain's voice intake handles both alike.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TextIO

from aivtube.contracts.control import ControlSurface, OpCommand, OpKind
from aivtube.contracts.events import (
    Alert,
    Event,
    Filtered,
    SegmentStarted,
    StateChanged,
    UserTranscript,
    UtteranceDone,
)
from aivtube.contracts.infra import Clock, EventBus, Overflow, Subscription
from aivtube.contracts.memory import MemoryStore
from aivtube.contracts.types import ChatMessage, MsgKind
from aivtube.infra.clock import DeadlineExceeded, deadline
from aivtube.panel.commands import new_command_id, validate_args
from aivtube.panel.ingest import chat_message

__all__ = [
    "HELP_TEXT",
    "Bits",
    "Chat",
    "ConsoleCommand",
    "Empty",
    "Help",
    "Invalid",
    "Mem",
    "Op",
    "Quit",
    "State",
    "Sub",
    "TextConsole",
    "Voice",
    "parse_line",
    "voice_submitter",
]

log = logging.getLogger("aivtube.console")

HELP_TEXT: Final = """\
คำสั่งโหมดข้อความ (text mode):
  <ข้อความ>                 สตรีมเมอร์พูดประโยคนี้
  /chat ชื่อ: ข้อความ        ส่งแชท
  /bits ชื่อ จำนวน [ข้อความ]  โดเนท Bits
  /sub ชื่อ เดือน [ข้อความ]   สมัครสมาชิก
  /op freeze | resume | skip | mute | unmute | live
  /op say <ข้อความ> | direct <คำสั่ง>
  /op mic open|ptt|deafened | ptt on|off | chat on|off | strict on|off
  /op llm <ชื่อ> | rollback | tools live|dry_run|off | reload | restart <ส่วน>
  /op approve <id> | deny <id>   ตอบเครื่องมือที่รออนุมัติ (ดู id ใน /state)
  /mem [active|quarantined|deleted]   ดูความจำ
  /state                     ดูสถานะ
  /help                      คำสั่งทั้งหมด
  /quit                      ออก"""


# --- parsed commands --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Voice:
    text: str


@dataclass(frozen=True, slots=True)
class Chat:
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class Bits:
    name: str
    amount: int
    text: str = ""


@dataclass(frozen=True, slots=True)
class Sub:
    name: str
    months: int
    text: str = ""


@dataclass(frozen=True, slots=True)
class Op:
    kind: OpKind
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Mem:
    status: str | None = None


@dataclass(frozen=True, slots=True)
class State:
    pass


@dataclass(frozen=True, slots=True)
class Help:
    pass


@dataclass(frozen=True, slots=True)
class Quit:
    pass


@dataclass(frozen=True, slots=True)
class Empty:
    pass


@dataclass(frozen=True, slots=True)
class Invalid:
    message: str


ConsoleCommand = Voice | Chat | Bits | Sub | Op | Mem | State | Help | Quit | Empty | Invalid

_ON_OFF: Final[Mapping[str, bool]] = {
    "on": True,
    "off": False,
    "เปิด": True,
    "ปิด": False,
    "1": True,
    "0": False,
}
_SIMPLE_OPS: Final[Mapping[str, OpKind]] = {
    "freeze": OpKind.FREEZE,
    "resume": OpKind.RESUME,
    "skip": OpKind.SKIP,
    "mute": OpKind.MUTE,
    "unmute": OpKind.UNMUTE,
    "live": OpKind.GO_LIVE,
    "go_live": OpKind.GO_LIVE,
    "golive": OpKind.GO_LIVE,
    "rollback": OpKind.LLM_ROLLBACK,
    "reload": OpKind.FILTER_RELOAD,
    "end": OpKind.END_STREAM,
}
_MEM_STATUSES: Final = ("active", "quarantined", "deleted")
#: Commands that belong to one character; the rest apply to every character (``None``).
_CHARACTER_KINDS: Final = frozenset(
    {OpKind.SAY, OpKind.DIRECT, OpKind.TTS_IDENTITY, OpKind.FAKE_CHAT, OpKind.INJECT_EVENT}
)


def _positive_int(raw: str, what: str) -> int:
    try:
        value = int(raw.replace(",", ""))
    except ValueError:
        raise ValueError(f"{what} ต้องเป็นตัวเลข") from None
    if value <= 0:
        raise ValueError(f"{what} ต้องมากกว่า 0")
    return value


def _parse_op(rest: str) -> ConsoleCommand:
    word, _, tail = rest.strip().partition(" ")
    word, tail = word.casefold(), tail.strip()
    if not word:
        return Invalid("ใช้: /op freeze|resume|skip|say <ข้อความ>|direct <คำสั่ง>|…")
    if word in _SIMPLE_OPS:
        return Op(_SIMPLE_OPS[word], {})
    kind: OpKind
    args: dict[str, Any]
    if word in ("say", "direct"):
        kind, args = (OpKind.SAY if word == "say" else OpKind.DIRECT), {"text": tail}
    elif word == "mic":
        kind, args = OpKind.MIC_MODE, {"mode": tail.casefold()}
    elif word in ("ptt", "chat", "strict"):
        if tail.casefold() not in _ON_OFF:
            return Invalid(f"ใช้: /op {word} on|off")
        on = _ON_OFF[tail.casefold()]
        if word == "ptt":
            kind, args = OpKind.PTT, {"active": on}
        elif word == "chat":
            kind, args = OpKind.CHAT_INTAKE, {"on": on}
        else:
            kind, args = OpKind.STRICT, {"on": on}
    elif word == "llm":
        kind, args = OpKind.LLM_USE, {"name": tail}
    elif word == "tools":
        kind, args = OpKind.TOOLS_MODE, {"mode": tail.casefold()}
    elif word == "restart":
        kind, args = OpKind.RESTART, {"component": tail}
    elif word in ("tts", "voice"):
        kind, args = OpKind.TTS_IDENTITY, {"identity": tail}
    elif word in ("approve", "deny"):
        if not tail or " " in tail:
            return Invalid(f"ใช้: /op {word} <id> (ดู id ใน /state)")
        kind, args = OpKind.APPROVE, {"request": tail, "approved": word == "approve"}
    else:
        return Invalid(f"ไม่รู้จักคำสั่ง /op {word} (พิมพ์ /help)")
    try:
        return Op(kind, validate_args(kind, args))
    except ValueError as exc:
        return Invalid(str(exc))


def parse_line(line: str) -> ConsoleCommand:
    """Parse one console line; never raises."""
    text = line.strip().lstrip("﻿")
    if not text:
        return Empty()
    if not text.startswith("/"):
        return Voice(text)
    head, _, rest = text[1:].partition(" ")
    head, rest = head.casefold(), rest.strip()
    try:
        if head == "chat":
            name, sep, message = rest.partition(":")
            if not sep or not name.strip() or not message.strip():
                return Invalid("ใช้: /chat ชื่อ: ข้อความ")
            return Chat(name.strip(), message.strip())
        if head == "bits":
            parts = rest.split(maxsplit=2)
            if len(parts) < 2:
                return Invalid("ใช้: /bits ชื่อ จำนวน [ข้อความ]")
            return Bits(parts[0], _positive_int(parts[1], "จำนวน bits"), " ".join(parts[2:]))
        if head == "sub":
            parts = rest.split(maxsplit=2)
            if len(parts) < 2:
                return Invalid("ใช้: /sub ชื่อ เดือน [ข้อความ]")
            return Sub(parts[0], _positive_int(parts[1], "จำนวนเดือน"), " ".join(parts[2:]))
    except ValueError as exc:
        return Invalid(str(exc))
    if head == "op":
        return _parse_op(rest)
    if head in ("mem", "memory"):
        status = rest.casefold() or None
        if status is not None and status not in _MEM_STATUSES:
            return Invalid("ใช้: /mem [active|quarantined|deleted]")
        return Mem(status)
    if head == "state":
        return State()
    if head in ("help", "h", "?"):
        return Help()
    if head in ("quit", "exit", "q"):
        return Quit()
    return Invalid(f"ไม่รู้จักคำสั่ง /{head} (พิมพ์ /help)")


# --- the console --------------------------------------------------------------------------------


def voice_submitter(bus: EventBus, *, character: str | None = None) -> Callable[[str], None]:
    """A ``submit_voice`` that publishes each typed line as ``UserTranscript``
    (``engine="console"``, no audio) on the loop thread."""

    def submit(text: str) -> None:
        bus.publish(
            UserTranscript(
                text=text, engine="console", latency_ms=0.0, audio_s=0.0, character=character
            )
        )

    return submit


class TextConsole:
    """Keyboard front end for ``aivtube run --text`` (see the module docstring)."""

    def __init__(
        self,
        *,
        submit_voice: Callable[[str], None],
        ingest_chat: Callable[[ChatMessage], None],
        control: ControlSurface,
        out: TextIO,
        clock: Clock,
        bus: EventBus | None = None,
        memory: MemoryStore | None = None,
        character: str = "pailin",
        display_name: str | None = None,
        stdin: TextIO | None = None,
        echo_captions: bool = True,
        cmd_timeout_s: float = 15.0,
    ) -> None:
        self._submit_voice = submit_voice
        self._ingest_chat = ingest_chat
        self._control = control
        self._out = out
        self._clock = clock
        self._bus = bus
        self._memory = memory
        self._character = character
        self._name = display_name or character
        self._stdin = stdin
        self._echo = echo_captions
        self._cmd_timeout = cmd_timeout_s
        # One reader thread per console, re-attached to each run() (a restart loses no line).
        self._io_lock = threading.Lock()
        self._target: tuple[asyncio.AbstractEventLoop, asyncio.Queue[str | None]] | None = None
        self._backlog: collections.deque[str | None] = collections.deque()
        self._reader: threading.Thread | None = None
        self._eof_seen = False  # a run consumed the end-of-input marker

    # --- output ---------------------------------------------------------------------------
    def print(self, text: str) -> None:
        """Write one line; survives consoles that cannot encode Thai and closed streams."""
        try:
            self._out.write(text + "\n")
        except UnicodeEncodeError:
            encoding = getattr(self._out, "encoding", None) or "ascii"
            safe = text.encode(encoding, "replace").decode(encoding, "replace")
            with contextlib.suppress(OSError, ValueError, UnicodeError):
                self._out.write(safe + "\n")
        except (OSError, ValueError):
            return
        with contextlib.suppress(OSError, ValueError):
            self._out.flush()

    # --- main loop --------------------------------------------------------------------------
    async def run(self) -> None:
        """Read and handle lines until ``/quit`` or end of input."""
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[str | None] = asyncio.Queue()
        sub: Subscription | None = None
        printer: asyncio.Task[None] | None = None
        if self._bus is not None:
            sub = self._bus.subscribe(
                SegmentStarted,
                UtteranceDone,
                Filtered,
                StateChanged,
                Alert,
                name="console",
                maxsize=256,
                overflow=Overflow.DROP_OLDEST,
            )
            printer = loop.create_task(self._print_events(sub), name="console-printer")
        self.print(f"โหมดข้อความ — พิมพ์แล้วกด Enter เพื่อคุยกับ{self._name} (/help ดูคำสั่ง)")
        self._attach(loop, lines)
        try:
            while True:
                line = await lines.get()
                if line is None:
                    self._eof_seen = True
                    break
                if not await self.handle(line):
                    break
        finally:
            self._detach(lines)
            if sub is not None:
                sub.close()
            if printer is not None:
                printer.cancel()
                await asyncio.wait({printer})

    # Every line goes through one ordered backlog; the loop moves it into the attached run's
    # queue (``_flush``). A run that ends hands its unread lines back to the front of the
    # backlog, so a restarted run() continues in reading order and no line is lost.
    def _attach(self, loop: asyncio.AbstractEventLoop, lines: asyncio.Queue[str | None]) -> None:
        with self._io_lock:
            self._target = (loop, lines)
            start = self._reader is None
            if start:
                stdin = self._stdin if self._stdin is not None else sys.stdin
                self._reader = threading.Thread(
                    target=self._read_stdin, args=(stdin,), name="console-stdin", daemon=True
                )
            ended = self._eof_seen and not self._backlog
        if ended:
            lines.put_nowait(None)  # input already ended in an earlier run
        self._flush()
        if start and self._reader is not None:
            self._reader.start()

    def _detach(self, lines: asyncio.Queue[str | None]) -> None:
        with self._io_lock:
            if self._target is not None and self._target[1] is lines:
                self._target = None
            unread: list[str | None] = []
            while not lines.empty():
                unread.append(lines.get_nowait())
            self._backlog.extendleft(reversed(unread))

    def _post(self, item: str | None) -> None:
        """Reader thread: queue a line (``None`` = end of input) and wake the attached run."""
        with self._io_lock:
            self._backlog.append(item)
            target = self._target
        if target is not None:
            with contextlib.suppress(RuntimeError):  # closed loop: the line waits in the backlog
                target[0].call_soon_threadsafe(self._flush)

    def _flush(self) -> None:
        """Loop thread: move the backlog, in order, into the attached run's queue."""
        with self._io_lock:
            target = self._target
            if target is None:
                return
            loop, lines = target
            try:
                on_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_loop = False
            if not on_loop:  # attached to another loop meanwhile: flush over there
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._flush)
                return
            items = list(self._backlog)
            self._backlog.clear()
        for item in items:
            lines.put_nowait(item)

    def _read_stdin(self, stdin: TextIO) -> None:
        """Reader thread: blocking ``readline`` until end of input."""
        try:
            while True:
                try:
                    line = stdin.readline()
                except UnicodeDecodeError:
                    continue
                if not line:
                    break
                self._post(line.rstrip("\r\n"))
        except (OSError, ValueError) as exc:  # closed or detached stdin
            log.info("console input ended: %r", exc)
        self._post(None)

    async def handle(self, line: str) -> bool:
        """Handle one line; returns False on ``/quit``. Never raises (except cancellation)."""
        cmd = parse_line(line)
        try:
            return await self._dispatch(cmd)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("console command failed: %r", line)
            self.print(f"[ผิดพลาด] {type(exc).__name__}: {exc}")
            return True

    async def _dispatch(self, cmd: ConsoleCommand) -> bool:
        match cmd:
            case Empty():
                return True
            case Quit():
                self.print("ลาก่อน")
                return False
            case Help():
                self.print(HELP_TEXT)
            case Invalid(message):
                self.print(f"[?] {message}")
            case Voice(text):
                self._submit_voice(text)
            case Chat(name, text):
                self._ingest_chat(chat_message(name, text, clock=self._clock))
            case Bits(name, amount, text):
                self._ingest_chat(
                    chat_message(
                        name,
                        text,
                        clock=self._clock,
                        kind=MsgKind.DONATION,
                        amount=float(amount),
                        currency="bits",
                        value_usd=round(amount / 100.0, 2),
                    )
                )
                self.print(f"[bits] {name} โดเนท {amount} bits")
            case Sub(name, months, text):
                self._ingest_chat(
                    chat_message(name, text, clock=self._clock, kind=MsgKind.SUB, months=months)
                )
                self.print(f"[sub] {name} สมัครสมาชิก {months} เดือน")
            case Op(kind, args):
                await self._op(kind, args)
            case Mem(status):
                await self._list_memories(status)
            case State():
                self._print_state()
        return True

    async def _op(self, kind: OpKind, args: Mapping[str, Any]) -> None:
        cmd = OpCommand(
            kind=kind,
            args=dict(args),
            character=self._character if kind in _CHARACTER_KINDS else None,
            operator="console",
            id=new_command_id(),
        )
        try:
            async with deadline(self._cmd_timeout, what=f"op {kind.value}", clock=self._clock):
                result = await self._control.execute(cmd)
        except DeadlineExceeded:
            self.print(f"[op] {kind.value}: หมดเวลา (core ไม่ตอบ)")
            return
        status = "สำเร็จ" if result.ok else "ไม่สำเร็จ"
        detail = f" — {result.detail}" if result.detail else ""
        self.print(f"[op] {kind.value}: {status}{detail} ({result.latency_ms:.1f} ms)")

    async def _list_memories(self, status: str | None) -> None:
        if self._memory is None:
            self.print("[mem] ไม่มีหน่วยความจำในโหมดนี้")
            return
        try:
            async with deadline(5.0, what="memory list", clock=self._clock):
                items = await self._memory.list_memories()
        except DeadlineExceeded:
            self.print("[mem] หมดเวลา")
            return
        shown = [m for m in items if (m.status == status if status else m.status != "deleted")]
        if not shown:
            self.print("[mem] ไม่มีรายการ")
            return
        for m in shown:
            flags = "".join(f" {f}" for f, on in (("ล็อก", m.locked), ("ปักหมุด", m.pinned)) if on)
            slot = f" s{m.slot}" if m.slot else ""
            self.print(f"  #{m.id}{slot} [{m.kind}/{m.status}{flags}] {m.text}")

    def _print_state(self) -> None:
        try:
            snap = self._control.snapshot()
        except Exception as exc:
            self.print(f"[state] อ่านสถานะไม่ได้: {exc}")
            return
        characters = snap.get("characters")
        if isinstance(characters, Mapping):
            for name, info in characters.items():
                state = info.get("state") if isinstance(info, Mapping) else info
                self.print(f"[state] {name}: {state}")
        for key in ("frozen", "paused", "muted", "live", "mic_mode", "ptt_active"):
            if key in snap:
                self.print(f"[state] {key}: {snap[key]}")
        llm = snap.get("llm")
        if isinstance(llm, Mapping) and llm.get("active"):
            self.print(f"[state] llm: {llm['active']}")
        tools = snap.get("tools")
        if isinstance(tools, Mapping) and tools.get("mode"):
            self.print(f"[state] tools: {tools['mode']}")
        approvals = snap.get("approvals")
        if isinstance(approvals, list):
            for item in approvals:
                if isinstance(item, Mapping):
                    self.print(
                        f"[รออนุมัติ] {item.get('id')}: {item.get('tool')} {item.get('args')} "
                        f"(/op approve {item.get('id')} | /op deny {item.get('id')})"
                    )

    # --- event echo -------------------------------------------------------------------------
    async def _print_events(self, sub: Subscription) -> None:
        async for event in sub:
            try:
                self._echo_event(event)
            except Exception:
                log.exception("console could not print %s", type(event).__name__)

    def _echo_event(self, event: Event) -> None:
        if event.character not in (None, self._character):
            return
        if isinstance(event, SegmentStarted):
            if self._echo:
                self.print(f"{self._name}: {event.caption}")
        elif isinstance(event, Filtered):
            if event.direction == "out" and self._echo:
                self.print(f"{self._name}: Filtered.")
            elif event.direction == "in":
                self.print(f"[กรอง] ข้อความขาเข้าถูกกรอง ({event.category or 'ไม่ระบุ'})")
        elif isinstance(event, UtteranceDone):
            if event.cancelled and self._echo:
                self.print(f"  (ถูกตัด: {event.reason or 'ไม่ทราบเหตุผล'})")
        elif isinstance(event, StateChanged):
            if event.new in ("paused", "pre_show") or event.old == "paused":
                self.print(f"[สถานะ] {event.old} → {event.new}")
        elif isinstance(event, Alert):
            self.print(f"[แจ้งเตือน/{event.level}] {event.message}")
