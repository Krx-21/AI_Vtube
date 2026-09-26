"""IPC messages: heartbeat death, schema validation at both ends, corr replies, ordering."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from ipc_kit import (
    TOKEN,
    frame,
    raw_peer,
    running_client,
    running_server,
    silent_core,
    wait_for,
)

from aivtube.contracts import ipc
from aivtube.infra import AsyncEventBus, DeadlineExceeded, SystemClock
from aivtube.ipc import IpcClient, IpcLinkError, IpcPeer, Link
from aivtube.testing.fakes import FakeClock

SEGMENT = {
    "utt": "u1",
    "seq": 0,
    "text": "สวัสดีค่ะ",
    "caption": "สวัสดีค่ะ",
    "emotion": None,
    "last": False,
    "kind": "speech",
}


@pytest.fixture
def clock() -> SystemClock:
    return SystemClock()


@pytest.fixture
def bus(clock: SystemClock) -> AsyncEventBus:
    return AsyncEventBus(clock)


# --- heartbeats ---------------------------------------------------------------------------


async def test_server_declares_a_silent_peer_dead_after_3_missed_heartbeats() -> None:
    clock = FakeClock()
    bus = AsyncEventBus(clock)
    lost: list[tuple[str, str]] = []
    ready: list[IpcPeer] = []
    async with running_server(clock, bus) as server:
        server.on_peer_lost(lambda role, why: lost.append((role, why)))
        server.on_peer_ready(ready.append)
        peer = await raw_peer(server.url)  # answers pings, never sends heartbeats
        try:
            await wait_for(lambda: bool(ready))
            await clock.run_for(2.5)
            assert lost == [] and server.peer("voice") is not None
            await clock.run_for(0.6)  # 3.1 s of silence: 3 heartbeats missed
            await wait_for(lambda: bool(lost))
            assert lost[0][0] == "voice" and "3 heartbeats missed" in lost[0][1]
            assert server.peer("voice") is None
            assert peer.types().count(ipc.HEARTBEAT) >= 3  # the core kept beating
        finally:
            await peer.close()


async def test_client_reports_link_lost_after_3_missed_heartbeats() -> None:
    clock = FakeClock()
    async with silent_core() as url:
        client = IpcClient(url, TOKEN, "voice", clock)
        lost: list[int] = []
        client.on_link_lost = lambda: lost.append(1)
        async with running_client(client):
            await wait_for(lambda: client.connected)
            await clock.run_for(2.0)
            assert client.connected and lost == []
            await clock.run_for(1.2)
            await wait_for(lambda: bool(lost))
            assert client.last_reason is not None and "heartbeats missed" in client.last_reason
            await wait_for(lambda: client.stats["connected"] == 2)  # and it reconnected


async def test_heartbeats_keep_a_quiet_link_alive(clock: SystemClock, bus: AsyncEventBus) -> None:
    lost: list[str] = []
    async with running_server(clock, bus, heartbeat_s=0.05) as server:
        server.on_peer_lost(lambda role, why: lost.append(why))
        client = IpcClient(server.url, TOKEN, "voice", clock, heartbeat_s=0.05)
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            await asyncio.sleep(0.5)  # ten intervals without any application message
            assert lost == [] and client.connected
            link = client.link
            assert link is not None and link.stats["rx"] >= 5


# --- validation -----------------------------------------------------------------------------


async def test_invalid_and_unknown_messages_are_dropped_and_the_link_survives(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    got: list[ipc.Envelope] = []
    async with running_server(clock, bus) as server:
        server.on(ipc.VAD_START, got.append)
        server.on(ipc.SEG_DONE, got.append)
        peer = await raw_peer(server.url)
        try:
            await wait_for(lambda: server.peer("voice") is not None)
            ws = peer.ws
            await ws.send("{broken")
            await ws.send(frame(ipc.VAD_START, {"t": "soon", "barge": False}, mid="a"))
            await ws.send(frame(ipc.VAD_START, {"t": 1.0}, mid="b"))  # missing barge
            await ws.send(
                frame(ipc.SEG_DONE, {"utt": "", "seq": 0, "heard": True, "heard_text": ""}, mid="c")
            )
            await ws.send(frame(ipc.VAD_START, {"t": 1.0, "barge": False}, mid="d", v=2))
            await ws.send(frame("from.the.future", {"x": 1}, mid="e"))
            await ws.send(frame(ipc.VAD_START, {"t": 1.0, "barge": True}, mid="ok"))
            await wait_for(lambda: bool(got))
            assert [e.id for e in got] == ["ok"] and got[0].data == {"t": 1.0, "barge": True}
            speer = server.peer("voice")
            assert speer is not None and speer.connected
            assert speer.link.stats["invalid_in"] == 5
            assert speer.link.stats["unknown_in"] == 1
        finally:
            await peer.close()


async def test_invalid_outgoing_messages_are_refused(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus) as server:
        client = IpcClient(server.url, TOKEN, "voice", clock)
        async with running_client(client):
            await wait_for(lambda: client.connected)
            assert client.post(ipc.VAD_START, {"t": 1.0}) is False  # missing barge: dropped
            with pytest.raises(ipc.IpcError):
                await client.send(ipc.VAD_END, {"t": 1.0, "audio_s": -1.0})
            with pytest.raises(ipc.IpcError):
                await client.send("made.up", {})
            assert client.post(ipc.VAD_END, {"t": 1.0, "audio_s": 0.5}) is True
            link = client.link
            assert link is not None and link.stats["invalid_out"] == 1


async def test_send_while_down_raises_and_post_drops(clock: SystemClock) -> None:
    client = IpcClient("ws://127.0.0.1:9/bus", TOKEN, "voice", clock)
    assert client.post(ipc.VAD_END, {"t": 1.0, "audio_s": 0.5}) is False
    with pytest.raises(IpcLinkError):
        await client.send(ipc.VAD_END, {"t": 1.0, "audio_s": 0.5})
    with pytest.raises(IpcLinkError):
        await client.request(ipc.PING, {})


# --- requests ------------------------------------------------------------------------------


async def test_speak_segment_is_answered_by_corr(clock: SystemClock, bus: AsyncEventBus) -> None:
    async with running_server(clock, bus) as server:
        client = IpcClient(server.url, TOKEN, "voice", clock)
        seen: list[int] = []

        def on_segment(env: ipc.Envelope) -> None:
            seen.append(env.data["seq"])
            client.reply(env, "busy" if env.data["seq"] >= 2 else "ok")

        client.on(ipc.SPEAK_SEGMENT, on_segment)
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            peer = server.peer("voice")
            assert peer is not None
            replies = [
                await peer.request(ipc.SPEAK_SEGMENT, {**SEGMENT, "seq": i}) for i in range(3)
            ]
            assert [r.data["status"] for r in replies] == ["ok", "ok", "busy"]
            assert all(r.type == ipc.REPLY and r.corr is not None for r in replies)
            assert len({r.corr for r in replies}) == 3
            assert seen == [0, 1, 2]


async def test_request_times_out_without_a_reply(clock: SystemClock, bus: AsyncEventBus) -> None:
    async with running_server(clock, bus) as server:
        client = IpcClient(server.url, TOKEN, "voice", clock)
        client.on(ipc.SPEAK_SEGMENT, lambda env: None)  # never answers
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            peer = server.peer("voice")
            assert peer is not None
            with pytest.raises(DeadlineExceeded):
                await peer.request(ipc.SPEAK_SEGMENT, SEGMENT, timeout=0.2)
            assert not peer.link._pending  # nothing left behind


async def test_pending_requests_fail_when_the_link_drops(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus) as server:
        client = IpcClient(server.url, TOKEN, "voice", clock)

        def drop(env: ipc.Envelope) -> None:
            client.disconnect("dropping mid-request")

        client.on(ipc.SPEAK_SEGMENT, drop)
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            peer = server.peer("voice")
            assert peer is not None
            with pytest.raises(IpcLinkError):
                await peer.request(ipc.SPEAK_SEGMENT, SEGMENT, timeout=5.0)


async def test_fake_clock_drives_the_request_deadline(bus: AsyncEventBus) -> None:
    clock = FakeClock()
    fbus = AsyncEventBus(clock)
    async with running_server(clock, fbus) as server:
        peer = await raw_peer(server.url)
        try:
            await wait_for(lambda: server.peer("voice") is not None)
            speer = server.peer("voice")
            assert speer is not None
            task = asyncio.create_task(speer.request(ipc.SPEAK_SEGMENT, SEGMENT, timeout=1.0))
            await asyncio.sleep(0.05)
            assert not task.done()
            await clock.run_for(1.0)
            with pytest.raises(DeadlineExceeded):
                await task
        finally:
            await peer.close()


# --- ordering and slow handlers -----------------------------------------------------------


async def test_handlers_run_in_order_and_a_slow_one_does_not_stall_liveness(
    clock: SystemClock, bus: AsyncEventBus
) -> None:
    async with running_server(clock, bus, heartbeat_s=0.05) as server:
        client = IpcClient(server.url, TOKEN, "voice", clock, heartbeat_s=0.05)
        order: list[str] = []

        async def slow(env: ipc.Envelope) -> None:
            order.append("configure:start")
            await asyncio.sleep(0.4)  # 8 heartbeat intervals: model loading
            order.append("configure:end")

        client.on(ipc.VOICE_CONFIGURE, slow)
        client.on(ipc.SPEAK_BEGIN, lambda env: order.append(f"begin:{env.data['utt']}"))
        client.on(ipc.VOICE_MUTE, lambda env: order.append(f"mute:{env.data['on']}"))
        lost: list[str] = []
        server.on_peer_lost(lambda role, why: lost.append(why))
        async with running_client(client):
            await wait_for(lambda: server.peer("voice") is not None)
            peer = server.peer("voice")
            assert peer is not None
            configure: dict[str, Any] = {
                "audio": {},
                "vad": {},
                "barge_in": {},
                "stt_chain": [],
                "tts": {"identities": {}, "backends": {}, "chunk": {}},
                "characters": {},
            }
            await peer.send(ipc.VOICE_CONFIGURE, configure)
            await peer.send(
                ipc.SPEAK_BEGIN,
                {"utt": "u1", "character": "pailin", "filler_after_s": None, "gate_open": True},
            )
            await peer.send(ipc.VOICE_MUTE, {"on": True})
            pong = await peer.request(ipc.PING, {"seq": 9}, timeout=0.2)  # answered at once
            assert pong.type == ipc.PONG and pong.data["ping_ts"] > 0
            await wait_for(lambda: len(order) == 4)
            assert order == ["configure:start", "configure:end", "begin:u1", "mute:True"]
            assert lost == [] and client.connected


def test_outgoing_queue_overflow_drops_droppable_frames_first(clock: SystemClock) -> None:
    async def nothing(env: ipc.Envelope) -> None:
        return None

    link = Link(
        object(),  # type: ignore[arg-type]  # the queue never touches the socket
        clock=clock,
        local_role="voice",
        peer_role="core",
        dispatch=nothing,
        max_queue=16,
    )
    for i in range(10):
        link.post(
            ipc.LIP_TRACK,
            {
                "utt": "u",
                "seq": i,
                "t0": 0.0,
                "fps": 60,
                "mouth": [0.1],
                "form": [0.5],
                "final": False,
            },
        )
    for i in range(10):
        assert link.post(ipc.SEG_DONE, {"utt": "u", "seq": i, "heard": True, "heard_text": ""})
    queued = [t for t, _ in link._out]
    assert len(queued) == 16 and queued.count(ipc.SEG_DONE) == 10  # lip tracks made room
    for i in range(10):
        link.post(
            ipc.LIP_TRACK,
            {
                "utt": "u",
                "seq": i,
                "t0": 0.0,
                "fps": 60,
                "mouth": [0.1],
                "form": [0.5],
                "final": False,
            },
        )
    assert [t for t, _ in link._out].count(ipc.SEG_DONE) == 10
    assert link.stats["dropped_overflow"] >= 14
