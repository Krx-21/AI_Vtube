"""UDP 47779 discovery: broadcast parsing, instance choice, a real UDP listener."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from typing import Any

import pytest

from aivtube.avatar import VTSDiscovery, discover_vts, parse_broadcast, pick_instance
from aivtube.avatar.discovery import resolve_vts_url
from aivtube.testing.fakes import FakeClock


def broadcast(port: int, title: str, instance: str | None = None, active: bool = True) -> bytes:
    return json.dumps(
        {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "timestamp": 1_760_000_000_000,
            "messageType": "VTubeStudioAPIStateBroadcast",
            "requestID": "VTubeStudioAPIStateBroadcast",
            "data": {
                "active": active,
                "port": port,
                "instanceID": instance or f"id-{port}",
                "windowTitle": title,
            },
        }
    ).encode()


def send_udp(port: int, *payloads: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for p in payloads:
            s.sendto(p, ("127.0.0.1", port))


def test_parse_valid_broadcast() -> None:
    inst = parse_broadcast(broadcast(8002, "VTube Studio Window 2", "abc"), "192.168.1.5")
    assert inst == {
        "active": True,
        "port": 8002,
        "instanceID": "abc",
        "windowTitle": "VTube Studio Window 2",
        "host": "192.168.1.5",
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[1, 2]",
        json.dumps({"messageType": "APIStateResponse", "data": {"port": 8001}}).encode(),
        json.dumps({"messageType": "VTubeStudioAPIStateBroadcast", "data": []}).encode(),
        json.dumps({"messageType": "VTubeStudioAPIStateBroadcast", "data": {"port": 0}}).encode(),
        json.dumps(
            {"messageType": "VTubeStudioAPIStateBroadcast", "data": {"port": True}}
        ).encode(),
        json.dumps(
            {"messageType": "VTubeStudioAPIStateBroadcast", "data": {"port": "8001"}}
        ).encode(),
        b"\xff\xfe",
    ],
)
def test_parse_rejects_junk(raw: bytes) -> None:
    assert parse_broadcast(raw, "127.0.0.1") is None


def _inst(port: int, title: str, active: bool = True) -> dict[str, Any]:
    return {"active": active, "port": port, "instanceID": f"i{port}", "windowTitle": title}


def test_pick_instance_rules() -> None:
    main, twin = _inst(8001, "VTube Studio"), _inst(8002, "VTube Studio Window 2")
    both = [twin, main]
    assert pick_instance(both, "vtube studio window 2") == twin  # case-insensitive title
    assert pick_instance(both, "Window 3") is None
    assert pick_instance(both, prefer_port=8002) == twin
    assert pick_instance(both) == main  # ambiguous → the main window
    assert pick_instance([twin]) == twin  # the only one
    assert pick_instance([_inst(8003, "custom"), _inst(8004, "other")]) is None
    assert pick_instance([_inst(8001, "VTube Studio", active=False)]) is None  # API off


def test_resolve_url_falls_back_to_config() -> None:
    insts = [_inst(8001, "VTube Studio"), _inst(8002, "VTube Studio Window 2")]
    url = "ws://127.0.0.1:8001"
    assert resolve_vts_url(url, "VTube Studio Window 2", insts) == "ws://127.0.0.1:8002"
    assert resolve_vts_url(url, "", insts) == url
    assert resolve_vts_url(url, "Window 9", insts) == url
    assert resolve_vts_url(url, "", []) == url
    assert resolve_vts_url("ws://127.0.0.1:8001", "", [_inst(8005, "moved")]) == (
        "ws://127.0.0.1:8005"
    )


async def test_listener_receives_fake_udp_packets() -> None:
    disc = VTSDiscovery(port=0, host="127.0.0.1")
    await disc.start()
    try:
        assert disc.running and disc.port > 0
        waiter = asyncio.ensure_future(disc.wait_for("VTube Studio Window 2", timeout=3.0))
        await asyncio.sleep(0.05)
        send_udp(
            disc.port,
            b"garbage",
            broadcast(8001, "VTube Studio", "a" * 32),
            broadcast(8002, "VTube Studio Window 2", "b" * 32),
        )
        found = await waiter
        assert found is not None and found["port"] == 8002 and found["host"] == "127.0.0.1"
        await asyncio.sleep(0.05)
        assert sorted(i["port"] for i in disc.instances()) == [8001, 8002]
        assert disc.received == 2
    finally:
        disc.close()
    assert not disc.running


async def test_wait_for_times_out_and_instances_go_stale() -> None:
    clock = FakeClock()
    disc = VTSDiscovery(port=0, host="127.0.0.1", clock=clock, stale_s=6.0)
    disc.feed(broadcast(8001, "VTube Studio"), "127.0.0.1")
    assert disc.pick() is not None
    waiter = asyncio.ensure_future(disc.wait_for("Window 2", timeout=1.0))
    await clock.run_for(1.1)
    assert waiter.done() and waiter.result() is None
    await clock.run_for(6.0)
    assert disc.instances() == [] and disc.pick() is None
    # `after` ignores broadcasts heard before a moment
    disc.feed(broadcast(8001, "VTube Studio"), "127.0.0.1")
    assert await disc.wait_for("", timeout=0.0) is not None
    later = asyncio.ensure_future(disc.wait_for("", timeout=5.0, after=clock.now()))
    await clock.run_for(0.5)
    assert not later.done()
    await clock.run_for(0.1)
    disc.feed(broadcast(8001, "VTube Studio"), "127.0.0.1")
    await clock.run_until_idle()
    assert later.done() and later.result() is not None


async def test_discover_vts_collects_broadcasts(free_udp_port: int) -> None:
    async def sender() -> None:
        await asyncio.sleep(0.1)
        send_udp(
            free_udp_port, broadcast(8001, "VTube Studio"), broadcast(8002, "VTube Studio Window 2")
        )

    task = asyncio.ensure_future(sender())
    found = await discover_vts(0.4, port=free_udp_port)
    await task
    assert sorted((i["windowTitle"], i["port"]) for i in found) == [
        ("VTube Studio", 8001),
        ("VTube Studio Window 2", 8002),
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows SO_REUSEADDR may share the port")
async def test_discover_vts_returns_empty_when_port_is_taken(free_udp_port: int) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        blocker.bind(("0.0.0.0", free_udp_port))  # no SO_REUSEADDR: the listener cannot bind
        assert await discover_vts(0.1, port=free_udp_port) == []
    finally:
        blocker.close()
