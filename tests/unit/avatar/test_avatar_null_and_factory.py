"""NullSink (contract subset) and building the avatar from the real config files."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from aivtube.avatar import (
    LiveAvatarDriver,
    NullSink,
    VTSSink,
    build_avatar_driver,
    build_avatar_sink,
)
from aivtube.config import load_character, load_config
from aivtube.contracts.avatar import AvatarSink
from aivtube.contracts.types import HealthState
from aivtube.infra import SystemClock
from aivtube.testing.contracts import avatar_sink_suite, case_id
from aivtube.testing.fakes import FakeClock

ROOT = Path(__file__).resolve().parents[3]

# NullSink renders nothing: its hotkeys never "work" and its health is DISABLED by design.
_NULL_CASES = [
    c
    for c in avatar_sink_suite(NullSink)
    if case_id(c).split(".")[-1] not in {"trigger_returns_bool", "run_connects_and_reports_ok"}
]


@pytest.mark.parametrize("case", _NULL_CASES, ids=case_id)
async def test_null_sink_contract(case: Callable[[], Awaitable[None]]) -> None:
    await case()


async def test_null_sink_behaviour() -> None:
    sink = NullSink()
    assert isinstance(sink, AvatarSink) and not sink.connected
    task = asyncio.ensure_future(sink.run())
    await asyncio.sleep(0)
    assert sink.connected
    sink.set_params({"MouthOpen": 0.5})
    await sink.set_emotion("happy")
    assert sink.frames == 1 and sink.last_values == {"MouthOpen": 0.5} and sink.emotion == "happy"
    assert await sink.trigger("happy") is False
    await sink.move(rotation=90.0)
    health = sink.health()
    assert health.state is HealthState.DISABLED and health.component == "avatar:null"
    sink.stop()
    await task
    assert not sink.connected


def test_build_vts_sink_from_config(tmp_path: Path) -> None:
    app = load_config(ROOT, env={})
    char = load_character(ROOT, "pailin", app=app)
    sink = build_avatar_sink(app, char, SystemClock(), discovery=False)
    assert isinstance(sink, VTSSink)
    assert sink.client.url == "ws://127.0.0.1:8001"
    assert sink.client.plugin_name == "AI_Vtube Brain"
    assert sink.client.token_path == char.resolve_path(char.avatar.token_file)
    assert sink.client.token_path.name == "vts_pailin.txt"
    assert sink.client.request_timeout == 2.0 and sink.client.max_inflight == 8
    assert sink.component == "avatar:pailin"
    assert sink.health().state is HealthState.STARTING
    assert sink._custom_name("MouthOpen") == "PailinMouthOpen"


def test_build_null_sink_when_disabled() -> None:
    app = load_config(ROOT, env={}, cli_overrides={"avatar.sink": "none"})
    char = load_character(ROOT, "pailin", app=app)
    sink = build_avatar_sink(app, char, SystemClock())
    assert isinstance(sink, NullSink) and sink.component == "avatar:pailin"
    browser = load_config(ROOT, env={}, cli_overrides={"avatar.sink": "browser"})
    assert isinstance(build_avatar_sink(browser, char, SystemClock()), NullSink)


def test_build_driver_from_config() -> None:
    app = load_config(ROOT, env={}, cli_overrides={"avatar.lead_ms": 60})
    char = load_character(ROOT, "pailin", app=app)
    drv = build_avatar_driver(app, char, NullSink(), FakeClock())
    assert isinstance(drv, LiveAvatarDriver)
    assert drv.fps == 60 and drv.lead_s == pytest.approx(0.060)
    assert drv.emotion_fade_s == pytest.approx(0.3)
    drv.set_emotion("happy")
    assert drv.emotion == "happy"
    drv.set_emotion("not-an-emotion")
    assert drv.emotion == "neutral"
