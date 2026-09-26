"""The ChatSource and ChatWindow contract suites against the real chat implementations."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest

from aivtube.chat import ScoredChatWindow, TwitchAnonIrc, WindowConfig, YouTubeListPoller
from aivtube.infra import SystemClock
from aivtube.testing.contracts import case_id, chat_source_suite, chat_window_suite
from aivtube.testing.fakes import FakeClock, FakeIrcServer, yt_text_message

_servers: list[FakeIrcServer] = []


def cases(suite: list[Any]) -> Any:
    return pytest.mark.parametrize("case", suite, ids=case_id)


@pytest.fixture
async def irc_servers() -> AsyncIterator[list[FakeIrcServer]]:
    yield _servers
    while _servers:
        await _servers.pop().stop()


async def _twitch_source() -> TwitchAnonIrc:
    server = FakeIrcServer()
    await server.start()
    _servers.append(server)
    return TwitchAnonIrc("pailin_th", clock=SystemClock(), url=server.url)


@cases(chat_source_suite(_twitch_source, expected_min=8))
async def test_twitch_anon_irc_contract(case: Callable[[], Any], irc_servers: Any) -> None:
    await case()


def _youtube_source() -> YouTubeListPoller:
    backlog = [yt_text_message("ข้อความเก่า", msg_id="old")]
    live = [yt_text_message(f"ข้อความใหม่ {i}", msg_id=f"new-{i}") for i in range(3)]
    pages = [backlog, live, [*live, yt_text_message("อีกหนึ่ง", msg_id="new-3")]]

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/videos"):
            items = [{"id": "vid", "liveStreamingDetails": {"activeLiveChatId": "chat"}}]
            return httpx2.Response(200, json={"items": items})
        token = request.url.params.get("pageToken", "")
        index = min(int(token.removeprefix("p")) if token else 0, len(pages) - 1)
        body = {
            "pollingIntervalMillis": 10,
            "nextPageToken": f"p{index + 1}",
            "items": pages[index],
        }
        return httpx2.Response(200, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return YouTubeListPoller(
        "key", video_id="vid", clock=SystemClock(), http=http, min_interval_s=0.01
    )


@cases(chat_source_suite(_youtube_source, expected_min=4))
async def test_youtube_list_poller_contract(case: Callable[[], Any]) -> None:
    await case()


_clock = FakeClock()


def _window() -> ScoredChatWindow:
    return ScoredChatWindow(_clock, WindowConfig(), ("ไพลิน", "pailin"), rng=random.Random(3))


@cases(chat_window_suite(_window, now=_clock.now))
def test_scored_chat_window_contract(case: Callable[[], None]) -> None:
    case()
