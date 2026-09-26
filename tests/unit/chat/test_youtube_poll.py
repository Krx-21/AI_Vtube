"""YouTubeListPoller with httpx2.MockTransport fixtures (and FakeYouTubeServer over real HTTP):
discovery, backlog skip, polling interval, super chats, memberships, gifts, quota, errors."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

from aivtube.chat import YouTubeListPoller, message_from_item, resolve_live_chat_id
from aivtube.chat.youtube_poll import (
    YouTubeApiError,
    _handle_path,
    live_video_id_from_handle,
    video_id_from_live_page,
)
from aivtube.contracts.chat import ChatSource
from aivtube.contracts.types import ChatMessage, HealthState, MsgKind, Platform
from aivtube.testing.fakes import FakeClock, FakeYouTubeServer, yt_super_chat, yt_text_message

VIDEO = "dQw4w9WgXcQ"
CHAT = "live-chat-1"
KEY = "test-api-key"


async def until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > end:
            raise AssertionError(f"timed out waiting for {predicate}")
        await asyncio.sleep(0.005)


def text_item(msg_id: str, text: str, **author: Any) -> dict[str, Any]:
    return yt_text_message(text, msg_id=msg_id, **author)


def gift_item(msg_id: str, combo: int, jewels: int = 10) -> dict[str, Any]:
    item = yt_text_message("", msg_id=msg_id, name="ผู้ให้ของขวัญ")
    item["snippet"] = {
        "type": "giftEvent",
        "liveChatId": CHAT,
        "publishedAt": item["snippet"]["publishedAt"],
        "giftDetails": {"giftName": "ดอกไม้", "jewelsAmount": jewels, "comboCount": combo},
    }
    return item


def event_item(msg_id: str, typ: str, details_key: str | None, details: dict[str, Any]) -> dict:
    item = yt_text_message("", msg_id=msg_id, name="สมาชิก")
    snippet: dict[str, Any] = {
        "type": typ,
        "liveChatId": CHAT,
        "publishedAt": item["snippet"]["publishedAt"],
    }
    if details_key:
        snippet[details_key] = details
    item["snippet"] = snippet
    return item


class YtApi:
    """A scripted YouTube Data API for ``httpx2.MockTransport``.

    ``pages[0]`` is the backlog (first page, no pageToken); ``pages[n]`` answers
    ``pageToken=page-n``. ``fail_next`` queues error answers for liveChat/messages calls.
    """

    def __init__(
        self,
        pages: list[list[dict[str, Any]]] | None = None,
        *,
        interval_ms: int | None = 5000,
        live_html: str | None = None,
    ) -> None:
        self.pages = pages if pages is not None else [[]]
        self.interval_ms = interval_ms
        self.live_html = live_html
        self.live_chat_id: str | None = CHAT
        self.requests: list[httpx2.Request] = []
        self.fail_next: list[tuple[int, str]] = []
        self.raise_next: list[Exception] = []
        self.offline_after: int | None = None

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))

    @property
    def list_calls(self) -> list[httpx2.Request]:
        return [r for r in self.requests if r.url.path.endswith("/liveChat/messages")]

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        q = request.url.params
        if path.endswith("/live"):
            return httpx2.Response(200 if self.live_html else 404, text=self.live_html or "")
        if q.get("key") != KEY:
            return self.error(400, "keyInvalid", "API key not valid.")
        if path == "/youtube/v3/videos":
            items = []
            if q.get("id") == VIDEO and q.get("part") == "liveStreamingDetails":
                details: dict[str, Any] = {"actualStartTime": "2026-09-25T12:00:00Z"}
                if self.live_chat_id:
                    details["activeLiveChatId"] = self.live_chat_id
                items.append({"id": VIDEO, "liveStreamingDetails": details})
            return httpx2.Response(200, json={"items": items})
        if path == "/youtube/v3/liveChat/messages":
            if self.raise_next:
                raise self.raise_next.pop(0)
            if self.fail_next:
                status, reason = self.fail_next.pop(0)
                return self.error(status, reason, f"{reason} (scripted)")
            assert q.get("part") == "id,snippet,authorDetails"
            token = q.get("pageToken", "")
            index = int(token.removeprefix("page-")) if token else 0
            items = self.pages[index] if index < len(self.pages) else []
            body: dict[str, Any] = {"nextPageToken": f"page-{min(index + 1, len(self.pages))}"}
            if self.interval_ms is not None:
                body["pollingIntervalMillis"] = self.interval_ms
            body["items"] = items
            if self.offline_after is not None and index >= self.offline_after:
                body["offlineAt"] = "2026-09-25T13:00:00Z"
            return httpx2.Response(200, json=body)
        return httpx2.Response(404, json={})

    @staticmethod
    def error(status: int, reason: str, message: str) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"error": {"code": status, "message": message, "errors": [{"reason": reason}]}},
        )


class Reader:
    def __init__(self, src: YouTubeListPoller) -> None:
        self.src = src
        self.got: list[ChatMessage] = []
        self.task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        async for m in self.src.messages():
            self.got.append(m)

    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.got]

    async def stop(self) -> None:
        await self.src.aclose()
        await asyncio.wait_for(self.task, 3.0)


def poller(api: YtApi, clock: Any, **kwargs: Any) -> YouTubeListPoller:
    kwargs.setdefault("video_id", VIDEO)
    return YouTubeListPoller(KEY, clock=clock, http=api.client(), **kwargs)


def backlog_page(fixtures_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (fixtures_dir / "youtube" / "live_chat_page.json").read_text(encoding="utf-8")
    )
    return data


# --- discovery -------------------------------------------------------------------------------
async def test_resolve_live_chat_id() -> None:
    api = YtApi()
    async with api.client() as http:
        assert await resolve_live_chat_id(http, KEY, VIDEO) == CHAT
        assert await resolve_live_chat_id(http, KEY, "unknownvid1") is None
        api.live_chat_id = None  # a finished broadcast has no activeLiveChatId
        assert await resolve_live_chat_id(http, KEY, VIDEO) is None
        with pytest.raises(YouTubeApiError) as err:
            await resolve_live_chat_id(http, "wrong", VIDEO)
    assert err.value.status == 400 and err.value.reason == "keyInvalid"
    first = api.requests[0].url
    assert first.host == "www.googleapis.com" and first.path == "/youtube/v3/videos"
    assert dict(first.params) == {"part": "liveStreamingDetails", "id": VIDEO, "key": KEY}


LIVE_HTML = (
    "<html><script>var ytInitialData = "
    + json.dumps(
        {
            "currentVideoEndpoint": {
                "clickTrackingParams": "abc",
                "commandMetadata": {"webCommandMetadata": {"url": f"/watch?v={VIDEO}"}},
                "watchEndpoint": {"videoId": VIDEO},
            },
            "contents": {"x": [1, 2, {"isLive": True}]},
        },
        separators=(",", ":"),
    )
    + ';</script><script>var ytInitialPlayerResponse = {"videoDetails":{"isLive":true}};'
    "</script></html>"
)


def test_video_id_from_live_page() -> None:
    assert video_id_from_live_page(LIVE_HTML) == VIDEO
    assert video_id_from_live_page(LIVE_HTML.replace('"isLive":true', '"isLive":false')) is None
    broken = LIVE_HTML.replace("var ytInitialData = ", "var ytInitialData = {broken ")
    assert video_id_from_live_page(broken) == VIDEO  # regex fallback
    assert video_id_from_live_page('"isLive":true but no data') is None


@pytest.mark.parametrize(
    ("handle", "path"),
    [
        ("@pailin_th", "@pailin_th"),
        ("pailin_th", "@pailin_th"),
        ("https://www.youtube.com/@pailin_th/live", "@pailin_th"),
        ("https://www.youtube.com/@pailin_th/", "@pailin_th"),
        ("UCabcdefghijklmnopqrstuv", "channel/UCabcdefghijklmnopqrstuv"),
    ],
)
def test_handle_path(handle: str, path: str) -> None:
    assert _handle_path(handle) == path


async def test_live_video_id_from_handle() -> None:
    api = YtApi(live_html=LIVE_HTML)
    async with api.client() as http:
        assert await live_video_id_from_handle(http, "pailin_th") == VIDEO
    request = api.requests[0]
    assert str(request.url) == "https://www.youtube.com/@pailin_th/live"
    assert "key" not in request.url.params  # the page scrape costs no quota
    api.live_html = None
    async with api.client() as http:
        assert await live_video_id_from_handle(http, "@pailin_th") is None


async def test_handle_discovery_is_confirmed_with_videos_list(real_clock: Any) -> None:
    api = YtApi([[], [text_item("h-1", "สวัสดีค่ะ")]], interval_ms=10, live_html=LIVE_HTML)
    src = poller(api, real_clock, video_id=None, handle="@pailin_th", min_interval_s=0.01)
    reader = Reader(src)
    await until(lambda: reader.ids == ["h-1"])
    paths = [r.url.path for r in api.requests[:3]]
    assert paths == ["/@pailin_th/live", "/youtube/v3/videos", "/youtube/v3/liveChat/messages"]
    assert src.live_chat_id == CHAT
    await reader.stop()


async def test_not_live_handle_degrades_and_retries(fake_clock: FakeClock) -> None:
    api = YtApi(live_html=None)
    src = poller(api, fake_clock, video_id=None, handle="@pailin_th")
    reader = Reader(src)
    await until(lambda: src.health().state is HealthState.DEGRADED)
    assert "not live" in src.health().detail and src.quota_used == 0
    api.live_html = LIVE_HTML
    await until(lambda: fake_clock.pending >= 1)
    await fake_clock.run_for(60.0)
    await until(lambda: src.live_chat_id == CHAT)
    await reader.stop()


# --- polling ---------------------------------------------------------------------------------
async def test_first_page_is_backlog_and_interval_honours_the_server(
    fake_clock: FakeClock, fixtures_dir: Path
) -> None:
    fixture = backlog_page(fixtures_dir)
    live = [text_item("new-1", "เพิ่งมาค่ะ"), text_item("new-2", "ไพลินเล่นเกมอะไร")]
    api = YtApi([fixture["items"], live], interval_ms=fixture["pollingIntervalMillis"])
    src = poller(api, fake_clock)
    assert isinstance(src, ChatSource) and src.platform is Platform.YOUTUBE
    reader = Reader(src)
    await fake_clock.run_until_idle()
    assert reader.got == [] and src.skipped_backlog == len(fixture["items"])
    assert src.quota_used == 2 and src.health().state is HealthState.OK
    assert src.last_interval == 5.0 and 5.0 in fake_clock.sleeps
    await fake_clock.run_for(4.9)
    assert len(api.list_calls) == 1
    await fake_clock.run_for(0.1)
    assert reader.ids == ["new-1", "new-2"]
    assert api.list_calls[1].url.params["pageToken"] == "page-1"
    assert api.list_calls[1].url.params["maxResults"] == "2000"
    assert src.quota_used == 3 and src.polls == 2
    api.interval_ms = 500  # faster than the 2 s floor
    await fake_clock.run_for(5.0)
    assert src.last_interval == 2.0
    api.interval_ms = None  # missing: fall back to 5 s
    await fake_clock.run_for(2.0)
    assert src.last_interval == 5.0
    await reader.stop()


async def test_super_chat_and_author_flags(fake_clock: FakeClock) -> None:
    sc = yt_super_chat("ไพลินร้องเพลงหน่อย", 100.0, "THB", msg_id="sc-1", name="ใจดี")
    usd = yt_super_chat("", 5.0, "USD", msg_id="sc-2")
    sticker = copy.deepcopy(usd)
    sticker["id"] = "st-1"
    details = sticker["snippet"].pop("superChatDetails")
    sticker["snippet"]["type"] = "superStickerEvent"
    sticker["snippet"]["superStickerDetails"] = {
        "amountMicros": details["amountMicros"],
        "currency": "JPY",
        "tier": 1,
    }
    flags = [
        text_item("mod-1", "ใจเย็นนะทุกคน", moderator=True),
        text_item("mem-1", "มาแล้วจ้า", sponsor=True, verified=True),
        text_item("own-1", "ขอบคุณทุกคน", owner=True),
    ]
    api = YtApi([[], [sc, usd, sticker, *flags]])
    src = poller(api, fake_clock)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    await fake_clock.run_for(5.0)
    by_id = {m.id: m for m in reader.got}
    donation = by_id["sc-1"]
    assert donation.kind is MsgKind.DONATION and donation.text == "ไพลินร้องเพลงหน่อย"
    assert (donation.amount, donation.currency) == (100.0, "THB")
    assert donation.value_usd == pytest.approx(2.8)
    assert donation.user.name == "ใจดี" and donation.platform is Platform.YOUTUBE
    assert by_id["sc-2"].value_usd == pytest.approx(5.0)
    assert by_id["st-1"].kind is MsgKind.DONATION and by_id["st-1"].currency == "JPY"
    assert by_id["mod-1"].user.is_mod and not by_id["mod-1"].user.is_sub
    assert by_id["mem-1"].user.is_sub and by_id["mem-1"].user.is_verified
    assert by_id["own-1"].user.is_broadcaster
    await reader.stop()


def test_memberships_and_ignored_types() -> None:
    new = event_item("ns-1", "newSponsorEvent", "newSponsorDetails", {"memberLevelName": "แฟน"})
    milestone = event_item(
        "ms-1",
        "memberMilestoneChatEvent",
        "memberMilestoneChatDetails",
        {"memberMonth": 7, "userComment": "ครบเจ็ดเดือน"},
    )
    gifting = event_item(
        "mg-1", "membershipGiftingEvent", "membershipGiftingDetails", {"giftMembershipsCount": 5}
    )
    m_new = message_from_item(new, received=1.0)
    m_ms = message_from_item(milestone, received=1.0)
    m_gift = message_from_item(gifting, received=1.0)
    assert m_new is not None and m_new.kind is MsgKind.SUB and m_new.user.is_sub
    assert m_ms is not None and m_ms.kind is MsgKind.SUB and m_ms.user.sub_months == 7
    assert m_ms.text == "ครบเจ็ดเดือน"
    assert m_gift is not None and m_gift.kind is MsgKind.GIFT_SUB and m_gift.amount == 5.0
    for typ in ("tombstone", "pollEvent", "userBannedEvent", "chatEndedEvent", "sponsorOnlyMode"):
        assert message_from_item(event_item("x", typ, None, {}), received=1.0) is None
    no_id = text_item("", "hi")
    no_id["id"] = ""
    assert message_from_item(no_id, received=1.0) is None
    display_only = text_item("d-1", "")
    del display_only["snippet"]["textMessageDetails"]
    display_only["snippet"]["displayMessage"] = "แสดงผล"
    shown = message_from_item(display_only, received=1.0)
    assert shown is not None and shown.text == "แสดงผล"


def test_published_at_maps_to_our_timebase() -> None:
    item = text_item("t-1", "hi", published_at="2026-09-25T12:00:00.000Z")
    wall = 1_790_337_602.5  # 2.5 s after publishedAt
    m = message_from_item(item, received=100.0, wall_now=wall)
    assert m is not None and m.ts == pytest.approx(97.5) and m.received == 100.0
    bad = text_item("t-2", "hi", published_at="not a date")
    m2 = message_from_item(bad, received=100.0, wall_now=wall)
    assert m2 is not None and m2.ts == 100.0


async def test_gift_combos_become_increments(fake_clock: FakeClock) -> None:
    api = YtApi([[gift_item("g-0", 1)], [gift_item("g-1", 1)], [gift_item("g-1", 3)]])
    api.pages.append([gift_item("g-1", 3), gift_item("g-0", 2)])
    src = poller(api, fake_clock)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    for _ in range(3):
        await fake_clock.run_for(5.0)
    got = [(m.id, m.amount, m.currency) for m in reader.got]
    assert got == [
        ("g-1", 10.0, "JEWEL"),
        ("g-1#3", 20.0, "JEWEL"),  # combo 1 -> 3 adds two more gifts
        ("g-0#2", 10.0, "JEWEL"),  # g-0 combo 1 was backlog; only the new gift counts
    ]
    assert src.duplicates == 1  # g-1 at combo 3 again
    assert reader.got[1].value_usd == pytest.approx(0.2)
    await reader.stop()


async def test_ids_are_deduped_across_pages(fake_clock: FakeClock) -> None:
    api = YtApi([[text_item("a", "เก่า")], [text_item("b", "ใหม่"), text_item("a", "เก่า")]])
    api.pages.append([text_item("b", "ใหม่"), text_item("c", "อีก")])
    src = poller(api, fake_clock)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    await fake_clock.run_for(10.0)
    assert reader.ids == ["b", "c"] and src.duplicates == 2
    await reader.stop()


# --- quota and errors ------------------------------------------------------------------------
async def test_quota_403_degrades_with_a_hint_and_backs_off(fake_clock: FakeClock) -> None:
    api = YtApi([[], [text_item("q-1", "ยังอยู่")]])
    src = poller(api, fake_clock, quota_backoff_s=900.0)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    api.fail_next.append((403, "quotaExceeded"))
    await fake_clock.run_for(5.0)
    health = src.health()
    assert health.state is HealthState.DEGRADED
    assert "quota" in health.detail and "โควตา" in health.detail and "403" in health.detail
    assert src.quota_used == 3  # videos + backlog + the refused call
    await fake_clock.run_for(899.0)
    assert len(api.list_calls) == 2
    await fake_clock.run_for(1.0)
    assert reader.ids == ["q-1"] and src.health().state is HealthState.OK
    await reader.stop()


async def test_rejected_key_degrades_with_a_hint(fake_clock: FakeClock) -> None:
    api = YtApi()
    src = YouTubeListPoller("wrong-key", video_id=VIDEO, clock=fake_clock, http=api.client())
    reader = Reader(src)
    await fake_clock.run_until_idle()
    health = src.health()
    assert health.state is HealthState.DEGRADED and "YOUTUBE_API_KEY" in health.detail
    assert fake_clock.next_deadline() == pytest.approx(fake_clock.now() + 900.0)
    await reader.stop()


async def test_forbidden_403_degrades_with_a_hint(fake_clock: FakeClock) -> None:
    api = YtApi()
    api.fail_next.append((403, "forbidden"))
    src = poller(api, fake_clock)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    detail = src.health().detail
    assert "Data API v3" in detail and "คีย์" in detail
    await reader.stop()


async def test_rate_limit_and_network_errors_back_off(fake_clock: FakeClock) -> None:
    api = YtApi([[], [text_item("r-1", "มาแล้ว")]])
    api.fail_next.extend([(403, "rateLimitExceeded"), (500, "backendError")])
    api.raise_next.append(httpx2.ConnectError("offline"))
    src = poller(api, fake_clock, backoff=(2.0, 60.0))
    src._rng.seed(0)
    reader = Reader(src)
    await fake_clock.run_until_idle()  # the first list call raises ConnectError
    assert "unreachable" in src.health().detail
    delays = []
    for _ in range(3):
        deadline = fake_clock.next_deadline()
        assert deadline is not None
        delays.append(deadline - fake_clock.now())
        await fake_clock.run_for(delays[-1])
    assert 1.6 <= delays[0] <= 2.4 and 3.2 <= delays[1] <= 4.8 and 6.4 <= delays[2] <= 9.6
    assert src.health().state is HealthState.OK and reader.got == []  # the backlog page
    await fake_clock.run_for(5.0)
    assert reader.ids == ["r-1"]
    await reader.stop()


async def test_ended_chat_and_404_re_resolve(fake_clock: FakeClock) -> None:
    api = YtApi([[], [text_item("e-1", "บ๊ายบาย")]])
    api.offline_after = 1
    src = poller(api, fake_clock, not_live_retry_s=60.0)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    await fake_clock.run_for(5.0)
    assert reader.ids == ["e-1"]  # the last page is still delivered
    assert src.live_chat_id is None and "ended" in src.health().detail
    api.offline_after = None
    api.fail_next.append((404, "liveChatNotFound"))
    await fake_clock.run_for(60.0)  # re-resolve (videos.list) then 404 on the list call
    assert src.live_chat_id is None and "gone" in src.health().detail
    videos_calls = [r for r in api.requests if r.url.path.endswith("/videos")]
    assert len(videos_calls) == 2
    await fake_clock.run_for(60.0)
    assert src.live_chat_id == CHAT and src.health().state is HealthState.OK
    assert src.skipped_backlog == 0  # the new first page was the (empty) backlog
    await reader.stop()


async def test_quota_soft_limit_doubles_the_interval(fake_clock: FakeClock) -> None:
    api = YtApi([[], [], []])
    src = poller(api, fake_clock, quota_soft_limit=3)
    reader = Reader(src)
    await fake_clock.run_until_idle()
    assert src.last_interval == 5.0  # 2 units used
    await fake_clock.run_for(5.0)
    assert src.quota_used == 3 and src.last_interval == 10.0
    assert src.health().state is HealthState.DEGRADED and "quota low" in src.health().detail
    await reader.stop()


# --- lifecycle -------------------------------------------------------------------------------
async def test_constructor_validation_and_from_config(real_clock: Any) -> None:
    with pytest.raises(ValueError):
        YouTubeListPoller(KEY, clock=real_clock)
    with pytest.raises(ValueError):
        YouTubeListPoller("", video_id=VIDEO, clock=real_clock)
    from aivtube.config.schema import YouTubePollConfig

    cfg = YouTubePollConfig(video_id=VIDEO, min_interval_s=3.0)
    src = YouTubeListPoller.from_config(cfg, api_key=KEY, clock=real_clock)
    assert src.video_id == VIDEO and src.handle is None and src.min_interval_s == 3.0
    await src.aclose()


async def test_aclose_closes_an_owned_client_and_is_idempotent(real_clock: Any) -> None:
    src = YouTubeListPoller(KEY, live_chat_id=CHAT, clock=real_clock)
    client = src._client()
    await src.aclose()
    await src.aclose()
    assert client.is_closed and src.health().state is HealthState.DOWN
    assert [m async for m in src.messages()] == []
    api = YtApi()
    shared = api.client()
    src2 = poller(api, real_clock)
    await src2.aclose()
    assert not shared.is_closed  # an injected client belongs to the caller
    await shared.aclose()


async def test_against_the_fake_youtube_server(real_clock: Any) -> None:
    backlog = [yt_text_message("ข้อความเก่า", msg_id="old-1")]
    async with (
        FakeYouTubeServer(backlog, polling_interval_ms=10, api_key=KEY) as server,
        httpx2.AsyncClient(trust_env=False) as http,
    ):
        src = YouTubeListPoller(
            KEY,
            video_id=server.video_id,
            clock=real_clock,
            http=http,
            api_root=server.api_root,
            min_interval_s=0.01,
        )
        reader = Reader(src)
        await until(lambda: src.skipped_backlog == 1)
        server.push(yt_super_chat("เป็นกำลังใจให้นะ", 50.0, msg_id="live-sc"))
        await until(lambda: reader.ids == ["live-sc"])
        assert reader.got[0].kind is MsgKind.DONATION and reader.got[0].amount == 50.0
        server.forbid()
        await until(lambda: src.health().state is HealthState.DEGRADED)
        assert "quota" in src.health().detail
        assert src.quota_used == server.quota_used
        await reader.stop()


@pytest.mark.network
async def test_live_youtube_key_smoke() -> None:
    """Real API (AIVTUBE_NETWORK=1 and YOUTUBE_API_KEY): a bogus video resolves to None."""
    import os

    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        pytest.skip("YOUTUBE_API_KEY is not set")
    async with httpx2.AsyncClient() as http:
        assert await resolve_live_chat_id(http, key, "aaaaaaaaaaa") is None
