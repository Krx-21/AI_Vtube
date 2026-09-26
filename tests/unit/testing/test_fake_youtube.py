"""FakeYouTubeServer: liveChatId lookup, backlog page, polling pages, quota and 403."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2

from aivtube.testing.fakes import FakeYouTubeServer, yt_super_chat, yt_text_message


async def test_backlog_then_new_messages_and_quota() -> None:
    srv = FakeYouTubeServer([yt_text_message("เก่า", msg_id="old-1")], polling_interval_ms=3000)
    root = await srv.start()
    try:
        async with httpx2.AsyncClient(base_url=root, trust_env=False) as http:
            video = (
                await http.get(
                    "/videos",
                    params={"part": "liveStreamingDetails", "id": srv.video_id, "key": "k"},
                )
            ).json()
            chat_id = video["items"][0]["liveStreamingDetails"]["activeLiveChatId"]
            params = {"liveChatId": chat_id, "part": "id,snippet,authorDetails", "key": "k"}
            first = (await http.get("/liveChat/messages", params=params)).json()
            assert [i["id"] for i in first["items"]] == ["old-1"] and first[
                "pollingIntervalMillis"
            ] == 3000
            empty = (
                await http.get(
                    "/liveChat/messages", params={**params, "pageToken": first["nextPageToken"]}
                )
            ).json()
            assert empty["items"] == [] and empty["nextPageToken"] == first["nextPageToken"]
            srv.push(
                yt_text_message("ใหม่", msg_id="new-1", moderator=True), yt_super_chat("เย้", 250.0)
            )
            page = (
                await http.get(
                    "/liveChat/messages", params={**params, "pageToken": empty["nextPageToken"]}
                )
            ).json()
            assert page["items"][0]["id"] == "new-1" and len(page["items"]) == 2
            assert page["items"][0]["authorDetails"]["isChatModerator"] is True
            sc = page["items"][1]["snippet"]["superChatDetails"]
            assert sc["amountMicros"] == "250000000" and sc["currency"] == "THB"
            assert srv.quota_used == 4
            srv.forbid()
            denied = await http.get("/liveChat/messages", params=params)
            assert (
                denied.status_code == 403
                and denied.json()["error"]["errors"][0]["reason"] == "quotaExceeded"
            )
    finally:
        await srv.stop()


def test_page_fixture_shape(fixtures_dir: Path) -> None:
    page = json.loads(
        (fixtures_dir / "youtube" / "live_chat_page.json").read_text(encoding="utf-8")
    )
    types = [i["snippet"]["type"] for i in page["items"]]
    assert types == ["textMessageEvent"] * 3 + ["superChatEvent"]
    assert page["items"][3]["snippet"]["superChatDetails"]["amountMicros"] == "100000000"
