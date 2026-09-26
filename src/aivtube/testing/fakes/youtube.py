"""``FakeYouTubeServer``: the YouTube Data API v3 live-chat endpoints over local HTTP (§3.8, §10).

Serves ``GET /youtube/v3/videos?part=liveStreamingDetails`` (the ``activeLiveChatId``) and
``GET /youtube/v3/liveChat/messages`` with ``pollingIntervalMillis``, ``nextPageToken`` and a
backlog on the first page, as ``liveChatMessages.list`` does. Every call costs one quota unit;
``forbid()`` switches to 403 ``quotaExceeded`` answers. Build items with ``yt_text_message`` and
``yt_super_chat``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from aiohttp import web

from aivtube.testing.fakes._http import HttpFixture

__all__ = ["FakeYouTubeServer", "yt_super_chat", "yt_text_message"]

_ids = itertools.count(1)


def _author(name: str, channel_id: str, **flags: bool) -> dict[str, Any]:
    return {
        "channelId": channel_id,
        "channelUrl": f"http://www.youtube.com/channel/{channel_id}",
        "displayName": name,
        "profileImageUrl": "https://yt3.ggpht.example/avatar.jpg",
        "isVerified": flags.get("verified", False),
        "isChatOwner": flags.get("owner", False),
        "isChatSponsor": flags.get("sponsor", False),
        "isChatModerator": flags.get("moderator", False),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def yt_text_message(
    text: str,
    *,
    name: str = "ผู้ชม",
    channel_id: str | None = None,
    msg_id: str | None = None,
    live_chat_id: str = "live-chat-1",
    published_at: str | None = None,
    moderator: bool = False,
    sponsor: bool = False,
    owner: bool = False,
    verified: bool = False,
) -> dict[str, Any]:
    """A ``textMessageEvent`` ``liveChatMessage`` resource."""
    n = next(_ids)
    cid = channel_id or f"UC{n:022d}"
    return {
        "kind": "youtube#liveChatMessage",
        "etag": f"etag-{n}",
        "id": msg_id or f"LCC.fake-{n}",
        "snippet": {
            "type": "textMessageEvent",
            "liveChatId": live_chat_id,
            "authorChannelId": cid,
            "publishedAt": published_at or _now(),
            "hasDisplayContent": True,
            "displayMessage": text,
            "textMessageDetails": {"messageText": text},
        },
        "authorDetails": _author(
            name, cid, moderator=moderator, sponsor=sponsor, owner=owner, verified=verified
        ),
    }


def yt_super_chat(
    comment: str,
    amount: float,
    currency: str = "THB",
    *,
    name: str = "ผู้สนับสนุน",
    channel_id: str | None = None,
    msg_id: str | None = None,
    live_chat_id: str = "live-chat-1",
    tier: int = 2,
) -> dict[str, Any]:
    """A ``superChatEvent`` resource; ``amountMicros`` is ``amount * 1e6`` as a string."""
    n = next(_ids)
    cid = channel_id or f"UC{n:022d}"
    display = f"{amount:,.2f} {currency}"
    return {
        "kind": "youtube#liveChatMessage",
        "etag": f"etag-{n}",
        "id": msg_id or f"LCC.fake-{n}",
        "snippet": {
            "type": "superChatEvent",
            "liveChatId": live_chat_id,
            "authorChannelId": cid,
            "publishedAt": _now(),
            "hasDisplayContent": True,
            "displayMessage": f'{display} from {name}: "{comment}"',
            "superChatDetails": {
                "amountMicros": str(round(amount * 1_000_000)),
                "currency": currency,
                "amountDisplayString": display,
                "userComment": comment,
                "tier": tier,
            },
        },
        "authorDetails": _author(name, cid),
    }


class FakeYouTubeServer(HttpFixture):
    """``start()`` returns the API root (``http://127.0.0.1:<port>/youtube/v3``).

    ``backlog`` is returned on the first page only; ``push(item)`` queues new items for the
    next poll. Instrumentation: ``calls`` (path, query), ``quota_used``.
    """

    def __init__(
        self,
        backlog: Sequence[dict[str, Any]] = (),
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        video_id: str = "dQw4w9WgXcQ",
        live_chat_id: str = "live-chat-1",
        polling_interval_ms: int = 5000,
        api_key: str | None = None,
    ) -> None:
        super().__init__(host, port)
        self.video_id = video_id
        self.live_chat_id = live_chat_id
        self.polling_interval_ms = polling_interval_ms
        self.api_key = api_key
        self._pages: list[list[dict[str, Any]]] = [list(backlog)]
        self._pending: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.quota_used = 0
        self.forbidden = False
        self.offline_at: str | None = None

    @property
    def api_root(self) -> str:
        return f"{self.root_url}/youtube/v3"

    async def start(self) -> str:
        await self._start_http()
        return self.api_root

    async def stop(self) -> None:
        await self._stop_http()

    def push(self, *items: dict[str, Any]) -> None:
        self._pending.extend(items)

    def forbid(self, on: bool = True) -> None:
        self.forbidden = on

    def end_stream(self) -> None:
        self.offline_at = _now()

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/youtube/v3/videos", self._videos)
        app.router.add_get("/youtube/v3/liveChat/messages", self._messages)
        return app

    def _gate(self, request: web.Request) -> web.Response | None:
        self.calls.append((request.path, dict(request.query)))
        self.quota_used += 1
        if self.api_key is not None and request.query.get("key") != self.api_key:
            return self._error(400, "keyInvalid", "API key not valid. Please pass a valid API key.")
        if self.forbidden:
            return self._error(
                403,
                "quotaExceeded",
                "The request cannot be completed because you have exceeded your quota.",
            )
        return None

    @staticmethod
    def _error(status: int, reason: str, message: str) -> web.Response:
        body = {
            "error": {
                "code": status,
                "message": message,
                "errors": [{"message": message, "domain": "youtube.quota", "reason": reason}],
            }
        }
        return web.json_response(body, status=status)

    async def _videos(self, request: web.Request) -> web.Response:
        denied = self._gate(request)
        if denied is not None:
            return denied
        items = []
        if request.query.get("id") == self.video_id:
            items.append(
                {
                    "kind": "youtube#video",
                    "id": self.video_id,
                    "liveStreamingDetails": {
                        "actualStartTime": "2026-09-25T12:00:00Z",
                        "concurrentViewers": "42",
                        "activeLiveChatId": self.live_chat_id,
                    },
                }
            )
        return web.json_response({"kind": "youtube#videoListResponse", "items": items})

    async def _messages(self, request: web.Request) -> web.Response:
        denied = self._gate(request)
        if denied is not None:
            return denied
        if request.query.get("liveChatId") != self.live_chat_id:
            return self._error(
                404,
                "liveChatNotFound",
                "The live chat that you are trying to retrieve cannot be found.",
            )
        token = request.query.get("pageToken", "")
        if self._pending:
            self._pages.append(self._pending)
            self._pending = []
        index = int(token.removeprefix("page-")) if token.startswith("page-") else 0
        if index < len(self._pages):
            items, next_index = self._pages[index], index + 1
        else:
            items, next_index = [], index
        body: dict[str, Any] = {
            "kind": "youtube#liveChatMessageListResponse",
            "pollingIntervalMillis": self.polling_interval_ms,
            "nextPageToken": f"page-{next_index}",
            "pageInfo": {"totalResults": len(items), "resultsPerPage": len(items)},
            "items": items,
        }
        if self.offline_at is not None:
            body["offlineAt"] = self.offline_at
        return web.json_response(body)
