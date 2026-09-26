"""``YouTubeListPoller``: YouTube live chat via ``liveChatMessages.list`` (ARCHITECTURE.md §3.8).

Selected in M1 only if spike S9 shows YouTube is the primary platform; ``streamList`` over
gRPC comes in M3. Reading needs only an API key (research chat.md: the list endpoint's 403 for
anonymous callers asks for "API Key or other form of API consumer identity"; the streamList
guide states API keys work explicitly).

- **Discovery.** A fixed ``live_chat_id``; else ``videos.list?part=liveStreamingDetails`` on
  the configured ``video_id`` (1 quota unit) for ``activeLiveChatId``; else the unofficial,
  zero-quota ``youtube.com/@handle/live`` page gives the live video id, which is then always
  confirmed with ``videos.list`` (``search.list`` has its own 100 calls/day bucket; not used).
- **Polling.** ``GET /liveChat/messages?part=id,snippet,authorDetails`` (1 unit per call),
  sleeping ``max(pollingIntervalMillis / 1000, min_interval_s)`` between calls. The first
  page after discovery replays recent history and is skipped as backlog.
- **Mapping.** ``superChatEvent``/``superStickerEvent`` → DONATION (``amountMicros / 1e6`` +
  currency), ``giftEvent`` → DONATION in jewels, ``newSponsorEvent``/
  ``memberMilestoneChatEvent`` → SUB, ``membershipGiftingEvent`` → GIFT_SUB.
  ``isChatOwner``/``isChatModerator``/``isChatSponsor``/``isVerified`` map to user flags.
- **Quota.** ``quota_used`` counts one unit per API call. A 403 (quota, disabled API, bad key)
  marks the source DEGRADED with a Thai/English hint and backs off for ``quota_backoff_s``;
  past ``quota_soft_limit`` units the poll interval doubles.

Every HTTP await has a deadline on the injected ``Clock`` (I2). The API key travels as the
documented ``key`` query parameter and is never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import random
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aivtube.chat._util import sleep_unless_set
from aivtube.contracts.infra import Clock
from aivtube.contracts.types import ChatMessage, ChatUser, Health, HealthState, MsgKind, Platform
from aivtube.infra.clock import deadline
from aivtube.infra.tasks import backoff_delay

if TYPE_CHECKING:
    import httpx2

__all__ = [
    "API_ROOT",
    "USD_PER_UNIT",
    "WEB_ROOT",
    "YouTubeApiError",
    "YouTubeListPoller",
    "live_video_id_from_handle",
    "message_from_item",
    "resolve_live_chat_id",
    "video_id_from_live_page",
]

log = logging.getLogger("aivtube.chat.youtube_poll")

API_ROOT = "https://www.googleapis.com/youtube/v3"
WEB_ROOT = "https://www.youtube.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aivtube/0.1"

USD_PER_UNIT: Mapping[str, float] = {
    "USD": 1.0,
    "THB": 0.028,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "KRW": 0.00073,
    "TWD": 0.031,
    "HKD": 0.128,
    "SGD": 0.74,
    "MYR": 0.22,
    "PHP": 0.018,
    "IDR": 0.000062,
    "VND": 0.00004,
    "INR": 0.012,
    "AUD": 0.66,
    "CAD": 0.73,
    "JEWEL": 0.01,  # YouTube jewels: unverified nominal value
}
"""Approximate exchange rates, used only to rank must-acknowledge messages."""

_VIDEO_ID = re.compile(r"^[\w-]{11}$")
_INITIAL_DATA = ("var ytInitialData = ", 'window["ytInitialData"] = ')
_ENDPOINT_RE = re.compile(
    r'"currentVideoEndpoint":\{.{0,800}?"watchEndpoint":\{"videoId":"([\w-]{11})"'
)
_QUOTA_REASONS = frozenset({"quotaExceeded", "dailyLimitExceeded"})
_RATE_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})
_GONE_REASONS = frozenset({"liveChatEnded", "liveChatNotFound", "liveChatDisabled"})


class YouTubeApiError(Exception):
    """A non-2xx answer from the Data API: HTTP ``status``, Google ``reason`` and message."""

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(f"HTTP {status} {reason or '-'}: {message}")
        self.status = status
        self.reason = reason
        self.message = message


def _api_error(resp: httpx2.Response) -> YouTubeApiError:
    reason, message = "", resp.reason_phrase or ""
    with contextlib.suppress(ValueError, AttributeError, TypeError):
        err = resp.json().get("error") or {}
        message = str(err.get("message") or message)
        errors = err.get("errors") or []
        reason = str((errors[0] or {}).get("reason") or err.get("status") or "") if errors else ""
        reason = reason or str(err.get("status") or "")
    return YouTubeApiError(resp.status_code, reason, message)


async def _get_json(
    http: Any,
    url: str,
    params: Mapping[str, Any],
    *,
    what: str,
    timeout_s: float,
    clock: Clock | None,
) -> dict[str, Any]:
    async with deadline(timeout_s, what=what, clock=clock):
        resp = await http.get(url, params=params)
    if resp.status_code >= 400:
        raise _api_error(resp)
    content = resp.content
    data = await asyncio.to_thread(json.loads, content) if len(content) > 200_000 else resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"{what}: expected a JSON object")
    return data


async def resolve_live_chat_id(
    http: Any,
    api_key: str,
    video_id: str,
    *,
    api_root: str = API_ROOT,
    timeout_s: float = 10.0,
    clock: Clock | None = None,
) -> str | None:
    """``videos.list?part=liveStreamingDetails`` → ``activeLiveChatId`` (1 quota unit).

    ``None`` when the video is unknown or not live; raises ``YouTubeApiError`` on HTTP errors.
    """
    data = await _get_json(
        http,
        f"{api_root}/videos",
        {"part": "liveStreamingDetails", "id": video_id, "key": api_key},
        what="youtube videos.list",
        timeout_s=timeout_s,
        clock=clock,
    )
    for item in data.get("items") or []:
        chat_id = (item.get("liveStreamingDetails") or {}).get("activeLiveChatId")
        if chat_id:
            return str(chat_id)
    return None


def _handle_path(handle: str) -> str:
    h = handle.strip().rstrip("/")
    if h.startswith(("http://", "https://")):
        path = h.split("youtube.com", 1)[-1] if "youtube.com" in h else h
        return path.removesuffix("/live").lstrip("/")
    if h.startswith("UC") and len(h) == 24:
        return f"channel/{h}"
    return "@" + h.lstrip("@")


def video_id_from_live_page(html: str) -> str | None:
    """The live video id in a ``/@handle/live`` page, or ``None`` if the channel is not live.

    Reads ``ytInitialData.currentVideoEndpoint.watchEndpoint.videoId`` (verified on real pages
    2026-09-25), only when ``"isLive":true`` is present. CPU-bound on ~1 MB of HTML: call it
    in a worker thread.
    """
    if '"isLive":true' not in html:
        return None
    for marker in _INITIAL_DATA:
        start = html.find(marker)
        if start < 0:
            continue
        with contextlib.suppress(ValueError):
            data, _ = json.JSONDecoder().raw_decode(html, start + len(marker))
            vid = (
                ((data.get("currentVideoEndpoint") or {}).get("watchEndpoint") or {}).get("videoId")
                if isinstance(data, dict)
                else None
            )
            if isinstance(vid, str) and _VIDEO_ID.match(vid):
                return vid
    match = _ENDPOINT_RE.search(html)
    return match.group(1) if match else None


async def live_video_id_from_handle(
    http: Any,
    handle: str,
    *,
    web_root: str = WEB_ROOT,
    timeout_s: float = 15.0,
    clock: Clock | None = None,
) -> str | None:
    """Unofficial, zero-quota: the live video id of a channel from ``/@handle/live``.

    Fragile (datacenter IPs get bot checks, several concurrent lives return varying ids), so
    always confirm the result with :func:`resolve_live_chat_id`.
    """
    url = f"{web_root}/{_handle_path(handle)}/live"
    async with deadline(timeout_s, what="youtube live page", clock=clock):
        resp = await http.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            follow_redirects=True,
        )
    if resp.status_code != 200:
        return None
    return await asyncio.to_thread(lambda: video_id_from_live_page(resp.text))


def _epoch(published: Any) -> float | None:
    if not isinstance(published, str) or not published:
        return None
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _map_ts(sent: float | None, received: float, wall_now: float | None) -> float:
    if sent is None or wall_now is None:
        return received
    delay = wall_now - sent
    return received - delay if 0.0 <= delay <= 30.0 else received


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _micros(details: Mapping[str, Any]) -> float:
    return _as_int(details.get("amountMicros")) / 1e6


def message_from_item(
    item: Mapping[str, Any],
    *,
    received: float,
    wall_now: float | None = None,
    fx: Mapping[str, float] = USD_PER_UNIT,
) -> ChatMessage | None:
    """Map a ``liveChatMessage`` resource to a ``ChatMessage`` (``None`` for types that are
    not chat or support, such as tombstones, polls, bans and ``chatEndedEvent``).

    A ``giftEvent`` maps to its whole combo (``jewelsAmount * comboCount``); the poller turns
    repeated combo updates into increments.
    """
    sn = item.get("snippet") or {}
    au = item.get("authorDetails") or {}
    typ = sn.get("type", "")
    months = _as_int((sn.get("memberMilestoneChatDetails") or {}).get("memberMonth"))
    user = ChatUser(
        platform=Platform.YOUTUBE,
        id=str(au.get("channelId") or sn.get("authorChannelId") or ""),
        name=str(au.get("displayName") or ""),
        is_broadcaster=bool(au.get("isChatOwner")),
        is_mod=bool(au.get("isChatModerator")),
        is_sub=bool(au.get("isChatSponsor"))
        or typ in ("newSponsorEvent", "memberMilestoneChatEvent"),
        sub_months=months,
        is_verified=bool(au.get("isVerified")),
    )
    ts = _map_ts(_epoch(sn.get("publishedAt")), received, wall_now)
    base: dict[str, Any] = {
        "platform": Platform.YOUTUBE,
        "id": str(item.get("id") or ""),
        "user": user,
        "ts": ts,
        "received": received,
        "raw": item,
    }
    if not base["id"]:
        return None
    if typ == "textMessageEvent":
        text = (sn.get("textMessageDetails") or {}).get("messageText") or sn.get("displayMessage")
        return ChatMessage(text=str(text or ""), **base)
    if typ in ("superChatEvent", "superStickerEvent"):
        details = sn.get("superChatDetails") or sn.get("superStickerDetails") or {}
        amount = _micros(details)
        currency = str(details.get("currency") or "")
        return ChatMessage(
            text=str(details.get("userComment") or ""),
            kind=MsgKind.DONATION,
            amount=amount,
            currency=currency,
            value_usd=amount * fx.get(currency, 0.0),
            **base,
        )
    if typ == "giftEvent":
        gift = sn.get("giftDetails") or {}
        jewels = float(
            _as_int(gift.get("jewelsAmount")) * max(1, _as_int(gift.get("comboCount"), 1))
        )
        return ChatMessage(
            text="",
            kind=MsgKind.DONATION,
            amount=jewels,
            currency="JEWEL",
            value_usd=jewels * fx.get("JEWEL", 0.0),
            **base,
        )
    if typ == "newSponsorEvent":
        return ChatMessage(text="", kind=MsgKind.SUB, amount=1.0, **base)
    if typ == "memberMilestoneChatEvent":
        comment = (sn.get("memberMilestoneChatDetails") or {}).get("userComment") or ""
        return ChatMessage(text=str(comment), kind=MsgKind.SUB, amount=1.0, **base)
    if typ == "membershipGiftingEvent":
        count = _as_int((sn.get("membershipGiftingDetails") or {}).get("giftMembershipsCount"), 1)
        return ChatMessage(text="", kind=MsgKind.GIFT_SUB, amount=float(count), **base)
    return None


@dataclass(slots=True)
class _Page:
    messages: list[ChatMessage]
    interval: float


class _NotLive(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class YouTubeListPoller:
    """``ChatSource`` for one YouTube live chat, polled with ``liveChatMessages.list``.

    Give one of ``live_chat_id``, ``video_id`` or ``handle``. ``http`` is an
    ``httpx2.AsyncClient`` (tests pass one with ``httpx2.MockTransport``); without it the
    poller owns a client and closes it in ``aclose()``. Instrumentation: ``quota_used``,
    ``polls``, ``duplicates``, ``skipped_backlog``, ``last_interval``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        video_id: str | None = None,
        handle: str | None = None,
        clock: Clock,
        http: Any | None = None,
        min_interval_s: float = 2.0,
        live_chat_id: str | None = None,
        api_root: str = API_ROOT,
        web_root: str = WEB_ROOT,
        max_results: int = 2000,
        request_timeout_s: float = 10.0,
        backoff: tuple[float, float] = (2.0, 60.0),
        not_live_retry_s: float = 60.0,
        quota_backoff_s: float = 900.0,
        daily_quota: int = 10_000,
        quota_soft_limit: int | None = None,
        default_interval_s: float = 5.0,
        fx: Mapping[str, float] = USD_PER_UNIT,
        dedupe_lru: int = 5000,
        rng: random.Random | None = None,
    ) -> None:
        if not (live_chat_id or video_id or handle):
            raise ValueError("YouTubeListPoller needs live_chat_id, video_id or handle")
        if not api_key:
            raise ValueError("a YouTube Data API key is required")
        self.platform: Platform = Platform.YOUTUBE
        self._api_key = api_key
        self.video_id = video_id or None
        self.handle = handle or None
        self._fixed_chat_id = live_chat_id or None
        self._clock = clock
        self._http = http
        self._owns_http = http is None
        self.min_interval_s = min_interval_s
        self.api_root = api_root.rstrip("/")
        self.web_root = web_root.rstrip("/")
        self.max_results = max_results
        self.request_timeout_s = request_timeout_s
        self.backoff = backoff
        self.not_live_retry_s = not_live_retry_s
        self.quota_backoff_s = quota_backoff_s
        self.daily_quota = daily_quota
        self.quota_soft_limit = (
            int(daily_quota * 0.8) if quota_soft_limit is None else quota_soft_limit
        )
        self.default_interval_s = default_interval_s
        self._fx = fx
        self._rng = rng or random.Random()
        self._dedupe_lru = dedupe_lru
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._gift_combo: OrderedDict[str, int] = OrderedDict()
        self.live_chat_id: str | None = None
        self._page_token: str | None = None
        self._closed = False
        self._closed_event = asyncio.Event()
        self._state = HealthState.STARTING
        self._detail = "resolving the live chat"
        self._since = clock.now()
        self.quota_used = 0
        self.polls = 0
        self.duplicates = 0
        self.skipped_backlog = 0
        self.last_interval: float | None = None

    @classmethod
    def from_config(
        cls, cfg: Any, *, api_key: str, clock: Clock, **kwargs: Any
    ) -> YouTubeListPoller:
        """Build from ``config.schema.YouTubePollConfig`` (duck-typed ``video_id``, ``handle``
        and ``min_interval_s``); the key comes from the secrets (``api_key_env``)."""
        return cls(
            api_key,
            video_id=cfg.video_id or None,
            handle=cfg.handle or None,
            clock=clock,
            min_interval_s=float(cfg.min_interval_s),
            **kwargs,
        )

    # --- ChatSource ------------------------------------------------------------------------
    def health(self) -> Health:
        component = f"chat:{self.platform.value}"
        if self._closed:
            return Health(component, HealthState.DOWN, "closed", self._since)
        return Health(component, self._state, self._detail, self._since)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_event.set()
        self._set_state(HealthState.DOWN, "closed")
        if self._owns_http and self._http is not None:
            http, self._http = self._http, None
            with contextlib.suppress(Exception):
                async with deadline(2.0, what="youtube http close", clock=self._clock):
                    await http.aclose()

    async def messages(self) -> AsyncIterator[ChatMessage]:
        import httpx2

        failures = 0
        while not self._closed:
            try:
                page = await self._poll_once()
            except _NotLive as exc:
                self._set_state(HealthState.DEGRADED, exc.detail)
                delay = self.not_live_retry_s
            except YouTubeApiError as exc:
                failures += 1
                delay = self._on_api_error(exc, failures)
            except (httpx2.HTTPError, OSError, TimeoutError, ValueError) as exc:
                failures += 1
                delay = self._backoff(failures)
                reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                reason = reason.replace(self._api_key, "***")
                self._set_state(
                    HealthState.DEGRADED,
                    f"YouTube unreachable ({reason}); retry in {delay:.0f} s · "
                    "ต่อ YouTube ไม่ได้ กำลังลองใหม่",
                )
                log.warning("youtube poll failed (%s); retry in %.1f s", reason, delay)
            else:
                failures = 0
                for msg in page.messages:
                    if self._closed:
                        return
                    yield msg
                delay = page.interval
            if not await self._sleep(delay):
                break

    # --- polling ---------------------------------------------------------------------------
    def _client(self) -> Any:
        if self._http is None:
            import httpx2

            self._http = httpx2.AsyncClient(
                timeout=self.request_timeout_s, headers={"User-Agent": USER_AGENT}
            )
        return self._http

    async def _resolve(self) -> str:
        if self._fixed_chat_id:
            return self._fixed_chat_id
        http = self._client()
        video_id = self.video_id
        if video_id is None and self.handle:
            video_id = await live_video_id_from_handle(
                http,
                self.handle,
                web_root=self.web_root,
                timeout_s=self.request_timeout_s,
                clock=self._clock,
            )
        if video_id is None:
            raise _NotLive(f"{self.handle} is not live · ช่องยังไม่ได้ไลฟ์")
        self.quota_used += 1
        chat_id = await resolve_live_chat_id(
            http,
            self._api_key,
            video_id,
            api_root=self.api_root,
            timeout_s=self.request_timeout_s,
            clock=self._clock,
        )
        if chat_id is None:
            raise _NotLive(f"video {video_id} has no active live chat · วิดีโอนี้ไม่มีแชทสด")
        return chat_id

    async def _poll_once(self) -> _Page:
        if self.live_chat_id is None:
            chat_id = await self._resolve()
            if chat_id != self.live_chat_id:
                self._page_token = None
            self.live_chat_id = chat_id
        params: dict[str, Any] = {
            "liveChatId": self.live_chat_id,
            "part": "id,snippet,authorDetails",
            "maxResults": self.max_results,
            "key": self._api_key,
        }
        backlog = self._page_token is None
        if not backlog:
            params["pageToken"] = self._page_token
        self.quota_used += 1
        self.polls += 1
        data = await _get_json(
            self._client(),
            f"{self.api_root}/liveChat/messages",
            params,
            what="youtube liveChatMessages.list",
            timeout_s=self.request_timeout_s,
            clock=self._clock,
        )
        token = data.get("nextPageToken")
        if isinstance(token, str) and token:
            self._page_token = token
        received, wall = self._clock.now(), self._clock.wall()
        out: list[ChatMessage] = []
        ended = bool(data.get("offlineAt"))
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            if (item.get("snippet") or {}).get("type") == "chatEndedEvent":
                ended = True
            msg = self._accept(item, received, wall)
            if msg is None:
                continue
            if backlog:
                self.skipped_backlog += 1
            else:
                out.append(msg)
        if ended:
            self.live_chat_id = None
            self._page_token = None
            self._set_state(HealthState.DEGRADED, "the live chat has ended · ไลฟ์จบแล้ว")
            return _Page(out, self.not_live_retry_s)
        interval = self._interval(data.get("pollingIntervalMillis"))
        if self.quota_used >= self.quota_soft_limit:
            self._set_state(
                HealthState.DEGRADED,
                f"quota low ({self.quota_used}/{self.daily_quota} units); polling every "
                f"{interval:.0f} s · โควตา YouTube ใกล้หมด",
            )
        else:
            self._set_state(HealthState.OK, f"polling {self.live_chat_id}")
        return _Page(out, interval)

    def _interval(self, polling_ms: Any) -> float:
        try:
            server = float(polling_ms) / 1000.0
        except (TypeError, ValueError):
            server = self.default_interval_s
        interval = max(server, self.min_interval_s)
        if self.quota_used >= self.quota_soft_limit:
            interval *= 2.0
        self.last_interval = interval
        return interval

    def _accept(self, item: Mapping[str, Any], received: float, wall: float) -> ChatMessage | None:
        msg = message_from_item(item, received=received, wall_now=wall, fx=self._fx)
        if msg is None:
            return None
        gift = (item.get("snippet") or {}).get("giftDetails")
        if (item.get("snippet") or {}).get("type") == "giftEvent" and isinstance(gift, dict):
            return self._gift_increment(msg, max(1, _as_int(gift.get("comboCount"), 1)))
        if msg.id in self._seen:
            self.duplicates += 1
            return None
        self._remember(self._seen, msg.id, None)
        return msg

    def _gift_increment(self, msg: ChatMessage, combo: int) -> ChatMessage | None:
        """YouTube reuses a giftEvent's id to update its ``comboCount``: dedupe on
        ``(id, comboCount)`` and yield only the jewels added since the last update."""
        prev = self._gift_combo.get(msg.id, 0)
        if combo <= prev:
            self.duplicates += 1
            return None
        self._remember(self._gift_combo, msg.id, combo)
        if prev == 0:
            return msg
        share = (combo - prev) / combo
        return dataclasses.replace(
            msg,
            id=f"{msg.id}#{combo}",
            amount=msg.amount * share,
            value_usd=msg.value_usd * share,
        )

    def _remember(self, table: OrderedDict[str, Any], key: str, value: Any) -> None:
        table[key] = value
        table.move_to_end(key)
        while len(table) > self._dedupe_lru:
            table.popitem(last=False)

    # --- errors, health, sleeping ----------------------------------------------------------
    def _backoff(self, failures: int) -> float:
        delay = backoff_delay(failures, self.backoff)
        return delay * (0.8 + 0.4 * self._rng.random())

    def _on_api_error(self, exc: YouTubeApiError, failures: int) -> float:
        reason, status = exc.reason, exc.status
        if reason in _QUOTA_REASONS:
            delay = self.quota_backoff_s
            hint = (
                "daily API quota exhausted (resets at midnight Pacific time) · "
                "โควตา YouTube API ของวันนี้หมดแล้ว"
            )
        elif reason in _RATE_REASONS or status == 429:
            delay = self._backoff(failures)
            hint = "polling too fast (rate limited) · YouTube จำกัดความถี่ กำลังชะลอ"
        elif reason in _GONE_REASONS or status == 404:
            self.live_chat_id = None
            self._page_token = None
            delay = self.not_live_retry_s
            hint = "the live chat is gone or disabled · แชทสดปิดหรือจบแล้ว"
        elif status in (400, 401, 403):
            delay = self.quota_backoff_s
            hint = (
                "API key rejected: check YOUTUBE_API_KEY and enable YouTube Data API v3 · "
                "คีย์ YouTube API ใช้ไม่ได้ ตรวจ YOUTUBE_API_KEY และเปิด YouTube Data API v3"
            )
        else:
            delay = self._backoff(failures)
            hint = "YouTube API error · YouTube API ขัดข้อง"
        self._set_state(
            HealthState.DEGRADED, f"{hint} (HTTP {status} {reason or '-'}; retry in {delay:.0f} s)"
        )
        log.warning("youtube api error %s %s; retry in %.1f s", status, reason or "-", delay)
        return delay

    def _set_state(self, state: HealthState, detail: str) -> None:
        if self._closed and state is not HealthState.DOWN:
            return
        if state is not self._state:
            self._since = self._clock.now()
        self._state = state
        self._detail = detail

    async def _sleep(self, seconds: float) -> bool:
        """Sleep on the clock unless ``aclose()`` happens first; False once closed."""
        if self._closed:
            return False
        return await sleep_unless_set(self._clock, seconds, self._closed_event)
