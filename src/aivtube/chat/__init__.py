"""Chat intake: platform sources and the scored chat window (ARCHITECTURE.md §3.8, §4.2, §4.3).

M1 ships :class:`TwitchAnonIrc` (anonymous read-only Twitch IRC), :class:`YouTubeListPoller`
(``liveChatMessages.list`` with an API key) and :class:`ScoredChatWindow`. EventSub/Helix,
YouTube streamList, alert services and TikTok follow in M3/M6.
"""

from aivtube.chat._text import AliasMatcher, dedupe_key, is_question, normalize
from aivtube.chat.twitch_irc import TwitchAnonIrc, parse_irc, parse_irc_line
from aivtube.chat.window import SUPPORT_KINDS, WEIGHTS, ScoredChatWindow, WindowConfig
from aivtube.chat.youtube_poll import (
    YouTubeApiError,
    YouTubeListPoller,
    live_video_id_from_handle,
    message_from_item,
    resolve_live_chat_id,
)

__all__ = [
    "SUPPORT_KINDS",
    "WEIGHTS",
    "AliasMatcher",
    "ScoredChatWindow",
    "TwitchAnonIrc",
    "WindowConfig",
    "YouTubeApiError",
    "YouTubeListPoller",
    "dedupe_key",
    "is_question",
    "live_video_id_from_handle",
    "message_from_item",
    "normalize",
    "parse_irc",
    "parse_irc_line",
    "resolve_live_chat_id",
]
