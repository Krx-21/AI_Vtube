"""Avatar output (ARCHITECTURE.md §3.7): VTube Studio client and sink, UDP discovery, the 60 Hz
driver (lip tracks, emotion baseline, idle motion, state poses) and the expression state
machine. The core uses it only through the ``AvatarSink``/``AvatarDriver`` Protocols.
"""

from aivtube.avatar.discovery import (
    DISCOVERY_PORT,
    VTSDiscovery,
    discover_vts,
    parse_broadcast,
    pick_instance,
    resolve_vts_url,
)
from aivtube.avatar.driver import POSES, LiveAvatarDriver, TickerLike
from aivtube.avatar.emotion import EmotionController, EmotionSpec
from aivtube.avatar.factory import build_avatar_driver, build_avatar_sink
from aivtube.avatar.idle_motion import HEAD_PARAMS, IdleMotion
from aivtube.avatar.null_sink import NullSink
from aivtube.avatar.vts_client import (
    VTSAPIError,
    VTSClient,
    VTSDisconnected,
    VTSErrorID,
)
from aivtube.avatar.vts_sink import VTSSink

__all__ = [
    "DISCOVERY_PORT",
    "HEAD_PARAMS",
    "POSES",
    "EmotionController",
    "EmotionSpec",
    "IdleMotion",
    "LiveAvatarDriver",
    "NullSink",
    "TickerLike",
    "VTSAPIError",
    "VTSClient",
    "VTSDisconnected",
    "VTSDiscovery",
    "VTSErrorID",
    "VTSSink",
    "build_avatar_driver",
    "build_avatar_sink",
    "discover_vts",
    "parse_broadcast",
    "pick_instance",
    "resolve_vts_url",
]
