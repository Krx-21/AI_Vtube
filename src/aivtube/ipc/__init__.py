"""Core <-> voice-worker IPC over a loopback WebSocket (ARCHITECTURE.md Appendix A, §2.1).

- ``IpcServer`` (core): ``ws://127.0.0.1:<ports.bus>/bus``; token and major-version check on
  ``hello``, a 5-ping clock check, 1 Hz heartbeats, one ``IpcPeer`` per role.
- ``IpcClient`` (voice worker): connects, says ``hello`` and reconnects with backoff;
  ``on_link_up`` / ``on_link_lost`` drive invariant I1.
- ``Link``: the shared per-connection machinery (validation of every message at both ends,
  in-order dispatch, ``corr``-correlated requests, heartbeat death after 3 misses).

The wire format, message types and JSON schemas live in ``aivtube.contracts.ipc``.
"""

from aivtube.ipc.client import HandshakeRejected, IpcClient
from aivtube.ipc.link import (
    CLOSE_BAD_TOKEN,
    CLOSE_DEAD,
    CLOSE_PROTOCOL,
    CLOSE_REPLACED,
    CLOSE_SHUTDOWN,
    CLOSE_VERSION,
    Handler,
    IpcLinkError,
    Link,
)
from aivtube.ipc.server import IpcPeer, IpcServer, is_loopback_host, major_version

__all__ = [
    "CLOSE_BAD_TOKEN",
    "CLOSE_DEAD",
    "CLOSE_PROTOCOL",
    "CLOSE_REPLACED",
    "CLOSE_SHUTDOWN",
    "CLOSE_VERSION",
    "Handler",
    "HandshakeRejected",
    "IpcClient",
    "IpcLinkError",
    "IpcPeer",
    "IpcServer",
    "Link",
    "is_loopback_host",
    "major_version",
]
