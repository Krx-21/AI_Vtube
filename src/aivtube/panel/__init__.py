"""Operator control panel, OBS overlays, hotkeys and alert ingest (ARCHITECTURE.md §2.10, §2.11).

- :class:`PanelServer`: token-protected aiohttp HTTP + WebSocket server on ``127.0.0.1:8770``
  serving the single-file operator SPA and the OBS overlays (imported lazily: aiohttp).
- :class:`CoreControl`: the core's ``ControlSurface``; routes operator commands to the brains,
  speech output, LLM router, tool registry, memory and safety gate, and audits every command.
- :class:`ToolApprovalQueue`: the ``approval`` callback of ``PolicyToolRegistry``; the operator
  answers pending tool calls with ``OpKind.APPROVE``.
- :mod:`aivtube.panel.commands`, :mod:`aivtube.panel.ingest`, :mod:`aivtube.panel.stats`:
  standard-library helpers shared with the text console.
"""

from typing import TYPE_CHECKING, Any

from aivtube.panel.commands import (
    FAST_KINDS,
    HOTKEYS,
    hotkey_command,
    hotkey_names,
    parse_op_command,
    validate_args,
)
from aivtube.panel.ingest import alert_message, chat_message
from aivtube.panel.stats import DEFAULT_BUDGETS, Budget, trace_summary

if TYPE_CHECKING:
    from aivtube.panel.approvals import ApprovalRequest, ToolApprovalQueue
    from aivtube.panel.control import BrainControl, CoreControl, OpAuditSink, OpHandler
    from aivtube.panel.server import ModerationActions, PanelOps, PanelServer

_LAZY = {
    "ApprovalRequest": "aivtube.panel.approvals",
    "ToolApprovalQueue": "aivtube.panel.approvals",
    "BrainControl": "aivtube.panel.control",
    "CoreControl": "aivtube.panel.control",
    "OpAuditSink": "aivtube.panel.control",
    "OpHandler": "aivtube.panel.control",
    "ModerationActions": "aivtube.panel.server",
    "PanelOps": "aivtube.panel.server",
    "PanelServer": "aivtube.panel.server",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "DEFAULT_BUDGETS",
    "FAST_KINDS",
    "HOTKEYS",
    "ApprovalRequest",
    "BrainControl",
    "Budget",
    "CoreControl",
    "ModerationActions",
    "OpAuditSink",
    "OpHandler",
    "PanelOps",
    "PanelServer",
    "ToolApprovalQueue",
    "alert_message",
    "chat_message",
    "hotkey_command",
    "hotkey_names",
    "parse_op_command",
    "trace_summary",
    "validate_args",
]
