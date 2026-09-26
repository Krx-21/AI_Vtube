"""Runtime infrastructure for the core and workers (ARCHITECTURE.md §2.4, §2.5, §2.10).

Event bus, task supervisor, clock and deadlines, precision ticker, logging, turn traces,
flight recorder, metrics, plugin registry and the loop-lag monitor. Standard library only.
Everything that measures time accepts a ``Clock`` so tests can drive it with a fake clock
(``PrecisionTicker`` and the lag watchdog are real threads by design).
"""

from aivtube.infra.bus import AsyncEventBus, BusSubscription
from aivtube.infra.clock import DeadlineExceeded, SystemClock, deadline
from aivtube.infra.flight import FlightRecorder, load_dump
from aivtube.infra.lag import LoopLagMonitor
from aivtube.infra.logging import (
    JsonFormatter,
    RedactionFilter,
    Redactor,
    add_secrets,
    current_redactor,
    setup_logging,
    shutdown_logging,
)
from aivtube.infra.metrics import MetricsRegistry
from aivtube.infra.registry import PluginRegistry
from aivtube.infra.tasks import SupervisedTasks, backoff_delay
from aivtube.infra.ticker import PrecisionTicker
from aivtube.infra.trace import STAGES, TurnTraceRecorder

__all__ = [
    "STAGES",
    "AsyncEventBus",
    "BusSubscription",
    "DeadlineExceeded",
    "FlightRecorder",
    "JsonFormatter",
    "LoopLagMonitor",
    "MetricsRegistry",
    "PluginRegistry",
    "PrecisionTicker",
    "RedactionFilter",
    "Redactor",
    "SupervisedTasks",
    "SystemClock",
    "TurnTraceRecorder",
    "add_secrets",
    "backoff_delay",
    "current_redactor",
    "deadline",
    "load_dump",
    "setup_logging",
    "shutdown_logging",
]
