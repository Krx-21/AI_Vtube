"""
Event bus implementation for the Pailin AI VTuber application.

Provides an async-compatible publish-subscribe event system for decoupling
application components and enabling extensibility.
"""

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the Pailin AI VTuber application."""

    NEW_MESSAGE = "new_message"
    AI_RESPONSE = "ai_response"
    TTS_START = "tts_start"
    TTS_END = "tts_end"
    STT_START = "stt_start"
    STT_END = "stt_end"
    ERROR = "error"
    MOOD_CHANGE = "mood_change"
    APP_START = "app_start"
    APP_STOP = "app_stop"


class EventBus:
    """
    Thread-safe async-compatible event bus for pub/sub messaging.

    Supports both synchronous and asynchronous event handlers.
    Sync handlers are executed in a thread pool to avoid blocking.
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._wildcard_handlers: list[Callable] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    def on(self, event_type: EventType, handler: Callable) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The event type to listen for.
            handler: Callable to invoke when event is emitted. Can be sync or async.
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            handler_name = getattr(handler, "__name__", repr(handler))
            logger.debug("Registered handler %s for event %s", handler_name, event_type.value)

    def off(self, event_type: EventType, handler: Callable) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            handler_name = getattr(handler, "__name__", repr(handler))
            logger.debug("Unregistered handler %s for event %s", handler_name, event_type.value)

    def on_any(self, handler: Callable) -> None:
        """
        Subscribe a wildcard handler that receives all events.

        Args:
            handler: Callable to invoke for any event. Can be sync or async.
        """
        if handler not in self._wildcard_handlers:
            self._wildcard_handlers.append(handler)
            handler_name = getattr(handler, "__name__", repr(handler))
            logger.debug("Registered wildcard handler %s", handler_name)

    def clear(self) -> None:
        """Remove all handlers (both specific and wildcard)."""
        self._handlers.clear()
        self._wildcard_handlers.clear()
        logger.debug("Cleared all event handlers")

    async def emit(self, event_type: EventType, **data: Any) -> None:
        """
        Emit an event to all registered handlers.

        Async handlers are awaited, sync handlers are run in a thread pool.

        Args:
            event_type: The event type to emit.
            **data: Arbitrary keyword arguments passed to handlers.
        """
        logger.debug("Emitting event %s with data: %s", event_type.value, data)

        # Collect all handlers (specific + wildcard)
        handlers = self._handlers[event_type][:] + self._wildcard_handlers[:]

        # Execute all handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Async handler - await it
                    await handler(event_type, **data)
                else:
                    # Sync handler - run in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self._executor, lambda: handler(event_type, **data))
            except Exception as e:
                handler_name = getattr(handler, "__name__", repr(handler))
                logger.error(
                    "Error in handler %s for event %s: %s",
                    handler_name,
                    event_type.value,
                    e,
                    exc_info=True,
                )


# Default singleton instance
default_bus = EventBus()


# Convenience functions that delegate to the default bus
def on(event_type: EventType, handler: Callable) -> None:
    """Subscribe a handler to an event type on the default bus."""
    default_bus.on(event_type, handler)


def off(event_type: EventType, handler: Callable) -> None:
    """Unsubscribe a handler from an event type on the default bus."""
    default_bus.off(event_type, handler)


def on_any(handler: Callable) -> None:
    """Subscribe a wildcard handler to all events on the default bus."""
    default_bus.on_any(handler)


async def emit(event_type: EventType, **data: Any) -> None:
    """Emit an event on the default bus."""
    await default_bus.emit(event_type, **data)
