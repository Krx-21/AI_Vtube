"""
Event system for the Pailin AI VTuber application.

Provides a modular pub/sub event bus for decoupling application components.
"""

from pailin_core.events.event_bus import (
    EventBus,
    EventType,
    default_bus,
    emit,
    off,
    on,
    on_any,
)

__all__ = [
    "EventBus",
    "EventType",
    "default_bus",
    "on",
    "off",
    "emit",
    "on_any",
]
