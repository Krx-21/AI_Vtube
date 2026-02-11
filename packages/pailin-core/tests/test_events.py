"""Tests for the event system."""

import asyncio
from unittest.mock import MagicMock

import pytest
from pailin_core.events import EventBus, EventType, default_bus, emit, off, on, on_any


class TestEventBus:
    """Tests for the EventBus class."""

    def setup_method(self) -> None:
        """Clear the default bus before each test."""
        default_bus.clear()

    def test_basic_subscribe_and_emit(self) -> None:
        """Test basic subscription and emission."""
        bus = EventBus()
        handler = MagicMock()

        bus.on(EventType.NEW_MESSAGE, handler)

        asyncio.run(bus.emit(EventType.NEW_MESSAGE, text="Hello"))

        handler.assert_called_once_with(EventType.NEW_MESSAGE, text="Hello")

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        """Test that async handlers are properly awaited."""
        bus = EventBus()
        called = []

        async def async_handler(event_type: EventType, **data: object) -> None:
            await asyncio.sleep(0.01)  # Simulate async work
            called.append((event_type, data))

        bus.on(EventType.AI_RESPONSE, async_handler)
        await bus.emit(EventType.AI_RESPONSE, response="Test response")

        assert len(called) == 1
        assert called[0][0] == EventType.AI_RESPONSE
        assert called[0][1] == {"response": "Test response"}

    @pytest.mark.asyncio
    async def test_sync_handler_in_executor(self) -> None:
        """Test that sync handlers are executed in thread pool."""
        bus = EventBus()
        called = []

        def sync_handler(event_type: EventType, **data: object) -> None:
            called.append((event_type, data))

        bus.on(EventType.TTS_START, sync_handler)
        await bus.emit(EventType.TTS_START, text="Hello")

        # Give executor time to complete
        await asyncio.sleep(0.1)

        assert len(called) == 1
        assert called[0][0] == EventType.TTS_START
        assert called[0][1] == {"text": "Hello"}

    @pytest.mark.asyncio
    async def test_wildcard_listener(self) -> None:
        """Test wildcard listener receives all events."""
        bus = EventBus()
        events_received = []

        async def wildcard_handler(event_type: EventType, **data: object) -> None:
            events_received.append(event_type)

        bus.on_any(wildcard_handler)

        await bus.emit(EventType.NEW_MESSAGE, text="test")
        await bus.emit(EventType.AI_RESPONSE, response="test")
        await bus.emit(EventType.TTS_START, text="test")

        assert len(events_received) == 3
        assert EventType.NEW_MESSAGE in events_received
        assert EventType.AI_RESPONSE in events_received
        assert EventType.TTS_START in events_received

    @pytest.mark.asyncio
    async def test_off_unsubscribe(self) -> None:
        """Test unsubscribing a handler."""
        bus = EventBus()
        handler = MagicMock()

        bus.on(EventType.ERROR, handler)
        await bus.emit(EventType.ERROR, message="error 1")

        bus.off(EventType.ERROR, handler)
        await bus.emit(EventType.ERROR, message="error 2")

        # Give executor time to complete
        await asyncio.sleep(0.1)

        # Handler should only be called once (before unsubscribe)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_removes_all_handlers(self) -> None:
        """Test that clear() removes all handlers."""
        bus = EventBus()
        handler1 = MagicMock()
        handler2 = MagicMock()
        wildcard = MagicMock()

        bus.on(EventType.NEW_MESSAGE, handler1)
        bus.on(EventType.AI_RESPONSE, handler2)
        bus.on_any(wildcard)

        bus.clear()

        await bus.emit(EventType.NEW_MESSAGE, text="test")
        await bus.emit(EventType.AI_RESPONSE, response="test")

        # Give executor time to complete
        await asyncio.sleep(0.1)

        handler1.assert_not_called()
        handler2.assert_not_called()
        wildcard.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self) -> None:
        """Test multiple handlers for the same event."""
        bus = EventBus()
        calls = []

        async def handler1(event_type: EventType, **data: object) -> None:
            calls.append("handler1")

        async def handler2(event_type: EventType, **data: object) -> None:
            calls.append("handler2")

        bus.on(EventType.APP_START, handler1)
        bus.on(EventType.APP_START, handler2)

        await bus.emit(EventType.APP_START)

        assert len(calls) == 2
        assert "handler1" in calls
        assert "handler2" in calls

    @pytest.mark.asyncio
    async def test_emit_with_data_kwargs(self) -> None:
        """Test emitting events with keyword arguments."""
        bus = EventBus()
        received_data = {}

        async def handler(event_type: EventType, **data: object) -> None:
            received_data.update(data)

        bus.on(EventType.NEW_MESSAGE, handler)
        await bus.emit(
            EventType.NEW_MESSAGE,
            text="Hello",
            user="Alice",
            timestamp=123456,
        )

        assert received_data["text"] == "Hello"
        assert received_data["user"] == "Alice"
        assert received_data["timestamp"] == 123456

    @pytest.mark.asyncio
    async def test_emitting_unknown_event_does_not_crash(self) -> None:
        """Test that emitting an event with no handlers doesn't crash."""
        bus = EventBus()

        # This should not raise any exception
        await bus.emit(EventType.MOOD_CHANGE, mood="happy")

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_stop_other_handlers(self) -> None:
        """Test that exception in one handler doesn't prevent others from running."""
        bus = EventBus()
        calls = []

        async def failing_handler(event_type: EventType, **data: object) -> None:
            raise ValueError("Handler error")

        async def working_handler(event_type: EventType, **data: object) -> None:
            calls.append("worked")

        bus.on(EventType.ERROR, failing_handler)
        bus.on(EventType.ERROR, working_handler)

        # Should not raise despite failing_handler exception
        await bus.emit(EventType.ERROR, message="test")

        assert "worked" in calls


class TestDefaultBus:
    """Tests for the default bus and convenience functions."""

    def setup_method(self) -> None:
        """Clear the default bus before each test."""
        default_bus.clear()

    @pytest.mark.asyncio
    async def test_convenience_functions_use_default_bus(self) -> None:
        """Test that convenience functions delegate to default_bus."""
        calls = []

        async def handler(event_type: EventType, **data: object) -> None:
            calls.append("called")

        on(EventType.APP_START, handler)
        await emit(EventType.APP_START)

        assert len(calls) == 1

        off(EventType.APP_START, handler)
        await emit(EventType.APP_START)

        # Should still be 1 since we unsubscribed
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_on_any_convenience_function(self) -> None:
        """Test on_any convenience function."""
        calls = []

        async def wildcard(event_type: EventType, **data: object) -> None:
            calls.append(event_type)

        on_any(wildcard)

        await emit(EventType.NEW_MESSAGE, text="test")
        await emit(EventType.AI_RESPONSE, response="test")

        assert len(calls) == 2


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_defined(self) -> None:
        """Test that all required event types are defined."""
        assert EventType.NEW_MESSAGE
        assert EventType.AI_RESPONSE
        assert EventType.TTS_START
        assert EventType.TTS_END
        assert EventType.STT_START
        assert EventType.STT_END
        assert EventType.ERROR
        assert EventType.MOOD_CHANGE
        assert EventType.APP_START
        assert EventType.APP_STOP

    def test_event_type_values(self) -> None:
        """Test that event types have correct string values."""
        assert EventType.NEW_MESSAGE.value == "new_message"
        assert EventType.APP_START.value == "app_start"
        assert EventType.APP_STOP.value == "app_stop"
