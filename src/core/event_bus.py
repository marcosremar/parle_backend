"""
Simple In-Memory Event Bus for Monolith
Replaces external message queues for internal communication
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from loguru import logger


class EventBus:
    """
    Simple in-memory event bus for monolith architecture
    Replaces external message queues (Redis, RabbitMQ) for internal communication
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._async_subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._enabled = True

    def subscribe(self, event_type: str, handler: Callable, async_handler: bool = False):
        """
        Subscribe to an event type

        Args:
            event_type: Event type name (e.g., 'user.created', 'conversation.started')
            handler: Handler function (sync or async)
            async_handler: Whether handler is async
        """
        if async_handler:
            self._async_subscribers[event_type].append(handler)
        else:
            self._subscribers[event_type].append(handler)
        logger.debug(f"✅ Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
        if handler in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, data: Any = None):
        """
        Publish an event

        Args:
            event_type: Event type name
            data: Event data
        """
        if not self._enabled:
            return

        logger.debug(f"📢 Publishing event: {event_type}")

        # Call sync handlers
        for handler in self._subscribers[event_type]:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"❌ Error in sync handler for {event_type}: {e}")

        # Call async handlers
        for handler in self._async_subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"❌ Error in async handler for {event_type}: {e}")

    def publish_sync(self, event_type: str, data: Any = None):
        """
        Publish an event synchronously (for sync code)

        Args:
            event_type: Event type name
            data: Event data
        """
        if not self._enabled:
            return

        logger.debug(f"📢 Publishing event (sync): {event_type}")

        # Call sync handlers
        for handler in self._subscribers[event_type]:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"❌ Error in sync handler for {event_type}: {e}")

    def disable(self):
        """Disable event bus (useful for testing)"""
        self._enabled = False

    def enable(self):
        """Enable event bus"""
        self._enabled = True

    def clear(self):
        """Clear all subscribers (useful for testing)"""
        self._subscribers.clear()
        self._async_subscribers.clear()


# Global singleton instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get global event bus instance (singleton)"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus():
    """Reset event bus (useful for testing)"""
    global _event_bus
    _event_bus = None
