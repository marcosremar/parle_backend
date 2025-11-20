"""Shared fixtures for REST Polling Service tests."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def polling_config():
    """REST Polling configuration."""
    return {
        "timeout": 5.0,
        "max_timeout": 5.0,
        "poll_interval": 1.0,
        "max_connections": 100,
        "max_requests_per_minute": 60,
        "enable_long_polling": True,
        "fallback_enabled": True,
        "retry_attempts": 3,
        "backoff_multiplier": 2.0
    }


@pytest.fixture
def mock_polling_client():
    """Mock polling client."""

    class MockPollingClient:
        def __init__(self):
            self.client_id = "polling_client_123"
            self.connected = False
            self.last_poll = None
            self.pending_messages = []
            self.poll_count = 0
            self.timeout_count = 0

        async def poll(self, timeout: float = 30.0) -> Dict:
            """Execute polling request."""
            self.poll_count += 1
            self.last_poll = datetime.now()

            # Simulate waiting for messages
            await asyncio.sleep(0.01)

            if self.pending_messages:
                messages = self.pending_messages.copy()
                self.pending_messages.clear()
                return {
                    "status": "success",
                    "messages": messages,
                    "has_more": False,
                    "timestamp": datetime.now().isoformat()
                }

            # Long polling timeout
            return {
                "status": "timeout",
                "messages": [],
                "has_more": False,
                "timestamp": datetime.now().isoformat()
            }

        async def send_message(self, message: Dict) -> bool:
            """Send message via polling."""
            self.pending_messages.append(message)
            return True

        def add_pending_message(self, message: Dict):
            """Add message to pending queue."""
            self.pending_messages.append(message)

    return MockPollingClient()


@pytest.fixture
def polling_manager():
    """Polling connection manager."""

    class PollingManager:
        def __init__(self):
            self.active_polls = {}
            self.timeout_handlers = {}
            self.message_queues = {}
            self.metrics = {
                "total_polls": 0,
                "successful_polls": 0,
                "timeout_polls": 0,
                "failed_polls": 0
            }

        async def register_poll(self, client_id: str, timeout: float) -> str:
            """Register new long-polling request."""
            poll_id = f"poll_{client_id}_{datetime.now().timestamp()}"
            self.active_polls[poll_id] = {
                "client_id": client_id,
                "timeout": timeout,
                "start_time": datetime.now(),
                "status": "active"
            }
            self.metrics["total_polls"] += 1
            return poll_id

        async def complete_poll(self, poll_id: str, messages: List[Dict]) -> Dict:
            """Complete polling request with messages."""
            if poll_id not in self.active_polls:
                return {"status": "error", "message": "Poll not found"}

            poll_info = self.active_polls[poll_id]
            poll_info["status"] = "completed"
            poll_info["end_time"] = datetime.now()

            self.metrics["successful_polls"] += 1

            return {
                "status": "success",
                "messages": messages,
                "poll_duration": (poll_info["end_time"] - poll_info["start_time"]).total_seconds()
            }

        async def timeout_poll(self, poll_id: str) -> Dict:
            """Handle polling timeout."""
            if poll_id in self.active_polls:
                self.active_polls[poll_id]["status"] = "timeout"
                self.metrics["timeout_polls"] += 1

            return {
                "status": "timeout",
                "messages": []
            }

        def get_active_polls(self, client_id: str) -> List[str]:
            """Get active polls for client."""
            return [
                poll_id for poll_id, info in self.active_polls.items()
                if info["client_id"] == client_id and info["status"] == "active"
            ]

        def add_message_to_queue(self, client_id: str, message: Dict):
            """Add message to client queue."""
            if client_id not in self.message_queues:
                self.message_queues[client_id] = []
            self.message_queues[client_id].append(message)

        def get_queued_messages(self, client_id: str) -> List[Dict]:
            """Get and clear queued messages."""
            messages = self.message_queues.get(client_id, [])
            self.message_queues[client_id] = []
            return messages

    return PollingManager()


@pytest.fixture
def timeout_handler():
    """Timeout management."""

    class TimeoutHandler:
        def __init__(self):
            self.timeouts = {}
            self.default_timeout = 30.0
            self.max_timeout=5.0

        def calculate_timeout(self, requested: float) -> float:
            """Calculate actual timeout."""
            return min(requested, self.max_timeout)

        async def wait_with_timeout(self, coro, timeout: float):
            """Execute coroutine with timeout."""
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                return {"status": "timeout"}

        def is_expired(self, start_time: datetime, timeout: float) -> bool:
            """Check if timeout expired."""
            elapsed = (datetime.now() - start_time).total_seconds()
            return elapsed >= timeout

    return TimeoutHandler()


@pytest.fixture
def fallback_manager():
    """Fallback mechanism manager."""

    class FallbackManager:
        def __init__(self):
            self.primary_type = "websocket"
            self.fallback_type = "polling"
            self.fallback_active = False
            self.fallback_reason = None
            self.fallback_count = 0

        async def check_primary_available(self) -> bool:
            """Check if primary connection type is available."""
            return not self.fallback_active

        async def activate_fallback(self, reason: str):
            """Activate fallback to polling."""
            self.fallback_active = True
            self.fallback_reason = reason
            self.fallback_count += 1

        async def deactivate_fallback(self):
            """Deactivate fallback."""
            self.fallback_active = False
            self.fallback_reason = None

        def get_connection_type(self) -> str:
            """Get current connection type."""
            return self.fallback_type if self.fallback_active else self.primary_type

    return FallbackManager()


@pytest.fixture
def rate_limiter():
    """Rate limiting for polling requests."""

    class RateLimiter:
        def __init__(self):
            self.max_requests = 60
            self.window_seconds = 60
            self.request_history = {}

        def is_allowed(self, client_id: str) -> bool:
            """Check if request is allowed."""
            now = datetime.now()

            if client_id not in self.request_history:
                self.request_history[client_id] = []

            # Clean old requests
            cutoff = now - timedelta(seconds=self.window_seconds)
            self.request_history[client_id] = [
                ts for ts in self.request_history[client_id]
                if ts > cutoff
            ]

            # Check limit
            if len(self.request_history[client_id]) >= self.max_requests:
                return False

            # Record request
            self.request_history[client_id].append(now)
            return True

        def get_remaining(self, client_id: str) -> int:
            """Get remaining requests in window."""
            if client_id not in self.request_history:
                return self.max_requests
            return self.max_requests - len(self.request_history[client_id])

    return RateLimiter()


@pytest.fixture
def retry_manager():
    """Retry management."""

    class RetryManager:
        def __init__(self):
            self.max_attempts = 3
            self.backoff_base = 1.0
            self.backoff_multiplier = 2.0
            self.attempts = {}

        def get_backoff_delay(self, client_id: str) -> float:
            """Calculate backoff delay."""
            attempt = self.attempts.get(client_id, 0)
            return self.backoff_base * (self.backoff_multiplier ** attempt)

        def record_attempt(self, client_id: str):
            """Record retry attempt."""
            self.attempts[client_id] = self.attempts.get(client_id, 0) + 1

        def should_retry(self, client_id: str) -> bool:
            """Check if should retry."""
            return self.attempts.get(client_id, 0) < self.max_attempts

        def reset(self, client_id: str):
            """Reset retry counter."""
            if client_id in self.attempts:
                del self.attempts[client_id]

    return RetryManager()


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {
            "id": "msg_1",
            "type": "text",
            "content": "Hello from polling",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "msg_2",
            "type": "notification",
            "content": "New update available",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "msg_3",
            "type": "system",
            "content": "Connection established",
            "timestamp": datetime.now().isoformat()
        }
    ]


@pytest.fixture
def performance_metrics():
    """Performance metrics collector."""

    class PerformanceMetrics:
        def __init__(self):
            self.poll_latencies = []
            self.timeout_rates = []
            self.throughput_samples = []
            self.concurrent_polls = 0
            self.max_concurrent = 0

        def record_poll_latency(self, latency_ms: float):
            """Record poll latency."""
            self.poll_latencies.append(latency_ms)

        def record_timeout(self, timed_out: bool):
            """Record timeout occurrence."""
            self.timeout_rates.append(1 if timed_out else 0)

        def record_throughput(self, messages_per_second: float):
            """Record throughput."""
            self.throughput_samples.append(messages_per_second)

        def increment_concurrent(self):
            """Increment concurrent polls."""
            self.concurrent_polls += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_polls)

        def decrement_concurrent(self):
            """Decrement concurrent polls."""
            self.concurrent_polls = max(0, self.concurrent_polls - 1)

        def get_avg_latency(self) -> float:
            """Get average latency."""
            return sum(self.poll_latencies) / len(self.poll_latencies) if self.poll_latencies else 0

        def get_timeout_rate(self) -> float:
            """Get timeout rate."""
            return sum(self.timeout_rates) / len(self.timeout_rates) if self.timeout_rates else 0

    return PerformanceMetrics()
