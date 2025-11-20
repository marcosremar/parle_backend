"""
Pytest configuration and fixtures for Session Service tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import redis.asyncio as aioredis
from datetime import datetime, timedelta
import uuid


# ============================================================================
# Test Configuration
# ============================================================================

# NOTE: Event loop fixture removed - pytest-asyncio now provides this automatically
# Custom event_loop fixtures can cause conflicts and deprecation warnings


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock(spec=aioredis.Redis)

    # Mock storage
    storage: Dict[str, Any] = {}
    expiry: Dict[str, datetime] = {}

    async def get(key: str):
        """Mock get operation."""
        if key in expiry and expiry[key] < datetime.now():
            del storage[key]
            del expiry[key]
            return None
        return storage.get(key)

    async def set(key: str, value: Any, ex: int = None):
        """Mock set operation."""
        storage[key] = value
        if ex:
            expiry[key] = datetime.now() + timedelta(seconds=ex)
        return True

    async def delete(key: str):
        """Mock delete operation."""
        if key in storage:
            del storage[key]
            if key in expiry:
                del expiry[key]
            return 1
        return 0

    async def exists(key: str):
        """Mock exists operation."""
        return 1 if key in storage else 0

    async def keys(pattern: str):
        """Mock keys operation."""
        import fnmatch
        return [k for k in storage.keys() if fnmatch.fnmatch(k, pattern)]

    async def expire(key: str, seconds: int):
        """Mock expire operation."""
        if key in storage:
            expiry[key] = datetime.now() + timedelta(seconds=seconds)
            return True
        return False

    async def ttl(key: str):
        """Mock TTL operation."""
        if key not in storage:
            return -2  # Key doesn't exist
        if key not in expiry:
            return -1  # No expiry set
        remaining = (expiry[key] - datetime.now()).total_seconds()
        return int(remaining) if remaining > 0 else -2

    async def ping():
        """Mock ping operation."""
        return True

    # Assign mock methods
    redis_mock.get = get
    redis_mock.set = set
    redis_mock.delete = delete
    redis_mock.exists = exists
    redis_mock.keys = keys
    redis_mock.expire = expire
    redis_mock.ttl = ttl
    redis_mock.ping = ping
    redis_mock._storage = storage
    redis_mock._expiry = expiry

    return redis_mock


@pytest.fixture
async def redis_pool():
    """Mock Redis connection pool."""
    pool_mock = MagicMock()
    pool_mock.max_connections = 10
    pool_mock.connection_class = aioredis.Connection
    return pool_mock


@pytest.fixture
async def redis_client(mock_redis):
    """Redis client fixture using mock."""
    return mock_redis


# ============================================================================
# Session Fixtures
# ============================================================================

@pytest.fixture
def session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())


@pytest.fixture
def user_id():
    """Generate a unique user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def session_data():
    """Sample session data."""
    return {
        "user_id": str(uuid.uuid4()),
        "username": "test_user",
        "email": "test@example.com",
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat(),
        "metadata": {
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
            "device": "desktop"
        }
    }


@pytest.fixture
def session_config():
    """Session service configuration."""
    return {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "session_ttl": 3600,  # 1 hour
        "max_sessions_per_user": 5,
        "session_prefix": "session:",
        "user_sessions_prefix": "user_sessions:"
    }


# ============================================================================
# Mock Session Manager
# ============================================================================

class MockSessionManager:
    """Mock Session Manager for testing."""

    def __init__(self, redis_client, config):
        self.redis = redis_client
        self.config = config
        self.sessions = {}

    async def create_session(self, user_id: str, data: dict) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session_key = f"{self.config['session_prefix']}{session_id}"

        session_data = {
            **data,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }

        import json
        await self.redis.set(
            session_key,
            json.dumps(session_data),
            ex=self.config['session_ttl']
        )

        self.sessions[session_id] = session_data
        return session_id

    async def get_session(self, session_id: str) -> dict:
        """Get session by ID."""
        session_key = f"{self.config['session_prefix']}{session_id}"
        import json
        data = await self.redis.get(session_key)
        if data:
            return json.loads(data)
        return None

    async def update_session(self, session_id: str, data: dict) -> bool:
        """Update session data."""
        session_key = f"{self.config['session_prefix']}{session_id}"
        existing = await self.get_session(session_id)
        if not existing:
            return False

        updated_data = {
            **existing,
            **data,
            "last_activity": datetime.now().isoformat()
        }

        import json
        await self.redis.set(
            session_key,
            json.dumps(updated_data),
            ex=self.config['session_ttl']
        )

        self.sessions[session_id] = updated_data
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        session_key = f"{self.config['session_prefix']}{session_id}"
        result = await self.redis.delete(session_key)
        if session_id in self.sessions:
            del self.sessions[session_id]
        return result > 0

    async def list_sessions(self, user_id: str = None) -> list:
        """List sessions."""
        if user_id:
            pattern = f"{self.config['session_prefix']}*"
            keys = await self.redis.keys(pattern)
            sessions = []
            import json
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    session = json.loads(data)
                    if session.get("user_id") == user_id:
                        sessions.append(session)
            return sessions
        else:
            return list(self.sessions.values())

    async def renew_session(self, session_id: str) -> bool:
        """Renew session TTL."""
        session_key = f"{self.config['session_prefix']}{session_id}"
        exists = await self.redis.exists(session_key)
        if exists:
            return await self.redis.expire(session_key, self.config['session_ttl'])
        return False


@pytest.fixture
async def session_manager(mock_redis, session_config):
    """Session manager instance."""
    return MockSessionManager(mock_redis, session_config)


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_sessions():
    """Generate multiple test sessions."""
    def _generate(count: int, user_id: str = None):
        sessions = []
        for i in range(count):
            sessions.append({
                "session_id": str(uuid.uuid4()),
                "user_id": user_id or str(uuid.uuid4()),
                "username": f"user_{i}",
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "index": i,
                    "test": True
                }
            })
        return sessions
    return _generate


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_redis(mock_redis):
    """Cleanup Redis storage after each test."""
    yield
    # Clear storage
    if hasattr(mock_redis, '_storage'):
        mock_redis._storage.clear()
    if hasattr(mock_redis, '_expiry'):
        mock_redis._expiry.clear()
