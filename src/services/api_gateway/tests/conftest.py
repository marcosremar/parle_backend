"""
Test Configuration and Fixtures for API Gateway Service.

Provides fixtures for:
- Mock HTTP clients
- Mock service communication
- Test request/response data
- Authentication mocks
- Rate limiting helpers
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def gateway_config():
    """API Gateway configuration for testing."""
    return {
        "port": 8000,
        "timeout": 5.0,
        "max_retries": 3,
        "rate_limit_requests": 100,
        "rate_limit_window": 60,
        "enable_auth": True,
        "enable_cors": True,
    }


@pytest.fixture
def service_registry():
    """Service registry for testing."""
    return {
        "session": {"host": "localhost", "port": 8001, "status": "healthy"},
        "llm": {"host": "localhost", "port": 8002, "status": "healthy"},
        "stt": {"host": "localhost", "port": 8003, "status": "healthy"},
        "tts": {"host": "localhost", "port": 8004, "status": "healthy"},
        "database": {"host": "localhost", "port": 8005, "status": "healthy"},
    }


# ============================================================================
# Mock HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for service calls."""
    client = AsyncMock()

    async def get(url: str, **kwargs):
        return MagicMock(
            status_code=200,
            json=lambda: {"status": "ok", "data": {}},
            text="Success"
        )

    async def post(url: str, **kwargs):
        return MagicMock(
            status_code=200,
            json=lambda: {"status": "ok", "data": {"id": "123"}},
            text="Success"
        )

    client.get = get
    client.post = post
    return client


# ============================================================================
# Request/Response Data Fixtures
# ============================================================================

@pytest.fixture
def valid_request_data():
    """Valid request data for testing."""
    return {
        "text": "Hello, world!",
        "language": "en-US",
        "user_id": "test_user_123"
    }


@pytest.fixture
def invalid_request_data():
    """Invalid request data for testing."""
    return {
        "text": "",  # Empty text
        # Missing required fields
    }


@pytest.fixture
def generate_requests():
    """Generate multiple test requests."""
    def _generate(count: int) -> List[Dict]:
        requests = []
        for i in range(count):
            requests.append({
                "text": f"Test message {i}",
                "language": "en-US",
                "user_id": f"user_{i}"
            })
        return requests
    return _generate


# ============================================================================
# Authentication Fixtures
# ============================================================================

@pytest.fixture
def valid_api_key():
    """Valid API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def invalid_api_key():
    """Invalid API key for testing."""
    return "invalid_key"


@pytest.fixture
def auth_headers(valid_api_key):
    """Valid authentication headers."""
    return {
        "Authorization": f"Bearer {valid_api_key}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def mock_auth_service():
    """Mock authentication service."""

    class MockAuthService:
        def __init__(self):
            self.valid_keys = {"test_api_key_12345", "admin_key"}
            self.sessions = {}

        async def validate_api_key(self, api_key: str) -> bool:
            """Validate API key."""
            return api_key in self.valid_keys

        async def create_session(self, user_id: str) -> str:
            """Create user session."""
            session_id = f"session_{user_id}_{datetime.now().timestamp()}"
            self.sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat()
            }
            return session_id

        async def validate_session(self, session_id: str) -> bool:
            """Validate session."""
            return session_id in self.sessions

    return MockAuthService()


# ============================================================================
# Rate Limiting Fixtures
# ============================================================================

@pytest.fixture
def rate_limiter():
    """Rate limiter for testing."""

    class RateLimiter:
        def __init__(self):
            self.requests = {}
            self.limit = 100
            self.window = 60  # seconds

        def check_limit(self, client_id: str) -> bool:
            """Check if client has exceeded rate limit."""
            now = datetime.now()

            if client_id not in self.requests:
                self.requests[client_id] = []

            # Clean old requests
            cutoff = now - timedelta(seconds=self.window)
            self.requests[client_id] = [
                ts for ts in self.requests[client_id]
                if ts > cutoff
            ]

            # Check limit
            if len(self.requests[client_id]) >= self.limit:
                return False

            # Add request
            self.requests[client_id].append(now)
            return True

        def get_remaining(self, client_id: str) -> int:
            """Get remaining requests for client."""
            if client_id not in self.requests:
                return self.limit

            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window)
            active_requests = [
                ts for ts in self.requests[client_id]
                if ts > cutoff
            ]
            return max(0, self.limit - len(active_requests))

    return RateLimiter()


# ============================================================================
# Service Communication Fixtures
# ============================================================================

@pytest.fixture
async def mock_service_comm():
    """Mock service communication layer."""
    comm = AsyncMock()

    async def call_service(service_name: str, endpoint: str, method: str = "GET", **kwargs):
        """Mock service call."""
        return {
            "success": True,
            "data": {
                "service": service_name,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat()
            }
        }

    comm.call_service = call_service
    return comm


# ============================================================================
# Request Validation Fixtures
# ============================================================================

@pytest.fixture
def request_validator():
    """Request validation helper."""

    class RequestValidator:
        def validate_required_fields(self, data: Dict, required: List[str]) -> bool:
            """Validate required fields are present."""
            return all(field in data for field in required)

        def validate_field_types(self, data: Dict, schema: Dict) -> bool:
            """Validate field types match schema."""
            for field, expected_type in schema.items():
                if field in data:
                    if not isinstance(data[field], expected_type):
                        return False
            return True

        def validate_json_schema(self, data: Dict, schema: Dict) -> bool:
            """Validate data against JSON schema."""
            # Simplified schema validation
            return self.validate_required_fields(
                data,
                schema.get("required", [])
            )

        def sanitize_input(self, text: str) -> str:
            """Sanitize user input."""
            # Remove potentially dangerous characters
            dangerous = ["<", ">", "&", '"', "'"]
            sanitized = text
            for char in dangerous:
                sanitized = sanitized.replace(char, "")
            return sanitized

    return RequestValidator()


# ============================================================================
# Error Response Fixtures
# ============================================================================

@pytest.fixture
def error_responses():
    """Common error response templates."""
    return {
        "400": {
            "error": "Bad Request",
            "message": "Invalid request data",
            "status_code": 400
        },
        "401": {
            "error": "Unauthorized",
            "message": "Authentication required",
            "status_code": 401
        },
        "403": {
            "error": "Forbidden",
            "message": "Access denied",
            "status_code": 403
        },
        "404": {
            "error": "Not Found",
            "message": "Resource not found",
            "status_code": 404
        },
        "429": {
            "error": "Too Many Requests",
            "message": "Rate limit exceeded",
            "status_code": 429
        },
        "500": {
            "error": "Internal Server Error",
            "message": "An internal error occurred",
            "status_code": 500
        },
        "503": {
            "error": "Service Unavailable",
            "message": "Service temporarily unavailable",
            "status_code": 503
        }
    }


# ============================================================================
# Routing Fixtures
# ============================================================================

@pytest.fixture
def route_registry():
    """Registry of available routes."""
    return {
        "/health": {"methods": ["GET"], "auth_required": False},
        "/api/sessions": {"methods": ["POST", "GET"], "auth_required": True},
        "/api/llm/chat": {"methods": ["POST"], "auth_required": True},
        "/api/stt/transcribe": {"methods": ["POST"], "auth_required": True},
        "/api/tts/synthesize": {"methods": ["POST"], "auth_required": True},
    }


@pytest.fixture
def mock_gateway_service(gateway_config, mock_service_comm):
    """Mock API Gateway Service."""

    class MockGatewayService:
        def __init__(self, config, comm):
            self.config = config
            self.comm = comm
            self.logger = MagicMock()

        async def health_check(self) -> Dict:
            """Health check."""
            return {
                "service": "api_gateway",
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }

        async def proxy_request(self, service: str, endpoint: str, method: str, data: Dict = None):
            """Proxy request to backend service."""
            return await self.comm.call_service(
                service_name=service,
                endpoint=endpoint,
                method=method,
                data=data
            )

    return MockGatewayService(gateway_config, mock_service_comm)


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
async def mock_gateway():
    """Comprehensive mock gateway for integration tests."""

    class MockGateway:
        def __init__(self):
            self.services_called = []
            self.metrics = {}
            self.cache = {}

        async def handle_request(self, request: Dict) -> Dict:
            """Handle complete request flow."""
            # Extract request details
            path = request.get("path", "/")
            method = request.get("method", "GET")
            headers = request.get("headers", {})
            body = request.get("body", {})

            # Simulate authentication
            auth_header = headers.get("Authorization", "")
            authenticated = auth_header.startswith("Bearer ") and "token" in auth_header.lower()

            # Check auth for protected routes
            if path.startswith("/api/v1/protected"):
                if not authenticated:
                    return {"status": 401, "error": {"type": "Unauthorized", "message": "Authentication required"}}

            # Handle specific routes
            if path == "/api/v1/failing-service":
                return {"status": 503, "error": {"type": "ServiceUnavailable", "message": "Service down"}}

            if path == "/api/v1/voice/chat":
                self.services_called = ["auth", "stt", "llm", "tts"]
                return {"status": 200, "services_called": self.services_called}

            if path.startswith("/api/v1/cached-resource"):
                cache_key = path
                if cache_key in self.cache:
                    return {"status": 200, "cache_hit": True, "data": self.cache[cache_key]}
                else:
                    data = {"resource": "data"}
                    self.cache[cache_key] = data
                    return {"status": 200, "cache_hit": False, "data": data}

            if path == "/api/v1/resilient-endpoint":
                return {"status": 200}

            if path == "/api/v1/transform":
                return {
                    "status": 200,
                    "data": {
                        "user_input": body.get("user_input"),
                        "format": "processed",
                        "transformed": True
                    }
                }

            if path == "/api/v1/conversation/analyze":
                return {
                    "status": 200,
                    "services_involved": ["conversation_store", "llm", "metrics"]
                }

            # Default response
            return {
                "status": 200,
                "data": {"message": "Success"},
                "authenticated": authenticated,
                "request_id": headers.get("X-Request-ID", ""),
                "headers": {"X-Request-ID": headers.get("X-Request-ID", "")},
                "metrics": {
                    "request_duration_ms": 25,
                    "services_latency": {"gateway": 5, "service": 20}
                }
            }

        async def proxy_request(self, request: Dict, target: str, **kwargs) -> Dict:
            """Proxy request to target service."""
            path = request.get("path", "/")
            method = request.get("method", "GET")
            headers = request.get("headers", {})

            # Handle slow service
            if target == "slow_service":
                return {
                    "status": 504,
                    "error": {"type": "GatewayTimeout", "message": "Request timeout"}
                }

            # Handle unreliable service
            if target == "unreliable_service":
                return {
                    "status": 200,
                    "retry_count": 2
                }

            # Handle failing service (circuit breaker)
            if target == "failing_service":
                return {
                    "status": 503,
                    "circuit_breaker_open": True,
                    "error": {"message": "Circuit breaker open"}
                }

            # Handle load balanced service
            if target == "load_balanced_service":
                import random
                actual_target = f"instance_{random.randint(1, 3)}"
                return {
                    "status": 200,
                    "actual_target": actual_target
                }

            # Handle request transformation
            if kwargs.get("transform_request"):
                return {
                    "status": 200,
                    "request_transformed": True,
                    "original_format": "external_format",
                    "transformed_format": "internal_format"
                }

            # Handle response transformation
            if kwargs.get("transform_response"):
                return {
                    "status": 200,
                    "response_transformed": True,
                    "transformed_data": {"result": "transformed"}
                }

            # Default proxy response
            return {
                "status": 200,
                "proxied_to": target,
                "data": {"service": target, "endpoint": path},
                "forwarded_headers": headers
            }

    return MockGateway()
