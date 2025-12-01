"""
HTTP Client with Connection Pooling
Reuses aiohttp.ClientSession for better performance
"""

import aiohttp
from loguru import logger

# Import constants with fallback
try:
    from src.core.constants import DEFAULT_CONNECT_TIMEOUT, DEFAULT_HTTP_TIMEOUT
except ImportError:
    DEFAULT_HTTP_TIMEOUT = 30
    DEFAULT_CONNECT_TIMEOUT = 10


class HTTPClient:
    """
    Singleton HTTP client with connection pooling

    Reuses aiohttp.ClientSession to avoid creating new connections
    for each request, improving performance significantly.

    Usage:
        session = await HTTPClient.get_session()
        async with session.get(url) as resp:
            data = await resp.json()
    """

    _session: aiohttp.ClientSession | None = None
    _lock = None

    @classmethod
    async def get_session(
        cls, timeout: aiohttp.ClientTimeout | None = None
    ) -> aiohttp.ClientSession:
        """
        Get or create shared aiohttp.ClientSession

        Args:
            timeout: Optional custom timeout (defaults to 30s total, 10s connect)

        Returns:
            aiohttp.ClientSession instance (reused across calls)
        """
        if cls._session is None or cls._session.closed:
            # Create new session with default timeout
            if timeout is None:
                timeout = aiohttp.ClientTimeout(
                    total=DEFAULT_HTTP_TIMEOUT, connect=DEFAULT_CONNECT_TIMEOUT
                )
            cls._session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("Created new HTTP client session")

        return cls._session

    @classmethod
    async def close_session(cls):
        """
        Close the shared HTTP session.

        Should be called during application shutdown to properly
        close connections and free resources.

        Note:
            This is automatically handled by the application lifecycle.
        """
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None
            logger.debug("Closed HTTP client session")

    @classmethod
    async def __aenter__(cls):
        """Async context manager entry"""
        return await cls.get_session()

    @classmethod
    async def __aexit__(cls, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Don't close on exit - keep session alive for reuse
