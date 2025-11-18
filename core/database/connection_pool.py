"""
Database Connection Pool Manager
Optimized connection pooling for PostgreSQL (future SQL migration)
"""
from typing import Optional
from contextlib import contextmanager
import logging
import os

try:
    from sqlalchemy import create_engine, pool, event
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not installed - connection pooling unavailable")

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """
    Manages database connection pooling with optimized settings

    🚀 OPTIMIZATIONS:
    - pool_size=20 (vs default 5) - More concurrent connections
    - max_overflow=10 (vs default 10) - Burst capacity for spikes
    - pool_pre_ping=True - Health checks before using connections
    - pool_recycle=3600 - Recycle connections every hour (prevents stale)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for connection pooling")

        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL must be provided or set in environment")

        # 🚀 OPTIMIZATION: Create engine with optimized pool settings
        self.engine = create_engine(
            self.database_url,
            poolclass=pool.QueuePool,  # Thread-safe connection pool
            pool_size=pool_size,  # Persistent connections
            max_overflow=max_overflow,  # Temporary connections for bursts
            pool_recycle=pool_recycle,  # Recycle connections to prevent stale
            pool_pre_ping=pool_pre_ping,  # Health check before use
            echo=echo,  # Log SQL statements (debug only)
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Register event listeners for monitoring
        self._register_event_listeners()

        logger.info(
            f"Connection pool initialized: "
            f"size={pool_size}, max_overflow={max_overflow}, "
            f"recycle={pool_recycle}s, pre_ping={pool_pre_ping}"
        )

    def _register_event_listeners(self):
        """Register SQLAlchemy event listeners for monitoring"""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("New database connection established")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            logger.debug("Connection returned to pool")

    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session from the pool

        Usage:
            with pool_manager.get_session() as session:
                result = session.query(User).filter_by(id=1).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_pool_status(self) -> dict:
        """
        Get current connection pool status

        Returns:
            dict: Pool statistics (size, checked_out, overflow, etc.)
        """
        pool_obj = self.engine.pool
        return {
            "size": pool_obj.size(),
            "checked_out": pool_obj.checkedout(),
            "overflow": pool_obj.overflow(),
            "checked_in": pool_obj.checkedin(),
        }

    def dispose(self):
        """Dispose of all connections in the pool"""
        self.engine.dispose()
        logger.info("Connection pool disposed")


# Singleton instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager(
    database_url: Optional[str] = None,
    **kwargs
) -> ConnectionPoolManager:
    """
    Get or create the connection pool manager instance

    Args:
        database_url: PostgreSQL connection URL (optional)
        **kwargs: Additional pool configuration options

    Returns:
        ConnectionPoolManager instance
    """
    global _pool_manager

    if _pool_manager is None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for connection pooling")

        _pool_manager = ConnectionPoolManager(database_url, **kwargs)

    return _pool_manager


# Example usage when migrating to SQL
if __name__ == "__main__":
    # Example configuration
    DATABASE_URL = "postgresql://user:password@localhost:5432/ultravox"

    # Initialize pool
    pool = ConnectionPoolManager(
        database_url=DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_recycle=3600,
        pool_pre_ping=True,
    )

    # Use in application
    with pool.get_session() as session:
        # Query database
        # result = session.query(User).filter_by(id=1).first()
        pass

    # Monitor pool
    status = pool.get_pool_status()
    print(f"Pool status: {status}")

    # Cleanup
    pool.dispose()
