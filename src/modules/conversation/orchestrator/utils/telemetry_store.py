"""
Telemetry Store - Request History Storage

Stores telemetry data for recent requests in memory (circular buffer).
Allows querying telemetry data via JSON API.
"""

import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class TelemetryRecord:
    """Single telemetry record for a request"""

    request_id: str
    service_name: str
    method: str
    path: str
    timestamp: float  # Unix timestamp
    timestamp_iso: str  # ISO format for readability
    request_size_bytes: int
    response_size_bytes: int
    processing_time_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class TelemetryStore:
    """
    Thread-safe in-memory storage for telemetry records

    Stores the last N requests in a circular buffer (FIFO).
    """

    def __init__(self, max_records: int = 1000):
        """
        Initialize telemetry store

        Args:
            max_records: Maximum number of records to keep (default: 1000)
        """
        self.max_records = max_records
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._records_by_id: Dict[str, TelemetryRecord] = {}

    def add_record(
        self,
        request_id: str,
        service_name: str,
        method: str,
        path: str,
        request_size_bytes: int,
        response_size_bytes: int,
        processing_time_ms: float,
        status_code: int,
        error: Optional[str] = None,
    ) -> None:
        """
        Add a telemetry record

        Args:
            request_id: Unique request identifier
            service_name: Name of the service
            method: HTTP method (GET, POST, etc.)
            path: Request path
            request_size_bytes: Request payload size
            response_size_bytes: Response payload size
            processing_time_ms: Processing time in milliseconds
            status_code: HTTP status code
            error: Error message if request failed
        """
        now = time.time()
        success = status_code < 400 and error is None

        record = TelemetryRecord(
            request_id=request_id,
            service_name=service_name.upper(),
            method=method,
            path=path,
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now).isoformat(),
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            processing_time_ms=round(processing_time_ms, 2),
            status_code=status_code,
            success=success,
            error=error,
        )

        with self._lock:
            # If buffer is full, remove oldest record from index
            if len(self._records) == self.max_records:
                oldest = self._records[0]
                self._records_by_id.pop(oldest.request_id, None)

            # Add new record
            self._records.append(record)
            self._records_by_id[request_id] = record

    def get_record(self, request_id: str) -> Optional[TelemetryRecord]:
        """
        Get a specific record by request ID

        Args:
            request_id: Request identifier

        Returns:
            TelemetryRecord if found, None otherwise
        """
        with self._lock:
            return self._records_by_id.get(request_id)

    def get_total_count(self) -> int:
        """
        Get total number of records currently in store

        Returns:
            Total number of records
        """
        with self._lock:
            return len(self._records)

    def get_latest(self, limit: int = 10) -> List[TelemetryRecord]:
        """
        Get latest N records

        Args:
            limit: Number of records to return (default: 10)

        Returns:
            List of latest telemetry records (newest first)
        """
        with self._lock:
            records = list(self._records)
            # Return newest first
            return list(reversed(records))[:limit]

    def get_by_service(
        self, service_name: str, limit: int = 10
    ) -> List[TelemetryRecord]:
        """
        Get latest records for a specific service

        Args:
            service_name: Service name to filter by
            limit: Number of records to return

        Returns:
            List of telemetry records for the service
        """
        service_upper = service_name.upper()
        with self._lock:
            records = [r for r in self._records if r.service_name == service_upper]
            return list(reversed(records))[:limit]

    def get_stats(self) -> Dict:
        """
        Get overall statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self._records:
                return {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_processing_time_ms": 0.0,
                    "total_data_bytes": 0,
                }

            total = len(self._records)
            successful = sum(1 for r in self._records if r.success)
            total_time = sum(r.processing_time_ms for r in self._records)
            total_data = sum(
                r.request_size_bytes + r.response_size_bytes for r in self._records
            )

            return {
                "total_requests": total,
                "success_rate": round((successful / total) * 100, 2),
                "avg_processing_time_ms": round(total_time / total, 2),
                "total_data_bytes": total_data,
                "oldest_timestamp": self._records[0].timestamp_iso,
                "newest_timestamp": self._records[-1].timestamp_iso,
            }

    def clear(self) -> None:
        """Clear all records"""
        with self._lock:
            self._records.clear()
            self._records_by_id.clear()


# Global telemetry store (singleton)
_global_store: Optional[TelemetryStore] = None


def get_telemetry_store() -> TelemetryStore:
    """
    Get the global telemetry store (singleton)

    Returns:
        Global TelemetryStore instance
    """
    global _global_store
    if _global_store is None:
        _global_store = TelemetryStore(max_records=1000)
    return _global_store
