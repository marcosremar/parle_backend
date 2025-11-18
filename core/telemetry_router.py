"""
Telemetry Router - JSON API for Telemetry Data

Provides HTTP endpoints to query telemetry data:
- GET /telemetry - Get latest requests
- GET /telemetry/{request_id} - Get specific request
- GET /telemetry/service/{service_name} - Get requests by service
- GET /telemetry/stats - Get overall statistics
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.core.telemetry_store import get_telemetry_store, TelemetryRecord


# Pydantic models for API responses
class TelemetryRecordResponse(BaseModel):
    """Telemetry record response"""

    request_id: str
    service_name: str
    method: str
    path: str
    timestamp: float
    timestamp_iso: str
    request_size_bytes: int
    response_size_bytes: int
    processing_time_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None

    class Config:
        """Configuration settings for """
        from_attributes = True


class TelemetryListResponse(BaseModel):
    """List of telemetry records"""

    total: int
    records: List[TelemetryRecordResponse]


class TelemetryStatsResponse(BaseModel):
    """Telemetry statistics"""

    total_requests: int
    success_rate: float
    avg_processing_time_ms: float
    total_data_bytes: int
    oldest_timestamp: Optional[str] = None
    newest_timestamp: Optional[str] = None


def create_telemetry_router() -> APIRouter:
    """
    Create telemetry router with all endpoints

    Returns:
        FastAPI APIRouter with telemetry endpoints
    """
    router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

    @router.get("/", response_model=TelemetryListResponse)
    async def get_latest_telemetry(
        limit: int = Query(default=10, ge=1, le=100, description="Number of records"),
    ):
        """
        Get latest telemetry records

        Args:
            limit: Number of records to return (1-100, default: 10)

        Returns:
            List of latest telemetry records (newest first)
        """
        store = get_telemetry_store()
        records = store.get_latest(limit=limit)

        return TelemetryListResponse(
            total=store.get_total_count(),  # Total in buffer, not just returned
            records=[TelemetryRecordResponse(**r.to_dict()) for r in records],
        )

    @router.get("/{request_id}", response_model=TelemetryRecordResponse)
    async def get_telemetry_by_id(request_id: str) -> TelemetryRecordResponse:
        """
        Get specific telemetry record by ID

        Args:
            request_id: Request identifier (e.g., "database_1763256627723")

        Returns:
            Telemetry record

        Raises:
            404: If request not found
        """
        store = get_telemetry_store()
        record = store.get_record(request_id)

        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Request not found: {request_id}"
            )

        return TelemetryRecordResponse(**record.to_dict())

    @router.get("/service/{service_name}", response_model=TelemetryListResponse)
    async def get_telemetry_by_service(
        service_name: str,
        limit: int = Query(default=10, ge=1, le=100, description="Number of records"),
    ):
        """
        Get telemetry records for a specific service

        Args:
            service_name: Service name (e.g., "database", "orchestrator")
            limit: Number of records to return (1-100, default: 10)

        Returns:
            List of telemetry records for the service
        """
        store = get_telemetry_store()
        records = store.get_by_service(service_name=service_name, limit=limit)

        return TelemetryListResponse(
            total=len(records),
            records=[TelemetryRecordResponse(**r.to_dict()) for r in records],
        )

    @router.get("/stats/overall", response_model=TelemetryStatsResponse)
    async def get_telemetry_stats() -> TelemetryStatsResponse:
        """
        Get overall telemetry statistics

        Returns:
            Statistics about all stored requests
        """
        store = get_telemetry_store()
        stats = store.get_stats()

        return TelemetryStatsResponse(**stats)

    @router.delete("/")
    async def clear_telemetry() -> Dict[str, Any]:
        """
        Clear all telemetry records

        Returns:
            Success message
        """
        store = get_telemetry_store()
        store.clear()

        return {"message": "Telemetry records cleared", "success": True}

    return router
