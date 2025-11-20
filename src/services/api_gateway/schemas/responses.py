"""
Common response models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    services: Dict[str, bool] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        """Configuration settings for """
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-09-19T14:00:00",
                "version": "1.0.0",
                "services": {
                    "ultravox": True,
                    "tts": True,
                    "stt": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    status_code: int = 400

    class Config:
        """Configuration settings for """
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid request",
                "detail": "Missing required field: text",
                "status_code": 400
            }
        }