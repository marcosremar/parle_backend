"""
Pydantic models for Acoustic Features Service
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ExtractFeaturesRequest(BaseModel):
    """Request for feature extraction"""
    audio_path: Optional[str] = None
    # For future: support base64 encoded audio


class ExtractFeaturesResponse(BaseModel):
    """Response with extracted features"""
    features: List[float]
    native_embedding_shape: List[int]
    learner_embedding_shape: List[int]
    fused_shape: List[int]
    model_info: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    device: str
    models_loaded: bool

