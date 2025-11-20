"""
Settings for API Gateway
"""

import os
from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """Configuration settings for server"""
    host: str = "0.0.0.0"
    port: int = Field(default_factory=lambda: int(os.getenv("API_GATEWAY_PORT", "8011")))  # ✅ Phase 3c: Correct default from PORT_MATRIX


class Settings(BaseModel):
    """Configuration settings for """
    server: ServerSettings = Field(default_factory=ServerSettings)


def get_settings():
    """Get application settings"""
    return Settings()