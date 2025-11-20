"""
Configuration module - Shared settings for all services
"""
from .settings import *
from .settings_service import SettingsService

__all__ = [
    'get_settings',
    'SettingsService',
    # Add other exports as needed
]

