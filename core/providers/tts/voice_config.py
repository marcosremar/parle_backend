"""
Voice Configuration Module - Centralizes voice mapping for TTS
"""

import os
from typing import Dict, Optional

# Voice IDs mapping to actual TTS voice names
VOICE_MAP = {
    # Portuguese voices
    "FEMALE_BR_1": "pt-BR-FranciscaNeural",
    "MALE_BR_1": "pt-BR-AntonioNeural",

    # English voices
    "FEMALE_US_1": "en-US-JennyNeural",
    "MALE_US_1": "en-US-GuyNeural",

    # Spanish voices
    "FEMALE_ES_1": "es-ES-ElviraNeural",
    "MALE_ES_1": "es-ES-AlvaroNeural",

    # Kokoro TTS voices (from config/settings.py)
    "pf_dora": "pf_dora",      # Portuguese female
    "pm_alex": "pm_alex",      # Portuguese male
    "af_bella": "af_bella",    # American female
    "am_michael": "am_michael", # American male
}

# Language defaults
DEFAULT_VOICES = {
    "pt": "FEMALE_BR_1",
    "pt-BR": "FEMALE_BR_1",
    "en": "FEMALE_US_1",
    "en-US": "FEMALE_US_1",
    "es": "FEMALE_ES_1",
    "es-ES": "FEMALE_ES_1",
}

def get_voice_name(voice_id: str, provider: str = "edge_tts") -> str:
    """
    Get actual voice name from voice ID

    Args:
        voice_id: Voice identifier (e.g., "FEMALE_BR_1")
        provider: TTS provider being used

    Returns:
        Actual voice name for the provider
    """
    # Try to get from environment first
    env_voice = os.environ.get(f"TTS_VOICE_{voice_id}")
    if env_voice:
        return env_voice

    # Get from mapping
    return VOICE_MAP.get(voice_id, voice_id)

def get_default_voice(language: str = "pt") -> str:
    """
    Get default voice ID for a language

    Args:
        language: Language code

    Returns:
        Default voice ID for the language
    """
    # Try to get from environment first
    env_default = os.environ.get(f"DEFAULT_VOICE_{language.upper()}")
    if env_default:
        return env_default

    return DEFAULT_VOICES.get(language, "FEMALE_BR_1")

def resolve_voice(voice_id: Optional[str], language: str = "pt", provider: str = "edge_tts") -> str:
    """
    Resolve voice ID to actual voice name, with fallback to language default

    Args:
        voice_id: Optional voice ID
        language: Language code
        provider: TTS provider

    Returns:
        Resolved voice name for the provider
    """
    if not voice_id:
        voice_id = get_default_voice(language)

    return get_voice_name(voice_id, provider)