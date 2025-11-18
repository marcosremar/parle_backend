#!/usr/bin/env python3
"""
Profile Validator - Validates if services can execute in current profile

Provides:
- Profile checking for GPU-dependent services
- Standard responses when service cannot execute
- Pre-generated mock data for testing
"""

import logging
import base64
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfileValidator:
    """Validates service execution against active profile"""

    # Pre-generated base64 audio for TTS mock response (silence, 1 second, 16kHz)
    MOCK_AUDIO_BASE64 = (
        "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQAAAAA="
    )

    @staticmethod
    def can_use_gpu() -> bool:
        """
        Check if current profile allows GPU usage

        Returns:
            True if GPU services can run locally, False otherwise
        """
        try:
            from src.core.managers.service_manager import ServiceManager

            # Try to get active profile
            config_path = Path("config/profiles.yaml")
            if not config_path.exists():
                logger.warning("⚠️  profiles.yaml not found, assuming GPU available")
                return True

            import yaml
            with open(config_path) as f:
                profiles_config = yaml.safe_load(f)

            active_profile = profiles_config.get("active_profile", "local")
            profiles = profiles_config.get("profiles", {})

            if active_profile not in profiles:
                logger.warning(f"⚠️  Profile '{active_profile}' not found, assuming GPU available")
                return True

            profile = profiles[active_profile]
            restrictions = profile.get("restrictions", {})

            # Check if GPU services are allowed locally
            allow_gpu_services = restrictions.get("allow_gpu_services", True)
            require_gpu_remote = restrictions.get("require_gpu_services_remote", False)

            # GPU is usable if:
            # 1. GPU services are allowed AND
            # 2. They are NOT required to be remote
            can_use = allow_gpu_services and not require_gpu_remote

            logger.info(f"🔍 Profile '{active_profile}': GPU available = {can_use}")
            return can_use

        except Exception as e:
            logger.error(f"❌ Error checking GPU availability: {e}")
            # Default to allowing GPU if we can't check
            return True

    @staticmethod
    def get_active_profile() -> str:
        """Get the currently active profile name"""
        try:
            config_path = Path("config/profiles.yaml")
            if not config_path.exists():
                return "unknown"

            import yaml
            with open(config_path) as f:
                profiles_config = yaml.safe_load(f)

            return profiles_config.get("active_profile", "unknown")

        except Exception as e:
            logger.error(f"❌ Error getting active profile: {e}")
            return "unknown"

    @staticmethod
    def get_standard_stt_response(audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Get standard STT response when GPU not available

        Args:
            audio_data: Optional audio data (unused, for signature compatibility)

        Returns:
            Standard STT response
        """
        profile = ProfileValidator.get_active_profile()

        return {
            "success": False,
            "error": "GPU_NOT_AVAILABLE",
            "message": f"STT service não disponível no perfil '{profile}' (sem GPU local)",
            "transcription": "[STT indisponível - sem GPU]",
            "profile": profile,
            "mock_response": True,
            "suggestion": "Use perfil 'gpu-machine' ou serviços externos (external_stt)"
        }

    @staticmethod
    def get_standard_tts_response(text: Optional[str] = None) -> Dict[str, Any]:
        """
        Get standard TTS response when GPU not available

        Returns pre-recorded audio in base64 format

        Args:
            text: Optional text to synthesize (unused, for signature compatibility)

        Returns:
            Standard TTS response with mock audio
        """
        profile = ProfileValidator.get_active_profile()

        return {
            "success": False,
            "error": "GPU_NOT_AVAILABLE",
            "message": f"TTS service não disponível no perfil '{profile}' (sem GPU local)",
            "audio_base64": ProfileValidator.MOCK_AUDIO_BASE64,
            "audio_format": "wav",
            "sample_rate": 16000,
            "duration_ms": 1000,
            "profile": profile,
            "mock_response": True,
            "suggestion": "Use perfil 'gpu-machine' ou serviços externos (external_tts)",
            "text_requested": text or "[Nenhum texto fornecido]"
        }

    @staticmethod
    def get_standard_llm_response(prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Get standard LLM response when GPU not available

        Args:
            prompt: Optional prompt (unused, for signature compatibility)

        Returns:
            Standard LLM response
        """
        profile = ProfileValidator.get_active_profile()

        return {
            "success": False,
            "error": "GPU_NOT_AVAILABLE",
            "message": f"LLM service não disponível no perfil '{profile}' (sem GPU local)",
            "response": "Desculpe, o modelo de linguagem não está disponível neste servidor. Por favor, use o perfil 'gpu-machine' ou configure serviços externos.",
            "model": "none",
            "profile": profile,
            "mock_response": True,
            "suggestion": "Use perfil 'gpu-machine' ou serviços externos (external_llm)",
            "prompt_requested": prompt or "[Nenhum prompt fornecido]"
        }

    @staticmethod
    def should_service_run(service_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a service should run in the current profile

        Args:
            service_name: Service identifier (e.g., 'stt', 'tts', 'llm')

        Returns:
            Tuple of (should_run: bool, reason: Optional[str])
        """
        gpu_services = ['stt', 'tts', 'llm']

        if service_name not in gpu_services:
            # Non-GPU service, always allow
            return True, None

        # GPU service - check profile
        can_use = ProfileValidator.can_use_gpu()

        if not can_use:
            profile = ProfileValidator.get_active_profile()
            reason = f"Service '{service_name}' requer GPU, mas perfil '{profile}' não permite GPU local"
            return False, reason

        return True, None


# Convenience functions for quick checks
def can_use_gpu() -> bool:
    """Check if GPU is available in current profile"""
    return ProfileValidator.can_use_gpu()


def get_stt_mock_response(audio_data: Optional[bytes] = None) -> Dict[str, Any]:
    """Get mock STT response (no GPU)"""
    return ProfileValidator.get_standard_stt_response(audio_data)


def get_tts_mock_response(text: Optional[str] = None) -> Dict[str, Any]:
    """Get mock TTS response with pre-recorded audio (no GPU)"""
    return ProfileValidator.get_standard_tts_response(text)


def get_llm_mock_response(prompt: Optional[str] = None) -> Dict[str, Any]:
    """Get mock LLM response (no GPU)"""
    return ProfileValidator.get_standard_llm_response(prompt)
