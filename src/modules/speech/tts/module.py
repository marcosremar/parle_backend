"""
TTS Module - Direct Python calls for Text-to-Speech
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule
from .providers.manager import TTSProviderManager


class TTSModule(BaseModule):
    """TTS Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("tts")
        self.manager = None
        self.providers = {}
        self.default_provider = None
    
    async def _initialize(self) -> bool:
        """Initialize TTS providers"""
        try:
            self.manager = TTSProviderManager()
            
            # Get available providers
            self.providers = {}
            for provider_name in self.manager.available_providers:
                try:
                    provider = self.manager.get_provider(provider_name)
                    if provider and provider.available:
                        self.providers[provider_name] = provider
                except:
                    pass
            
            # Default is gtts if available
            self.default_provider = self.providers.get("gtts")
            
            self.logger.info(f"✅ TTS Module initialized with {len(self.providers)} providers")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TTS Module: {e}")
            return False
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        provider: Optional[str] = None,
        language: str = "pt",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            provider: Provider name (gtts, elevenlabs)
            language: Language code
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Dict with audio_base64, format, provider, voice_id
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get provider name
            provider_name = provider or "gtts"
            
            # Use manager to synthesize
            result = await self.manager.synthesize_speech(
                text=text,
                provider=provider_name,
                voice=voice_id or language,
                speed=speed,
                language=language
            )
            
            return {
                "audio_base64": result.get("audio_data"),
                "format": result.get("format", "wav"),
                "provider": result.get("provider", provider_name),
                "voice_id": result.get("voice"),
                "duration": result.get("duration"),
                "sample_rate": result.get("sample_rate")
            }
        except Exception as e:
            self.logger.error(f"❌ TTS synthesis failed: {e}")
            raise
    
    async def get_voices(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get available voices"""
        if not self.initialized:
            await self.initialize()
        
        provider_name = provider or "gtts"
        
        try:
            provider_instance = self.manager.get_provider(provider_name)
            if provider_instance and hasattr(provider_instance, "get_available_voices"):
                voices = provider_instance.get_available_voices()
                return {
                    "voices": voices,
                    "provider": provider_name
                }
        except:
            pass
        
        return {"voices": [], "provider": provider_name}
