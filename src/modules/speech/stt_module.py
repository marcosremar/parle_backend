"""
STT Module - Direct Python calls for Speech-to-Text
"""

import base64
import tempfile
import asyncio
from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class STTModule(BaseModule):
    """STT Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("stt")
        self.provider = None
        self.initialized = False
    
    async def _initialize(self) -> bool:
        """Initialize STT provider"""
        try:
            # Import Groq provider
            from src.services.stt.app_complete import GroqTranscriptionProvider
            
            self.provider = GroqTranscriptionProvider()
            self.initialized = True
            self.logger.info("✅ STT Module initialized with Groq provider")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize STT Module: {e}")
            # Try to initialize anyway (provider might be available later)
            self.initialized = True
            return True
    
    async def transcribe(
        self,
        audio_base64: Optional[str] = None,
        audio_url: Optional[str] = None,
        language: str = "pt",
        model: str = "whisper-large-v3"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_base64: Base64 encoded audio data
            audio_url: URL to audio file
            language: Language code (pt, en, etc.)
            model: Whisper model to use
            
        Returns:
            Dict with text, language, duration, model, provider
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Decode base64 audio if provided
            audio_data = None
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
            elif audio_url:
                # Download from URL
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(audio_url) as resp:
                        audio_data = await resp.read()
            
            if not audio_data:
                raise ValueError("Either audio_base64 or audio_url must be provided")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # Read audio data from temp file
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Transcribe using provider
                result = await self.provider.transcribe_audio(
                    audio_data=audio_bytes,
                    language=language,
                    model=model
                )
                
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", language),
                    "duration": result.get("duration"),
                    "model": model,
                    "provider": "groq"
                }
            finally:
                # Cleanup temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            raise
    
    async def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        return {
            "models": ["whisper-large-v3", "whisper-large-v2", "whisper-medium"],
            "provider": "groq"
        }
