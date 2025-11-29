#!/usr/bin/env python3
"""
Talker Class - Encapsulate conversation processing logic

Cloud-based API processing:
- STT: Groq Whisper API
- LLM: Groq Llama 3.1-8B / LiteLLM
- TTS: HTTP TTS Service (ElevenLabs, etc)

This abstraction keeps the main Orchestrator pipeline clean and simple.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class AbstractTalker(ABC):
    """
    Base class for conversation processing

    A Talker handles the complete conversation pipeline:
    1. Audio Input → Transcription
    2. Text → LLM Response
    3. Response Text → Audio Output
    """

    def __init__(self, name: str):
        self.name = name
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time_ms": 0
        }

    @abstractmethod
    async def initialize(self):
        """Initialize resources (models, API clients, etc)"""
        pass

    @abstractmethod
    async def process_turn(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process complete conversation turn

        Args:
            audio_data: Input audio bytes (PCM int16)
            sample_rate: Audio sample rate in Hz
            system_prompt: Optional system prompt for LLM
            conversation_history: Optional conversation context
            voice_id: Optional voice ID for TTS

        Returns:
            Dict with:
                - success: bool
                - transcript: str (user's speech)
                - text: str (AI response)
                - audio: bytes (AI response audio)
                - metrics: Dict (timing breakdown)
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        avg_time = (self.stats["total_time_ms"] / self.stats["total_calls"]
                   if self.stats["total_calls"] > 0 else 0)

        return {
            **self.stats,
            "average_time_ms": int(avg_time),
            "success_rate": (self.stats["successful_calls"] / self.stats["total_calls"]
                           if self.stats["total_calls"] > 0 else 0)
        }


class Talker(AbstractTalker):
    """
    Talker - Cloud API-based processing

    Pipeline:
    1. Audio → Groq Whisper API → Transcript
    2. Transcript → Groq Llama 3.1-8B / LiteLLM → Text Response
    3. Text Response → HTTP TTS Service → Audio Response

    Requires:
    - API keys for STT, LLM, and TTS services
    - All services are external (no GPU required)
    """

    def __init__(self, service_clients: Dict[str, Any]):
        super().__init__("Talker")
        self.clients = service_clients
        self.stt_client = None
        self.llm_client = None
        self.tts_client = None

    async def initialize(self):
        """Initialize API clients"""
        logger.info("🌐 Initializing Talker (Cloud APIs)...")

        try:
            # Get clients from service_clients dict
            self.stt_client = self.clients.get("stt")
            self.llm_client = self.clients.get("llm")
            self.tts_client = self.clients.get("tts")

            if not self.stt_client:
                raise RuntimeError("stt client not found")
            if not self.llm_client:
                raise RuntimeError("llm client not found")
            if not self.tts_client:
                raise RuntimeError("tts client not found")

            logger.info("✅ API clients ready")
            logger.info("   - STT: Groq Whisper")
            logger.info("   - LLM: Groq Llama / LiteLLM")
            logger.info("   - TTS: HTTP TTS Service")

            logger.info("🎉 Talker ready!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Talker: {e}")
            raise

    async def process_turn(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process turn using cloud APIs"""

        start_time = time.time()
        self.stats["total_calls"] += 1

        try:
            logger.info(f"🌐 Talker processing: {len(audio_data)} bytes @ {sample_rate}Hz")

            # ==========================================
            # STEP 1: Groq Whisper (Audio → Transcript)
            # ==========================================
            stt_start = time.time()

            transcript_result = await self.stt_client.transcribe(
                audio_data=audio_data,
                language="pt"  # Portuguese
            )

            transcript = transcript_result.get("text", "") if isinstance(transcript_result, dict) else transcript_result

            stt_time = (time.time() - stt_start) * 1000

            logger.info(f"📝 Groq Whisper: {transcript[:50]}... ({stt_time:.0f}ms)")

            # ==========================================
            # STEP 2: Groq Llama 3.1-8B (Text → Response)
            # ==========================================
            llm_start = time.time()

            response_text = await self.llm_client.generate(
                text=transcript,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                conversation_history=conversation_history or []
            )

            llm_time = (time.time() - llm_start) * 1000

            logger.info(f"🤖 Groq LLM: {response_text[:50]}... ({llm_time:.0f}ms)")

            # ==========================================
            # STEP 3: HTTP TTS (Text → Audio)
            # ==========================================
            tts_start = time.time()

            audio_response = await self.tts_client.synthesize(
                text=response_text,
                voice_id=voice_id or None,
                format="wav"
            )

            tts_time = (time.time() - tts_start) * 1000

            logger.info(f"🔊 HTTP TTS: {len(audio_response)} bytes ({tts_time:.0f}ms)")

            # ==========================================
            # STEP 4: Return Result
            # ==========================================
            total_time = (time.time() - start_time) * 1000

            self.stats["successful_calls"] += 1
            self.stats["total_time_ms"] += total_time

            logger.info(f"✅ Talker completed in {total_time:.0f}ms")

            return {
                "success": True,
                "transcript": transcript,
                "text": response_text,
                "audio": audio_response,
                "talker": "external",
                "metrics": {
                    "stt_time_ms": int(stt_time),
                    "llm_time_ms": int(llm_time),
                    "tts_time_ms": int(tts_time),
                    "total_time_ms": int(total_time),
                    "gpu_used": False
                }
            }

        except Exception as e:
            logger.error(f"❌ Talker error: {e}")
            self.stats["failed_calls"] += 1

            return {
                "success": False,
                "error": str(e),
                "talker": "external"
            }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Talker...")
        # HTTP clients are managed by orchestrator's session
        logger.info("✅ Talker cleanup complete")


class TalkerFactory:
    """
    Factory to create Talker instance

    Always creates Talker (all services are external, no GPU needed)
    """

    @staticmethod
    async def create_talker(
        service_clients: Dict[str, Any],
        **kwargs  # Accept but ignore legacy parameters (gpu_available, force_external)
    ) -> AbstractTalker:
        """
        Create Talker instance

        Args:
            service_clients: Dict of service clients (stt, llm, tts)
            **kwargs: Ignored (for backward compatibility)

        Returns:
            Initialized Talker instance
        """

        try:
            logger.info("🎯 Creating Talker (cloud APIs)")
            talker = Talker(service_clients)
            await talker.initialize()
            return talker
        except Exception as e:
            logger.error(f"❌ Failed to create Talker: {e}")
            raise
