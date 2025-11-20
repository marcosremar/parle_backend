#!/usr/bin/env python3
"""
Talker Classes - Encapsulate conversation processing logic

Two implementations:
1. InternalTalker - GPU-based local processing (Ultravox multimodal + HTTP TTS)
2. ExternalTalker - Cloud-based API processing (Groq Whisper + Llama 3.1-8B + HTTP TTS)

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


class InternalTalker(AbstractTalker):
    """
    Internal Talker - GPU-based local processing

    Pipeline:
    1. Audio → Ultravox Universal (multimodal: STT + LLM in one) → Text Response
    2. Text Response → HTTP TTS Service → Audio Response

    Requires:
    - GPU available (checked via torch.cuda.is_available())
    - Ultravox Universal module loaded in-process
    - HTTP TTS service available

    Performance: Ultra-low latency (0 HTTP overhead)
    """

    def __init__(self):
        super().__init__("InternalTalker")
        self.ultravox = None  # Ultravox Universal (multimodal)
        self.gpu_available = False

    async def initialize(self):
        """Initialize in-process modules (Ultravox)"""
        logger.info("🚀 Initializing InternalTalker (GPU-based)...")

        try:
            # Check GPU availability
            import torch
            self.gpu_available = torch.cuda.is_available()

            if not self.gpu_available:
                raise RuntimeError("GPU not available - InternalTalker requires GPU")

            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

            # Load Ultravox Universal (multimodal: audio → text response)
            from src.services.llm.ultravox.ultravox_universal import UltravoxUniversal
            self.ultravox = UltravoxUniversal()
            await self.ultravox.initialize()
            logger.info("✅ Ultravox Universal loaded (multimodal)")

            # TTS will use HTTP service
            logger.info("✅ TTS will use HTTP service")

            logger.info("🎉 InternalTalker ready - ultra-low latency mode enabled!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize InternalTalker: {e}")
            raise

    async def process_turn(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process turn using GPU-based local models"""

        start_time = time.time()
        self.stats["total_calls"] += 1

        try:
            logger.info(f"🎤 InternalTalker processing: {len(audio_data)} bytes @ {sample_rate}Hz")

            # ==========================================
            # STEP 1: Ultravox Universal (Audio → Text Response)
            # ==========================================
            stt_llm_start = time.time()

            # Convert bytes to numpy float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            logger.debug(f"🔄 Audio conversion: {len(audio_data)} bytes → {len(audio_array)} samples")

            # Call Ultravox (does STT + LLM in one pass)
            ultravox_result = await self.ultravox.process_audio(
                audio_array=audio_array,
                sample_rate=sample_rate,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                conversation_history=conversation_history or []
            )

            transcript = ultravox_result.get("transcript", "")
            response_text = ultravox_result.get("text", "")

            stt_llm_time = (time.time() - stt_llm_start) * 1000

            logger.info(f"🤖 Ultravox: {transcript[:50]}... → {response_text[:50]}... ({stt_llm_time:.0f}ms)")

            # ==========================================
            # STEP 2: HTTP TTS (Text → Audio)
            # ==========================================
            tts_start = time.time()

            # Use HTTP TTS service
            from src.services.orchestrator.service_clients import TTSClient
            tts_client = TTSClient()
            audio_response = await tts_client.synthesize(
                text=response_text,
                voice_id=voice_id or None
            )

            tts_time = (time.time() - tts_start) * 1000

            logger.info(f"🔊 HTTP TTS: {len(audio_response)} bytes ({tts_time:.0f}ms)")

            # ==========================================
            # STEP 3: Return Result
            # ==========================================
            total_time = (time.time() - start_time) * 1000

            self.stats["successful_calls"] += 1
            self.stats["total_time_ms"] += total_time

            logger.info(f"✅ InternalTalker completed in {total_time:.0f}ms")

            return {
                "success": True,
                "transcript": transcript,
                "text": response_text,
                "audio": audio_response,
                "talker": "internal",
                "metrics": {
                    "stt_llm_time_ms": int(stt_llm_time),
                    "tts_time_ms": int(tts_time),
                    "total_time_ms": int(total_time),
                    "gpu_used": True
                }
            }

        except Exception as e:
            logger.error(f"❌ InternalTalker error: {e}")
            self.stats["failed_calls"] += 1

            return {
                "success": False,
                "error": str(e),
                "talker": "internal"
            }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up InternalTalker...")
        # Models are in-process, no cleanup needed
        logger.info("✅ InternalTalker cleanup complete")


class ExternalTalker(AbstractTalker):
    """
    External Talker - Cloud API-based processing

    Pipeline:
    1. Audio → Groq Whisper API → Transcript
    2. Transcript → Groq Llama 3.1-8B → Text Response
    3. Text Response → HTTP TTS Service → Audio Response

    Requires:
    - GROQ_API_KEY for Whisper + LLM
    - HTTP TTS service available

    Performance: Higher latency (HTTP overhead) but no GPU required
    """

    def __init__(self, service_clients: Dict[str, Any]):
        super().__init__("ExternalTalker")
        self.clients = service_clients
        self.stt_client = None
        self.llm_client = None
        self.tts_client = None

    async def initialize(self):
        """Initialize API clients"""
        logger.info("🌐 Initializing ExternalTalker (Cloud APIs)...")

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

            logger.info("✅ External API clients ready")
            logger.info("   - STT: Groq Whisper")
            logger.info("   - LLM: Groq Llama 3.1-8B")
            logger.info("   - TTS: HTTP TTS Service")

            logger.info("🎉 ExternalTalker ready!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ExternalTalker: {e}")
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
            logger.info(f"🌐 ExternalTalker processing: {len(audio_data)} bytes @ {sample_rate}Hz")

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

            logger.info(f"✅ ExternalTalker completed in {total_time:.0f}ms")

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
            logger.error(f"❌ ExternalTalker error: {e}")
            self.stats["failed_calls"] += 1

            return {
                "success": False,
                "error": str(e),
                "talker": "external"
            }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up ExternalTalker...")
        # HTTP clients are managed by orchestrator's session
        logger.info("✅ ExternalTalker cleanup complete")


class TalkerFactory:
    """
    Factory to create the appropriate Talker based on GPU profile

    Decision logic:
    - If GPU available + gpu_machine profile: InternalTalker
    - Otherwise: ExternalTalker
    """

    @staticmethod
    async def create_talker(
        gpu_available: bool,
        service_clients: Dict[str, Any],
        force_external: bool = False
    ) -> AbstractTalker:
        """
        Create appropriate Talker instance

        Args:
            gpu_available: Whether GPU is available
            service_clients: Dict of service clients for ExternalTalker
            force_external: Force ExternalTalker (for testing)

        Returns:
            Initialized Talker instance
        """

        try:
            if gpu_available and not force_external:
                logger.info("🎯 Creating InternalTalker (GPU available)")
                talker = InternalTalker()
            else:
                if force_external:
                    logger.info("🎯 Creating ExternalTalker (forced by flag)")
                else:
                    logger.info("🎯 Creating ExternalTalker (GPU not available)")
                talker = ExternalTalker(service_clients)

            await talker.initialize()
            return talker
        except Exception as e:
            logger.error(f"❌ Failed to create Talker: {e}")
            raise
