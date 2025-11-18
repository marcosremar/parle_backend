#!/usr/bin/env python3
"""
Binary Conversation Pipeline
Pipeline otimizada com comunicação HTTP binária para Ultravox e Kokoro
"""

import asyncio
import aiohttp
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from pathlib import Path

# Imports específicos
from src.core.http_clients.binary_client import UltravoxHTTPClient, KokoroHTTPClient
from src.services.external_stt.transcription.groq_transcription import GroqTranscription
from src.core.conversational_context.context_manager import ConversationalContext
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)

class InputType(Enum):
    """Input type enumeration"""
    AUDIO = "audio"
    TEXT = "text"

class BinaryConversationPipeline:
    """
    Pipeline de conversação com comunicação HTTP binária
    Ultravox e Kokoro rodam como serviços HTTP isolados
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize binary conversation pipeline

        Args:
            device: Device for local models (cuda/cpu)
        """
        self.device = device
        self.is_initialized = False

        # Clientes HTTP binários
        self.ultravox_client = None
        self.kokoro_client = None

        # Componentes locais
        self.stt = None  # Speech-to-Text (Groq) - local
        self.conversation_context = None  # Context manager - local

        # Session management
        self.session_id = "binary_pipeline_session"

        # Configurações de endpoints
        self.ultravox_url = "http://127.0.0.1:8100"
        self.kokoro_url = "http://127.0.0.1:8101"

        # Load configuration from settings
        from core.settings import get_settings
        settings = get_settings()

        self.config = {
            'language': settings.pipeline.language,
            'voice_id': settings.pipeline.voice_id,
            'max_tokens': 512,
            'temperature': 0.7
        }

        logger.info(f"📝 📋 Loaded binary pipeline config: language={self.config['language']}, voice={self.config['voice_id']}")

    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("📝 🚀 Initializing Binary Conversation Pipeline...")

        # 1. Inicializar contexto conversacional (local)
        logger.info("📝 🧠 Initializing advanced conversation context...")
        from src.core.conversational_context.context_manager import ConversationalContext
        self.conversation_context = ConversationalContext()
        logger.info("📝 ✅ Advanced conversation context initialized")

        # 2. Inicializar STT (local - Groq API)
        logger.info("📝 🎤 Initializing Groq STT...")
        from src.services.external_stt.transcription.groq_transcription import GroqTranscription
        self.stt = GroqTranscription()
        await self.stt.initialize()
        logger.info("📝 ✅ Groq STT initialized")

        # 3. Inicializar clientes HTTP binários
        logger.info("📝 🔗 Initializing HTTP binary clients...")

        # Cliente Ultravox
        self.ultravox_client = UltravoxHTTPClient(self.ultravox_url)

        # Cliente Kokoro
        self.kokoro_client = KokoroHTTPClient(self.kokoro_url)

        logger.info("📝 ✅ HTTP binary clients initialized")

        # 4. Verificar saúde dos serviços
        await self._health_check()

        self.is_initialized = True
        logger.info("📝 ✅ Binary Conversation Pipeline initialized successfully!")

    async def _health_check(self):
        """Verificar saúde dos serviços HTTP"""
        logger.info("📝 🏥 Checking services health...")

        try:
            # Check Ultravox
            async with self.ultravox_client as client:
                ultravox_health = await client.health_check()
                logger.info(f"📝 🤖 Ultravox status: {ultravox_health.get('status', 'unknown')}")

            # Check Kokoro
            async with self.kokoro_client as client:
                kokoro_health = await client.health_check()
                logger.info(f"📝 🎵 Kokoro status: {kokoro_health.get('status', 'unknown')}")

            logger.info("📝 ✅ All services healthy")

        except Exception as e:
            logger.warning(f"📝 ⚠️ Service health check failed: {e}")

    async def process_conversation(
        self,
        input_data: Union[str, bytes, np.ndarray],
        input_type: InputType,
        session_id: Optional[str] = None,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process conversation input with binary HTTP communication

        Args:
            input_data: Text string or audio data
            input_type: Type of input (text or audio)
            session_id: Session identifier
            voice_id: Voice ID for TTS

        Returns:
            Dict with response text, audio, and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")

        start_time = time.time()
        session_id = session_id or self.session_id
        voice_id = voice_id or self.config['voice_id']

        logger.info(f"📝 🎯 Processing {input_type.value} input (session: {session_id})")

        try:
            # Step 1: Convert input to text
            if input_type == InputType.TEXT:
                input_text = input_data
            elif input_type == InputType.AUDIO:
                # Transcribe audio using local Groq STT
                input_text = await self._transcribe_audio(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

            logger.info(f"📝 📝 Input text: '{input_text[:100]}...'")

            # Step 2: Get conversation context
            enhanced_prompt = await self.conversation_context.get_enhanced_prompt(
                input_text,
                session_id
            )

            # Step 3: Generate response using Ultravox HTTP binary
            response_text = await self._generate_response_binary(
                enhanced_prompt,
                session_id
            )

            logger.info(f"📝 💬 Response: '{response_text[:100]}...'")

            # Step 4: Update conversation context
            await self.conversation_context.add_interaction(
                session_id,
                input_text,
                response_text
            )

            # Step 5: Generate audio using Kokoro HTTP binary
            audio_output = await self._generate_audio_binary(
                response_text,
                voice_id,
                session_id
            )

            processing_time = (time.time() - start_time) * 1000

            # Prepare response
            result = {
                'success': True,
                'input_text': input_text,
                'response': response_text,
                'audio_output': audio_output,
                'session_id': session_id,
                'voice_id': voice_id,
                'processing_time_ms': processing_time,
                'pipeline_type': 'binary_http',
                'input_type': input_type.value,
                'metrics': {
                    'total_time_ms': processing_time,
                    'audio_size_bytes': len(audio_output) if audio_output else 0
                }
            }

            logger.info(f"📝 ✅ Conversation processed successfully ({processing_time:.2f}ms)")
            return result

        except Exception as e:
            logger.error(f"📝 ❌ Error in conversation processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'processing_time_ms': (time.time() - start_time) * 1000
            }

    async def _transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Transcribe audio using local Groq STT"""
        logger.info("📝 🎤 Transcribing audio with Groq STT...")

        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            # Convert to bytes (this is a simplified conversion)
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = audio_data

        transcript = await self.stt.transcribe_audio(audio_bytes)
        logger.info(f"📝 📝 Transcript: '{transcript[:100]}...'")
        return transcript

    async def _generate_response_binary(
        self,
        prompt: str,
        session_id: str
    ) -> str:
        """Generate response using Ultravox HTTP binary client"""
        logger.info("📝 🤖 Generating response with Ultravox (binary HTTP)...")

        try:
            async with self.ultravox_client as client:
                response = await client.generate_text(
                    text=prompt,
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature'],
                    session_id=session_id
                )

                logger.info(f"📝 ✅ Ultravox response generated: {len(response)} characters")
                return response

        except Exception as e:
            logger.error(f"📝 ❌ Ultravox binary generation failed: {e}")
            # Fallback response
            return "Desculpe, não consegui processar sua solicitação no momento."

    async def _generate_audio_binary(
        self,
        text: str,
        voice_id: str,
        session_id: str
    ) -> bytes:
        """Generate audio using Kokoro HTTP binary client"""
        logger.info(f"📝 🎵 Generating audio with Kokoro (binary HTTP, voice: {voice_id})...")

        try:
            async with self.kokoro_client as client:
                audio_data = await client.synthesize_speech(
                    text=text,
                    voice_id=voice_id,
                    session_id=session_id
                )

                logger.info(f"📝 ✅ Kokoro audio generated: {len(audio_data)} bytes")
                return audio_data

        except Exception as e:
            logger.error(f"📝 ❌ Kokoro binary synthesis failed: {e}")
            # Return empty audio on failure
            return b''

    async def warmup(self):
        """Warm up the pipeline with test requests"""
        logger.info("📝 🔥 Warming up binary pipeline...")

        try:
            # Test text processing
            test_result = await self.process_conversation(
                "Olá, este é um teste do pipeline binário.",
                InputType.TEXT,
                session_id="warmup_session"
            )

            if test_result.get('success'):
                logger.info("📝 ✅ Binary pipeline warmup successful")
            else:
                logger.warning(f"📝 ⚠️ Binary pipeline warmup warning: {test_result.get('error')}")

        except Exception as e:
            logger.warning(f"📝 ⚠️ Binary pipeline warmup failed: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        metrics = {
            'pipeline_type': 'binary_http',
            'is_initialized': self.is_initialized,
            'device': self.device,
            'config': self.config,
            'services': {
                'ultravox_url': self.ultravox_url,
                'kokoro_url': self.kokoro_url,
                'stt': 'groq_local',
                'context': 'local'
            }
        }

        # Get service health if possible
        try:
            if self.ultravox_client:
                async with self.ultravox_client as client:
                    metrics['ultravox_health'] = await client.health_check()

            if self.kokoro_client:
                async with self.kokoro_client as client:
                    metrics['kokoro_health'] = await client.health_check()

        except Exception as e:
            metrics['health_check_error'] = str(e)

        return metrics

    def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("📝 🧹 Cleaning up binary pipeline...")

        # Close HTTP clients if needed
        # (aiohttp clients are closed automatically in context managers)

        self.is_initialized = False
        logger.info("📝 ✅ Binary pipeline cleaned up")