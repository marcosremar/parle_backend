#!/usr/bin/env python3
"""
Pipeline de Conversação por TEXTO
Processa conversas com entrada de texto (sem áudio)
"""

import asyncio
from pathlib import Path
import time
import logging
from typing import Optional, Dict, Any
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getenv("ULTRAVOX_HOME", str(Path(__file__).parent.parent.parent.parent)))

from src.core.interfaces import IAudioProcessor, ITextToSpeech, IMemoryStore
from src.core.error_handling import (
    with_error_handling as handle_errors,
    validate_input,
    with_circuit_breaker,
    ProcessingError
)
from src.core.graceful_degradation import degradation_manager
from src.core.pipeline.streaming_handler import StreamingHandler
# DEPRECATED: modules.memory removed (replaced by conversation_store service)
# from modules.memory import SimpleMemoryStore
from src.core.managers.session_manager import session_manager, get_or_create_session
from src.core.context_manager import create_context_manager, get_context_manager

logger = logging.getLogger(__name__)


class TextConversationPipeline:
    """
    Pipeline de Conversação por TEXTO - Para entrada de texto (chat)

    Características:
    - Entrada: texto puro (sem áudio)
    - Processamento: Ultravox em modo texto
    - Saída: texto + áudio sintetizado (opcional)
    - Streaming com Pydantic

    NOTA: Para conversas por áudio, use AudioConversationPipeline
    """

    def __init__(self):
        # Services
        self.audio_processor: Optional[IAudioProcessor] = None  # Ultravox LLM
        self.tts: Optional[ITextToSpeech] = None
        self.memory_store: Optional[IMemoryStore] = None
        
        # Pipeline state
        self._initialized = False
        self.processing_queue = asyncio.Queue(maxsize=100)
        
        # Streaming handler
        self.streaming_handler: Optional[StreamingHandler] = None
        
        # Stats
        self.total_messages = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    async def setup(self) -> None:
        """Initialize pipeline components"""
        if self._initialized:
            return
            
        logger.info("🚀 Iniciando configuração da Text Pipeline...")

        # 1. Configure Ultravox (LLM para texto)
        logger.info("📡 Configurando módulo Ultravox (LLM)...")
        try:
            from src.services.llm.ultravox.ultravox_vllm import UltravoxVLLM
            self.audio_processor = UltravoxVLLM()
            logger.info("✅ Ultravox LLM configurado")
        except Exception as e:
            logger.error(f"❌ Falha ao configurar Ultravox: {e}")
            raise ProcessingError(f"Failed to setup Ultravox: {e}", component="ultravox")

        # 2. Configure Kokoro TTS (opcional)
        logger.info("🎵 Configurando módulo Kokoro TTS...")
        try:
            from src.services.tts.kokoro import KokoroTTS
            self.tts = KokoroTTS()
            logger.info("✅ Kokoro TTS configurado")
        except Exception as e:
            logger.warning(f"⚠️ TTS não disponível: {e}")
            self.tts = None
        
        # 3. Configure Memory Store
        logger.info("🧠 Configurando módulo de Memória...")
        self.memory_store = SimpleMemoryStore()
        
        # 4. Initialize all services
        logger.info("🔧 Initializing all services...")
        # Initialize Ultravox
        if hasattr(self.audio_processor, 'initialize'):
            logger.info("   Inicializando Ultravox...")
            if asyncio.iscoroutinefunction(self.audio_processor.initialize):
                await self.audio_processor.initialize()
            else:
                self.audio_processor.initialize()

        # Initialize TTS
        if self.tts and hasattr(self.tts, 'initialize'):
            logger.info("   Inicializando Kokoro TTS...")
            if asyncio.iscoroutinefunction(self.tts.initialize):
                await self.tts.initialize()
            else:
                self.tts.initialize()
        
        # Initialize memory store
        if hasattr(self.memory_store, 'initialize'):
            await self.memory_store.initialize()
        
        # 5. Initialize StreamingHandler
        if self.tts:
            logger.info("🎧 Initializing StreamingHandler for text pipeline...")
            self.streaming_handler = StreamingHandler(
                kokoro_tts=self.tts,
                websocket_callback=None
            )
            logger.info("✅ StreamingHandler initialized")
        
        self._initialized = True
        logger.info("✅ Text Pipeline configurada com sucesso!")
    
    @handle_errors(component="text_pipeline", operation="process_text")
    @validate_input(lambda x: isinstance(x, str) and len(x) > 0, "Invalid text input")
    async def process_text(self,
                          text: str,
                          session_id: str = "default",
                          voice_id: str = "pf_dora",
                          generate_audio: bool = True,
                          enable_streaming: bool = False) -> Dict[str, Any]:
        """
        Process text input through the pipeline
        
        Args:
            text: User text input
            session_id: Session identifier
            voice_id: Voice for TTS
            generate_audio: Whether to generate audio response
            enable_streaming: Whether to use streaming
            
        Returns:
            Response with text and optional audio
        """
        if not self._initialized:
            raise ProcessingError("Pipeline not initialized", component="text_pipeline")
        
        processing_start = time.time()
        self.total_messages += 1
        
        try:
            logger.info(f"📝 Processing text input: '{text[:50]}...'")
            
            # 1. Get or create session
            session = get_or_create_session(session_id)
            context_manager = get_context_manager(session_id)
            
            # 2. Build context
            context_data = await context_manager.build_context(
                user_message=text,
                include_rag=True
            )
            
            formatted_context = self._format_context_for_prompt(context_data)
            
            # 3. Process with Ultravox LLM
            logger.info("🧠 Processing with Ultravox LLM...")
            start_llm = time.time()
            
            # Use ask_text for text input
            llm_func = lambda: self.audio_processor.ask_text(
                text,
                user_prompt=formatted_context if formatted_context else None
            )
            
            fallback_response = await degradation_manager.execute_with_fallback(
                service_name="ultravox_llm",
                primary_func=llm_func,
                correlation_id=session_id
            )
            
            text_response = fallback_response.data if fallback_response.success else "Desculpe, não consegui processar sua mensagem."
            llm_time = (time.time() - start_llm) * 1000
            
            logger.info(f"🧠 LLM response ({llm_time:.0f}ms): {text_response[:50]}...")
            
            # 4. Update context and memory
            context_manager.add_message("user", text)
            context_manager.add_message("assistant", text_response)
            await context_manager.add_to_memory(text, text_response)
            
            # Save to session
            session_manager.add_interaction(
                session_id=session_id,
                user_message=text,
                assistant_response=text_response
            )
            
            # Save to memory store
            if self.memory_store:
                # Adicionar mensagem do usuário
                await self.memory_store.add_message(
                    session_id,
                    role="user",
                    content=text
                )
                # Adicionar resposta do assistente
                await self.memory_store.add_message(
                    session_id,
                    role="assistant",
                    content=text_response
                )
            
            # 5. Generate audio if requested
            audio_data = None
            tts_time = 0
            
            if generate_audio and self.tts:
                logger.info("🎵 Generating audio response...")
                
                if enable_streaming and self.streaming_handler:
                    # Use streaming handler
                    streaming_result = await self.streaming_handler.process_with_streaming(
                        llm_response=text_response,
                        session_id=session_id,
                        voice_id=voice_id,
                        enable_suggestions=False
                    )
                    
                    # Get combined audio from streaming
                    if streaming_result.get("audio_chunks"):
                        audio_data = b"".join(streaming_result["audio_chunks"])
                        tts_time = streaming_result.get("timing", {}).get("total_ms", 0)
                else:
                    # Regular TTS
                    start_tts = time.time()
                    audio_bytes = await self.tts.synthesize(text_response, voice_id=voice_id)
                    
                    if audio_bytes:
                        # Convert to base64 for JSON response
                        import base64
                        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
                        
                    tts_time = (time.time() - start_tts) * 1000
                
                logger.info(f"🎵 TTS complete ({tts_time:.0f}ms)")
            
            # Calculate total time
            total_time = (time.time() - processing_start) * 1000
            
            # Log timing
            logger.info(f"⏱️ Text Pipeline Timing: LLM={llm_time:.0f}ms TTS={tts_time:.0f}ms Total={total_time:.0f}ms")
            
            # Build response
            result = {
                "text": text_response,
                "session_id": session_id,
                "processing_time_ms": total_time,
                "timing": {
                    "llm_ms": llm_time,
                    "tts_ms": tts_time,
                    "total_ms": total_time
                }
            }
            
            if audio_data:
                result["audio_data"] = audio_data
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Error in text pipeline: {e}")
            raise ProcessingError(f"Text processing failed: {e}", component="text_pipeline")
    
    def _format_context_for_prompt(self, context_data: Dict[str, Any]) -> str:
        """Format context data for LLM prompt"""
        if not context_data or not context_data.get('context'):
            return ""
        
        parts = []
        
        # Add recent messages
        if context_data.get('recent_messages'):
            parts.append("Conversa recente:")
            for msg in context_data['recent_messages'][-3:]:  # Last 3 messages
                role = "Usuário" if msg['role'] == 'user' else "Assistente"
                parts.append(f"{role}: {msg['content'][:100]}...")
        
        # Add memories
        if context_data.get('memories'):
            parts.append("\nMemórias relevantes:")
            for memory in context_data['memories'][:2]:  # Top 2 memories
                parts.append(f"- {memory['content'][:100]}...")
        
        return "\n".join(parts) if parts else ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        uptime = time.time() - self.start_time
        return {
            "status": "operational" if self._initialized else "not_initialized",
            "total_messages": self.total_messages,
            "total_errors": self.total_errors,
            "error_rate": (self.total_errors / max(1, self.total_messages)) * 100,
            "uptime_seconds": uptime,
            "messages_per_minute": (self.total_messages / max(1, uptime)) * 60
        }
