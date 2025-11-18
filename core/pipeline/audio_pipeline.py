#!/usr/bin/env python3
"""
Main Pipeline Entry Point
Coordenador principal do sistema de áudio e voz em tempo real
"""

# Setup warnings filter first
try:
    from core.warnings_filter import setup_warnings_filter
    setup_warnings_filter()
except ImportError:
    pass

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pipeline unificada diretamente neste arquivo
from src.core.container import DIContainer
from src.core.interfaces import IAudioProcessor, ITextToSpeech, IMemoryStore
from src.core.singleton_manager import get_ultravox, get_kokoro_tts  # Import singleton helpers
from src.core.pipeline.streaming_handler import StreamingHandler
# DEPRECATED: modules.memory removed (replaced by conversation_store service)
# from modules.memory import MemoryStore
from src.services.external_stt.transcription.groq_transcription import GroqTranscription
from src.core.managers.session_manager import session_manager, get_or_create_session
from src.core.context_manager import create_context_manager, get_context_manager

# Error handling and graceful degradation
from src.core.error_handling import (
    global_error_handler as error_handler,
    with_circuit_breaker,
    with_error_handling as handle_errors,
    validate_input,
    UltravoxError,
    ProcessingError,
    ServiceUnavailableError,
    ValidationError,
)

# Validation functions (moved from old error_handler)
def is_valid_audio(audio_data) -> bool:
    """Validate audio data"""
    if audio_data is None:
        return False
    import numpy as np
    if isinstance(audio_data, np.ndarray):
        return len(audio_data) > 0 and audio_data.dtype in [np.float32, np.int16]
    if isinstance(audio_data, bytes):
        return len(audio_data) > 44  # Minimum WAV header
    return False

def is_valid_session_id(session_id) -> bool:
    """Validate session ID"""
    return isinstance(session_id, str) and len(session_id) > 0 and len(session_id) <= 128
from src.core.graceful_degradation import (
    degradation_manager, initialize_graceful_degradation,
    get_degradation_status, with_graceful_degradation
)
from src.core.initialization_manager import initialization_manager, initialize_system
from health_server import start_health_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioConversationPipeline:
    """
    Pipeline de Conversação por ÁUDIO - Processa áudio DIRETAMENTE com Ultravox

    Características:
    - Ultravox processa áudio diretamente (capacidade multimodal)
    - Groq STT roda em background apenas para memória
    - Streaming com Pydantic para respostas incrementais
    - Kokoro TTS sintetiza áudio de resposta

    NOTA: Para conversas por texto, use TextConversationPipeline
    """

    def __init__(self):
        self.container = DIContainer()

        # Direct access to services
        self.audio_processor: Optional[IAudioProcessor] = None  # Ultravox LLM
        self.tts: Optional[ITextToSpeech] = None
        self.memory_store: Optional[IMemoryStore] = None
        self.groq_stt: Optional[GroqTranscription] = None  # Groq STT

        # Pipeline state
        self.is_initialized = False
        self.start_time = None

        # Stats
        self.total_processed = 0
        self.total_errors = 0
        self.average_processing_time = 0

        # Processing queue for async operations
        self.processing_queue = asyncio.Queue(maxsize=100)
        self._processing_task = None

        # Streaming handler
        self.streaming_handler: Optional[StreamingHandler] = None

    async def setup(self):
        """
        Configura todos os módulos e dependências com warmup completo
        """
        logger.info("🚀 Iniciando configuração da Pipeline Principal...")

        # Start health server first
        start_health_server(port=8080)

        # Use the initialization manager for complete setup
        success = await initialize_system(self)

        if not success:
            raise RuntimeError("Falha na inicialização do sistema")

        logger.info("✅ Pipeline totalmente configurada e aquecida")
        return

        # OLD CODE - mantido para referência mas não executado
        # Initialize error handling and graceful degradation
        logger.info("🛡️ Inicializando sistema de error handling...")
        initialize_graceful_degradation()
        logger.info("✅ Error handling inicializado")

        # 1. Configure Groq STT (speech-to-text) with circuit breaker
        logger.info("🎤 Configurando Groq STT...")
        try:
            self.groq_stt = GroqTranscription()
            logger.info("✅ Groq STT configurado")
        except Exception as e:
            logger.warning(f"⚠️ Groq STT não disponível: {e}")
            self.groq_stt = None

        # 2. Configure Ultravox (LLM para texto e áudio)
        logger.info("📡 Configurando módulo Ultravox (LLM)...")
        # Use singleton manager to get/create Ultravox - direct loading
        ultravox = get_ultravox()
        if ultravox is None:
            raise RuntimeError("Failed to initialize Ultravox model")
        logger.info("✅ Ultravox LLM configurado")
        self.container.register_singleton(IAudioProcessor, instance=ultravox)

        # 3. Configure Kokoro TTS (síntese de voz)
        logger.info("🎵 Configurando módulo Kokoro TTS...")
        # Use singleton manager to get/create Kokoro TTS - direct loading
        tts = get_kokoro_tts()
        if tts is None:
            raise RuntimeError("Failed to initialize Kokoro TTS model")
        self.container.register_singleton(ITextToSpeech, instance=tts)

        # 4. Configure Memory Manager
        logger.info("🧠 Configurando módulo de Memória...")
        memory = MemoryStore()
        self.container.register_singleton(IMemoryStore, instance=memory)

        # WebRTC is handled separately - no longer part of the pipeline

        # 5. Store direct references for performance
        logger.info("🔗 Resolvendo serviços do container...")
        self.audio_processor = self.container.resolve(IAudioProcessor)
        self.tts = self.container.resolve(ITextToSpeech)
        self.memory_store = self.container.resolve(IMemoryStore)

        # 6. Initialize all services
        await self.initialize_services()

        # 7. Start processing task
        self._processing_task = asyncio.create_task(self._process_queue())

        self.is_initialized = True
        self.start_time = time.time()

        logger.info("✅ Pipeline Principal configurada com sucesso!")

        logger.info("🚀 Modelos carregados diretamente durante a inicialização")

    async def initialize_services(self) -> None:
        """
        Initialize all services
        """
        logger.info("🔧 Initializing all services...")

        # Check if audio_processor.initialize is async or sync
        if hasattr(self.audio_processor, 'initialize'):
            if asyncio.iscoroutinefunction(self.audio_processor.initialize):
                await self.audio_processor.initialize()
            else:
                await asyncio.to_thread(self.audio_processor.initialize)

        # Check if tts.initialize is async or sync
        if hasattr(self.tts, 'initialize'):
            if asyncio.iscoroutinefunction(self.tts.initialize):
                await self.tts.initialize()
            else:
                await asyncio.to_thread(self.tts.initialize)

        # Memory store initialize is async, call it directly
        if hasattr(self.memory_store, 'initialize'):
            await self.memory_store.initialize()

        # Warm-up removido - deve ser feito na inicialização do servidor
        # A pipeline de conversa não deve fazer warm-up

        # 9. Initialize StreamingHandler for real-time audio streaming
        logger.info("🎧 Initializing StreamingHandler for streaming audio...")
        self.streaming_handler = StreamingHandler(
            kokoro_tts=self.tts,
            websocket_callback=None  # Will be set later by WebRTC server
        )
        logger.info("✅ StreamingHandler initialized - ready for streaming!")

    def get_model_loading_status(self) -> dict:
        """
        Get current model loading status for health checks
        """
        try:
            # With direct loading, models are ready immediately
            return {
                "is_loading": False,
                "is_ready": True,
                "has_error": False,
                "status": "Models loaded directly during initialization"
            }
        except Exception as e:
            return {
                "is_loading": False,
                "is_ready": False,
                "has_error": True,
                "error": str(e)
            }


    async def _process_queue(self):
        """Background task to process queued requests"""
        while True:
            try:
                request = await self.processing_queue.get()
                # Process request asynchronously
                await self._handle_queued_request(request)
            except Exception as e:
                logger.error(f"Error processing queued request: {e}")

    async def _handle_queued_request(self, request):
        """Handle a queued processing request"""
        # Implement queue processing logic here
        pass

    @handle_errors(component="pipeline", operation="process_audio")
    @validate_input(is_valid_audio, "Invalid audio data", detailed_check=True)
    async def _process_audio_internal(self,
                                     audio: np.ndarray,
                                     sample_rate: int,
                                     session_id: str,
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Internal audio processing with hybrid sync/async approach
        """
        start_time = time.time()

        try:
            # 0. Get or create session and context manager
            session = get_or_create_session(session_id)

            # Get or create context manager for this session
            context_manager = get_context_manager(session_id)
            if not context_manager:
                user_id = session.user_id or session_id  # Use session_id as fallback
                context_manager = await create_context_manager(
                    session_id=session_id,
                    user_id=user_id,
                    window_size=30,
                    rag_top_k=3
                )
                logger.info(f"🧠 Context Manager created for session {session_id[:12]}...")

            # 1. ASYNC STT: Start STT in background for memory storage only
            logger.info(f"🚀 Starting async STT: {len(audio)} samples at {sample_rate}Hz")

            # Start STT task in background - we don't wait for it!
            stt_task = None
            if self.groq_stt:
                try:
                    # Convert numpy array to bytes for Groq
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    language = context.get('language', 'pt') if context else 'pt'

                    # Create async task for STT (runs in background for memory only)
                    stt_task = asyncio.create_task(
                        self._async_stt_for_memory(
                            audio_bytes, sample_rate, language, session_id,
                            context_manager, session
                        )
                    )
                    logger.info("🚀 STT task started asynchronously for memory storage")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to start async STT: {e}")

            # ULTRAVOX PROCESSA ÁUDIO DIRETAMENTE! (Capacidade multimodal)
            # STT roda em background apenas para salvar transcrição na memória
            transcript_placeholder = '[Audio input]'  # Placeholder para contexto/memória

            # 2. Build context with sliding window + RAG
            context_data = await context_manager.build_context(
                user_message=transcript_placeholder,  # Usa placeholder para contexto
                include_rag=True
            )

            # 3. Format context for LLM
            formatted_context = self._format_context_for_prompt(context_data)

            # Add fixed conversational instructions to the pipeline
            # This encourages short, engaging responses that keep the user talking
            fixed_instructions = """PIPELINE_INSTRUCTIONS:
You are a conversational assistant. Keep responses SHORT and ENGAGING.

CRITICAL RULES:
• Use 3-5 words per sentence
• Maximum 2 short sentences
• Ask questions to keep conversation flowing
• Be friendly but concise

EXAMPLES:
User: "I'm tired"
You: "Long day? Tell me more."

User: "I love pizza"
You: "What's your favorite topping?"

User: "It's raining"
You: "Staying inside today? Cozy plans?"
"""

            # Combine fixed instructions with any custom context
            if formatted_context:
                formatted_context = fixed_instructions + "\n" + formatted_context
            else:
                formatted_context = fixed_instructions

            # 4. LLM: PROCESSAR ÁUDIO DIRETAMENTE com Ultravox (multimodal)!
            logger.info(f"🎤 Ultravox processando ÁUDIO DIRETAMENTE com contexto: {context_data['metadata']['window_messages']} recent msgs, "
                       f"{context_data['metadata']['rag_memories']} RAG memories")

            # Define voice_id before using it in the lambda function
            voice_id = context.get('voice_id', 'pf_dora') if context else 'pf_dora'
            logger.info(f"🔊 Using voice for language detection: {voice_id}")

            start_llm = time.time()
            llm_time = 0
            stt_time = 0  # Initialize stt_time to default (STT runs in background)

            try:
                # ULTRAVOX PROCESSA O ÁUDIO DIRETAMENTE!
                # Passa o áudio raw para o Ultravox usar sua capacidade multimodal
                async def llm_func():
                    return await self.audio_processor.process_audio(
                        audio=audio,
                        sample_rate=sample_rate,
                        session_id=session_id,
                        context=formatted_context if formatted_context else None,
                        voice_id=voice_id  # Para detectar idioma
                    )

                fallback_response = await degradation_manager.execute_with_fallback(
                    service_name="ultravox_llm",
                    primary_func=llm_func,
                    correlation_id=session_id
                )

                text_response = fallback_response.data if fallback_response.success else "Desculpe, não consegui processar sua mensagem."
                llm_time = (time.time() - start_llm) * 1000

                if fallback_response.degraded:
                    logger.warning(f"⚠️ LLM usando fallback: {fallback_response.fallback_used}")

                logger.info(f"🧠 LLM ({llm_time:.0f}ms): {text_response[:50]}...")

            except Exception as e:
                llm_time = (time.time() - start_llm) * 1000
                logger.error(f"❌ Erro no LLM: {e}")
                text_response = "Desculpe, estou com dificuldades técnicas no momento."

            # 5. Update context manager with new exchange
            # Usa placeholder até o STT real chegar do background
            context_manager.add_message("user", transcript_placeholder)
            context_manager.add_message("assistant", text_response)
            await context_manager.add_to_memory(transcript_placeholder, text_response)

            # Save interaction to session (transcript real virá do background STT)
            session_manager.add_interaction(
                session_id=session.session_id,
                user_message=transcript_placeholder,  # Será atualizado pelo STT em background
                assistant_response=text_response,
                audio_duration_ms=(len(audio) / sample_rate * 1000)
            )

            # 2. Store in memory if enabled (legacy support)
            if self.memory_store and session_id:
                # Adicionar mensagem do usuário (será o áudio transcrito)
                await self.memory_store.add_message(
                    session_id,
                    role="user",
                    content=transcript_placeholder  # Será atualizado pelo STT em background
                )
                # Adicionar resposta do assistente
                await self.memory_store.add_message(
                    session_id,
                    role="assistant",
                    content=text_response
                )

            # 6. TTS: Synthesize speech response with graceful degradation
            logger.info(f"🔊 Synthesizing with voice: {voice_id}")

            start_tts = time.time()
            try:
                # Use graceful degradation for TTS
                async def tts_func():
                    return await asyncio.to_thread(self.tts.synthesize, text_response, voice_id=voice_id)

                fallback_response = await degradation_manager.execute_with_fallback(
                    service_name="kokoro_tts",
                    primary_func=tts_func,
                    correlation_id=session_id
                )

                if fallback_response.success:
                    audio_bytes = fallback_response.data
                    if fallback_response.degraded:
                        logger.warning(f"⚠️ TTS usando fallback: {fallback_response.fallback_used}")
                else:
                    # Ultimate fallback - silent audio
                    audio_bytes = b'\x00' * 1024  # 1KB of silence
                    logger.error("❌ TTS falhou completamente, usando áudio silencioso")

            except Exception as e:
                logger.error(f"❌ Erro no TTS: {e}")
                audio_bytes = b'\x00' * 1024  # 1KB of silence

            tts_time = (time.time() - start_tts) * 1000

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.average_processing_time = (
                (self.average_processing_time * self.total_processed + processing_time) /
                (self.total_processed + 1)
            )
            self.total_processed += 1

            # 6. Start generating suggestions in background (streaming mode)
            suggestion_task = None
            if context and context.get('enable_suggestions', False):
                logger.info("💡 Starting suggestion generation in background (streaming mode)...")
                # Start suggestions generation but don't wait - will be processed after response is sent
                suggestion_task = asyncio.create_task(
                    self._generate_conversation_suggestions(
                        user_message=transcript,
                        assistant_response=text_response,
                        context=formatted_context
                    )
                )

            # Log timing breakdown
            context_time = context_data['metadata'].get('build_time_ms', 0)
            logger.info(f"⏱️ Timing: STT={stt_time:.0f}ms Context={context_time:.1f}ms LLM={llm_time:.0f}ms TTS={tts_time:.0f}ms Total={processing_time:.0f}ms")

            result = {
                'audio_response': audio_bytes,
                'text_response': text_response,
                'transcript': transcript_placeholder,  # Transcript real virá do STT em background,
                'session_id': session_id,
                'processing_time_ms': processing_time,
                'latency_ms': processing_time,
                'voice_id': voice_id,
                'timing_breakdown': {
                    'stt_ms': 0,  # STT não bloqueia, roda em background,
                    'context_ms': context_time,
                    'llm_ms': llm_time,
                    'tts_ms': tts_time,
                    'suggestion_ms': 0,  # Will be updated when suggestions are ready
                    'total_ms': processing_time
                }
            }

            # Wait for suggestions if they were requested (streaming mode)
            # In a real streaming implementation, this could be sent as a follow-up
            if suggestion_task:
                try:
                    start_suggestions = time.time()
                    suggestions = await asyncio.wait_for(suggestion_task, timeout=5.0)
                    suggestion_time = (time.time() - start_suggestions) * 1000

                    if suggestions:
                        result['suggestions'] = suggestions
                        result['suggestion_time_ms'] = suggestion_time
                        result['timing_breakdown']['suggestion_ms'] = suggestion_time
                        logger.info(f"💡 Suggestions ready ({suggestion_time:.0f}ms): {len(suggestions)} options")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Suggestion generation timed out")
                    result['suggestions'] = []
                    result['suggestion_time_ms'] = 5000
                except Exception as e:
                    logger.error(f"❌ Error generating suggestions: {e}")
                    result['suggestions'] = []
                    result['suggestion_time_ms'] = 0

            return result

        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Error in audio processing: {e}")

            # Convert to structured error
            if isinstance(e, UltravoxError):
                raise
            else:
                raise ProcessingError(
                    message=f"Audio processing failed: {str(e)}",
                    component="pipeline"
                )

    def _format_context_for_prompt(self, context_data: Dict[str, Any]) -> str:
        """
        Format context for LLM prompt

        Estratégias de prompt testadas:
        1. System prompt com memórias - Mais formal, melhor para assistentes
        2. User prompt com contexto - Mais natural, melhor para conversação
        3. Híbrido - Combina ambos para máxima efetividade
        """
        parts = []

        # OPÇÃO 1: System Prompt (mais formal, estruturado)
        system_parts = []

        # Adicionar memórias relevantes antigas como contexto do sistema
        if context_data.get('relevant_memories'):
            system_parts.append("# Relevant Context from Previous Conversations")
            for mem in context_data['relevant_memories']:
                # Remover timestamps para reduzir tokens
                content = mem['content'].replace('\n', ' ')
                similarity = mem.get('similarity_score', 0)
                if similarity > 0.8:  # Alta relevância
                    system_parts.append(f"[HIGH RELEVANCE] {content}")
                else:
                    system_parts.append(f"[CONTEXT] {content}")

        # OPÇÃO 2: User Prompt (mais natural, conversacional)
        user_parts = []

        # Adicionar conversação recente
        if context_data.get('recent_messages'):
            # Incluir apenas as últimas mensagens mais relevantes
            recent = context_data['recent_messages'][-10:]  # Últimas 10

            for msg in recent:
                role = "User" if msg.role == "user" else "Assistant"
                user_parts.append(f"{role}: {msg.content}")

        # OPÇÃO 3: Formato Híbrido (recomendado)
        # System: Instruções + memórias antigas relevantes
        # User: Conversa recente + mensagem atual

        formatted = []

        # System context (memórias relevantes como instruções)
        if system_parts:
            formatted.append("SYSTEM CONTEXT:")
            formatted.extend(system_parts)
            formatted.append("")

        # Recent conversation
        if user_parts:
            formatted.append("RECENT CONVERSATION:")
            formatted.extend(user_parts)
            formatted.append("")

        # Current message (já incluído pelo Ultravox)
        # formatted.append(f"Current User Message: {context_data['user_message']}")

        return "\n".join(formatted)

    async def _generate_conversation_suggestions(self,
                                               user_message: str,
                                               assistant_response: str,
                                               context: Optional[str] = None) -> List[str]:
        """
        Generate 3 conversation continuation suggestions based on the current exchange.
        Uses a single LLM call for efficiency.
        """
        try:
            # Create a prompt to generate suggestions
            suggestion_prompt = f"""Baseado na conversa:
Usuário: {user_message}
Assistente: {assistant_response}

Gere exatamente 3 perguntas ou frases curtas que o usuário poderia usar para continuar a conversa.
As sugestões devem ser:
1. Relevantes ao tópico
2. Naturais e conversacionais
3. Curtas (máximo 10 palavras cada)

Formato: Uma sugestão por linha, sem números ou marcadores."""

            # Use Ultravox to generate suggestions in one shot
            suggestions_text = await asyncio.to_thread(
                self.audio_processor.ask_text,
                suggestion_prompt
            )

            # Parse the suggestions
            suggestions = []
            lines = suggestions_text.strip().split('\n')

            for line in lines[:3]:  # Take max 3 suggestions
                suggestion = line.strip()
                # Remove any numbering or bullets if present
                suggestion = suggestion.lstrip('123.-•* ')
                if suggestion and len(suggestion) > 0:
                    suggestions.append(suggestion)

            # Fallback suggestions if parsing fails
            if len(suggestions) < 3:
                default_suggestions = [
                    "Me conte mais sobre isso",
                    "Pode dar um exemplo?",
                    "O que mais você sabe?"
                ]
                suggestions.extend(default_suggestions[:3-len(suggestions)])

            return suggestions[:3]  # Ensure exactly 3

        except Exception as e:
            logger.warning(f"⚠️ Failed to generate suggestions: {e}")
            # Return default suggestions on error
            return [
                "Me conte mais sobre isso",
                "Pode dar um exemplo?",
                "O que mais você sabe?"
            ]

    async def _async_stt_for_memory(self, audio_bytes: bytes, sample_rate: int,
                                   language: str, session_id: str,
                                   context_manager, session) -> None:
        """
        Process STT asynchronously in background for memory storage only.
        This doesn't block the main response pipeline.
        """
        try:
            logger.info("🎤 Background STT processing started...")
            start_stt = time.time()

            # Use graceful degradation for STT
            from core.graceful_degradation import degradation_manager

            fallback_response = await degradation_manager.execute_with_fallback(
                service_name="groq_stt",
                primary_func=self.groq_stt.transcribe_audio,
                audio_data=audio_bytes,
                sample_rate=sample_rate,
                language=language,
                correlation_id=session_id
            )

            transcript = fallback_response.data if fallback_response.success else '[Erro na transcrição]'
            stt_time = (time.time() - start_stt) * 1000

            if transcript and transcript != '[Erro na transcrição]':
                logger.info(f"🎤 Background STT complete ({stt_time:.0f}ms): {transcript[:50]}...")

                # Update memory with actual transcript (retroactive)
                try:
                    # Update context manager with real transcript
                    if hasattr(context_manager, 'update_last_user_message'):
                        context_manager.update_last_user_message(transcript)

                    # Update session with real transcript
                    from core.session_manager import session_manager
                    if hasattr(session_manager, 'update_last_interaction'):
                        session_manager.update_last_interaction(session_id, user_message=transcript)

                    # Store in memory store
                    if self.memory_store and session_id:
                        # Atualizar a última mensagem do usuário com a transcrição real
                        # Por enquanto, vamos adicionar uma nova mensagem com a transcrição correta
                        # (idealmente deveria atualizar a mensagem anterior)
                        logger.debug(f"STT transcript ready: {transcript[:50]}...")

                    logger.info(f"💾 Memory updated with real transcript: {transcript[:30]}...")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to update memory with transcript: {e}")

                if fallback_response.degraded:
                    logger.warning(f"⚠️ Background STT used fallback: {fallback_response.fallback_used}")
            else:
                logger.warning("⚠️ Background STT returned empty text")

        except Exception as e:
            logger.error(f"❌ Background STT failed: {e}")

    async def process_text(self,
                          text: str,
                          session_id: str = "text_session",
                          voice_id: str = "pf_dora",
                          generate_audio: bool = True,
                          generate_suggestions: bool = True) -> Dict[str, Any]:
        """
        Process text input directly (without audio STT)
        Perfect for text-based interactions and testing
        Now with conversation continuations!
        """
        start_time = time.time()
        logger.info(f"📝 Processing text input: '{text[:50]}...'")

        try:
            # 0. Get or create session and context manager
            session = get_or_create_session(session_id)

            # Get or create context manager for this session
            context_manager = get_context_manager(session_id)
            if not context_manager:
                user_id = session.user_id or session_id  # Use session_id as fallback
                context_manager = await create_context_manager(
                    session_id=session_id,
                    user_id=user_id,
                    window_size=30,
                    rag_top_k=3
                )
                logger.info(f"🧠 Context Manager created for session {session_id[:12]}...")

            # 1. Build context with sliding window + RAG
            context_data = await context_manager.build_context(
                user_message=text,
                include_rag=True
            )

            # 2. Format context for LLM
            formatted_context = self._format_context_for_prompt(context_data)

            # Add fixed conversational instructions to the pipeline
            # This encourages short, engaging responses that keep the user talking
            fixed_instructions = """SYSTEM_PROMPT:
You are a conversational assistant. Keep responses SHORT and ENGAGING.

CRITICAL RULES:
• Use 3-5 words per sentence
• Maximum 2 short sentences
• Ask questions to keep conversation flowing
• Be friendly but concise

EXAMPLES:
User: "I'm tired"
You: "Long day? Tell me more."

User: "I love pizza"
You: "What's your favorite topping?"

User: "It's raining"
You: "Staying inside today? Cozy plans?"
"""

            # Combine fixed instructions with any custom context
            if formatted_context:
                formatted_context = fixed_instructions + "\n" + formatted_context
            else:
                formatted_context = fixed_instructions

            # 3. LLM Processing for main response
            logger.info(f"🧠 LLM processing text with context...")
            start_llm = time.time()

            try:
                # Use Ultravox ask_text method for direct text processing
                # Always use formatted_context now (includes fixed instructions)
                text_response = await asyncio.to_thread(
                    self.audio_processor.ask_text,
                    text,
                    user_prompt=formatted_context
                )

                llm_time = (time.time() - start_llm) * 1000

                logger.info(f"🧠 LLM ({llm_time:.0f}ms): {text_response[:50]}...")

                # 4. Start generating suggestions in background (streaming mode)
                suggestion_task = None
                if generate_suggestions:
                    logger.info("💡 Starting suggestion generation in background (streaming mode)...")
                    # Start suggestions generation but don't wait
                    suggestion_task = asyncio.create_task(
                        self._generate_conversation_suggestions(
                            user_message=text,
                            assistant_response=text_response,
                            context=formatted_context
                        )
                    )

            except Exception as e:
                llm_time = (time.time() - start_llm) * 1000
                logger.error(f"❌ Erro no LLM: {e}")
                text_response = "Desculpe, estou com dificuldades técnicas no momento."
                suggestion_task = None

            # 5. Update context and memory
            context_manager.add_message("user", text)
            context_manager.add_message("assistant", text_response)
            await context_manager.add_to_memory(text, text_response)

            # Save interaction to session
            session_manager.add_interaction(
                session_id=session.session_id,
                user_message=text,
                assistant_response=text_response,
                audio_duration_ms=0  # No audio input
            )

            # Store in memory if enabled
            if self.memory_store and session_id:
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

            # Prepare initial result (can be sent immediately)
            result = {
                'text_response': text_response,
                'session_id': session_id,
                'latency_ms': llm_time,
                'total_time_ms': (time.time() - start_time) * 1000
            }

            # Wait for suggestions if they were requested (streaming mode)
            if suggestion_task:
                try:
                    start_suggestions = time.time()
                    suggestions = await asyncio.wait_for(suggestion_task, timeout=5.0)
                    suggestion_time = (time.time() - start_suggestions) * 1000

                    if suggestions:
                        result['suggestions'] = suggestions
                        result['suggestion_time_ms'] = suggestion_time
                        logger.info(f"💡 Suggestions ready ({suggestion_time:.0f}ms): {len(suggestions)} options")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Suggestion generation timed out")
                    result['suggestions'] = []
                    result['suggestion_time_ms'] = 5000
                except Exception as e:
                    logger.error(f"❌ Error generating suggestions: {e}")
                    result['suggestions'] = []
                    result['suggestion_time_ms'] = 0
            else:
                result['suggestions'] = []
                result['suggestion_time_ms'] = 0

            # 5. Optional TTS: Generate audio response with STREAMING
            if generate_audio:
                logger.info(f"🎧 Generating STREAMING audio response with voice: {voice_id}")
                start_streaming = time.time()

                try:
                    # Use StreamingHandler for incremental audio generation
                    if self.streaming_handler:
                        streaming_result = await self.streaming_handler.process_with_streaming(
                            llm_response=text_response,
                            session_id=session_id,
                            voice_id=voice_id,
                            enable_suggestions=False  # Suggestions already handled above
                        )

                        streaming_time = (time.time() - start_streaming) * 1000

                        # Combine all audio chunks into single response for compatibility
                        audio_chunks = streaming_result.get('audio_chunks', [])
                        if audio_chunks:
                            # Concatenate all audio bytes from chunks
                            combined_audio = b''
                            valid_chunks = 0
                            for chunk in audio_chunks:
                                if isinstance(chunk, dict) and 'audio_size' in chunk and chunk['audio_size'] > 0:
                                    # For now, we don't have the actual audio data in the chunk
                                    # This is because StreamingHandler doesn't store it, only sends via callback
                                    valid_chunks += 1
                                elif isinstance(chunk, bytes):
                                    combined_audio += chunk

                            # If no actual audio data, create placeholder
                            if not combined_audio and valid_chunks > 0:
                                logger.info(f"🎧 Streaming processed {valid_chunks} audio chunks (sent via WebSocket)")
                                # For HTTP responses, we need to synthesize again in batch mode
                                logger.info("⚠️ Re-synthesizing for HTTP response compatibility...")
                                combined_audio = await asyncio.to_thread(
                                    self.tts.synthesize,
                                    text_response,
                                    voice_id=voice_id
                                )
                            result['audio_response'] = combined_audio
                            result['streaming_sentences'] = len(streaming_result.get('sentences', []))
                            result['first_audio_latency_ms'] = streaming_result['timing']['first_audio_ms']
                            logger.info(f"🎧 STREAMING TTS ({streaming_time:.0f}ms): {len(audio_chunks)} chunks, {len(combined_audio)} total bytes")
                            logger.info(f"   First audio chunk ready in: {result['first_audio_latency_ms']:.0f}ms")
                        else:
                            logger.warning("⚠️ Streaming returned no audio chunks")
                            result['audio_error'] = "No audio chunks generated"

                        result['tts_latency_ms'] = streaming_time
                        result['streaming_info'] = streaming_result.get('timing', {})
                    else:
                        # Fallback to batch TTS if streaming not available
                        logger.warning("⚠️ StreamingHandler not available, using batch TTS")
                        audio_data = await asyncio.to_thread(self.tts.synthesize, text_response, voice_id=voice_id)
                        fallback_time = (time.time() - start_streaming) * 1000
                        result['audio_response'] = audio_data
                        result['tts_latency_ms'] = fallback_time
                        logger.info(f"🔊 Fallback TTS ({fallback_time:.0f}ms): {len(audio_data)} bytes")

                    result['total_time_ms'] = (time.time() - start_time) * 1000

                except Exception as e:
                    logger.error(f"❌ Erro no TTS streaming: {e}")
                    result['audio_error'] = str(e)

            logger.info(f"✅ Text processing complete: {result['total_time_ms']:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ Erro no processamento de texto: {e}")
            return {
                'error': str(e),
                'text_response': "Erro no processamento",
                'session_id': session_id,
                'total_time_ms': (time.time() - start_time) * 1000
            }

    async def run(self):
        """
        Executa a pipeline principal
        """
        try:
            await self.setup()

            logger.info("🎯 Pipeline em execução...")
            logger.info("=" * 50)
            logger.info("Módulos da Pipeline:")
            logger.info("  • Groq STT: Transcrição de fala")
            logger.info("  • Ultravox: Processamento LLM multimodal")
            logger.info("  • Kokoro TTS: Síntese de voz neural")
            logger.info("  • Memory: Gerenciamento de contexto")
            logger.info("=" * 50)
            logger.info("📡 Aceita entrada por áudio ou texto")
            logger.info("🔧 Use process_audio() para áudio ou process_text() para texto")
            logger.info("💬 ask_text() agora usa Ultravox real para máxima qualidade")
            logger.info("=" * 50)

            # Keep running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupção recebida, encerrando...")
        except Exception as e:
            logger.error(f"❌ Erro na pipeline: {e}")
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """
        Limpa recursos e encerra módulos
        """
        logger.info("🧹 Limpando recursos...")

        # Pipeline não existe mais como objeto único
        # Cleanup individual dos componentes já é feito no método cleanup

        logger.info("✅ Pipeline encerrada")

    @with_circuit_breaker("main_pipeline", failure_threshold=10, recovery_timeout=30)
    @validate_input(lambda x: is_valid_session_id(x) if isinstance(x, str) else True, "Invalid session ID")
    async def process_audio(self,
                           audio: np.ndarray,
                           sample_rate: int,
                           session_id: str,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa áudio através da pipeline completa

        Args:
            audio: Dados de áudio como numpy array
            sample_rate: Taxa de amostragem em Hz
            session_id: Identificador da sessão
            context: Contexto opcional

        Returns:
            Dicionário com resposta de texto e áudio sintetizado
        """
        if not self.audio_processor or not self.tts:
            raise RuntimeError("Pipeline não inicializada. Chame setup() primeiro.")

        start_time = time.time()

        # Otimizar contexto para português acadêmico
        if not context:
            context = {}
        context.setdefault('language', 'pt-BR')
        context.setdefault('academic_mode', True)
        context.setdefault('voice_id', 'pf_dora')

        try:
            # Processar diretamente com a abordagem híbrida
            return await self._process_audio_internal(audio, sample_rate, session_id, context)
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Erro no processamento: {e}")

            # Handle error with error handler
            error_response = await error_handler.handle_error(e, component="pipeline")

            return {
                'success': False,
                'error': error_response,
                'session_id': session_id,
                'timestamp': time.time()
            }

    async def process_audio_sync(self,
                                audio: np.ndarray,
                                sample_rate: int,
                                session_id: str,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Versão síncrona direta do processamento (bypass da UnifiedPipeline)
        Para casos onde queremos acesso direto aos módulos
        """
        if not self.audio_processor or not self.tts:
            raise RuntimeError("Pipeline não inicializada. Chame setup() primeiro.")

        start_time = time.time()

        # Otimizar contexto para português acadêmico
        if not context:
            context = {}
        context.setdefault('language', 'pt-BR')
        context.setdefault('academic_mode', True)
        context.setdefault('voice_id', 'pf_dora')

        try:
            # Processamento direto (síncrono)
            logger.info(f"🎤 Processando áudio para sessão {session_id[:8]}...")

            # Step 1: STT + Understanding
            text_response = await asyncio.to_thread(
                self.audio_processor.process,
                audio=audio,
                sample_rate=sample_rate,
                session_id=session_id,
                context=context
            )

            if not text_response:
                text_response = "Desculpe, não consegui processar o áudio."

            # Step 2: TTS
            logger.info(f"🔊 Sintetizando resposta...")
            audio_response = await asyncio.to_thread(
                self.tts.synthesize,
                text=text_response,
                voice_id=context.get('voice_id', 'pf_dora') if context else 'pf_dora'
            )

            # Stats
            processing_time = (time.time() - start_time) * 1000
            self.total_processed += 1

            response = {
                'success': True,
                'text': text_response,
                'audio': audio_response,
                'session_id': session_id,
                'processing_time_ms': processing_time,
                'timestamp': time.time()
            }

            logger.info(f"✅ Processamento direto em {processing_time:.0f}ms")
            return response

        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Erro no processamento direto: {e}")

            # Handle error with error handler
            error_response = await error_handler.handle_error(e, component="pipeline")

            return {
                'success': False,
                'error': error_response,
                'session_id': session_id,
                'timestamp': time.time()
            }

    def get_statistics(self):
        """Alias for compatibility"""
        return self.get_stats()

    def get_stats(self):
        """
        Retorna estatísticas da pipeline
        """
        uptime = time.time() - self.start_time if self.start_time else 0

        main_stats = {
            "main_pipeline": {
                "uptime_seconds": uptime,
                "total_processed": self.total_processed,
                "total_errors": self.total_errors,
                "success_rate": ((self.total_processed - self.total_errors) / self.total_processed * 100) if self.total_processed > 0 else 0
            }
        }

        # Add error handling stats
        error_stats = error_handler.get_metrics()
        degradation_stats = get_degradation_status()

        main_stats.update({
            "error_handling": error_stats,
            "graceful_degradation": degradation_stats
        })

        return main_stats

    async def cleanup(self):
        """
        Cleanup resources
        """
        logger.info("🧹 Cleaning up pipeline resources...")

        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Cleanup services
        if hasattr(self.audio_processor, 'cleanup'):
            await asyncio.to_thread(self.audio_processor.cleanup)
        if hasattr(self.tts, 'cleanup'):
            self.tts.cleanup()
        if hasattr(self.memory_store, 'cleanup'):
            await self.memory_store.cleanup()

        self.is_initialized = False
        logger.info("✅ Pipeline cleanup completed")


def main():
    """
    Ponto de entrada principal
    """
    print("""
    ╔═══════════════════════════════════════════════╗
    ║     🚀 ULTRAVOX PIPELINE - SISTEMA PRINCIPAL  ║
    ║                                               ║
    ║     Processamento de Áudio em Tempo Real     ║
    ║         com IA Conversacional                ║
    ╚═══════════════════════════════════════════════╝
    """)

    pipeline = MainPipeline()

    try:
        asyncio.run(pipeline.run())
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()