#!/usr/bin/env python3
"""
Local Conversation Pipeline with Automatic LLM Failover
Usa os serviços HTTP em vez de carregar modelos diretamente
Ideal para WebRTC e outros clientes que não precisam carregar modelos

⚠️ DEPRECATED: This class is deprecated and will be removed in a future version.
Use OrchestratorClient (src/core/orchestrator_client.py) and Orchestrator Service instead.

The Orchestrator Service provides the same functionality but:
- Runs as a separate HTTP service (http://localhost:8900)
- Enables horizontal scaling
- Centralizes orchestration logic
- Better separation of concerns

Migration guide:
    # Old (deprecated):
    from pipeline.conversation import LocalConversationPipeline
    pipeline = LocalConversationPipeline()
    await pipeline.process(audio_data)

    # New (recommended):
    from src.core.orchestrator_client import OrchestratorClient
    client = OrchestratorClient()
    await client.initialize()
    await client.process_turn(audio_data, session_id)

Features:
- Automatic failover from primary (Ultravox) to fallback (Groq) LLM
- Session management with Redis
- Scenario-based conversation configuration
- State preservation during failures
"""

import asyncio
import logging
import os
import warnings
import aiohttp
import json
import base64
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add project to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.http_clients.binary_client import BinaryHTTPClient
from src.core.audio.binary_protocol import BinaryProtocol
from src.core.conversational_context.context_manager import ConversationalContext
from src.core.pipeline.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.core.settings import get_pipeline_failover_settings
from src.core.http_reliability import get_circuit_breaker, http_call_with_retry

logger = logging.getLogger(__name__)

class LocalConversationPipeline:
    """
    Pipeline de conversação que usa serviços HTTP locais
    Não carrega modelos, apenas faz chamadas para APIs locais

    Includes automatic failover from primary to fallback LLM
    """

    def __init__(self,
                 ultravox_url: str = None,
                 kokoro_url: str = None,
                 language: str = "Portuguese",
                 voice: str = "pf_dora",
                 enable_context: bool = True,
                 max_context_messages: int = 10,
                 enable_failover: bool = True):
        """
        Inicializa pipeline HTTP

        ⚠️ DEPRECATED: Use OrchestratorClient instead.
        See module docstring for migration guide.

        Args:
            ultravox_url: URL do serviço Ultravox (primary LLM)
            kokoro_url: URL do serviço Kokoro TTS
            language: Idioma padrão
            voice: Voz padrão
            enable_failover: Enable automatic LLM failover
        """
        # Emit deprecation warning
        warnings.warn(
            "LocalConversationPipeline is deprecated. "
            "Use OrchestratorClient (src/core/orchestrator_client.py) instead. "
            "See module docstring for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )

        self.ultravox_url = ultravox_url or os.getenv('LLM_SERVICE_URL', 'http://localhost:8100')
        self.kokoro_url = kokoro_url or os.getenv('TTS_SERVICE_URL', 'http://localhost:8101')
        self.language = language
        self.voice = voice
        self.session = None

        # Service URLs (environment variables with defaults)
        self.scenarios_url = os.getenv('SCENARIOS_SERVICE_URL', 'http://localhost:8700')
        self.session_service_url = os.getenv('SESSION_SERVICE_URL', 'http://localhost:8800')
        self.conversation_service_url = os.getenv('CONVERSATION_SERVICE_URL', 'http://localhost:8010')
        self.stt_service_url = os.getenv('STT_SERVICE_URL', 'http://localhost:8099')
        self.external_llm_url = os.getenv('EXTERNAL_LLM_SERVICE_URL', 'http://localhost:8110')

        # Initialize conversational context (shared module)
        self.enable_context = enable_context
        self.context_manager = None
        if enable_context:
            self.context_manager = ConversationalContext(
                max_context_messages=max_context_messages,
                context_window_size=4,
                enable_long_term_memory=True,
                enable_embeddings_search=True
            )

        # Initialize circuit breaker for failover
        self.enable_failover = enable_failover
        self.circuit_breaker = None
        if enable_failover:
            try:
                failover_settings = get_pipeline_failover_settings()
                self.circuit_breaker = CircuitBreaker(
                    CircuitBreakerConfig(
                        failure_threshold=failover_settings.failure_threshold,
                        recovery_timeout=failover_settings.recovery_timeout,
                        half_open_max_calls=failover_settings.half_open_max_calls,
                        primary_timeout=failover_settings.primary_timeout,
                        fallback_timeout=failover_settings.fallback_timeout
                    )
                )
                logger.info("🔌 Circuit breaker initialized for automatic failover")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize circuit breaker: {e}")
                self.enable_failover = False

        logger.info(f"🔗 LocalConversationPipeline configurado:")
        logger.info(f"   Primary LLM (Ultravox): {ultravox_url}")
        logger.info(f"   Fallback LLM (External): {self.external_llm_url}")
        logger.info(f"   TTS (Kokoro): {kokoro_url}")
        logger.info(f"   Scenarios Service: {self.scenarios_url}")
        logger.info(f"   Session Service: {self.session_service_url}")
        logger.info(f"   Language: {language}, Voice: {voice}")
        logger.info(f"   Context: {enable_context}, Failover: {enable_failover}")

    async def initialize(self):
        """Inicializa cliente HTTP"""
        self.session = aiohttp.ClientSession()

        # Verifica se os serviços estão rodando
        await self._check_services()

        logger.info("✅ HTTPConversationPipeline inicializado")

    async def _check_services(self):
        """Verifica se os serviços HTTP estão disponíveis com timeout"""
        try:
            # Testa Ultravox (5s timeout)
            async with asyncio.timeout(5.0):
                async with self.session.get(f"{self.ultravox_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✅ Ultravox: {data.get('status')}")
                    else:
                        raise Exception(f"Ultravox não disponível: {resp.status}")

            # Testa Kokoro (5s timeout)
            async with asyncio.timeout(5.0):
                async with self.session.get(f"{self.kokoro_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✅ Kokoro: {data.get('status')} ({data.get('voices_available')} vozes)")
                    else:
                        raise Exception(f"Kokoro não disponível: {resp.status}")

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Timeout ao verificar serviços: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Erro ao verificar serviços: {e}")
            raise

    async def process_audio(self,
                          audio_data: bytes,
                          session_id: str = "default",
                          voice_id: Optional[str] = None,
                          sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Processa áudio usando pipeline HTTP com failover automático

        Args:
            audio_data: Dados de áudio
            session_id: ID da sessão
            voice_id: ID da voz (opcional)
            sample_rate: Taxa de amostragem

        Returns:
            Resultado processado com informação sobre qual LLM foi usado
        """
        try:
            voice_id = voice_id or self.voice

            logger.info(f"🎤 Processando áudio HTTP: {len(audio_data)} bytes, session={session_id}")

            # Optional: Get session and scenario for system prompt and conversation context
            system_prompt = None
            conversation_id = None
            session_data = await self._get_session(session_id)
            if session_data:
                conversation_id = session_data.get("conversation_id")
                scenario = await self._get_scenario(session_data.get("scenario_id"))
                if scenario:
                    system_prompt = scenario.get("system_prompt")
                    logger.info(f"📝 Using scenario: {scenario.get('name')}")

            # 1. Call LLM with automatic failover and conversation context preservation
            text_response, llm_used = await self._call_llm_with_failover(
                audio_data, sample_rate, system_prompt, conversation_id
            )

            if not text_response:
                return {
                    "success": False,
                    "error": "Both primary and fallback LLMs failed"
                }

            logger.info(f"🤖 LLM ({llm_used}) respondeu: {text_response[:100]}...")

            # 2. Update session with active LLM
            if session_data:
                await self._update_session_llm(session_id, llm_used)

            # 3. Sintetizar resposta com Kokoro
            audio_response = await self._call_kokoro(text_response, voice_id)

            if not audio_response:
                return {
                    "success": False,
                    "error": "Kokoro não retornou áudio"
                }

            logger.info(f"🔊 Kokoro gerou: {len(audio_response)} bytes de áudio")

            return {
                "success": True,
                "text": text_response,
                "audio": audio_response,
                "voice_id": voice_id,
                "session_id": session_id,
                "llm_used": llm_used,  # NEW: Track which LLM was used
                "input_size": len(audio_data),
                "output_size": len(audio_response),
                "circuit_state": self.circuit_breaker.get_state() if self.circuit_breaker else None
            }

        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from Session Service (5s timeout)"""
        try:
            async with asyncio.timeout(5.0):
                async with self.session.get(f"{self.session_service_url}/api/sessions/{session_id}") as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"⚠️ Session {session_id} not found: {resp.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout getting session {session_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get session: {e}")
            return None

    async def _get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario from Scenarios Service (cached, 5s timeout)"""
        try:
            async with asyncio.timeout(5.0):
                async with self.session.get(f"{self.scenarios_url}/api/scenarios/{scenario_id}") as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"⚠️ Scenario {scenario_id} not found: {resp.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout getting scenario {scenario_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get scenario: {e}")
            return None

    async def _update_session_llm(self, session_id: str, llm_type: str):
        """Update which LLM is serving this session (5s timeout)"""
        try:
            async with asyncio.timeout(5.0):
                async with self.session.put(
                    f"{self.session_service_url}/api/sessions/{session_id}/llm",
                    json={"active_llm": llm_type}
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"✅ Updated session {session_id} LLM to {llm_type}")
                    else:
                        logger.warning(f"⚠️ Failed to update session LLM: {resp.status}")
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout updating session {session_id} LLM")
        except Exception as e:
            logger.error(f"❌ Failed to update session LLM: {e}")

    async def _call_ultravox(self, audio_data: bytes, sample_rate: int) -> Optional[str]:
        """Chama serviço Ultravox (Primary LLM) com timeout 10s"""
        try:
            # Preparar metadados
            metadata = {
                'sample_rate': sample_rate,
                'max_tokens': 512,
                'voice_id': self.voice
            }

            # Empacotar no protocolo binário
            binary_data = BinaryProtocol.pack_audio_request(audio_data, metadata)

            # Fazer requisição com timeout 10s
            headers = {'Content-Type': 'application/octet-stream'}

            async with asyncio.timeout(10.0):
                async with self.session.post(
                    f"{self.ultravox_url}/generate/audio",
                    data=binary_data,
                    headers=headers
                ) as resp:

                    if resp.status == 200:
                        response_data = await resp.read()

                        # Desempacotar resposta
                        msg_type, resp_metadata, text_bytes = BinaryProtocol.unpack_message(response_data)
                        return text_bytes.decode('utf-8')
                    else:
                        error_text = await resp.text()
                        logger.error(f"Ultravox erro {resp.status}: {error_text}")
                        raise Exception(f"Ultravox returned status {resp.status}")

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout ao chamar Ultravox: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro ao chamar Ultravox: {e}")
            raise

    async def _call_external_llm(self, text: str, system_prompt: Optional[str] = None,
                                 conversation_id: Optional[str] = None) -> Optional[str]:
        """Call external LLM (Groq) - Fallback LLM with conversation context (10s timeout)"""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history for context continuity
            if self.context_manager and conversation_id:
                try:
                    # Get recent conversation context
                    context_messages = self.context_manager.get_context_for_llm(conversation_id)
                    messages.extend(context_messages)
                    logger.info(f"📚 Added {len(context_messages)} context messages for conversation {conversation_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load conversation context: {e}")

            # Add current message
            messages.append({"role": "user", "content": text})

            async with asyncio.timeout(10.0):
                async with self.session.post(
                    f"{self.external_llm_url}/chat",
                    json={"messages": messages},
                    headers={'Content-Type': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data.get("text", data.get("response", ""))

                        # Save response to context if available
                        if self.context_manager and conversation_id:
                            self.context_manager.add_message(conversation_id, "user", text)
                            self.context_manager.add_message(conversation_id, "assistant", response_text)

                        return response_text
                    else:
                        error_text = await resp.text()
                        logger.error(f"External LLM erro {resp.status}: {error_text}")
                        raise Exception(f"External LLM returned status {resp.status}")

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout ao chamar External LLM: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro ao chamar External LLM: {e}")
            raise

    async def _transcribe_audio_stt(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using STT service (for fallback LLM, 15s timeout)"""
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            async with asyncio.timeout(15.0):
                async with self.session.post(
                    f"{self.stt_service_url}/transcribe",
                    json={"audio": audio_b64},
                    headers={'Content-Type': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("text", "")
                    else:
                        error_text = await resp.text()
                        logger.error(f"STT erro {resp.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout ao transcrever áudio")
            return None
        except Exception as e:
            logger.error(f"Erro ao transcrever áudio: {e}")
            return None

    async def _call_llm_with_failover(self, audio_data: bytes, sample_rate: int,
                                      system_prompt: Optional[str] = None,
                                      conversation_id: Optional[str] = None) -> tuple[Optional[str], str]:
        """
        Call LLM with automatic failover from primary to fallback
        Preserves conversation context during failover for seamless user experience

        Returns:
            Tuple of (response_text, llm_used: "primary"|"fallback")
        """
        if not self.enable_failover or not self.circuit_breaker:
            # No failover, just call primary
            try:
                response = await self._call_ultravox(audio_data, sample_rate)
                return response, "primary"
            except Exception as e:
                logger.error(f"❌ Primary LLM failed and failover disabled: {e}")
                return None, "primary"

        # Use circuit breaker for automatic failover
        async def primary_fn(context):
            """Primary LLM function (Ultravox)"""
            return await self._call_ultravox(context["audio_data"], context["sample_rate"])

        async def fallback_fn(context):
            """Fallback LLM function (External Groq) with conversation context"""
            # Step 1: Transcribe audio using STT
            transcription = await self._transcribe_audio_stt(context["audio_data"])
            if not transcription:
                raise Exception("STT transcription failed")

            # Step 2: Call external LLM with conversation context for continuity
            response = await self._call_external_llm(
                text=transcription,
                system_prompt=context.get("system_prompt"),
                conversation_id=context.get("conversation_id")
            )
            return response

        try:
            result, llm_used = await self.circuit_breaker.call_with_fallback(
                primary_fn=primary_fn,
                fallback_fn=fallback_fn,
                context={
                    "audio_data": audio_data,
                    "sample_rate": sample_rate,
                    "system_prompt": system_prompt,
                    "conversation_id": conversation_id  # NEW: Pass conversation ID for context
                }
            )
            return result, llm_used

        except Exception as e:
            logger.error(f"❌ Both primary and fallback LLMs failed: {e}")
            return None, "failed"

    async def _call_kokoro(self, text: str, voice_id: str, sample_rate: int = 16000) -> Optional[bytes]:
        """Chama serviço Kokoro TTS (10s timeout)"""
        try:
            # Usar endpoint binário (33% menor - sem base64 overhead)
            data = {
                "text": text,
                "voice_id": voice_id,
                "speed": 1.0,
                "sample_rate": sample_rate
            }

            headers = {'Content-Type': 'application/json'}

            async with asyncio.timeout(10.0):
                async with self.session.post(
                    f"{self.kokoro_url}/synthesize",
                    json=data,
                    headers=headers
                ) as resp:

                    if resp.status == 200:
                        audio_data = await resp.read()
                        return audio_data
                    else:
                        error_text = await resp.text()
                        logger.error(f"Kokoro erro {resp.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout ao chamar Kokoro")
            return None
        except Exception as e:
            logger.error(f"Erro ao chamar Kokoro: {e}")
            return None

    async def generate_text(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Gera texto usando Ultravox JSON endpoint (10s timeout)"""
        try:
            data = {
                "text": prompt,
                "max_tokens": max_tokens
            }

            headers = {'Content-Type': 'application/json'}
            endpoint = f"{self.ultravox_url}/json/generate"
            logger.info(f"🤖 Enviando para Ultravox: {endpoint}")
            logger.info(f"🤖 Dados: {data}")

            async with asyncio.timeout(10.0):
                async with self.session.post(
                    endpoint,
                    json=data,
                    headers=headers
                ) as resp:

                    if resp.status == 200:
                        response_data = await resp.json()
                        logger.info(f"🤖 Ultravox response_data: {response_data}")
                        logger.info(f"🤖 Ultravox response field: {response_data.get('response')}")
                        # Ultravox retorna 'response', não 'text'
                        result = response_data.get('response', response_data.get('text', ''))
                        logger.info(f"🤖 Ultravox result: {result}")
                        return result
                    else:
                        error_text = await resp.text()
                        logger.error(f"Ultravox JSON erro {resp.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout ao gerar texto com Ultravox")
            return None
        except Exception as e:
            logger.error(f"Erro ao gerar texto: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def synthesize_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
        """Sintetiza fala usando Kokoro"""
        voice_id = voice_id or self.voice
        return await self._call_kokoro(text, voice_id)

    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcreve áudio usando Ultravox (se suportado, 15s timeout)"""
        try:
            headers = {'Content-Type': 'audio/wav'}

            async with asyncio.timeout(15.0):
                async with self.session.post(
                    f"{self.ultravox_url}/transcribe",
                    data=audio_data,
                    headers=headers
                ) as resp:

                    if resp.status == 200:
                        response_data = await resp.json()
                        return response_data.get('transcription', '')
                    else:
                        logger.warning(f"Transcrição não disponível: {resp.status}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout na transcrição")
            return None
        except Exception as e:
            logger.error(f"Erro na transcrição: {e}")
            return None

    async def get_status(self) -> Dict[str, Any]:
        """Obtém status dos serviços (5s timeout)"""
        try:
            ultravox_status = "unknown"
            kokoro_status = "unknown"

            # Status Ultravox
            try:
                async with asyncio.timeout(5.0):
                    async with self.session.get(f"{self.ultravox_url}/health") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            ultravox_status = data.get('status', 'unknown')
            except (asyncio.TimeoutError, Exception):
                pass

            # Status Kokoro
            try:
                async with asyncio.timeout(5.0):
                    async with self.session.get(f"{self.kokoro_url}/health") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            kokoro_status = data.get('status', 'unknown')
            except (asyncio.TimeoutError, Exception):
                pass

            return {
                "pipeline_type": "http",
                "ultravox": {
                    "url": self.ultravox_url,
                    "status": ultravox_status
                },
                "kokoro": {
                    "url": self.kokoro_url,
                    "status": kokoro_status
                },
                "language": self.language,
                "voice": self.voice
            }

        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Limpa recursos"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("🧹 HTTPConversationPipeline limpo")

    async def __aenter__(self):
        """Context manager"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.cleanup()

    async def process(self,
                     input_data: Any = None,
                     request_data: Dict[str, Any] = None,
                     voice_id: Optional[str] = None,
                     language: Optional[str] = None,
                     return_audio: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Compatibilidade com ConversationController
        Processa requisições do WebRTC

        Args:
            request_data: Dados da requisição

        Returns:
            Resultado processado
        """
        try:
            logger.info(f"🔄 HTTPConversationPipeline.process chamado: input_data={input_data}, voice_id={voice_id}")
            # Compatibilidade com diferentes assinaturas
            if input_data is not None and not request_data:
                # Chamada do ConversationController
                voice_id = voice_id or self.voice
                logger.info(f"🔄 Processando input_data, voice_id={voice_id}")

                if isinstance(input_data, str):
                    # Processamento de texto
                    logger.info(f"🔄 Chamando generate_text com: {input_data}")
                    response_text = await self.generate_text(input_data)
                    logger.info(f"🔄 generate_text retornou: {response_text}")
                    if not response_text:
                        return {
                            "success": False,
                            "error": "Failed to generate response"
                        }

                    if return_audio:
                        audio_data = await self.synthesize_speech(response_text, voice_id)
                        return {
                            "success": True,
                            "text": response_text,
                            "response": response_text,  # Para compatibilidade com ConversationController
                            "audio": audio_data,
                            "audio_output": audio_data,  # Para compatibilidade com ConversationController
                            "voice_id": voice_id,
                            "metrics": {}  # Para compatibilidade com ConversationController
                        }
                    else:
                        return {
                            "success": True,
                            "text": response_text,
                            "response": response_text,  # Para compatibilidade com ConversationController
                            "voice_id": voice_id,
                            "metrics": {}  # Para compatibilidade com ConversationController
                        }

                elif isinstance(input_data, bytes):
                    # Processamento de áudio
                    return await self.process_audio(
                        audio_data=input_data,
                        voice_id=voice_id
                    )

                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input_data type: {type(input_data)}"
                    }

            # Processamento via request_data (chamada direta)
            if not request_data:
                return {
                    "success": False,
                    "error": "No input provided"
                }

            request_type = request_data.get('type')
            session_id = request_data.get('session_id', 'default')

            if request_type == 'text':
                # Processamento de texto
                text = request_data.get('text', '')
                if not text:
                    return {
                        "success": False,
                        "error": "Text field is required"
                    }

                # Gerar resposta com Ultravox
                response_text = await self.generate_text(text)
                if not response_text:
                    return {
                        "success": False,
                        "error": "Failed to generate response"
                    }

                # Sintetizar com Kokoro
                audio_data = await self.synthesize_speech(response_text, self.voice)
                if not audio_data:
                    return {
                        "success": False,
                        "error": "Failed to synthesize audio"
                    }

                return {
                    "success": True,
                    "type": "text",
                    "text": response_text,
                    "audio": audio_data,
                    "session_id": session_id
                }

            elif request_type == 'audio':
                # Processamento de áudio
                audio_data = request_data.get('audio')
                if not audio_data:
                    return {
                        "success": False,
                        "error": "Audio data is required"
                    }

                # Se áudio estiver em base64, decodificar
                if isinstance(audio_data, str):
                    import base64
                    audio_data = base64.b64decode(audio_data)

                return await self.process_audio(
                    audio_data=audio_data,
                    session_id=session_id,
                    voice_id=self.voice
                )

            else:
                return {
                    "success": False,
                    "error": f"Unsupported request type: {request_type}"
                }

        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}")
            return {
                "success": False,
                "error": str(e)
            }