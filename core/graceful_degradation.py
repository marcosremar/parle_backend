#!/usr/bin/env python3
"""
Sistema de Graceful Degradation para Pipeline Ultravox
Permite que o sistema continue funcionando mesmo com falhas parciais
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
import json

from .error_handler import UltravoxError, ServiceUnavailableError, ErrorContext

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status dos serviços"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


@dataclass
class FallbackResponse:
    """Resposta de fallback"""
    success: bool
    data: Any
    message: str
    fallback_used: str
    degraded: bool = True


class ServiceHealthMonitor:
    """Monitor de saúde dos serviços"""

    def __init__(self):
        self.service_status: Dict[str, ServiceStatus] = {}
        self.last_health_check: Dict[str, float] = {}
        self.health_check_interval = 30  # segundos

    async def check_service_health(self, service_name: str, health_check_func: Callable) -> ServiceStatus:
        """Verifica saúde de um serviço"""
        try:
            current_time = time.time()
            last_check = self.last_health_check.get(service_name, 0)

            # Se verificou recentemente, retorna status cached
            if current_time - last_check < self.health_check_interval:
                return self.service_status.get(service_name, ServiceStatus.UNAVAILABLE)

            # Executar health check
            is_healthy = await health_check_func()
            status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNAVAILABLE

            self.service_status[service_name] = status
            self.last_health_check[service_name] = current_time

            logger.info(f"Service {service_name} health check: {status.value}")
            return status

        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            self.service_status[service_name] = ServiceStatus.UNAVAILABLE
            return ServiceStatus.UNAVAILABLE

    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Obtém status atual do serviço"""
        return self.service_status.get(service_name, ServiceStatus.UNAVAILABLE)

    def get_all_status(self) -> Dict[str, str]:
        """Obtém status de todos os serviços"""
        return {name: status.value for name, status in self.service_status.items()}


class GracefulDegradationManager:
    """Gerenciador de degradação graceful"""

    def __init__(self):
        self.health_monitor = ServiceHealthMonitor()
        self.fallback_strategies: Dict[str, List[Callable]] = {}
        self.degradation_config = {
            'enable_fallbacks': True,
            'max_fallback_attempts': 3,
            'fallback_timeout': 10.0
        }

    def register_fallback(self, service_name: str, fallback_func: Callable, priority: int = 1):
        """Registra estratégia de fallback para um serviço"""
        if service_name not in self.fallback_strategies:
            self.fallback_strategies[service_name] = []

        self.fallback_strategies[service_name].append((priority, fallback_func))
        # Ordenar por prioridade (menor número = maior prioridade)
        self.fallback_strategies[service_name].sort(key=lambda x: x[0])

        logger.info(f"Registered fallback for {service_name} with priority {priority}")

    async def execute_with_fallback(self,
                                   service_name: str,
                                   primary_func: Callable,
                                   *args,
                                   correlation_id: Optional[str] = None,
                                   **kwargs) -> FallbackResponse:
        """Executa função com fallback automático"""

        context = ErrorContext(correlation_id=correlation_id)

        # Tentar função primária
        try:
            result = await self._execute_with_timeout(primary_func, *args, **kwargs)
            return FallbackResponse(
                success=True,
                data=result,
                message=f"{service_name} executed successfully",
                fallback_used="none",
                degraded=False
            )

        except Exception as primary_error:
            logger.warning(f"Primary service {service_name} failed: {primary_error}")

            # Verificar se fallbacks estão habilitados
            if not self.degradation_config['enable_fallbacks']:
                raise primary_error

            # Tentar fallbacks
            return await self._try_fallbacks(service_name, primary_error, context, *args, **kwargs)

    async def _try_fallbacks(self,
                           service_name: str,
                           primary_error: Exception,
                           context: ErrorContext,
                           *args,
                           **kwargs) -> FallbackResponse:
        """Tenta estratégias de fallback"""

        fallback_strategies = self.fallback_strategies.get(service_name, [])

        if not fallback_strategies:
            logger.error(f"No fallback strategies registered for {service_name}")
            raise primary_error

        last_error = primary_error

        for priority, fallback_func in fallback_strategies:
            try:
                logger.info(f"Trying fallback {fallback_func.__name__} for {service_name}")

                result = await self._execute_with_timeout(fallback_func, *args, **kwargs)

                return FallbackResponse(
                    success=True,
                    data=result,
                    message=f"Fallback {fallback_func.__name__} successful",
                    fallback_used=fallback_func.__name__,
                    degraded=True
                )

            except Exception as fallback_error:
                logger.warning(f"Fallback {fallback_func.__name__} failed: {fallback_error}")
                last_error = fallback_error
                continue

        # Todos os fallbacks falharam
        logger.error(f"All fallbacks failed for {service_name}")
        raise ServiceUnavailableError(
            service=service_name,
            message=f"Service and all fallbacks unavailable. Last error: {last_error}",
            context=context
        )

    async def _execute_with_timeout(self, func: Callable, *args, **kwargs):
        """Executa função com timeout"""
        timeout = self.degradation_config['fallback_timeout']

        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            else:
                # Para funções síncronas, executar em thread
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")


# Singleton global
degradation_manager = GracefulDegradationManager()


# Fallback implementations específicas
class STTFallbacks:
    """Fallbacks para Speech-to-Text"""

    @staticmethod
    async def mock_transcription(*args, **kwargs) -> str:
        """Fallback que retorna transcrição simulada"""
        return "[Audio não pôde ser transcrito - usando modo degradado]"

    @staticmethod
    async def cached_transcription(audio_data, session_id: str, **kwargs) -> str:
        """Fallback usando cache de transcrições anteriores"""
        # Implementação simplificada - em produção usaria Redis/memcache
        cache_key = f"stt_cache_{session_id}_last"
        # return get_from_cache(cache_key) or "[Áudio não disponível]"
        return "[Última transcrição em cache não disponível]"


class LLMFallbacks:
    """Fallbacks para Large Language Model"""

    @staticmethod
    async def predefined_responses(text: str, **kwargs) -> str:
        """Fallback com respostas predefinidas"""
        responses = {
            "olá": "Olá! Como posso ajudá-lo?",
            "oi": "Oi! Em que posso ser útil?",
            "como você está": "Estou bem, obrigado por perguntar!",
            "qual seu nome": "Sou um assistente virtual.",
            "help": "Como posso ajudá-lo hoje?",
            "obrigado": "De nada! Fico feliz em ajudar."
        }

        text_lower = text.lower().strip()

        # Busca por palavras-chave
        for key, response in responses.items():
            if key in text_lower:
                return f"{response} (Modo de resposta básica ativo)"

        return "Desculpe, estou com dificuldades técnicas. Pode repetir sua pergunta? (Modo degradado)"

    @staticmethod
    async def simple_echo(text: str, **kwargs) -> str:
        """Fallback que ecoa a entrada com processamento básico"""
        if len(text.strip()) == 0:
            return "Não consegui compreender sua mensagem."

        # Processamento básico
        if "?" in text:
            return f"Você perguntou: '{text}'. Infelizmente não posso responder no momento devido a problemas técnicos."
        else:
            return f"Você disse: '{text}'. Obrigado por sua mensagem."


class TTSFallbacks:
    """Fallbacks para Text-to-Speech"""

    @staticmethod
    async def silent_audio(**kwargs) -> bytes:
        """Retorna áudio silencioso"""
        import numpy as np

        # 1 segundo de silêncio em 16kHz, mono
        silence = np.zeros(16000, dtype=np.float32)

        # Converter para bytes (WAV format simplificado)
        audio_int16 = (silence * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    async def text_only_response(text: str, **kwargs) -> Dict[str, Any]:
        """Fallback que retorna apenas texto"""
        return {
            "audio": None,
            "text": text,
            "message": "Síntese de voz temporariamente indisponível - apenas texto",
            "degraded": True
        }


# Registrar fallbacks automaticamente
def register_default_fallbacks():
    """Registra fallbacks padrão do sistema"""

    # STT Fallbacks
    degradation_manager.register_fallback("groq_stt", STTFallbacks.cached_transcription, priority=1)
    degradation_manager.register_fallback("groq_stt", STTFallbacks.mock_transcription, priority=2)

    # LLM Fallbacks
    degradation_manager.register_fallback("ultravox_llm", LLMFallbacks.predefined_responses, priority=1)
    degradation_manager.register_fallback("ultravox_llm", LLMFallbacks.simple_echo, priority=2)

    # TTS Fallbacks
    degradation_manager.register_fallback("kokoro_tts", TTSFallbacks.text_only_response, priority=1)
    degradation_manager.register_fallback("kokoro_tts", TTSFallbacks.silent_audio, priority=2)

    logger.info("Default fallback strategies registered")


# Decorators para aplicar graceful degradation
def with_graceful_degradation(service_name: str):
    """Decorator para aplicar graceful degradation automaticamente"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            return await degradation_manager.execute_with_fallback(
                service_name, func, *args, **kwargs
            )
        return wrapper
    return decorator


# Health check implementations
class HealthChecks:
    """Implementações de health check para serviços"""

    @staticmethod
    async def groq_health_check() -> bool:
        """Health check para Groq STT"""
        try:
            from src.services.external_stt.transcription.groq_transcription import GroqTranscription
            groq = GroqTranscription()
            return groq.api_key is not None
        except ImportError as e:
            logger.debug(f"Groq STT module not available: {e}")
            return False
        except (AttributeError, TypeError) as e:
            logger.debug(f"Groq STT health check failed: {e}")
            return False

    @staticmethod
    async def ultravox_health_check() -> bool:
        """Health check para Ultravox LLM"""
        try:
            from src.services.llm.ultravox import UltravoxAPI
            # Verificação simples se consegue instanciar
            api = UltravoxAPI(system_prompt="test", auto_initialize=False)
            return api is not None
        except ImportError as e:
            logger.debug(f"Ultravox LLM module not available: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.debug(f"Ultravox LLM health check failed: {e}")
            return False

    @staticmethod
    async def kokoro_health_check() -> bool:
        """Health check para Kokoro TTS"""
        try:
            from src.services.tts.kokoro import KokoroTTS
            # Verificação básica
            return True  # Se conseguiu importar, assume que está ok
        except ImportError as e:
            logger.debug(f"Kokoro TTS module not available: {e}")
            return False


# Configurar sistema na inicialização
def initialize_graceful_degradation():
    """Inicializa sistema de graceful degradation"""
    register_default_fallbacks()

    # Registrar health checks
    health_monitor = degradation_manager.health_monitor

    # Configurações de degradação
    degradation_manager.degradation_config.update({
        'enable_fallbacks': True,
        'max_fallback_attempts': 3,
        'fallback_timeout': 15.0,
        'health_check_interval': 60
    })

    logger.info("Graceful degradation system initialized")


# Métricas e status
def get_degradation_status() -> Dict[str, Any]:
    """Retorna status do sistema de degradação"""
    return {
        "degradation_enabled": degradation_manager.degradation_config['enable_fallbacks'],
        "services": degradation_manager.health_monitor.get_all_status(),
        "registered_fallbacks": {
            service: len(strategies)
            for service, strategies in degradation_manager.fallback_strategies.items()
        },
        "config": degradation_manager.degradation_config
    }