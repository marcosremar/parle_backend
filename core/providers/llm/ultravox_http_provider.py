"""
Ultravox HTTP Provider
Provider para comunicar com serviço Ultravox HTTP usando arquitetura de providers
"""

import logging
import os
import aiohttp
import sys
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
from .base import BaseLLMProvider

# Add project root to path for Communication Manager
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.managers.communication_manager import ServiceCommunicationManager, Priority

logger = logging.getLogger(__name__)

class UltravoxHTTPProvider(BaseLLMProvider):
    """Provider para serviço Ultravox HTTP"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', os.getenv('LLM_SERVICE_URL', 'http://localhost:8100'))
        self.timeout = config.get('timeout', 60)
        self.session = None
        self.comm_manager = None

        logger.info(f"🤖 UltravoxHTTPProvider configurado: {self.base_url}")

    async def initialize(self):
        """Inicializa cliente HTTP e Communication Manager"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

        # Initialize Communication Manager for intelligent protocol selection
        self.comm_manager = ServiceCommunicationManager(self.session)
        await self.comm_manager.initialize()
        self.comm_manager.set_preference('ultravox', primary='binary', fallback='json')
        logger.info("🔗 Communication Manager initialized")

        # Verifica se serviço está disponível
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Ultravox HTTP: {data.get('status')}")
                else:
                    raise Exception(f"Ultravox não disponível: {resp.status}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar Ultravox: {e}")
            raise

    async def generate(self,
                      messages: list,
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.7,
                      **kwargs) -> str:
        """
        Gera texto usando endpoint JSON do Ultravox

        Args:
            messages: Lista de mensagens (formato OpenAI)
            max_tokens: Máximo de tokens
            temperature: Temperatura

        Returns:
            Texto gerado
        """
        try:
            # Converter mensagens para prompt simples
            if isinstance(messages, list) and messages:
                prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
            else:
                prompt = str(messages)

            data = {
                "text": prompt,
                "max_tokens": max_tokens or 512
            }

            headers = {'Content-Type': 'application/json'}

            async with self.session.post(
                f"{self.base_url}/json/generate",
                json=data,
                headers=headers
            ) as resp:

                if resp.status == 200:
                    response_data = await resp.json()
                    return response_data.get('text', '')
                else:
                    error = await resp.text()
                    logger.error(f"Ultravox erro {resp.status}: {error}")
                    raise Exception(f"Ultravox falhou: {error}")

        except Exception as e:
            logger.error(f"Erro ao gerar texto: {e}")
            raise

    async def generate_stream(self,
                             messages: list,
                             max_tokens: Optional[int] = None,
                             temperature: float = 0.7,
                             **kwargs) -> AsyncGenerator[str, None]:
        """Stream não suportado pelo HTTP provider"""
        result = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield result

    async def process_audio(self,
                           audio_data: bytes,
                           sample_rate: int = 16000,
                           max_tokens: Optional[int] = None,
                           priority: Priority = Priority.NORMAL) -> str:
        """
        Processa áudio usando Communication Manager (auto-seleciona binary/JSON)

        Args:
            audio_data: Dados de áudio
            sample_rate: Taxa de amostragem
            max_tokens: Máximo de tokens
            priority: Prioridade da requisição (REALTIME, NORMAL, DEBUG)

        Returns:
            Texto transcrito/gerado
        """
        try:
            if not self.comm_manager:
                await self.initialize()

            # Preparar metadados
            metadata = {
                'max_tokens': max_tokens or 512
            }

            # Call using Communication Manager (auto-selects binary for performance)
            # Endpoint is resolved from settings.yaml automatically
            result = await self.comm_manager.call_audio_service(
                service_name='ultravox',
                audio_data=audio_data,
                sample_rate=sample_rate,
                metadata=metadata,
                priority=priority,
                endpoint_name='audio'  # Looks up service_endpoints.ultravox.endpoints.audio in settings
            )

            logger.info(f"🤖 Ultravox responded via {result.get('protocol_used')}: {result.get('text', '')[:100]}...")
            return result.get('text', '')

        except Exception as e:
            logger.error(f"Erro ao processar áudio: {e}")
            raise

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcreve áudio (se suportado)

        Args:
            audio_data: Dados de áudio

        Returns:
            Texto transcrito
        """
        try:
            headers = {'Content-Type': 'audio/wav'}

            async with self.session.post(
                f"{self.base_url}/transcribe",
                data=audio_data,
                headers=headers
            ) as resp:

                if resp.status == 200:
                    response_data = await resp.json()
                    return response_data.get('transcription', '')
                else:
                    # Fallback para processamento de áudio
                    return await self.process_audio(audio_data)

        except Exception as e:
            logger.error(f"Erro na transcrição: {e}")
            # Fallback para processamento de áudio
            return await self.process_audio(audio_data)

    async def get_status(self) -> Dict[str, Any]:
        """Obtém status do serviço"""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"status": "error", "code": resp.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def cleanup(self):
        """Limpa recursos"""
        if self.session:
            await self.session.close()
            self.session = None

    @property
    def is_multimodal(self) -> bool:
        """Ultravox suporta áudio"""
        return True

    @property
    def supported_modes(self) -> list:
        """Modos suportados"""
        return ["text", "audio"]