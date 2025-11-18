#!/usr/bin/env python3
"""
Cliente HTTP binário para comunicação entre serviços
Facilita comunicação entre pipeline e serviços HTTP isolados
"""

import aiohttp
import logging
from typing import Dict, Any, Optional, Tuple
import sys
import os

# Adicionar projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.audio.binary_protocol import BinaryProtocol

logger = logging.getLogger(__name__)

class BinaryHTTPClient:
    """Cliente HTTP para comunicação binária entre serviços"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None

    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Garantir que a sessão está ativa"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def send_text_request(
        self,
        endpoint: str,
        text: str,
        metadata: Dict[str, Any],
        msg_type: int = BinaryProtocol.TYPE_TEXT_REQUEST
    ) -> Tuple[int, Dict[str, Any], bytes]:
        """
        Enviar requisição de texto via protocolo binário

        Args:
            endpoint: Endpoint do serviço (ex: "/generate")
            text: Texto a ser enviado
            metadata: Metadados adicionais
            msg_type: Tipo da mensagem binária

        Returns:
            Tuple com (tipo_resposta, metadados_resposta, dados_resposta)
        """
        await self._ensure_session()

        # Empacotar mensagem binária
        binary_message = BinaryProtocol.pack_text_message(text, metadata, msg_type)

        url = f"{self.base_url}{endpoint}"

        logger.debug(f"🔗 Enviando requisição binária para {url}")
        logger.debug(f"📝 Texto: '{text[:100]}...'")
        logger.debug(f"📦 Tamanho binário: {len(binary_message)} bytes")

        try:
            async with self.session.post(
                url,
                data=binary_message,
                headers={'Content-Type': 'application/octet-stream'}
            ) as response:

                if response.content_type == 'application/octet-stream':
                    # Resposta binária
                    response_data = await response.read()
                    msg_type, response_metadata, data = BinaryProtocol.unpack_message(response_data)

                    logger.debug(f"✅ Resposta binária recebida: tipo {msg_type}")
                    return msg_type, response_metadata, data
                else:
                    # Resposta JSON (provavelmente erro)
                    response_json = await response.json()
                    raise Exception(f"HTTP {response.status}: {response_json}")

        except Exception as e:
            logger.error(f"❌ Erro na requisição binária: {e}")
            raise

    async def send_audio_request(
        self,
        endpoint: str,
        audio_data: bytes,
        metadata: Dict[str, Any],
        msg_type: int = BinaryProtocol.TYPE_AUDIO_REQUEST
    ) -> Tuple[int, Dict[str, Any], bytes]:
        """
        Enviar requisição de áudio via protocolo binário

        Args:
            endpoint: Endpoint do serviço (ex: "/synthesize")
            audio_data: Dados de áudio em bytes
            metadata: Metadados adicionais
            msg_type: Tipo da mensagem binária

        Returns:
            Tuple com (tipo_resposta, metadados_resposta, dados_resposta)
        """
        await self._ensure_session()

        # Empacotar mensagem binária
        binary_message = BinaryProtocol.pack_audio_message(audio_data, metadata, msg_type)

        url = f"{self.base_url}{endpoint}"

        logger.debug(f"🔗 Enviando requisição de áudio para {url}")
        logger.debug(f"🎵 Tamanho áudio: {len(audio_data)} bytes")
        logger.debug(f"📦 Tamanho binário: {len(binary_message)} bytes")

        try:
            async with self.session.post(
                url,
                data=binary_message,
                headers={'Content-Type': 'application/octet-stream'}
            ) as response:

                if response.content_type == 'application/octet-stream':
                    # Resposta binária
                    response_data = await response.read()
                    msg_type, response_metadata, data = BinaryProtocol.unpack_message(response_data)

                    logger.debug(f"✅ Resposta de áudio recebida: tipo {msg_type}")
                    return msg_type, response_metadata, data
                else:
                    # Resposta JSON (provavelmente erro)
                    response_json = await response.json()
                    raise Exception(f"HTTP {response.status}: {response_json}")

        except Exception as e:
            logger.error(f"❌ Erro na requisição de áudio: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Verificar saúde do serviço"""
        await self._ensure_session()

        url = f"{self.base_url}/health"

        try:
            async with self.session.get(url) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"❌ Erro no health check: {e}")
            raise

class UltravoxHTTPClient(BinaryHTTPClient):
    """Cliente específico para o servidor Ultravox HTTP"""

    def __init__(self, base_url: str = "http://127.0.0.1:8100", timeout: int = 60):
        super().__init__(base_url, timeout)

    async def generate_text(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        session_id: Optional[str] = None
    ) -> str:
        """
        Gerar resposta de texto via Ultravox

        Args:
            text: Texto de entrada
            max_tokens: Máximo de tokens para gerar
            temperature: Temperatura para geração
            session_id: ID da sessão (opcional)

        Returns:
            Texto gerado pelo Ultravox
        """
        metadata = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'session_id': session_id or 'http_client'
        }

        msg_type, response_metadata, response_data = await self.send_text_request(
            "/generate",
            text,
            metadata
        )

        if msg_type == BinaryProtocol.TYPE_ERROR:
            error_msg = response_data.decode('utf-8')
            raise Exception(f"Ultravox error: {error_msg}")

        return response_data.decode('utf-8')

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        max_tokens: int = 512,
        session_id: Optional[str] = None
    ) -> str:
        """
        Processar áudio via Ultravox

        Args:
            audio_data: Dados de áudio
            sample_rate: Taxa de amostragem
            max_tokens: Máximo de tokens
            session_id: ID da sessão

        Returns:
            Texto gerado a partir do áudio
        """
        metadata = {
            'sample_rate': sample_rate,
            'max_tokens': max_tokens,
            'session_id': session_id or 'http_client_audio'
        }

        msg_type, response_metadata, response_data = await self.send_audio_request(
            "/generate/audio",
            audio_data,
            metadata
        )

        if msg_type == BinaryProtocol.TYPE_ERROR:
            error_msg = response_data.decode('utf-8')
            raise Exception(f"Ultravox audio error: {error_msg}")

        return response_data.decode('utf-8')

class KokoroHTTPClient(BinaryHTTPClient):
    """Cliente específico para o servidor Kokoro TTS HTTP"""

    def __init__(self, base_url: str = "http://127.0.0.1:8101", timeout: int = 30):
        super().__init__(base_url, timeout)

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str = "pt_dora",
        speed: float = 1.0,
        session_id: Optional[str] = None
    ) -> bytes:
        """
        Sintetizar fala via Kokoro TTS

        Args:
            text: Texto para sintetizar
            voice_id: ID da voz
            speed: Velocidade da fala
            session_id: ID da sessão

        Returns:
            Dados de áudio WAV
        """
        metadata = {
            'voice_id': voice_id,
            'speed': speed,
            'session_id': session_id or 'http_client_tts'
        }

        msg_type, response_metadata, response_data = await self.send_text_request(
            "/synthesize",
            text,
            metadata
        )

        if msg_type == BinaryProtocol.TYPE_ERROR:
            error_msg = response_data.decode('utf-8')
            raise Exception(f"Kokoro TTS error: {error_msg}")

        return response_data

    async def synthesize_batch(
        self,
        texts: list,
        voice_id: str = "pt_dora",
        session_id: Optional[str] = None
    ) -> bytes:
        """
        Sintetizar múltiplos textos em lote

        Args:
            texts: Lista de textos
            voice_id: ID da voz
            session_id: ID da sessão

        Returns:
            Áudio concatenado
        """
        combined_text = '\n'.join(texts)

        metadata = {
            'voice_id': voice_id,
            'session_id': session_id or 'http_client_batch'
        }

        msg_type, response_metadata, response_data = await self.send_text_request(
            "/synthesize/batch",
            combined_text,
            metadata
        )

        if msg_type == BinaryProtocol.TYPE_ERROR:
            error_msg = response_data.decode('utf-8')
            raise Exception(f"Kokoro batch error: {error_msg}")

        return response_data

    async def list_voices(self) -> Dict[str, Any]:
        """Listar vozes disponíveis"""
        await self._ensure_session()

        url = f"{self.base_url}/voices"

        try:
            async with self.session.get(url) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"❌ Erro ao listar vozes: {e}")
            raise