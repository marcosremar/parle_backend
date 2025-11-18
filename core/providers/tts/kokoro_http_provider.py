"""
Kokoro HTTP Provider
Provider para comunicar com serviço Kokoro TTS HTTP usando arquitetura de providers
"""

import logging
import aiohttp
from typing import Dict, Any, Optional, List
from .base import BaseTTSProvider
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)

class KokoroHTTPProvider(BaseTTSProvider):
    """Provider para serviço Kokoro TTS HTTP"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(**config)
        self.base_url = config.get('base_url', 'http://localhost:8101')
        self.timeout = config.get('timeout', 30)
        self.default_voice = config.get('voice', 'pf_dora')
        self.session = None

        logger.info(f"🔊 KokoroHTTPProvider configurado: {self.base_url}")

    async def initialize(self):
        """Inicializa cliente HTTP"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

        # Verifica se serviço está disponível
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    voices_count = data.get('voices_available', 0)
                    logger.info(f"✅ Kokoro HTTP: {data.get('status')} ({voices_count} vozes)")
                else:
                    raise Exception(f"Kokoro não disponível: {resp.status}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar Kokoro: {e}")
            raise

    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        speed: float = 1.0,
                        sample_rate: int = 16000,
                        format: str = "wav",
                        bitrate: Optional[str] = None,
                        **kwargs) -> bytes:
        """
        Sintetiza texto em áudio

        Args:
            text: Texto para sintetizar
            voice: ID da voz (opcional)
            speed: Velocidade da fala
            sample_rate: Taxa de amostragem (8000, 16000, ou 24000 Hz)
            format: Formato de saída (wav, mp3, opus, ogg) - padrão: wav
            bitrate: Bitrate (opcional, ex: "128k", "64k")

        Returns:
            Dados de áudio no formato especificado
        """
        try:
            voice_id = voice or self.default_voice

            # Usar endpoint binário (mais rápido - sem base64 overhead)
            data = {
                "text": text,
                "voice_id": voice_id,
                "speed": speed,
                "sample_rate": sample_rate,
                "format": format
            }

            # Adicionar bitrate se especificado
            if bitrate:
                data["bitrate"] = bitrate

            headers = {'Content-Type': 'application/json'}

            async with self.session.post(
                f"{self.base_url}/synthesize",
                json=data,
                headers=headers
            ) as resp:

                if resp.status == 200:
                    # Resposta binária direta (33% menor que base64)
                    audio_data = await resp.read()
                    logger.debug(f"🔊 Kokoro gerou {len(audio_data)} bytes para voz {voice_id} ({sample_rate}Hz, {format.upper()})")
                    return audio_data
                else:
                    error = await resp.text()
                    logger.error(f"Kokoro erro {resp.status}: {error}")
                    raise Exception(f"Síntese falhou: {error}")

        except Exception as e:
            logger.error(f"Erro ao sintetizar: {e}")
            raise

    async def get_voices(self) -> List[Dict[str, Any]]:
        """
        Lista vozes disponíveis

        Returns:
            Lista de vozes disponíveis
        """
        try:
            async with self.session.get(f"{self.base_url}/voices") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"Não foi possível obter vozes: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Erro ao obter vozes: {e}")
            return []

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

    async def validate_voice(self, voice: str) -> bool:
        """
        Valida se uma voz existe

        Args:
            voice: ID da voz

        Returns:
            True se a voz existe
        """
        try:
            voices = await self.get_voices()
            return any(v.get('id') == voice for v in voices)
        except Exception as e:
            return False

    async def cleanup(self):
        """Limpa recursos"""
        if self.session:
            await self.session.close()
            self.session = None

    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get list of available voices (sync wrapper)

        Args:
            language: Filter by language (optional)

        Returns:
            List of voice dictionaries
        """
        # For async implementation, use get_voices() instead
        return []

    def get_audio_format(self) -> str:
        """
        Get the audio format produced by this provider

        Returns:
            Audio format (default: "wav", but configurable via format parameter)
        """
        return self.default_format

    @property
    def supported_formats(self) -> List[str]:
        """Formatos de áudio suportados"""
        return ["wav", "mp3", "opus", "ogg"]

    @property
    def default_format(self) -> str:
        """Formato padrão"""
        return "wav"