#!/usr/bin/env python3
"""
External Conversation Pipeline
Pipeline que usa providers externos (APIs de terceiros)
- STT: Groq Whisper (rápido para transcrição)
- LLM: Groq Llama 3.1 8B
- TTS: Azure Speech Services (mapeamento de vozes)
"""

import asyncio
import logging
import aiohttp
import json
import base64
import numpy as np
from typing import Dict, Any, Optional, List
from src.core.providers.tts.voice_config import resolve_voice
from src.core.conversational_context.context_manager import ConversationalContext
from src.core.configurations.config_manager import config_manager

logger = logging.getLogger(__name__)

class ExternalConversationPipeline:
    """
    Pipeline de conversação que usa providers externos (APIs de terceiros)
    Não carrega modelos localmente, usa apenas APIs externas
    """

    def __init__(self,
                 groq_api_key: Optional[str] = None,
                 azure_speech_key: Optional[str] = None,
                 azure_region: Optional[str] = None,
                 language: str = "Portuguese",
                 voice: str = "pf_dora",
                 enable_context: bool = True,
                 max_context_messages: int = 10):
        """
        Initialize external conversation pipeline

        Args:
            groq_api_key: Groq API key for STT and LLM
            azure_speech_key: Azure Speech Services key for TTS
            azure_region: Azure region for Speech Services
            language: Target language
            voice: Voice ID for TTS
        """
        # Load configurations transparently using configuration manager
        self.groq_api_key = groq_api_key or config_manager.get_env("GROQ_API_KEY")
        self.azure_speech_key = azure_speech_key or config_manager.get_env("AZURE_SPEECH_KEY")
        self.azure_region = azure_region or config_manager.get_env("AZURE_REGION", "eastus")
        self.language = language
        self.voice = voice
        self.session = None
        self.is_initialized = False

        # URLs for external APIs
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.azure_tts_url = f"https://{self.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1"

        # API models from configuration manager (with default fallbacks)
        self.groq_stt_model = config_manager.get_from_settings("api.groq.stt_model", "whisper-large-v3")
        self.groq_llm_model = config_manager.get_from_settings("api.groq.llm_model", "llama-3.1-8b-instant")
        # Otimizado: formato mais rápido (360ms vs 455ms do 32k)
        self.azure_tts_format = "audio-16khz-64kbitrate-mono-mp3"  # Sweet spot: rápido + qualidade
        self.groq_language = config_manager.get_from_settings("api.groq.language", "pt")

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

        logger.info("🌐 ExternalConversationPipeline inicializado com providers externos")
        logger.info(f"   Context: {enable_context}")

    async def initialize(self) -> Any:
        """Initialize the pipeline"""
        if self.is_initialized:
            return

        self.session = aiohttp.ClientSession()
        self.is_initialized = True
        logger.info("✅ ExternalConversationPipeline inicializado")

    async def cleanup(self) -> Any:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.is_initialized = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def _transcribe_with_groq_whisper(self, audio_data: bytes) -> Optional[str]:
        """
        Transcrever áudio usando Groq Whisper (rápido)
        """
        if not self.groq_api_key:
            logger.error("❌ Groq API key não configurada")
            return None

        try:
            # Groq Whisper API endpoint
            url = f"{self.groq_base_url}/audio/transcriptions"

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
            }

            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
            data.add_field('model', self.groq_stt_model)
            data.add_field('language', self.groq_language)

            async with self.session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    transcription = result.get('text', '')
                    logger.info(f"🎤 Groq Whisper transcription: '{transcription[:100]}...'")
                    return transcription
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Groq Whisper error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"❌ Erro na transcrição Groq Whisper: {e}")
            return None

    async def _generate_with_groq_llama(self, text: str) -> Optional[str]:
        """
        Gerar resposta usando Groq Llama 3.1 8B
        """
        if not self.groq_api_key:
            logger.error("❌ Groq API key não configurada")
            return None

        try:
            url = f"{self.groq_base_url}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }

            system_prompt = (
                "Você é um assistente virtual em português brasileiro. "
                "Responda de forma natural, direta e útil. "
                "Mantenha respostas concisas mas informativas."
            )

            payload = {
                "model": self.groq_llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result['choices'][0]['message']['content']
                    logger.info(f"🤖 Groq Llama response: '{response_text[:100]}...'")
                    return response_text
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Groq Llama error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"❌ Erro na geração Groq Llama: {e}")
            return None

    async def _synthesize_with_azure_speech(self, text: str) -> Optional[bytes]:
        """
        Sintetizar voz usando Azure Speech Services (otimizado para velocidade)
        """
        if not self.azure_speech_key:
            logger.error("❌ Azure Speech key não configurada")
            return None

        try:
            # Mapear voz Kokoro para Azure
            azure_voice = self._map_kokoro_to_azure_voice(self.voice)

            # Headers otimizados
            headers = {
                "Ocp-Apim-Subscription-Key": self.azure_speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self.azure_tts_format,
                "User-Agent": "ultravox-pipeline/1.0"
            }

            # SSML otimizado para velocidade (prosody rate="fast")
            ssml = f"""<speak version='1.0' xml:lang='pt-BR'>
                <voice xml:lang='pt-BR' name='{azure_voice}'>
                    <prosody rate="1.1" pitch="0%">
                        {text}
                    </prosody>
                </voice>
            </speak>"""

            # Timeout mais agressivo para detectar problemas de latência rapidamente
            timeout = aiohttp.ClientTimeout(total=5.0, connect=2.0)

            async with self.session.post(
                self.azure_tts_url,
                headers=headers,
                data=ssml.encode('utf-8'),
                timeout=timeout
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    logger.info(f"🔊 Azure TTS synthesized: {len(audio_data)} bytes")
                    return audio_data
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Azure TTS error: {response.status} - {error_text}")
                    return None

        except asyncio.TimeoutError:
            logger.error("❌ Azure TTS timeout - região pode estar lenta")
            return None
        except Exception as e:
            logger.error(f"❌ Erro na síntese Azure TTS: {e}")
            return None

    def _map_kokoro_to_azure_voice(self, kokoro_voice: str) -> str:
        """
        Mapear vozes Kokoro para equivalentes Azure
        """
        voice_mapping = {
            # Portuguese voices (otimizado: Francisca é 2x mais rápida que as outras)
            "pf_dora": "pt-BR-FranciscaNeural",  # Portuguese female - MAIS RÁPIDA (342ms)
            "pm_alex": "pt-BR-AntonioNeural",    # Portuguese male (422ms)

            # English voices
            "af_bella": "en-US-JennyNeural",     # American female
            "am_michael": "en-US-GuyNeural",     # American male
        }

        azure_voice = voice_mapping.get(kokoro_voice, "pt-BR-FranciscaNeural")
        logger.info(f"🎭 Voice mapping: {kokoro_voice} → {azure_voice}")
        return azure_voice

    async def process_audio(self,
                          audio_data: bytes,
                          session_id: str = "external_session",
                          voice_id: Optional[str] = None,
                          sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process audio through external pipeline
        """
        logger.info(f"🎯 Processing audio with external providers: {len(audio_data)} bytes")

        if not self.is_initialized:
            await self.initialize()

        try:
            # 1. STT: Groq Whisper
            logger.info("🎤 Step 1: Transcription with Groq Whisper...")
            transcription = await self._transcribe_with_groq_whisper(audio_data)

            if not transcription:
                return {
                    'success': False,
                    'error': 'Transcription failed',
                    'text': '',
                    'audio': b'',
                    'input_size': len(audio_data),
                    'output_size': 0,
                    'voice_id': voice_id or self.voice
                }

            # 2. LLM: Groq Llama 3.1 8B
            logger.info("🤖 Step 2: Response generation with Groq Llama 3.1 8B...")
            response_text = await self._generate_with_groq_llama(transcription)

            if not response_text:
                return {
                    'success': False,
                    'error': 'Text generation failed',
                    'text': transcription,
                    'audio': b'',
                    'input_size': len(audio_data),
                    'output_size': 0,
                    'voice_id': voice_id or self.voice
                }

            # 3. TTS: Azure Speech Services
            logger.info("🔊 Step 3: Speech synthesis with Azure Speech Services...")
            audio_response = await self._synthesize_with_azure_speech(response_text)

            if not audio_response:
                return {
                    'success': False,
                    'error': 'Speech synthesis failed',
                    'text': response_text,
                    'audio': b'',
                    'input_size': len(audio_data),
                    'output_size': 0,
                    'voice_id': voice_id or self.voice
                }

            logger.info("✅ External pipeline processing complete")
            return {
                'success': True,
                'text': response_text,
                'audio': audio_response,
                'input_size': len(audio_data),
                'output_size': len(audio_response),
                'voice_id': voice_id or self.voice,
                'transcription': transcription,
                'providers': {
                    'stt': 'groq_whisper',
                    'llm': 'groq_llama_3.1_8b',
                    'tts': 'azure_speech'
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in external pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'audio': b'',
                'input_size': len(audio_data),
                'output_size': 0,
                'voice_id': voice_id or self.voice
            }

    async def process_text(self,
                         text: str,
                         session_id: str = "external_session",
                         voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process text through external pipeline (skip STT)
        """
        logger.info(f"📝 Processing text with external providers: '{text[:100]}...'")

        if not self.is_initialized:
            await self.initialize()

        try:
            # 1. LLM: Groq Llama 3.1 8B
            logger.info("🤖 Step 1: Response generation with Groq Llama 3.1 8B...")
            response_text = await self._generate_with_groq_llama(text)

            if not response_text:
                return {
                    'success': False,
                    'error': 'Text generation failed',
                    'text': text,
                    'audio': b'',
                    'voice_id': voice_id or self.voice
                }

            # 2. TTS: Azure Speech Services
            logger.info("🔊 Step 2: Speech synthesis with Azure Speech Services...")
            audio_response = await self._synthesize_with_azure_speech(response_text)

            if not audio_response:
                return {
                    'success': False,
                    'error': 'Speech synthesis failed',
                    'text': response_text,
                    'audio': b'',
                    'voice_id': voice_id or self.voice
                }

            logger.info("✅ External text pipeline processing complete")
            return {
                'success': True,
                'text': response_text,
                'audio': audio_response,
                'voice_id': voice_id or self.voice,
                'input_text': text,
                'providers': {
                    'llm': 'groq_llama_3.1_8b',
                    'tts': 'azure_speech'
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in external text pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': text,
                'audio': b'',
                'voice_id': voice_id or self.voice
            }