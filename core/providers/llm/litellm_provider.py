"""
LiteLLM Provider - Unified interface for multiple LLM providers
Supports: Groq, OpenAI, Anthropic, Azure, Cohere, and many more
"""

import os
import logging
from typing import Optional, Dict, Any
import litellm
from litellm import completion, transcription
import asyncio

from .base import BaseLLMProvider
from src.core.exceptions import UltravoxError, wrap_exception

# Import Groq client for direct STT access
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq client not available. STT will use LiteLLM fallback.")

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseLLMProvider):
    """
    Unified LLM provider using LiteLLM
    Supports multiple providers with the same interface
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize LiteLLM provider

        Args:
            model: Model string (e.g., "groq/llama3-70b-8192", "gpt-4", "claude-3-opus")
            api_key: API key for the provider (or set in environment)
        """
        self.model = model
        self.provider = model.split('/')[0] if '/' in model else model.split('-')[0]

        # Set API keys based on provider
        if api_key:
            if self.provider == "groq":
                os.environ["GROQ_API_KEY"] = api_key
            elif self.provider in ["gpt", "openai"]:
                os.environ["OPENAI_API_KEY"] = api_key
            elif self.provider == "claude":
                os.environ["ANTHROPIC_API_KEY"] = api_key

        # Initialize direct Groq client for STT if available
        self.groq_client = None
        if self.provider == "groq" and GROQ_AVAILABLE:
            try:
                groq_api_key = api_key or os.environ.get("GROQ_API_KEY")
                if groq_api_key:
                    self.groq_client = Groq(
                        api_key=groq_api_key,
                        timeout=60.0  # 60 second timeout
                    )
                    logger.info("✅ Direct Groq client initialized for STT")
                else:
                    logger.warning("⚠️ Groq API key not found, will use LiteLLM fallback")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")

        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported params automatically
        litellm.set_verbose = kwargs.get("verbose", False)

        # Store additional settings
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens", 1000)

        logger.info(f"Initialized LiteLLM provider with model: {self.model}")

    async def generate(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: float = None,
                      max_tokens: int = None,
                      **kwargs) -> str:
        """
        Generate text response using LiteLLM

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # Use asyncio to make it truly async
            response = await asyncio.to_thread(
                completion,
                model=self.model,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def transcribe(self, audio_file_path: str, language: str = "pt") -> str:
        """
        Transcribe audio using provider's Whisper API
        Currently supports: OpenAI, Groq

        Args:
            audio_file_path: Path to audio file
            language: Language code

        Returns:
            Transcribed text
        """
        try:
            # Check provider support
            if self.provider not in ["groq", "openai", "gpt"]:
                raise NotImplementedError(f"Transcription not supported for {self.provider}")

            # Use direct Groq client for better reliability
            if self.provider == "groq" and self.groq_client:
                logger.info("🎤 Using direct Groq client for transcription")
                return await self._transcribe_with_groq_direct(audio_file_path, language)

            # Fallback to LiteLLM for other providers or if Groq client unavailable
            logger.info("🎤 Using LiteLLM for transcription")
            return await self._transcribe_with_litellm(audio_file_path, language)

        except Exception as e:
            logger.error(f"❌ Error transcribing audio: {e}")
            raise

    async def _transcribe_with_groq_direct(self, audio_file_path: str, language: str) -> str:
        """
        Transcribe using direct Groq client (more reliable)
        """
        try:
            # Check file size (Groq has 25MB limit)
            file_size = os.path.getsize(audio_file_path)
            if file_size > 25 * 1024 * 1024:  # 25MB
                raise ValueError(f"Audio file too large: {file_size} bytes (limit: 25MB)")

            logger.info(f"📤 Transcribing with Groq: {file_size} bytes, language={language}")

            with open(audio_file_path, 'rb') as audio_file:
                # Use asyncio.to_thread to make sync call async
                response = await asyncio.to_thread(
                    self.groq_client.audio.transcriptions.create,
                    file=audio_file,
                    model="whisper-large-v3",
                    language=language,
                    response_format="text",
                    timeout=60.0
                )

            logger.info(f"✅ Groq transcription successful: {len(response)} characters")
            return response

        except Exception as e:
            logger.error(f"❌ Direct Groq transcription failed: {e}")
            # Fall back to LiteLLM if direct client fails
            logger.info("🔄 Falling back to LiteLLM...")
            return await self._transcribe_with_litellm(audio_file_path, language)

    async def _transcribe_with_litellm(self, audio_file_path: str, language: str) -> str:
        """
        Transcribe using LiteLLM (fallback method)
        """
        # For Groq, use groq/whisper-large-v3
        # For OpenAI, use whisper-1
        if self.provider == "groq":
            model = "groq/whisper-large-v3"
        else:
            model = "whisper-1"

        # Open file and pass to transcription API
        with open(audio_file_path, 'rb') as audio_file:
            # For Groq, we need to be more explicit about parameters
            if self.provider == "groq":
                response = await asyncio.to_thread(
                    transcription,
                    model=model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            else:
                response = await asyncio.to_thread(
                    transcription,
                    model=model,
                    file=audio_file,
                    language=language
                )

        # Handle response format
        if self.provider == "groq" and hasattr(response, 'text'):
            return response.text
        elif self.provider == "groq":
            # Groq with response_format="text" returns string directly
            return str(response)
        else:
            return response.text

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": self.provider,
            "supports_transcription": self.provider in ["groq", "openai", "gpt"],
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens
        }

    @staticmethod
    def list_available_models():
        """List all available models in LiteLLM"""
        return litellm.model_list

    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given model and token count"""
        try:
            return litellm.completion_cost(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens
            )
        except Exception as e:
            return 0.0