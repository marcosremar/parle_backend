"""
TTS Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, APIRouter
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import logging
import base64
import io
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file if it exists
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Only set if not already in environment
                if key and not os.getenv(key):
                    os.environ[key] = value

# Try to import local utils (fallback to core if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router, service_instance=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "tts",
        "port": 8103,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "tts": {
        "provider": "huggingface",
        "model": "default",
        "default_voice": "af_heart",
        "default_language": "en-us",
        "timeout_seconds": 30,
        "max_retries": 3
    }
}

def get_config():
    """Get external tts service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class TTSRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice to use (auto-selected if not specified)")
    provider: Optional[str] = Field(default=None, description="Provider: huggingface or elevenlabs (auto-selected if not specified)")
    language: Optional[str] = Field(default="en-us", description="Language code")
    model: Optional[str] = Field(default=None, description="Model to use (provider-specific)")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")
    
    @field_validator('voice', mode='before')
    @classmethod
    def normalize_voice(cls, v):
        """Normalize voice field: convert invalid values to None"""
        # Accept any string value - normalization happens later in the endpoint
        return v

class TTSResponse(BaseModel):
    """Text-to-speech response"""
    audio_data: str  # Base64 encoded audio
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    format: str = "wav"
    voice: str
    model: str
    provider: str

class VoiceInfo(BaseModel):
    """Voice information"""
    id: str
    name: str
    language: str
    gender: str
    description: Optional[str] = None

# ============================================================================
# Hugging Face TTS Provider (Standalone)
# ============================================================================

class HuggingFaceTTSProvider:
    """Hugging Face Inference API TTS provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('HF_API_KEY') or os.getenv('HUGGINGFACE_API_KEY')
        self.available = False

        if not self.api_key:
            print("⚠️  No Hugging Face API key provided - TTS functionality disabled")
            return

        # Try to import huggingface_hub
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
            self.available = True
            print("✅ Hugging Face TTS provider initialized")
        except ImportError:
            print("⚠️  huggingface_hub not available - TTS functionality disabled")
        except Exception as e:
            print(f"⚠️  Failed to initialize Hugging Face client: {e}")
            self.available = False

    def _get_voice_config(self, voice: str) -> Dict[str, Any]:
        """Get voice configuration - Expanded voice set"""
        voice_configs = {
            # American Female voices
            "af_heart": {
                "voice": "af_heart",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (warm, expressive)",
                "gender": "female",
                "accent": "american"
            },
            "af_nicole": {
                "voice": "af_nicole",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (clear, professional)",
                "gender": "female",
                "accent": "american"
            },
            "af_alloy": {
                "voice": "af_alloy",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (neutral, calm)",
                "gender": "female",
                "accent": "american"
            },
            "af_sarah": {
                "voice": "af_sarah",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (youthful, energetic)",
                "gender": "female",
                "accent": "american"
            },
            "af_kore": {
                "voice": "af_kore",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (soft, gentle)",
                "gender": "female",
                "accent": "american"
            },
            "af_bella": {
                "voice": "af_bella",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (sweet, melodic)",
                "gender": "female",
                "accent": "american"
            },

            # American Male voices
            "am_adam": {
                "voice": "am_adam",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (deep, authoritative)",
                "gender": "male",
                "accent": "american"
            },
            "am_alex": {
                "voice": "am_alex",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (clear, friendly)",
                "gender": "male",
                "accent": "american"
            },
            "am_michael": {
                "voice": "am_michael",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (confident, professional)",
                "gender": "male",
                "accent": "american"
            },
            "am_fenrir": {
                "voice": "am_fenrir",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (strong, resonant)",
                "gender": "male",
                "accent": "american"
            },
            "am_levi": {
                "voice": "am_levi",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (warm, approachable)",
                "gender": "male",
                "accent": "american"
            },

            # British voices (if available)
            "bf_alice": {
                "voice": "bf_alice",
                "lang_code": "b",
                "speed": 1.0,
                "description": "British female voice (elegant, clear)",
                "gender": "female",
                "accent": "british"
            },
            "bm_george": {
                "voice": "bm_george",
                "lang_code": "b",
                "speed": 1.0,
                "description": "British male voice (refined, articulate)",
                "gender": "male",
                "accent": "british"
            }
        }
        return voice_configs.get(voice, voice_configs["af_heart"])

    async def synthesize_speech(self, text: str, voice: str = "af_heart", **kwargs) -> Dict[str, Any]:
        """Synthesize speech using Hugging Face"""
        if not self.available:
            raise HTTPException(status_code=503, detail="TTS provider not available")

        voice_config = self._get_voice_config(voice)

        try:
            # Prepare the payload
            payload = {
                "inputs": text,
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }

            # Add voice parameters if available
            if "voice" in voice_config:
                payload["parameters"] = {
                    "voice": voice_config["voice"]
                }

            # Make API call to Hugging Face Inference API
            import time
            start_time = time.time()

            # Use the InferenceClient for text-to-speech
            audio_bytes = self.client.text_to_speech(
                text=text,
                model="default"
            )

            end_time = time.time()

            # Convert audio bytes to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Try to get audio info (basic WAV parsing)
            try:
                import wave
                import io
                wav_info = wave.open(io.BytesIO(audio_bytes), 'rb')
                sample_rate = wav_info.getframerate()
                n_frames = wav_info.getnframes()
                duration = n_frames / sample_rate if sample_rate > 0 else None
                wav_info.close()
            except:
                sample_rate = 24000  # Default sample rate
                duration = None

            return {
                "audio_data": audio_b64,
                "duration": duration,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "model": "default",
                "provider": "huggingface",
                "latency_ms": (end_time - start_time) * 1000
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices"""
        if not self.available:
            return []

        return [
            {
                "id": "af_heart",
                "name": "Heart",
                "language": "en",
                "gender": "female",
                "description": "Female voice (warm, expressive)",
                "provider": "huggingface"
            },
            {
                "id": "af_nicole",
                "name": "Nicole",
                "language": "en",
                "gender": "female",
                "description": "Female voice (clear, professional)",
                "provider": "huggingface"
            },
            {
                "id": "af_alloy",
                "name": "Alloy",
                "language": "en",
                "gender": "female",
                "description": "Female voice (neutral, calm)",
                "provider": "huggingface"
            },
            {
                "id": "am_adam",
                "name": "Adam",
                "language": "en",
                "gender": "male",
                "description": "Male voice (deep, authoritative)",
                "provider": "huggingface"
            },
            {
                "id": "am_alex",
                "name": "Alex",
                "language": "en",
                "gender": "male",
                "description": "Male voice (clear, friendly)",
                "provider": "huggingface"
            }
        ]

# ============================================================================
# Eleven Labs TTS Provider (Standalone)
# ============================================================================

class ElevenLabsTTSProvider:
    """Eleven Labs TTS provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY') or os.getenv('ELEVEN_LABS_API_KEY')
        self.available = False
        self.client = None
        self._valid_voices = None  # Cache for valid voices from API

        if not self.api_key:
            print("⚠️  No Eleven Labs API key provided - Eleven Labs functionality disabled")
            return

        # Try to import elevenlabs
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
            self.available = True
            print("✅ Eleven Labs TTS provider initialized")
        except ImportError:
            print("⚠️  elevenlabs not available - Eleven Labs functionality disabled")
        except Exception as e:
            print(f"⚠️  Failed to initialize Eleven Labs client: {e}")
            self.available = False

    # ElevenLabs voice name → voice_id mapping (official voices from Eleven Labs)
    # This is a fallback if API call fails
    VOICE_MAPPING = {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",  # Female, clear
        "Drew": "29vD33N1CtxCmqQRPOHJ",    # Male, well-rounded
        "Clyde": "2EiwWnXFnvU5JabPnv8n",   # Male, war veteran
        "Paul": "5Q0t7uMcjvnagumLfvZi",    # Male, ground reporter
        "Domi": "AZnzlk1XvdvUeBnXmlld",    # Female, strong
        "Dave": "CYw3kZ02Hs0563khs1Fj",    # Male, conversational
        "Fin": "D38z5RcWu1voky8WS1ja",     # Male, sailor
        "Bella": "EXAVITQu4vr4xnSDxMaL",   # Female, soft
        "Antoni": "ErXwobaYiN019PkySvjV",  # Male, well-rounded
        "Thomas": "GBv7mTt0atIp3Br8iCZE",  # Male, calm
        "Charlie": "IKne3meq5aSn9XLyUdCD", # Male, casual
        "Emily": "LcfcDJNUP1GQjkzn1xUU",   # Female, calm
        "Elli": "MF3mGyEYCl7XYWbV9V6O",    # Female, emotional
        "Josh": "TxGEqnHWrfWFTfGW9XjX",    # Male, deep
        "Arnold": "VR6AewLTigWG4xSOukaG",  # Male, crisp
        "Adam": "pNInz6obpgDQGcFmaJgB",    # Male, deep
        "Sam": "yoZ06aMxZJJ28mfd3POQ",     # Male, raspy
    }

    def _fetch_valid_voices_from_api(self) -> Dict[str, str]:
        """Fetch valid voices from Eleven Labs API"""
        if not self.available or not self.client:
            return self.VOICE_MAPPING
        
        try:
            # Fetch voices from API - try different methods depending on SDK version
            try:
                # Newer SDK version
                voices_response = self.client.voices.get_all()
                voices_list = voices_response.voices if hasattr(voices_response, 'voices') else voices_response
            except AttributeError:
                # Older SDK version or different API structure
                try:
                    voices_list = self.client.voices.get_all()
                except:
                    voices_list = []
            
            valid_voices = {}
            
            # Handle different response formats
            if isinstance(voices_list, list):
                for voice in voices_list:
                    if hasattr(voice, 'name') and hasattr(voice, 'voice_id'):
                        voice_name = voice.name
                        voice_id = voice.voice_id
                        valid_voices[voice_name] = voice_id
                    elif isinstance(voice, dict):
                        voice_name = voice.get('name')
                        voice_id = voice.get('voice_id')
                        if voice_name and voice_id:
                            valid_voices[voice_name] = voice_id
            
            # Only use API results if we got valid voices, otherwise use fallback
            if valid_voices:
                self._valid_voices = valid_voices
                logger.info(f"✅ Fetched {len(valid_voices)} voices from Eleven Labs API")
                return valid_voices
            else:
                logger.warning(f"⚠️  No valid voices found in API response, using fallback mapping")
                return self.VOICE_MAPPING
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch voices from API, using fallback mapping: {e}")
            return self.VOICE_MAPPING

    def get_valid_voices(self) -> Dict[str, str]:
        """Get valid voices (from API cache or fallback)"""
        if self._valid_voices is None:
            self._valid_voices = self._fetch_valid_voices_from_api()
        return self._valid_voices

    def is_valid_voice(self, voice: str) -> bool:
        """Check if voice is valid for Eleven Labs"""
        if not voice:
            return False
        valid_voices = self.get_valid_voices()
        return voice in valid_voices

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available Eleven Labs voices"""
        if not self.available:
            return []

        valid_voices = self.get_valid_voices()
        female_voices = ["Rachel", "Domi", "Bella", "Emily", "Elli", "Lily", "Molly"]
        
        return [
            {
                "id": voice_name,
                "name": voice_name,
                "language": "en",
                "gender": "female" if voice_name in female_voices else "male",
                "description": f"Eleven Labs {voice_name} voice",
                "provider": "elevenlabs"
            }
            for voice_name in valid_voices.keys()
        ]

    async def synthesize_speech(self, text: str, voice: str = "Rachel", model: str = "eleven_turbo_v2_5", **kwargs) -> Dict[str, Any]:
        """Synthesize speech using Eleven Labs"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Eleven Labs provider not available")

        # Normalize None, "None", or empty string to None
        if voice in (None, "None", ""):
            voice = None

        # CRITICAL: Normalize known invalid voices BEFORE validation
        # This prevents errors from invalid voices stored in sessions
        known_invalid_voices = ["pf_dora"]
        if voice in known_invalid_voices:
            logger.warning(f"⚠️  Voice '{voice}' is known to be invalid for Eleven Labs, using default")
            voice = None

        # If no voice specified, use default
        if not voice:
            voice = "Rachel"
            logger.info(f"🎤 No voice specified, using default: {voice}")

        # STRICT VALIDATION: Only accept valid Eleven Labs voices
        valid_voices = self.get_valid_voices()
        if voice not in valid_voices:
            available_voices = ", ".join(sorted(valid_voices.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice}' is not valid for Eleven Labs. Available voices: {available_voices}"
            )
        
        # Get voice ID - this will always succeed since we validated above
        voice_id = valid_voices[voice]
        logger.info(f"🎤 Eleven Labs voice: {voice} (ID: {voice_id})")

        try:
            import time
            start_time = time.time()

            # Use the correct Eleven Labs API (text_to_speech method)
            # Use default model if None
            model_id = model or "eleven_turbo_v2_5"
            audio_data = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id
            )

            end_time = time.time()

            # Convert to base64
            import base64
            # audio_data is a generator, convert to bytes
            audio_bytes = b""
            for chunk in audio_data:
                if isinstance(chunk, bytes):
                    audio_bytes += chunk
                else:
                    audio_bytes += chunk
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "audio_data": audio_b64,
                "duration": None,  # Eleven Labs doesn't provide duration
                "sample_rate": 44100,  # Eleven Labs default
                "format": "mp3",
                "voice": voice,
                "model": model or "eleven_turbo_v2_5",  # Default model if None
                "provider": "elevenlabs",
                "latency_ms": (end_time - start_time) * 1000
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Eleven Labs synthesis failed: {str(e)}")


# ============================================================================
# Unified TTS Provider Manager
# ============================================================================

class TTSProviderManager:
    """Manages multiple TTS providers"""

    def __init__(self):
        self.providers = {}
        self.available_providers = []

        # Initialize Hugging Face provider
        try:
            hf_provider = HuggingFaceTTSProvider()
            if hf_provider.available:
                self.providers["huggingface"] = hf_provider
                self.available_providers.append("huggingface")
                print("✅ Hugging Face provider registered")
        except Exception as e:
            print(f"⚠️  Hugging Face provider initialization failed: {e}")

        # Initialize Eleven Labs provider (default provider)
        try:
            elevenlabs_provider = ElevenLabsTTSProvider()
            if elevenlabs_provider.available:
                self.providers["elevenlabs"] = elevenlabs_provider
                # Add Eleven Labs first to make it the default
                self.available_providers.insert(0, "elevenlabs")
                print("✅ Eleven Labs provider registered (DEFAULT)")
        except Exception as e:
            print(f"⚠️  Eleven Labs provider initialization failed: {e}")

        print(f"📊 Available providers: {', '.join(self.available_providers) if self.available_providers else 'None'}")

    def get_provider(self, provider_name: str):
        """Get a specific provider"""
        return self.providers.get(provider_name)

    def get_available_voices(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available voices, optionally filtered by provider"""
        voices = []

        if provider and provider in self.providers:
            voices.extend(self.providers[provider].get_available_voices())
        else:
            # Get voices from all providers
            for provider_name, provider_instance in self.providers.items():
                provider_voices = provider_instance.get_available_voices()
                voices.extend(provider_voices)

        return voices

    async def synthesize_speech(self, text: str, provider: str = "huggingface", voice: str = None, **kwargs) -> Dict[str, Any]:
        """Synthesize speech using the specified provider"""
        if provider not in self.providers:
            raise HTTPException(status_code=400, detail=f"Provider '{provider}' not available")

        provider_instance = self.providers[provider]

        # Normalize None, "None", or empty string to None
        if voice in (None, "None", ""):
            voice = None

        # CRITICAL: Validate and normalize voice for Eleven Labs BEFORE calling provider
        # This prevents errors from invalid voices stored in sessions
        if provider == "elevenlabs" and voice:
            # First, check against known invalid voices (hardcoded for safety)
            known_invalid_voices = ["pf_dora"]
            if voice in known_invalid_voices:
                logger.warning(f"⚠️  Voice '{voice}' is known to be invalid for Eleven Labs, using default")
                voice = None
            else:
                # Then, validate against provider's valid voices list
                try:
                    if hasattr(provider_instance, 'is_valid_voice'):
                        is_valid = provider_instance.is_valid_voice(voice)
                        logger.info(f"🔍 Voice validation: '{voice}' -> valid={is_valid}")
                        if not is_valid:
                            logger.warning(f"⚠️  Voice '{voice}' is not valid for Eleven Labs, using default")
                            voice = None
                except Exception as e:
                    logger.error(f"❌ Error validating voice '{voice}': {e}")
                    # On error, normalize to None to prevent downstream errors
                    voice = None

        # Set default voice based on provider if not specified
        if not voice:
            if provider == "elevenlabs":
                voice = "Rachel"
            elif provider == "huggingface":
                voice = "af_heart"
            else:
                voice = "Rachel"
            logger.info(f"🎤 Using default voice '{voice}' for provider '{provider}'")

        return await provider_instance.synthesize_speech(text=text, voice=voice, **kwargs)

# ============================================================================
# Global Provider Manager Instance
# ============================================================================

try:
    tts_provider_manager = TTSProviderManager()
    provider_available = len(tts_provider_manager.available_providers) > 0
except Exception as e:
    print(f"⚠️  TTS provider manager failed: {e}")
    tts_provider_manager = None
    provider_available = False

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="External TTS Service", version="1.0.0")

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    providers_info = {}
    if tts_provider_manager:
        for provider_name in tts_provider_manager.available_providers:
            provider_instance = tts_provider_manager.get_provider(provider_name)
            if provider_instance:
                providers_info[provider_name] = {
                    "available": provider_instance.available,
                    "voices": len(provider_instance.get_available_voices())
                }
    
    return {
        "status": "healthy" if provider_available else "degraded",
        "service": "tts",
        "timestamp": datetime.now().isoformat(),
        "providers": providers_info,
        "available_providers": tts_provider_manager.available_providers if tts_provider_manager else []
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    if not tts_provider_manager:
        raise HTTPException(status_code=503, detail="TTS providers not available")

    try:
        # Auto-select provider if not specified
        provider = request.provider
        voice = request.voice
        
        # Auto-select provider if not specified
        if not provider:
            # Prefer Eleven Labs if available
            if "elevenlabs" in tts_provider_manager.available_providers:
                provider = "elevenlabs"
            elif "huggingface" in tts_provider_manager.available_providers:
                provider = "huggingface"
            else:
                raise HTTPException(status_code=503, detail="No TTS providers available")
        
        # Normalize None, "None", or empty string to None
        if voice in (None, "None", ""):
            voice = None
        
        # CRITICAL: Reject invalid voices IMMEDIATELY before any processing
        # This must happen before provider selection to prevent errors
        # Note: Invalid voices are validated by provider-specific checks below
        
        # CRITICAL: Normalize invalid voices for Eleven Labs BEFORE processing
        if provider == "elevenlabs" and voice:
            # Check if voice is valid for Eleven Labs before processing
            elevenlabs_provider = tts_provider_manager.get_provider("elevenlabs")
            if elevenlabs_provider:
                if not elevenlabs_provider.is_valid_voice(voice):
                    logger.warning(f"⚠️  Voice '{voice}' is not valid for Eleven Labs, using default")
                    voice = None
        
        logger.info(f"🎤 TTS Request - Voice: '{voice}', Provider: '{provider}'")
        
        result = await tts_provider_manager.synthesize_speech(
            text=request.text,
            provider=provider,
            voice=voice,
            model=request.model,
            api_key=request.api_key
        )

        return TTSResponse(**result)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/voices")
async def get_voices(provider: Optional[str] = None):
    """Get available voices, optionally filtered by provider"""
    if not tts_provider_manager:
        raise HTTPException(status_code=503, detail="TTS providers not available")

    try:
        voices = tts_provider_manager.get_available_voices(provider=provider)
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available models from all providers"""
    if not tts_provider_manager:
        raise HTTPException(status_code=503, detail="TTS providers not available")

    models = []

    # Hugging Face models
    if "huggingface" in tts_provider_manager.available_providers:
        models.append({
            "id": "default",
            "provider": "huggingface",
            "description": "Hugging Face TTS model",
            "languages": ["en"],
            "voices": ["af_heart", "af_nicole", "af_alloy", "af_sarah", "af_kore", "af_bella",
                      "am_adam", "am_alex", "am_michael", "am_fenrir", "am_levi",
                      "bf_alice", "bm_george"]
        })

    # Eleven Labs models
    if "elevenlabs" in tts_provider_manager.available_providers:
        elevenlabs_provider = tts_provider_manager.get_provider("elevenlabs")
        if elevenlabs_provider and elevenlabs_provider.available:
            models.append({
                "id": "eleven_turbo_v2_5",
                "provider": "elevenlabs",
                "description": "Eleven Labs Turbo v2.5 - Fast, high-quality TTS",
                "languages": ["en", "es", "fr", "de", "pt", "it", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
                "voices": list(elevenlabs_provider.VOICE_MAPPING.keys())
            })

    return {"models": models}

@app.post("/synthesize-stream")
async def synthesize_speech_stream(request: TTSRequest):
    """Synthesize speech and stream the audio"""
    if not tts_provider_manager:
        raise HTTPException(status_code=503, detail="TTS providers not available")
    
    try:
        # Auto-select provider if not specified
        provider = request.provider
        if not provider:
            if "elevenlabs" in tts_provider_manager.available_providers:
                provider = "elevenlabs"
            elif "huggingface" in tts_provider_manager.available_providers:
                provider = "huggingface"
            else:
                raise HTTPException(status_code=503, detail="No TTS providers available")
        
        result = await tts_provider_manager.synthesize_speech(
            text=request.text,
            provider=provider,
            voice=request.voice,
            model=request.model,
            api_key=request.api_key
        )

        # Decode base64 to bytes for streaming
        audio_bytes = base64.b64decode(result["audio_data"])

        from fastapi.responses import StreamingResponse
        import io

        def audio_generator():
            yield audio_bytes

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=\"tts_output.wav\"",
                "X-Voice": result["voice"],
                "X-Model": result["model"],
                "X-Provider": result["provider"]
            }
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Stream synthesis failed: {str(e)}")

# Add standard endpoints
# Create a minimal service instance for health checks
class MinimalService:
    async def health_check(self):
        return {"status": "healthy", "service": "tts"}
    def get_service_info(self):
        return {"service": "tts", "version": "1.0.0", "status": "running"}

minimal_service = MinimalService()
router = APIRouter()
add_standard_endpoints(router, minimal_service, "tts")
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing External TTS Service...")
    print(f"   Provider Available: {provider_available}")
    if tts_provider_manager:
        total_voices = 0
        for provider_name in tts_provider_manager.available_providers:
            provider_instance = tts_provider_manager.get_provider(provider_name)
            if provider_instance:
                voices = provider_instance.get_available_voices()
                print(f"   {provider_name.title()}: {len(voices)} voices")
                total_voices += len(voices)
        print(f"   Total Voices: {total_voices}")
    print("✅ External TTS Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8103"))
    print(f"Starting External TTS Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
