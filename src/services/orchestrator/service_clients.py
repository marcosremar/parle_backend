#!/usr/bin/env python3
"""
Service Clients - HTTP clients for all downstream services
Clean abstraction layer for orchestrator to communicate with other services
"""

import aiohttp
import base64
import logging
import os
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Priority enum for service calls (simplified - no communication manager needed)
from enum import Enum

class Priority(str, Enum):
    """Request priority levels"""
    REALTIME = "realtime"      # WebRTC, low-latency required
    NORMAL = "normal"          # Standard API calls
    DEBUG = "debug"            # Testing/debugging, prefer JSON

logger = logging.getLogger(__name__)


class ServiceClientError(Exception):
    """Base exception for service client errors"""
    pass


class BaseServiceClient:
    """Base HTTP client with common functionality using direct HTTP calls"""

    def __init__(self, service_name: str, base_url: Optional[str] = None, is_module_service: bool = False):
        self.service_name = service_name
        self.base_url = base_url or self._get_service_url(service_name)
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_module_service = is_module_service  # True if service runs in-process (MODULE)

    def _get_service_url(self, service_name: str) -> str:
        """Get service URL from environment or default ports (with Nomad service discovery support)"""
        env_var = f"{service_name.upper()}_SERVICE_URL"
        default_ports = {
            "stt": "http://localhost:8099",
            "tts": "http://localhost:8103",
            "llm": "http://localhost:8110",
            "orchestrator": "http://localhost:8500",
            "conversation_store": "http://localhost:8800",
            "conversation_history": "http://localhost:8501",
            "session": "http://localhost:8600",
            "scenarios": "http://localhost:8700",
        }
        return os.getenv(env_var, default_ports.get(service_name, f"http://localhost:8000"))

    async def initialize(self, session: aiohttp.ClientSession):
        """Initialize with shared aiohttp session"""
        self.session = session
        logger.info(f"✅ {self.service_name} client initialized with HTTP")

    async def _get(self, path: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Generic GET request using direct HTTP"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise ServiceClientError(f"{self.service_name} GET {path} failed ({resp.status}): {error_text}")
        except aiohttp.ClientError as e:
            raise ServiceClientError(f"{self.service_name} GET {path} error: {e}")

    async def _post(self, path: str, data: Any = None, json_data: Dict = None,
                   headers: Dict = None, timeout: float = 30.0) -> Any:
        """Generic POST request using direct HTTP"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        try:
            async with self.session.post(
                url,
                data=data,
                json=json_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    # Try JSON first, fallback to bytes
                    try:
                        return await resp.json()
                    except:
                        return await resp.read()
                else:
                    error_text = await resp.text()
                    raise ServiceClientError(f"{self.service_name} POST {path} failed ({resp.status}): {error_text}")
        except aiohttp.ClientError as e:
            raise ServiceClientError(f"{self.service_name} POST {path} error: {e}")

    async def _put(self, path: str, json_data: Dict, timeout: float = 5.0) -> Dict[str, Any]:
        """Generic PUT request"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        try:
            async with self.session.put(url, json=json_data,
                                       timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise ServiceClientError(
                        f"{self.service_name} PUT {path} failed ({resp.status}): {error_text}"
                    )
        except aiohttp.ClientError as e:
            raise ServiceClientError(f"{self.service_name} PUT {path} error: {e}")

    async def health_check(self) -> bool:
        """
        Check if service is healthy via HTTP GET /health check.
        """
        try:
            result = await self._get("/health", timeout=2.0)
            return result.get("status") in ["healthy", "ok", "running"]
        except Exception as e:
            logger.debug(f"HTTP health check failed for {self.service_name}: {e}")
            return False


class LLMClient(BaseServiceClient):
    """Ultravox LLM service client (Primary LLM)"""

    def __init__(self):
        super().__init__("llm")

    async def process_audio(self, audio_data: bytes, sample_rate: int = 16000,
                           max_tokens: int = 512, voice_id: str = None,
                           system_prompt: str = None, priority: Priority = Priority.NORMAL) -> Dict[str, Any]:
        """
        Process audio through Ultravox (integrated STT + LLM) using Communication Manager

        Args:
            audio_data: Audio bytes (PCM or WAV)
            sample_rate: Sample rate in Hz
            max_tokens: Maximum tokens to generate
            voice_id: Voice ID for language detection
            system_prompt: Optional system prompt/context for the LLM
            priority: Request priority (REALTIME for binary, DEBUG for JSON)

        Returns:
            Dict with 'text' (response) and 'transcript' (optional)
        """
        try:
            # Convert audio to base64 for HTTP transmission
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            # Prepare request data
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id
            }

            if system_prompt:
                request_data["system_prompt"] = system_prompt

            # Call LLM service via HTTP
            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)

            logger.info(f"🤖 LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get('text', ''),
                "transcript": result.get('transcript', ''),
                "metadata": result.get('metadata', {}),
                "latency_ms": result.get('latency_ms', 0)
            }

        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            raise ServiceClientError(f"LLM processing failed: {e}")


class TTSClient(BaseServiceClient):
    """TTS service client"""

    def __init__(self):
        super().__init__("tts")

    async def synthesize(self, text: str, voice_id: str = None,
                        speed: float = 1.0, sample_rate: int = 16000,
                        format: str = "wav") -> bytes:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            voice_id: Voice ID
            speed: Speech speed
            sample_rate: Sample rate (8000, 16000, or 24000 Hz)
            format: Audio format (wav, mp3, opus, or ogg)

        Returns:
            Audio bytes in specified format
        """
        try:
            # Normalize invalid voices - reject invalid voices for Eleven Labs
            if voice_id:
                # List of valid Eleven Labs voices
                valid_elevenlabs_voices = ["Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave", "Fin", "Bella", "Antoni", "Thomas", "Charlie", "Emily", "Elli", "Josh", "Arnold", "Adam", "Sam"]
                if voice_id not in valid_elevenlabs_voices:
                    logger.warning(f"⚠️  Voice '{voice_id}' is not valid, normalizing to None")
                    voice_id = None
                elif not voice_id.strip():
                    voice_id = None
            
            data = {
                "text": text,
                "voice": voice_id,  # TTS service expects "voice" not "voice_id"
                "speed": speed,
                "sample_rate": sample_rate,
                "format": format
            }

            # Use binary endpoint (33% smaller payload - no base64 overhead)
            audio_data = await self._post(
                "/synthesize",
                json_data=data,
                timeout=20.0
            )

            logger.info(f"🔊 TTS generated: {len(audio_data)} bytes ({sample_rate}Hz, {format})")

            return audio_data

        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            raise ServiceClientError(f"TTS synthesis failed: {e}")


class STTClient(BaseServiceClient):
    """STT (Speech-to-Text) service client"""

    def __init__(self):
        super().__init__("stt")

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio to text

        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to list of int16 samples
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize to float range [-1, 1]
            audio_list = (audio_array / 32768.0).tolist()

            result = await self._post(
                "/audio/transcribe",
                json_data={"audio": audio_list},
                timeout=15.0
            )

            text = result.get("text", "")
            logger.info(f"📝 STT transcribed: {text[:100]}...")

            return text

        except Exception as e:
            logger.error(f"❌ STT error: {e}")
            raise ServiceClientError(f"STT transcription failed: {e}")


class ExternalUltravoxClient(BaseServiceClient):
    """External Ultravox service client (Groq STT + LLM) - Simplified Ultravox alternative"""

    def __init__(self):
        super().__init__("external_ultravox")

    async def process_audio(self, audio_data: bytes, sample_rate: int = 16000,
                           max_tokens: int = 512, voice_id: str = None,
                           system_prompt: str = None, priority: Priority = Priority.NORMAL) -> Dict[str, Any]:
        """
        Process audio through External LLM service via HTTP

        Same interface as LLMClient.process_audio() for easy orchestrator switching

        Args:
            audio_data: Audio bytes (PCM or WAV)
            sample_rate: Sample rate in Hz
            max_tokens: Maximum tokens to generate
            voice_id: Voice ID for language detection
            system_prompt: Optional system prompt/context for the LLM
            priority: Request priority (not used in HTTP mode)

        Returns:
            Dict with 'text' (response), 'transcript', 'metadata', and 'latency_ms'
        """
        try:
            # Convert audio to base64 for HTTP transmission
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            # Prepare request data
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id
            }

            if system_prompt:
                request_data["system_prompt"] = system_prompt

            # Call External LLM service via HTTP
            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)

            logger.info(f"🤖 External LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get('text', ''),
                "transcript": result.get('transcript', ''),
                "metadata": result.get('metadata', {}),
                "latency_ms": result.get('latency_ms', 0)
            }

        except Exception as e:
            logger.error(f"❌ External LLM error: {e}")
            raise ServiceClientError(f"External LLM processing failed: {e}")


class ExternalLLMClient(BaseServiceClient):
    """External LLM service client (Groq, OpenAI) - Fallback LLM"""

    def __init__(self):
        super().__init__("llm", is_module_service=True)  # Renamed from external_llm

    async def generate(self, text: str, system_prompt: Optional[str] = None,
                      conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate response from external LLM (Groq)

        Args:
            text: User input text
            system_prompt: Optional system prompt
            conversation_history: Optional conversation context

        Returns:
            Generated response text
        """
        try:
            messages = []

            # Add system prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history for context
            if conversation_history:
                messages.extend(conversation_history)

            # Add current message
            messages.append({"role": "user", "content": text})

            result = await self._post(
                "/chat",
                json_data={
                    "messages": messages,
                    "model": "groq/llama-3.1-8b-instant"
                },
                timeout=20.0
            )

            # Handle response format: {"choices": [{"message": {"content": "..."}}]}
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
            else:
                # Fallback to old format
                response_text = result.get("text", result.get("response", ""))

            if not response_text:
                raise ServiceClientError("External LLM returned empty response")

            logger.info(f"🤖 External LLM responded: {response_text[:100]}...")

            return response_text

        except Exception as e:
            logger.error(f"❌ External LLM error: {e}")
            raise ServiceClientError(f"External LLM generation failed: {e}")


class SessionClient(BaseServiceClient):
    """Session service client"""

    def __init__(self):
        super().__init__("session", is_module_service=True)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            return await self._get(f"/api/sessions/{session_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ Session {session_id} not found")
            return None

    async def create_session(self, conversation_id: str = None, scenario_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Create a new session with optional specific session_id"""
        try:
            session_data = {
                "scenario_id": scenario_id or "default",  # Required field with default value
            }
            if conversation_id:
                session_data["conversation_id"] = conversation_id
            if session_id:
                session_data["session_id"] = session_id

            result = await self._post("/api/sessions", json_data=session_data)
            created_session_id = result.get("id")
            logger.debug(f"✅ Created session: {created_session_id}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise

    async def update_session_llm(self, session_id: str, llm_type: str) -> bool:
        """Update which LLM is serving this session"""
        try:
            await self._put(
                f"/api/sessions/{session_id}/llm",
                json_data={"active_llm": llm_type}
            )
            logger.debug(f"✅ Updated session {session_id} LLM to {llm_type}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to update session LLM")
            return False


class ScenariosClient(BaseServiceClient):
    """Scenarios service client"""

    def __init__(self):
        super().__init__("scenarios", is_module_service=True)

    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario configuration"""
        try:
            return await self._get(f"/api/scenarios/{scenario_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ Scenario {scenario_id} not found")
            return None

    async def validate_turn(
        self,
        scenario_id: str,
        user_message: str,
        expected_topics: List[str],
        turn_number: int = 1,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate conversation turn against scenario context

        Args:
            scenario_id: Scenario identifier
            user_message: User's message text
            expected_topics: Expected topics/keywords
            turn_number: Current turn number
            session_id: Optional session identifier

        Returns:
            Dict with validation result and adapted system prompt
        """
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/validate-turn",
                json_data={
                    "user_message": user_message,
                    "expected_topics": expected_topics,
                    "turn_number": turn_number,
                    "session_id": session_id
                },
                timeout=5.0
            )

            logger.debug(f"✅ Validated turn: coherence={result.get('coherence_score'):.2f}")
            return result

        except ServiceClientError as e:
            logger.error(f"❌ Turn validation failed: {e}")
            raise

    async def initialize_scenario_state(
        self,
        scenario_id: str,
        session_id: str,
        expected_topics: List[str]
    ) -> Dict[str, Any]:
        """
        Initialize scenario state for a new conversation session

        Args:
            scenario_id: Scenario identifier
            session_id: Session identifier
            expected_topics: Expected topics for scenario

        Returns:
            Initial state
        """
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/initialize-state",
                json_data={
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "expected_topics": expected_topics
                }
            )

            logger.debug(f"🎬 Initialized scenario state for session {session_id}")
            return result

        except ServiceClientError as e:
            logger.error(f"❌ State initialization failed: {e}")
            raise

    async def get_scenario_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current scenario state for a session"""
        try:
            return await self._get(f"/api/scenarios/state/{session_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ State for session {session_id} not found")
            return None

    async def update_scenario_state(
        self,
        session_id: str,
        validation_result: Dict[str, Any]
    ) -> bool:
        """
        Update scenario state with validation metrics

        Args:
            session_id: Session identifier
            validation_result: Validation metrics to store

        Returns:
            Success status
        """
        try:
            await self._post(
                f"/api/scenarios/state/{session_id}/update",
                json_data=validation_result
            )
            logger.debug(f"✅ Updated scenario state for session {session_id}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to update scenario state")
            return False

    async def delete_scenario_state(self, session_id: str) -> bool:
        """Delete scenario state for a session"""
        try:
            await self._post(f"/api/scenarios/state/{session_id}/delete", json_data={})
            logger.debug(f"✅ Deleted scenario state for session {session_id}")
            return True
        except ServiceClientError:
            return False


class ConversationStoreClient(BaseServiceClient):
    """Conversation store service client"""

    def __init__(self):
        super().__init__("conversation_store")

    async def add_turn(self, conversation_id: str, user_audio: bytes = None,
                      user_text: str = "", ai_text: str = "",
                      ai_audio: bytes = None) -> bool:
        """Save conversation turn"""
        try:
            turn_data = {
                "user_text": user_text,
                "ai_text": ai_text
            }

            if user_audio:
                turn_data["user_audio"] = base64.b64encode(user_audio).decode()
            if ai_audio:
                turn_data["ai_audio"] = base64.b64encode(ai_audio).decode()

            await self._post(
                f"/api/conversations/{conversation_id}/turn",
                json_data=turn_data
            )

            logger.debug(f"💾 Saved turn to conversation {conversation_id}")
            return True

        except ServiceClientError:
            logger.warning(f"⚠️ Failed to save conversation turn")
            return False

    async def get_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for context"""
        try:
            result = await self._get(f"/api/conversations/{conversation_id}/messages?limit={limit}")
            return result.get("messages", [])
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to get conversation context")
            return []

    async def create_conversation(self) -> Dict[str, Any]:
        """Create a new conversation"""
        try:
            result = await self._post("/api/conversations", json_data={})
            logger.debug(f"✅ Created conversation: {result.get('conversation_id')}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create conversation: {e}")
            raise


class ExternalSTTClient(BaseServiceClient):
    """External STT service client (Groq Whisper API)"""

    def __init__(self):
        super().__init__("stt", is_module_service=True)  # Renamed from external_stt

    async def transcribe(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio using Groq Whisper API

        Args:
            audio_data: Audio bytes (WAV, MP3, etc)
            language: Language code (default: en)

        Returns:
            Dict with 'text' (transcription) and 'metadata'
        """
        try:
            # Convert audio to base64 for JSON transport
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            result = await self._post(
                "/transcribe",
                json_data={
                    "audio_base64": audio_base64,
                    "language": language
                },
                timeout=15.0
            )

            text = result.get("text", "")
            logger.info(f"📝 External STT transcribed: {text[:100]}...")

            return {
                "text": text,
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"❌ External STT error: {e}")
            raise ServiceClientError(f"External STT transcription failed: {e}")


class ExternalTTSClient(BaseServiceClient):
    """External TTS service client"""

    def __init__(self):
        super().__init__("tts", is_module_service=True)  # Renamed from external_tts

    async def synthesize(self, text: str, voice: str = "af_heart",
                        speed: float = 1.0, format: str = "wav") -> bytes:
        """
        Synthesize speech using external TTS API

        Args:
            text: Text to synthesize
            voice: Voice ID (af_heart, af_nicole, af_sky, am_adam, am_michael)
            speed: Speech speed multiplier
            format: Audio format (wav, mp3)

        Returns:
            Audio bytes in specified format
        """
        try:
            result = await self._post(
                "/synthesize",
                json_data={
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "format": format
                },
                timeout=20.0
            )

            # Response contains audio_base64
            audio_base64 = result.get("audio_base64", "")
            if not audio_base64:
                raise ServiceClientError("External TTS returned no audio")

            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"🔊 External TTS generated: {len(audio_bytes)} bytes ({format})")

            return audio_bytes

        except Exception as e:
            logger.error(f"❌ External TTS error: {e}")
            raise ServiceClientError(f"External TTS synthesis failed: {e}")


class UserClient(BaseServiceClient):
    """User service client"""

    def __init__(self):
        super().__init__("user")

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data"""
        try:
            return await self._get(f"/api/users/{user_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ User {user_id} not found")
            return None

    async def create_user(self, user_id: str, name: str = None, email: str = None) -> Dict[str, Any]:
        """Create a new user"""
        try:
            user_data = {"user_id": user_id}
            if name:
                user_data["name"] = name
            if email:
                user_data["email"] = email

            result = await self._post("/api/users", json_data=user_data)
            logger.debug(f"✅ Created user: {user_id}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create user: {e}")
            raise

    async def authenticate(self, user_id: str, token: str) -> bool:
        """Authenticate user with token"""
        try:
            result = await self._post(
                "/api/auth/verify",
                json_data={"user_id": user_id, "token": token}
            )
            return result.get("valid", False)
        except ServiceClientError:
            return False


class DatabaseClient(BaseServiceClient):
    """Database service client"""

    def __init__(self):
        super().__init__("database", is_module_service=True)

    async def query(self, table: str, filters: Dict = None) -> List[Dict[str, Any]]:
        """Query database"""
        try:
            result = await self._post(
                "/api/query",
                json_data={"table": table, "filters": filters or {}}
            )
            return result.get("results", [])
        except ServiceClientError as e:
            logger.error(f"❌ Database query failed: {e}")
            return []

    async def insert(self, table: str, data: Dict) -> bool:
        """Insert data into database"""
        try:
            await self._post(
                "/api/insert",
                json_data={"table": table, "data": data}
            )
            return True
        except ServiceClientError:
            return False


class FileStorageClient(BaseServiceClient):
    """File storage service client"""

    def __init__(self):
        super().__init__("file_storage", is_module_service=True)

    async def upload_file(self, file_data: bytes, filename: str,
                         user_id: str = None) -> Dict[str, Any]:
        """Upload file to storage"""
        try:
            file_base64 = base64.b64encode(file_data).decode('utf-8')

            result = await self._post(
                "/api/upload",
                json_data={
                    "file_data": file_base64,
                    "filename": filename,
                    "user_id": user_id
                },
                timeout=30.0
            )

            file_id = result.get("file_id")
            logger.debug(f"✅ Uploaded file: {filename} (ID: {file_id})")
            return result

        except ServiceClientError as e:
            logger.error(f"❌ File upload failed: {e}")
            raise

    async def get_file(self, file_id: str) -> Optional[bytes]:
        """Download file from storage"""
        try:
            result = await self._get(f"/api/files/{file_id}")
            file_base64 = result.get("file_data", "")
            if file_base64:
                return base64.b64decode(file_base64)
            return None
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to get file: {file_id}")
            return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from storage"""
        try:
            await self._post(f"/api/files/{file_id}/delete", json_data={})
            logger.debug(f"✅ Deleted file: {file_id}")
            return True
        except ServiceClientError:
            return False


class WebSocketClient(BaseServiceClient):
    """WebSocket service client for real-time communication"""

    def __init__(self):
        super().__init__("websocket")

    async def send_notification(self, user_id: str, message: Dict[str, Any]) -> bool:
        """Send real-time notification via WebSocket"""
        try:
            await self._post(
                f"/api/notify/{user_id}",
                json_data=message
            )
            logger.debug(f"✅ Sent WebSocket notification to {user_id}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to send WebSocket notification")
            return False

    async def broadcast(self, message: Dict[str, Any]) -> bool:
        """Broadcast message to all connected clients"""
        try:
            await self._post("/api/broadcast", json_data=message)
            logger.debug("✅ Broadcasted message via WebSocket")
            return True
        except ServiceClientError:
            return False


class ConversationHistoryClient(BaseServiceClient):
    """Conversation history service client"""

    def __init__(self):
        super().__init__("conversation_history")

    async def get_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            result = await self._get(f"/api/history/{conversation_id}?limit={limit}")
            return result.get("history", [])
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to get conversation history")
            return []

    async def add_entry(self, conversation_id: str, entry: Dict[str, Any]) -> bool:
        """Add entry to conversation history"""
        try:
            await self._post(f"/api/history/{conversation_id}", json_data=entry)
            return True
        except ServiceClientError:
            return False


class RestPollingClient(BaseServiceClient):
    """REST polling service client"""

    def __init__(self):
        super().__init__("rest_polling")

    async def poll(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Poll an endpoint"""
        try:
            query_string = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
            path = f"{endpoint}?{query_string}" if query_string else endpoint
            return await self._get(path)
        except ServiceClientError:
            return {}


class NeuralCodecClient(BaseServiceClient):
    """Neural codec service client for audio processing"""

    def __init__(self):
        super().__init__("neural_codec")

    async def encode_audio(self, audio_data: bytes) -> bytes:
        """Encode audio using neural codec"""
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            result = await self._post(
                "/api/encode",
                json_data={"audio": audio_base64}
            )
            encoded_base64 = result.get("encoded", "")
            return base64.b64decode(encoded_base64)
        except ServiceClientError:
            logger.warning("⚠️ Neural codec encoding failed")
            return audio_data  # Return original on failure

    async def decode_audio(self, encoded_data: bytes) -> bytes:
        """Decode audio using neural codec"""
        try:
            encoded_base64 = base64.b64encode(encoded_data).decode('utf-8')
            result = await self._post(
                "/api/decode",
                json_data={"encoded": encoded_base64}
            )
            audio_base64 = result.get("audio", "")
            return base64.b64decode(audio_base64)
        except ServiceClientError:
            logger.warning("⚠️ Neural codec decoding failed")
            return encoded_data  # Return original on failure


class WebRTCClient(BaseServiceClient):
    """WebRTC service client"""

    def __init__(self):
        super().__init__("webrtc")

    async def create_offer(self, session_id: str) -> Dict[str, Any]:
        """Create WebRTC offer"""
        try:
            result = await self._post(
                f"/api/offer/{session_id}",
                json_data={}
            )
            return result
        except ServiceClientError:
            return {}

    async def handle_answer(self, session_id: str, answer: Dict[str, Any]) -> bool:
        """Handle WebRTC answer"""
        try:
            await self._post(
                f"/api/answer/{session_id}",
                json_data=answer
            )
            return True
        except ServiceClientError:
            return False


class WebRTCSignalingClient(BaseServiceClient):
    """WebRTC signaling service client"""

    def __init__(self):
        super().__init__("webrtc_signaling")

    async def send_signal(self, session_id: str, signal: Dict[str, Any]) -> bool:
        """Send signaling message"""
        try:
            await self._post(
                f"/api/signal/{session_id}",
                json_data=signal
            )
            return True
        except ServiceClientError:
            return False

    async def get_signals(self, session_id: str) -> List[Dict[str, Any]]:
        """Get signaling messages for session"""
        try:
            result = await self._get(f"/api/signals/{session_id}")
            return result.get("signals", [])
        except ServiceClientError:
            return []


class ViberGatewayClient(BaseServiceClient):
    """Viber gateway service client"""

    def __init__(self):
        super().__init__("viber_gateway")

    async def send_message(self, user_id: str, message: str) -> bool:
        """Send message via Viber"""
        try:
            await self._post(
                "/api/send",
                json_data={"user_id": user_id, "message": message}
            )
            logger.debug(f"✅ Sent Viber message to {user_id}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to send Viber message")
            return False

    async def receive_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process received Viber message"""
        try:
            result = await self._post("/api/receive", json_data=message_data)
            return result
        except ServiceClientError:
            return {}


class APIGatewayClient(BaseServiceClient):
    """API Gateway client (for reverse communication)"""

    def __init__(self):
        super().__init__("api_gateway")

    async def register_route(self, route: str, service: str, endpoint: str) -> bool:
        """Register route in API Gateway"""
        try:
            await self._post(
                "/api/routes/register",
                json_data={"route": route, "service": service, "endpoint": endpoint}
            )
            return True
        except ServiceClientError:
            return False

    async def get_routes(self) -> List[Dict[str, Any]]:
        """Get all registered routes"""
        try:
            result = await self._get("/api/routes")
            return result.get("routes", [])
        except ServiceClientError:
            return []

            if not file_base64:
                return None

            return base64.b64decode(file_base64)

        except ServiceClientError:
            logger.warning(f"⚠️ File {file_id} not found")
            return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from storage"""
        try:
            await self._post(f"/api/files/{file_id}/delete", json_data={})
            logger.debug(f"✅ Deleted file: {file_id}")
            return True
        except ServiceClientError:
            return False


# Factory function to create all clients
def create_service_clients(config: Optional[Dict[str, str]] = None) -> Dict[str, BaseServiceClient]:
    """
    Create all service clients using Communication Manager for routing

    Module services (in-process, execution_mode: internal):
    - llm, stt, tts (renamed from external_*)
    - session, scenarios, orchestrator
    - file_storage, database, communication

    HTTP services (separate process, execution_mode: external or remote):
    - REMOVED: llm, tts, stt (services deleted)
    - user, conversation_store
    - api_gateway, websocket, webrtc

    Args:
        config: Optional dictionary (preserved for compatibility, not used)

    Returns:
        Dictionary of service name -> client instance
    """
    clients = {
        # AI Services (Local) - REMOVED: llm, tts, stt services have been removed
        # "llm": LLMClient(),  # REMOVED - service deleted
        # "tts": TTSClient(),  # REMOVED - service deleted
        # "stt": STTClient(),  # REMOVED - service deleted

        # AI Services (External/Remote) - MODULE services (in-process, lightweight API wrappers)
        "external_ultravox": ExternalUltravoxClient(),
        "llm": ExternalLLMClient(),  # Renamed from external_llm
        "stt": ExternalSTTClient(),  # Renamed from external_stt
        "tts": ExternalTTSClient(),  # Renamed from external_tts

        # Data & State Services
        "session": SessionClient(),  # MODULE service (in-process CRUD)
        "scenarios": ScenariosClient(),  # MODULE service (in-process CRUD)
        "conversation_store": ConversationStoreClient(),  # HTTP service
        "conversation_history": ConversationHistoryClient(),  # HTTP service
        "user": UserClient(),  # HTTP service
        "database": DatabaseClient(),  # MODULE service (in-process)
        "file_storage": FileStorageClient(),  # MODULE service (in-process)

        # Communication Services
        "websocket": WebSocketClient(),  # HTTP service - Real-time communication
        "rest_polling": RestPollingClient(),  # HTTP service - REST polling
        "webrtc": WebRTCClient(),  # HTTP service - WebRTC communication
        "webrtc_signaling": WebRTCSignalingClient(),  # HTTP service - WebRTC signaling

        # Audio Processing Services
        "neural_codec": NeuralCodecClient(),  # HTTP service - Neural audio codec

        # Gateway Services
        "api_gateway": APIGatewayClient(),  # HTTP service - API Gateway (reverse communication)
        "viber_gateway": ViberGatewayClient()  # HTTP service - Viber integration
    }

    return clients
