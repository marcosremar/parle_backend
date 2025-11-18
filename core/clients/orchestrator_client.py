#!/usr/bin/env python3
"""
Orchestrator Client - Reusable HTTP client for all gateways

This client provides a clean interface for WebRTC, WebSocket, and API Gateway
to communicate with the Orchestrator Service, replacing direct usage of
LocalConversationPipeline.

Features:
- Simple async API
- Automatic error handling
- Consistent timeout management
- Health checking
- Statistics retrieval
- Binary protocol support for performance (1.65x-9.54x faster)
- Automatic failover from binary to JSON
"""

import aiohttp
import asyncio
import base64
import logging
from typing import Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add project root to path for Communication Manager
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.managers.communication_manager import ServiceCommunicationManager, Priority
from src.core.http_reliability import get_circuit_breaker

logger = logging.getLogger(__name__)


class OrchestratorClientError(Exception):
    """Exception raised when Orchestrator client operations fail"""
    pass


class OrchestratorClient:
    """
    HTTP client for Orchestrator Service

    Usage:
        client = OrchestratorClient()
        await client.initialize()

        result = await client.process_turn(
            audio_data=audio_bytes,
            session_id="session_123"
        )

        await client.cleanup()
    """

    def __init__(self, orchestrator_url: Optional[str] = None, use_binary_protocol: bool = True):
        """
        Initialize Orchestrator client

        Args:
            orchestrator_url: URL of orchestrator service
                            (default: from env ORCHESTRATOR_URL or http://localhost:8900)
            use_binary_protocol: Use binary protocol for better performance (default: True)
                               Falls back to JSON automatically if binary fails
        """
        self.orchestrator_url = (
            orchestrator_url or
            os.getenv("ORCHESTRATOR_URL", "http://localhost:8900")
        ).rstrip('/')

        self.use_binary_protocol = use_binary_protocol
        self.session: Optional[aiohttp.ClientSession] = None
        self.comm_manager: Optional[ServiceCommunicationManager] = None

        protocol_note = " (binary protocol enabled)" if use_binary_protocol else " (JSON only)"
        logger.info(f"🎯 OrchestratorClient initialized: {self.orchestrator_url}{protocol_note}")

    async def initialize(self):
        """
        Initialize HTTP session and Communication Manager
        Must be called before making requests
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()
            logger.info("✅ OrchestratorClient session created")

        if self.comm_manager is None and self.use_binary_protocol:
            self.comm_manager = ServiceCommunicationManager(self.session)
            await self.comm_manager.initialize()
            # Set orchestrator preference: binary primary, JSON fallback
            self.comm_manager.set_preference('orchestrator', primary='binary', fallback='json')
            logger.info("🔗 Communication Manager initialized for orchestrator")

    async def process_turn(self,
                          audio_data: bytes,
                          session_id: str,
                          sample_rate: int = 16000,
                          voice_id: Optional[str] = None,
                          force_external_llm: bool = False,
                          timeout: float = 60.0) -> Dict[str, Any]:
        """
        Process a conversation turn through the Orchestrator

        This is the main method that replaces LocalConversationPipeline.process_turn()

        Args:
            audio_data: Input audio bytes (WAV or PCM)
            session_id: Session identifier
            sample_rate: Audio sample rate (default 16000)
            voice_id: TTS voice ID (optional, uses session default if not provided)
            force_external_llm: Force use of external LLM (skip primary) for benchmarking
            timeout: Request timeout in seconds (default 60s)

        Returns:
            Dict containing:
                - success: bool - Whether processing succeeded
                - text: str - AI response text
                - audio: bytes - AI response audio (WAV)
                - transcript: str - User input transcript
                - session_id: str - Session identifier
                - llm_used: str - Which LLM was used ("primary" or "fallback")
                - voice_id: str - Voice ID used
                - metrics: Dict - Processing metrics
                - error: str - Error message (if success=False)

        Raises:
            OrchestratorClientError: If request fails or orchestrator returns error
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            # Use Communication Manager if binary protocol is enabled
            if self.use_binary_protocol and self.comm_manager:
                logger.debug(f"📤 Calling orchestrator via Communication Manager: session={session_id}, audio={len(audio_data)} bytes")

                # Prepare metadata
                metadata = {
                    "session_id": session_id,
                }
                if voice_id:
                    metadata["voice_id"] = voice_id
                if force_external_llm:
                    metadata["force_external_llm"] = force_external_llm

                # Call using Communication Manager with automatic protocol selection
                result = await self.comm_manager.call_audio_service(
                    service_name='orchestrator',
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    metadata=metadata,
                    priority=Priority.REALTIME,
                    endpoint_name='process_turn'
                )

                logger.debug(f"📥 Orchestrator response via {result.get('protocol_used', 'unknown')}: "
                            f"success={result.get('success')}, llm={result.get('llm_used')}")

                return result

            else:
                # Fallback to JSON-only mode (backward compatibility)
                logger.debug(f"📤 Calling orchestrator via JSON: session={session_id}, audio={len(audio_data)} bytes")

                # Encode audio to base64 for JSON transport
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # Prepare request payload
                payload = {
                    "audio": audio_b64,
                    "session_id": session_id,
                    "sample_rate": sample_rate
                }

                if voice_id:
                    payload["voice_id"] = voice_id
                if force_external_llm:
                    payload["force_external_llm"] = force_external_llm

                # Call orchestrator
                url = f"{self.orchestrator_url}/api/orchestrator/process-turn"

                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise OrchestratorClientError(
                            f"Orchestrator returned {resp.status}: {error_text}"
                        )

                    result = await resp.json()

                # Decode base64 audio response if present
                if result.get("success") and result.get("audio"):
                    audio_data_resp = result["audio"]
                    if isinstance(audio_data_resp, str):
                        result["audio"] = base64.b64decode(audio_data_resp)
                    elif isinstance(audio_data_resp, dict):
                        # Audio is a dict (possibly with metadata), keep as is
                        pass
                    # else: already bytes, keep as is

                logger.debug(f"📥 Orchestrator response: success={result.get('success')}, "
                            f"llm={result.get('llm_used')}")

                return result

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Orchestrator timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Orchestrator timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Orchestrator client error: {e}")
            raise OrchestratorClientError(f"Failed to call orchestrator: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in orchestrator client: {e}")
            raise OrchestratorClientError(f"Unexpected error: {e}")

    async def process_text_conversation(self,
                                        message: str,
                                        session_id: str,
                                        voice_id: Optional[str] = None,
                                        timeout: float = 30.0) -> Dict[str, Any]:
        """
        Process a text-only conversation through the Orchestrator

        Args:
            message: User message text
            session_id: Session identifier
            voice_id: Optional voice ID for TTS response
            timeout: Request timeout in seconds (default 30s)

        Returns:
            Dict containing:
                - success: bool - Whether processing succeeded
                - response: str - AI response text
                - session_id: str - Session identifier
                - context_size: int - Number of messages in context
                - messages_count: int - Total messages in conversation
                - metrics: Dict - Processing metrics
                - error: str - Error message (if success=False)

        Raises:
            OrchestratorClientError: If request fails or orchestrator returns error
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            logger.debug(f"📤 Calling orchestrator text conversation: session={session_id}, message_len={len(message)}")

            # Prepare request payload
            payload = {
                "message": message,
                "session_id": session_id
            }

            if voice_id:
                payload["voice_id"] = voice_id

            # Call orchestrator text conversation endpoint
            url = f"{self.orchestrator_url}/api/orchestrator/conversation"

            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise OrchestratorClientError(
                        f"Orchestrator text conversation failed ({resp.status}): {error_text}"
                    )

                result = await resp.json()

            logger.debug(f"📥 Orchestrator text response: success={result.get('success')}, "
                        f"context_size={result.get('context_size')}")

            return result

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Orchestrator text conversation timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Orchestrator text conversation timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Orchestrator text conversation client error: {e}")
            raise OrchestratorClientError(f"Failed to call orchestrator text conversation: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in orchestrator text conversation: {e}")
            raise OrchestratorClientError(f"Unexpected error: {e}")

    async def health_check(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Check health of orchestrator and downstream services

        Args:
            timeout: Request timeout in seconds (default 5s)

        Returns:
            Dict with:
                - status: str - Overall status ("healthy", "degraded", "unhealthy")
                - orchestrator: Dict - Orchestrator info
                - services: Dict[str, bool] - Health of each downstream service
                - fallback: Dict[str, bool] - Health of fallback components
                - circuit_breaker: Dict - Circuit breaker state
                - stats: Dict - Processing statistics

        Raises:
            OrchestratorClientError: If health check fails
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            url = f"{self.orchestrator_url}/api/orchestrator/health"

            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise OrchestratorClientError(
                        f"Health check failed ({resp.status}): {error_text}"
                    )

                return await resp.json()

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Health check timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Health check timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Health check error: {e}")
            raise OrchestratorClientError(f"Health check failed: {e}")

    async def get_stats(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Get orchestrator processing statistics

        Args:
            timeout: Request timeout in seconds (default 5s)

        Returns:
            Dict with statistics:
                - total_turns: int
                - successful_turns: int
                - failed_turns: int
                - primary_llm_count: int
                - fallback_llm_count: int
                - average_processing_time_ms: int
                - success_rate: float
                - primary_llm_rate: float

        Raises:
            OrchestratorClientError: If request fails
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            url = f"{self.orchestrator_url}/api/orchestrator/stats"

            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise OrchestratorClientError(
                        f"Stats request failed ({resp.status}): {error_text}"
                    )

                return await resp.json()

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Stats request timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Stats request timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Stats request error: {e}")
            raise OrchestratorClientError(f"Stats request failed: {e}")

    async def reset_stats(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Reset orchestrator statistics counters

        Args:
            timeout: Request timeout in seconds (default 5s)

        Returns:
            Dict with confirmation message

        Raises:
            OrchestratorClientError: If request fails
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            url = f"{self.orchestrator_url}/api/orchestrator/reset-stats"

            async with self.session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise OrchestratorClientError(
                        f"Reset stats failed ({resp.status}): {error_text}"
                    )

                return await resp.json()

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Reset stats timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Reset stats timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Reset stats error: {e}")
            raise OrchestratorClientError(f"Reset stats failed: {e}")

    async def reset_circuit_breaker(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Manually reset the circuit breaker (force primary LLM retry)

        Args:
            timeout: Request timeout in seconds (default 5s)

        Returns:
            Dict with confirmation message

        Raises:
            OrchestratorClientError: If request fails
        """
        if not self.session:
            raise OrchestratorClientError("Client not initialized. Call initialize() first.")

        try:
            url = f"{self.orchestrator_url}/api/orchestrator/reset-circuit-breaker"

            async with self.session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise OrchestratorClientError(
                        f"Reset circuit breaker failed ({resp.status}): {error_text}"
                    )

                return await resp.json()

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Reset circuit breaker timeout (exceeded {timeout}s): {e}")
            raise OrchestratorClientError(f"Reset circuit breaker timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Reset circuit breaker error: {e}")
            raise OrchestratorClientError(f"Reset circuit breaker failed: {e}")

    async def cleanup(self):
        """
        Cleanup HTTP session and Communication Manager
        Should be called when client is no longer needed
        """
        if self.comm_manager:
            await self.comm_manager.cleanup()
            self.comm_manager = None
            logger.info("✅ Communication Manager cleaned up")

        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ OrchestratorClient session closed")

    async def __aenter__(self):
        """Support for async context manager"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager"""
        await self.cleanup()


# Convenience function for simple usage
async def create_orchestrator_client(orchestrator_url: Optional[str] = None) -> OrchestratorClient:
    """
    Create and initialize an OrchestratorClient

    Args:
        orchestrator_url: Optional URL override

    Returns:
        Initialized OrchestratorClient

    Example:
        client = await create_orchestrator_client()
        result = await client.process_turn(audio_data, session_id)
        await client.cleanup()
    """
    client = OrchestratorClient(orchestrator_url)
    await client.initialize()
    return client
