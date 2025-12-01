"""
Communication Service Clients

Clients for WebSocket, WebRTC, REST polling, and neural codec services.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .base import BaseServiceClient, ServiceClientError

logger = logging.getLogger(__name__)


class WebSocketClient(BaseServiceClient):
    """WebSocket service client for real-time communication"""

    def __init__(self) -> None:
        super().__init__("websocket")

    async def send_notification(self, user_id: str, message: dict[str, Any]) -> bool:
        """Send real-time notification via WebSocket"""
        try:
            await self._post(f"/api/notify/{user_id}", json_data=message)
            logger.debug(f"✅ Sent WebSocket notification to {user_id}")
            return True
        except ServiceClientError:
            logger.warning("⚠️ Failed to send WebSocket notification")
            return False

    async def broadcast(self, message: dict[str, Any]) -> bool:
        """Broadcast message to all connected clients"""
        try:
            await self._post("/api/broadcast", json_data=message)
            logger.debug("✅ Broadcasted message via WebSocket")
            return True
        except ServiceClientError:
            return False


class RestPollingClient(BaseServiceClient):
    """REST polling service client"""

    def __init__(self) -> None:
        super().__init__("rest_polling")

    async def poll(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Poll an endpoint"""
        try:
            query_string = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
            path = f"{endpoint}?{query_string}" if query_string else endpoint
            return await self._get(path)
        except ServiceClientError:
            return {}


class NeuralCodecClient(BaseServiceClient):
    """Neural codec service client for audio processing"""

    def __init__(self) -> None:
        super().__init__("neural_codec")

    async def encode_audio(self, audio_data: bytes) -> bytes:
        """Encode audio using neural codec"""
        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            result = await self._post("/api/encode", json_data={"audio": audio_base64})
            encoded_base64 = result.get("encoded", "")
            return base64.b64decode(encoded_base64)
        except ServiceClientError:
            logger.warning("⚠️ Neural codec encoding failed")
            return audio_data  # Return original on failure

    async def decode_audio(self, encoded_data: bytes) -> bytes:
        """Decode audio using neural codec"""
        try:
            encoded_base64 = base64.b64encode(encoded_data).decode("utf-8")
            result = await self._post("/api/decode", json_data={"encoded": encoded_base64})
            audio_base64 = result.get("audio", "")
            return base64.b64decode(audio_base64)
        except ServiceClientError:
            logger.warning("⚠️ Neural codec decoding failed")
            return encoded_data  # Return original on failure


class WebRTCClient(BaseServiceClient):
    """WebRTC service client"""

    def __init__(self) -> None:
        super().__init__("webrtc")

    async def create_offer(self, session_id: str) -> dict[str, Any]:
        """Create WebRTC offer"""
        try:
            result = await self._post(f"/api/offer/{session_id}", json_data={})
            return result
        except ServiceClientError:
            return {}

    async def handle_answer(self, session_id: str, answer: dict[str, Any]) -> bool:
        """Handle WebRTC answer"""
        try:
            await self._post(f"/api/answer/{session_id}", json_data=answer)
            return True
        except ServiceClientError:
            return False


class WebRTCSignalingClient(BaseServiceClient):
    """WebRTC signaling service client"""

    def __init__(self) -> None:
        super().__init__("webrtc_signaling")

    async def send_signal(self, session_id: str, signal: dict[str, Any]) -> bool:
        """Send signaling message"""
        try:
            await self._post(f"/api/signal/{session_id}", json_data=signal)
            return True
        except ServiceClientError:
            return False

    async def get_signals(self, session_id: str) -> list[dict[str, Any]]:
        """Get signaling messages for session"""
        try:
            result = await self._get(f"/api/signals/{session_id}")
            return result.get("signals", [])
        except ServiceClientError:
            return []


class ViberGatewayClient(BaseServiceClient):
    """Viber gateway service client"""

    def __init__(self) -> None:
        super().__init__("viber_gateway")

    async def send_message(self, user_id: str, message: str) -> bool:
        """Send message via Viber"""
        try:
            await self._post("/api/send", json_data={"user_id": user_id, "message": message})
            logger.debug(f"✅ Sent Viber message to {user_id}")
            return True
        except ServiceClientError:
            logger.warning("⚠️ Failed to send Viber message")
            return False

    async def receive_message(self, message_data: dict[str, Any]) -> dict[str, Any]:
        """Process received Viber message"""
        try:
            result = await self._post("/api/receive", json_data=message_data)
            return result
        except ServiceClientError:
            return {}
