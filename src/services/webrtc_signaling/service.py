"""
WebRTC Signaling Service - BaseService Implementation
Handles WebRTC peer connection signaling via WebSocket
"""

import sys
import asyncio
from pathlib import Path

# Add project to path FIRST (before src.core imports)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
, increment_metric, set_gauge
from loguru import logger

# Setup logging and metrics for Webrtc Signaling Service
.parent / "tmp" / "metrics")

from typing import Dict, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from .utils.base_service import BaseService

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext
from src.core.shared.models.config_models import WebRTCSignalingConfig, PortConfig

# WebRTC support
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.media import MediaBlackhole
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("⚠️ aiortc not available, WebRTC will use simplified SDP")

class SignalingManager:
    """Manages WebRTC signaling connections"""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.peer_mappings: Dict[str, str] = {}  # client_id -> peer_id

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Register new client connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"✅ Client connected: {client_id}")
        logger.info(f"📊 Active connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str) -> None:
        """Disconnect client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.peer_mappings:
            del self.peer_mappings[client_id]
        logger.info(f"❌ Client disconnected: {client_id}")
        logger.info(f"📊 Active connections: {len(self.active_connections)}")

    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
                return True
            except Exception as e:
                increment_metric("service_initializations", "webrtc_signaling", status="error")
                logger.error(f"Error sending to {client_id}: {e}")
                return False

    async def broadcast(self, message: dict, exclude: str = None) -> None:
        """Broadcast message to all clients except excluded"""
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")

    def get_peer_id(self, client_id: str) -> str:
        """Get peer ID for client"""
        return self.peer_mappings.get(client_id)

    def set_peer_mapping(self, client_id: str, peer_id: str) -> None:
        """Set peer mapping"""
        self.peer_mappings[client_id] = peer_id

class WebRTCSignalingService(BaseService):
    """WebRTC Signaling Service using BaseService"""

    def __init__(self, config: Dict = None, context: Optional[ServiceContext] = None) -> None:
        # Pass context to BaseService (DI support)
        super().__init__(context=context, config=config)

        # Load configuration from SettingsService (via DI)
        self.webRTCSignaling_config: Optional[WebRTCSignalingConfig] = None
        if self.settings:
            try:
                self.webRTCSignaling_config = WebRTCSignalingConfig.from_settings(self.settings)
                self.logger.info(f"✅ WebRTCSignalingConfig loaded via SettingsService")
            except Exception as e:
                self.logger.error(f"Failed to load WebRTCSignalingConfig from SettingsService: {e}")
                self.webRTCSignaling_config = WebRTCSignalingConfig()  # Use defaults

        # Log DI status
        if self.context:
            self.logger.info("🎯 WebRTC Signaling Service initialized with ServiceContext (DI enabled)")
        else:
            self.logger.warning("⚠️  WebRTC Signaling Service initialized without ServiceContext (legacy mode)")

        self.signaling_manager = SignalingManager()

        # Store peer connections for each client
        self.peer_connections: Dict[str, RTCPeerConnection] = {}

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
        from .routes import create_router

        router = create_router(self)
        self.router.include_router(router)

    async def initialize(self) -> bool:
        """Initialize WebRTC Signaling service"""
        try:
            # Update metrics
            increment_metric("service_initializations", "webrtc_signaling", status="success")

            self.logger.info("✅ WebRTC Signaling Service initialized")
            return True
        except Exception as e:
            increment_metric("service_initializations", "webrtc_signaling", status="error")
            self.logger.error(f"❌ Failed to initialize WebRTC Signaling Service: {e}")
            return False

    async def health_check(self) -> Dict:
        """Perform health check"""
        return {
            "status": "healthy",
            "active_connections": len(self.signaling_manager.active_connections),
            "timestamp": datetime.now().isoformat()
        }

    async def add_ice_candidate(self, client_id: str, candidate_data: dict) -> bool:
        """
        Add ICE candidate to peer connection

        Args:
            client_id: Client identifier
            candidate_data: ICE candidate data from client

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if client_id not in self.peer_connections:
                self.logger.warning(f"⚠️ No peer connection for {client_id}")
                return False

            if not AIORTC_AVAILABLE:
                self.logger.warning("⚠️ aiortc not available, ignoring ICE candidate")
                return False

            pc = self.peer_connections[client_id]

            # aiortc handles ICE candidates automatically during offer/answer
            # We don't need to manually add them
            self.logger.info(f"🧊 ICE candidate received for {client_id} (handled automatically by aiortc)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to add ICE candidate for {client_id}: {e}")
            return False

    async def generate_answer(self, client_sdp: dict, client_id: str) -> dict:
        """
        Generate a valid SDP answer from the client's offer using aiortc

        Args:
            client_sdp: Client's SDP offer (dict with 'type' and 'sdp' keys)
            client_id: Client identifier

        Returns:
            dict: SDP answer with 'type' and 'sdp' keys
        """
        try:
            self.logger.info(f"🔧 Generating SDP answer for client: {client_id}")

            if not AIORTC_AVAILABLE:
                raise RuntimeError("aiortc is not available - cannot generate WebRTC answer")

            # Extract the SDP string from the offer
            offer_sdp = client_sdp.get("sdp", "")

            if not offer_sdp or not isinstance(offer_sdp, str):
                raise ValueError(f"Invalid SDP offer: {type(offer_sdp)}")

            # Create RTCPeerConnection for this client
            pc = RTCPeerConnection(
                configuration=RTCConfiguration(
                    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
                )
            )

            # Store the peer connection
            self.peer_connections[client_id] = pc

            # Setup data channel handler
            @pc.on("datachannel")
            def on_datachannel(channel):
                self.logger.info(f"📡 Data channel established for {client_id}: {channel.label}")

                @channel.on("message")
                def on_message(message):
                    self.logger.info(f"📨 Data channel message from {client_id}: {len(message) if isinstance(message, bytes) else message}")

            # Set remote description (client's offer)
            offer = RTCSessionDescription(sdp=offer_sdp, type=client_sdp.get("type", "offer"))
            await pc.setRemoteDescription(offer)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Return answer as dict
            answer_dict = {
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp
            }

            self.logger.info(f"✅ Generated real WebRTC answer for {client_id}")
            return answer_dict

        except Exception as e:
            self.logger.error(f"❌ Failed to generate SDP answer: {e}", exc_info=True)
            # Clean up peer connection on error
            if client_id in self.peer_connections:
                await self.peer_connections[client_id].close()
                del self.peer_connections[client_id]
            raise

    async def shutdown(self):
        """Cleanup resources"""
        # Close all peer connections
        for client_id, pc in list(self.peer_connections.items()):
            try:
                await pc.close()
                self.logger.info(f"🔌 Closed peer connection for {client_id}")
            except Exception as e:
                self.logger.error(f"Error closing peer connection for {client_id}: {e}")
        self.peer_connections.clear()

        # Close all WebSocket connections
        for client_id in list(self.signaling_manager.active_connections.keys()):
            self.signaling_manager.disconnect(client_id)
        self.logger.info("🛑 WebRTC Signaling Service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    import os
    from fastapi import FastAPI
    # telemetry_middleware removed import add_telemetry_middleware

    port = int(os.getenv("WEBRTC_SIGNALING_PORT", "8021"))  # Dynamic allocation supported via PortPool

    config = {
        "name": "webrtc_signaling",
        "port": port,
        "host": "0.0.0.0"
    }

    service = WebRTCSignalingService(config)

    # Create FastAPI app
    app = FastAPI(title="WebRTC Signaling Service")
    # add_telemetry_middleware removed, "webrtc_signaling")

    # Add basic health endpoint for testing
    @app.get("/health")
    async def health():
        return await service.health_check()

    # Include service router
    app.include_router(service.get_router())

    uvicorn.run(app, host="0.0.0.0", port=port)
