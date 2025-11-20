"""
WebRTC Gateway Service - BaseService Implementation
Handles WebRTC communication for real-time media
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
from loguru import logger

# Setup logging and metrics for WebRTC Service
.parent / "tmp" / "metrics")

from .utils.base_service import BaseService

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext
from src.core.shared.models.config_models import WebRTCConfig, PortConfig

class WebRTCService(BaseService):
    """WebRTC Gateway Service using BaseService"""

    def __init__(self, config: Dict = None, context: Optional[ServiceContext] = None) -> None:
        # Initialize attributes BEFORE calling super().__init__()
        # (because BaseService.__init__ calls _setup_router which needs these)
        self.orchestrator_url = None  # Set during initialize()
        self.websockets: Set[WebSocket] = set()
        self.start_time = time.time()
        self.task_integration_available = False

        # Pass context to BaseService (DI support)
        super().__init__(context=context, config=config)

        # Load configuration from SettingsService (via DI)
        self.webRTC_config: Optional[WebRTCConfig] = None
        if self.settings:
            try:
                self.webRTC_config = WebRTCConfig.from_settings(self.settings)
                self.logger.info(f"✅ WebRTCConfig loaded via SettingsService")
            except Exception as e:
                self.logger.error(f"Failed to load WebRTCConfig from SettingsService: {e}")
                self.webRTC_config = WebRTCConfig()  # Use defaults

        # Log DI status
        if self.context:
            self.logger.info("🎯 WebRTC Service initialized with ServiceContext (DI enabled)")
            self.logger.info(f"   - Logger: ✅ injected (scoped)")
            self.logger.info(f"   - Communication: ✅ injected ({type(self.comm).__name__})")
        else:
            self.logger.warning("⚠️  WebRTC Service initialized without ServiceContext (legacy mode)")

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
        from .routes import create_router

        router = create_router(self)
        self.router.include_router(router)

    async def _handle_ws_message(self, websocket: WebSocket, message: dict) -> None:
        """Handle WebSocket text message"""
        msg_type = message.get("type")

        if msg_type == "ping":
            await websocket.send_json({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })

        elif msg_type == "offer":
            # WebRTC signaling - echo for now
            await websocket.send_json({
                "type": "answer",
                "sdp": message.get("sdp")
            })

        elif msg_type == "ice-candidate":
            # Echo ICE candidate
            await websocket.send_json({
                "type": "ice-candidate",
                "candidate": message.get("candidate")
            })

    async def _handle_ws_audio(self, websocket: WebSocket, audio_data: bytes) -> None:
        """Handle WebSocket binary audio data"""
        try:
            import base64

            # Encode audio to base64 for HTTP transport
            audio_b64 = base64.b64encode(audio_data).decode()

            # Call orchestrator service via Communication Manager (fast!)
            result = await self.comm.call_service(
                service_name="orchestrator",
                endpoint_path="/process-turn",
                method="POST",
                json_data={
                    "audio": audio_b64,
                    "session_id": "webrtc_ws_session",
                    "sample_rate": 16000
                },
                timeout=30.0
            )

            # Send result back
            await websocket.send_json({
                "type": "audio_response",
                "success": result.get("success", False),
                "transcript": result.get("transcript"),
                "response": result.get("text"),
                "audio": result.get("audio"),  # Already base64 encoded
                "error": result.get("error")
            })

        except Exception as e:
            self.logger.error(f"Error processing WebSocket audio: {e}")
            await websocket.send_json({
                "type": "audio_response",
                "success": False,
                "error": str(e)
            })

    async def initialize(self) -> bool:
        """Initialize WebRTC service"""
        try:
            import os
            # WebRTC is just an interface - orchestrator service handles the actual processing
            # No heavy initialization needed here
            # ✅ Phase 8: Use environment variable for service manager URL
            self.orchestrator_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888") + '/api/orchestrator'

            self.logger.info("✅ WebRTC Service initialized (lightweight mode)")
            self.logger.info(f"   Will call orchestrator at: {self.orchestrator_url}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize WebRTC Service: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    async def health_check(self) -> Dict:
        """Perform health check"""
        uptime = time.time() - self.start_time
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "orchestrator_configured": self.orchestrator_url is not None,
            "websocket_connections": len(self.websockets),
            "timestamp": datetime.now().isoformat()
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        # Close all WebSocket connections
        for ws in list(self.websockets):
            try:
                await ws.close()
            except Exception as e:
                # WebSocket already closed or connection error
                self.logger.debug(f"Failed to close WebSocket: {e}")
        self.websockets.clear()

        self.logger.info("🛑 WebRTC Service shutdown complete")

# For standalone execution
if __name__ == "__main__":
    import uvicorn
    import asyncio
    from fastapi import FastAPI
    from src.core.utils.port_manager import ensure_port_available
    # telemetry_middleware removed import add_telemetry_middleware

    # Create service with default config
    port = int(os.getenv("WEBRTC_PORT", "8020"))

    # Ensure port is available
    logger.info(f"Ensuring port {port} is available...")
    ensure_port_available(port)

    config = {
        "name": "webrtc",
        "port": port,
        "host": "0.0.0.0"
    }

    service = WebRTCService(config)

    # Initialize
    async def init_service():
        await service.initialize()

    # Skip initialization to avoid dependency timeouts in test mode
    # asyncio.run(init_service())

    # Create FastAPI app
    app = FastAPI(title="WebRTC Service")
    # add_telemetry_middleware removed, "webrtc")

    # Add basic health endpoint for testing (works without full initialization)
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "webrtc",
            "initialized": False  # Standalone test mode
        }

    # Include service router
    app.include_router(service.get_router())

    # Run server
    print(f"Starting WebRTC Service on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
