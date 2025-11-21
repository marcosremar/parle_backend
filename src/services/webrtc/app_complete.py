"""
Webrtc Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, APIRouter
from typing import Dict, Optional, Any
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router, service=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "webrtc",
        "port": 10100,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get webrtc service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "10100"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Webrtc Service",
    version="1.0.0",
    description="Webrtc Service for Parle Backend"
)

# Add telemetry middleware
# add_telemetry_middleware removed
# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the webrtc service"""
    global service
    try:
        logger.info("🚀 Initializing Webrtc Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Try to import and initialize service
        try:
            from src.services.webrtc.service import WebrtcService
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager
            
            # Create minimal ServiceCommunicationManager
            class MinimalServiceCommunicationManager:
                def get_service_url(self, service_name): 
                    return None
                def send_request(self, *args, **kwargs): 
                    return None
            
            comm = MinimalServiceCommunicationManager()
            
            context = ServiceContext.create(
                service_name="webrtc",
                comm=comm,
                config={"name": "webrtc", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"
            )
            
            service = WebrtcService(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ Webrtc Service initialized successfully")
                return True
            logger.info("   Running in minimal mode")
        except ImportError:
            logger.warning("⚠️  Service classes not found - running in minimal mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e} - running in minimal mode")
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "webrtc",
        "version": "1.0.0",
        "status": "running",
        "description": "Webrtc Service"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service:
        try:
            return await service.health_check()
        except Exception as e:
            logger.warning(f"Health check error: {e}")
    
    return {
        "status": "healthy",
        "service": "webrtc",
        "initialized": service is not None
    }

# Try to mount service routes
try:
    from src.services.webrtc.routes import router as service_router
    app.include_router(service_router)
    logger.info("✅ Service routes mounted")
except ImportError:
    logger.warning("⚠️  Service routes not found - using basic endpoints only")

# Add standard endpoints
router = APIRouter()
if service:
    add_standard_endpoints(router, service, "webrtc")
else:
    add_standard_endpoints(router, None, "webrtc")
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Startup event - initialize service"""
    await initialize_service()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10100"))
    logger.info(f"🚀 Starting Webrtc Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
