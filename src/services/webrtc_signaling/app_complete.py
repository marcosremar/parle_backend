"""
Webrtc Signaling Service Standalone - Consolidated for Nomad deployment
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

        pass

    def add_standard_endpoints(router, service=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "webrtc_signaling",
        "port": 10200,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get webrtc_signaling service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "10200"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Webrtc Signaling Service",
    version="1.0.0",
    description="Webrtc Signaling Service for Parle Backend"
)

# Telemetry middleware removed - services are now independent

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the webrtc_signaling service"""
    global service
    try:
        logger.info("🚀 Initializing Webrtc Signaling Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Try to import and initialize service
        try:
            from src.services.webrtc_signaling.service import WebrtcsignalingService
            from src.services.orchestrator.utils.context import ServiceContext
            # Communication manager removed - services use HTTP directly
            
            # Create minimal ServiceContext (no communication manager needed - using HTTP directly)
            context = ServiceContext.create(
                service_name="webrtc_signaling",
                comm=None,  # Services communicate via HTTP directly
                config={"name": "webrtc_signaling", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"
            )
            
            service = WebrtcsignalingService(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ Webrtc Signaling Service initialized successfully")
                return True
        except ImportError as e:
            logger.warning(f"⚠️  Service class not found: {e}")
            logger.info("   Running in minimal mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e}")
            logger.info("   Running in minimal mode")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Webrtc Signaling Service: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "webrtc_signaling",
        "version": "1.0.0",
        "status": "running",
        "description": "Webrtc Signaling Service"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service:
        try:
            return await service.health_check()
        except Exception:
            pass
    
    return {
        "status": "healthy",
        "service": "webrtc_signaling",
        "initialized": service is not None
    }

# Try to mount service routes
try:
    from src.services.webrtc_signaling.routes import router as service_router
    app.include_router(service_router)
    logger.info("✅ Service routes mounted")
except ImportError:
    logger.warning("⚠️  Service routes not found - using basic endpoints only")
except Exception as e:
    logger.warning(f"⚠️  Failed to mount service routes: {e}")

# Add standard endpoints
router = APIRouter()
try:
    if service:
        add_standard_endpoints(router, service, "webrtc_signaling")
    else:
        add_standard_endpoints(router, None, "webrtc_signaling")
except Exception as e:
    logger.warning(f"⚠️  Failed to add standard endpoints: {e}")

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
    port = int(os.getenv("PORT", "10200"))
    logger.info(f"🚀 Starting Webrtc Signaling Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
