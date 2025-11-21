"""
Rest Polling Service Standalone - Consolidated for Nomad deployment
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
    try:
        from src.core.route_helpers import add_standard_endpoints
        from src.core.metrics import increment_metric, set_gauge
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
        "name": "rest_polling",
        "port": 8700,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get rest_polling service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "8700"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Rest Polling Service",
    version="1.0.0",
    description="Rest Polling Service for Parle Backend"
)

# Add telemetry middleware
# Add telemetry middleware
# add_telemetry_middleware removed

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the rest_polling service"""
    global service
    try:
        logger.info("🚀 Initializing Rest Polling Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Try to import and initialize service
        try:
            from src.services.rest_polling.service import RestpollingService
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager
            
            # Create minimal ServiceContext
            try:
                comm = ServiceCommunicationManager()
            except:
                # Fallback mock communication manager
                class MockComm:
                    def get_service_url(self, service_name): return None
                    def send_request(self, *args, **kwargs): return None
                comm = MockComm()
            
            context = ServiceContext.create(
                service_name="rest_polling",
                comm=comm,
                config={"name": "rest_polling", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"
            )
            
            service = RestpollingService(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ Rest Polling Service initialized successfully")
                return True
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e}")
    except Exception as e:
        logger.warning(f"⚠️  Service initialization error: {e}")
    logger.info("   Running in minimal mode")
    return True
# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "rest_polling",
        "version": "1.0.0",
        "status": "running",
        "description": "Rest Polling Service"
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
        "service": "rest_polling",
        "initialized": service is not None
    }

# Try to mount service routes
try:
    from src.services.rest_polling.routes import router as service_router
    app.include_router(service_router)
    logger.info("✅ Service routes mounted")
except Exception as e:
    logger.warning(f"⚠️  Failed to mount service routes: {e}")

# Add standard endpoints
router = APIRouter()
try:
    if service:
        add_standard_endpoints(router, service, "rest_polling")
    else:
        add_standard_endpoints(router, None, "rest_polling")
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
    port = int(os.getenv("PORT", "8700"))
    logger.info(f"🚀 Starting Rest Polling Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
