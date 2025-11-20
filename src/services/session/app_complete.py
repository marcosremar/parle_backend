"""
Session Service Standalone - Consolidated for Nomad deployment
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

# Try to import local utils (fallback to core if not available)
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
        "name": "session",
        "port": 8600,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get session service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "8600"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Session Service",
    version="1.0.0",
    description="Session Service for Parle Backend"
)

# Add telemetry middleware
try:
    # add_telemetry_middleware removed, "session")
except Exception as e:
    logger.warning(f"Failed to add telemetry middleware: {e}")

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the session service"""
    global service
    try:
        logger.info("🚀 Initializing Session Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Try to import and initialize service
        try:
            from src.services.session.service import SessionService
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager
            
            # Create minimal ServiceContext
            try:
                comm = ServiceCommunicationManager()
            except Exception:
                class MockComm:
                    def get_service_url(self, service_name): return None
                    def send_request(self, *args, **kwargs): return None
                comm = MockComm()
            
            context = ServiceContext.create(
                service_name="session",
                comm=comm,
                config={"name": "session", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"
            )
            
            service = SessionService(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ Session Service initialized successfully")
                return True
        except ImportError as e:
            logger.warning(f"⚠️  Service class not found: {e}")
            logger.info("   Running in minimal mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e}")
            logger.info("   Running in minimal mode")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Session Service: {e}")
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
        "service": "session",
        "version": "1.0.0",
        "status": "running",
        "description": "Session Service"
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
        "service": "session",
        "initialized": service is not None
    }

# Try to mount service routes
try:
    from src.services.session.routes import router as service_router
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
        add_standard_endpoints(router, service, "session")
    else:
        add_standard_endpoints(router, None, "session")
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
    port = int(os.getenv("PORT", "8600"))
    logger.info(f"🚀 Starting Session Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
