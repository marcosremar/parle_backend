"""
Orchestrator Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, APIRouter
from typing import Dict, Optional, Any
from datetime import datetime
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
        "name": "orchestrator",
        "port": 8500,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get orchestrator service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "8500"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Orchestrator Service",
    version="1.0.0",
    description="Orchestrator Service for Parle Backend"
)

# Telemetry middleware removed - services are now independent

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the orchestrator service"""
    global service
    try:
        logger.info("🚀 Initializing Orchestrator Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Try to import and initialize service
        try:
            from src.services.orchestrator.service import OrchestratorService
            from .utils.unified_context import ServiceContext
            
            # Create minimal ServiceContext (no communication manager needed - using HTTP directly)
            context = ServiceContext.create(
                service_name="orchestrator",
                comm=None,  # Services communicate via HTTP directly
                config={"name": "orchestrator", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"
            )
            
            service = OrchestratorService(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ Orchestrator Service initialized successfully")
                return True
        except ImportError as e:
            logger.warning(f"⚠️  Service class not found: {e}")
            logger.info("   Running in minimal mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e}")
            logger.info("   Running in minimal mode")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Orchestrator Service: {e}")
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
        "service": "orchestrator",
        "version": "1.0.0",
        "status": "running",
        "description": "Orchestrator Service"
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint (liveness)"""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "initialized": service is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/live")
async def liveness_probe():
    """Liveness probe - checks if service is alive"""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "probe": "liveness",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/ready")
async def readiness_probe():
    """Readiness probe - checks if service is ready to accept traffic"""
    if not service or not hasattr(service, 'orchestrator') or not service.orchestrator:
        return {
            "status": "unhealthy",
            "service": "orchestrator",
            "probe": "readiness",
            "message": "Service not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Check dependencies
        health_status = await service.orchestrator._health_check_services()
        
        # Determine if ready
        critical_services = ["llm", "stt", "tts"]
        critical_healthy = all(health_status.get(s, False) for s in critical_services)
        
        if critical_healthy:
            status = "healthy"
            message = "Service is ready"
        else:
            status = "degraded"
            message = "Service is ready but some dependencies are unhealthy"
        
        return {
            "status": status,
            "service": "orchestrator",
            "probe": "readiness",
            "message": message,
            "dependencies": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "orchestrator",
            "probe": "readiness",
            "message": f"Readiness check failed: {e}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with comprehensive information"""
    if not service or not hasattr(service, 'orchestrator') or not service.orchestrator:
        return {
            "status": "unhealthy",
            "service": "orchestrator",
            "message": "Service not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Check dependencies
        health_status = await service.orchestrator._health_check_services()
        
        # Collect metrics
        metrics = {}
        if service.orchestrator:
            metrics["clients_initialized"] = len(service.orchestrator.clients)
            metrics["in_process_mode"] = service.orchestrator.in_process_mode
        
        # Determine overall status
        critical_services = ["llm", "stt", "tts"]
        critical_healthy = all(health_status.get(s, False) for s in critical_services)
        all_healthy = all(health_status.values())
        
        if all_healthy:
            status = "healthy"
            message = "All systems operational"
        elif critical_healthy:
            status = "degraded"
            message = "Critical services healthy, some non-critical services unavailable"
        else:
            status = "unhealthy"
            message = "Critical services unavailable"
        
        return {
            "status": status,
            "service": "orchestrator",
            "message": message,
            "dependencies": health_status,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "orchestrator",
            "message": f"Health check failed: {e}",
            "timestamp": datetime.now().isoformat()
        }

# Try to mount service routes
try:
    from src.services.orchestrator.routes import create_router
    if service:
        service_router = create_router(service)
        app.include_router(service_router, prefix="/api", tags=["orchestrator"])
        logger.info("✅ Service routes mounted with orchestrator service")
    else:
        service_router = create_router(None)
        app.include_router(service_router, prefix="/api", tags=["orchestrator"])
        logger.info("✅ Service routes mounted (basic mode)")
except ImportError:
    logger.warning("⚠️  Service routes not found - using basic endpoints only")
except Exception as e:
    logger.warning(f"⚠️  Failed to mount service routes: {e}")

# Add standard endpoints
router = APIRouter()
try:
    if service:
        add_standard_endpoints(router, service, "orchestrator")
    else:
        add_standard_endpoints(router, None, "orchestrator")
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
    port = int(os.getenv("ORCHESTRATOR_PORT", os.getenv("PORT", "8500")))
    logger.info(f"🚀 Starting Orchestrator Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
