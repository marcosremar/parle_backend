"""
API Gateway Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI, APIRouter
from typing import Dict, Any
from loguru import logger

# Add project root to path for core imports
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
    except ImportError as e:
        logger.warning(f"Some core modules not available: {e}")
        # Fallback implementations for standalone mode
        def increment_metric(name, value=1, labels=None):
            pass

        def set_gauge(name, value, labels=None):
            pass

        def add_standard_endpoints(router, service_instance=None, service_name=None):
            pass

# Import API Gateway service
try:
    from src.services.api_gateway.service import APIGatewayService
    from src.services.api_gateway.routes import create_router
except ImportError as e:
    logger.error(f"Failed to import API Gateway service: {e}")
    raise

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "api_gateway",
        "port": 8010,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get API Gateway service configuration"""
    config = DEFAULT_CONFIG.copy()
    # Override with environment variables
    port = int(os.getenv("PORT", "8010"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="API Gateway Service",
    version="1.0.0",
    description="Unified API Gateway for speech-to-speech processing"
)

# Add telemetry middleware
try:
    # add_telemetry_middleware removed, "api_gateway")
except Exception as e:
    logger.warning(f"Failed to add telemetry middleware: {e}")

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service: APIGatewayService = None

async def initialize_service():
    """Initialize the API Gateway service"""
    global service
    try:
        logger.info("🚀 Initializing API Gateway Service...")
        logger.info(f"   Port: {config['service']['port']}")
        
        # Create minimal ServiceContext for standalone mode
        try:
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager
            
            # Create a minimal communication manager for standalone
            try:
                comm = ServiceCommunicationManager()
            except Exception:
                # If communication manager fails, create a mock
                class MockComm:
                    def get_service_url(self, service_name): return None
                    def send_request(self, *args, **kwargs): return None
                comm = MockComm()
            
            # Create ServiceContext using factory method
            context = ServiceContext.create(
                service_name="api_gateway",
                comm=comm,
                config={"name": "api_gateway", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"  # Standalone service
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to create ServiceContext: {e}")
            logger.warning("   Trying alternative approach...")
            # Try direct instantiation as fallback
            try:
                from src.core.unified_context import ServiceContext
                from loguru import logger as loguru_logger
                context = ServiceContext(
                    service_name="api_gateway",
                    logger=loguru_logger,
                    comm=None,  # Will be set to None
                    config={"name": "api_gateway", "port": config["service"]["port"]}
                )
            except Exception as e2:
                logger.error(f"❌ Failed to create ServiceContext: {e2}")
                context = None
        
        # Create service configuration
        service_config = {
            "name": config["service"]["name"],
            "port": config["service"]["port"],
            "host": config["service"]["host"]
        }
        
        # Create service instance with context
        if context:
            service = APIGatewayService(config=service_config, context=context)
        else:
            # Fallback - try without context (may fail)
            service = APIGatewayService(config=service_config)
        
        # Initialize service
        success = await service.initialize()
        if not success:
            logger.error("❌ Failed to initialize API Gateway Service")
            return False
        
        logger.info("✅ API Gateway Service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API Gateway Service: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service:
        return await service.health_check()
    return {
        "status": "unhealthy",
        "service": "api-gateway",
        "message": "Service not initialized"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "api-gateway",
        "version": "1.0.0",
        "status": "running",
        "description": "Unified API Gateway for speech-to-speech processing"
    }

# Mount service router after initialization
@app.on_event("startup")
async def startup():
    """Startup event - initialize service and mount routes"""
    global service
    
    logger.info("🚀 Starting API Gateway Service...")
    logger.info(f"   Port: {config['service']['port']}")
    
    # Initialize service
    success = await initialize_service()
    if not success:
        logger.error("❌ Service initialization failed")
        return
    
    # Mount service router
    if service:
        try:
            # Get router from service
            service_router = service.get_router()
            if service_router:
                app.include_router(service_router)
                logger.info("✅ Service router mounted")
            
            # Also include core routes
            try:
                core_router = create_router(service)
                if core_router is not None:
                    app.include_router(core_router)
                    logger.info("✅ Core router mounted")
                else:
                    logger.warning("⚠️  create_router returned None - skipping core router")
            except Exception as e:
                logger.warning(f"⚠️  Failed to mount core router: {e}")
            
            # Try to mount additional routers
            try:
                from src.services.api_gateway.routers import health, scenarios, session, internal, rest_polling
                
                # Mount health router
                try:
                    app.include_router(health.router)
                    logger.info("✅ Health router mounted")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to mount health router: {e}")
                
                # Mount essential routers
                try:
                    app.include_router(scenarios.router)
                    logger.info("✅ Scenarios router mounted")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to mount scenarios router: {e}")
                
                try:
                    app.include_router(session.router)
                    logger.info("✅ Session router mounted")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to mount session router: {e}")
                
                try:
                    app.include_router(internal.router)
                    logger.info("✅ Internal router mounted")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to mount internal router: {e}")
                
                try:
                    app.include_router(rest_polling.router)
                    logger.info("✅ REST Polling router mounted")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to mount rest_polling router: {e}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to import additional routers: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to mount service router: {e}")
            import traceback
            traceback.print_exc()
    
    # Add standard endpoints
    try:
        router = APIRouter()
        # Create a minimal service instance for health checks
        class MinimalService:
            async def health_check(self):
                return {"status": "healthy", "service": "api_gateway"}
            def get_service_info(self):
                return {"service": "api_gateway", "version": "1.0.0", "status": "running"}
        
        minimal_service = MinimalService()
        add_standard_endpoints(router, minimal_service, "api_gateway")
        app.include_router(router)
    except Exception as e:
        logger.warning(f"⚠️  Failed to add standard endpoints: {e}")
    
    logger.info("✅ API Gateway Service started successfully!")

@app.on_event("shutdown")
async def shutdown():
    """Shutdown event - cleanup resources"""
    global service
    if service:
        try:
            await service.shutdown()
            logger.info("✅ API Gateway Service shut down successfully")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    logger.info(f"Starting API Gateway Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

