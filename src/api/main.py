"""
API Principal - Monolito Modular
Consolida todos os serviços em uma única aplicação FastAPI
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set monolith mode
os.environ["MONOLITH_MODE"] = "true"

# Load unified configuration
from src.core.config import get_config
config = get_config()

# Module instances (singletons)
_modules: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("🚀 Starting Monolith API...")
    logger.info("   Mode: MONOLITH (direct Python calls)")
    
    try:
        # Initialize all modules
        await initialize_modules()
        logger.info("✅ All modules initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize modules: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Monolith API...")
    await cleanup_modules()
    logger.info("✅ Cleanup complete")


async def initialize_modules():
    """Initialize all service modules as singletons"""
    global _modules
    
    logger.info("📦 Initializing modules...")
    
    # Import modules lazily to avoid circular dependencies
    # Modules will be initialized on first access
    
    _modules["initialized"] = True
    logger.info("✅ Module system ready")


async def cleanup_modules():
    """Cleanup all module resources"""
    global _modules
    
    logger.info("🧹 Cleaning up modules...")
    
    # Cleanup each module if it has cleanup method
    for name, module in _modules.items():
        if name == "initialized":
            continue
        if hasattr(module, "cleanup"):
            try:
                if hasattr(module.cleanup, "__call__"):
                    if hasattr(module.cleanup, "__await__"):
                        await module.cleanup()
                    else:
                        module.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {name}: {e}")
    
    _modules.clear()


def get_module(module_name: str) -> Any:
    """Get a module instance (lazy initialization)"""
    global _modules
    
    if module_name not in _modules:
        # Lazy import and initialization
        try:
            from src.modules import module_factory
            _modules[module_name] = module_factory.create(module_name)
            logger.debug(f"✅ Module '{module_name}' initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize module '{module_name}': {e}")
            raise
    
    return _modules[module_name]


# Create FastAPI app
app = FastAPI(
    title="Parle Backend API",
    version="1.0.0",
    description="Monolith Modular API - All services in one process",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "parle_backend_api",
        "version": "1.0.0",
        "mode": "monolith",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "monolith",
        "modules_initialized": _modules.get("initialized", False)
    }

# Import and mount consolidated router
# Single router with all endpoints organized by domain
try:
    from src.api.routers.api import router as api_router
    
    # Mount single consolidated router
    app.include_router(api_router, prefix="/api", tags=["api"])
    
    logger.info("✅ Consolidated API router mounted")
except ImportError as e:
    logger.warning(f"⚠️  API router not available: {e}")
    logger.info("   Router will be created incrementally")
except Exception as e:
    logger.warning(f"⚠️  Error mounting router: {e}")
    import traceback
    logger.debug(traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    port = config.server.port
    host = config.server.host
    logger.info(f"🚀 Starting Monolith API on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=config.server.reload,
        workers=config.server.workers if not config.server.reload else 1
    )
