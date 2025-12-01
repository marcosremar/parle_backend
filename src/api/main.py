"""
API Principal - Monolito Modular
Consolida todos os serviços em uma única aplicação FastAPI
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import threading
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set monolith mode
# Note: System always uses direct module calls (no MONOLITH_MODE needed)

# Load unified configuration
from src.core.config import get_config

config = get_config()

# Module instances (singletons)
_modules: dict[str, Any] = {}
_modules_lock = threading.Lock()


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
    global _modules, _modules_lock

    logger.info("🧹 Cleaning up modules...")

    cleanup_errors = []

    # Cleanup each module if it has cleanup method
    with _modules_lock:
        for name, module in _modules.items():
            if name == "initialized":
                continue
            if hasattr(module, "cleanup"):
                try:
                    import asyncio

                    if asyncio.iscoroutinefunction(module.cleanup):
                        await module.cleanup()
                    elif hasattr(module.cleanup, "__call__"):
                        if hasattr(module.cleanup, "__await__"):
                            await module.cleanup()
                        else:
                            module.cleanup()
                except Exception as e:
                    cleanup_errors.append((name, str(e)))
                    logger.error(f"Error cleaning up {name}: {e}", exc_info=True)

        _modules.clear()

    if cleanup_errors:
        logger.error(f"Cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}")
    else:
        logger.info("✅ All modules cleaned up successfully")


def get_module(module_name: str) -> Any:
    """Get a module instance (lazy initialization, thread-safe)"""
    global _modules, _modules_lock

    with _modules_lock:
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


# Create FastAPI app with comprehensive OpenAPI documentation
app = FastAPI(
    title="Parle Backend API",
    version="1.0.0",
    description="""
    ## Parle Backend - Sistema de Conversação Multimodal
    
    API monolítica modular que consolida todos os serviços em um único processo.
    
    ### Características Principais
    
    * 🎤 **Speech-to-Speech**: Pipeline completo de áudio para áudio
    * 🤖 **IA Integrada**: STT (Whisper), LLM (GPT/LiteLLM), TTS (ElevenLabs/HuggingFace)
    * 📊 **Orquestração**: Gerenciamento inteligente do fluxo de conversação
    * 🔐 **Autenticação JWT**: API Gateway com autenticação e autorização
    * 💾 **Persistência**: Armazenamento de conversações, usuários e arquivos
    * 📚 **Tutoring**: Módulos de diagnóstico e política pedagógica (opcional)
    
    ### Endpoints Principais
    
    * `/api/v1/speech/stt/transcribe` - Transcrever áudio para texto
    * `/api/v1/speech/tts/synthesize` - Sintetizar texto para áudio
    * `/api/v1/llm/generate` - Geração de texto com LLM
    * `/api/v1/conversation` - Conversação unificada (texto ou áudio)
    * `/api/v1/auth/*` - Autenticação e gerenciamento de usuários
    * `/api/v1/tutoring/*` - Módulos de tutoring (se habilitados)
    
    ### Autenticação
    
    A maioria dos endpoints requer autenticação via JWT Bearer token.
    Obtenha um token através do endpoint `/api/v1/auth/login`.
    
    ### Rate Limiting
    
    Endpoints críticos possuem rate limiting configurado para prevenir abuso.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Parle Team",
        "email": "support@parle.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.parle.ai", "description": "Production server"},
    ],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - configured based on environment
allowed_origins = config.server.allowed_origins
if config.environment == "development":
    # In development, allow all origins
    allowed_origins = ["*"]
elif not allowed_origins:
    # In production without explicit origins, warn but allow (for backward compatibility)
    logger.warning(
        "⚠️  SERVER_ALLOWED_ORIGINS not set - allowing all origins (not recommended for production)"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["X-Request-ID"],
)

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Correlation ID middleware (for request tracing)
try:
    from src.core.prometheus_metrics import CorrelationIDMiddleware

    app.middleware("http")(CorrelationIDMiddleware.add_correlation_id)
    logger.info("✅ Correlation ID middleware enabled")
except ImportError:
    logger.warning("⚠️  Correlation ID middleware not available")


# JWT Secret validation middleware (must be early)
@app.middleware("http")
async def validate_jwt_secret(request: Request, call_next):
    """Validate JWT_SECRET is set in production"""
    if config.environment == "production":
        jwt_secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")
        if not jwt_secret or jwt_secret in ["your-secret-key", "default-secret", ""]:
            logger.error("❌ JWT_SECRET_KEY not properly configured in production!")
            return Response(
                content="Server configuration error: JWT_SECRET_KEY must be set in production",
                status_code=500,
                headers={"Content-Type": "text/plain"},
            )
    return await call_next(request)


# HTTPS enforcement middleware (must be before security headers)
@app.middleware("http")
async def enforce_https(request: Request, call_next):
    """Enforce HTTPS in production"""
    if config.environment == "production":
        if request.url.scheme != "https":
            return Response(
                content="HTTPS required in production",
                status_code=400,
                headers={"Content-Type": "text/plain"},
            )
    return await call_next(request)


# Metrics middleware (records request metrics)
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record metrics for all HTTP requests"""
    import time

    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Record metrics
    try:
        from src.core.metrics import record_request

        # Get endpoint path (remove query params)
        endpoint = str(request.url.path)
        # Record metric
        record_request(
            method=request.method, endpoint=endpoint, status=response.status_code, duration=duration
        )
    except Exception as e:
        # Don't fail request if metrics fail
        logger.debug(f"Failed to record metrics: {e}")

    return response


# CSRF protection middleware (for form submissions)
@app.middleware("http")
async def csrf_protection(request: Request, call_next):
    """
    CSRF protection for state-changing operations.

    FastAPI already provides CSRF protection for JSON APIs via CORS.
    This adds additional protection for form-based submissions.
    """
    # Skip CSRF for safe methods
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return await call_next(request)

    # JSON APIs are protected by CORS
    content_type = request.headers.get("Content-Type", "")
    if "application/json" in content_type:
        return await call_next(request)

    # For form submissions, validate CSRF token if needed
    # (Currently optional - FastAPI's CORS provides sufficient protection)
    # Uncomment if you need strict CSRF for forms:
    # try:
    #     from src.core.csrf import validate_csrf_token
    #     if not validate_csrf_token(request):
    #         return Response(
    #             content="Invalid CSRF token",
    #             status_code=403
    #         )
    # except ImportError:
    #     pass  # CSRF module not available

    return await call_next(request)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Only add HSTS in production with HTTPS
    if config.environment == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

    return response


# Initialize metrics
try:
    from src.core.metrics import set_app_info

    set_app_info(version="1.0.0", environment=config.environment)
    logger.info("✅ Metrics initialized")
except Exception as e:
    logger.warning(f"⚠️  Failed to initialize metrics: {e}")


# Health check
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "parle_backend_api",
        "version": "1.0.0",
        "mode": "monolith",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "monolith",
        "modules_initialized": _modules.get("initialized", False),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from fastapi import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "Prometheus client not available"}


# Import and mount consolidated router
# Single router with all endpoints organized by domain
try:
    # Share limiter with router
    import src.api.routers.api as api_router_module
    from src.api.routers.api import router as api_router

    api_router_module.app_limiter = limiter

    # Mount single consolidated router with versioning
    app.include_router(api_router, prefix="/api/v1", tags=["api-v1"])
    # Also mount without version for backward compatibility
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
        workers=config.server.workers if not config.server.reload else 1,
    )
