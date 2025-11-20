#!/usr/bin/env python3
"""
Script para criar app_complete.py para serviços migrados
"""
import os
from pathlib import Path

PARLE_DIR = Path("/Users/marcos/Documents/projects/backend/parle_backend")

# Portas padrão para cada serviço
SERVICE_PORTS = {
    "orchestrator": 8500,
    "session": 8600,
    "rest_polling": 8700,
    "conversation_store": 8800,
    "tts": 8900,
    "stt": 9000,
    "diarization": 9100,
    "vad_service": 9200,
    "sentiment_analysis": 9300,
    "broadcaster": 9400,
    "communication_strategy": 9500,
    "group_orchestrator": 9600,
    "group_session": 9700,
    "metrics_testing": 9800,
    "runpod_llm": 9900,
    "streaming_orchestrator": 10000,
    "webrtc": 10100,
    "webrtc_signaling": 10200,
    "discord_voice": 10300,
    "viber_gateway": 10400,
    "whatsapp_gateway": 10500,
}

def get_service_info(service_name: str):
    """Obtém informações sobre o serviço"""
    service_dir = PARLE_DIR / "src" / "services" / service_name
    
    has_service_py = (service_dir / "service.py").exists()
    has_routes_py = (service_dir / "routes.py").exists()
    has_config_py = (service_dir / "config.py").exists()
    
    return {
        "has_service": has_service_py,
        "has_routes": has_routes_py,
        "has_config": has_config_py,
        "port": SERVICE_PORTS.get(service_name, 8000)
    }

def create_app_complete(service_name: str, info: dict):
    """Cria app_complete.py para um serviço"""
    service_dir = PARLE_DIR / "src" / "services" / service_name
    app_file = service_dir / "app_complete.py"
    
    if app_file.exists():
        print(f"  ⚠️  app_complete.py já existe - pulando")
        return False
    
    port = info["port"]
    
    template = f'''"""
{service_name.replace("_", " ").title()} Service Standalone - Consolidated for Nomad deployment
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
    from src.core.route_helpers import add_standard_endpoints
    from src.core.metrics import increment_metric, set_gauge
    from src.core.telemetry_middleware import add_telemetry_middleware
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_telemetry_middleware(app, service_name):
        pass

    def add_standard_endpoints(router, service=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {{
    "service": {{
        "name": "{service_name}",
        "port": {port},
        "host": "0.0.0.0"
    }},
    "logging": {{
        "level": "INFO",
        "format": "json"
    }}
}}

def get_config():
    """Get {service_name} service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PORT", "{port}"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="{service_name.replace("_", " ").title()} Service",
    version="1.0.0",
    description="{service_name.replace("_", " ").title()} Service for Parle Backend"
)

# Add telemetry middleware
try:
    add_telemetry_middleware(app, "{service_name}")
except Exception as e:
    logger.warning(f"Failed to add telemetry middleware: {{e}}")

# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service = None

async def initialize_service():
    """Initialize the {service_name} service"""
    global service
    try:
        logger.info("🚀 Initializing {service_name.replace("_", " ").title()} Service...")
        logger.info(f"   Port: {{config['service']['port']}}")
        
        # Try to import and initialize service
        try:
            from src.services.{service_name}.service import {service_name.replace("_", "").title()}Service
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
                service_name="{service_name}",
                comm=comm,
                config={{"name": "{service_name}", "port": config["service"]["port"]}},
                profile="standalone",
                execution_mode="external"
            )
            
            service = {service_name.replace("_", "").title()}Service(config=config, context=context)
            success = await service.initialize()
            if success:
                logger.info("✅ {service_name.replace("_", " ").title()} Service initialized successfully")
                return True
        except ImportError as e:
            logger.warning(f"⚠️  Service class not found: {{e}}")
            logger.info("   Running in minimal mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {{e}}")
            logger.info("   Running in minimal mode")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize {service_name.replace("_", " ").title()} Service: {{e}}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {{
        "service": "{service_name}",
        "version": "1.0.0",
        "status": "running",
        "description": "{service_name.replace("_", " ").title()} Service"
    }}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service:
        try:
            return await service.health_check()
        except Exception:
            pass
    
    return {{
        "status": "healthy",
        "service": "{service_name}",
        "initialized": service is not None
    }}

# Try to mount service routes
try:
    from src.services.{service_name}.routes import router as service_router
    app.include_router(service_router)
    logger.info("✅ Service routes mounted")
except ImportError:
    logger.warning("⚠️  Service routes not found - using basic endpoints only")
except Exception as e:
    logger.warning(f"⚠️  Failed to mount service routes: {{e}}")

# Add standard endpoints
router = APIRouter()
try:
    if service:
        add_standard_endpoints(router, service, "{service_name}")
    else:
        add_standard_endpoints(router, None, "{service_name}")
except Exception as e:
    logger.warning(f"⚠️  Failed to add standard endpoints: {{e}}")

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
    port = int(os.getenv("PORT", "{port}"))
    logger.info(f"🚀 Starting {service_name.replace("_", " ").title()} Service on port {{port}}")
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
    
    app_file.write_text(template, encoding='utf-8')
    return True

def main():
    """Cria app_complete.py para todos os serviços migrados"""
    services = [
        "orchestrator", "session", "rest_polling", "conversation_store",
        "tts", "stt", "diarization", "vad_service", "sentiment_analysis",
        "broadcaster", "communication_strategy", "group_orchestrator",
        "group_session", "metrics_testing", "runpod_llm",
        "streaming_orchestrator", "webrtc", "webrtc_signaling",
        "discord_voice", "viber_gateway", "whatsapp_gateway"
    ]
    
    print(f"🚀 Criando app_complete.py para {len(services)} serviços")
    
    created = 0
    skipped = 0
    
    for service in services:
        print(f"\n📦 {service}")
        info = get_service_info(service)
        
        if create_app_complete(service, info):
            created += 1
            print(f"  ✅ app_complete.py criado (porta {info['port']})")
        else:
            skipped += 1
    
    print(f"\n📊 Resumo:")
    print(f"   ✅ Criados: {created}")
    print(f"   ⏭️  Pulados: {skipped}")

if __name__ == "__main__":
    main()

