"""
Pipeline Management Endpoints

Pipeline validation, testing, and monitoring.
"""

from fastapi import APIRouter
from datetime import datetime
import asyncio
import aiohttp
from src.core.logging import setup_logging
from src.core.service_manager.core import HealthStatus, HEALTH_CHECK_TIMEOUT
from src.core.service_manager.models import ProcessStatus

# Setup logging
logger = setup_logging("endpoints-pipeline", level="INFO")

# Create router
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Manager instance
_manager = None

def set_manager(manager):
    """Set the manager instance"""
    global _manager
    _manager = manager

def get_manager():
    """Get the manager instance"""
    return _manager


@router.get("s/validate")
async def validate_pipelines():
    """Valida ambas as pipelines (WebRTC interna e API externa)"""
    manager = get_manager()

    results = {
        "timestamp": datetime.now().isoformat(),
        "pipelines": {},
        "services_status": {}
    }

    # Verificar serviços necessários
    required_services = ["llm", "tts", "websocket", "stt"]
    for service_id in required_services:
        service = manager.services.get(service_id)
        if service:
            is_running = manager.check_port(service.port)
            is_healthy = is_running and (service.health_status == HealthStatus.HEALTHY if hasattr(service, 'health_status') else False)
            results["services_status"][service_id] = {
                "name": service.name,
                "port": service.port,
                "healthy": is_healthy,
                "status": service.process_status.value if isinstance(service.process_status, ProcessStatus) else str(service.process_status)
            }

    # Helper function to check health endpoint with aiohttp
    async def check_health_endpoint(url: str) -> bool:
        """Check health endpoint using aiohttp (non-blocking)"""
        try:
            timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    # Validar Pipeline WebRTC Interna
    try:
        # WebRTC -> WebSocket -> Ultravox -> Kokoro TTS
        # Run health checks in parallel using asyncio.gather for better performance
        webrtc_ok, websocket_ok = await asyncio.gather(
            check_health_endpoint("http://localhost:8010/health"),
            check_health_endpoint("http://localhost:8020/health")
        )

        results["pipelines"]["webrtc_internal"] = {
            "status": "healthy" if webrtc_ok and websocket_ok else "unhealthy",
            "flow": "WebRTC(8010) → WebSocket(8020) → Ultravox(8100) → Kokoro(8101)",
            "endpoints": {
                "webrtc": webrtc_ok,
                "websocket": websocket_ok
            }
        }
    except Exception as e:
        results["pipelines"]["webrtc_internal"] = {
            "status": "error",
            "error": str(e)
        }

    # Validar Pipeline API Externa
    try:
        # API Gateway -> Ultravox/STT -> Kokoro TTS
        # Run health checks in parallel using asyncio.gather for better performance
        api_ok, llm_ok, tts_ok = await asyncio.gather(
            check_health_endpoint("http://localhost:8099/health"),
            check_health_endpoint("http://localhost:8100/health"),
            check_health_endpoint("http://localhost:8101/health")
        )

        results["pipelines"]["api_external"] = {
            "status": "healthy" if api_ok and llm_ok and tts_ok else "unhealthy",
            "flow": "API(8099) → Ultravox(8100)/STT(8200) → Kokoro(8101)",
            "endpoints": {
                "api_gateway": api_ok,
                "llm": llm_ok,
                "tts": tts_ok
            }
        }
    except Exception as e:
        results["pipelines"]["api_external"] = {
            "status": "error",
            "error": str(e)
        }

    # Log validation results
    if all(p.get("status") == "healthy" for p in results["pipelines"].values()):
        logger.info(f"✅ All pipelines validated successfully", extra={"results": results})
    else:
        logger.warning(f"⚠️ Some pipelines failed validation", extra={"results": results})

    return results


@router.get("/webrtc")
async def get_webrtc_pipeline_info():
    """Redirect to Orchestrator for WebRTC pipeline information"""
    return {
        "message": "Pipeline information is available through the Orchestrator service",
        "orchestrator_url": "http://localhost:8900/pipelines/webrtc",
        "status": "redirect"
    }


@router.get("/api")
async def get_api_pipeline_info():
    """Redirect to Orchestrator for API pipeline information"""
    return {
        "message": "Pipeline information is available through the Orchestrator service",
        "orchestrator_url": "http://localhost:8900/pipelines/api",
        "status": "redirect"
    }


@router.post("/webrtc/test")
async def test_webrtc_pipeline():
    """Test WebRTC pipeline"""
    return {"message": "WebRTC pipeline test endpoint", "status": "not_implemented"}


@router.post("/api/test")
async def test_api_pipeline():
    """Test API pipeline"""
    return {"message": "API pipeline test endpoint", "status": "not_implemented"}


@router.get("/metrics")
async def get_pipeline_metrics():
    """Get pipeline metrics"""
    return {"message": "Pipeline metrics endpoint", "status": "not_implemented"}


@router.get("/compare")
async def compare_pipelines():
    """Compare pipeline performance"""
    return {"message": "Pipeline comparison endpoint", "status": "not_implemented"}
