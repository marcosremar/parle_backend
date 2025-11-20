"""
Models Router
Endpoints for model management and status - aggregates from microservices
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging
import os
import httpx
from typing import Dict, Any, Optional
import asyncio

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)

# Global Communication Manager (initialized from API Gateway service)
comm_manager: Optional['ServiceCommunicationManager'] = None

def set_comm_manager(cm):
    """Set Communication Manager instance from parent service"""
    global comm_manager
    comm_manager = cm

# Service endpoints to query for model status (environment variables with defaults)
SERVICE_ENDPOINTS = {
    "llm": os.getenv('LLM_SERVICE_URL', 'http://localhost:8100'),
    "tts": os.getenv('TTS_SERVICE_URL', 'http://localhost:8101'),
    "stt": os.getenv('STT_SERVICE_URL', 'http://localhost:8099')
}


async def get_service_model_info(service_name: str, service_url: str) -> Dict[str, Any]:
    """Get model information from a specific service"""
    try:
        if comm_manager:
            # Use Communication Manager for health check
            data = await comm_manager.call_text_service(
                service_name=service_name,
                text="",
                endpoint="/health"
            )
            return {
                "service": service_name,
                "status": "loaded",
                "healthy": True,
                "model_info": data.get("model", {}),
                "metrics": data.get("metrics", {})
            }
        else:
            # Fallback to direct HTTP
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "service": service_name,
                        "status": "loaded",
                        "healthy": True,
                        "model_info": data.get("model", {}),
                        "metrics": data.get("metrics", {})
                    }
                else:
                    return {
                        "service": service_name,
                        "status": "error",
                        "healthy": False,
                        "error": f"Service returned status {response.status_code}"
                    }
    except Exception as e:
        return {
            "service": service_name,
            "status": "unavailable",
            "healthy": False,
            "error": str(e)
        }


@router.get("/status")
async def get_models_status():
    """
    Get status of models across all services

    Returns:
        Model initialization status, memory usage, and metrics from all services
    """
    try:
        # Query all services in parallel
        tasks = [
            get_service_model_info(name, url)
            for name, url in SERVICE_ENDPOINTS.items()
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        all_healthy = all(r["healthy"] for r in results)
        models_info = {r["service"]: r for r in results}

        # Calculate total GPU memory if available
        total_gpu_memory = 0
        for result in results:
            if "metrics" in result and "gpu_memory_mb" in result["metrics"]:
                total_gpu_memory += result["metrics"]["gpu_memory_mb"]

        return JSONResponse(content={
            "success": True,
            "all_services_healthy": all_healthy,
            "models": models_info,
            "total_gpu_memory_mb": total_gpu_memory if total_gpu_memory > 0 else "unknown",
            "services_checked": len(SERVICE_ENDPOINTS)
        })

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.post("/warmup")
async def warmup_models():
    """
    Force model initialization (warmup) across all services
    Useful for ensuring models are loaded before first request
    """
    try:
        logger.info("🔥 Forcing model warmup across all services...")

        warmup_results = {}

        # Attempt to warmup each service via Communication Manager
        if comm_manager:
            # LLM Service warmup
            try:
                await comm_manager.call_text_service(
                    service_name="llm",
                    text="",
                    endpoint="/warmup"
                )
                warmup_results["llm"] = {"success": True, "status": "warmed"}
            except asyncio.TimeoutError:
                logger.warning("LLM warmup timeout via Communication Manager")
                warmup_results["llm"] = {"success": False, "status": "timeout"}
            except ConnectionError as e:
                logger.warning(f"LLM service connection failed: {e}")
                warmup_results["llm"] = {"success": False, "status": "unavailable"}
            except Exception as e:
                logger.error(f"Unexpected error during LLM warmup: {e}")
                warmup_results["llm"] = {"success": False, "status": "unavailable"}

            # TTS Service warmup
            try:
                await comm_manager.call_text_service(
                    service_name="tts",
                    text="Test",
                    endpoint="/synthesize",
                    extra_params={"text": "Test", "voice": "af_bella"}
                )
                warmup_results["tts"] = {"success": True, "status": "warmed"}
            except asyncio.TimeoutError:
                logger.warning("TTS warmup timeout via Communication Manager")
                warmup_results["tts"] = {"success": False, "status": "timeout"}
            except ConnectionError as e:
                logger.warning(f"TTS service connection failed: {e}")
                warmup_results["tts"] = {"success": False, "status": "unavailable"}
            except Exception as e:
                logger.error(f"Unexpected error during TTS warmup: {e}")
                warmup_results["tts"] = {"success": False, "status": "unavailable"}

            # STT Service warmup
            try:
                await comm_manager.call_text_service(
                    service_name="stt",
                    text="",
                    endpoint="/health"
                )
                warmup_results["stt"] = {"success": True, "status": "ready"}
            except asyncio.TimeoutError:
                logger.warning("STT warmup timeout via Communication Manager")
                warmup_results["stt"] = {"success": False, "status": "timeout"}
            except ConnectionError as e:
                logger.warning(f"STT service connection failed: {e}")
                warmup_results["stt"] = {"success": False, "status": "unavailable"}
            except Exception as e:
                logger.error(f"Unexpected error during STT warmup: {e}")
                warmup_results["stt"] = {"success": False, "status": "unavailable"}
        else:
            # Fallback to direct HTTP
            async with httpx.AsyncClient(timeout=30.0) as client:
                # LLM Service warmup
                try:
                    llm_response = await client.post(f"{SERVICE_ENDPOINTS['llm']}/warmup")
                    warmup_results["llm"] = {
                        "success": llm_response.status_code == 200,
                        "status": "warmed" if llm_response.status_code == 200 else "failed"
                    }
                except asyncio.TimeoutError:
                    logger.warning("LLM warmup timeout via HTTP")
                    warmup_results["llm"] = {"success": False, "status": "timeout"}
                except (ConnectionError, httpx.ConnectError) as e:
                    logger.warning(f"LLM service HTTP connection failed: {e}")
                    warmup_results["llm"] = {"success": False, "status": "unavailable"}
                except Exception as e:
                    logger.error(f"Unexpected error during LLM HTTP warmup: {e}")
                    warmup_results["llm"] = {"success": False, "status": "unavailable"}

                # TTS Service warmup
                try:
                    tts_response = await client.post(
                        f"{SERVICE_ENDPOINTS['tts']}/synthesize",
                        json={"text": "Test", "voice": "af_bella"}
                    )
                    warmup_results["tts"] = {
                        "success": tts_response.status_code == 200,
                        "status": "warmed" if tts_response.status_code == 200 else "failed"
                    }
                except asyncio.TimeoutError:
                    logger.warning("TTS warmup timeout via HTTP")
                    warmup_results["tts"] = {"success": False, "status": "timeout"}
                except (ConnectionError, httpx.ConnectError) as e:
                    logger.warning(f"TTS service HTTP connection failed: {e}")
                    warmup_results["tts"] = {"success": False, "status": "unavailable"}
                except Exception as e:
                    logger.error(f"Unexpected error during TTS HTTP warmup: {e}")
                    warmup_results["tts"] = {"success": False, "status": "unavailable"}

                # STT Service warmup
                try:
                    stt_response = await client.get(f"{SERVICE_ENDPOINTS['stt']}/health")
                    warmup_results["stt"] = {
                        "success": stt_response.status_code == 200,
                        "status": "ready" if stt_response.status_code == 200 else "failed"
                    }
                except asyncio.TimeoutError:
                    logger.warning("STT warmup timeout via HTTP")
                    warmup_results["stt"] = {"success": False, "status": "timeout"}
                except (ConnectionError, httpx.ConnectError) as e:
                    logger.warning(f"STT service HTTP connection failed: {e}")
                    warmup_results["stt"] = {"success": False, "status": "unavailable"}
                except Exception as e:
                    logger.error(f"Unexpected error during STT HTTP warmup: {e}")
                    warmup_results["stt"] = {"success": False, "status": "unavailable"}

        all_successful = all(r["success"] for r in warmup_results.values())

        return JSONResponse(content={
            "success": all_successful,
            "message": "Model warmup completed" if all_successful else "Some services failed to warm up",
            "results": warmup_results
        })

    except Exception as e:
        logger.error(f"Error during model warmup: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.delete("/clear")
async def clear_models():
    """
    Clear models from memory (for debugging/testing)
    WARNING: This will affect all active connections

    Note: Individual services manage their own models.
    This endpoint is for compatibility - actual clearing depends on service implementation.
    """
    try:
        logger.warning("🗑️ Requesting model clear - services manage their own lifecycle")

        return JSONResponse(content={
            "success": True,
            "message": "Model clear request sent",
            "warning": "Services manage their own model lifecycle. Manual restart may be required.",
            "note": "Use Service Manager to restart services if needed"
        })

    except Exception as e:
        logger.error(f"Error clearing models: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/info")
async def get_model_info():
    """
    Get detailed information about available models
    """
    try:
        model_info = {
            "llm": {
                "service": "LLM Service",
                "port": 8100,
                "model": "Ultravox v0.2",
                "description": "Multimodal LLM for audio and text processing",
                "capabilities": ["speech-to-text", "text-generation", "audio-understanding"]
            },
            "tts": {
                "service": "TTS Service",
                "port": 8101,
                "model": "Eleven Labs",
                "description": "High-quality text-to-speech synthesis",
                "capabilities": ["text-to-speech", "multi-voice", "multilingual"]
            },
            "stt": {
                "service": "STT Service",
                "port": 8099,
                "model": "Whisper",
                "description": "Speech recognition and transcription",
                "capabilities": ["speech-to-text", "language-detection", "timestamp-generation"]
            }
        }

        return JSONResponse(content={
            "success": True,
            "models": model_info,
            "total_services": len(model_info)
        })

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )