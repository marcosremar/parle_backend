"""
REST Polling Router - Proxy to REST Polling Service
Forwards requests from /api/rest_polling/* to rest_polling service
"""

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rest_polling", tags=["rest-polling"])

# REST Polling service URL (environment variable with default)
REST_POLLING_SERVICE_URL = os.getenv('REST_POLLING_SERVICE_URL', 'http://localhost:8106')


async def proxy_request(
    request: Request,
    path: str,
    method: str = "GET",
) -> Response:
    """
    Proxy request to REST Polling service

    Args:
        request: FastAPI request object
        path: Target path (without /api/rest_polling prefix)
        method: HTTP method

    Returns:
        Proxied response from rest_polling service
    """
    try:
        # Build target URL
        target_url = f"{REST_POLLING_SERVICE_URL}{path}"

        # Get request body if present
        body = None
        if method in ["POST", "PUT", "PATCH"]:
            body = await request.body()

        # Forward headers (excluding host)
        headers = dict(request.headers)
        headers.pop("host", None)

        # Make request to rest_polling service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
            )

            # Return proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )

    except httpx.TimeoutException:
        logger.error(f"Timeout proxying to rest_polling service: {target_url}")
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError:
        logger.error(f"Connection error to rest_polling service: {REST_POLLING_SERVICE_URL}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception as e:
        logger.error(f"Error proxying to rest_polling service: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@router.get("/health")
async def health(request: Request):
    """Health check - proxy to rest_polling service"""
    return await proxy_request(request, "/health", "GET")


@router.post("/api/session/create")
async def create_session(request: Request):
    """Create REST polling session"""
    return await proxy_request(request, "/api/session/create", "POST")


@router.get("/api/session/{session_id}/poll")
async def poll_session(session_id: str, request: Request):
    """Poll for messages"""
    return await proxy_request(request, f"/api/session/{session_id}/poll", "GET")


@router.post("/api/session/{session_id}/message")
async def send_message(session_id: str, request: Request):
    """Send message to session"""
    return await proxy_request(request, f"/api/session/{session_id}/message", "POST")


@router.post("/api/session/{session_id}/audio")
async def send_audio(session_id: str, request: Request):
    """Send audio to session"""
    return await proxy_request(request, f"/api/session/{session_id}/audio", "POST")


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete session"""
    return await proxy_request(request, f"/api/session/{session_id}", "DELETE")


@router.get("/api/session/{session_id}/status")
async def session_status(session_id: str, request: Request):
    """Get session status"""
    return await proxy_request(request, f"/api/session/{session_id}/status", "GET")


@router.post("/api/tts/synthesize")
async def synthesize_tts(request: Request):
    """Synthesize TTS"""
    return await proxy_request(request, "/api/tts/synthesize", "POST")


@router.get("/validate")
async def validate(request: Request):
    """Validate REST polling service"""
    return await proxy_request(request, "/validate", "GET")
