"""
Helper functions for API testing
"""

import httpx
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path


async def wait_for_api_ready(
    url: str,
    endpoint: str = "/health",
    max_retries: int = 30,
    delay: float = 1.0
) -> bool:
    """
    Wait for API to be ready
    
    Args:
        url: API base URL
        endpoint: Health check endpoint
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        
    Returns:
        True if API is ready
    """
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}{endpoint}")
                if response.status_code == 200:
                    return True
        except Exception:
            pass
        
        await asyncio.sleep(delay)
    
    return False


async def create_test_session(
    api_url: str,
    user_id: str = "test_user"
) -> Optional[str]:
    """
    Create a test session via API
    
    Args:
        api_url: API base URL
        user_id: User ID for session
        
    Returns:
        Session ID or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{api_url}/api/conversation/session/create",
                data={"user_id": user_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("id") or data.get("session_id")
    except Exception:
        pass
    
    return None


async def send_conversation_request(
    api_url: str,
    session_id: str,
    message: Optional[str] = None,
    audio_base64: Optional[str] = None,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Send conversation request to API
    
    Args:
        api_url: API base URL
        session_id: Session ID
        message: Text message (optional)
        audio_base64: Audio in base64 (optional)
        **kwargs: Additional parameters
        
    Returns:
        Response data or None if failed
    """
    try:
        data = {"session_id": session_id, **kwargs}
        
        if message:
            data["message"] = message
        elif audio_base64:
            data["audio_base64"] = audio_base64
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/api/conversation",
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    
    return None


def validate_api_response(response: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate API response structure
    
    Args:
        response: Response dictionary
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present
    """
    return all(field in response for field in required_fields)


def extract_audio_from_response(response: Dict[str, Any]) -> Optional[bytes]:
    """
    Extract audio bytes from API response
    
    Args:
        response: API response dictionary
        
    Returns:
        Audio bytes or None
    """
    import base64
    
    # Try different possible field names
    audio_data = response.get("audio") or response.get("audio_base64") or response.get("data")
    
    if audio_data:
        if isinstance(audio_data, bytes):
            return audio_data
        elif isinstance(audio_data, str):
            try:
                return base64.b64decode(audio_data)
            except Exception:
                pass
    
    return None
