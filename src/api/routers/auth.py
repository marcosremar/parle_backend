"""
Auth Router - User
Usa módulo interno para chamadas diretas Python
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import jwt
import os

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Module instance (lazy initialization)
_user_module = None


def get_user_module():
    """Get User module instance"""
    global _user_module
    if _user_module is None:
        from src.modules import create
        _user_module = create("user")
    return _user_module


# Pydantic models
class LoginRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class CreateUserRequest(BaseModel):
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")
    full_name: Optional[str] = Field(None, description="Full name")


def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        secret = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    payload = verify_token(credentials.credentials)
    user_module = get_user_module()
    user = await user_module.get_user(payload.get("user_id"))
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


@router.get("/health")
async def health():
    """Health check for auth services"""
    return {"status": "ok", "services": ["user"]}


@router.post("/user/login")
async def login(request: LoginRequest):
    """User login endpoint using direct module"""
    try:
        user_module = get_user_module()
        result = await user_module.login(
            email=request.email,
            password=request.password
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/register")
async def register(request: CreateUserRequest):
    """User registration endpoint using direct module"""
    try:
        user_module = get_user_module()
        result = await user_module.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return current_user


@router.get("/user/{user_id}")
async def get_user(user_id: str):
    """Get user by ID"""
    try:
        user_module = get_user_module()
        user = await user_module.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise HTTPException(status_code=500, detail=str(e))
