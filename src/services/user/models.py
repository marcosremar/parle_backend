"""
Pydantic Models for User Service
Data validation and serialization
"""
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict


# Request Models
class UserRegister(BaseModel):
    """User registration request"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class UserUpdate(BaseModel):
    """User profile update request"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    preferences: Optional[Dict] = None


class PasswordChange(BaseModel):
    """Password change request"""
    old_password: str
    new_password: str


class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str
    expires_in_days: Optional[int] = 365


# Response Models
class UserProfile(BaseModel):
    """User profile response"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    created_at: str
    last_login: Optional[str]
    preferences: Dict
    is_active: bool
    is_admin: bool = False


class SessionInfo(BaseModel):
    """Session information response"""
    session_id: str
    user_id: str
    created_at: str
    expires_at: str
    is_active: bool


class APIKeyInfo(BaseModel):
    """API key information response"""
    key_id: str
    name: str
    key_preview: str  # First/last chars only
    created_at: str
    expires_at: str
    is_active: bool
    last_used: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response with token"""
    session_token: str
    user: UserProfile
    expires_at: str


class APIKeyResponse(BaseModel):
    """API key creation response"""
    api_key: str  # Full key shown only on creation
    key_id: str
    name: str
    expires_at: str
