#!/usr/bin/env python3
"""
JWT Authentication Middleware for FastAPI
Secure API endpoints with JWT tokens

Features:
- JWT token generation and validation
- Role-based access control (RBAC)
- Token refresh
- Blacklist support
- API key authentication (alternative)

⚠️  SECURITY FIX: JWT secret now loaded from environment variables (not hardcoded)
"""

import logging
import jwt
import time
import os
import hmac
from typing import Optional, Dict, List, Callable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"  # Service-to-service communication
    GUEST = "guest"


@dataclass
class AuthConfig:
    """
    Authentication configuration

    Args:
        secret_key: JWT secret key (loaded from JWT_SECRET_KEY env var)
        algorithm: JWT algorithm (HS256, RS256, etc.)
        access_token_expire_minutes: Access token expiration time
        refresh_token_expire_days: Refresh token expiration time
        require_auth: Require authentication for all endpoints
        public_paths: List of paths that don't require authentication
        api_key_header: Header name for API key authentication
        api_keys: Valid API keys (alternative to JWT)
        enabled: Enable/disable authentication

    ⚠️  SECURITY: JWT_SECRET_KEY MUST be set in environment!
    """
    secret_key: str = None  # Will be loaded from environment
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    require_auth: bool = True
    public_paths: List[str] = None
    api_key_header: str = "X-API-Key"
    api_keys: Set[str] = None
    enabled: bool = True

    def __post_init__(self):
        # ⚠️  SECURITY FIX: Load JWT secret from environment
        if self.secret_key is None:
            self.secret_key = os.getenv("JWT_SECRET_KEY")

            # Validate secret is provided
            if not self.secret_key:
                error_msg = (
                    "❌ CRITICAL SECURITY ERROR: JWT_SECRET_KEY environment variable not set!\n"
                    "   This is required for JWT authentication.\n"
                    "   Generate a secure secret: python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
                    "   Then set: export JWT_SECRET_KEY='<generated-secret>'\n"
                    "   Or add to .env file: JWT_SECRET_KEY=<generated-secret>"
                )
                logger.critical(error_msg)
                raise ValueError(error_msg)

            # Validate secret length (minimum 32 characters)
            if len(self.secret_key) < 32:
                raise ValueError(
                    f"❌ JWT_SECRET_KEY must be at least 32 characters long (current: {len(self.secret_key)})"
                )

            logger.info("✅ JWT secret loaded from JWT_SECRET_KEY environment variable")

        if self.public_paths is None:
            self.public_paths = [
                "/",
                "/health",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/auth/login",
                "/auth/register"
            ]
        if self.api_keys is None:
            self.api_keys = set()


class JWTAuth:
    """
    JWT authentication handler

    Usage:
        auth = JWTAuth(config=AuthConfig(secret_key="your-secret-key"))

        # Create access token
        token = auth.create_access_token(user_id="user123", role=UserRole.USER)

        # Verify token
        payload = auth.verify_token(token)
        print(f"User: {payload['sub']}, Role: {payload['role']}")
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        """
        Initialize JWT auth

        Args:
            config: Authentication configuration
        """
        self.config = config or AuthConfig()
        self.blacklist: Set[str] = set()  # Revoked tokens

        if self.config.enabled:
            logger.info(
                f"🔐 JWT Authentication initialized\n"
                f"   Algorithm: {self.config.algorithm}\n"
                f"   Access token expiration: {self.config.access_token_expire_minutes}min\n"
                f"   Refresh token expiration: {self.config.refresh_token_expire_days}days\n"
                f"   API keys enabled: {len(self.config.api_keys) > 0}"
            )

    def create_access_token(
        self,
        user_id: str,
        role: UserRole = UserRole.USER,
        extra_claims: Optional[Dict] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            user_id: User identifier
            role: User role
            extra_claims: Additional claims to include

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.config.access_token_expire_minutes)

        claims = {
            "sub": user_id,  # Subject (user ID)
            "role": role.value,
            "iat": int(now.timestamp()),  # Issued at
            "exp": int(expire.timestamp()),  # Expiration
            "type": "access"
        }

        if extra_claims:
            claims.update(extra_claims)

        token = jwt.encode(claims, self.config.secret_key, algorithm=self.config.algorithm)
        return token

    def create_refresh_token(self, user_id: str) -> str:
        """
        Create JWT refresh token

        Args:
            user_id: User identifier

        Returns:
            JWT refresh token string
        """
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.refresh_token_expire_days)

        claims = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "type": "refresh"
        }

        token = jwt.encode(claims, self.config.secret_key, algorithm=self.config.algorithm)
        return token

    def verify_token(self, token: str, token_type: str = "access") -> Dict:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string
            token_type: Expected token type (access or refresh)

        Returns:
            Token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            # Decode token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )

            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )

    def revoke_token(self, token: str):
        """
        Revoke (blacklist) a token

        Args:
            token: JWT token to revoke
        """
        self.blacklist.add(token)
        logger.info(f"🚫 Token revoked (blacklist size: {len(self.blacklist)})")

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify API key

        Args:
            api_key: API key to verify

        Returns:
            True if valid
        """
        return api_key in self.config.api_keys

    def check_permission(self, required_role: UserRole, user_role: str) -> bool:
        """
        Check if user has required role

        Args:
            required_role: Minimum required role
            user_role: User's role

        Returns:
            True if user has permission
        """
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.SERVICE: 2,
            UserRole.ADMIN: 3
        }

        user_level = role_hierarchy.get(UserRole(user_role), 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for JWT authentication

    Usage:
        app = FastAPI()

        # Add authentication middleware
        app.add_middleware(
            AuthenticationMiddleware,
            config=AuthConfig(
                secret_key="your-secret-key",
                require_auth=True
            )
        )

        # Use in routes
        @app.get("/protected")
        async def protected_route(request: Request):
            user_id = request.state.user_id
            role = request.state.user_role
            return {"user": user_id, "role": role}
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[AuthConfig] = None
    ):
        """
        Initialize middleware

        Args:
            app: FastAPI application
            config: Authentication configuration
        """
        super().__init__(app)
        self.config = config or AuthConfig()
        self.auth = JWTAuth(config=self.config)

    async def dispatch(self, request: Request, call_next):
        """Process request through authentication"""
        if not self.config.enabled:
            return await call_next(request)

        # Check if path is public
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Try API key authentication first
        api_key = request.headers.get(self.config.api_key_header)
        if api_key and self.auth.verify_api_key(api_key):
            request.state.user_id = "api_key_user"
            request.state.user_role = UserRole.SERVICE.value
            request.state.auth_method = "api_key"
            return await call_next(request)

        # Try JWT authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Missing or invalid Authorization header",
                    "message": "Please provide 'Authorization: Bearer <token>' header"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = auth_header.split(" ")[1]

        try:
            # Verify token
            payload = self.auth.verify_token(token)

            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.user_role = payload.get("role")
            request.state.auth_method = "jwt"
            request.state.token_payload = payload

            # Process request
            response = await call_next(request)
            return response

        except HTTPException as e:
            logger.warning(f"🚫 Authentication failed: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "Authentication failed",
                    "message": e.detail
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require authentication)"""
        return any(path.startswith(public_path) for public_path in self.config.public_paths)


# Global auth instance
_jwt_auth: Optional[JWTAuth] = None


def get_jwt_auth(config: Optional[AuthConfig] = None) -> JWTAuth:
    """Get global JWT auth instance (singleton)"""
    global _jwt_auth

    if _jwt_auth is None:
        _jwt_auth = JWTAuth(config=config)

    return _jwt_auth
