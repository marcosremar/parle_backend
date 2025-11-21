"""
API Gateway Service - JWT Authentication Gateway
Simplified version focused on JWT authentication and API proxying
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel
import requests
from loguru import logger

# Add project root to path for core imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import local utils (fallback to core if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback to core imports
    try:
        from src.core.route_helpers import add_standard_endpoints
        from src.core.metrics import increment_metric, set_gauge
    except ImportError:
        # Fallback implementations for standalone mode
        def increment_metric(name, value=1, labels=None):
            pass

        def set_gauge(name, value, labels=None):
            pass

        def add_standard_endpoints(router, service_instance=None, service_name=None):
            pass

# Import API Gateway service
try:
    from src.services.api_gateway.service import APIGatewayService
    from src.services.api_gateway.routes import create_router
    raise

except ImportError:
    APIGatewayService = None
    create_router = None

# ============================================================================
# JWT Configuration
# ============================================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# JWT Security
security = HTTPBearer(auto_error=False)

# ============================================================================
# Pydantic Models for Authentication
# ============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

# ============================================================================
# JWT Helper Functions
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """Create JWT refresh token (longer expiration)"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # 7 days
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.DecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error verifying token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Dependency to get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)

    # Check if it's an access token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "api_gateway",
        "port": 8000,  # Changed from 8010 to 8000 for API Gateway standard port
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get API Gateway service configuration"""
    config = DEFAULT_CONFIG.copy()
    # Override with environment variables
    port = int(os.getenv("PORT", "8000"))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="API Gateway Service",
    version="1.0.0",
    description="Unified API Gateway for speech-to-speech processing"
)


# ============================================================================
# Service Initialization
# ============================================================================

config = get_config()
service: APIGatewayService = None

async def initialize_service():
    """Initialize the API Gateway service"""
    global service
    try:
        logger.info("🚀 Initializing API Gateway Service...")
        logger.info(f"   Port: {config['service']['port']}")

        # Try to initialize full service
        try:
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager

            # Create minimal communication manager for standalone
            class MockComm:
                def get_service_url(self, service_name): return None
                def send_request(self, *args, **kwargs): return None
            comm = MockComm()

            # Create ServiceContext using factory method
            context = ServiceContext.create(
                service_name="api_gateway",
                comm=comm,
                config={"name": "api_gateway", "port": config["service"]["port"]},
                profile="standalone",
                execution_mode="external"  # Standalone service
            )

            if APIGatewayService:
                service = APIGatewayService(config=config, context=context)
                success = await service.initialize()
                if success:
                    logger.info("✅ API Gateway Service initialized successfully")
                    return True

        except ImportError:
            logger.warning("⚠️  Service class not found - using basic endpoints only")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize service: {e}")

        logger.info("✅ API Gateway initialized (basic mode)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize API Gateway Service: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Authentication Routes
# ============================================================================

@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """Login endpoint - validate credentials and return JWT tokens"""
    try:
        # TODO: Replace with actual user service validation
        # For now, accept any email/password combination
        if not credentials.email or not credentials.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password required"
            )

        # Call User Service for authentication (with service discovery)
        user_service_base = os.getenv("USER_SERVICE_URL", "http://localhost:8201")
        user_service_url = f"{user_service_base}/login"
        logger.info(f"Calling user service: {user_service_url}")
        user_response = requests.post(
            user_service_url,
            json={
                "username": credentials.email,  # User service uses username field
                "password": credentials.password
            },
            timeout=10
        )
        logger.info(f"User service response status: {user_response.status_code}")

        if user_response.status_code != 200:
            # Handle specific error messages from user service
            error_detail = "Invalid credentials"
            try:
                error_data = user_response.json()
                error_detail = error_data.get("detail", error_detail)
            except:
                pass

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=error_detail
            )

        # Parse user service response
        user_data = user_response.json()
        user_id = str(user_data.get("user", {}).get("id"))
        email = user_data.get("user", {}).get("email", credentials.email)

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user data from authentication service"
            )

        # Create JWT tokens
        access_token = create_access_token({"sub": user_id, "email": email})
        refresh_token = create_refresh_token({"sub": user_id, "email": email})

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token_endpoint(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = verify_token(request.refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Create new access token
        user_id = payload.get("sub")
        email = payload.get("email")

        access_token = create_access_token({"sub": user_id, "email": email})
        refresh_token = create_refresh_token({"sub": user_id, "email": email})

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user_id": current_user.get("sub"),
        "email": current_user.get("email"),
        "token_type": current_user.get("type"),
        "issued_at": current_user.get("iat"),
        "expires_at": current_user.get("exp")
    }

# ============================================================================
# Protected API Routes (with JWT validation)
# ============================================================================

@app.get("/api/v1/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    """Proxy to conversation history service"""
    try:
        # Proxy to conversation_history service
        conversation_history_url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8501")
        user_id = current_user.get("sub")

        # Forward request with user_id as query parameter
        params = {"user_id": user_id}
        response = requests.get(f"{conversation_history_url}/api/v1/conversations", params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Conversation history service error: {response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Conversation service unavailable"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying to conversation_history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unavailable"
        )

@app.post("/api/v1/conversations")
async def create_conversation(request: dict, current_user: dict = Depends(get_current_user)):
    """Create new conversation via conversation_history service"""
    try:
        # Proxy to conversation_history service
        conversation_history_url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8501")
        user_id = current_user.get("sub")

        # Add user_id to query parameters
        params = {"user_id": user_id}
        response = requests.post(f"{conversation_history_url}/api/v1/conversations",
                               params=params, json=request, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Conversation history service error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create conversation: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying to conversation_history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unavailable"
        )

@app.post("/api/v1/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, request: dict, current_user: dict = Depends(get_current_user)):
    """Add message to conversation via conversation_history service"""
    try:
        # Proxy to conversation_history service
        conversation_history_url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8501")
        user_id = current_user.get("sub")

        # Add user_id to query parameters
        params = {"user_id": user_id}
        response = requests.post(f"{conversation_history_url}/api/v1/conversations/{conversation_id}/messages",
                               params=params, json=request, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Conversation history service error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to add message: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying to conversation_history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unavailable"
        )

@app.get("/api/v1/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get conversation messages via conversation_history service"""
    try:
        # Proxy to conversation_history service
        conversation_history_url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8501")
        user_id = current_user.get("sub")

        # Forward request with user_id as query parameter
        params = {"user_id": user_id}
        response = requests.get(f"{conversation_history_url}/api/v1/conversations/{conversation_id}/messages",
                              params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Conversation history service error: {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get messages: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying to conversation_history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unavailable"
        )

@app.get("/api/v1/health")
async def protected_health_check(current_user: dict = Depends(get_current_user)):
    """Protected health check"""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "user_authenticated": True,
        "user_id": current_user.get("sub")
    }

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service:
        return await service.health_check()
    # In basic mode, service is still functional
    return {
        "status": "healthy",
        "service": "api-gateway",
        "mode": "basic",
        "jwt_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "api-gateway",
        "version": "1.0.0",
        "status": "running",
        "description": "API Gateway with JWT Authentication for speech-to-speech processing",
        "features": ["JWT Authentication", "API Proxy", "Rate Limiting", "Request Routing"],
        "endpoints": {
            "auth": {
                "login": "POST /auth/login",
                "refresh": "POST /auth/refresh",
                "me": "GET /auth/me"
            },
            "api": {
                "conversations": "GET /api/v1/conversations",
                "health": "GET /api/v1/health"
            },
            "public": {
                "health": "GET /health",
                "root": "GET /"
            }
        },
        "jwt_config": {
            "algorithm": ALGORITHM,
            "access_token_expiry": f"{ACCESS_TOKEN_EXPIRE_MINUTES} minutes",
            "refresh_token_expiry": "7 days"
        }
    }

# Mount service router after initialization
@app.on_event("startup")
async def startup():
    """Startup event - initialize service and mount routes"""
    global service
    
    logger.info("🚀 Starting API Gateway Service...")
    logger.info(f"   Port: {config['service']['port']}")
    
    # Initialize service
    success = await initialize_service()
    if not success:
        logger.error("❌ Service initialization failed")
        return
    
    # Mount service router
    if service:
        try:
            # Get router from service
            service_router = service.get_router()
            if service_router:
                app.include_router(service_router)
                logger.info("✅ Service router mounted")

            # Also include core routes
            try:
                core_router = create_router(service)
                if core_router is not None:
                    app.include_router(core_router)
                    logger.info("✅ Core router mounted")
                else:
                    logger.warning("⚠️  create_router returned None - skipping core router")
            except Exception as e:
                logger.warning(f"⚠️  Failed to mount core router: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to mount service router: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Shutdown event - cleanup resources"""
    global service
    if service:
        try:
            await service.shutdown()
            logger.info("✅ API Gateway Service shut down successfully")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting API Gateway Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

