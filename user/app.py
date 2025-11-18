"""FastAPI app for User Service - Standalone"""
from fastapi import FastAPI
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.user.routes import create_router
from src.services.user.storage import users_db
from src.services.user.core.auth import hash_password
from src.core.telemetry_middleware import add_telemetry_middleware
from datetime import datetime
import secrets

# Create FastAPI app
app = FastAPI(title="User Service", version="1.0.0")
add_telemetry_middleware(app, "user")

# Include user router (already includes /validate endpoint)
app.include_router(create_router(service=None))

# Startup event
@app.on_event("startup")
async def startup():
    """Initialize default admin user"""
    admin_id = "admin_" + secrets.token_hex(8)
    users_db[admin_id] = {
        "user_id": admin_id,
        "username": "admin",
        "email": "admin@ultravox.local",
        "password_hash": hash_password("admin123"),
        "full_name": "System Administrator",
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "preferences": {"theme": "dark", "language": "pt-BR"},
        "is_active": True,
        "is_admin": True
    }
    print(f"✅ User Service initialized - Admin user created")

# Health endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "user"}
