"""FastAPI app for User Service - Simplified version for Nomad"""
from fastapi import FastAPI
import sys
from pathlib import Path

# Ajustar paths
current_dir = Path(__file__).parent
v2_root = current_dir.parent
project_root = v2_root.parent

# Adicionar apenas os paths necessários (sem core para evitar conflitos)
sys.path.insert(0, str(project_root / "src"))

# Importar módulos do serviço user
from src.services.user.routes import create_router
from src.services.user.storage import users_db
from src.services.user.core.auth import hash_password

from datetime import datetime
import secrets

# Create FastAPI app
app = FastAPI(title="User Service", version="1.0.0")

# Tentar adicionar telemetry middleware (opcional)
try:
    from src.core.telemetry_middleware import add_telemetry_middleware
    add_telemetry_middleware(app, "user")
except ImportError:
    print("⚠️  Telemetry middleware não disponível, continuando sem ele")

# Include user router
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run(app, host="0.0.0.0", port=port)

