"""
User Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
import json
import uuid
import secrets
import sqlite3
from passlib.context import CryptContext
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from src.core.route_helpers import add_standard_endpoints
    from src.core.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "user",
        "port": 8200,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "database": {
        "path": "/tmp/user_service.db",
    },
    "security": {
        "session_expiry_hours": 24,
        "password_min_length": 12,
        "api_key_length": 32
    }
}

def get_config():
    """Get user service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class UserRole(str, Enum):
    """User roles"""
    USER = "user"
    ADMIN = "admin"

class UserRegister(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=12)
    full_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str

class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None

class UserProfile(BaseModel):
    """User profile response"""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime

class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=1, max_length=50)
    expires_days: Optional[int] = Field(None, ge=1, le=365)

class APIKeyResponse(BaseModel):
    """API key response"""
    id: str
    name: str
    key: str  # Only returned on creation
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime]

class LoginResponse(BaseModel):
    """Login response"""
    user: UserProfile
    session_token: str
    expires_at: datetime

class SessionResponse(BaseModel):
    """Session response"""
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database_connected: bool
    total_users: int
    total_sessions: int
    total_api_keys: int
    timestamp: datetime

# ============================================================================
# Password Hashing (Standalone)
# ============================================================================

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4,
    argon2__hash_len=32,
    argon2__salt_len=16,
)

def hash_password(password: str) -> str:
    """Hash password using Argon2"""
    if not password or len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against Argon2 hash"""
    if not password or not password_hash:
        return False
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False

# ============================================================================
# SQLite Database Manager (Standalone)
# ============================================================================

class UserDatabase:
    """
    SQLite database for managing users, sessions, and API keys
    Standalone version without external dependencies
    """

    def __init__(self, db_path: str = "/tmp/user_service.db"):
        """Initialize user database"""
        # Ensure data directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_file)
        self.conn: Optional[sqlite3.Connection] = None

        print(f"📁 User database path: {self.db_path}")
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Create users table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Create sessions table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Create api_keys table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Create indexes for performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)')

            self.conn.commit()
            print("✅ User database initialized")

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.conn = None

    def is_connected(self) -> bool:
        """Check database connection"""
        return self.conn is not None

    # ============================================================================
    # User Management
    # ============================================================================

    def create_user(self, user_data: UserRegister, role: UserRole = UserRole.USER) -> UserProfile:
        """Create a new user"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        # Check if username or email already exists
        if self.get_user_by_username(user_data.username):
            raise HTTPException(status_code=400, detail="Username already exists")
        if self.get_user_by_email(user_data.email):
            raise HTTPException(status_code=400, detail="Email already exists")

        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        try:
            password_hash = hash_password(user_data.password)

            self.conn.execute('''
                INSERT INTO users (
                    id, username, email, password_hash, full_name, role, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_data.username,
                user_data.email,
                password_hash,
                user_data.full_name,
                role.value,
                True,
                now,
                now
            ))

            self.conn.commit()
            return self.get_user_by_id(user_id)

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    def get_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID"""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_user_profile(row)

        except Exception:
            return None

    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user by username"""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_user_profile(row)

        except Exception:
            return None

    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email"""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute('SELECT * FROM users WHERE email = ?', (email,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_user_profile(row)

        except Exception:
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[UserProfile]:
        """Authenticate user with username and password"""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute(
                'SELECT * FROM users WHERE username = ? AND is_active = 1',
                (username,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Verify password
            stored_hash = row['password_hash']
            if not verify_password(password, stored_hash):
                return None

            return self._row_to_user_profile(row)

        except Exception:
            return None

    def update_user(self, user_id: str, update_data: UserUpdate) -> UserProfile:
        """Update user"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        user = self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Build update query
        update_fields = []
        params = []

        if update_data.email is not None:
            # Check if email is already taken (skip if it's the same user)
            if update_data.email != user.email:
                existing = self.get_user_by_email(update_data.email)
                if existing:
                    raise HTTPException(status_code=400, detail="Email already exists")
            update_fields.append('email = ?')
            params.append(update_data.email)

        if update_data.full_name is not None:
            update_fields.append('full_name = ?')
            params.append(update_data.full_name)

        if update_data.is_active is not None:
            update_fields.append('is_active = ?')
            params.append(update_data.is_active)

        if not update_fields:
            return user  # No changes

        # Add updated_at
        update_fields.append('updated_at = ?')
        params.append(datetime.now().isoformat())

        # Add user_id
        params.append(user_id)

        try:
            # Build query safely
            set_clause = ", ".join(update_fields)
            query = f'UPDATE users SET {set_clause} WHERE id = ?'

            # Debug
            print(f"DEBUG: Query: {query}")
            print(f"DEBUG: Params: {params}")

            self.conn.execute(query, params)
            self.conn.commit()

            return self.get_user_by_id(user_id)

        except Exception as e:
            print(f"DEBUG: Update error: {str(e)}")
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

    def list_users(self) -> List[UserProfile]:
        """List all users"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.execute('SELECT * FROM users ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [self._row_to_user_profile(row) for row in rows]

        except Exception:
            return []

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if not self.conn:
            return False

        try:
            # Delete associated sessions and API keys first
            self.conn.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
            self.conn.execute('DELETE FROM api_keys WHERE user_id = ?', (user_id,))

            # Delete user
            cursor = self.conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            deleted = cursor.rowcount > 0
            self.conn.commit()
            return deleted

        except Exception as e:
            self.conn.rollback()
            return False

    # ============================================================================
    # Session Management
    # ============================================================================

    def create_session(self, user_id: str, expiry_hours: int = 24) -> SessionResponse:
        """Create a new session"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(hours=expiry_hours)

        try:
            self.conn.execute('''
                INSERT INTO sessions (id, user_id, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                now.isoformat(),
                expires_at.isoformat(),
                True
            ))

            self.conn.commit()

            return SessionResponse(
                id=session_id,
                user_id=user_id,
                created_at=now,
                expires_at=expires_at,
                is_active=True
            )

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

    def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Get session by ID"""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return SessionResponse(
                id=row['id'],
                user_id=row['user_id'],
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']),
                is_active=row['is_active']
            )

        except Exception:
            return None

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        if not self.conn:
            return False

        try:
            cursor = self.conn.execute(
                'UPDATE sessions SET is_active = 0 WHERE id = ?',
                (session_id,)
            )
            invalidated = cursor.rowcount > 0
            self.conn.commit()
            return invalidated

        except Exception as e:
            self.conn.rollback()
            return False

    def list_user_sessions(self, user_id: str) -> List[SessionResponse]:
        """List sessions for a user"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.execute(
                'SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,)
            )
            rows = cursor.fetchall()

            return [
                SessionResponse(
                    id=row['id'],
                    user_id=row['user_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    is_active=row['is_active']
                ) for row in rows
            ]

        except Exception:
            return []

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        if not self.conn:
            return 0

        try:
            now = datetime.now().isoformat()
            cursor = self.conn.execute(
                'UPDATE sessions SET is_active = 0 WHERE expires_at < ? AND is_active = 1',
                (now,)
            )
            cleaned = cursor.rowcount
            self.conn.commit()
            return cleaned

        except Exception as e:
            self.conn.rollback()
            return 0

    # ============================================================================
    # API Key Management
    # ============================================================================

    def create_api_key(self, user_id: str, name: str, expires_days: Optional[int] = None) -> tuple[APIKeyResponse, str]:
        """Create a new API key"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        api_key_id = str(uuid.uuid4())
        api_key_value = secrets.token_urlsafe(32)  # Generate secure API key
        api_key_hash = hash_password(api_key_value)  # Hash for storage

        now = datetime.now()
        expires_at = None
        if expires_days:
            expires_at = now + timedelta(days=expires_days)

        try:
            self.conn.execute('''
                INSERT INTO api_keys (id, user_id, name, key_hash, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                api_key_id,
                user_id,
                name,
                api_key_hash,
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                True
            ))

            self.conn.commit()

            return APIKeyResponse(
                id=api_key_id,
                name=name,
                key=api_key_value,  # Return the actual key only on creation
                user_id=user_id,
                created_at=now,
                expires_at=expires_at
            ), api_key_value

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")

    def get_api_key_by_value(self, api_key_value: str) -> Optional[APIKeyResponse]:
        """Get API key by value (for authentication)"""
        if not self.conn:
            return None

        try:
            # Get all active API keys and check against the provided value
            cursor = self.conn.execute('SELECT * FROM api_keys WHERE is_active = 1')
            rows = cursor.fetchall()

            for row in rows:
                if verify_password(api_key_value, row['key_hash']):
                    expires_at = None
                    if row['expires_at']:
                        expires_at = datetime.fromisoformat(row['expires_at'])
                        if expires_at < datetime.now():
                            continue  # Expired

                    return APIKeyResponse(
                        id=row['id'],
                        name=row['name'],
                        key="",  # Don't return the actual key
                        user_id=row['user_id'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        expires_at=expires_at
                    )

            return None

        except Exception:
            return None

    def list_user_api_keys(self, user_id: str) -> List[APIKeyResponse]:
        """List API keys for a user"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.execute(
                'SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,)
            )
            rows = cursor.fetchall()

            api_keys = []
            for row in rows:
                expires_at = None
                if row['expires_at']:
                    expires_at = datetime.fromisoformat(row['expires_at'])

                api_keys.append(APIKeyResponse(
                    id=row['id'],
                    name=row['name'],
                    key="",  # Don't return the actual key
                    user_id=row['user_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    expires_at=expires_at
                ))

            return api_keys

        except Exception:
            return []

    def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Revoke an API key"""
        if not self.conn:
            return False

        try:
            cursor = self.conn.execute(
                'UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ?',
                (api_key_id, user_id)
            )
            revoked = cursor.rowcount > 0
            self.conn.commit()
            return revoked

        except Exception as e:
            self.conn.rollback()
            return False

    # ============================================================================
    # Statistics
    # ============================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.conn:
            return {
                "total_users": 0,
                "total_sessions": 0,
                "total_api_keys": 0,
                "active_sessions": 0,
                "active_api_keys": 0
            }

        try:
            # Get counts
            user_count = self.conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            session_count = self.conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
            api_key_count = self.conn.execute('SELECT COUNT(*) FROM api_keys').fetchone()[0]

            # Get active counts
            active_sessions = self.conn.execute(
                'SELECT COUNT(*) FROM sessions WHERE is_active = 1 AND expires_at > ?',
                (datetime.now().isoformat(),)
            ).fetchone()[0]

            active_api_keys = self.conn.execute(
                'SELECT COUNT(*) FROM api_keys WHERE is_active = 1 AND (expires_at IS NULL OR expires_at > ?)',
                (datetime.now().isoformat(),)
            ).fetchone()[0]

            return {
                "total_users": user_count,
                "total_sessions": session_count,
                "total_api_keys": api_key_count,
                "active_sessions": active_sessions,
                "active_api_keys": active_api_keys
            }

        except Exception:
            return {
                "total_users": 0,
                "total_sessions": 0,
                "total_api_keys": 0,
                "active_sessions": 0,
                "active_api_keys": 0
            }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _row_to_user_profile(self, row) -> UserProfile:
        """Convert database row to UserProfile"""
        return UserProfile(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            full_name=row['full_name'],
            role=UserRole(row['role']),
            is_active=row['is_active'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    def create_admin_user(self):
        """Create default admin user"""
        try:
            admin_user = self.get_user_by_username("admin")
            if not admin_user:
                admin_data = UserRegister(
                    username="admin",
                    email="admin@example.com",
                    password="admin12345678",
                    full_name="System Administrator"
                )
                self.create_user(admin_data, UserRole.ADMIN)
                print("✅ Admin user created (username: admin, password: admin12345678)")
        except Exception as e:
            print(f"⚠️  Failed to create admin user: {e}")

# ============================================================================
# Global Database Instance
# ============================================================================

try:
    config = get_config()
    user_db = UserDatabase(db_path=config["database"]["path"])
    print("✅ User Database initialized")
except Exception as e:
    print(f"⚠️  User Database failed: {e}")
    user_db = None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="User Service", version="1.0.0", description="Complete User Service with all endpoints")

security = HTTPBearer()

# ============================================================================
# Authentication Helpers
# ============================================================================

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current user from session token"""
    session_id = credentials.credentials
    session = user_db.get_session(session_id) if user_db else None

    if not session or not session.is_active:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Check expiration
    if session.expires_at < datetime.now():
        raise HTTPException(status_code=401, detail="Session expired")

    user = user_db.get_user_by_id(session.user_id) if user_db else None
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user

def get_current_user_or_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current user from session token or API key"""
    token = credentials.credentials

    # Try session first
    session = user_db.get_session(token) if user_db else None
    if session and session.is_active and session.expires_at > datetime.now():
        user = user_db.get_user_by_id(session.user_id) if user_db else None
        if user:
            return user

    # Try API key
    api_key = user_db.get_api_key_by_value(token) if user_db else None
    if api_key:
        user = user_db.get_user_by_id(api_key.user_id) if user_db else None
        if user:
            return user

    raise HTTPException(status_code=401, detail="Invalid authentication")

def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Require admin privileges"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ============================================================================
# Routes
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = user_db.get_stats() if user_db else {}
    return HealthResponse(
        status="healthy" if user_db and user_db.is_connected() else "unhealthy",
        database_connected=user_db.is_connected() if user_db else False,
        total_users=stats.get("total_users", 0),
        total_sessions=stats.get("total_sessions", 0),
        total_api_keys=stats.get("total_api_keys", 0),
        timestamp=datetime.now()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "user-service",
        "version": "1.0.0",
        "status": "running",
        "description": "Complete User Service with authentication, sessions, and API keys"
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.get_stats()

# ==================== User Management ====================

@app.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """Register a new user"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.create_user(user_data)

@app.post("/login", response_model=LoginResponse)
async def login(user_data: UserLogin):
    """Login user and create session"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    user = user_db.authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session = user_db.create_session(user.id)
    return LoginResponse(
        user=user,
        session_token=session.id,
        expires_at=session.expires_at
    )

@app.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    """Logout user (invalidate session)"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get session from header
    auth_header = current_user.get("_auth_header")
    if auth_header:
        session_id = auth_header.replace("Bearer ", "")
        user_db.invalidate_session(session_id)

    return {"message": "Logged out successfully"}

@app.get("/users/me", response_model=UserProfile)
async def get_current_user_profile(current_user: Dict = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.put("/users/me", response_model=UserProfile)
async def update_current_user(update_data: UserUpdate, current_user: Dict = Depends(get_current_user)):
    """Update current user profile"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.update_user(current_user["id"], update_data)

@app.get("/users", response_model=List[UserProfile])
async def list_users(_: Dict = Depends(require_admin)):
    """List all users (admin only)"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.list_users()

@app.delete("/users/{user_id}")
async def delete_user(user_id: str, _: Dict = Depends(require_admin)):
    """Delete user (admin only)"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    deleted = user_db.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User deleted successfully"}

# ==================== Session Management ====================

@app.get("/sessions", response_model=List[SessionResponse])
async def list_user_sessions(current_user: Dict = Depends(get_current_user)):
    """List current user's sessions"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.list_user_sessions(current_user["id"])

@app.delete("/sessions/{session_id}")
async def invalidate_session(session_id: str, current_user: Dict = Depends(get_current_user)):
    """Invalidate a specific session"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    # Check if session belongs to current user
    session = user_db.get_session(session_id)
    if not session or session.user_id != current_user["id"]:
        raise HTTPException(status_code=404, detail="Session not found")

    invalidated = user_db.invalidate_session(session_id)
    if not invalidated:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session invalidated successfully"}

# ==================== API Key Management ====================

@app.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(api_key_data: APIKeyCreate, current_user: Dict = Depends(get_current_user)):
    """Create a new API key"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    api_key_response, _ = user_db.create_api_key(
        current_user["id"],
        api_key_data.name,
        api_key_data.expires_days
    )
    return api_key_response

@app.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(current_user: Dict = Depends(get_current_user)):
    """List user's API keys"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return user_db.list_user_api_keys(current_user["id"])

@app.delete("/api-keys/{api_key_id}")
async def revoke_api_key(api_key_id: str, current_user: Dict = Depends(get_current_user)):
    """Revoke an API key"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    revoked = user_db.revoke_api_key(api_key_id, current_user["id"])
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"message": "API key revoked successfully"}

# ==================== Admin Endpoints ====================

@app.post("/admin/create-admin")
async def create_admin_user(_: Dict = Depends(require_admin)):
    """Create admin user (admin only)"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    user_db.create_admin_user()
    return {"message": "Admin user creation attempted"}

@app.post("/admin/cleanup-sessions")
async def cleanup_expired_sessions(_: Dict = Depends(require_admin)):
    """Clean up expired sessions (admin only)"""
    if not user_db:
        raise HTTPException(status_code=503, detail="Database not available")

    cleaned = user_db.cleanup_expired_sessions()
    return {"message": f"Cleaned up {cleaned} expired sessions"}

# Add standard endpoints
router = APIRouter()
add_standard_endpoints(router)
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing User Service...")
    print(f"   Database Path: {config['database']['path']}")

    if user_db and user_db.is_connected():
        stats = user_db.get_stats()
        print(f"   Total Users: {stats['total_users']}")
        print(f"   Total Sessions: {stats['total_sessions']}")
        print(f"   Total API Keys: {stats['total_api_keys']}")

        # Create admin user if it doesn't exist
        user_db.create_admin_user()
    else:
        print("   ❌ Database connection failed")

    print("✅ User Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8200"))
    print(f"Starting User Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    if not session or not session.get("is_active"):
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Check expiration
    expires_at = datetime.fromisoformat(session["expires_at"])
    if datetime.now() > expires_at:
        raise HTTPException(status_code=401, detail="Session expired")
    
    user = get_user_by_id(session["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Require admin privileges"""
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ==================== Health & Info ====================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "user"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "user-service", "version": "1.0.0", "status": "running"}

@app.get("/stats")
async def stats():
    """Get service statistics"""
    return get_stats()

# ==================== User Management ====================

@app.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """Register a new user"""
    try:
        # Verificar se username ou email já existe
        for user in users_db.values():
            if user["username"] == user_data.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
            if user["email"] == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Criar novo usuário usando auth_simple
        user_id = f"user_{secrets.token_hex(8)}"
        user = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),  # Usa auth_simple
            "full_name": user_data.full_name,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "preferences": {},
            "is_active": True,
            "is_admin": False
        }
        
        users_db[user_id] = user
        
        # Remove password hash from response
        user_response = user.copy()
        user_response.pop("password_hash", None)
        return UserProfile(**user_response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login", response_model=LoginResponse)
async def login(credentials: UserLogin):
    """Login user and create session"""
    user = get_user_by_username(credentials.username)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Verificar se password_hash existe (usar get para evitar KeyError)
    password_hash = user.get("password_hash")
    if not password_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    try:
        if not verify_password(credentials.password, password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="User account is inactive")
    
    # Create session
    session = create_session(user["user_id"])
    
    # Update last login
    user["last_login"] = datetime.now().isoformat()
    
    # Remove password hash
    user_response = user.copy()
    user_response.pop("password_hash", None)
    
    return LoginResponse(
        session_token=session["session_id"],
        user=UserProfile(**user_response),
        expires_at=session["expires_at"]
    )

@app.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user), authorization: Optional[str] = Header(None)):
    """Logout user and invalidate session"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        invalidate_session(token)
    return {"message": "Logged out successfully"}

@app.get("/users/me", response_model=UserProfile)
async def get_my_profile(current_user: Dict = Depends(get_current_user)):
    """Get current user profile"""
    user = current_user.copy()
    user.pop("password_hash", None)
    return UserProfile(**user)

@app.put("/users/me", response_model=UserProfile)
async def update_my_profile(update_data: UserUpdate, current_user: Dict = Depends(get_current_user)):
    """Update current user profile"""
    user_id = current_user["user_id"]
    updated_user = update_user(
        user_id=user_id,
        email=update_data.email,
        full_name=update_data.full_name,
        preferences=update_data.preferences
    )
    updated_user.pop("password_hash", None)
    return UserProfile(**updated_user)

@app.get("/users", response_model=List[UserProfile])
async def list_users(current_user: Dict = Depends(require_admin)):
    """List all users (admin only)"""
    users = list_all_users()
    return [UserProfile(**user) for user in users]

@app.delete("/users/{user_id}")
async def delete_user(user_id: str, current_user: Dict = Depends(require_admin)):
    """Delete a user (admin only)"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Don't allow self-deletion
    if user_id == current_user["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    del users_db[user_id]
    return {"message": "User deleted successfully"}

# ==================== Sessions ====================

@app.get("/sessions")
async def get_my_sessions(current_user: Dict = Depends(get_current_user)):
    """Get current user's sessions"""
    sessions = list_user_sessions(current_user["user_id"])
    return sessions

# ==================== API Keys ====================

@app.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(key_data: APIKeyCreate, current_user: Dict = Depends(get_current_user)):
    """Create a new API key for current user"""
    api_key_data = create_api_key(
        user_id=current_user["user_id"],
        name=key_data.name,
        expires_in_days=key_data.expires_in_days or 365
    )
    return APIKeyResponse(
        api_key=api_key_data["api_key"],
        key_id=api_key_data["key_id"],
        name=api_key_data["name"],
        expires_at=api_key_data["expires_at"]
    )

@app.get("/api-keys")
async def list_my_api_keys(current_user: Dict = Depends(get_current_user)):
    """List current user's API keys"""
    keys = list_user_api_keys(current_user["user_id"])
    # Mask API keys in response
    for key in keys:
        if "api_key" in key:
            full_key = key["api_key"]
            key["key_preview"] = f"{full_key[:8]}...{full_key[-4:]}" if len(full_key) > 12 else "***"
            del key["api_key"]
    return keys

@app.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: str, current_user: Dict = Depends(get_current_user)):
    """Delete an API key"""
    # Verificar se a key pertence ao usuário
    user_keys = list_user_api_keys(current_user["user_id"])
    key_exists = any(k.get("key_id") == key_id for k in user_keys)
    if not key_exists:
        raise HTTPException(status_code=404, detail="API key not found")
    
    revoke_api_key_by_id(key_id)
    return {"message": "API key revoked successfully"}

# Startup event
@app.on_event("startup")
async def startup():
    """Initialize default admin user"""
    import secrets
    admin_id = "admin_" + secrets.token_hex(8)
    # Usar auth_simple para criar admin também
    admin_password = "admin12345678"  # Mínimo 12 caracteres
    users_db[admin_id] = {
        "user_id": admin_id,
        "username": "admin",
        "email": "admin@ultravox.local",
        "password_hash": hash_password(admin_password),  # Usa auth_simple
        "full_name": "System Administrator",
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "preferences": {"theme": "dark", "language": "pt-BR"},
        "is_active": True,
        "is_admin": True
    }
    print(f"✅ User Service initialized - Admin user created (username: admin, password: {admin_password})")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run(app, host="0.0.0.0", port=port)

