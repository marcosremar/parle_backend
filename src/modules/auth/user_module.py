"""
User Module - Direct Python calls for User management
"""

from typing import Dict, Optional, Any
from loguru import logger
import secrets
from datetime import datetime

from src.modules.base_module import BaseModule


class UserModule(BaseModule):
    """User Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("user")
        self.storage = None
    
    async def _initialize(self) -> bool:
        """Initialize user storage"""
        try:
            # Import user storage
            from src.services.user.storage import users_db
            
            self.storage = users_db
            
            # Create default admin if needed
            if not any(u.get("username") == "admin" for u in self.storage.values()):
                admin_id = "admin_" + secrets.token_hex(8)
                from src.services.user.core.auth import hash_password
                
                self.storage[admin_id] = {
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
            
            self.logger.info(f"✅ User Module initialized with {len(self.storage)} users")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize User Module: {e}")
            return False
    
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Dict with user_id, username, token, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            from src.services.user.core.auth import verify_password
            
            # Find user by email
            user = None
            for user_data in self.storage.values():
                if user_data.get("email") == email:
                    user = user_data
                    break
            
            if not user:
                raise ValueError("Invalid credentials")
            
            # Verify password
            if not verify_password(password, user.get("password_hash")):
                raise ValueError("Invalid credentials")
            
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            
            # Generate token (simplified - use JWT in production)
            import jwt
            import os
            secret = os.getenv("JWT_SECRET_KEY", "your-secret-key")
            token = jwt.encode(
                {"user_id": user["user_id"], "email": email},
                secret,
                algorithm="HS256"
            )
            
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "token": token,
                "is_admin": user.get("is_admin", False)
            }
        except Exception as e:
            self.logger.error(f"❌ Login failed: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        if not self.initialized:
            await self.initialize()
        
        user = self.storage.get(user_id)
        if user:
            # Remove sensitive data
            user_copy = user.copy()
            user_copy.pop("password_hash", None)
            return user_copy
        return None
    
    async def create_user(self, username: str, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """Create new user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            from src.services.user.core.auth import hash_password
            
            # Check if user exists
            for user_data in self.storage.values():
                if user_data.get("email") == email or user_data.get("username") == username:
                    raise ValueError("User already exists")
            
            # Create user
            user_id = "user_" + secrets.token_hex(8)
            self.storage[user_id] = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password_hash": hash_password(password),
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "is_active": True,
                "is_admin": False,
                **kwargs
            }
            
            user = self.storage[user_id].copy()
            user.pop("password_hash", None)
            return user
        except Exception as e:
            self.logger.error(f"❌ User creation failed: {e}")
            raise
