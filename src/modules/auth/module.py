"""
User/Auth Module - Direct Python calls for user management and authentication
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule
try:
    from .storage import users_db
    from .auth import hash_password, verify_password
except ImportError:
    # Fallback to services if local files not available
    from src.services.user.storage import users_db
    from src.services.user.core.auth import hash_password, verify_password


class UserModule(BaseModule):
    """User Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("user")
        self.users_db = users_db
    
    async def _initialize(self) -> bool:
        """Initialize user module"""
        try:
            self.logger.info("✅ User Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  User module initialization warning: {e}")
            return True
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Check if user already exists
            if email in self.users_db:
                raise ValueError("User with this email already exists")
            
            # Hash password
            hashed_password = hash_password(password)
            
            # Create user
            import secrets
            from datetime import datetime
            user_id = f"user_{secrets.token_hex(8)}"
            
            user = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password_hash": hashed_password,
                "full_name": full_name,
                "created_at": datetime.now().isoformat(),
                "active": True
            }
            
            self.users_db[email] = user
            return user
        except Exception as e:
            self.logger.error(f"❌ User creation failed: {e}")
            raise
    
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            user = self.users_db.get(email)
            if not user:
                raise ValueError("Invalid email or password")
            
            if not verify_password(password, user.get("password_hash")):
                raise ValueError("Invalid email or password")
            
            # Generate token (simplified)
            import secrets
            token = secrets.token_urlsafe(32)
            
            return {
                "token": token,
                "user_id": user.get("user_id"),
                "email": email,
                "username": user.get("username")
            }
        except Exception as e:
            self.logger.error(f"❌ Login failed: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        if not self.initialized:
            await self.initialize()
        
        for user in self.users_db.values():
            if user.get("user_id") == user_id:
                result = user.copy()
                result.pop("password_hash", None)  # Don't return password
                return result
        return None
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        if not self.initialized:
            await self.initialize()
        
        user = self.users_db.get(email)
        if user:
            result = user.copy()
            result.pop("password_hash", None)
            return result
        return None
