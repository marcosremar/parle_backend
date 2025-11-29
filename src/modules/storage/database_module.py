"""
Database Module - Direct Python calls for Database operations
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class DatabaseModule(BaseModule):
    """Database Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("database")
        self.db_client = None
    
    async def _initialize(self) -> bool:
        """Initialize database client"""
        try:
            # Database client - using in-memory for now
            # TODO: Migrate UserDatabase to modules if needed
            
            # Using in-memory storage for now
            self.logger.info("✅ Database Module initialized (in-memory storage)")
            self.db_client = None
            self._data = {}
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Database Module: {e}")
            # Fallback to in-memory storage
            self.db_client = None
            self._data = {}
            return True
    
    async def set_data(
        self,
        user_id: str,
        key: str,
        value: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> bool:
        """Set data for a user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db_client:
                self.db_client.set_data(
                    user_id=user_id,
                    key=key,
                    value=value,
                    metadata=metadata
                )
                return True
            else:
                # Fallback to in-memory
                if user_id not in self._data:
                    self._data[user_id] = {}
                self._data[user_id][key] = {
                    "value": value,
                    "metadata": metadata or {}
                }
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to set data: {e}")
            return False
    
    async def get_data(
        self,
        user_id: str,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """Get data for a user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db_client:
                return self.db_client.get_data(user_id=user_id, key=key)
            else:
                # Fallback to in-memory
                return self._data.get(user_id, {}).get(key, {}).get("value")
        except Exception as e:
            self.logger.error(f"❌ Failed to get data: {e}")
            return None
    
    async def delete_data(
        self,
        user_id: str,
        key: str
    ) -> bool:
        """Delete data for a user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db_client:
                return self.db_client.delete_data(user_id=user_id, key=key)
            else:
                # Fallback to in-memory
                if user_id in self._data and key in self._data[user_id]:
                    del self._data[user_id][key]
                    return True
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to delete data: {e}")
            return False
    
    async def list_keys(
        self,
        user_id: str,
        prefix: Optional[str] = None
    ) -> List[str]:
        """List keys for a user, optionally filtered by prefix"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db_client:
                return self.db_client.list_keys(user_id=user_id, prefix=prefix)
            else:
                # Fallback to in-memory
                keys = list(self._data.get(user_id, {}).keys())
                if prefix:
                    keys = [k for k in keys if k.startswith(prefix)]
                return keys
        except Exception as e:
            self.logger.error(f"❌ Failed to list keys: {e}")
            return []
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get database statistics for a user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db_client:
                return self.db_client.get_stats(user_id=user_id)
            else:
                # Fallback to in-memory
                user_data = self._data.get(user_id, {})
                return {
                    "total_keys": len(user_data),
                    "user_id": user_id
                }
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {e}")
            return {}
