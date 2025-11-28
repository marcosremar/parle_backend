"""
Database Module - Direct Python calls for Database operations
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class DatabaseModule(BaseModule):
    """Database Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("database")
        self.db = None
    
    async def _initialize(self) -> bool:
        """Initialize database"""
        try:
            # Try to import DatabaseStorage from services (with fallback)
            try:
                from src.services.database.app_complete import DatabaseStorage
                import os
                from pathlib import Path
                
                # Create storage path
                storage_path = os.getenv("STORAGE_DIR", str(Path.home() / ".cache" / "parle_backend" / "database"))
                os.makedirs(storage_path, exist_ok=True)
                db_path = os.path.join(storage_path, "database.db")
                
                self.db = DatabaseStorage(db_path)
                self.logger.info("✅ Database Module initialized")
                return True
            except ImportError:
                # Fallback: DatabaseStorage not available, use in-memory
                self.logger.warning("⚠️  DatabaseStorage not available, using in-memory storage")
                self.db = None
                self._data = {}
                return True
        except Exception as e:
            self.logger.warning(f"⚠️  Database not available: {e}")
            self.db = None
            self._data = {}
            return True
    
    async def set_data(
        self,
        user_id: str,
        key: str,
        value: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set data for user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db:
                # UserDatabase pode ter set_data síncrono ou assíncrono
                if hasattr(self.db, 'set_data'):
                    result = self.db.set_data(user_id=user_id, key=key, value=value)
                    if hasattr(result, '__await__'):
                        result = await result
                    return result if isinstance(result, dict) else {"success": True}
            # Fallback to in-memory
            if not hasattr(self, '_data'):
                self._data = {}
            if user_id not in self._data:
                self._data[user_id] = {}
            self._data[user_id][key] = value
            return {"success": True}
        except Exception as e:
            self.logger.error(f"❌ Failed to set data: {e}")
            raise
    
    async def get_data(
        self,
        user_id: str,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """Get data for user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.db and hasattr(self.db, 'get_data'):
                # DatabaseStorage.get_data is synchronous and returns dict or None
                result = self.db.get_data(user_id=user_id, key=key)
                # get_data returns {"value": ..., "metadata": ...} or None
                if result:
                    return result.get("value") if isinstance(result, dict) else result
                return None
            # Fallback to in-memory
            if not hasattr(self, '_data'):
                self._data = {}
            return self._data.get(user_id, {}).get(key)
        except Exception as e:
            self.logger.error(f"❌ Failed to get data: {e}")
            return None
