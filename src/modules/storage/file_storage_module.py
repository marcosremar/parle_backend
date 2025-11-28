"""
File Storage Module - Direct Python calls for File storage
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
from loguru import logger

from src.modules.base_module import BaseModule


class FileStorageModule(BaseModule):
    """File Storage Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("file_storage")
        self.storage_manager = None
    
    async def _initialize(self) -> bool:
        """Initialize file storage manager"""
        try:
            # Import file storage manager
            from src.services.file_storage.app_complete import FileStorageManager
            from src.services.file_storage.app_complete import get_config
            
            config = get_config()
            storage_config = config.get("storage", {})
            
            self.storage_manager = FileStorageManager(
                base_path=storage_config.get("base_path", "/tmp/file_storage"),
                max_file_size=storage_config.get("max_file_size", 100 * 1024 * 1024)
            )
            
            self.logger.info("✅ File Storage Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize File Storage Module: {e}")
            # Fallback to basic storage
            self.storage_manager = None
            self._files = {}
            return True
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Upload a file"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.storage_manager:
                # Create a mock UploadFile-like object
                from fastapi import UploadFile
                from io import BytesIO
                
                file_obj = UploadFile(
                    filename=filename,
                    file=BytesIO(file_content)
                )
                
                result = await self.storage_manager.upload_file(
                    file=file_obj,
                    tags=tags,
                    metadata=metadata
                )
                
                # Convert to dict if needed
                if hasattr(result, 'dict'):
                    return result.dict()
                return result
            else:
                # Fallback to in-memory
                import secrets
                from datetime import datetime
                
                file_id = f"file_{secrets.token_hex(8)}"
                file_data = {
                    "file_id": file_id,
                    "filename": filename,
                    "file_size": len(file_content),
                    "upload_date": datetime.now().isoformat(),
                    "tags": tags or [],
                    "metadata": metadata or {},
                    "content": file_content
                }
                self._files[file_id] = file_data
                return file_data
        except Exception as e:
            self.logger.error(f"❌ File upload failed: {e}")
            raise
    
    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download a file by ID"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.storage_manager:
                file_path = self.storage_manager.download_file(file_id)
                if file_path and file_path.exists():
                    with open(file_path, 'rb') as f:
                        return f.read()
                return None
            else:
                # Fallback to in-memory
                file_data = self._files.get(file_id)
                if file_data:
                    return file_data.get("content")
                return None
        except Exception as e:
            self.logger.error(f"❌ File download failed: {e}")
            return None
    
    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.storage_manager:
                metadata = self.storage_manager.get_file_metadata(file_id)
                if metadata:
                    if hasattr(metadata, 'dict'):
                        return metadata.dict()
                    return metadata
                return None
            else:
                # Fallback to in-memory
                file_data = self._files.get(file_id)
                if file_data:
                    # Return metadata without content
                    result = file_data.copy()
                    result.pop("content", None)
                    return result
                return None
        except Exception as e:
            self.logger.error(f"❌ Failed to get file metadata: {e}")
            return None
    
    async def list_files(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List files, optionally filtered by tags"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.storage_manager:
                files = self.storage_manager.list_files(tags=tags, limit=limit)
                # Convert to list of dicts
                result = []
                for f in files:
                    if hasattr(f, 'dict'):
                        result.append(f.dict())
                    else:
                        result.append(f)
                return result
            else:
                # Fallback to in-memory
                files = list(self._files.values())
                if tags:
                    files = [
                        f for f in files
                        if any(tag in f.get("tags", []) for tag in tags)
                    ]
                # Remove content from results
                for f in files:
                    f.pop("content", None)
                return files[:limit]
        except Exception as e:
            self.logger.error(f"❌ Failed to list files: {e}")
            return []
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.storage_manager:
                return self.storage_manager.delete_file(file_id)
            else:
                # Fallback to in-memory
                if file_id in self._files:
                    del self._files[file_id]
                    return True
                return False
        except Exception as e:
            self.logger.error(f"❌ File deletion failed: {e}")
            return False
