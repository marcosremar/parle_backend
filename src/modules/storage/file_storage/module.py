"""
File Storage Module - Direct Python calls for File storage
"""

from typing import Dict, Optional, Any, List

from src.modules.base_module import BaseModule


class FileStorageModule(BaseModule):
    """File Storage Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("file_storage")
        self.manager = None
    
    async def _initialize(self) -> bool:
        """Initialize file storage manager"""
        try:
            from .manager import FileStorageManager
            self.manager = FileStorageManager(
                base_path="/tmp/file_storage",
                max_file_size=100 * 1024 * 1024
            )
            self.logger.info("✅ File Storage Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  File storage not available: {e}")
            self.manager = None
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
            if self.manager:
                from fastapi import UploadFile
                from io import BytesIO
                
                file_obj = UploadFile(
                    filename=filename,
                    file=BytesIO(file_content)
                )
                
                result = await self.manager.upload_file(
                    file=file_obj,
                    tags=tags,
                    metadata=metadata
                )
                return result
            else:
                # Fallback to in-memory
                import secrets
                file_id = f"file_{secrets.token_hex(8)}"
                file_data = {
                    "file_id": file_id,
                    "filename": filename,
                    "file_size": len(file_content),
                    "upload_date": None,
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
            if self.manager:
                file_path = self.manager.download_file(file_id)
                if file_path and file_path.exists():
                    import asyncio
                    return await asyncio.to_thread(lambda: open(file_path, 'rb').read())
                return None
            else:
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
            if self.manager:
                return self.manager.get_file_metadata(file_id)
            else:
                file_data = self._files.get(file_id)
                if file_data:
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
        """List files"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.manager:
                return self.manager.list_files(tag=tags[0] if tags else None, limit=limit)
            else:
                files = list(self._files.values())
                if tags:
                    files = [
                        f for f in files
                        if any(tag in f.get("tags", []) for tag in tags)
                    ]
                for f in files:
                    f.pop("content", None)
                return files[:limit]
        except Exception as e:
            self.logger.error(f"❌ Failed to list files: {e}")
            return []
