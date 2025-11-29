"""
File Storage Manager
"""

import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import HTTPException, UploadFile
import aiofiles
import mimetypes


class FileStorageManager:
    """Manages file storage operations"""

    def __init__(self, base_path: str = "/tmp/file_storage", max_file_size: int = 100*1024*1024):
        self.base_path = Path(base_path)
        self.max_file_size = max_file_size
        self.metadata_file = self.base_path / "metadata.json"

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.file_metadata: Dict[str, Dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load file metadata from disk"""
        if self.metadata_file.exists():
            try:
                import json
                with open(self.metadata_file, 'r') as f:
                    self.file_metadata = json.load(f)
            except Exception as e:
                self.file_metadata = {}

    def _save_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.file_metadata, f, indent=2, default=str)
        except Exception as e:
            pass

    def _generate_file_id(self, filename: str) -> str:
        """Generate unique file ID"""
        return f"{uuid.uuid4().hex}_{int(datetime.now().timestamp())}"

    def _get_file_path(self, file_id: str) -> Path:
        """Get file path for file ID"""
        return self.base_path / file_id

    def _validate_file_size(self, file_size: int) -> bool:
        """Validate file size"""
        return file_size <= self.max_file_size

    def _validate_file_extension(self, filename: str) -> bool:
        """Validate file extension"""
        allowed_extensions = [".wav", ".mp3", ".mp4", ".jpg", ".png", ".txt", ".json"]
        _, ext = os.path.splitext(filename.lower())
        return ext in allowed_extensions

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def upload_file(self, file: UploadFile, tags: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a file"""
        if not self._validate_file_size(file.size or 0):
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {self.max_file_size} bytes")

        if not self._validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail="File extension not allowed")

        file_id = self._generate_file_id(file.filename)
        file_path = self._get_file_path(file_id)

        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)

            file_hash = self._calculate_file_hash(file_path)

            now = datetime.now()
            file_metadata = {
                "file_id": file_id,
                "filename": file.filename,
                "original_filename": file.filename,
                "file_size": len(content),
                "content_type": file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream",
                "upload_date": now.isoformat(),
                "last_accessed": now.isoformat(),
                "tags": tags or [],
                "metadata": metadata or {},
                "file_hash": file_hash,
                "file_path": str(file_path)
            }

            self.file_metadata[file_id] = file_metadata
            self._save_metadata()

            return file_metadata

        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    def download_file(self, file_id: str) -> Path:
        """Get file path for download"""
        if file_id not in self.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._get_file_path(file_id)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        self.file_metadata[file_id]["last_accessed"] = datetime.now().isoformat()
        self._save_metadata()

        return file_path

    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata"""
        if file_id not in self.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")
        return self.file_metadata[file_id]

    def list_files(self, tag: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List files with optional filtering"""
        filtered_metadata = list(self.file_metadata.values())
        if tag:
            filtered_metadata = [f for f in filtered_metadata if tag in f.get("tags", [])]

        sorted_files = sorted(
            filtered_metadata,
            key=lambda x: x.get("upload_date", ""),
            reverse=True
        )

        return sorted_files[offset:offset + limit]

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        if file_id not in self.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._get_file_path(file_id)
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            pass

        del self.file_metadata[file_id]
        self._save_metadata()
        return True
