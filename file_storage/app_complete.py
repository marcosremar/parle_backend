"""
File Storage Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Request, APIRouter
from fastapi.responses import FileResponse, Response
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import json
import aiofiles
import hashlib
import mimetypes
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
        "name": "file_storage",
        "port": 8107,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "storage": {
        "base_path": "/tmp/file_storage",
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "allowed_extensions": [".wav", ".mp3", ".mp4", ".jpg", ".png", ".txt", ".json"],
        "cleanup_interval_hours": 24,
        "max_age_days": 30
    }
}

def get_config():
    """Get file storage service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class FileMetadata(BaseModel):
    """File metadata model"""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    upload_date: datetime
    last_accessed: Optional[datetime] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    file_size: int
    content_type: str
    upload_date: datetime
    url: str

class FileListResponse(BaseModel):
    """File list response"""
    files: List[FileMetadata]
    total_count: int
    total_size: int

class StorageStats(BaseModel):
    """Storage statistics"""
    total_files: int
    total_size: int
    storage_used: int
    storage_available: Optional[int] = None
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None

# ============================================================================
# File Storage Manager
# ============================================================================

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

        print(f"✅ File Storage Manager initialized at {self.base_path}")

    def _load_metadata(self):
        """Load file metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.file_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.file_metadata)} files")
            except Exception as e:
                print(f"⚠️  Failed to load metadata: {e}")
                self.file_metadata = {}
        else:
            print("ℹ️  No existing metadata file found")

    def _save_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.file_metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save metadata: {e}")

    def _generate_file_id(self, filename: str) -> str:
        """Generate unique file ID"""
        import uuid
        return f"{uuid.uuid4().hex}_{int(datetime.now().timestamp())}"

    def _get_file_path(self, file_id: str) -> Path:
        """Get file path for file ID"""
        return self.base_path / file_id

    def _validate_file_size(self, file_size: int) -> bool:
        """Validate file size"""
        return file_size <= self.max_file_size

    def _validate_file_extension(self, filename: str) -> bool:
        """Validate file extension"""
        allowed_extensions = DEFAULT_CONFIG["storage"]["allowed_extensions"]
        _, ext = os.path.splitext(filename.lower())
        return ext in allowed_extensions

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def upload_file(self, file: UploadFile, tags: List[str] = None, metadata: Dict[str, Any] = None) -> FileUploadResponse:
        """Upload a file"""
        # Validate file size
        if not self._validate_file_size(file.size or 0):
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {self.max_file_size} bytes")

        # Validate file extension
        if not self._validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail="File extension not allowed")

        # Generate file ID
        file_id = self._generate_file_id(file.filename)
        file_path = self._get_file_path(file_id)

        try:
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)

            # Calculate hash
            file_hash = self._calculate_file_hash(file_path)

            # Create metadata
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

            # Store metadata
            self.file_metadata[file_id] = file_metadata
            self._save_metadata()

            return FileUploadResponse(
                file_id=file_id,
                filename=file.filename,
                file_size=len(content),
                content_type=file_metadata["content_type"],
                upload_date=now,
                url=f"/files/{file_id}"
            )

        except Exception as e:
            # Clean up file if upload failed
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

        # Update last accessed
        self.file_metadata[file_id]["last_accessed"] = datetime.now().isoformat()
        self._save_metadata()

        return file_path

    def get_file_metadata(self, file_id: str) -> FileMetadata:
        """Get file metadata"""
        if file_id not in self.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")

        metadata = self.file_metadata[file_id]
        return FileMetadata(**metadata)

    def list_files(self, tag: Optional[str] = None, limit: int = 100, offset: int = 0) -> FileListResponse:
        """List files with optional filtering"""
        files = []
        total_size = 0

        # Filter files
        filtered_metadata = self.file_metadata.values()
        if tag:
            filtered_metadata = [f for f in filtered_metadata if tag in f.get("tags", [])]

        # Sort by upload date (newest first)
        sorted_files = sorted(
            filtered_metadata,
            key=lambda x: x.get("upload_date", ""),
            reverse=True
        )

        # Apply pagination
        paginated_files = sorted_files[offset:offset + limit]

        # Convert to FileMetadata objects
        for file_data in paginated_files:
            files.append(FileMetadata(**file_data))
            total_size += file_data.get("file_size", 0)

        return FileListResponse(
            files=files,
            total_count=len(sorted_files),
            total_size=total_size
        )

    def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        if file_id not in self.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._get_file_path(file_id)

        # Delete file from disk
        if file_path.exists():
            file_path.unlink()

        # Remove metadata
        del self.file_metadata[file_id]
        self._save_metadata()

        return True

    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics"""
        if not self.file_metadata:
            return StorageStats(
                total_files=0,
                total_size=0,
                storage_used=0
            )

        total_files = len(self.file_metadata)
        total_size = sum(f.get("file_size", 0) for f in self.file_metadata.values())

        # Calculate storage used (approximate)
        storage_used = total_size
        for file_data in self.file_metadata.values():
            file_path = file_data.get("file_path")
            if file_path and Path(file_path).exists():
                storage_used += Path(file_path).stat().st_size

        # Find oldest/newest files
        upload_dates = [f.get("upload_date") for f in self.file_metadata.values() if f.get("upload_date")]
        if upload_dates:
            oldest_file = min(upload_dates)
            newest_file = max(upload_dates)
        else:
            oldest_file = None
            newest_file = None

        return StorageStats(
            total_files=total_files,
            total_size=total_size,
            storage_used=storage_used,
            oldest_file=oldest_file,
            newest_file=newest_file
        )

    def cleanup_old_files(self, max_age_days: int = 30):
        """Clean up old files"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        files_to_delete = []

        for file_id, file_data in self.file_metadata.items():
            upload_date_str = file_data.get("upload_date")
            if upload_date_str:
                try:
                    upload_date = datetime.fromisoformat(upload_date_str)
                    if upload_date < cutoff_date:
                        files_to_delete.append(file_id)
                except:
                    pass

        for file_id in files_to_delete:
            try:
                self.delete_file(file_id)
                print(f"🗑️  Cleaned up old file: {file_id}")
            except:
                pass

        return len(files_to_delete)

# ============================================================================
# Global Storage Manager Instance
# ============================================================================

try:
    config = get_config()
    storage_manager = FileStorageManager(
        base_path=config["storage"]["base_path"],
        max_file_size=config["storage"]["max_file_size"]
    )
    print("✅ File Storage Manager initialized")
except Exception as e:
    print(f"⚠️  File Storage Manager failed: {e}")
    storage_manager = None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="File Storage Service", version="1.0.0")

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    if not storage_manager:
        return {
            "status": "unhealthy",
            "service": "file_storage",
            "timestamp": datetime.now().isoformat(),
            "error": "Storage manager not initialized"
        }

    try:
        stats = storage_manager.get_storage_stats()
        return {
            "status": "healthy",
            "service": "file_storage",
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "total_files": stats.total_files,
                "total_size": stats.total_size,
                "storage_used": stats.storage_used
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "file_storage",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: str = Form(""),
    metadata: str = Form("{}")
):
    """Upload a file"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    try:
        # Parse tags and metadata
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        metadata_dict = json.loads(metadata) if metadata else {}

        result = await storage_manager.upload_file(file, tags_list, metadata_dict)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/files/{file_id}")
async def download_file(file_id: str):
    """Download a file"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    try:
        file_path = storage_manager.download_file(file_id)
        metadata = storage_manager.get_file_metadata(file_id)

        return FileResponse(
            path=file_path,
            filename=metadata.filename,
            media_type=metadata.content_type
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/files/{file_id}/metadata")
async def get_file_metadata(file_id: str):
    """Get file metadata"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    return storage_manager.get_file_metadata(file_id)

@app.get("/files")
async def list_files(
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List files"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    return storage_manager.list_files(tag=tag, limit=limit, offset=offset)

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    storage_manager.delete_file(file_id)
    return {"message": "File deleted successfully"}

@app.get("/stats")
async def get_storage_stats():
    """Get storage statistics"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    return storage_manager.get_storage_stats()

@app.post("/cleanup")
async def cleanup_old_files(max_age_days: int = 30):
    """Clean up old files"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")

    deleted_count = storage_manager.cleanup_old_files(max_age_days)
    return {"message": f"Cleaned up {deleted_count} old files"}

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
    print("🚀 Initializing File Storage Service...")
    print(f"   Storage Path: {config['storage']['base_path']}")
    print(f"   Max File Size: {config['storage']['max_file_size']} bytes")

    if storage_manager:
        stats = storage_manager.get_storage_stats()
        print(f"   Total Files: {stats.total_files}")
        print(f"   Storage Used: {stats.storage_used} bytes")
    print("✅ File Storage Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8107"))
    print(f"Starting File Storage Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
