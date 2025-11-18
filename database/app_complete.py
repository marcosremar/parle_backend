"""
Database Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Header, APIRouter, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import logging
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
import hashlib
import pickle
import numpy as np
from loguru import logger

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available - vector search disabled")

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from src.core.route_helpers import add_standard_endpoints
    from src.core.metrics import increment_metric, set_gauge
    from src.core.config_models import DatabaseConfig
    from src.core.context import ServiceContext
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    class DatabaseConfig:
        def __init__(self):
            self.redis_url = "redis://localhost:6379"
            self.redis_db = 0

    class ServiceContext:
        def __init__(self):
            self.logger = logger

    def add_standard_endpoints(router):
        pass

# ============================================================================
# Database Configuration (Standalone)
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "database",
        "port": 8300,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "database": {
        "runpod_volume_path": os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"),
        "storage_path": os.getenv("DATABASE_STORAGE_PATH", None),
        "max_active_sessions": 100,
        "session_ttl_minutes": 15,
        "sqlite": {
            "wal_mode": True,
            "pool_size": 5,
            "timeout_seconds": 10
        },
        "faiss": {
            "dimension": 384,  # Default embedding dimension
            "index_type": "IndexFlatIP",  # Inner product for cosine similarity
            "normalize_vectors": True
        },
        "realtime": {
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "redis_db": int(os.getenv("REDIS_DB", "0"))
        }
    }
}

def get_config():
    """Get database service configuration"""
    config = DEFAULT_CONFIG.copy()

    # Override for local development
    if not os.path.exists("/runpod-volume"):
        config["database"]["runpod_volume_path"] = "/tmp"
        config["database"]["storage_path"] = os.path.join(os.getcwd(), "data")

    return config

# ============================================================================
# Database Models
# ============================================================================

class SetDataRequest(BaseModel):
    key: str
    value: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class GetDataRequest(BaseModel):
    key: str

class SearchDataRequest(BaseModel):
    query: str
    metadata_filter: Optional[Dict[str, Any]] = None
    top_k: int = 5

class DeleteDataRequest(BaseModel):
    key: str

class ListKeysRequest(BaseModel):
    prefix: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    database_stats: Dict[str, Any]

class InitializeUserRequest(BaseModel):
    user_id: str
    quota_mb: Optional[int] = 1000
    metadata: Optional[Dict[str, Any]] = None

class CreateDocumentRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class CreateConversationRequest(BaseModel):
    title: str
    metadata: Optional[Dict[str, Any]] = None

class AddMessageRequest(BaseModel):
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# Database Storage Implementation
# ============================================================================

class DatabaseStorage:
    """Simple SQLite-based storage for user data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_data (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')
            conn.commit()

    def set_data(self, user_id: str, key: str, value: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Store key-value data for user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO user_data (user_id, key, value, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, json.dumps(value), json.dumps(metadata) if metadata else None))
            conn.commit()

    def get_data(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT value, metadata FROM user_data
                WHERE user_id = ? AND key = ?
            ''', (user_id, key))
            row = cursor.fetchone()
            if row:
                return {
                    "value": json.loads(row[0]),
                    "metadata": json.loads(row[1]) if row[1] else None
                }
        return None

    def delete_data(self, user_id: str, key: str) -> bool:
        """Delete data by key"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM user_data WHERE user_id = ? AND key = ?
            ''', (user_id, key))
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self, user_id: str, prefix: Optional[str] = None) -> List[str]:
        """List all keys for user, optionally filtered by prefix"""
        with sqlite3.connect(self.db_path) as conn:
            if prefix:
                cursor = conn.execute('''
                    SELECT key FROM user_data
                    WHERE user_id = ? AND key LIKE ?
                    ORDER BY key
                ''', (user_id, f"{prefix}%"))
            else:
                cursor = conn.execute('''
                    SELECT key FROM user_data
                    WHERE user_id = ?
                    ORDER BY key
                ''', (user_id,))
            return [row[0] for row in cursor.fetchall()]

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get storage statistics for user"""
        with sqlite3.connect(self.db_path) as conn:
            # Count records
            cursor = conn.execute('SELECT COUNT(*) FROM user_data WHERE user_id = ?', (user_id,))
            record_count = cursor.fetchone()[0]

            # Count documents
            cursor = conn.execute('SELECT COUNT(*) FROM documents WHERE user_id = ?', (user_id,))
            doc_count = cursor.fetchone()[0]

            # Count conversations
            cursor = conn.execute('SELECT COUNT(*) FROM conversations WHERE user_id = ?', (user_id,))
            conv_count = cursor.fetchone()[0]

            return {
                "user_id": user_id,
                "total_records": record_count,
                "total_documents": doc_count,
                "total_conversations": conv_count,
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# Vector Store Implementation (FAISS)
# ============================================================================

class FAISSVectorStore:
    """Simple FAISS-based vector store"""

    def __init__(self, dimension: int = 384, index_type: str = "IndexFlatIP"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")

        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # For larger datasets, use IVF
            nlist = min(100, max(4, len(self.documents) // 39))  # Rule of thumb
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def add_document(self, doc_id: str, text: str, embedding: List[float], metadata: Optional[Dict] = None):
        """Add document to vector store"""
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if DEFAULT_CONFIG["database"]["faiss"]["normalize_vectors"]:
            vector = self._normalize_vector(vector)

        self.index.add(vector)
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "vector": vector.flatten().tolist()
        })

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if DEFAULT_CONFIG["database"]["faiss"]["normalize_vectors"]:
            query_vector = self._normalize_vector(query_vector)

        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(distances[0][i])
                })

        return results

# ============================================================================
# Global Storage Instances
# ============================================================================

config = get_config()
storage_path = config["database"]["storage_path"] or f"{config['database']['runpod_volume_path']}/database"
os.makedirs(storage_path, exist_ok=True)

# Global instances
db_storage = DatabaseStorage(f"{storage_path}/database.db")
vector_store = FAISSVectorStore(
    dimension=config["database"]["faiss"]["dimension"],
    index_type=config["database"]["faiss"]["index_type"]
) if FAISS_AVAILABLE else None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Database Service", version="1.0.0")

# ============================================================================
# Authentication Helper
# ============================================================================

def get_user_from_auth(authorization: Optional[str] = Header(None)) -> str:
    """Extract user_id from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Use: Bearer <user_id>"
        )

    user_id = authorization.replace("Bearer ", "")
    return user_id

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="database",
        timestamp=datetime.now().isoformat(),
        database_stats={
            "total_users": 0,  # Simplified for standalone
            "total_records": 0,
            "storage_path": storage_path
        }
    )

@app.post("/api/v1/initialize-user")
async def initialize_user(request: InitializeUserRequest):
    """Initialize user storage"""
    try:
        # For now, just acknowledge - storage is ready when accessed
        return {
            "status": "initialized",
            "user_id": request.user_id,
            "quota_mb": request.quota_mb,
            "message": f"User {request.user_id} storage initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data")
async def set_data(request: SetDataRequest, user_id: str = Depends(get_user_from_auth)):
    """Store key-value data"""
    try:
        db_storage.set_data(user_id, request.key, request.value, request.metadata)
        return {"status": "stored", "key": request.key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/{key}")
async def get_data(key: str, user_id: str = Depends(get_user_from_auth)):
    """Retrieve data by key"""
    try:
        result = db_storage.get_data(user_id, key)
        if result is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/data/{key}")
async def delete_data(key: str, user_id: str = Depends(get_user_from_auth)):
    """Delete data by key"""
    try:
        deleted = db_storage.delete_data(user_id, key)
        if not deleted:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"status": "deleted", "key": key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/keys")
async def list_keys(prefix: Optional[str] = None, user_id: str = Depends(get_user_from_auth)):
    """List user keys"""
    try:
        keys = db_storage.list_keys(user_id, prefix)
        return {"keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats(user_id: str = Depends(get_user_from_auth)):
    """Get user storage statistics"""
    try:
        stats = db_storage.get_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents")
async def create_document(request: CreateDocumentRequest, user_id: str = Depends(get_user_from_auth)):
    """Create a document"""
    try:
        doc_id = f"doc_{hashlib.md5(f'{user_id}_{request.title}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"

        # Store in SQLite
        with sqlite3.connect(f"{storage_path}/database.db") as conn:
            conn.execute('''
                INSERT INTO documents (id, user_id, title, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, user_id, request.title, request.content, json.dumps(request.metadata) if request.metadata else None))
            conn.commit()

        return {
            "id": doc_id,
            "title": request.title,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/conversations")
async def create_conversation(request: CreateConversationRequest, user_id: str = Depends(get_user_from_auth)):
    """Create a conversation"""
    try:
        conv_id = f"conv_{hashlib.md5(f'{user_id}_{request.title}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"

        with sqlite3.connect(f"{storage_path}/database.db") as conn:
            conn.execute('''
                INSERT INTO conversations (id, user_id, title, metadata)
                VALUES (?, ?, ?, ?)
            ''', (conv_id, user_id, request.title, json.dumps(request.metadata) if request.metadata else None))
            conn.commit()

        return {
            "id": conv_id,
            "title": request.title,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/messages")
async def add_message(request: AddMessageRequest, user_id: str = Depends(get_user_from_auth)):
    """Add message to conversation"""
    try:
        msg_id = f"msg_{hashlib.md5(f'{user_id}_{request.conversation_id}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"

        with sqlite3.connect(f"{storage_path}/database.db") as conn:
            conn.execute('''
                INSERT INTO messages (id, conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (msg_id, request.conversation_id, request.role, request.content, json.dumps(request.metadata) if request.metadata else None))
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": request.conversation_id,
            "status": "added"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/vector-search")
async def vector_search(request: SearchDataRequest, user_id: str = Depends(get_user_from_auth)):
    """Search in vector store (requires embeddings in request)"""
    try:
        if not FAISS_AVAILABLE or vector_store is None:
            raise HTTPException(status_code=503, detail="Vector search not available - FAISS not installed")

        # For now, expect embedding in metadata
        if not request.metadata or "embedding" not in request.metadata:
            raise HTTPException(status_code=400, detail="Embedding required in metadata.embedding")

        embedding = request.metadata["embedding"]
        results = vector_store.search(embedding, request.top_k)

        return {
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/vector-add")
async def vector_add(request: CreateDocumentRequest, user_id: str = Depends(get_user_from_auth)):
    """Add document to vector store (requires embedding in metadata)"""
    try:
        if not FAISS_AVAILABLE or vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store not available - FAISS not installed")

        if not request.metadata or "embedding" not in request.metadata:
            raise HTTPException(status_code=400, detail="Embedding required in metadata.embedding")

        doc_id = f"vec_{hashlib.md5(f'{user_id}_{request.title}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        embedding = request.metadata["embedding"]

        vector_store.add_document(doc_id, request.content, embedding, request.metadata)

        return {
            "id": doc_id,
            "status": "added_to_vector_store"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    print("🚀 Initializing Database Service...")
    print(f"   Storage path: {storage_path}")
    print(f"   Database: {storage_path}/database.db")
    print(f"   Vector dimension: {config['database']['faiss']['dimension']}")
    print("✅ Database Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8300"))
    print(f"Starting Database Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
