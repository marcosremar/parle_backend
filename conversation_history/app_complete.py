"""
Conversation History Service Standalone - Consolidated for Nomad deployment
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
import asyncio
import hashlib
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
        "name": "conversation_history",
        "port": 8010,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "conversation_history": {
        "storage_path": os.getenv("CONVERSATION_HISTORY_STORAGE_PATH", None),
        "max_active_sessions": 1000,
        "session_ttl_minutes": 60,
        "sqlite": {
            "wal_mode": True,
            "pool_size": 10,
            "timeout_seconds": 30
        }
    }
}

def get_config():
    """Get conversation history service configuration"""
    config = DEFAULT_CONFIG.copy()

    # Override for local development
    if not os.path.exists("/runpod-volume"):
        config["conversation_history"]["storage_path"] = os.path.join(os.getcwd(), "data")

    return config

# ============================================================================
# Database Models (Pydantic)
# ============================================================================

class Message(BaseModel):
    message_id: str = None
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    created_at: datetime = None
    speaker_id: Optional[str] = None  # Multi-speaker support

    def __init__(self, **data):
        super().__init__(**data)
        if self.message_id is None:
            self.message_id = f"msg_{hashlib.md5(f'{self.conversation_id}_{self.content[:50]}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"
        if self.created_at is None:
            self.created_at = datetime.now()

class Conversation(BaseModel):
    conversation_id: str = None
    user_id: str
    title: str = "New Chat"
    metadata: Optional[Dict[str, Any]] = {}
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.conversation_id is None:
            self.conversation_id = f"conv_{hashlib.md5(f'{self.user_id}_{self.title}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

# ============================================================================
# Storage Implementation
# ============================================================================

class ConversationStorage:
    """SQLite-based conversation and message storage"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    speaker_id TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')
            conn.commit()

    async def create_conversation(self, user_id: str, title: str = "New Chat", metadata: Optional[Dict] = None) -> Conversation:
        """Create new conversation"""
        conv = Conversation(user_id=user_id, title=title, metadata=metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversations (id, user_id, title, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conv.conversation_id, conv.user_id, conv.title, json.dumps(conv.metadata), conv.created_at.isoformat(), conv.updated_at.isoformat()))
            conn.commit()

        return conv

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT user_id, title, metadata, created_at, updated_at FROM conversations
                WHERE id = ?
            ''', (conversation_id,))

            row = cursor.fetchone()
            if row:
                return Conversation(
                    conversation_id=conversation_id,
                    user_id=row[0],
                    title=row[1],
                    metadata=json.loads(row[2]) if row[2] else {},
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4])
                )
        return None

    async def save_message(self, conversation_id: str, message: Message):
        """Save message to conversation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO messages (id, conversation_id, role, content, metadata, speaker_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.message_id,
                conversation_id,
                message.role,
                message.content,
                json.dumps(message.metadata),
                message.speaker_id,
                message.created_at.isoformat()
            ))
            conn.commit()

    async def get_messages(self, conversation_id: str, limit: int = 50, offset: int = 0) -> List[Message]:
        """Get messages for conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, role, content, metadata, speaker_id, created_at FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (conversation_id, limit, offset))

            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    message_id=row[0],
                    conversation_id=conversation_id,
                    role=row[1],
                    content=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    speaker_id=row[4],
                    created_at=datetime.fromisoformat(row[5])
                ))
            return messages

    async def search_conversations(self, user_id: str, query: str, limit: int = 10) -> List[Conversation]:
        """Search conversations by title or content"""
        with sqlite3.connect(self.db_path) as conn:
            # Search in conversation titles
            cursor = conn.execute('''
                SELECT id, user_id, title, metadata, created_at, updated_at FROM conversations
                WHERE user_id = ? AND title LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, f'%{query}%', limit))

            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    conversation_id=row[0],
                    user_id=row[1],
                    title=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                ))

            # If not enough results, search in message content
            if len(conversations) < limit:
                cursor = conn.execute('''
                    SELECT DISTINCT c.id, c.user_id, c.title, c.metadata, c.created_at, c.updated_at
                    FROM conversations c
                    JOIN messages m ON c.id = m.conversation_id
                    WHERE c.user_id = ? AND m.content LIKE ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                ''', (user_id, f'%{query}%', limit - len(conversations)))

                for row in cursor.fetchall():
                    conversations.append(Conversation(
                        conversation_id=row[0],
                        user_id=row[1],
                        title=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5])
                    ))

            return conversations

    async def get_user_conversations(self, user_id: str, limit: int = 20, offset: int = 0) -> List[Conversation]:
        """Get user's conversations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, user_id, title, metadata, created_at, updated_at FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))

            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    conversation_id=row[0],
                    user_id=row[1],
                    title=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                ))
            return conversations

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Count conversations
            cursor = conn.execute('SELECT COUNT(*) FROM conversations WHERE user_id = ?', (user_id,))
            conv_count = cursor.fetchone()[0]

            # Count messages
            cursor = conn.execute('''
                SELECT COUNT(*) FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ?
            ''', (user_id,))
            msg_count = cursor.fetchone()[0]

            return {
                "user_id": user_id,
                "total_conversations": conv_count,
                "total_messages": msg_count,
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# Request/Response Models
# ============================================================================

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"
    metadata: Optional[Dict[str, Any]] = {}

class SaveMessageRequest(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    speaker_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    stats: Dict[str, Any]

# ============================================================================
# Global Storage Instance
# ============================================================================

config = get_config()
storage_path = config["conversation_history"]["storage_path"] or f"{os.getcwd()}/data"
os.makedirs(storage_path, exist_ok=True)

storage = ConversationStorage(f"{storage_path}/conversation_history.db")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Conversation History Service", version="1.0.0")

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
    try:
        # Get some basic stats
        total_conversations = 0
        total_messages = 0

        with sqlite3.connect(f"{storage_path}/conversation_history.db") as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM messages')
            total_messages = cursor.fetchone()[0]

        return HealthResponse(
            status="healthy",
            service="conversation_history",
            timestamp=datetime.now().isoformat(),
            stats={
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "storage_path": storage_path
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/v1/conversations")
async def create_conversation(request: CreateConversationRequest, user_id: str = Depends(get_user_from_auth)):
    """Create new conversation"""
    try:
        conversation = await storage.create_conversation(
            user_id=user_id,
            title=request.title,
            metadata=request.metadata
        )
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: str = Depends(get_user_from_auth)):
    """Get conversation by ID"""
    try:
        conversation = await storage.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "metadata": conversation.metadata,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations")
async def get_user_conversations(limit: int = 20, offset: int = 0, user_id: str = Depends(get_user_from_auth)):
    """Get user's conversations"""
    try:
        conversations = await storage.get_user_conversations(user_id, limit, offset)
        return {
            "conversations": [
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title,
                    "metadata": conv.metadata,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                }
                for conv in conversations
            ],
            "total": len(conversations),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/conversations/{conversation_id}/messages")
async def save_message(conversation_id: str, request: SaveMessageRequest, user_id: str = Depends(get_user_from_auth)):
    """Save message to conversation"""
    try:
        # Verify conversation exists and belongs to user
        conversation = await storage.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message = Message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata,
            speaker_id=request.speaker_id
        )

        await storage.save_message(conversation_id, message)

        return {
            "message_id": message.message_id,
            "conversation_id": conversation_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, limit: int = 50, offset: int = 0, user_id: str = Depends(get_user_from_auth)):
    """Get messages for conversation"""
    try:
        # Verify conversation exists and belongs to user
        conversation = await storage.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = await storage.get_messages(conversation_id, limit, offset)

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "speaker_id": msg.speaker_id,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ],
            "total": len(messages),
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search")
async def search_conversations(request: SearchRequest, user_id: str = Depends(get_user_from_auth)):
    """Search conversations"""
    try:
        conversations = await storage.search_conversations(user_id, request.query, request.limit)

        return {
            "query": request.query,
            "results": [
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title,
                    "metadata": conv.metadata,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                }
                for conv in conversations
            ],
            "total_results": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats(user_id: str = Depends(get_user_from_auth)):
    """Get user statistics"""
    try:
        stats = await storage.get_stats(user_id)
        return stats
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
    print("🚀 Initializing Conversation History Service...")
    print(f"   Storage path: {storage_path}")
    print(f"   Database: {storage_path}/conversation_history.db")
    print("✅ Conversation History Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    print(f"Starting Conversation History Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
