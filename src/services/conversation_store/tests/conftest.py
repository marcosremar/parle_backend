"""Shared fixtures for Conversation Store Service tests."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4


@pytest.fixture
def store_config():
    """Conversation store configuration."""
    return {
        "max_conversations": 10000,
        "max_messages_per_conversation": 1000,
        "retention_days": 90,
        "enable_search": True,
        "enable_archival": True,
        "auto_archive_days": 30,
        "compression_enabled": True,
        "encryption_enabled": True
    }


@pytest.fixture
def conversation_store():
    """In-memory conversation store."""

    class ConversationStore:
        def __init__(self):
            self.conversations = {}
            self.messages = {}
            self.indexes = {
                "by_user": {},
                "by_date": {},
                "by_topic": {}
            }
            self.metrics = {
                "total_conversations": 0,
                "total_messages": 0,
                "searches_performed": 0,
                "archived_count": 0
            }

        async def create_conversation(self, user_id: str, metadata: Dict = None) -> str:
            """Create new conversation."""
            conversation_id = str(uuid4())

            self.conversations[conversation_id] = {
                "id": conversation_id,
                "user_id": user_id,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "status": "active",
                "metadata": metadata or {},
                "message_count": 0
            }

            # Index by user
            if user_id not in self.indexes["by_user"]:
                self.indexes["by_user"][user_id] = []
            self.indexes["by_user"][user_id].append(conversation_id)

            self.metrics["total_conversations"] += 1
            return conversation_id

        async def save_message(self, conversation_id: str, message: Dict) -> str:
            """Save message to conversation."""
            if conversation_id not in self.conversations:
                raise ValueError("Conversation not found")

            message_id = str(uuid4())
            message_data = {
                "id": message_id,
                "conversation_id": conversation_id,
                "content": message.get("content", ""),
                "role": message.get("role", "user"),
                "timestamp": datetime.now(),
                "metadata": message.get("metadata", {})
            }

            # Store message
            if conversation_id not in self.messages:
                self.messages[conversation_id] = []
            self.messages[conversation_id].append(message_data)

            # Update conversation
            self.conversations[conversation_id]["updated_at"] = datetime.now()
            self.conversations[conversation_id]["message_count"] += 1

            self.metrics["total_messages"] += 1
            return message_id

        async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
            """Get conversation by ID."""
            return self.conversations.get(conversation_id)

        async def get_messages(self, conversation_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
            """Get messages from conversation."""
            if conversation_id not in self.messages:
                return []

            messages = self.messages[conversation_id]
            return messages[offset:offset + limit]

        async def search_conversations(self, query: str, user_id: str = None) -> List[Dict]:
            """Search conversations."""
            self.metrics["searches_performed"] += 1

            results = []
            for conv_id, conv in self.conversations.items():
                # Filter by user if specified
                if user_id and conv["user_id"] != user_id:
                    continue

                # Search in messages
                if conv_id in self.messages:
                    for msg in self.messages[conv_id]:
                        if query.lower() in msg["content"].lower():
                            results.append(conv)
                            break

            return results

        async def archive_conversation(self, conversation_id: str) -> bool:
            """Archive conversation."""
            if conversation_id not in self.conversations:
                return False

            self.conversations[conversation_id]["status"] = "archived"
            self.conversations[conversation_id]["archived_at"] = datetime.now()
            self.metrics["archived_count"] += 1
            return True

        async def delete_conversation(self, conversation_id: str) -> bool:
            """Delete conversation."""
            if conversation_id not in self.conversations:
                return False

            del self.conversations[conversation_id]
            if conversation_id in self.messages:
                del self.messages[conversation_id]
            return True

        async def get_user_conversations(self, user_id: str) -> List[Dict]:
            """Get all conversations for user."""
            conv_ids = self.indexes["by_user"].get(user_id, [])
            return [self.conversations[cid] for cid in conv_ids if cid in self.conversations]

    return ConversationStore()


@pytest.fixture
def sample_conversation():
    """Sample conversation data."""
    return {
        "user_id": "user_123",
        "metadata": {
            "topic": "technical_support",
            "channel": "web",
            "language": "en"
        }
    }


@pytest.fixture
def sample_messages():
    """Sample messages."""
    return [
        {
            "role": "user",
            "content": "Hello, I need help with my account",
            "metadata": {"sentiment": "neutral"}
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your account. What seems to be the issue?",
            "metadata": {"sentiment": "positive"}
        },
        {
            "role": "user",
            "content": "I can't log in to my account",
            "metadata": {"sentiment": "negative"}
        },
        {
            "role": "assistant",
            "content": "Let me help you reset your password",
            "metadata": {"sentiment": "helpful"}
        }
    ]


@pytest.fixture
def search_engine():
    """Search engine for conversations."""

    class SearchEngine:
        def __init__(self):
            self.index = {}
            self.search_count = 0

        def index_message(self, message_id: str, content: str, metadata: Dict):
            """Index message for search."""
            words = content.lower().split()
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append({
                    "message_id": message_id,
                    "metadata": metadata
                })

        def search(self, query: str) -> List[Dict]:
            """Search for query."""
            self.search_count += 1
            query_words = query.lower().split()

            results = []
            for word in query_words:
                if word in self.index:
                    results.extend(self.index[word])

            # Remove duplicates
            seen = set()
            unique_results = []
            for result in results:
                if result["message_id"] not in seen:
                    seen.add(result["message_id"])
                    unique_results.append(result)

            return unique_results

        def get_search_suggestions(self, partial: str) -> List[str]:
            """Get search suggestions."""
            suggestions = []
            for word in self.index.keys():
                if word.startswith(partial.lower()):
                    suggestions.append(word)
            return suggestions[:5]

    return SearchEngine()


@pytest.fixture
def archival_manager():
    """Archival and retention manager."""

    class ArchivalManager:
        def __init__(self):
            self.archived_conversations = {}
            self.retention_policy_days = 90
            self.auto_archive_days = 30

        async def should_archive(self, conversation: Dict) -> bool:
            """Check if conversation should be archived."""
            updated_at = conversation.get("updated_at")
            if not updated_at:
                return False

            days_inactive = (datetime.now() - updated_at).days
            return days_inactive >= self.auto_archive_days

        async def should_delete(self, conversation: Dict) -> bool:
            """Check if conversation should be deleted."""
            created_at = conversation.get("created_at")
            if not created_at:
                return False

            days_old = (datetime.now() - created_at).days
            return days_old >= self.retention_policy_days

        async def archive(self, conversation_id: str, conversation: Dict):
            """Archive conversation."""
            self.archived_conversations[conversation_id] = {
                **conversation,
                "archived_at": datetime.now(),
                "status": "archived"
            }

        async def restore(self, conversation_id: str) -> Optional[Dict]:
            """Restore archived conversation."""
            if conversation_id in self.archived_conversations:
                conv = self.archived_conversations[conversation_id]
                conv["status"] = "active"
                del self.archived_conversations[conversation_id]
                return conv
            return None

        def get_archived_count(self) -> int:
            """Get count of archived conversations."""
            return len(self.archived_conversations)

    return ArchivalManager()


@pytest.fixture
def pagination_helper():
    """Pagination helper."""

    class PaginationHelper:
        def __init__(self):
            self.default_limit = 50
            self.max_limit = 100

        def paginate(self, items: List, page: int = 1, limit: int = None) -> Dict:
            """Paginate items."""
            limit = min(limit or self.default_limit, self.max_limit)
            offset = (page - 1) * limit

            total_items = len(items)
            total_pages = (total_items + limit - 1) // limit

            return {
                "items": items[offset:offset + limit],
                "page": page,
                "limit": limit,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }

    return PaginationHelper()


@pytest.fixture
def performance_tracker():
    """Performance tracking."""

    class PerformanceTracker:
        def __init__(self):
            self.query_times = []
            self.write_times = []
            self.search_times = []

        def record_query(self, duration_ms: float):
            """Record query duration."""
            self.query_times.append(duration_ms)

        def record_write(self, duration_ms: float):
            """Record write duration."""
            self.write_times.append(duration_ms)

        def record_search(self, duration_ms: float):
            """Record search duration."""
            self.search_times.append(duration_ms)

        def get_avg_query_time(self) -> float:
            """Get average query time."""
            return sum(self.query_times) / len(self.query_times) if self.query_times else 0

        def get_avg_write_time(self) -> float:
            """Get average write time."""
            return sum(self.write_times) / len(self.write_times) if self.write_times else 0

        def get_avg_search_time(self) -> float:
            """Get average search time."""
            return sum(self.search_times) / len(self.search_times) if self.search_times else 0

        def get_p95_query_time(self) -> float:
            """Get P95 query time."""
            if not self.query_times:
                return 0
            sorted_times = sorted(self.query_times)
            index = int(len(sorted_times) * 0.95)
            return sorted_times[min(index, len(sorted_times) - 1)]

    return PerformanceTracker()


@pytest.fixture
def conversation_validator():
    """Conversation data validator."""

    class ConversationValidator:
        def validate_conversation(self, conversation: Dict) -> bool:
            """Validate conversation data."""
            required_fields = ["id", "user_id", "created_at", "status"]
            return all(field in conversation for field in required_fields)

        def validate_message(self, message: Dict) -> bool:
            """Validate message data."""
            required_fields = ["role", "content"]
            valid_roles = ["user", "assistant", "system"]

            has_fields = all(field in message for field in required_fields)
            valid_role = message.get("role") in valid_roles

            return has_fields and valid_role

        def sanitize_content(self, content: str) -> str:
            """Sanitize message content."""
            # Remove potentially harmful content
            sanitized = content.strip()
            # Limit length
            max_length = 10000
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length]
            return sanitized

    return ConversationValidator()
