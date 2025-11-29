# Database Indexes - Recommended Indexes

This document lists recommended database indexes for optimal query performance.

## Indexes for Conversation Store

### Primary Indexes (Already Implemented)
- `conversation_id` - Primary key for conversations
- `user_id` - For user conversation lookups

### Recommended Additional Indexes

```sql
-- For fast user conversation listing
CREATE INDEX IF NOT EXISTS idx_user_conversations_user_id 
ON user_conversations(user_id);

-- For fast message retrieval by conversation
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
ON messages(conversation_id);

-- For timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
ON messages(timestamp);

-- For user_id lookups in conversations
CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
ON conversations(user_id);
```

## Indexes for Student Model

```sql
-- For skill mastery queries
CREATE INDEX IF NOT EXISTS idx_interactions_user_skill 
ON interactions(user_id, skill_id);

-- For CEFR progress queries
CREATE INDEX IF NOT EXISTS idx_interactions_user_timestamp 
ON interactions(user_id, timestamp);

-- For error pattern analysis
CREATE INDEX IF NOT EXISTS idx_interactions_has_error 
ON interactions(has_error) WHERE has_error = true;
```

## Indexes for Session Management

```sql
-- For session lookups
CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
ON sessions(user_id);

-- For expired session cleanup
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
ON sessions(expires_at);
```

## Implementation Notes

- These indexes are recommendations based on query patterns
- Actual implementation depends on the database backend (SQLite, PostgreSQL, etc.)
- Indexes should be created during database migration
- Monitor query performance and adjust indexes as needed

## Performance Impact

- **Without indexes**: O(n) full table scans
- **With indexes**: O(log n) index lookups
- **Trade-off**: Slightly slower writes, much faster reads
