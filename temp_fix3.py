import sys
import json
import sqlite3
sys.path.insert(0, 'tests/e2e')
from pathlib import Path

# Test the function directly
def _load_seeded_conversations(db_path: Path):
    """Carrega conversas marcadas como seed CEFR no banco."""
    if not db_path.exists():
        return []

    conversations = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT id, user_id, metadata FROM conversations WHERE user_id LIKE 'cefr_%'")
        rows = cursor.fetchall()
        print(f"Rows from DB: {len(rows)}")
        for conv_id, user_id, metadata_raw in rows:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            print(f"Processing {user_id}, metadata: {metadata}")
            cefr_level = metadata.get("cefr_level")
            if not cefr_level:
                continue

            messages_cursor = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conv_id,),
            )
            message_rows = messages_cursor.fetchall()
            user_messages = [row[1] for row in message_rows if row[0] == "user"]
            text = " ".join(user_messages).strip()

            conversations.append({
                "user_id": user_id,
                "cefr_level": cefr_level,
                "text": text,
            })

    return conversations

convs = _load_seeded_conversations(Path('data/conversation_history.db'))
print(f'Conversas carregadas: {len(convs)}')
if convs:
    print(f'Primeira: {convs[0].get("user_id", "N/A")}')
    print(f'Níveis: {set(c.get("cefr_level", "N/A") for c in convs)}')
else:
    print('Nenhuma conversa encontrada')
