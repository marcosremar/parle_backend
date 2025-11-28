import sys
sys.path.insert(0, 'tests/e2e')
from test_cefr_conversation_history import _load_seeded_conversations
from pathlib import Path

convs = _load_seeded_conversations(Path('data/conversation_history.db'))
print(f'Conversas carregadas: {len(convs)}')
if convs:
    print(f'Primeira: {convs[0].get("user_id", "N/A")}')
    print(f'Níveis: {set(c.get("cefr_level", "N/A") for c in convs)}')
else:
    print('Nenhuma conversa encontrada')
