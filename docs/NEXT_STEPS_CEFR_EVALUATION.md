# Próximos Passos - Avaliação CEFR com Conversas Geradas

## 📊 Estado Atual

✅ **Implementado:**
- Geração automática de conversas CEFR com Gemini Flash 2.5
- 12 conversas geradas (2 usuários × 6 níveis CEFR)
- Classificador híbrido com validação AKT (Opção 1)
- Teste de avaliação de conversas gravadas
- Precisão de 91.7% (11/12) sem validação AKT

⚠️ **Pendente:**
- Serviços não estão rodando (LLM, student_model, linguistic_analysis)
- Validação AKT não foi testada (0/12 conversas validadas)
- Um caso C1 foi classificado como C2 (precisão 50% para C1)

---

## 🔴 ALTA PRIORIDADE

### 1. Iniciar Serviços Necessários

Execute em **3 terminais separados**:

**Terminal 1 - LLM Service:**
```bash
cd /Users/marcos/Documents/projects/backend/parle_backend
python3 -m uvicorn src.services.llm.app_complete:app --host 0.0.0.0 --port 8006
```

**Terminal 2 - Student Model Service (para AKT):**
```bash
cd /Users/marcos/Documents/projects/backend/parle_backend
python3 -m uvicorn src.services.student_model.app_complete:app --host 0.0.0.0 --port 8900
```

**Terminal 3 - Linguistic Analysis Service:**
```bash
cd /Users/marcos/Documents/projects/backend/parle_backend
./main.sh start:linguistic
```

**Verificar se estão rodando:**
```bash
curl http://localhost:8006/health  # LLM
curl http://localhost:8900/health  # Student Model
curl http://localhost:8901/health  # Linguistic Analysis
```

### 2. Rodar Avaliação Completa com AKT

**Opção A - Usando script automatizado:**
```bash
./scripts/run_cefr_evaluation.sh
```

**Opção B - Manualmente:**
```bash
export LLM_SERVICE_URL=http://localhost:8006
export STUDENT_MODEL_URL=http://localhost:8900
python3 -m pytest tests/e2e/test_cefr_conversation_history.py -v -s
```

**Resultados esperados:**
- 12 conversas avaliadas
- Validação AKT ativa para todas
- Relatórios gerados em `tests/e2e/reports/conversation_history/`
- Precisão esperada: > 90% (com AKT pode melhorar)

---

## 🟡 MÉDIA PRIORIDADE

### 3. Comparar Resultados com/sem AKT

**Analisar impacto da validação AKT:**
```bash
# Ver resumo da avaliação
cat tests/e2e/reports/conversation_history/conversation_history_summary.json | python3 -m json.tool

# Comparar confiança média
python3 << 'PYTHON'
import json
from pathlib import Path

summary = json.loads(Path('tests/e2e/reports/conversation_history/conversation_history_summary.json').read_text())

# Calcular métricas
total = len(summary)
akt_validated = sum(1 for s in summary if s.get('akt_validated', False))
accuracy = sum(1 for s in summary if s['predicted_level'] == s['cefr_level']) / total
avg_confidence = sum(s['confidence'] for s in summary) / total

print(f"📊 Métricas da Avaliação:")
print(f"   Total de conversas: {total}")
print(f"   AKT validado: {akt_validated}/{total} ({akt_validated/total*100:.1f}%)")
print(f"   Precisão: {accuracy*100:.1f}%")
print(f"   Confiança média: {avg_confidence:.1f}%")
PYTHON
```

**Comparar com avaliação de textos estáticos:**
```bash
# Rodar validação de textos estáticos
python3 tests/e2e/validate_classifier.py

# Comparar resultados
# - tests/e2e/reports/validation_hybrid_*.json (textos estáticos)
# - tests/e2e/reports/conversation_history/conversation_history_summary.json (conversas)
```

### 4. Melhorar Prompts CEFR (se necessário)

**Problema identificado:**
- C1 teve 50% de precisão (1/2 classificado como C2)
- Conversa C1 muito complexa, confundida com C2

**Solução:**
1. Editar `scripts/seed_cefr_conversations.py`
2. Ajustar prompt C1 para ser mais específico sobre limites
3. Adicionar exemplos de diferenças C1 vs C2
4. Regenerar conversas C1:
   ```bash
   # Deletar conversas C1 existentes
   sqlite3 data/conversation_history.db "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id LIKE 'cefr_c1_%'); DELETE FROM conversations WHERE user_id LIKE 'cefr_c1_%';"
   
   # Regenerar apenas C1
   python3 scripts/seed_cefr_conversations.py
   ```

---

## 🟢 BAIXA PRIORIDADE

### 5. Gerar Relatório Final Consolidado

**Criar documento comparativo:**
- Precisão: Textos estáticos vs Conversas geradas
- Impacto do AKT na precisão
- Análise por nível CEFR
- Recomendações de melhorias

### 6. Expandir Dataset de Conversas

**Para melhorar robustez:**
- Gerar 4-5 usuários por nível (atualmente 2)
- Adicionar mais cenários variados
- Incluir casos edge (limites entre níveis)

**Modificar script:**
```python
# Em scripts/seed_cefr_conversations.py, mudar:
for user_num in [1, 2]:  # Atual
# Para:
for user_num in [1, 2, 3, 4, 5]:  # Expandido
```

---

## 📈 Métricas de Sucesso

**Objetivos:**
- ✅ Precisão geral > 90% (atual: 91.7%)
- ⏳ Precisão por nível > 80% (C1 precisa melhorar)
- ⏳ Validação AKT ativa em 100% das conversas
- ⏳ Confiança média > 50% (atual: 45.3%)

**Próximas métricas a monitorar:**
- Impacto do AKT na precisão (+5-10% esperado)
- Confiança ajustada por convergência AKT
- Taxa de detecção de inconsistências

---

## 🛠️ Comandos Úteis

**Verificar conversas no banco:**
```bash
sqlite3 data/conversation_history.db "SELECT user_id, metadata FROM conversations WHERE user_id LIKE 'cefr_%';"
```

**Ver mensagens de uma conversa:**
```bash
sqlite3 data/conversation_history.db "SELECT role, content FROM messages WHERE conversation_id = (SELECT id FROM conversations WHERE user_id = 'cefr_a1_user_1') ORDER BY created_at;"
```

**Regenerar todas as conversas:**
```bash
# Deletar todas
sqlite3 data/conversation_history.db "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id LIKE 'cefr_%'); DELETE FROM conversations WHERE user_id LIKE 'cefr_%';"

# Regenerar
python3 scripts/seed_cefr_conversations.py
```

**Ver relatórios gerados:**
```bash
ls -lh tests/e2e/reports/conversation_history/
cat tests/e2e/reports/conversation_history/conversation_history_summary.json | python3 -m json.tool
```

---

## 📝 Notas

- **AKT Validation:** A validação AKT (Opção 1) ajusta confiança baseado em convergência, não muda o nível identificado
- **Serviços:** Todos os serviços devem estar rodando para avaliação completa
- **Modelo LLM:** Sistema usa Gemini Flash 2.5 via OpenRouter
- **Banco de Dados:** Conversas armazenadas em `data/conversation_history.db`

---

**Última atualização:** 2025-11-23
