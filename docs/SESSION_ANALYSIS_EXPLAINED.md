# Como o Sistema Analisa Toda a Sessão (Não Apenas uma Frase)

## 🎯 Pergunta do Usuário

> "O sistema não avalia apenas a frase, mas analisa toda a sessão para tirar essas conclusões. Como isso acontece exatamente?"

## 📊 Resposta: 3 Níveis de Análise

O sistema opera em **3 níveis de análise** que se complementam:

### 1. Análise por Turno (Frase Individual) ✅

**O que é**: Analisa cada frase que o estudante fala individualmente.

**Como funciona**:
```
Turno 1: "Eu fui ao mercado"
  ↓
Diagnostic Module → analyze_turn()
  ↓
Retorna:
  - errors: []
  - correct_skills: ["vocabulary_basic", "verb_conjugation_past"]
  - linguistic_features: {tense: "past", person: "1st"}
```

**Uso**: Correção imediata, feedback instantâneo.

### 2. Análise de Sessão (Agregação de Turnos) ✅ NOVO

**O que é**: Agrega dados de **todos os turnos da sessão atual** para identificar padrões.

**Como funciona**:
```
Turno 1: 0 erros, 2 skills corretas
Turno 2: 1 erro (3ª pessoa), 1 skill correta
Turno 3: 1 erro (3ª pessoa), 1 skill correta
Turno 4: 0 erros, 2 skills corretas
Turno 5: 1 erro (3ª pessoa), 1 skill correta
  ↓
Session Analyzer → analyze_session()
  ↓
Retorna:
  - total_turns: 5
  - avg_errors_per_turn: 0.6
  - error_trends: [0, 1, 1, 0, 1]  # Padrão: erros na 3ª pessoa
  - improving: False  # Não está melhorando
  - problematic_skills: ["verb_conjugation_past"]
  - linguistic_patterns: {
      "person:3rd": {error_rate: 1.0}  # 100% de erros na 3ª pessoa!
    }
  - recommendations: [
      "Você está tendo dificuldades com verb_conjugation_past (60% de erros)",
      "Especialmente na 3ª pessoa (100% de erros). Pratique mais esta área."
    ]
```

**Uso**: Identificar problemas que persistem na sessão, tendências de melhora/piora.

### 3. Análise Histórica (Agregação Temporal) ✅

**O que é**: Agrega dados dos **últimos 90 dias** de interações para identificar padrões de longo prazo.

**Como funciona**:
```
InteractionHistory (últimos 90 dias)
  - 45 interações com "person:3rd"
  - 30 erros, 15 acertos
  - error_rate: 66.7%
  ↓
_analyze_linguistic_error_patterns()
  ↓
Retorna:
  - problematic_features: [
      {
        "feature_key": "person:3rd",
        "error_rate": 0.667,
        "total_attempts": 45
      }
    ]
  - summary: "Maior dificuldade: person 3rd (67% de erros em 45 tentativas)"
```

**Uso**: Identificar dificuldades crônicas, progresso geral ao longo do tempo.

## 🔄 Como Está Integrado

### No Orchestrator

```python
# STEP 2.5: Analyze Current Turn (frase individual)
turn_analysis = await diagnostic_module.analyze_turn(
    user_text=user_transcript,
    valid_skills=valid_skills
)

# STEP 1.7: Compose Prompt (usa análise + histórico)
interpretable_state = await student_model.get_interpretable_knowledge_state(user_id)
# interpretable_state inclui linguistic_error_patterns (90 dias)

session_analysis = {
    "historical_patterns": interpretable_state.get("linguistic_error_patterns"),
    "session_context": f"Conversa com {len(conversation_history)} mensagens anteriores"
}

prompt_context = {
    "current_turn_analysis": turn_analysis,  # Análise da frase atual
    "session_analysis": session_analysis,     # Padrões históricos (90 dias)
    "conversation_history": conversation_history  # Últimas 10 mensagens
}
```

### No Prompt Pedagógico

O `TurnAnalysisLayer` renderiza:

```
[Current Turn Analysis]
Erros identificados (1):
  1. grammar em 'verb_conjugation_past' (medium)

✅ Skills usadas corretamente (2):
  - vocabulary_basic
  - article_definite

📊 Features linguísticas identificadas:
  Tempo: past, Pessoa: 3rd, Número: singular

[Análise da Sessão Completa]
Padrões históricos identificados (últimos 90 dias):
  Features problemáticas:
    - person 3rd: 75% de erros
    - tense past: 60% de erros
  Features dominadas:
    - person 1st: 90% de acertos
  Resumo: Maior dificuldade: person 3rd (75% de erros em 8 tentativas)

📊 Contexto da sessão: Conversa com 5 mensagens anteriores
```

## 🧠 Como o AKT Usa Histórico

O **AKT (Attentive Knowledge Tracing)** carrega **últimas 30 interações** do banco:

```python
# Em student_model/app_complete.py → assess_response()
recent_interactions = db.query(InteractionHistory).filter(
    InteractionHistory.user_id == user_id,
    InteractionHistory.skill_id == skill_id,
    InteractionHistory.timestamp >= datetime.now(timezone.utc) - timedelta(days=30)
).limit(30).all()  # Últimas 30 interações

# Restaura para AKT
for interaction in recent_interactions:
    tracer.interaction_history.append({
        'correct': interaction.correct,
        'timestamp': interaction.timestamp,
        'features': {
            'linguistic_features': {...}  # Features históricas
        }
    })
```

**Como AKT usa**:
1. **Attention weights**: Interações mais recentes têm mais peso
2. **Semantic similarity**: Compara interação atual com históricas
   - Se sempre erra na 3ª pessoa, interações com `person:3rd` têm mais peso
3. **Pattern adaptation**: Se sempre erra na 3ª pessoa, ajusta `p_T` para essa feature

## 📈 Exemplo Completo: Como Conclusões São Tiradas

### Cenário: Estudante fala 5 frases na sessão

**Turno 1**: "Eu fui ao mercado"
- Análise: ✅ 0 erros, skills corretas

**Turno 2**: "Ele comprou pão"
- Análise: ❌ 1 erro (verb_conjugation_past - 3ª pessoa)

**Turno 3**: "Ela foi ao banco"
- Análise: ❌ 1 erro (verb_conjugation_past - 3ª pessoa)

**Turno 4**: "Nós compramos frutas"
- Análise: ✅ 0 erros

**Turno 5**: "Eles foram ao parque"
- Análise: ❌ 1 erro (verb_conjugation_past - 3ª pessoa plural)

### Conclusões Tiradas

#### 1. Por Turno (Individual)
- Turno 5: "Erro na 3ª pessoa plural"

#### 2. Por Sessão (Agregação)
- **Padrão identificado**: Sempre erra na 3ª pessoa (singular e plural)
- **Tendência**: Não está melhorando (erros continuam)
- **Recomendação**: "Foque em praticar 3ª pessoa"

#### 3. Por Histórico (90 dias)
- **Padrão crônico**: "person:3rd" tem 75% de erros em 45 tentativas
- **Conclusão**: Dificuldade persistente, não apenas nesta sessão

### Como o Prompt Usa Isso

```
[Current Turn Analysis]
Erros identificados (1):
  1. grammar em 'verb_conjugation_past' (medium)
     → Erro na conjugação do verbo no passado (3ª pessoa plural)

[Análise da Sessão Completa]
Padrões históricos identificados (últimos 90 dias):
  Features problemáticas:
    - person 3rd: 75% de erros (45 tentativas)
  Resumo: Maior dificuldade: person 3rd (75% de erros em 45 tentativas)

💡 Instruções pedagógicas:
  - Foque nos erros identificados ao responder
  - O estudante tem dificuldade crônica com 3ª pessoa (75% de erros)
  - Use correção explícita e pratique mais esta área
```

## ✅ Resumo: Como Funciona Exatamente

1. **Análise por Turno**: Cada frase é analisada individualmente
2. **Armazenamento**: Cada análise é salva em `InteractionHistory` com `linguistic_features`
3. **Agregação de Sessão**: `SessionAnalyzer` agrega turnos da sessão atual
4. **Agregação Histórica**: `_analyze_linguistic_error_patterns()` agrega últimos 90 dias
5. **AKT usa histórico**: Carrega últimas 30 interações para attention e similaridade
6. **Prompt recebe tudo**: 
   - Análise do turno atual
   - Padrões históricos (90 dias)
   - Contexto da sessão (conversation_history)
7. **LLM gera resposta**: Baseada em análise de turno + padrões de sessão + histórico

**Tudo integrado para tirar conclusões baseadas em múltiplos níveis de análise!**

