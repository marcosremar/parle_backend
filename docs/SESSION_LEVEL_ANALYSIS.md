# Análise de Sessão Completa - Como o Sistema Analisa Toda a Conversa

## 🎯 Objetivo

O sistema não analisa apenas **uma frase individual**, mas agrega dados de **toda a sessão/conversa** para tirar conclusões sobre o progresso do estudante. Este documento explica **exatamente como isso acontece**.

## 📊 Níveis de Análise

O sistema opera em **3 níveis de análise**:

### 1. Análise por Turno (Frase Individual) ✅
- **Quando**: A cada frase que o estudante fala
- **O que analisa**: 
  - Erros na frase atual
  - Skills usadas corretamente na frase atual
  - Features linguísticas da frase atual
- **Localização**: `diagnostic_module` → `analyze_turn()`

### 2. Análise de Sessão (Agregação de Turnos) ✅ NOVO
- **Quando**: Agrega dados de múltiplos turnos na mesma sessão
- **O que analisa**:
  - Tendências de erro (melhorando/piorando ao longo da sessão)
  - Skills problemáticas na sessão (não apenas no turno atual)
  - Padrões linguísticos recorrentes na sessão
  - Progresso geral da sessão
- **Localização**: `diagnostic_module` → `SessionAnalyzer.analyze_session()`

### 3. Análise Histórica (Agregação Temporal) ✅
- **Quando**: Agrega dados dos últimos 90 dias
- **O que analisa**:
  - Padrões de erro por feature linguística (ex: sempre erra na 3ª pessoa)
  - Progresso CEFR ao longo do tempo
  - Skills dominadas vs. problemáticas
- **Localização**: `student_model` → `_analyze_linguistic_error_patterns()`

## 🔄 Como Funciona na Prática

### Exemplo: Sessão com 5 Turnos

```
Turno 1: "Eu fui ao mercado"
  ↓
Análise Turno 1:
  - ✅ Skills corretas: vocabulary_basic, verb_conjugation_past
  - ❌ Erros: 0
  - Features: tense=past, person=1st

Turno 2: "Ele comprou pão"
  ↓
Análise Turno 2:
  - ✅ Skills corretas: vocabulary_basic
  - ❌ Erros: 1 (verb_conjugation_past - 3ª pessoa)
  - Features: tense=past, person=3rd

Turno 3: "Ela foi ao banco"
  ↓
Análise Turno 3:
  - ✅ Skills corretas: vocabulary_basic
  - ❌ Erros: 1 (verb_conjugation_past - 3ª pessoa)
  - Features: tense=past, person=3rd

Turno 4: "Nós compramos frutas"
  ↓
Análise Turno 4:
  - ✅ Skills corretas: vocabulary_basic, verb_conjugation_past
  - ❌ Erros: 0
  - Features: tense=past, person=1st_plural

Turno 5: "Eles foram ao parque"
  ↓
Análise Turno 5:
  - ✅ Skills corretas: vocabulary_basic
  - ❌ Erros: 1 (verb_conjugation_past - 3ª pessoa plural)
  - Features: tense=past, person=3rd_plural
```

### Agregação de Sessão

O `SessionAnalyzer` agrega todos os turnos:

```python
session_analysis = {
    "total_turns": 5,
    "total_errors": 3,
    "avg_errors_per_turn": 0.6,
    "error_trends": [
        {"turn": 1, "error_count": 0},
        {"turn": 2, "error_count": 1},
        {"turn": 3, "error_count": 1},
        {"turn": 4, "error_count": 0},
        {"turn": 5, "error_count": 1}
    ],
    "improving": False,  # Não está melhorando (erros continuam)
    "skill_progress": {
        "verb_conjugation_past": {
            "total_uses": 5,
            "correct_count": 2,
            "error_count": 3,
            "error_rate": 0.6,
            "success_rate": 0.4
        }
    },
    "problematic_skills": ["verb_conjugation_past"],  # 60% de erros
    "linguistic_patterns": {
        "person:3rd": {
            "count": 3,
            "errors": 3,
            "error_rate": 1.0  # 100% de erros na 3ª pessoa!
        },
        "person:1st": {
            "count": 2,
            "errors": 0,
            "error_rate": 0.0  # 0% de erros na 1ª pessoa
        }
    },
    "recommendations": [
        "Você está tendo dificuldades com verb_conjugation_past (60% de erros)",
        "Especialmente na 3ª pessoa (100% de erros). Pratique mais esta área."
    ]
}
```

## 🔧 Integração no Sistema

### 1. Análise por Turno (Já Implementado)

**Fluxo**:
```
User Speech → STT → Diagnostic Module → Analyze Turn
  ↓
Retorna: errors, correct_skills, linguistic_features, semantic_mapping
  ↓
Usado em: current_turn_analysis no prompt
```

### 2. Análise de Sessão (NOVO - Implementado)

**Fluxo**:
```
Multiple Turns → Session Analyzer → Aggregate Data
  ↓
Retorna: session_summary, error_trends, skill_progress, linguistic_patterns
  ↓
Usado em: session_analysis no prompt
```

**Como é construído**:
- O orchestrator coleta `conversation_history` (últimas 10 mensagens)
- Usa `interpretable_knowledge_state` que já agrega dados históricos
- `linguistic_error_patterns` agrega últimos 90 dias
- Passa para `session_analysis` no prompt

### 3. Análise Histórica (Já Implementado)

**Fluxo**:
```
InteractionHistory (90 dias) → _analyze_linguistic_error_patterns()
  ↓
Agrega por feature linguística
  ↓
Retorna: problematic_features, mastered_features, all_features
  ↓
Usado em: interpretable_knowledge_state → linguistic_error_patterns
```

## 📋 Como o AKT Usa Histórico

O **AKT (Attentive Knowledge Tracing)** usa histórico de **30 interações** (attention window):

```python
# Em student_model/app_complete.py → assess_response()
recent_interactions = db.query(InteractionHistory).filter(
    InteractionHistory.user_id == user_id,
    InteractionHistory.skill_id == skill_id,
    InteractionHistory.timestamp >= datetime.now(timezone.utc) - timedelta(days=30)
).order_by(InteractionHistory.timestamp.desc()).limit(30).all()  # Últimas 30 interações

# Restaura histórico para AKT
for interaction in reversed(recent_interactions):
    tracer.interaction_history.append({
        'correct': interaction.correct,
        'timestamp': interaction.timestamp,
        'features': {
            'difficulty': ...,
            'linguistic_features': ...  # Features linguísticas históricas
        }
    })
```

**Como AKT usa isso**:
1. **Attention weights**: Interações mais recentes têm mais peso
2. **Semantic similarity**: Compara interação atual com históricas (mesmas features linguísticas = mais peso)
3. **Pattern adaptation**: Se sempre erra na 3ª pessoa, ajusta `p_T` para essa feature

## 🎯 Como o Prompt Usa Análise de Sessão

O prompt pedagógico agora inclui:

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
    - tense present: 85% de acertos
  Resumo: Maior dificuldade: person 3rd (75% de erros em 8 tentativas)

📊 Contexto da sessão: Conversa com 5 mensagens anteriores
```

## 🔄 Fluxo Completo Integrado

```
1. User fala frase 1
   ↓
2. Analyze Turn 1 → errors, skills, features
   ↓
3. Update Knowledge (AKT usa histórico de 30 interações)
   ↓
4. Compose Prompt:
   - current_turn_analysis (frase 1)
   - session_analysis (agrega histórico de 90 dias)
   - conversation_history (últimas 10 mensagens)
   ↓
5. LLM gera resposta adaptada

... (mesmo processo para frase 2, 3, 4, 5)

6. Após 5 turnos:
   - Session Analyzer pode agregar todos os 5 turnos
   - Identifica padrões na sessão (ex: sempre erra na 3ª pessoa)
   - Gera recomendações baseadas na sessão completa
```

## 📊 Agregações Diferentes

### Por Turno (Frase Individual)
- **Escopo**: 1 frase
- **Dados**: Erros, skills, features da frase atual
- **Uso**: Correção imediata, feedback instantâneo

### Por Sessão (Conversa Atual)
- **Escopo**: Todos os turnos da sessão atual
- **Dados**: Tendências, padrões recorrentes na sessão
- **Uso**: Identificar problemas que persistem na sessão

### Por Histórico (90 dias)
- **Escopo**: Últimas 30-90 dias de interações
- **Dados**: Padrões de longo prazo, progresso CEFR
- **Uso**: Identificar dificuldades crônicas, progresso geral

## ✅ Implementação

### Endpoint Novo: Análise de Sessão

```python
POST /api/diagnostic/analyze_session
{
    "session_turns": [
        {"analysis": {...}},  # Turno 1
        {"analysis": {...}},  # Turno 2
        ...
    ],
    "conversation_history": [...]
}
```

### Integração no Orchestrator

O orchestrator agora:
1. Analisa cada turno individualmente
2. Usa `interpretable_knowledge_state` que já agrega histórico
3. Passa `session_analysis` para o prompt (com padrões históricos)

### Integração no Prompt

`TurnAnalysisLayer` agora renderiza:
- Análise do turno atual
- **Análise da sessão completa** (padrões históricos)

## 🎯 Resultado

O sistema agora:
1. ✅ Analisa cada frase individualmente
2. ✅ Agrega dados de toda a sessão
3. ✅ Usa histórico de 90 dias para identificar padrões
4. ✅ Tira conclusões baseadas em múltiplos níveis de análise
5. ✅ Adapta pedagogia baseado em análise de sessão completa

**Tudo integrado e funcionando!**

