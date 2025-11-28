# Análise de Padrões de Erro por Feature Linguística

## 📋 Visão Geral

O sistema usa **LLM para analisar a fala do usuário** e identificar quais **features linguísticas** estão causando mais erros. Esta análise é integrada em múltiplos pontos do sistema para fornecer feedback didático preciso.

## 🔄 Fluxo de Integração

### 1. Extração de Features pelo LLM

**Localização**: `diagnostic_module/llm_client.py` → `analyze_grammar()`

O LLM analisa cada turno do usuário e extrai:

```json
{
  "errors": [
    {
      "error_type": "grammar",
      "skill_id": "verb_conjugation_past",
      "linguistic_features": {
        "tense": "past",
        "person": "3rd",
        "number": "singular",
        "mood": "indicative"
      }
    }
  ],
  "correct_skills": ["vocabulary_basic"],
  "linguistic_features": {
    "tense": "present",
    "person": "1st",
    "register": "informal"
  }
}
```

**Features extraídas**:
- **Verbos**: `tense`, `person`, `number`, `mood`, `aspect`
- **Vocabulário**: `register` (formal/informal/neutral), `domain` (family/travel/emotions/daily_life/general)
- **Artigos**: `article_type` (definite/indefinite)
- **Preposições**: `preposition_type` (location/basic)

### 2. Armazenamento no Banco de Dados

**Localização**: `student_model/app_complete.py` → `assess_response()`

As linguistic_features são armazenadas em `InteractionHistory.semantic_features` como JSON:

```python
semantic_features_data = {
    'linguistic_features': linguistic_features,
    'difficulty': context.get('difficulty'),
    'complexity': context.get('complexity'),
    'error_type': context.get('error_type'),
    'error_severity': context.get('error_severity')
}
```

### 3. Agregação e Análise de Padrões

**Localização**: `student_model/app_complete.py` → `_analyze_linguistic_error_patterns()`

A função agrega todas as interações dos últimos 90 dias e calcula:

- **Taxa de erro por feature**: `error_rate = error_count / total_attempts`
- **Features problemáticas**: `error_rate > 0.5` e `>= 5 tentativas`
- **Features dominadas**: `error_rate < 0.2` e `>= 5 tentativas`

**Exemplo de resultado**:
```json
{
  "total_interactions_analyzed": 45,
  "features_analyzed": 12,
  "problematic_features": [
    {
      "feature_key": "person:3rd",
      "feature_type": "person",
      "feature_value": "3rd",
      "total_attempts": 8,
      "error_count": 6,
      "correct_count": 2,
      "error_rate": 0.75,
      "success_rate": 0.25
    }
  ],
  "mastered_features": [
    {
      "feature_key": "tense:present",
      "feature_type": "tense",
      "feature_value": "present",
      "total_attempts": 10,
      "error_count": 1,
      "correct_count": 9,
      "error_rate": 0.1,
      "success_rate": 0.9
    }
  ],
  "summary": "Maior dificuldade: person 3rd (75% de erros em 8 tentativas). Melhor domínio: tense present (90% de acertos em 10 tentativas)"
}
```

### 4. Integração no Interpretable Knowledge State

**Localização**: `student_model/app_complete.py` → `get_interpretable_knowledge_state()`

A análise de padrões é incluída no estado interpretável e gera recomendações:

```python
if linguistic_error_patterns.get("problematic_features"):
    problematic = linguistic_error_patterns["problematic_features"][:2]
    for feature in problematic:
        feature_name = feature.get("feature_key", "").replace(":", " ")
        error_rate = feature.get("error_rate", 0.0)
        recommendations.append(
            f"Você está tendo dificuldades com {feature_name} "
            f"({error_rate:.0%} de erros). Pratique mais esta área."
        )
```

### 5. Uso no Pedagogical Policy

**Localização**: `pedagogical_policy/prompt_composer/layers/student_state_layer.py`

O `interpretable_knowledge_state` (que inclui `linguistic_error_patterns`) é passado para o `StudentStateLayer`, que renderiza no prompt:

```
[Análise Detalhada do Progresso]
Progresso por dimensão:
  - grammar: 65%
  - vocabulary: 72%
  - pronunciation: 58%

Skills que precisam de prática:
  - person 3rd: 25% (75% de erros)
  - tense past: 40% (60% de erros)

Recomendações pedagógicas:
  - Você está tendo dificuldades com person 3rd (75% de erros). Pratique mais esta área.
```

### 6. Adaptação de Parâmetros no AKT

**Localização**: `student_model/knowledge_tracer/akt_tracer.py` → `_adapt_parameters()`

O AKT rastreia padrões de erro por feature em tempo real:

```python
# Rastrear padrão "person:3rd"
pattern_key = "person:3rd"
if error_rate > 0.7 and total_attempts >= 3:
    # Aumentar p_T (precisa de mais prática)
    self.adaptive_p_T = min(0.3, self.adaptive_p_T + 0.02)
```

## 🔌 Endpoints Disponíveis

### 1. Análise de Padrões de Erro (Novo)

**Endpoint**: `GET /api/student/{user_id}/linguistic_error_patterns`

**Resposta**:
```json
{
  "total_interactions_analyzed": 45,
  "features_analyzed": 12,
  "problematic_features": [
    {
      "feature_key": "person:3rd",
      "feature_type": "person",
      "feature_value": "3rd",
      "total_attempts": 8,
      "error_count": 6,
      "error_rate": 0.75,
      "success_rate": 0.25
    }
  ],
  "mastered_features": [...],
  "all_features": [...],
  "summary": "Maior dificuldade: person 3rd (75% de erros)..."
}
```

### 2. Estado Interpretável (Inclui Padrões)

**Endpoint**: `GET /api/student/{user_id}/interpretable_knowledge_state`

**Resposta inclui**:
- `linguistic_error_patterns`: Análise completa de padrões
- `recommendations`: Recomendações baseadas em padrões problemáticos

## 📊 Como Funciona na Prática

### Exemplo: Estudante sempre erra na 3ª pessoa

1. **Turno 1**: "Ele foi ao mercado" → LLM extrai `person:3rd`, `tense:past`
   - Erro detectado → `linguistic_features` salvas com `person:3rd`

2. **Turno 2**: "Ela comprou pão" → LLM extrai `person:3rd`, `tense:past`
   - Erro detectado → `linguistic_features` salvas

3. **Após 5+ erros com `person:3rd`**:
   - `_analyze_linguistic_error_patterns()` identifica padrão
   - `problematic_features` inclui `person:3rd` com `error_rate: 0.75`
   - Recomendação gerada: "Você está tendo dificuldades com person 3rd (75% de erros)"
   - AKT ajusta `p_T` para essa feature específica

4. **No próximo prompt pedagógico**:
   - `StudentStateLayer` inclui: "Skills que precisam de prática: person 3rd (75% de erros)"
   - LLM recebe contexto e pode focar em praticar 3ª pessoa

## 🎯 Integração Completa

```
User Speech
    ↓
LLM Analysis (diagnostic_module)
    ↓
Extract linguistic_features (tense, person, number, register, domain)
    ↓
Store in InteractionHistory.semantic_features
    ↓
AKT tracks patterns in real-time (feature_patterns)
    ↓
_analyze_linguistic_error_patterns() aggregates historical data
    ↓
get_interpretable_knowledge_state() includes patterns
    ↓
Pedagogical Policy receives patterns
    ↓
StudentStateLayer renders in prompt
    ↓
LLM generates targeted feedback
```

## ✅ Status da Implementação

- ✅ LLM extrai linguistic_features de cada turno
- ✅ Features armazenadas no banco de dados
- ✅ Análise de padrões agregando dados históricos
- ✅ Padrões incluídos no interpretable_knowledge_state
- ✅ Recomendações geradas baseadas em padrões
- ✅ Integrado no pedagogical_policy (prompt)
- ✅ AKT adapta parâmetros baseado em padrões
- ✅ Endpoint dedicado criado: `/api/student/{user_id}/linguistic_error_patterns`

## 🔍 Verificação

Para verificar se está funcionando:

1. **Fazer várias interações** com erros em uma feature específica (ex: sempre errar na 3ª pessoa)
2. **Chamar endpoint**: `GET /api/student/{user_id}/linguistic_error_patterns`
3. **Verificar** se `problematic_features` inclui a feature problemática
4. **Verificar** se `interpretable_knowledge_state` inclui recomendações sobre essa feature
5. **Verificar** se o prompt pedagógico inclui informações sobre a feature problemática

