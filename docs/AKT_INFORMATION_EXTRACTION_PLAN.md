# Plano de Implementação: Extração de Informações do AKT

**Versão:** 1.0  
**Data:** 2025-11-22  
**Status:** Planejamento

---

## 📋 Sumário Executivo

Este documento apresenta um plano completo para implementar a extração e utilização de diferentes tipos de informação que o **Attentive Knowledge Tracing (AKT)** pode fornecer sobre o progresso de aprendizado dos estudantes.

O AKT já está implementado no sistema, mas seu potencial completo ainda não está sendo explorado. Este plano detalha **9 tipos de informação** que podem ser extraídas e como implementá-las de forma incremental.

---

## 🎯 Objetivos

1. **Maximizar o uso do AKT** para obter insights sobre o aprendizado do estudante
2. **Expor informações acionáveis** via API para outros serviços (Pedagogical Policy, Learning Path, UI)
3. **Melhorar a adaptação pedagógica** baseada em dados ricos do AKT
4. **Fornecer feedback detalhado** para estudantes e professores

---

## 📊 Tipos de Informação a Implementar

### Visão Geral

| # | Tipo de Informação | Prioridade | Complexidade | Status Atual |
|---|-------------------|------------|--------------|--------------|
| 1 | Probabilidade de domínio por skill | Alta | Baixa | ✅ Implementado |
| 2 | Probabilidade de acerto futuro | Alta | Média | ⚠️ Parcial |
| 3 | Curva de aprendizado | Média | Média | ❌ Não implementado |
| 4 | Skills fortes e fracas | Alta | Baixa | ✅ Implementado |
| 5 | Pontos fracos por feature linguística | Alta | Média | ✅ Implementado |
| 6 | Impacto da dificuldade (IRT) | Baixa | Alta | ❌ Não implementado |
| 7 | Esquecimento e necessidade de revisão | Média | Alta | ⚠️ Parcial |
| 8 | Resumo por dimensão/CEFR | Alta | Baixa | ✅ Implementado |
| 9 | Taxas de acerto por skill | Média | Baixa | ⚠️ Parcial |

---

## 1️⃣ Probabilidade de Domínio por Skill

### 📝 Descrição

Extrair, para cada skill, um valor contínuo **0.0–1.0** representando o quanto o aluno domina aquela habilidade.

### ✅ Status Atual

**✅ IMPLEMENTADO**

- Já disponível em `SkillMastery.mastery_probability`
- Exposto via `/api/student/{user_id}/skills`

### 🔧 Melhorias Propostas

#### Backend

- [ ] Adicionar método helper: `get_mastery_for_user(user_id, skill_id)`
- [ ] Garantir que mastery é sempre atualizado após `assess_response`
- [ ] Adicionar validação de range [0.0, 1.0]

#### API

- [ ] Novo endpoint dedicado (opcional):
  ```
  GET /api/student/{user_id}/skill/{skill_id}/mastery
  
  Response:
  {
    "skill_id": "verb_conjugation_past",
    "mastery_probability": 0.78,
    "mastery_status": "mastered",
    "last_updated": "2025-11-22T01:30:00Z"
  }
  ```

#### Uso Didático

- [ ] Pedagogical Policy usa para decidir estratégia:
  - `mastery < 0.4` → **teach** (ensinar)
  - `0.4 ≤ mastery ≤ 0.7` → **reinforce** (reforçar)
  - `mastery > 0.7` → **challenge** (desafiar)

### 📅 Timeline

**Fase 1 (já concluída):** Implementação base  
**Fase 2 (1 semana):** Melhorias e otimizações

---

## 2️⃣ Probabilidade de Acerto Futuro

### 📝 Descrição

Calcular a probabilidade de o aluno **acertar a próxima questão** de uma skill específica, considerando:
- Mastery atual
- Dificuldade do item
- Performance recente
- Esquecimento temporal

### ⚠️ Status Atual

**⚠️ PARCIALMENTE IMPLEMENTADO**

- Método `predict_performance` existe no `AttentiveKnowledgeTracer`
- Não está exposto via API
- Não é usado ativamente na adaptação pedagógica

### 🔧 Implementação

#### Backend

**Arquivo:** `src/services/student_model/app_complete.py`

```python
@app.get("/api/student/{user_id}/skill/{skill_id}/prediction")
async def predict_skill_performance(
    user_id: str,
    skill_id: str,
    difficulty: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Prediz a probabilidade de o estudante acertar uma questão desta skill
    """
    # 1. Carregar mastery e histórico
    mastery = db.query(SkillMastery).filter(
        SkillMastery.user_id == user_id,
        SkillMastery.skill_id == skill_id
    ).first()
    
    if not mastery:
        raise HTTPException(status_code=404, detail="Skill not found for user")
    
    # 2. Obter parâmetros AKT
    from .skill_registry import get_akt_params, get_skill_difficulty
    akt_params = get_akt_params(skill_id)
    tracer = AttentiveKnowledgeTracer(akt_params)
    
    # 3. Restaurar estado do tracer
    tracer.mastery_probability = mastery.mastery_probability
    # Carregar histórico recente...
    
    # 4. Obter dificuldade
    if difficulty is None:
        difficulty = get_skill_difficulty(skill_id)
    
    # 5. Calcular predição
    prediction = tracer.predict_performance(skill_id, difficulty)
    
    return {
        "skill_id": skill_id,
        "predicted_correct_probability": round(prediction, 4),
        "current_mastery": round(mastery.mastery_probability, 4),
        "difficulty": difficulty,
        "interpretation": _interpret_prediction(prediction),
        "recommendation": _get_recommendation(prediction)
    }

def _interpret_prediction(prediction: float) -> str:
    if prediction < 0.3:
        return "muito_dificil"
    elif prediction < 0.5:
        return "dificil"
    elif prediction < 0.7:
        return "adequado"
    elif prediction < 0.8:
        return "facil"
    else:
        return "muito_facil"

def _get_recommendation(prediction: float) -> str:
    if prediction < 0.3:
        return "Oferecer scaffolding ou revisar conceitos básicos"
    elif prediction < 0.7:
        return "Zona ideal para prática - continuar"
    else:
        return "Considerar aumentar dificuldade ou mudar de skill"
```

#### API

**Endpoint:**
```
GET /api/student/{user_id}/skill/{skill_id}/prediction?difficulty=0.5
```

**Response:**
```json
{
  "skill_id": "verb_conjugation_past",
  "predicted_correct_probability": 0.6325,
  "current_mastery": 0.7845,
  "difficulty": 0.5,
  "interpretation": "adequado",
  "recommendation": "Zona ideal para prática - continuar"
}
```

#### Integração com Pedagogical Policy

**Arquivo:** `src/services/pedagogical_policy/strategy_engine.py`

```python
async def select_next_skill(user_id: str, available_skills: List[str]) -> str:
    """
    Seleciona próxima skill baseada em predições do AKT
    """
    predictions = []
    
    for skill_id in available_skills:
        # Consultar Student Model
        prediction = await student_model_client.predict_performance(user_id, skill_id)
        predictions.append({
            "skill_id": skill_id,
            "prediction": prediction["predicted_correct_probability"],
            "distance_from_optimal": abs(prediction["predicted_correct_probability"] - 0.5)
        })
    
    # Ordenar por proximidade de 0.5 (zona ótima de aprendizado)
    predictions.sort(key=lambda x: x["distance_from_optimal"])
    
    return predictions[0]["skill_id"]
```

#### Uso Didático

- [ ] Learning Path usa para selecionar próxima skill
- [ ] Pedagogical Policy ajusta scaffolding baseado na predição
- [ ] UI mostra "nível de desafio" para o aluno

### 📅 Timeline

**Semana 1:**
- [ ] Implementar endpoint no Student Model
- [ ] Testes unitários

**Semana 2:**
- [ ] Integrar com Pedagogical Policy
- [ ] Integrar com Learning Path
- [ ] Testes E2E

---

## 3️⃣ Curva de Aprendizado por Skill

### 📝 Descrição

Visualizar a **evolução temporal** do mastery de uma skill, mostrando:
- Pontos de prática (timestamp, correct, mastery após interação)
- Tendência (melhorando, estável, regredindo)
- Marcos importantes (primeira vez acertou, alcançou mastery, etc.)

### ❌ Status Atual

**❌ NÃO IMPLEMENTADO**

### 🔧 Implementação

#### Backend

**Arquivo:** `src/services/student_model/app_complete.py`

```python
@app.get("/api/student/{user_id}/skill/{skill_id}/learning_curve")
async def get_learning_curve(
    user_id: str,
    skill_id: str,
    days_back: int = 90,
    db: Session = Depends(get_db)
):
    """
    Retorna a curva de aprendizado de uma skill ao longo do tempo
    """
    from datetime import datetime, timedelta, timezone
    
    # 1. Buscar interações
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    interactions = db.query(InteractionHistory).filter(
        InteractionHistory.user_id == user_id,
        InteractionHistory.skill_id == skill_id,
        InteractionHistory.timestamp >= cutoff_date
    ).order_by(InteractionHistory.timestamp.asc()).all()
    
    if not interactions:
        raise HTTPException(status_code=404, detail="No interaction history found")
    
    # 2. Construir curva
    curve_points = []
    for interaction in interactions:
        # Obter mastery após essa interação (pode estar no context ou calcular)
        mastery_after = _extract_mastery_from_interaction(interaction, db)
        
        curve_points.append({
            "timestamp": interaction.timestamp.isoformat(),
            "correct": interaction.correct,
            "mastery_after": round(mastery_after, 4),
            "iteration": len(curve_points) + 1
        })
    
    # 3. Calcular estatísticas e tendência
    initial_mastery = curve_points[0]["mastery_after"]
    final_mastery = curve_points[-1]["mastery_after"]
    mastery_gain = final_mastery - initial_mastery
    
    # Calcular tendência (comparar primeira metade vs segunda metade)
    mid_point = len(curve_points) // 2
    first_half_avg = sum(p["mastery_after"] for p in curve_points[:mid_point]) / mid_point
    second_half_avg = sum(p["mastery_after"] for p in curve_points[mid_point:]) / (len(curve_points) - mid_point)
    trend = second_half_avg - first_half_avg
    
    # Identificar marcos
    milestones = []
    for i, point in enumerate(curve_points):
        # Primeira vez acertou
        if point["correct"] and (i == 0 or not curve_points[i-1]["correct"]):
            milestones.append({
                "type": "first_correct" if i < 5 else "correct_after_errors",
                "timestamp": point["timestamp"],
                "iteration": i + 1
            })
        
        # Alcançou mastery
        if point["mastery_after"] >= 0.7 and (i == 0 or curve_points[i-1]["mastery_after"] < 0.7):
            milestones.append({
                "type": "reached_mastery",
                "timestamp": point["timestamp"],
                "iteration": i + 1,
                "mastery": point["mastery_after"]
            })
    
    return {
        "skill_id": skill_id,
        "user_id": user_id,
        "total_interactions": len(curve_points),
        "date_range": {
            "start": curve_points[0]["timestamp"],
            "end": curve_points[-1]["timestamp"]
        },
        "statistics": {
            "initial_mastery": round(initial_mastery, 4),
            "final_mastery": round(final_mastery, 4),
            "mastery_gain": round(mastery_gain, 4),
            "trend": round(trend, 4),
            "trend_interpretation": _interpret_trend(trend)
        },
        "milestones": milestones,
        "curve": curve_points
    }

def _interpret_trend(trend: float) -> str:
    if trend > 0.1:
        return "melhorando_significativamente"
    elif trend > 0.05:
        return "melhorando"
    elif trend > -0.05:
        return "estavel"
    elif trend > -0.1:
        return "regredindo"
    else:
        return "regredindo_significativamente"
```

#### API

**Endpoint:**
```
GET /api/student/{user_id}/skill/{skill_id}/learning_curve?days_back=30
```

**Response:**
```json
{
  "skill_id": "verb_conjugation_past",
  "user_id": "user_123",
  "total_interactions": 25,
  "date_range": {
    "start": "2025-10-22T10:00:00Z",
    "end": "2025-11-22T15:30:00Z"
  },
  "statistics": {
    "initial_mastery": 0.2500,
    "final_mastery": 0.7845,
    "mastery_gain": 0.5345,
    "trend": 0.1234,
    "trend_interpretation": "melhorando_significativamente"
  },
  "milestones": [
    {
      "type": "first_correct",
      "timestamp": "2025-10-23T11:00:00Z",
      "iteration": 3
    },
    {
      "type": "reached_mastery",
      "timestamp": "2025-11-15T14:20:00Z",
      "iteration": 18,
      "mastery": 0.7123
    }
  ],
  "curve": [
    {
      "timestamp": "2025-10-22T10:00:00Z",
      "correct": false,
      "mastery_after": 0.2500,
      "iteration": 1
    },
    ...
  ]
}
```

#### Uso Didático / UX

- [ ] Dashboard do aluno: gráfico de progresso por skill
- [ ] Dashboard do professor: visualização de múltiplos alunos
- [ ] Pedagogical Policy: detectar estagnação e mudar estratégia

### 📅 Timeline

**Semana 1-2:**
- [ ] Implementar backend e endpoint
- [ ] Testes

**Semana 3:**
- [ ] Integração com UI (gráfico)

---

## 4️⃣ Skills Fortes e Fracas

### 📝 Descrição

Classificar skills do aluno em **fortes** (bem dominadas) e **fracas** (necessitam mais prática).

### ✅ Status Atual

**✅ IMPLEMENTADO**

- Disponível em `/api/student/{user_id}/interpretable_knowledge_state`
- Retorna `strong_skills` e `weak_skills`

### 🔧 Melhorias Propostas

#### Backend

- [ ] Definir thresholds configuráveis:
  ```python
  STRONG_SKILL_THRESHOLD = 0.8
  WEAK_SKILL_THRESHOLD = 0.4
  ```

- [ ] Adicionar categoria intermediária "learning" (0.4–0.8)

- [ ] Endpoint dedicado (opcional):
  ```
  GET /api/student/{user_id}/skills/classification
  ```

#### Uso Didático

- [ ] Prompt pedagógico menciona skills fortes para motivação
- [ ] Foca em skills fracas para prática direcionada
- [ ] Conecta skills fortes com fracas (scaffolding)

### 📅 Timeline

**Semana 1:** Refinamentos e testes

---

## 5️⃣ Pontos Fracos por Feature Linguística

### 📝 Descrição

Identificar quais **features linguísticas específicas** causam mais erros:
- `tense: past` vs `present` vs `future`
- `person: 1st` vs `2nd` vs `3rd`
- `number: singular` vs `plural`
- `register: formal` vs `informal`
- `domain: travel` vs `family` vs `work`

### ✅ Status Atual

**✅ IMPLEMENTADO**

- Endpoint: `/api/student/{user_id}/linguistic_error_patterns`
- Retorna features problemáticas e dominadas

### 🔧 Melhorias Propostas

#### Backend

- [ ] Adicionar filtragem por tipo de feature:
  ```
  GET /api/student/{user_id}/linguistic_error_patterns?feature_type=tense
  ```

- [ ] Adicionar comparação temporal:
  ```json
  {
    "feature_key": "tense:past",
    "error_rate_30_days_ago": 0.75,
    "error_rate_now": 0.35,
    "improvement": 0.40
  }
  ```

#### Uso Didático

- [ ] Pedagogical Policy gera instruções específicas:
  - "Foque no passado na 3ª pessoa do singular"
  - "Pratique vocabulário informal em contextos de viagem"

- [ ] Learning Path prioriza exercícios com essas features

### 📅 Timeline

**Semana 1-2:** Melhorias incrementais

---

## 6️⃣ Impacto da Dificuldade (IRT) por Skill

### 📝 Descrição

Analisar como o aluno performa em **diferentes níveis de dificuldade** da mesma skill:
- Fácil (difficulty < 0.3): 90% de acerto
- Médio (0.3 ≤ difficulty ≤ 0.7): 60% de acerto
- Difícil (difficulty > 0.7): 30% de acerto

### ❌ Status Atual

**❌ NÃO IMPLEMENTADO**

### 🔧 Implementação

#### Backend

**Arquivo:** `src/services/student_model/app_complete.py`

```python
@app.get("/api/student/{user_id}/skill/{skill_id}/difficulty_profile")
async def get_difficulty_profile(
    user_id: str,
    skill_id: str,
    db: Session = Depends(get_db)
):
    """
    Analisa performance do aluno em diferentes níveis de dificuldade
    """
    # 1. Buscar interações com difficulty
    interactions = db.query(InteractionHistory).filter(
        InteractionHistory.user_id == user_id,
        InteractionHistory.skill_id == skill_id
    ).all()
    
    # 2. Agrupar por faixas de dificuldade
    buckets = {
        "facil": {"range": (0.0, 0.3), "correct": 0, "total": 0},
        "medio": {"range": (0.3, 0.7), "correct": 0, "total": 0},
        "dificil": {"range": (0.7, 1.0), "correct": 0, "total": 0}
    }
    
    for interaction in interactions:
        difficulty = _extract_difficulty(interaction)
        if difficulty is None:
            continue
        
        # Classificar em bucket
        if difficulty < 0.3:
            bucket = "facil"
        elif difficulty < 0.7:
            bucket = "medio"
        else:
            bucket = "dificil"
        
        buckets[bucket]["total"] += 1
        if interaction.correct:
            buckets[bucket]["correct"] += 1
    
    # 3. Calcular accuracy por bucket
    profile = {}
    for bucket_name, bucket_data in buckets.items():
        if bucket_data["total"] > 0:
            accuracy = bucket_data["correct"] / bucket_data["total"]
        else:
            accuracy = None
        
        profile[bucket_name] = {
            "difficulty_range": bucket_data["range"],
            "total_attempts": bucket_data["total"],
            "correct_attempts": bucket_data["correct"],
            "accuracy": round(accuracy, 4) if accuracy is not None else None
        }
    
    # 4. Análise e recomendação
    analysis = _analyze_difficulty_profile(profile)
    
    return {
        "skill_id": skill_id,
        "user_id": user_id,
        "profile": profile,
        "analysis": analysis
    }

def _analyze_difficulty_profile(profile: Dict) -> Dict:
    facil_acc = profile.get("facil", {}).get("accuracy")
    medio_acc = profile.get("medio", {}).get("accuracy")
    dificil_acc = profile.get("dificil", {}).get("accuracy")
    
    # Detectar padrões
    patterns = []
    if facil_acc and facil_acc > 0.8 and dificil_acc and dificil_acc < 0.3:
        patterns.append("trava_em_dificil")
        recommendation = "Introduzir exercícios de dificuldade média para construir ponte"
    elif all([facil_acc, medio_acc, dificil_acc]) and min(facil_acc, medio_acc, dificil_acc) > 0.7:
        patterns.append("consistente_em_todas_dificuldades")
        recommendation = "Skill bem dominada - considerar aumentar complexidade geral"
    else:
        recommendation = "Continuar praticando em dificuldade média"
    
    return {
        "patterns": patterns,
        "recommendation": recommendation
    }
```

#### Uso Didático

- [ ] Learning Path cria "escada" de dificuldade progressiva
- [ ] Pedagogical Policy ajusta velocidade de progressão

### 📅 Timeline

**Semana 2-3:** Implementação completa

---

## 7️⃣ Esquecimento e Necessidade de Revisão (FoLiBi)

### 📝 Descrição

Detectar skills onde o mastery **caiu significativamente** devido a falta de prática, e recomendar revisões.

### ⚠️ Status Atual

**⚠️ PARCIALMENTE IMPLEMENTADO**

- FoLiBi está implementado no AKT (`_compute_folibi_bias`)
- Não há endpoint específico para detectar esquecimento

### 🔧 Implementação

#### Backend

```python
@app.get("/api/student/{user_id}/forgetting_risks")
async def detect_forgetting_risks(
    user_id: str,
    threshold_drop: float = 0.15,  # Drop de 15% ou mais
    days_inactive: int = 7,  # Skills sem prática por 7+ dias
    db: Session = Depends(get_db)
):
    """
    Detecta skills com alto risco de esquecimento
    """
    from datetime import datetime, timedelta, timezone
    
    # 1. Buscar todas as skills do aluno
    masteries = db.query(SkillMastery).filter(
        SkillMastery.user_id == user_id
    ).all()
    
    at_risk = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_inactive)
    
    for mastery in masteries:
        # 2. Verificar última interação
        last_interaction = db.query(InteractionHistory).filter(
            InteractionHistory.user_id == user_id,
            InteractionHistory.skill_id == mastery.skill_id
        ).order_by(InteractionHistory.timestamp.desc()).first()
        
        if not last_interaction:
            continue
        
        # 3. Se inativo por muito tempo
        if last_interaction.timestamp < cutoff_date:
            days_since_practice = (datetime.now(timezone.utc) - last_interaction.timestamp).days
            
            # 4. Estimar queda de mastery por esquecimento (FoLiBi)
            estimated_drop = _estimate_forgetting_drop(
                current_mastery=mastery.mastery_probability,
                days_inactive=days_since_practice
            )
            
            if estimated_drop >= threshold_drop:
                at_risk.append({
                    "skill_id": mastery.skill_id,
                    "current_mastery": round(mastery.mastery_probability, 4),
                    "estimated_drop": round(estimated_drop, 4),
                    "days_since_practice": days_since_practice,
                    "last_practice_date": last_interaction.timestamp.isoformat(),
                    "priority": "alta" if estimated_drop > 0.25 else "media"
                })
    
    # 5. Ordenar por prioridade (maior drop primeiro)
    at_risk.sort(key=lambda x: x["estimated_drop"], reverse=True)
    
    return {
        "user_id": user_id,
        "skills_at_risk": at_risk,
        "total_at_risk": len(at_risk),
        "recommendation": "Revisar skills de alta prioridade primeiro"
    }

def _estimate_forgetting_drop(current_mastery: float, days_inactive: int) -> float:
    """
    Estima queda de mastery baseado em FoLiBi
    """
    # Parâmetro de decay do FoLiBi
    linear_decay_factor = 0.3
    
    # Decay por dia (simplificado)
    daily_decay = 0.01  # 1% por dia
    
    # Decay total
    total_decay = min(current_mastery * 0.5, daily_decay * days_inactive)
    
    return total_decay
```

#### Uso Didático

- [ ] Learning Path agenda revisões automáticas (spaced repetition)
- [ ] Notificações para o aluno: "Revisar verbos no passado"
- [ ] Dashboard do professor: alunos com mais skills em risco

### 📅 Timeline

**Semana 2-4:** Implementação e integração com spaced repetition

---

## 8️⃣ Resumo por Dimensão / CEFR

### 📝 Descrição

Agregar masteries por:
- **Dimensões:** grammar, vocabulary, pronunciation
- **Níveis CEFR:** A1, A2, B1, B2, C1, C2

### ✅ Status Atual

**✅ IMPLEMENTADO**

- Endpoint: `/api/student/{user_id}/cefr_progress`
- Retorna `dimension_progress` e `cefr_details`

### 🔧 Melhorias Propostas

#### Backend

- [ ] Adicionar histórico de evolução CEFR:
  ```
  GET /api/student/{user_id}/cefr_progress/history?days_back=90
  ```

- [ ] Comparação com média de outros alunos no mesmo nível

#### Uso Didático

- [ ] UI: barras de progresso por dimensão
- [ ] Certificados quando alcança novo nível CEFR
- [ ] Pedagogical Policy ajusta complexidade baseado no nível

### 📅 Timeline

**Semana 1:** Melhorias incrementais

---

## 9️⃣ Taxas de Acerto (Accuracy) por Skill

### 📝 Descrição

Calcular **taxa de acerto** (accuracy) por skill e global:
- Total de acertos / total de tentativas
- Correlacionar com mastery probability

### ⚠️ Status Atual

**⚠️ PARCIALMENTE IMPLEMENTADO**

- Dados existem em `InteractionHistory`
- Não há endpoint dedicado

### 🔧 Implementação

#### Backend

```python
@app.get("/api/student/{user_id}/accuracy_stats")
async def get_accuracy_stats(
    user_id: str,
    days_back: int = 30,
    db: Session = Depends(get_db)
):
    """
    Calcula estatísticas de acerto do aluno
    """
    from datetime import datetime, timedelta, timezone
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    # 1. Buscar todas as interações
    interactions = db.query(InteractionHistory).filter(
        InteractionHistory.user_id == user_id,
        InteractionHistory.timestamp >= cutoff_date
    ).all()
    
    if not interactions:
        raise HTTPException(status_code=404, detail="No interactions found")
    
    # 2. Calcular global
    total_interactions = len(interactions)
    total_correct = sum(1 for i in interactions if i.correct)
    global_accuracy = total_correct / total_interactions
    
    # 3. Calcular por skill
    skill_stats = {}
    for interaction in interactions:
        skill_id = interaction.skill_id
        if skill_id not in skill_stats:
            skill_stats[skill_id] = {"correct": 0, "total": 0}
        
        skill_stats[skill_id]["total"] += 1
        if interaction.correct:
            skill_stats[skill_id]["correct"] += 1
    
    # Calcular accuracy por skill
    skill_accuracies = {}
    for skill_id, stats in skill_stats.items():
        accuracy = stats["correct"] / stats["total"]
        skill_accuracies[skill_id] = {
            "accuracy": round(accuracy, 4),
            "total_attempts": stats["total"],
            "correct_attempts": stats["correct"]
        }
    
    # 4. Top e bottom skills por accuracy
    sorted_skills = sorted(skill_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    top_skills = sorted_skills[:5]
    bottom_skills = sorted_skills[-5:]
    
    return {
        "user_id": user_id,
        "period_days": days_back,
        "global_accuracy": round(global_accuracy, 4),
        "total_interactions": total_interactions,
        "skill_accuracies": skill_accuracies,
        "top_performing_skills": [{"skill_id": s[0], **s[1]} for s in top_skills],
        "lowest_performing_skills": [{"skill_id": s[0], **s[1]} for s in bottom_skills]
    }
```

#### Uso Didático

- [ ] Mensagens motivacionais: "Sua taxa de acerto subiu 15%!"
- [ ] Comparação com mastery: "Mesmo com 80% de acerto, seu mastery está em 60% - precisamos de mais prática para consolidar"

### 📅 Timeline

**Semana 1:** Implementação rápida

---

## 📅 Cronograma Geral

### Fase 1: Melhorias Rápidas (Semana 1-2)

✅ **Prioridade Alta - Já Implementados:**
- [x] Probabilidade de domínio por skill
- [x] Skills fortes e fracas
- [x] Pontos fracos por feature linguística
- [x] Resumo por dimensão/CEFR

⚡ **Prioridade Alta - Implementação Rápida:**
- [ ] Probabilidade de acerto futuro (endpoint + integração)
- [ ] Taxas de acerto por skill

### Fase 2: Funcionalidades Médias (Semana 3-4)

- [ ] Curva de aprendizado por skill
- [ ] Esquecimento e revisão (FoLiBi completo)

### Fase 3: Funcionalidades Avançadas (Semana 5-6)

- [ ] Impacto da dificuldade (IRT profile)
- [ ] Histórico de evolução CEFR
- [ ] Comparações com outros alunos

---

## 🎯 Próximos Passos

### Imediato (esta semana)

1. **Implementar endpoint de predição** (`predict_performance`)
2. **Implementar endpoint de accuracy stats**
3. **Integrar predição com Pedagogical Policy**

### Curto prazo (próximas 2 semanas)

4. **Implementar curva de aprendizado**
5. **Completar sistema de detecção de esquecimento**
6. **Criar dashboards de visualização**

### Médio prazo (próximo mês)

7. **Implementar análise de dificuldade (IRT)**
8. **Sistema de spaced repetition inteligente**
9. **Comparações e benchmarks**

---

## 📊 Métricas de Sucesso

- [ ] **Cobertura:** 100% dos endpoints planejados implementados
- [ ] **Integração:** Todos os serviços (Pedagogical Policy, Learning Path) usando informações do AKT
- [ ] **Performance:** Endpoints respondem em < 200ms
- [ ] **Testes:** Cobertura de testes > 80%
- [ ] **Documentação:** Todos os endpoints documentados com exemplos

---

## 🔗 Referências

- [AKT Implementation](./AKT_IMPLEMENTATION.md)
- [AKT Prediction Explained](./AKT_PREDICTION_EXPLAINED.md)
- [Integration Flow](./INTEGRATION_FLOW.md)
- [Linguistic Features Analysis](./LINGUISTIC_FEATURES_ANALYSIS.md)

---

**Última atualização:** 2025-11-22  
**Próxima revisão:** Semanalmente durante implementação

