# Como o AKT Prediz Acertos/Erros

## 📊 Visão Geral

O AKT (Attentive Knowledge Tracing) é um modelo que **prediz a probabilidade de um aluno acertar uma questão** baseado no seu histórico de interações. Esta predição é usada para:

1. **Adaptar dificuldade** dos exercícios
2. **Recomendar próximas skills** para praticar
3. **Identificar pontos fracos** do estudante
4. **Otimizar o caminho de aprendizado**

---

## 🔍 Como Funciona a Predição

### Fórmula Base

O AKT usa uma fórmula probabilística para predizer acertos:

```
P(acertar) = P(acertar|sabe) × P(sabe) + P(acertar|não sabe) × P(não sabe)
```

Onde:
- **P(sabe)**: Probabilidade atual de domínio (mastery probability) - **0.0 a 1.0**
- **P(acertar|sabe)**: Probabilidade de acertar quando sabe (p_G) - **~0.85**
- **P(acertar|não sabe)**: Probabilidade de acertar por "chute" (p_S) - **~0.3**

### Exemplo Prático

Se um aluno tem:
- **Mastery = 0.6** (60% de chance de saber)
- **p_G = 0.85** (85% de chance de acertar quando sabe)
- **p_S = 0.3** (30% de chance de acertar por chute quando não sabe)

Então:
```
P(acertar) = 0.85 × 0.6 + 0.3 × 0.4
           = 0.51 + 0.12
           = 0.63 (63% de chance de acertar)
```

---

## 🧠 Componentes da Predição no AKT

### 1. **Mastery Probability (P(sabe))**

É a **probabilidade atual de domínio** da skill, calculada a partir de:
- Histórico de interações (últimas 30 interações)
- Taxa de acerto recente
- Padrões de erro (ex: sempre erra na 3ª pessoa)
- Tempo desde última prática (esquecimento)

**Como evolui:**
- ✅ **Aumenta** quando o aluno acerta (especialmente em sequência)
- ❌ **Diminui** quando o aluno erra (mas não muito, para não ser punitivo)
- ⏰ **Decai** com o tempo (esquecimento - FoLiBi)

### 2. **Ajuste por Dificuldade (IRT/Rasch)**

O AKT ajusta a predição baseado na **dificuldade do skill**:

```python
# Skills mais difíceis reduzem p_G e p_S
adjusted_p_G = p_G × (1.0 - difficulty × 0.3)
adjusted_p_S = p_S × (1.0 - difficulty × 0.2)
```

**Exemplo:**
- Skill fácil (difficulty = 0.2): p_G ajustado = 0.85 × 0.94 = **0.80**
- Skill difícil (difficulty = 0.8): p_G ajustado = 0.85 × 0.76 = **0.65**

### 3. **Attention sobre Histórico**

O AKT não olha apenas a última interação. Ele usa **attention** para considerar:
- **Últimas 30 interações** (attention_window)
- **Peso maior** para interações recentes
- **Padrões temporais** (ex: melhorou nas últimas 5 tentativas?)

**Como funciona:**
```python
# Peso decai com o tempo
weight = temporal_decay ^ (tempo_desde_interação)
```

Interações mais recentes têm **mais influência** na predição.

### 4. **Adaptação por Features Linguísticas**

O AKT adapta a predição baseado em **padrões de erro por feature**:

**Exemplo:**
- Aluno sempre erra quando usa **3ª pessoa do singular**
- Aluno sempre acerta quando usa **1ª pessoa do plural**

O modelo ajusta `p_T` (probabilidade de aprender) para essas features específicas.

---

## 📈 Exemplo Completo de Predição

### Cenário: Aluno praticando `verb_conjugation_past`

**Histórico:**
- Interação 1: ❌ Errou (mastery: 0.2 → 0.15)
- Interação 2: ❌ Errou (mastery: 0.15 → 0.12)
- Interação 3: ✅ Acertou (mastery: 0.12 → 0.25)
- Interação 4: ✅ Acertou (mastery: 0.25 → 0.40)
- Interação 5: ✅ Acertou (mastery: 0.40 → 0.55)

**Estado atual:**
- Mastery = **0.55** (55% de chance de saber)
- Dificuldade do skill = **0.5** (média)
- Última interação foi há **2 dias**

**Cálculo da predição:**

1. **Ajuste por dificuldade:**
   ```
   adjusted_p_G = 0.85 × (1.0 - 0.5 × 0.3) = 0.85 × 0.85 = 0.72
   adjusted_p_S = 0.3 × (1.0 - 0.5 × 0.2) = 0.3 × 0.9 = 0.27
   ```

2. **Ajuste por esquecimento (FoLiBi):**
   ```
   # Decay de 2 dias
   mastery_with_forgetting = 0.55 × (0.95 ^ 2) = 0.55 × 0.90 = 0.50
   ```

3. **Predição final:**
   ```
   P(acertar) = 0.72 × 0.50 + 0.27 × 0.50
              = 0.36 + 0.135
              = 0.495 ≈ 50%
   ```

**Interpretação:** O aluno tem **50% de chance de acertar** a próxima questão de `verb_conjugation_past`.

---

## 🎯 Como Usar a Predição no Sistema

### 1. **Adaptação de Dificuldade**

```python
prediction = akt.predict_performance(skill_id="verb_conjugation_past")

if prediction > 0.8:
    # Muito fácil - aumentar dificuldade ou mudar de skill
    next_skill = get_harder_skill()
elif prediction < 0.3:
    # Muito difícil - oferecer scaffolding ou skill mais fácil
    next_skill = get_easier_skill()
else:
    # Zona ideal de aprendizado (30-80%)
    next_skill = current_skill  # Continuar praticando
```

### 2. **Recomendação de Skills**

```python
# Priorizar skills com predição na zona ideal
recommended_skills = []
for skill_id in available_skills:
    prediction = akt.predict_performance(skill_id)
    if 0.3 <= prediction <= 0.8:
        recommended_skills.append((skill_id, prediction))

# Ordenar por proximidade de 0.5 (zona ótima)
recommended_skills.sort(key=lambda x: abs(x[1] - 0.5))
```

### 3. **Identificação de Pontos Fracos**

```python
# Skills com predição baixa são pontos fracos
weak_skills = []
for skill_id in all_skills:
    prediction = akt.predict_performance(skill_id)
    if prediction < 0.3:
        weak_skills.append({
            "skill_id": skill_id,
            "prediction": prediction,
            "recommendation": "Necessita mais prática"
        })
```

---

## 🔄 Atualização da Predição

A predição **muda a cada interação**:

1. **Aluno acerta:**
   - Mastery aumenta
   - Próxima predição será **maior**

2. **Aluno erra:**
   - Mastery diminui (mas não muito)
   - Próxima predição será **menor**

3. **Tempo passa:**
   - Mastery decai (esquecimento)
   - Próxima predição será **menor**

4. **Padrões identificados:**
   - Se sempre erra em uma feature específica, predição ajusta para essa feature

---

## 📊 Interpretação dos Valores

| Predição | Interpretação | Ação Recomendada |
|----------|---------------|------------------|
| **0.0 - 0.3** | Muito difícil | Oferecer scaffolding, skill mais fácil, ou revisar conceitos básicos |
| **0.3 - 0.5** | Difícil mas aprendível | Zona ideal para prática com suporte |
| **0.5 - 0.7** | Nível adequado | Zona ótima de aprendizado - continuar praticando |
| **0.7 - 0.8** | Fácil | Praticar para consolidar |
| **0.8 - 1.0** | Muito fácil | Mudar para skill mais difícil ou revisar ocasionalmente |

---

## 🧪 Testando a Predição

Você pode testar a predição usando o endpoint:

```python
# Obter predição para uma skill
GET /api/student/{user_id}/skills

# Ou diretamente no código:
from src.services.student_model.knowledge_tracer import AttentiveKnowledgeTracer

akt = AttentiveKnowledgeTracer()
prediction = akt.predict_performance(skill_id="verb_conjugation_past")
print(f"Probabilidade de acertar: {prediction:.2%}")
```

---

## 📚 Referências

- **AKT Paper**: "Attentive Knowledge Tracing" (2020)
- **FoLiBi**: "Forgetting-aware Linear Bias for Attentive Knowledge Tracing" (2023)
- **IRT/Rasch**: Item Response Theory para ajuste de dificuldade

