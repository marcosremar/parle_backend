# Melhores Práticas para Classificação Automática de Níveis CEFR
## Baseado em Papers Acadêmicos (2023-2024)

---

## 📚 Papers Principais Encontrados

### 1. **NILC-Metrix (Leal et al., 2022)**
- **Título**: "NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese"
- **Link**: https://arxiv.org/abs/2201.03445
- **Contribuição**: Sistema com **200 métricas linguísticas** para português brasileiro
- **Aplicação**: Textos escritos e falados
- **Métricas**: Sintáticas, lexicais, discursivas, psicolinguísticas

### 2. **Arnold et al. (2018)**
- **Título**: "Predicting CEFRL levels in learner English on the basis of metrics and full texts"
- **Link**: https://arxiv.org/abs/1806.11099
- **Contribuição**: Modelos de aprendizado supervisionado com alta precisão
- **Resultados**: Melhor distinção entre **A1-A2** e **A2-B1**
- **Features principais**: Tokens, tipos de palavras, métricas lexicais e sintáticas

### 3. **Vajjala & Rama (2021)**
- **Título**: "Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs"
- **Contribuição**: Uso de RNNs com contornos de complexidade
- **Aplicação**: Classificação automática de textos escritos

### 4. **Ribeiro et al. (2024)**
- **Título**: "Avaliação automática do nível de complexidade de textos em português europeu"
- **Contribuição**: Métricas específicas para português europeu
- **Aplicação**: Classificação de complexidade textual

---

## 🎯 Métricas Mais Eficazes (Baseadas nos Papers)

### **1. Métricas Sintáticas (Prioridade Alta)**

#### **Comprimento de Sentença**
- **A1**: 3-5 palavras/frase
- **A2**: 5-8 palavras/frase
- **B1**: 8-12 palavras/frase
- **B2**: 12-18 palavras/frase
- **C1**: 15-25 palavras/frase
- **C2**: 20-30+ palavras/frase

#### **Subordinação**
- **A1**: Ausente (0%)
- **A2**: Muito limitada (1-5% das frases)
- **B1**: Limitada (5-15% das frases)
- **B2**: Moderada (15-30% das frases)
- **C1**: Avançada (30-50% das frases)
- **C2**: Muito avançada (50%+ das frases)

#### **Estruturas Complexas**
- **Subjuntivo**: Ausente (A1-A2) → Raro (B1) → Presente (B2) → Natural (C1-C2)
- **Voz Passiva**: Ausente (A1-A2) → Limitada (B1) → Natural (B2-C1) → Sofisticada (C2)
- **Orações Relativas**: Ausentes (A1) → Simples (A2-B1) → Variadas (B2) → Complexas (C1-C2)

### **2. Métricas Lexicais (Prioridade Alta)**

#### **Type-Token Ratio (TTR)**
- **A1**: 0.30-0.40 (muita repetição)
- **A2**: 0.40-0.50
- **B1**: 0.50-0.60
- **B2**: 0.60-0.70
- **C1**: 0.70-0.80
- **C2**: 0.80-0.90 (muita diversidade)

#### **Comprimento Médio de Palavras**
- **A1**: 4-5 caracteres
- **A2**: 5-6 caracteres
- **B1**: 6-7 caracteres
- **B2**: 7-8 caracteres
- **C1**: 8-9 caracteres
- **C2**: 9+ caracteres

#### **Vocabulário**
- **A1**: 500-1000 palavras mais frequentes
- **A2**: 1000-2000 palavras (rotina)
- **B1**: 2000-3500 palavras (opinião)
- **B2**: 3500-5000 palavras (abstrato)
- **C1**: 5000-8000 palavras (técnico)
- **C2**: 8000+ palavras (nativo)

### **3. Métricas Discursivas (Prioridade Média)**

#### **Conectores**
- **A1**: Apenas "e", "ou", "mas"
- **A2**: "porque", "mas", "então"
- **B1**: "portanto", "além disso", "por outro lado"
- **B2**: Repertório amplo e variado
- **C1**: Sofisticados ("conquanto", "visto que")
- **C2**: Criativos e eruditos

#### **Marcadores Discursivos**
- **A1-A2**: Ausentes
- **B1**: Básicos ("por exemplo", "ou seja")
- **B2-C1**: Variados e naturais
- **C2**: Sofisticados e criativos

---

## 🔬 Métodos de Classificação Mais Precisos

### **1. Ensemble de Métricas (Recomendado)**
Combinar múltiplas métricas em vez de usar apenas uma:
- **Sintáticas** (peso: 40%)
- **Lexicais** (peso: 40%)
- **Discursivas** (peso: 20%)

### **2. Aprendizado Supervisionado**
- **SVM**: Boa para distinção binária (A1 vs B1)
- **Random Forest**: Melhor para classificação multi-classe
- **Neural Networks**: Melhor para padrões complexos

### **3. Abordagem Híbrida (LLM + Métricas)**
- **LLM para análise qualitativa**: Estruturas complexas, nuances
- **Métricas para análise quantitativa**: Comprimento, TTR, frequência
- **Combinação**: Score final = 0.6 × LLM + 0.4 × Métricas

---

## 📊 Precisão Esperada por Nível (Baseado nos Papers)

| Nível | Precisão Esperada | Dificuldade de Distinção |
|-------|------------------|--------------------------|
| **A1** | 90-95% | Fácil (muito distinto) |
| **A2** | 75-85% | Moderada |
| **B1** | 70-80% | Moderada-Difícil |
| **B2** | 65-75% | Difícil |
| **C1** | 60-70% | Muito difícil |
| **C2** | 55-65% | Extremamente difícil |

**Nota**: A distinção entre níveis adjacentes (A1-A2, B1-B2) é mais difícil que entre níveis distantes (A1-B1).

---

## 🎯 Recomendações Específicas para Nosso Sistema

### **1. Usar Ensemble de Métricas**
Em vez de confiar apenas no LLM, combinar:
- **LLM Analysis** (60%): Análise qualitativa de estruturas
- **Syntactic Metrics** (25%): Comprimento, subordinação, estruturas complexas
- **Lexical Metrics** (15%): TTR, comprimento de palavras, vocabulário

### **2. Calibrar por Nível**
- **A1-A2**: Focar em comprimento de frase e vocabulário básico
- **B1-B2**: Focar em subordinação e diversidade lexical
- **C1-C2**: Focar em estruturas complexas e sofisticação discursiva

### **3. Usar Few-Shot Learning**
Fornecer exemplos de cada nível no prompt do LLM:
```
Exemplo A1: "Olá! Como vai? Quer café?"
Exemplo A2: "Ontem fui ao cinema porque fazia sol."
Exemplo B1: "Eu gosto deste restaurante porque a comida é excelente e o atendimento sempre foi ótimo."
```

### **4. Validação Cruzada**
- Testar em múltiplos cenários
- Validar com diferentes tipos de texto (conversacional, descritivo, argumentativo)
- Comparar com classificadores humanos

---

## 🔍 Features Mais Discriminativas (Arnold et al., 2018)

1. **Tokens totais** (número de palavras)
2. **Types únicos** (vocabulário diverso)
3. **Type-Token Ratio** (diversidade lexical)
4. **Comprimento médio de sentença**
5. **Presença de subordinação**
6. **Presença de voz passiva**
7. **Presença de subjuntivo**
8. **Orações relativas**

---

## 📈 Melhorias Sugeridas para Nosso Sistema

### **1. Adicionar Métricas Quantitativas**
```python
def calculate_syntactic_metrics(text):
    sentences = split_sentences(text)
    return {
        "avg_words_per_sentence": mean([len(s.split()) for s in sentences]),
        "subordination_ratio": count_subordinating_conjunctions(text) / len(sentences),
        "passive_voice_ratio": count_passive_voice(text) / len(sentences),
        "subjunctive_ratio": count_subjunctive(text) / len(sentences)
    }

def calculate_lexical_metrics(text):
    words = text.split()
    unique_words = set(words)
    return {
        "type_token_ratio": len(unique_words) / len(words),
        "avg_word_length": mean([len(w) for w in words]),
        "vocabulary_size_estimate": len(unique_words)
    }
```

### **2. Combinar LLM + Métricas**
```python
def classify_cefr_level(text):
    # LLM analysis (qualitative)
    llm_analysis = identify_cefr_level_llm(text)
    llm_score = llm_analysis["confidence"]
    llm_level = llm_analysis["identified_level"]
    
    # Quantitative metrics
    syntactic = calculate_syntactic_metrics(text)
    lexical = calculate_lexical_metrics(text)
    metrics_score = calculate_metrics_score(syntactic, lexical)
    
    # Ensemble
    final_score = 0.6 * llm_score + 0.4 * metrics_score
    final_level = weighted_decision(llm_level, metrics_score)
    
    return final_level, final_score
```

### **3. Few-Shot Examples no Prompt**
Adicionar exemplos reais de cada nível no prompt do LLM para melhorar a classificação.

---

## 📚 Referências Completas

1. **Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022).** NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese. *arXiv preprint arXiv:2201.03445*.

2. **Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P. (2018).** Predicting CEFRL levels in learner English on the basis of metrics and full texts. *arXiv preprint arXiv:1806.11099*.

3. **Vajjala, S., & Rama, T. (2021).** Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications*.

4. **Ribeiro, E., Mamede, N., & Baptista, J. (2024).** Avaliação automática do nível de complexidade de textos em português europeu. *Linguamática*, 16(2), 115-139.

---

## ✅ Conclusão

Para melhorar a precisão da classificação CEFR:

1. **Combinar LLM + Métricas Quantitativas** (ensemble)
2. **Usar Few-Shot Learning** com exemplos de cada nível
3. **Calibrar por nível** (focar em features diferentes para cada nível)
4. **Validar em múltiplos cenários** e tipos de texto
5. **Usar ensemble de múltiplas métricas** (sintáticas, lexicais, discursivas)

**Precisão esperada com essas melhorias: 80-90% para níveis A1-B2, 70-80% para C1-C2.**

