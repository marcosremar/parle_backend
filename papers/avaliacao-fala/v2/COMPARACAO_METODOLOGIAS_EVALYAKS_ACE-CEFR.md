# Comparação de Metodologias: EvalYaks vs Ace-CEFR

## 📊 Visão Geral

| Aspecto | EvalYaks (2024) | Ace-CEFR (2025) |
|---------|-----------------|-----------------|
| **Objetivo** | Avaliar transcrições de fala CEFR B2 | Avaliar dificuldade linguística de textos conversacionais |
| **Foco** | Avaliação de desempenho em exame | Classificação de nível CEFR |
| **Modelo Base** | Mistral Instruct 7B v0.2 | BERT, PaLM 2-L, Linear Regression |
| **Fine-tuning** | LoRA (Parameter Efficient) | Fine-tuning completo (BERT) |
| **Dataset** | 3,060 conversas sintéticas | 890 trechos conversacionais |
| **Precisão** | 96% (acceptable accuracy) | MSE 0.48 (supera humanos) |

---

## 🔬 METODOLOGIA DETALHADA

### 1. EvalYaks (2024)

#### **Abordagem Geral:**
Sistema de 6 modelos especializados para avaliação automática de exames CEFR B2 Speaking.

#### **Etapas da Metodologia:**

**Etapa 1: Geração de Dataset Sintético**
- **Ferramenta:** GPT-4 Turbo (Jan'24)
- **Processo:**
  - Geração de conversas simuladas para 4 partes do exame B2
  - Cada conversa inclui scores para 3 critérios:
    - Grammar and Vocabulary
    - Discourse Management
    - Interactive Communication
  - 25 perfis de usuários em 5 níveis de proficiência
  - 3,060 data points totais (671, 667, 927, 795 para partes 1-4)
- **Validação:** 3 especialistas com 4+ anos de experiência
- **Alinhamento:** Validação dupla por 2 de 3 especialistas

**Etapa 2: Baseline com LLMs Padrão**
- Teste de LLMs do LMSYS leaderboard
- Dois tipos de prompts: com e sem contexto
- Resultado: LLMs padrão não são satisfatórios

**Etapa 3: Criação de Instruction Datasets**
- **Recursos utilizados:**
  - Conversas validadas (3,060)
  - English Vocabulary Profile (até B2)
  - CEFR-SP WikiAuto dataset
- **Instruções criadas:**
  - 3 conjuntos de instruções variadas
  - Versões parafraseadas para aumentar dataset
  - Total: 7,345 instruction data points (partes 1-4)
  - 3,072 para vocabulário (English Vocabulary Profile)
  - 19,142 para sentenças (CEFR-SP WikiAuto)

**Etapa 4: Fine-tuning com LoRA**
- **Modelo Base:** Mistral Instruct 7B v0.2
- **Método:** LoRA (Low-Rank Adaptation)
- **Razão:** Parameter Efficient Fine-Tuning (PEFT)
- **6 Modelos Criados:**
  1. EvalYaks Part 1 (avaliação parte 1)
  2. EvalYaks Part 2 (avaliação parte 2)
  3. EvalYaks Part 3 (avaliação parte 3)
  4. EvalYaks Part 4 (avaliação parte 4)
  5. EvalYaks Vocab (identificar/gerar vocabulário B2)
  6. EvalYaks CEFR (identificar/gerar sentenças B2)

#### **Métricas de Avaliação:**
- **Acceptable Accuracy:** 96%
- **Degree of Variation:** 0.35 níveis
- **Performance:** 3x melhor que próximo melhor modelo
- **Classificação:**
  - Accurate: scores perfeitamente alinhados
  - Partly accurate: pelo menos 1 critério correto
  - Acceptable: desvio de 1 ponto na mesma direção
  - Inaccurate: outros casos

---

### 2. Ace-CEFR (2025)

#### **Abordagem Geral:**
Dataset e modelos para avaliar dificuldade linguística de textos conversacionais curtos.

#### **Etapas da Metodologia:**

**Etapa 1: Criação do Dataset Ace-CEFR**
- **Tamanho:** 890 trechos conversacionais
- **Comprimento médio:** 12 palavras (mediana: 10)
- **Distribuição:**
  - A1: 131
  - A2/A2+: 180
  - B1/B1+: 169
  - B2/B2+: 186
  - C1: 107
  - C2: 116
- **Fontes:**
  - Organização de pesquisa (272)
  - Autorados especificamente (255)
  - Gerados por LLMs (198)
  - Aprendizes de teste (101)
  - Dados públicos web (64)

**Etapa 2: Anotação por Especialistas**
- **Especialistas:** Mestrado em Applied Linguistics + 10+ anos experiência
- **Processo:**
  - Cada texto anotado por pelo menos 2 raters independentes
  - Colaboração em guideline document
  - Ratings convertidos para escala numérica (A1=1, A2=2, A2+=2.5, etc.)
  - Média para consenso
  - Adjudicação para casos com diferença > 1 ponto (5% dos casos)
- **Inter-rater Agreement:** QWK = 0.89

**Etapa 3: Avaliação de Modelos**
- **3 Tipos de Modelos Testados:**

  1. **Linear Regression (Surface Features)**
     - Features: comprimento médio de palavras, comprimento médio de sentenças
     - Latência: ~50µs (CPU)
     - MSE: 0.81

  2. **BERT-based Model**
     - Fine-tuning completo do BERT
     - Latência: ~100ms (CPU) / ~10ms (API)
     - MSE: 0.48 (melhor que humanos)

  3. **PaLM 2-L (Few-shot)**
     - Prompting few-shot
     - Latência: ~1s (API)
     - MSE: 0.48 (com prompts separados para palavras vs frases)

**Etapa 4: Otimização**
- Re-adjudicação dos 20 piores casos de cada modelo
- 123 casos de adjudicação total
- Separação de prompts para palavras vs frases (melhoria significativa)

#### **Métricas de Avaliação:**
- **Métrica Principal:** Mean Squared Error (MSE)
- **Escala:** 1-6 (correspondente a A1-C2)
- **Baseline Humano:** MSE 0.75
- **Melhor Modelo:** MSE 0.48 (BERT-based)
- **Latência:** 10ms-100ms para produção

---

## 🔄 COMPARAÇÃO DETALHADA

### **1. Objetivo e Escopo**

| Aspecto | EvalYaks | Ace-CEFR |
|---------|----------|----------|
| **Objetivo Principal** | Avaliar desempenho em exame B2 | Classificar dificuldade de textos |
| **Foco** | Avaliação de 4 partes do exame | Classificação de nível CEFR |
| **Aplicação** | Scoring de exames | Filtragem/ajuste de LLMs |
| **Especificidade** | Muito específico (B2 Speaking) | Geral (todos os níveis A1-C2) |

### **2. Dataset**

| Aspecto | EvalYaks | Ace-CEFR |
|---------|----------|----------|
| **Tamanho** | 3,060 conversas | 890 trechos |
| **Geração** | Sintético (GPT-4) | Misto (sintético + real) |
| **Validação** | 3 especialistas, validação dupla | 2+ raters, QWK 0.89 |
| **Distribuição** | Focada em B2, balanceada por scores | Uniforme A1-C2 |
| **Comprimento** | Conversas completas | Trechos curtos (média 12 palavras) |

### **3. Modelos e Arquitetura**

| Aspecto | EvalYaks | Ace-CEFR |
|---------|----------|----------|
| **Modelo Base** | Mistral Instruct 7B | BERT, PaLM 2-L, Linear |
| **Fine-tuning** | LoRA (PEFT) | Fine-tuning completo (BERT) |
| **Número de Modelos** | 6 modelos especializados | 3 tipos de modelos |
| **Tamanho** | 7B parâmetros | Varia (BERT base, PaLM 2-L) |
| **Eficiência** | Alta (LoRA é eficiente) | Média (fine-tuning completo) |

### **4. Treinamento**

| Aspecto | EvalYaks | Ace-CEFR |
|---------|----------|----------|
| **Instruction Tuning** | ✅ Sim (3 tipos de instruções) | ❌ Não (fine-tuning direto) |
| **Datasets Adicionais** | English Vocabulary Profile, CEFR-SP | Apenas Ace-CEFR |
| **Data Augmentation** | Paráfrases de instruções | Re-adjudicação de casos difíceis |
| **Total de Data Points** | ~30,000+ (com augmentation) | 890 (445 train, 445 test) |

### **5. Avaliação e Métricas**

| Aspecto | EvalYaks | Ace-CEFR |
|---------|----------|----------|
| **Métrica Principal** | Acceptable Accuracy | Mean Squared Error (MSE) |
| **Resultado** | 96% acceptable accuracy | MSE 0.48 (vs 0.75 humano) |
| **Comparação Humana** | 3x melhor que próximo modelo | Supera humanos (0.48 vs 0.75) |
| **Variação** | 0.35 níveis | Não reportado |
| **Latência** | Não reportado | 10ms-100ms (BERT) |

### **6. Pontos Fortes e Fracos**

#### **EvalYaks - Pontos Fortes:**
- ✅ Especialização em exame B2 específico
- ✅ 6 modelos especializados para diferentes tarefas
- ✅ Instruction tuning bem estruturado
- ✅ Dataset grande e validado (3,060 conversas)
- ✅ Alta precisão (96%)
- ✅ LoRA eficiente (baixo custo)

#### **EvalYaks - Pontos Fracos:**
- ⚠️ Específico apenas para B2
- ⚠️ Requer dataset sintético grande
- ⚠️ 6 modelos para manter
- ⚠️ Latência não reportada

#### **Ace-CEFR - Pontos Fortes:**
- ✅ Cobre todos os níveis CEFR (A1-C2)
- ✅ Dataset pequeno mas eficaz (890 trechos)
- ✅ Supera avaliação humana
- ✅ Baixa latência (10ms-100ms)
- ✅ Múltiplos modelos testados
- ✅ Dataset público disponível

#### **Ace-CEFR - Pontos Fracos:**
- ⚠️ Dataset menor (890 vs 3,060)
- ⚠️ Foco em classificação, não avaliação detalhada
- ⚠️ Não usa instruction tuning
- ⚠️ Fine-tuning completo mais caro

---

## 🎯 DIFERENÇAS PRINCIPAIS

### **1. Abordagem de Fine-tuning**

**EvalYaks:**
- **LoRA (Parameter Efficient Fine-Tuning)**
- Vantagem: Baixo custo, eficiente
- Desvantagem: Pode ter limitações de capacidade

**Ace-CEFR:**
- **Fine-tuning Completo (BERT)**
- Vantagem: Máxima capacidade de aprendizado
- Desvantagem: Mais caro, requer mais recursos

### **2. Instruction Tuning**

**EvalYaks:**
- ✅ Usa instruction tuning extensivamente
- 3 tipos de instruções + paráfrases
- Instruções estruturadas com roles, steps, exemplos

**Ace-CEFR:**
- ❌ Não usa instruction tuning
- Fine-tuning direto no dataset
- Few-shot prompting para PaLM 2-L

### **3. Especialização**

**EvalYaks:**
- 6 modelos especializados
- Cada modelo para tarefa específica
- Foco em exame B2 completo

**Ace-CEFR:**
- Modelos gerais para classificação
- Um modelo para todos os níveis
- Foco em classificação de dificuldade

### **4. Dataset**

**EvalYaks:**
- Dataset sintético grande (3,060)
- Focado em B2, balanceado por scores
- Conversas completas de exame

**Ace-CEFR:**
- Dataset menor mas diversificado (890)
- Distribuição uniforme A1-C2
- Trechos curtos conversacionais

---

## 💡 INSIGHTS E RECOMENDAÇÕES

### **Para Nossa Implementação:**

1. **Combinar Abordagens:**
   - Usar LoRA (EvalYaks) para eficiência
   - Usar instruction tuning (EvalYaks) para melhor performance
   - Usar dataset diversificado (Ace-CEFR) para robustez

2. **Dataset:**
   - Gerar dataset sintético (como EvalYaks)
   - Validar com especialistas (ambos)
   - Balancear distribuição (como Ace-CEFR)

3. **Modelos:**
   - Considerar modelos especializados (EvalYaks)
   - Testar múltiplos tipos (Ace-CEFR)
   - Priorizar latência para produção (Ace-CEFR)

4. **Métricas:**
   - Usar acceptable accuracy (EvalYaks)
   - Considerar MSE para regressão (Ace-CEFR)
   - Comparar com baseline humano (ambos)

---

## 📊 RESUMO EXECUTIVO

| Aspecto | EvalYaks | Ace-CEFR | Melhor para |
|---------|----------|----------|-------------|
| **Precisão** | 96% | MSE 0.48 | EvalYaks (mais específico) |
| **Eficiência** | LoRA | Fine-tuning completo | EvalYaks (mais eficiente) |
| **Cobertura** | B2 apenas | A1-C2 | Ace-CEFR (mais amplo) |
| **Latência** | Não reportado | 10-100ms | Ace-CEFR (otimizado) |
| **Dataset** | 3,060 (sintético) | 890 (misto) | EvalYaks (maior) |
| **Instruction Tuning** | ✅ Sim | ❌ Não | EvalYaks (mais sofisticado) |

---

**Conclusão:** Ambas as metodologias são complementares. EvalYaks é melhor para avaliação detalhada de exames, enquanto Ace-CEFR é melhor para classificação rápida de dificuldade. Para nossa implementação, devemos combinar os pontos fortes de ambas.

---

**Última atualização:** 2025-11-23
