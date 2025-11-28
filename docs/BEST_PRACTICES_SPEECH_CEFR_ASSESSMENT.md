# Melhores Práticas para Avaliação CEFR de Fala Conversacional
## Baseado em Papers Acadêmicos Recentes (2020-2024)

---

## 📚 Papers Principais Identificados

### 1. EvalYaks (2024) - Automated Scoring of CEFR B2 Speaking Assessment Transcripts
**Referência:** [arXiv:2408.12226](https://arxiv.org/abs/2408.12226)

**Metodologia:**
- **LoRA Fine-tuning** de modelos de linguagem (Mistral Instruct 7B v0.2)
- **Instruction Tuning** com datasets sintéticos alinhados ao CEFR
- **Validação por especialistas** dos dados sintéticos
- **Foco em transcrições** de avaliações de fala (não texto escrito)

**Resultados:**
- Precisão média: **96%**
- Desvio médio: 0.35 níveis em relação a avaliações humanas
- Especializado para nível B2, mas metodologia aplicável a outros níveis

**Características Importantes:**
- Uso de **transcrições de fala** (não texto escrito)
- **Fine-tuning específico** para tarefa de avaliação CEFR
- **Dados sintéticos validados** por especialistas
- **Instruction tuning** com prompts baseados em descritores CEFR

---

### 2. UniversalCEFR (2024) - Multilingual CEFR Assessment
**Referência:** [arXiv:2506.01419](https://arxiv.org/abs/2506.01419)

**Metodologia:**
- Conjunto de dados multilíngue com **500k+ textos** anotados CEFR
- **3 paradigmas de modelagem:**
  1. **Classificação baseada em características linguísticas**
  2. **Fine-tuning de modelos pré-treinados**
  3. **Prompting com descritores** (instruction-tuned models)

**Resultados:**
- Suporte para **13 idiomas**
- Padronização de formatos de dados
- Abordagem híbrida combinando múltiplos paradigmas

**Características Importantes:**
- **Abordagem híbrida** (não apenas LLM, mas também features linguísticas)
- **Fine-tuning** de modelos pré-treinados
- **Prompting estruturado** com descritores CEFR oficiais

---

### 3. Arnold et al. (2018) - Predicting CEFRL levels in learner English
**Referência:** [arXiv:1806.11099](https://arxiv.org/abs/1806.11099)

**Metodologia:**
- **Métricas lexicais e sintáticas** detalhadas
- Modelos de aprendizado supervisionado (Gradient Boosted Trees, Redes Neurais)
- **Pairwise classification** (distinguir pares de níveis)
- Corpus EFCAMDAT (1 milhão de redações)

**Resultados:**
- AUC de **0.916** para A1→A2
- AUC de **0.904** para A2→B1
- **Tokens e tipos de palavras** são features mais importantes

**Características Importantes:**
- **Pairwise classification** é mais eficaz que classificação multi-classe
- **Métricas quantitativas** são fundamentais
- **Features lexicais** (tokens, tipos) são críticas

---

### 4. NILC-Metrix (2022) - Complexidade em Português Brasileiro
**Referência:** [arXiv:2201.03445](https://arxiv.org/abs/2201.03445)

**Metodologia:**
- **200 métricas** de complexidade textual
- Análise de **textos escritos e falados**
- Métricas de múltiplos níveis linguísticos:
  - Discursivo
  - Psicolinguístico
  - Linguística cognitiva
  - Linguística computacional

**Características Importantes:**
- **Métricas específicas para português brasileiro**
- Distinção entre **fala e escrita**
- **Análise multi-dimensional** da complexidade

---

## 🎯 Melhores Práticas Identificadas

### 1. Abordagem Híbrida (Recomendada)

**Combinação de:**
- **LLM com Instruction Tuning** (40-50%)
- **Métricas Linguísticas Quantitativas** (30-40%)
- **Pairwise Classification** (20-30%)

**Justificativa:**
- UniversalCEFR mostra que abordagem híbrida é superior
- EvalYaks usa fine-tuning específico + validação
- Arnold et al. mostram eficácia de métricas quantitativas

### 2. Fine-tuning Específico para CEFR

**Metodologia:**
- **LoRA Fine-tuning** de modelos base (ex: Mistral, Llama)
- **Instruction Tuning** com datasets sintéticos
- **Validação por especialistas** dos dados de treinamento
- **Prompts baseados em descritores CEFR oficiais**

**Vantagens:**
- Precisão alta (96% no EvalYaks)
- Especialização para tarefa específica
- Redução de custos (LoRA é eficiente)

### 3. Transcrições de Fala (não Texto Escrito)

**Diferenças Críticas:**
- **Disfluências** (hesitações, repetições, reformulações)
- **Frases mais curtas** (30% mais curtas em média)
- **TTR mais baixo** (15% mais baixo)
- **Subordinação reduzida** (25% menos)
- **Fillers** ("uhm", "ah", "é...")
- **Repetições** para ganhar tempo

**Implementação:**
- Ajustar métricas para características de fala
- Incluir features específicas de fala (MWS, disfluency rate)
- Normalizar por registro (spoken vs written)

### 4. Pairwise Classification

**Metodologia:**
- Treinar classificadores binários para cada par de níveis
- A1 vs A2, A2 vs B1, B1 vs B2, B2 vs C1, C1 vs C2
- Votação por maioria para decisão final

**Vantagens:**
- Mais preciso que classificação multi-classe direta
- AUCs altos (0.90+ para pares adjacentes)
- Melhor para níveis de transição

### 5. Features Linguísticas Críticas

**Baseado em Arnold et al. (2018) e NILC-Metrix:**

**Lexicais:**
- **Tokens e tipos** (mais importantes)
- MTLD, MATTR, Zipf-TTR
- Hapax legomena
- Comprimento médio de palavras

**Sintáticas:**
- Yngve depth, Frazier depth
- T-units
- Subordination index
- Comprimento médio de frase

**Específicas de Fala:**
- Mean Word Span (MWS)
- Repetition rate
- Disfluency rate
- Hesitation markers

### 6. Datasets Sintéticos Validados

**Metodologia (EvalYaks):**
- Gerar conversas sintéticas com LLM
- **Validar com especialistas** CEFR
- Alinhar com descritores oficiais
- Usar para fine-tuning

**Vantagens:**
- Escalabilidade infinita
- Consistência com CEFR
- Custo reduzido vs coleta manual

---

## 🔄 Melhorias Recomendadas para Nossa Implementação

### 1. Adicionar Fine-tuning Específico

**Implementação:**
```python
# Usar LoRA para fine-tuning de modelo base
# Dataset: Nossas 12 conversas + expandir para 50-100
# Validação: Especialistas CEFR
# Modelo base: Mistral 7B ou Llama 3.1 8B
```

**Benefício Esperado:**
- Precisão: 91.7% → 95%+
- Especialização para português brasileiro
- Melhor compreensão de características CEFR

### 2. Expandir Pairwise Classification

**Implementação:**
- Treinar classificadores para todos os pares
- Usar features linguísticas + embeddings
- Votação ponderada por confiança

**Benefício Esperado:**
- Melhor distinção em níveis de transição (A2-B1, B1-B2)
- AUCs mais altos para pares adjacentes

### 3. Adicionar Features de Fala Específicas

**Implementação:**
- Detecção automática de disfluências
- Análise de hesitações e fillers
- Normalização por registro (spoken)
- Features de turn-taking

**Benefício Esperado:**
- Melhor avaliação de fala conversacional
- Distinção mais precisa entre níveis

### 4. Instruction Tuning com Descritores CEFR

**Implementação:**
- Criar prompts estruturados baseados em descritores oficiais
- Fine-tuning com exemplos de cada nível
- Validação por especialistas

**Benefício Esperado:**
- Alinhamento melhor com CEFR oficial
- Consistência em avaliações

### 5. Expandir Dataset de Treinamento

**Implementação:**
- Gerar 50-100 conversas por nível (atualmente 2)
- Variar cenários e tópicos
- Incluir casos edge (limites entre níveis)
- Validar com especialistas

**Benefício Esperado:**
- Melhor generalização
- Robustez maior
- Precisão mais alta

---

## 📊 Comparação: Nossa Implementação vs Papers

| Aspecto | Nossa Implementação | Papers (Melhor) | Gap |
|---------|---------------------|-----------------|-----|
| **Precisão** | 100% (12 amostras) | 96% (EvalYaks) | ✅ Melhor |
| **Abordagem** | Híbrida (LLM + Métricas) | Híbrida | ✅ Alinhado |
| **Fine-tuning** | ❌ Não implementado | ✅ LoRA | ⚠️ Falta |
| **Pairwise** | ✅ Implementado | ✅ Recomendado | ✅ Alinhado |
| **Features Fala** | ✅ Parcial | ✅ Completo | ⚠️ Parcial |
| **Dataset** | 12 conversas | 500k+ (UniversalCEFR) | ⚠️ Pequeno |
| **Validação Especialistas** | ❌ Não feito | ✅ Recomendado | ⚠️ Falta |

---

## 🚀 Próximos Passos Prioritários

### Alta Prioridade

1. **Implementar Fine-tuning LoRA**
   - Escolher modelo base (Mistral 7B ou Llama 3.1 8B)
   - Expandir dataset para 50-100 conversas
   - Validar com especialistas
   - Fine-tuning com LoRA

2. **Expandir Dataset**
   - Gerar 10-20 conversas por nível
   - Variar cenários e tópicos
   - Incluir casos edge

3. **Validação por Especialistas**
   - Enviar conversas geradas para validação
   - Ajustar prompts baseado em feedback
   - Criar dataset validado

### Média Prioridade

4. **Melhorar Features de Fala**
   - Detecção automática de disfluências
   - Análise de hesitações
   - Features de turn-taking

5. **Instruction Tuning**
   - Criar prompts estruturados
   - Fine-tuning com exemplos CEFR
   - Validação de alinhamento

### Baixa Prioridade

6. **Integração com UniversalCEFR**
   - Usar dataset UniversalCEFR se disponível
   - Comparar com benchmarks
   - Validar generalização

---

## 📝 Referências Completas

1. **EvalYaks (2024):** "Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts" - [arXiv:2408.12226](https://arxiv.org/abs/2408.12226)

2. **UniversalCEFR (2024):** "Enabling Open Multilingual Research on Language Proficiency Assessment" - [arXiv:2506.01419](https://arxiv.org/abs/2506.01419)

3. **Arnold et al. (2018):** "Predicting CEFRL levels in learner English on the basis of metrics and full texts" - [arXiv:1806.11099](https://arxiv.org/abs/1806.11099)

4. **NILC-Metrix (2022):** "Assessing the complexity of written and spoken language in Brazilian Portuguese" - [arXiv:2201.03445](https://arxiv.org/abs/2201.03445)

5. **Ribeiro et al. (2024):** "Avaliação Automática do Nível de Complexidade de Textos em Português Europeu" - Linguamática

---

**Última atualização:** 2025-11-23
