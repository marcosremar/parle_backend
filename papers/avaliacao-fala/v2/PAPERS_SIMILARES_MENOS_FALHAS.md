# Papers Similares com Menos Falhas Metodológicas

## 📋 Resumo

Este documento lista papers similares ao Mohammadi (2025) mas com metodologia mais rigorosa e menos falhas.

---

## ✅ PAPERS RECOMENDADOS

### 1. **McKnight et al. (2023) - Automatic Assessment of Conversational Speaking Tests**

**Mencionado no Mohammadi (2025):**
> "Compared to McKnight et al. [6], which also investigated dialogue-based assessment, their work applied regression-based scoring using a wav2vec 2.0 + Longformer model"

**Por que pode ser melhor:**
- ✅ Usa **regression-based scoring** (mais apropriado que classification)
- ✅ Combina **wav2vec 2.0 + Longformer** (multimodal)
- ✅ Foca em **conversational speaking tests** (similar ao objetivo)
- ✅ Metodologia diferente pode ser mais robusta

**Buscar:** arxiv "McKnight" "Automatic assessment conversational speaking tests" 2023

---

### 2. **Lo et al. (2025) - An Effective Automated Speaking Assessment Approach**

**Mencionado no Mohammadi (2025):**
> "while Lo et al. [26] reported high proficiency classification accuracy using wav2vec 2.0, their results were achieved in a speaker-dependent setting"

**Título completo (provável):**
"An effective automated speaking assessment approach to mitigating data scarcity and imbalanced distribution"

**Por que pode ser melhor:**
- ✅ **Especificamente trata desbalanceamento** (falha crítica do Mohammadi)
- ✅ **Speaker-independent** (como Mohammadi, mas pode ter melhor metodologia)
- ✅ Foca em **data scarcity** (problema real)
- ✅ Mais recente (2025)

**Buscar:** arxiv "Lo" "automated speaking assessment" "data scarcity" 2025

---

### 3. **Banno & Matassoni (2022, 2023) - Cross-Corpora Experiments**

**Papers:**
- "Proficiency assessment of L2 spoken English using wav2vec 2.0" (2023)
- "Cross-corpora experiments of automatic proficiency assessment" (2022)

**Por que pode ser melhor:**
- ✅ **Cross-corpora experiments** (testa em múltiplos datasets)
- ✅ Validação mais robusta
- ✅ Menos dependente de um dataset específico
- ✅ Pode ter datasets públicos

**Buscar:** arxiv "Banno" "Matassoni" "L2 proficiency assessment" wav2vec 2022 2023

---

### 4. **Singla et al. (2021) - Speaker-Conditioned Hierarchical Modeling**

**Título:**
"Speaker-Conditioned Hierarchical Modeling for Automated Speech Scoring"

**Por que pode ser melhor:**
- ✅ Metodologia inovadora (hierarchical modeling)
- ✅ Melhoria de **6.92%** vs baselines (resultados claros)
- ✅ Usa contexto do falante (mais robusto)
- ✅ Pode ter análise mais completa

**Buscar:** arxiv "Singla" "Speaker-Conditioned Hierarchical Modeling" 2021

---

## 🔍 CRITÉRIOS DE QUALIDADE

### **O que buscar em cada paper:**

1. **✅ Dataset Público ou Bem Documentado**
   - Dataset disponível ou descrito em detalhes
   - Sem datasets privados não disponíveis

2. **✅ Métricas Completas**
   - Precision, Recall, F1 (não apenas accuracy)
   - Matriz de confusão
   - Análise de erros

3. **✅ Baseline Humano**
   - Comparação com avaliação humana
   - Concordância inter-avaliadores

4. **✅ Análise de Erros**
   - Matriz de confusão
   - Identificação de padrões de erro
   - Análise de casos difíceis

5. **✅ Dataset Balanceado ou Técnicas de Balanceamento**
   - Distribuição equilibrada
   - Técnicas de balanceamento aplicadas

6. **✅ Validação Rigorosa**
   - Test sets adequados
   - Speaker-independent
   - Cross-validation ou hold-out apropriado

7. **✅ Reprodutibilidade**
   - Código disponível (opcional mas desejável)
   - Hiperparâmetros reportados
   - Metodologia clara

---

## 📊 COMPARAÇÃO ESPERADA

| Aspecto | Mohammadi (2025) | Papers Recomendados |
|---------|------------------|---------------------|
| **Dataset** | ❌ Privado não disponível | ✅ Público ou bem documentado |
| **Métricas** | ⚠️ Incompletas (NC) | ✅ Completas |
| **Análise Erros** | ❌ Ausente | ✅ Presente |
| **Baseline Humano** | ❌ Ausente | ✅ Provavelmente presente |
| **Balanceamento** | ❌ Desbalanceado | ✅ Balanceado ou técnicas aplicadas |
| **Reprodutibilidade** | ❌ Baixa | ✅ Alta |

---

## 🎯 PRÓXIMOS PASSOS

1. **Buscar papers específicos:**
   - McKnight et al. (2023) - arxiv
   - Lo et al. (2025) - arxiv
   - Banno & Matassoni (2022, 2023) - arxiv
   - Singla et al. (2021) - arxiv

2. **Analisar cada paper:**
   - Verificar se tem datasets públicos
   - Verificar se tem métricas completas
   - Verificar se tem análise de erros
   - Verificar se tem baseline humano

3. **Comparar metodologias:**
   - Identificar pontos fortes
   - Identificar pontos fracos
   - Recomendar melhor abordagem

---

**Última atualização:** 2025-11-23
