# Análise Comparativa: Papers com Menos Falhas Metodológicas

## 📋 Resumo Executivo

Este documento compara papers similares ao Mohammadi (2025) para identificar estudos com metodologia mais rigorosa e menos falhas.

---

## ✅ PAPER ANALISADO: Banno et al. (2022)

### **Informações Básicas:**
- **Título:** "L2 Proficiency Assessment Using Self-Supervised Speech Representations"
- **Ano:** 2022
- **arXiv:** 2211.08849
- **Dataset:** Linguaskill (Cambridge English Language Assessment)

---

## 📊 COMPARAÇÃO DETALHADA

### **1. Dataset**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Disponibilidade** | ❌ Privado não disponível | ⚠️ Privado mas bem documentado |
| **Documentação** | ⚠️ Limitada | ✅ Muito detalhada |
| **Tamanho Train** | Não especificado claramente | ✅ 31,475 speakers |
| **Tamanho Test** | ⚠️ 6-12 speakers | ✅ 1,049 + 712 speakers |
| **Balanceamento** | ❌ Desbalanceado (A1=191k, C1=5k) | ✅ Balanceado (gênero, proficiência) |
| **Diversidade** | ⚠️ Limitada | ✅ ~30 L1s (línguas nativas) |

**Vencedor:** ✅ **Banno (2022)** - Dataset muito melhor documentado e balanceado

---

### **2. Métricas Reportadas**

| Métrica | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Accuracy** | ✅ 85% | ⚠️ Não reportado (regression) |
| **Precision** | ❌ NC (Not Computed) para modelos ruins | ✅ Não aplicável (regression) |
| **Recall** | ❌ NC (Not Computed) para modelos ruins | ✅ Não aplicável (regression) |
| **F1 Score** | ❌ NC (Not Computed) para modelos ruins | ✅ Não aplicável (regression) |
| **RMSE** | ❌ Não reportado | ✅ 0.356-0.455 |
| **PCC** | ✅ 0.82 (correlação) | ✅ 0.893-0.943 |
| **SRC** | ❌ Não reportado | ✅ 0.911-0.947 |
| **% ≤ 0.5** | ❌ Não reportado | ✅ 72.1-85.0% |
| **% ≤ 1.0** | ❌ Não reportado | ✅ 97.3-99.4% |

**Vencedor:** ✅ **Banno (2022)** - Métricas muito mais completas e apropriadas para regression

---

### **3. Análise de Erros**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Matriz de Confusão** | ❌ Ausente | ⚠️ Não aplicável (regression) |
| **Análise por Nível** | ❌ Ausente | ✅ Análise por parte do exame |
| **Padrões de Erro** | ❌ Não identificados | ✅ Discussão de limitações |
| **Casos Difíceis** | ❌ Não analisados | ✅ Análise de partes problemáticas |

**Vencedor:** ✅ **Banno (2022)** - Análise mais detalhada e discussão de limitações

---

### **4. Baseline e Comparação**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Baseline Humano** | ❌ Ausente | ⚠️ Scores de examinadores operacionais |
| **Concordância Inter-avaliadores** | ❌ Não reportada | ⚠️ Não reportada explicitamente |
| **Comparação com Outros Modelos** | ⚠️ Limitada | ✅ 3 sistemas comparados |
| **Combinação de Modelos** | ⚠️ Multi-task (gênero) | ✅ Combinações lineares testadas |

**Vencedor:** ✅ **Banno (2022)** - Comparação mais abrangente

---

### **5. Validação e Reprodutibilidade**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Speaker-Independent** | ✅ Sim | ✅ Sim (implícito) |
| **Cross-Validation** | ⚠️ 10-fold (ANGLISH apenas) | ✅ Hold-out apropriado |
| **Test Sets Separados** | ⚠️ Pequenos | ✅ 2 test sets (LinGen, LinBus) |
| **Hiperparâmetros** | ⚠️ Parcialmente reportados | ✅ Detalhados por parte |
| **Reprodutibilidade** | ❌ Baixa (dataset privado) | ⚠️ Média (dataset privado mas documentado) |

**Vencedor:** ✅ **Banno (2022)** - Validação mais rigorosa

---

### **6. Metodologia**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Abordagem** | ⚠️ Classification | ✅ Regression (mais apropriado) |
| **Modelos** | ⚠️ wav2vec 2.0, BERT separados | ✅ wav2vec 2.0, BERT, Standard combinados |
| **Ensemble** | ❌ Não mencionado | ✅ Ensemble usado |
| **Calibração** | ❌ Não mencionada | ✅ OLS calibration |
| **Multi-task** | ⚠️ Gênero (questionável) | ❌ Não usado |

**Vencedor:** ✅ **Banno (2022)** - Metodologia mais apropriada (regression vs classification)

---

## 📈 RESULTADOS COMPARADOS

### **Mohammadi (2025):**
- Accuracy: 85%
- Correlação: 0.82
- Dataset: EFCAMDAT (texto escrito usado para fala), ANGLISH, Private

### **Banno (2022):**
- RMSE: 0.356-0.455
- PCC: 0.893-0.943
- SRC: 0.911-0.947
- % ≤ 0.5: 72.1-85.0%
- % ≤ 1.0: 97.3-99.4%
- Dataset: Linguaskill (fala real, bem documentado)

**Interpretação:**
- Banno tem **correlação muito maior** (0.89-0.94 vs 0.82)
- Banno tem **85% dentro de meio nível** (vs 85% accuracy do Mohammadi)
- Banno usa **regression** (mais apropriado que classification)

---

## ✅ PONTOS FORTES DO BANNO (2022)

1. **✅ Dataset Balanceado**
   - Balanceado para gênero e proficiência
   - ~30 L1s (diversidade)
   - Test sets grandes (1,049 + 712)

2. **✅ Métricas Completas**
   - RMSE, PCC, SRC
   - % dentro de meio nível
   - % dentro de um nível

3. **✅ Análise Detalhada**
   - Performance por parte do exame
   - Discussão de limitações
   - Análise de diferentes tipos de resposta

4. **✅ Comparação Robusta**
   - 3 sistemas comparados
   - Combinações testadas
   - Ensemble usado

5. **✅ Metodologia Apropriada**
   - Regression (não classification)
   - Calibração com OLS
   - Hiperparâmetros detalhados

---

## ⚠️ LIMITAÇÕES DO BANNO (2022)

1. **⚠️ Dataset Privado**
   - Linguaskill não disponível publicamente
   - Mas muito bem documentado

2. **⚠️ Sem Matriz de Confusão**
   - Não mostra confusão entre níveis
   - Mas tem análise por parte

3. **⚠️ Sem Baseline Humano Explícito**
   - Não reporta concordância inter-avaliadores
   - Mas usa scores de examinadores operacionais

---

## 🎯 CONCLUSÃO

### **Banno (2022) é SIGNIFICATIVAMENTE MELHOR que Mohammadi (2025):**

✅ **Menos Falhas:**
- Métricas completas (vs incompletas)
- Dataset balanceado (vs desbalanceado)
- Test sets grandes (vs pequenos)
- Análise detalhada (vs ausente)
- Metodologia apropriada (regression vs classification)

✅ **Melhor Performance:**
- Correlação 0.89-0.94 (vs 0.82)
- 85% dentro de meio nível
- Combinações melhoram resultados

✅ **Mais Rigoroso:**
- Validação mais completa
- Comparação com múltiplos baselines
- Hiperparâmetros detalhados

---

## 📚 RECOMENDAÇÃO

**Usar Banno (2022) como referência principal** em vez de Mohammadi (2025) porque:

1. ✅ Metodologia mais rigorosa
2. ✅ Menos falhas metodológicas
3. ✅ Resultados melhores
4. ✅ Análise mais completa
5. ✅ Dataset melhor documentado

---

**Última atualização:** 2025-11-23

