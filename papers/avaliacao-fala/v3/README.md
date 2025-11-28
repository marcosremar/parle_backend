# Papers com Menos Falhas Metodológicas - v3

Este diretório contém papers similares ao Mohammadi (2025) mas com metodologia mais rigorosa e menos falhas.

---

## 📚 Papers Disponíveis

### 1. Banno et al. (2022) ⭐ RECOMENDADO

- **Arquivo:** `Banno_2022.pdf` / `Banno_2022.md`
- **Referência:** arXiv:2211.08849
- **Título:** "L2 Proficiency Assessment Using Self-Supervised Speech Representations"
- **Dataset:** Linguaskill (Cambridge English Language Assessment)
- **Modelos:** wav2vec 2.0, BERT, Standard grader (hand-crafted features)

**Por que é melhor que Mohammadi (2025):**
- ✅ **Métricas completas** (RMSE, PCC, SRC, % ≤ 0.5, % ≤ 1.0)
- ✅ **Dataset balanceado** (gênero, proficiência)
- ✅ **Test sets grandes** (1,049 + 712 speakers)
- ✅ **Análise detalhada** (por parte do exame)
- ✅ **Metodologia apropriada** (regression vs classification)
- ✅ **Comparação robusta** (3 sistemas + combinações)
- ✅ **Resultados melhores** (PCC 0.89-0.94 vs 0.82)

**Limitações:**
- ⚠️ Dataset privado (mas bem documentado)
- ⚠️ Sem matriz de confusão (mas análise por parte)
- ⚠️ Sem baseline humano explícito (mas usa scores operacionais)

---

## 📊 Análise Comparativa

Ver documento completo: `ANALISE_COMPARATIVA.md`

### **Resumo:**

| Aspecto | Mohammadi (2025) | Banno (2022) |
|---------|------------------|--------------|
| **Métricas** | ❌ Incompletas (NC) | ✅ Completas |
| **Balanceamento** | ❌ Desbalanceado | ✅ Balanceado |
| **Test Set** | ⚠️ Pequeno (6-12) | ✅ Grande (1,761 total) |
| **Análise** | ❌ Sem análise de erros | ✅ Análise detalhada |
| **Metodologia** | ⚠️ Classification | ✅ Regression |
| **Resultados** | ⚠️ 85% accuracy, 0.82 corr | ✅ 0.89-0.94 PCC, 85% ≤ 0.5 |

**Vencedor:** ✅ **Banno (2022)** - Significativamente melhor

---

## ⚠️ Papers Não Encontrados

Os seguintes papers foram mencionados mas não estão disponíveis publicamente:

1. **Lo et al. (2025)** - "An effective automated speaking assessment approach to mitigating data scarcity and imbalanced distribution"
   - Pode não estar publicado ainda
   - Pode estar em processo de revisão

2. **McKnight et al. (2023)** - "Automatic assessment of conversational speaking tests"
   - Pode não estar no arxiv
   - Pode estar em conferência sem acesso público

3. **Singla et al. (2021)** - "Speaker-Conditioned Hierarchical Modeling for Automated Speech Scoring"
   - ACM Conference (pode requerer acesso)
   - Pode estar disponível via ACM Digital Library

4. **Banno & Matassoni (2023)** - "Proficiency assessment of L2 spoken English using wav2vec 2.0"
   - IEEE SLT (pode requerer acesso)
   - Pode estar disponível via IEEE Xplore

---

## 💡 Recomendações

### **Para Nossa Implementação:**

1. **Usar Banno (2022) como referência principal:**
   - Metodologia mais rigorosa
   - Menos falhas metodológicas
   - Resultados melhores
   - Análise mais completa

2. **Aplicar lições aprendidas:**
   - Usar **regression** em vez de classification quando apropriado
   - Reportar **métricas completas** (RMSE, PCC, SRC)
   - **Balancear datasets** ou usar técnicas de balanceamento
   - **Análise detalhada** por componente
   - **Comparar múltiplos baselines**

3. **Evitar falhas do Mohammadi:**
   - Não usar dataset de texto escrito para fala
   - Não usar datasets privados sem documentação
   - Reportar todas as métricas
   - Fazer análise de erros
   - Comparar com baseline humano

---

## 📋 Estatísticas

- **Total de papers:** 1 (Banno 2022)
- **Formato:** PDF + Markdown
- **Análise:** Completa
- **Comparação:** vs Mohammadi (2025)

---

## 🔗 Links Úteis

- **Análise Comparativa:** `ANALISE_COMPARATIVA.md`
- **Falhas do Mohammadi:** `../v2/FALHAS_METODOLOGICAS_MOHAMMADI_2025.md`
- **Papers Similares:** `../v2/PAPERS_SIMILARES_MENOS_FALHAS.md`

---

**Última atualização:** 2025-11-23

