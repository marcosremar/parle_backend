# Papers sobre Fala com Dados Reais (Atualizados - 2019-2025)
## Similar ao Arnold (2018), mas sobre fala e mais recentes

---

## ✅ PAPERS ENCONTRADOS COM DADOS REAIS

### **1. Mohammadi et al. (2025) - Automatic Proficiency Assessment in L2 English Learners**

**📅 Ano:** 2025  
**🔗 Link:** https://arxiv.org/abs/2505.02615

**Dataset:**
- ✅ **EFCAMDAT** (English First Cambridge open language Database) - **DADOS REAIS**
- ✅ **ANGLISH** - **DADOS REAIS**
- ✅ Dataset privado adicional - **DADOS REAIS**

**Metodologia:**
- **Modelos de Áudio:**
  - 2D CNN
  - CNN baseada em frequência
  - ResNet
  - **wav2vec 2.0** (pré-treinado) - melhor performance
- **Modelos de Texto:**
  - Fine-tuning de **BERT** para avaliação baseada em transcrições

**Foco:**
- Avaliação abrangente de proficiência em inglês L2
- Análise de **sinal de fala** E **transcrições**
- Classificação de proficiência oral

**Vantagens:**
- ✅ **100% dados reais** (EFCAMDAT, ANGLISH)
- ✅ Análise multimodal (áudio + texto)
- ✅ Modelos modernos (wav2vec 2.0, BERT)
- ✅ Mais recente (2025)

**Limitações:**
- ⚠️ Não especifica níveis CEFR específicos
- ⚠️ Foco em inglês (não português)

---

### **2. CELPE-Bras Study (2025) - Dimensionalidade das Escalas de Avaliação Oral**

**📅 Ano:** 2025  
**🔗 Link:** https://www.scielo.br/j/ep/a/KWYysnwZJK7xFL6NfvkjdND/

**Dataset:**
- ✅ Dados reais do **Certificado de Proficiência em Língua Portuguesa para Estrangeiros (CELPE-Bras)**
- ✅ Falantes não nativos de português
- ✅ Dados autênticos de exames

**Foco:**
- Análise das escalas de avaliação da proficiência oral
- Dimensionalidade das escalas
- Metodologia de avaliação

**Vantagens:**
- ✅ **100% dados reais** (exames CELPE-Bras)
- ✅ **Português** (relevante para nosso projeto!)
- ✅ Dados de exames oficiais
- ✅ Mais recente (2025)

**Limitações:**
- ⚠️ Foco em análise de escalas, não em classificação automática
- ⚠️ Pode não ter implementação de modelos automatizados

---

## ⚠️ PAPERS COM DADOS SINTÉTICOS (Para Comparação)

### **3. EvalYaks (2024) - Automated Scoring of CEFR B2 Speaking**

**📅 Ano:** 2024  
**Dataset:** ❌ 100% sintético (GPT-4 Turbo)  
**Foco:** CEFR B2 Speaking Assessment  
**Precisão:** 96%  
**Limitação:** Dados sintéticos validados por especialistas

---

## 📊 COMPARAÇÃO: Arnold (2018) vs Papers Atualizados

| Aspecto | Arnold (2018) | Mohammadi (2025) | CELPE-Bras (2025) |
|---------|---------------|------------------|-------------------|
| **Ano** | 2018 | 2025 | 2025 |
| **Dados** | ✅ 100% reais (EFCAMDAT) | ✅ 100% reais (EFCAMDAT, ANGLISH) | ✅ 100% reais (CELPE-Bras) |
| **Tipo** | Texto escrito | **Fala (áudio + texto)** | **Fala (português)** |
| **Língua** | Inglês | Inglês | **Português** |
| **Modelos** | Gradient Boosted Trees, Neural Networks | wav2vec 2.0, BERT | Análise de escalas |
| **CEFR** | A1, A2, B1 | Não especificado | CELPE-Bras (equivalente CEFR) |
| **AUC/Accuracy** | AUC 0.91-0.92 | Não reportado | Não reportado |

---

## 💡 RECOMENDAÇÕES

### **Para Nossa Implementação:**

1. **Mohammadi et al. (2025):**
   - ✅ Mais similar ao Arnold (usa EFCAMDAT)
   - ✅ Foca em **fala** (não texto)
   - ✅ Modelos modernos (wav2vec 2.0, BERT)
   - ✅ **Baixar e analisar este paper!**

2. **CELPE-Bras (2025):**
   - ✅ **Português** (muito relevante!)
   - ✅ Dados reais de exames
   - ✅ Pode ter insights sobre avaliação de fala em português
   - ✅ **Baixar e analisar este paper!**

3. **Próximos Passos:**
   - Baixar paper Mohammadi (2025)
   - Baixar paper CELPE-Bras (2025)
   - Comparar metodologias
   - Adaptar para nosso contexto

---

## 🔍 PRÓXIMOS PASSOS

1. **Baixar Papers:**
   - [ ] Mohammadi et al. (2025) - https://arxiv.org/abs/2505.02615
   - [ ] CELPE-Bras (2025) - https://www.scielo.br/j/ep/a/KWYysnwZJK7xFL6NfvkjdND/

2. **Analisar:**
   - Metodologias usadas
   - Modelos e arquiteturas
   - Resultados e métricas
   - Aplicabilidade ao nosso projeto

3. **Adaptar:**
   - Incorporar insights dos papers
   - Melhorar nossa implementação
   - Validar com dados reais

---

**Última atualização:** 2025-11-23
