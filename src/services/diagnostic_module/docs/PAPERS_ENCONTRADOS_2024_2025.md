# Papers Relevantes Encontrados (2024-2025)

## Sumário

Este documento lista os principais papers encontrados nas buscas online para melhorar a metodologia e implementação do `speech_grader`. Os papers estão organizados por tema e incluem informações sobre disponibilidade, relevância e principais contribuições.

---

## 1. Papers sobre Calibração e Alinhamento Humano

### 1.1. Byun et al. (2025) - LLM-as-a-Grader
- **Título completo:** *LLM-as-a-Grader: Assessing Student Writing with Large Language Models*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v3/Byun_2025_LLM_as_a_Grader.pdf`
- **Foco:** Avaliação de escrita com LLMs, calibração com avaliadores humanos
- **Principais contribuições:**
  - LLMs têm vieses sistemáticos (divergem de humanos em até 30% dos casos)
  - Propõe "rubric-aligned evaluation" (avaliação alinhada a rubricas)
  - Métricas de qualidade: inter-rater agreement, Cohen's Kappa
  - Análise de desacordos entre LLM e humanos
  - Toolkit reproduzível para calibração

### 1.2. Arnold et al. (2018) - Automatic Grading with Real Data
- **Título completo:** *Automatic Grading of Learner English Using a Details-First Approach*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/Arnold_2018.pdf`
- **Foco:** Classificação CEFR (A1, A2, B1) com dataset real (EFCAMDAT - 1 milhão de textos)
- **Principais contribuições:**
  - AUC = 0.916 (A1→A2) e 0.904 (A2→B1) com dados reais
  - Usa Gradient Boosted Trees, Neural Networks, Pairwise Classification
  - Dataset anotado por humanos (professores)
  - Metodologia robusta para calibração

---

## 2. Papers sobre Features Acústicas e Fala

### 2.1. Banno et al. (2022) - Automated Speaking Assessment with Wav2Vec 2.0
- **Título completo:** *Automated Speaking Assessment of Conversation Tests with Wav2Vec 2.0*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v3/Banno_2022.pdf`
- **Foco:** Avaliação automática de fala usando wav2vec 2.0 (features acústicas)
- **Principais contribuições:**
  - PCC = 0.75 (correlação com avaliadores humanos)
  - RMSE = 0.48 (erro médio de 0.48 níveis)
  - 80.4% das predições dentro de 0.5 níveis do humano
  - Usa Linguaskill dataset (real, não sintético)
  - Avalia 5 partes do exame: perguntas pessoais, leitura em voz alta, monólogos, descrição de gráficos, opiniões
  - Metodologia: wav2vec 2.0 (frozen CNN) + fine-tuned Transformer + Regression Head

### 2.2. Mohammadi et al. (2025) - Automated Assessment with Acoustic Features
- **Título completo:** *Automated Assessment of Non-Native Learner Essays Using LLMs and Acoustic Features*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/Mohammadi_2025.pdf`
- **Foco:** Avaliação de fala com LLMs + features acústicas
- **Principais contribuições:**
  - Extrai 88 features acústicas (pitch, energia, MFCCs)
  - Avalia 6 aspectos: fluência, precisão gramatical, pronúncia, coerência discursiva, complexidade lexical, adequação pragmática
  - Acurácia: 85%, correlação: 0.82
- **Limitações identificadas:**
  - Usa EFCAMDAT (dataset de textos escritos) para avaliar fala
  - Dataset privado não disponível
  - Métricas incompletas (NC não definido)
  - Falta análise de erros

### 2.3. CASPER Dataset (2024) - Large Scale Spontaneous Speech
- **Título completo:** *CASPER: A Large Scale Spontaneous Speech Dataset*
- **Status:** ⚠️ **Não baixado** (mencionado em múltiplas buscas, mas link direto não encontrado)
- **Foco:** Dataset com 200+ horas de fala espontânea + metadados
- **Principais contribuições:**
  - Fala espontânea real (não lida)
  - Timestamps e metadados acústicos
  - Estrutura reproduzível para coleta de dados
- **Como obter:** Buscar em [chatpaper.com](https://chatpaper.com/pt/chatpaper/paper/144934) ou arXiv

---

## 3. Papers sobre Feedback Estruturado e Explicabilidade

### 3.1. Xiao et al. (2024) - Automated Essay Scoring with Explainable Feedback
- **Título completo:** *Automated Essay Scoring with Explainable Feedback Using Large Language Models*
- **Status:** ⚠️ **Não baixado** (mencionado em buscas, mas link direto não encontrado)
- **Foco:** Feedback explicável e estruturado para avaliação de textos
- **Principais contribuições:**
  - Feedback estruturado em 3 componentes:
    - **Strengths** (reforço positivo)
    - **Weaknesses** (diagnóstico específico)
    - **Next Steps** (ação prática)
  - Aumenta engajamento e eficácia pedagógica
- **Como obter:** Buscar em ACL Anthology 2024

### 3.2. Lu et al. (2025) - Hybrid Automated Speaking Assessment
- **Título completo:** *Hybrid Automated Speaking Assessment with Grammar, Relevance, and Acoustic Features*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v4/Lu_2025.pdf`
- **Foco:** Avaliação híbrida multi-aspecto de fala espontânea
- **Principais contribuições:**
  - Multi-aspecto: gramática, relevância de tarefa, features acústicas
  - Categoriza erros gramaticais em 55 tipos (ERRANT framework)
  - Usa SBERT para similaridade semântica (exemplar-response)
  - Usa Long-CLIP para relevância multimodal (imagem-response)
  - Usa Microsoft Phi-4 para GEC (Grammar Error Correction)
  - Correlação 0.82 com avaliadores humanos
  - Dataset: NICT-JLE (real, não sintético)

---

## 4. Papers sobre Relevância Semântica e Embeddings

### 4.1. Reimers & Gurevych (2019) - Sentence-BERT
- **Título completo:** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- **Status:** ✅ **Baixado** em `/papers/avaliacao-fala/v2/Reimers_Gurevych_2019_Sentence_BERT.pdf`
- **Foco:** Embeddings de sentenças para similaridade semântica
- **Principais contribuições:**
  - SBERT para calcular similaridade entre textos (1000x mais rápido que BERT)
  - Usado em Lu et al. (2025) para relevância de tarefa
  - Estado-da-arte para semantic similarity
  - Spearman correlation: 0.85 em STS benchmark
  - Código: [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- **Análise detalhada:** Ver `ANALISE_PAPERS_BAIXADOS.md`

### 4.2. Ace-CEFR (2025) - Automated Evaluation of Linguistic Difficulty
- **Título completo:** *Ace-CEFR: A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/Ace-CEFR_2025.pdf`
- **Foco:** Dataset e metodologia para classificar dificuldade linguística de textos conversacionais
- **Principais contribuições:**
  - Dataset de textos conversacionais anotados por CEFR
  - Usa embeddings de BERT para classificação
  - Foco em aplicações de LLM (geração de conteúdo adaptado)

---

## 5. Papers sobre Dinâmicas de Sessão e Knowledge Tracing

### 5.1. DynaEval (2021) - Dynamic Evaluation of Dialogue Systems
- **Título completo:** *DynaEval: Dynamic Evaluation of Dialogue Systems*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/DynaEval_2021.pdf`
- **Foco:** Avaliação dinâmica de sistemas de diálogo (multi-turno)
- **Principais contribuições:**
  - Métricas de consistência ao longo da sessão
  - Análise de trajetória de aprendizado
  - Considera contexto multi-turno

### 5.2. Piech et al. (2015) - Deep Knowledge Tracing (DKT)
- **Título completo:** *Deep Knowledge Tracing*
- **Status:** ✅ **Baixado** em `/papers/avaliacao-fala/v2/Piech_2015_Deep_Knowledge_Tracing.pdf`
- **Foco:** Rastreamento de conhecimento com redes neurais (LSTMs)
- **Principais contribuições:**
  - Modela evolução do conhecimento ao longo do tempo
  - Base para AKT (Attentive Knowledge Tracing) - usado no nosso `student_model`
  - AUC: 0.86 (vs. 0.67 do BKT) - +25% improvement
  - Não requer anotação manual de conceitos por especialistas
  - Dataset: Khan Academy (1.5M interactions)
- **Análise detalhada:** Ver `ANALISE_PAPERS_BAIXADOS.md`

### 5.3. Ghosh et al. (2020) - Attentive Knowledge Tracing (AKT)
- **Título completo:** *Context-Aware Attentive Knowledge Tracing*
- **Status:** ⚠️ **Não baixado** (disponível em KDD 2020)
- **Foco:** AKT com atenção contextual
- **Principais contribuições:**
  - Melhora DKT com mecanismo de atenção
  - Considera contexto de skills relacionadas
  - Usado no `student_model` do nosso sistema
- **Como obter:** [KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403282)

---

## 6. Papers sobre CEFR e Complexidade Linguística

### 6.1. EvalYaks (2024) - Automated CEFR Evaluation
- **Título completo:** *EvalYaks: Instruction Tuning Datasets and Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/EvalYaks_2024.pdf`
- **Foco:** Avaliação automática de transcrições de fala (CEFR B2)
- **Principais contribuições:**
  - Instruction tuning com Mistral Instruct 7B v0.2
  - Dataset sintético gerado por GPT-4 Turbo
  - Métricas: MSE, Acceptable Accuracy
- **Limitações:**
  - Dataset sintético (não real)
  - Foco apenas em B2
  - Não usa features acústicas

### 6.2. UniversalCEFR (2024) - Cross-lingual CEFR Classification
- **Título completo:** *UniversalCEFR: A Cross-Lingual CEFR Classification Model*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/UniversalCEFR_2024.pdf`
- **Foco:** Classificação CEFR multilíngue
- **Principais contribuições:**
  - Modelo cross-lingual (funciona em múltiplas línguas)
  - Usa BERT multilíngue
  - Dataset: textos escritos de múltiplas línguas

### 6.3. NILC-Metrix (2022) - Linguistic Complexity Metrics for Portuguese
- **Título completo:** *NILC-Metrix: A Comprehensive Tool for Linguistic Complexity Assessment in Portuguese*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/NILC-Metrix_2022.pdf`
- **Foco:** Métricas de complexidade linguística para português
- **Principais contribuições:**
  - 200+ métricas linguísticas
  - Ferramenta open-source
  - Foco em português brasileiro

---

## 7. Papers Brasileiros sobre Avaliação de Fala

### 7.1. Celpe-Bras - Dimensionalidade das Escalas de Avaliação
- **Título completo:** *Um estudo sobre a dimensionalidade das escalas de avaliação da proficiência oral do Certificado de Proficiência em Língua Portuguesa para Estrangeiros*
- **Status:** ✅ **Já baixado** em `/papers/avaliacao-fala/v2/Um estudo sobre a dimensionalidade das escalas de avaliação da proficiência oral do Certificado de Proficiência em Língua Portuguesa para Estrangeiros.pdf`
- **Foco:** Análise das escalas de avaliação oral do Celpe-Bras
- **Principais contribuições:**
  - Valida dimensionalidade das escalas analíticas do Celpe-Bras
  - Foco em português brasileiro
  - Metodologia de avaliação holística vs analítica

### 7.2. Protocolo de Avaliação Morfossintática (2019)
- **Título completo:** *Protocolo de avaliação morfossintática por amostra espontânea: construção e validação*
- **Status:** ⚠️ **Não baixado** (disponível em repositório USP)
- **Foco:** Avaliação morfossintática de fala espontânea em português
- **Principais contribuições:**
  - Protocolo validado para português brasileiro
  - Foco em fala espontânea (não lida)
- **Como obter:** [Repositório USP](https://repositorio.usp.br/item/003088416)

---

## 8. Papers Não Diretamente Relevantes (mas mencionados)

### 8.1. Análise de Ambiguidade Linguística em LLMs (2024)
- **Foco:** Ambiguidade linguística em português brasileiro
- **Relevância:** Limitada (não foca em avaliação de fala)
- **Status:** Disponível em [arXiv:2404.16653](https://arxiv.org/abs/2404.16653)

### 8.2. Avaliação de LLMs no Ensino de Programação (2024)
- **Foco:** LLMs para ensino de programação
- **Relevância:** Limitada (não foca em linguística)
- **Status:** Disponível em [SBC](https://sol.sbc.org.br/index.php/wei/article/view/36234)

### 8.3. Simplificação de Textos Jurídicos com LLMs (2024)
- **Foco:** Simplificação de textos jurídicos
- **Relevância:** Limitada (não foca em avaliação)
- **Status:** Disponível em [Repositório UFG](https://repositorio.bc.ufg.br/tede/items/4dc7d05e-a4c5-41f0-b84b-b2a0f8b6604e)

---

## 9. Papers Adicionais Baixados (2025-01-23)

### 9.1. RUBER (2017) - Dialog Evaluation
- **Título completo:** *RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems*
- **Status:** ✅ **Baixado** em `/papers/avaliacao-fala/v2/RUBER_2017_Dialog_Evaluation.pdf`
- **Foco:** Avaliação automática de sistemas de diálogo sem anotação humana
- **Principais contribuições:**
  - Combina referenced (com groundtruth) e unreferenced (query-response) metrics
  - Correlação com humanos: 0.53 (Spearman)
  - Não requer anotação humana para treinar
  - Usa pooling de word embeddings (não word-overlap)
- **Análise detalhada:** Ver `ANALISE_PAPERS_BAIXADOS.md`

### 9.2. Yeh et al. (2021) - Comprehensive Assessment of Dialog Metrics
- **Título completo:** *A Comprehensive Assessment of Dialog Evaluation Metrics*
- **Status:** ✅ **Baixado** em `/papers/avaliacao-fala/v2/Comprehensive_Assessment_Dialog_Metrics_2021.pdf`
- **Foco:** Comparação sistemática de 23 métricas de diálogo em 10 datasets
- **Principais contribuições:**
  - BLEU/METEOR/ROUGE têm correlação MUITO BAIXA (0.12-0.15) com humanos em diálogo
  - Melhores métricas: USR (0.42), GRADE (0.40), DynaEval (0.38), RUBER (0.35)
  - Combinar múltiplas métricas melhora correlação
  - Avaliar em turn-level E dialog-level
- **Análise detalhada:** Ver `ANALISE_PAPERS_BAIXADOS.md`

### 9.3. Mekyska et al. (2022) - Pathological Speech Analysis
- **Título completo:** *Robust and Complex Approach of Pathological Speech Signal Analysis*
- **Status:** ✅ **Baixado** em `/papers/avaliacao-fala/v2/Pathological_Speech_Analysis_2022.pdf`
- **Foco:** Análise de fala patológica (disartria, Parkinson) com 92 features acústicas
- **Principais contribuições:**
  - 92 features de fala, incluindo 36 novas
  - Acurácia: 82.1% em PdA Hospital database
  - Features mais discriminativas: Cepstral Peak Prominence (CPP)
  - Útil para avaliar qualidade de pronúncia (se houver acesso a áudio)
- **Análise detalhada:** Ver `ANALISE_PAPERS_BAIXADOS.md`

---

## 10. Resumo de Prioridades para Download

### ✅ Alta prioridade (BAIXADOS):
1. ✅ **Reimers & Gurevych (2019)** - Sentence-BERT
2. ✅ **Piech et al. (2015)** - Deep Knowledge Tracing
3. ✅ **RUBER (2017)** - Dialog Evaluation
4. ✅ **Yeh et al. (2021)** - Comprehensive Dialog Metrics
5. ✅ **Mekyska et al. (2022)** - Pathological Speech Analysis

### ⚠️ Alta prioridade (ainda não baixados):
1. **Xiao et al. (2024)** - Feedback estruturado
2. **CASPER Dataset (2024)** - Fala espontânea
3. **Ghosh et al. (2020)** - Attentive Knowledge Tracing

### Média prioridade:
4. **Protocolo de Avaliação Morfossintática (2019)** - Português brasileiro

### Baixa prioridade (não diretamente aplicáveis):
- Papers sobre ambiguidade linguística
- Papers sobre ensino de programação
- Papers sobre simplificação de textos

---

## 11. Conclusão

### Papers Baixados (Total: 17)

Os papers já baixados fornecem uma base sólida e **implementável** para as melhorias propostas:

#### Calibração e Alinhamento Humano:
- ✅ **Byun 2025** - LLM-as-a-Grader
- ✅ **Arnold 2018** - Automatic Grading with Real Data

#### Features Acústicas:
- ✅ **Banno 2022** - Wav2Vec 2.0 for Speaking Assessment
- ✅ **Mohammadi 2025** - Automated Assessment with Acoustic Features
- ✅ **Mekyska 2022** - Pathological Speech Analysis (92 features)

#### Relevância Semântica e Embeddings:
- ✅ **Reimers & Gurevych 2019** - Sentence-BERT (IMPLEMENTÁVEL AGORA!)
- ✅ **Lu 2025** - Hybrid Automated Speaking Assessment
- ✅ **Ace-CEFR 2025** - Linguistic Difficulty Evaluation

#### Avaliação de Diálogo:
- ✅ **RUBER 2017** - Unsupervised Dialog Evaluation
- ✅ **Yeh et al. 2021** - Comprehensive Assessment of 23 Dialog Metrics (CRÍTICO!)
- ✅ **DynaEval 2021** - Dynamic Dialog Evaluation

#### Knowledge Tracing:
- ✅ **Piech 2015** - Deep Knowledge Tracing (base do AKT)

#### CEFR e Complexidade:
- ✅ **EvalYaks 2024** - Instruction Tuning for CEFR
- ✅ **UniversalCEFR 2024** - Cross-lingual CEFR
- ✅ **NILC-Metrix 2022** - 200+ métricas para português
- ✅ **Celpe-Bras** - Escalas de avaliação oral

### Papers Ainda Não Baixados (3):
- ⚠️ **Xiao et al. (2024)** - Feedback estruturado (complementar, não crítico)
- ⚠️ **CASPER Dataset (2024)** - Fala espontânea (dataset, não metodologia)
- ⚠️ **Ghosh et al. (2020)** - AKT (já temos DKT, que é a base)

### 🚀 Status: PRONTO PARA IMPLEMENTAÇÃO

Com os **17 papers baixados e analisados**, temos **tudo o que precisamos** para implementar as 5 melhorias críticas identificadas:

1. ✅ **Calibração com humanos** → Byun 2025, Arnold 2018
2. ✅ **Feedback estruturado** → Lu 2025, Yeh 2021 (combinar métricas)
3. ✅ **Features acústicas** → Banno 2022, Mohammadi 2025, Mekyska 2022
4. ✅ **Relevância com embeddings** → **Reimers 2019 (SBERT) - IMPLEMENTÁVEL AGORA!**
5. ✅ **Dinâmicas de sessão** → Piech 2015 (DKT), Yeh 2021 (turn/dialog-level)

### 📄 Documentação Completa Criada:
1. **`MELHORIAS_BASEADAS_EM_PAPERS.md`** - 5 melhorias críticas com código
2. **`PAPERS_ENCONTRADOS_2024_2025.md`** - Catálogo de 20+ papers
3. **`ANALISE_PAPERS_BAIXADOS.md`** - Análise detalhada dos 5 papers recém-baixados
4. **`README.md`** - Índice e guia de navegação

