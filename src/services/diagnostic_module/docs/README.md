# Documentação do `speech_grader`

## Visão Geral

O **`speech_grader`** é o serviço responsável pela avaliação automática de fala no sistema Parle. Ele combina:
- **LLMs** (Gemini 2.5 via OpenRouter) para análise qualitativa
- **Métricas linguísticas** (complexidade sintática, diversidade lexical, coerência)
- **AKT** (Attentive Knowledge Tracing) para validação cruzada com progresso do aluno
- **Descritores CEFR** e rubricas analíticas (Celpe-Bras, EvalYaks, Ace-CEFR)

---

## Documentos Disponíveis

### 1. 📘 [METODOLOGIA.md](./METODOLOGIA.md)
**Fundamentos teóricos e científicos do serviço**

Descreve:
- Base teórica (CEFR, Celpe-Bras, avaliação de fala)
- Referências bibliográficas completas (APA)
- Como cada paper contribui para a metodologia
- Abordagem multi-aspecto (fluência, gramática, vocabulário, conteúdo, interação)
- Integração de LLMs, métricas quantitativas e AKT

**Público-alvo:** Pesquisadores, linguistas, designers de sistema

---

### 2. 🛠️ [IMPLEMENTACAO.md](./IMPLEMENTACAO.md)
**Guia técnico de implementação**

Descreve:
- Arquitetura interna do serviço
- Fluxo de dados com outros microserviços (`orchestrator`, `student_model`, `linguistic_analysis`)
- Módulos internos e responsabilidades
- Formatos de requisição/resposta (JSON)
- Testes e validação

**Público-alvo:** Desenvolvedores, engenheiros de software

---

### 3. 🚀 [MELHORIAS_BASEADAS_EM_PAPERS.md](./MELHORIAS_BASEADAS_EM_PAPERS.md)
**Melhorias propostas para elevar o sistema ao estado-da-arte**

Identifica **5 lacunas críticas** e propõe soluções baseadas em papers recentes (2024-2025):

1. **Calibração com Avaliadores Humanos** (Prioridade 1)
   - Endpoint `/api/diagnostic/calibrate` para corrigir vieses sistemáticos do LLM
   - Baseado em: Byun et al. (2025), Arnold et al. (2018)

2. **Feedback Pedagógico Estruturado** (Prioridade 2)
   - Saída JSON com `strengths`, `weaknesses`, `next_steps`, `priority`
   - Baseado em: Xiao et al. (2024), Lu et al. (2025)

3. **Integração de Features Acústicas** (Prioridade 3)
   - Metadados ASR (taxa de fala, confiança, pausas)
   - wav2vec 2.0 para features acústicas profundas
   - Baseado em: Banno et al. (2022), Mohammadi et al. (2025)

4. **Relevância de Tarefa com Embeddings Semânticos** (Prioridade 4)
   - SBERT para similaridade com exemplar e cobertura de tópicos
   - Baseado em: Lu et al. (2025), Reimers & Gurevych (2019)

5. **Análise de Dinâmicas de Sessão** (Prioridade 5)
   - Métricas de consistência, trajetória, engajamento, anomalias
   - Baseado em: DynaEval (2021), DKT/AKT

**Público-alvo:** Desenvolvedores, product managers, pesquisadores

---

### 4. 📚 [PAPERS_ENCONTRADOS_2024_2025.md](./PAPERS_ENCONTRADOS_2024_2025.md)
**Catálogo de papers relevantes encontrados nas buscas**

Lista papers organizados por tema:
- Calibração e alinhamento humano
- Features acústicas e fala
- Feedback estruturado e explicabilidade
- Relevância semântica e embeddings
- Dinâmicas de sessão e knowledge tracing
- CEFR e complexidade linguística
- Papers brasileiros sobre avaliação de fala

Inclui:
- Status de download (✅ baixado, ⚠️ não baixado)
- Principais contribuições
- Limitações identificadas
- Como obter (links)

**Público-alvo:** Pesquisadores, desenvolvedores

---

### 5. 🎤 [PAPERS_INTERSPEECH_RELEVANTES.md](./PAPERS_INTERSPEECH_RELEVANTES.md)
**Papers da conferência Interspeech (2021-2024)**

Lista **12 papers da Interspeech** organizados por prioridade:
- ⭐⭐⭐⭐⭐ Muito Alta: SpeechBERTScore (2024), AlignNet (2024)
- ⭐⭐⭐⭐ Alta: Robustness of wav2vec 2.0 (2023), Data Augmentation (2020)
- ⭐⭐⭐ Média: Deep LSTM + wav2vec (2022), Speaker Verification (2021)
- ⭐⭐ Baixa: Papers de contexto clínico e pré-processamento

**Público-alvo:** Pesquisadores, desenvolvedores

---

### 6. 📊 [ANALISE_PAPERS_INTERSPEECH_2024.md](./ANALISE_PAPERS_INTERSPEECH_2024.md)
**Análise detalhada de 2 papers da Interspeech 2024**

Analisa em profundidade:
1. **Acoustic Feature Mixup** (Do et al., 2024)
   - Data augmentation para scores desbalanceados
   - Error-rate features (ASR vs. resposta esperada)
   - +29% de melhoria em aspectos desbalanceados

2. **Wav2Vec2.0 for Children with Cochlear Implants** (Lee et al., 2024)
   - Múltiplos modelos wav2vec para diferentes populações
   - Multi-head attention para fusão de embeddings
   - +51% de melhoria vs. baseline

Inclui:
- Código de exemplo para implementação
- Impacto esperado no `speech_grader`
- Próximos passos de implementação

**Público-alvo:** Desenvolvedores, pesquisadores

---

## Fluxo de Leitura Recomendado

### Para **Pesquisadores/Linguistas:**
1. [METODOLOGIA.md](./METODOLOGIA.md) → Entender fundamentos teóricos
2. [PAPERS_ENCONTRADOS_2024_2025.md](./PAPERS_ENCONTRADOS_2024_2025.md) → Explorar literatura
3. [PAPERS_INTERSPEECH_RELEVANTES.md](./PAPERS_INTERSPEECH_RELEVANTES.md) → Papers da Interspeech
4. [ANALISE_PAPERS_INTERSPEECH_2024.md](./ANALISE_PAPERS_INTERSPEECH_2024.md) → Análise detalhada
5. [MELHORIAS_BASEADAS_EM_PAPERS.md](./MELHORIAS_BASEADAS_EM_PAPERS.md) → Ver como papers informam melhorias

### Para **Desenvolvedores:**
1. [IMPLEMENTACAO.md](./IMPLEMENTACAO.md) → Entender arquitetura e fluxo de dados
2. [ANALISE_PAPERS_INTERSPEECH_2024.md](./ANALISE_PAPERS_INTERSPEECH_2024.md) → Código de exemplo
3. [MELHORIAS_BASEADAS_EM_PAPERS.md](./MELHORIAS_BASEADAS_EM_PAPERS.md) → Ver roadmap de melhorias
4. [METODOLOGIA.md](./METODOLOGIA.md) → Contexto teórico (opcional)

### Para **Product Managers:**
1. [MELHORIAS_BASEADAS_EM_PAPERS.md](./MELHORIAS_BASEADAS_EM_PAPERS.md) → Prioridades e impacto
2. [ANALISE_PAPERS_INTERSPEECH_2024.md](./ANALISE_PAPERS_INTERSPEECH_2024.md) → Estado-da-arte 2024
3. [IMPLEMENTACAO.md](./IMPLEMENTACAO.md) → Entender capacidades atuais
4. [METODOLOGIA.md](./METODOLOGIA.md) → Validação científica

---

## Status Atual (Novembro 2025)

### ✅ Implementado
- Análise CEFR multi-aspecto (A1-C2)
- Integração com AKT para validação cruzada
- Métricas linguísticas quantitativas (MTLD, subordinação, etc.)
- Análise gramatical com categorização de erros
- Relevância de tarefa (básica, via LLM)
- Testes E2E com conversas geradas e gravadas
- Precisão: **91.67%** (11/12 conversas corretas)

### 🔄 Em Planejamento (Roadmap)
- Calibração com avaliadores humanos
- Feedback estruturado (JSON)
- Features acústicas (metadados ASR + wav2vec 2.0)
- Relevância semântica com SBERT
- Análise de dinâmicas de sessão

---

## Referências Principais

### Papers Fundamentais (já baixados):
1. **Banno et al. (2022)** - Automated Speaking Assessment with Wav2Vec 2.0
2. **Lu et al. (2025)** - Hybrid Automated Speaking Assessment
3. **Byun et al. (2025)** - LLM-as-a-Grader
4. **Arnold et al. (2018)** - Automatic Grading with Real Data
5. **EvalYaks (2024)** - Instruction Tuning for CEFR Evaluation
6. **Ace-CEFR (2025)** - Automated Evaluation of Linguistic Difficulty
7. **Celpe-Bras** - Escalas de Avaliação Oral (Português Brasileiro)
8. **Do et al. (2024)** - Acoustic Feature Mixup (Interspeech 2024) ⭐ NOVO
9. **Lee et al. (2024)** - Wav2Vec2.0 Multi-Embedding (Interspeech 2024) ⭐ NOVO

### Papers Complementares (a baixar):
- Xiao et al. (2024) - Explainable Feedback
- Reimers & Gurevych (2019) - Sentence-BERT
- CASPER Dataset (2024) - Spontaneous Speech
- Piech et al. (2015) - Deep Knowledge Tracing
- Ghosh et al. (2020) - Attentive Knowledge Tracing

---

## Contato e Contribuições

Para dúvidas, sugestões ou contribuições:
- Abra uma issue no repositório
- Consulte a documentação principal do projeto: `/README.md`
- Consulte a documentação do sistema completo: `/docs/`

---

## Changelog

### 2025-01-23 ⭐ NOVO
- ✅ Baixados e analisados **2 papers da Interspeech 2024**
- ✅ **Acoustic Feature Mixup** (Do et al., 2024) - Data augmentation para scores desbalanceados
- ✅ **Wav2Vec Multi-Embedding** (Lee et al., 2024) - Múltiplos modelos + multi-head attention
- ✅ Criado [PAPERS_INTERSPEECH_RELEVANTES.md](./PAPERS_INTERSPEECH_RELEVANTES.md) com 12 papers
- ✅ Criado [ANALISE_PAPERS_INTERSPEECH_2024.md](./ANALISE_PAPERS_INTERSPEECH_2024.md) com código de exemplo
- ✅ Identificado impacto esperado: **+10-15% em PCC**

### 2025-01-XX
- ✅ Criada documentação completa do `speech_grader`
- ✅ Identificadas 5 melhorias críticas baseadas em papers 2024-2025
- ✅ Catalogados 20+ papers relevantes
- ✅ Definido roadmap de implementação (3 fases)

### 2025-01-XX (anterior)
- ✅ Implementado classificador híbrido CEFR (LLM + métricas)
- ✅ Integrado AKT para validação cruzada
- ✅ Alcançado 91.67% de precisão em testes E2E

