# 🎤 Papers da Interspeech Relevantes para o `speech_grader`

## Objetivo

Este documento lista papers da conferência **Interspeech** (2021-2024) que podem contribuir para melhorar a metodologia e implementação do `speech_grader`.

**Interspeech** é a principal conferência internacional em processamento de fala, organizada pela ISCA (International Speech Communication Association).

---

## 📊 Papers Identificados (12 papers)

### 🥇 Alta Relevância (Implementação Direta)

#### 1. **SpeechBERTScore** (Interspeech 2024)

**Título:** *SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech Generation Leveraging NLP Evaluation Metrics*

**Autores:** Não especificado nas buscas

**Ano:** 2024

**Contribuição:**
- Métrica automática que combina representações de **fala + texto**
- Usa BERTScore (similar ao SBERT) para avaliar qualidade de fala
- Reference-aware (com groundtruth)

**Aplicação no `speech_grader`:**
- Avaliar qualidade de fala gerada pelo aluno
- Combinar avaliação acústica (wav2vec) + textual (SBERT)
- Métrica híbrida para scoring

**Link:** [ResearchGate](https://www.researchgate.net/publication/383652976_SpeechBERTScore_Reference-Aware_Automatic_Evaluation_of_Speech_Generation_Leveraging_NLP_Evaluation_Metrics)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐⭐⭐ MUITO ALTA

---

#### 2. **AlignNet** (Interspeech 2024)

**Título:** *AlignNet: Learning Dataset Score Alignment Functions to Enable Better Training of Speech Quality Estimators*

**Autores:** Não especificado

**Ano:** 2024

**Contribuição:**
- Aprende funções de **alinhamento de pontuações** entre diferentes datasets
- Permite usar dados mais diversos para treinamento
- Melhora estimadores de qualidade de fala

**Aplicação no `speech_grader`:**
- Alinhar scores de diferentes avaliadores humanos
- Calibração automática entre datasets
- Melhorar treinamento do modelo wav2vec

**Link:** [papers.cool](https://papers.cool/venue/INTERSPEECH.2024?group=Analysis+and+Assessment)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐⭐⭐ MUITO ALTA (relacionado à calibração)

---

#### 3. **On the Robustness of wav2vec 2.0 Based Speaker Recognition Systems** (Interspeech 2023)

**Título:** *On the Robustness of wav2vec 2.0 Based Speaker Recognition Systems*

**Autores:** Novoselov et al.

**Ano:** 2023

**Contribuição:**
- Investiga **robustez do wav2vec 2.0** em diferentes domínios
- Testa em telefone, microfone, canais cruzados
- Recomenda **data augmentation** durante fine-tuning

**Aplicação no `speech_grader`:**
- Melhorar robustez do modelo wav2vec
- Técnicas de data augmentation
- Generalização em condições variáveis

**Link:** [ISCA Archive](https://www.isca-archive.org/interspeech_2023/novoselov23_interspeech.html)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐⭐ ALTA

---

#### 4. **Deep LSTM Spoken Term Detection Using Wav2Vec 2.0 Recognizer** (Interspeech 2022)

**Título:** *Deep LSTM Spoken Term Detection Using Wav2Vec 2.0 Recognizer*

**Autores:** Švec et al.

**Ano:** 2022

**Contribuição:**
- Combina **wav2vec 2.0 + LSTM profunda**
- Detecção de termos falados em grandes documentos
- Supera sistemas DNN-HMM anteriores

**Aplicação no `speech_grader`:**
- Detectar palavras-chave específicas (marcadores de nível CEFR)
- Identificar estruturas gramaticais na fala
- Análise de vocabulário

**Link:** [ISCA Archive](https://www.isca-archive.org/interspeech_2022/svec22_interspeech.html)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐ MÉDIA

---

### 🥈 Média Relevância (Insights Metodológicos)

#### 5. **Exploring Wav2vec 2.0 on Speaker Verification and Language Identification** (Interspeech 2021)

**Título:** *Exploring Wav2vec 2.0 on Speaker Verification and Language Identification*

**Autores:** Fan et al.

**Ano:** 2021

**Contribuição:**
- wav2vec 2.0 para **verificação de locutor** e **identificação de idioma**
- EER = 3.61% em VoxCeleb1 (speaker verification)
- EER = 3.47% em AP17-OLR (language ID)

**Aplicação no `speech_grader`:**
- Identificar sotaque/variante do português (BR vs. PT)
- Verificar consistência do locutor (anti-fraude)
- Adaptar modelo para variantes linguísticas

**Link:** [ISCA Archive](https://www.isca-archive.org/interspeech_2021/fan21_interspeech.html)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐ MÉDIA

---

#### 6. **Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings** (Interspeech 2021)

**Título:** *Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings*

**Autores:** Pepino et al.

**Ano:** 2021

**Contribuição:**
- Reconhecimento de **emoções** usando wav2vec 2.0
- Implementação oficial disponível no GitHub
- Embeddings capturam informações prosódicas

**Aplicação no `speech_grader`:**
- Avaliar **prosódia emocional** (entonação, ênfase)
- Detectar confiança vs. hesitação na fala
- Análise de expressividade

**Link:** [GitHub](https://github.com/habla-liaa/ser-with-w2v2)

**Status:** ⚠️ Não baixado (mas código disponível)

**Prioridade:** ⭐⭐⭐ MÉDIA

---

#### 7. **Data Augmentation using Prosody and False Starts to Recognize Non-Native Children's Speech** (Interspeech 2020)

**Título:** *Data Augmentation using Prosody and False Starts to Recognize Non-Native Children's Speech*

**Autores:** Não especificado

**Ano:** 2020

**Contribuição:**
- Técnicas de **data augmentation** para fala não nativa
- Uso de **prosódia** e **falsos começos**
- Melhora reconhecimento de fala infantil

**Aplicação no `speech_grader`:**
- Data augmentation para treinamento do wav2vec
- Lidar com hesitações e falsos começos (característicos de L2)
- Melhorar robustez para fala não nativa

**Link:** [arXiv](https://arxiv.org/abs/2008.12914)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐⭐⭐ ALTA

---

### 🥉 Baixa Relevância (Contexto Geral)

#### 8. **Alzheimer's Dementia Recognition through Spontaneous Speech: The ADReSS Challenge** (Interspeech 2020)

**Título:** *Alzheimer's Dementia Recognition through Spontaneous Speech: The ADReSS Challenge*

**Autores:** Não especificado

**Ano:** 2020

**Contribuição:**
- Detecção de Alzheimer via **fala espontânea**
- Dataset balanceado + metodologias padronizadas
- Classificação + regressão de scores neuropsicológicos

**Aplicação no `speech_grader`:**
- Metodologias para análise de fala espontânea
- Benchmarks padronizados
- Técnicas de classificação + regressão

**Link:** [arXiv](https://arxiv.org/abs/2004.06833)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐ BAIXA (contexto clínico, não educacional)

---

#### 9. **The INTERSPEECH 2020 Deep Noise Suppression Challenge** (Interspeech 2020)

**Título:** *The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results*

**Autores:** Não especificado

**Ano:** 2020

**Contribuição:**
- **Supressão de ruído** em tempo real
- Datasets + framework de teste subjetivo
- Melhora qualidade perceptiva da fala

**Aplicação no `speech_grader`:**
- Pré-processamento de áudio ruidoso
- Melhorar robustez em ambientes não controlados
- Técnicas de limpeza de áudio

**Link:** [arXiv](https://arxiv.org/abs/2005.13981)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐ BAIXA (pré-processamento, não avaliação)

---

#### 10. **Investigating the Impact of Speech Compression on the Acoustics of Dysarthric Speech** (Interspeech 2022)

**Título:** *Investigating the Impact of Speech Compression on the Acoustics of Dysarthric Speech*

**Autores:** Não especificado

**Ano:** 2022

**Contribuição:**
- Impacto da **compressão de áudio** em fala disártrica
- Preservação de características acústicas críticas
- Análise de codecs de áudio

**Aplicação no `speech_grader`:**
- Escolher codec adequado para armazenamento
- Garantir preservação de features acústicas
- Otimizar armazenamento sem perder qualidade

**Link:** [papers.cool](https://papers.cool/venue/INTERSPEECH.2022?group=Analysis+and+Assessment)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐ MUITO BAIXA (otimização técnica)

---

#### 11. **Zero-Shot Cross-Lingual Aphasia Detection Using Automatic Speech Recognition** (Interspeech 2022)

**Título:** *Zero-Shot Cross-Lingual Aphasia Detection Using Automatic Speech Recognition*

**Autores:** Não especificado

**Ano:** 2022

**Contribuição:**
- Detecção de **afasia** em múltiplos idiomas
- **Zero-shot** (sem dados anotados por idioma)
- Usa ASR pré-treinados

**Aplicação no `speech_grader`:**
- Técnicas de transfer learning cross-lingual
- Adaptação para português sem dados massivos
- Detecção de fala comprometida

**Link:** [papers.cool](https://papers.cool/venue/INTERSPEECH.2022?group=Analysis+and+Assessment)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐⭐ BAIXA (contexto clínico)

---

#### 12. **Interspeech Pathology Challenge: Investigations into Speaker and Sentence Specific Effects** (Interspeech 2012)

**Título:** *Interspeech Pathology Challenge: Investigations into Speaker and Sentence Specific Effects*

**Autores:** Não especificado

**Ano:** 2012

**Contribuição:**
- Efeitos específicos de **falante** e **sentença** em patologias
- Importância de considerar variações individuais
- Metodologias de coleta de dados

**Aplicação no `speech_grader`:**
- Considerar variações individuais na avaliação
- Normalização por falante
- Design de tarefas de avaliação

**Link:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8043657/)

**Status:** ⚠️ Não baixado

**Prioridade:** ⭐ MUITO BAIXA (muito antigo, contexto clínico)

---

## 📊 Resumo por Prioridade

### ⭐⭐⭐⭐⭐ Muito Alta (Implementar Agora)
1. **SpeechBERTScore** (2024) - Métrica híbrida fala+texto
2. **AlignNet** (2024) - Alinhamento de scores para calibração

### ⭐⭐⭐⭐ Alta (Próximas Sprints)
3. **Robustness of wav2vec 2.0** (2023) - Data augmentation
4. **Data Augmentation for Non-Native Speech** (2020) - Prosódia + falsos começos

### ⭐⭐⭐ Média (Futuro)
5. **Deep LSTM + wav2vec** (2022) - Detecção de termos
6. **Speaker Verification + Language ID** (2021) - Identificação de variantes
7. **Emotion Recognition** (2021) - Prosódia emocional

### ⭐⭐ Baixa (Contexto)
8. **ADReSS Challenge** (2020) - Fala espontânea (contexto clínico)
9. **Deep Noise Suppression** (2020) - Pré-processamento
10. **Zero-Shot Aphasia Detection** (2022) - Transfer learning
11. **Speech Compression** (2022) - Otimização técnica

### ⭐ Muito Baixa (Não prioritário)
12. **Pathology Challenge** (2012) - Muito antigo

---

## 🎯 Recomendações de Implementação

### Fase 1 (Imediata) - Papers 2024
1. **Baixar e analisar SpeechBERTScore**
   - Implementar métrica híbrida fala+texto
   - Combinar com SBERT + wav2vec 2.0
   - Avaliar qualidade de fala de forma holística

2. **Baixar e analisar AlignNet**
   - Implementar alinhamento de scores
   - Melhorar calibração com humanos
   - Usar dados de múltiplos avaliadores

### Fase 2 (Próximas Sprints) - Robustez
3. **Implementar data augmentation** (Novoselov 2023 + paper 2020)
   - Prosódia variada
   - Falsos começos
   - Ruído de fundo
   - Variações de velocidade

### Fase 3 (Futuro) - Features Avançadas
4. **Explorar detecção de termos** (Švec 2022)
   - Identificar marcadores de nível CEFR
   - Detectar estruturas gramaticais específicas

5. **Adicionar análise de emoção/prosódia** (Pepino 2021)
   - Avaliar expressividade
   - Detectar confiança vs. hesitação

---

## 📥 Próximos Passos

### 1. Baixar Papers Prioritários
```bash
cd /Users/marcos/Documents/projects/backend/parle_backend/papers/avaliacao-fala/v2

# SpeechBERTScore (2024)
# AlignNet (2024)
# Robustness of wav2vec 2.0 (2023)
# Data Augmentation for Non-Native Speech (2020)
```

### 2. Converter para Markdown
```bash
python -c "
import pdf4llm

papers = [
    'SpeechBERTScore_2024.pdf',
    'AlignNet_2024.pdf',
    'Robustness_wav2vec_2023.pdf',
    'Data_Augmentation_NonNative_2020.pdf'
]

for pdf in papers:
    md = pdf4llm.to_markdown(pdf)
    with open(pdf.replace('.pdf', '.md'), 'w') as f:
        f.write(md)
"
```

### 3. Analisar e Documentar
- Criar `ANALISE_PAPERS_INTERSPEECH.md`
- Extrair insights práticos
- Propor implementações

### 4. Atualizar Documentação
- Adicionar referências em `IMPLEMENTACAO.md`
- Atualizar `RASTREABILIDADE_PAPERS.md`
- Atualizar `PAPERS_ENCONTRADOS_2024_2025.md`

---

## 🔗 Links Úteis

**Interspeech Archive:**
- [ISCA Archive](https://www.isca-archive.org/)
- [Interspeech 2024](https://interspeech2024.org/)
- [Interspeech 2023](https://interspeech2023.org/)

**Papers por Tópico:**
- [Analysis and Assessment (2024)](https://papers.cool/venue/INTERSPEECH.2024?group=Analysis+and+Assessment)
- [Analysis and Assessment (2022)](https://papers.cool/venue/INTERSPEECH.2022?group=Analysis+and+Assessment)

---

## ✅ Conclusão

Identificamos **12 papers da Interspeech** (2012-2024) relevantes para o `speech_grader`, com **2 papers de 2024** de **prioridade muito alta**:

1. **SpeechBERTScore** - Métrica híbrida estado-da-arte
2. **AlignNet** - Alinhamento de scores para calibração

Esses papers complementam perfeitamente a implementação atual baseada em **Banno et al. (2022)** e podem elevar o sistema ao **estado-da-arte absoluto** em avaliação automática de fala.

**Próximo passo:** Baixar e analisar SpeechBERTScore e AlignNet! 🚀

