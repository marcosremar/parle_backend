# Papers da Interspeech 2024

## 📊 Visão Geral

Este diretório contém **2 papers da Interspeech 2024** baixados e convertidos para Markdown.

**Interspeech** é a principal conferência internacional em processamento de fala, organizada pela ISCA (International Speech Communication Association).

---

## 📄 Papers Baixados

### 1. Acoustic Feature Mixup (Interspeech 2024)

**Arquivo:** `Acoustic_Feature_Mixup_2024.pdf` (492KB) + `.md`

**Título:** *Acoustic Feature Mixup for Balanced Multi-aspect Pronunciation Assessment*

**Autores:** Heejin Do, Wonjun Lee, Gary Geunbae Lee (POSTECH, South Korea)

**Resumo:**
- Propõe **Acoustic Feature Mixup** para lidar com dados desbalanceados
- Duas estratégias: Static AM (linear) e Dynamic AM (não-linear)
- Usa **GOP features** + **Error-Rate features** (ASR vs. resposta esperada)
- **Resultados:** +29% em aspectos desbalanceados (Stress, Completeness)

**Aplicação no `speech_grader`:**
- Data augmentation durante treinamento do wav2vec 2.0
- Error-rate features para detecção direta de erros
- Melhorar performance em níveis CEFR desbalanceados (A1, C2)

**Link:** [arXiv:2406.15723](https://arxiv.org/abs/2406.15723)

---

### 2. Wav2Vec2.0 for Children with Cochlear Implants (Interspeech 2024)

**Arquivo:** `Wav2Vec_Cochlear_Implants_2024.pdf` (264KB) + `.md`

**Título:** *Automatic Assessment of Speech Production Skills for Children with Cochlear Implants Using Wav2Vec2.0 Acoustic Embeddings*

**Autores:** Seonwoo Lee, Sunhee Kim, Minhwa Chung (Seoul National University)

**Resumo:**
- Usa **múltiplos modelos Wav2Vec2.0** (adultos + crianças)
- Combina embeddings com **Multi-Head Attention**
- Inclui **Phoneme Embeddings** como referência
- **Resultados:** +51% vs. baseline (PCC = 0.63)

**Aplicação no `speech_grader`:**
- Treinar múltiplos modelos wav2vec para diferentes populações (nativos + L2)
- Multi-head attention para fusão de embeddings
- Phoneme embeddings como ground truth

**Link:** [ISCA Archive](https://www.isca-archive.org/interspeech_2024/lee24e_interspeech.pdf)

---

## 📊 Comparação

| Aspecto | Paper 1 (Mixup) | Paper 2 (Multi-Embedding) |
|---------|-----------------|---------------------------|
| **Problema** | Dados desbalanceados | Populações específicas |
| **Solução** | Data augmentation | Múltiplos modelos |
| **Técnica** | Interpolação de features | Multi-head attention |
| **Melhoria** | +29% (desbalanceados) | +51% (vs. baseline) |
| **Complexidade** | Baixa | Média |
| **Prioridade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 Impacto Esperado no `speech_grader`

### Antes (Implementação Atual)
- wav2vec 2.0 único: PCC = 0.75
- Sem data augmentation
- Sem error-rate features

### Depois (Com Papers Interspeech 2024)
- **Múltiplos wav2vec + Multi-Head Attention:** +9%
- **Acoustic Feature Mixup:** +29% em aspectos desbalanceados
- **Error-Rate Features:** Detecção direta de erros

**Impacto Total Estimado:** PCC = **0.85-0.88** 🎯

---

## 📚 Documentação Completa

Para análise detalhada com código de exemplo:
- `/src/services/diagnostic_module/docs/ANALISE_PAPERS_INTERSPEECH_2024.md`

Para lista completa de papers da Interspeech:
- `/src/services/diagnostic_module/docs/PAPERS_INTERSPEECH_RELEVANTES.md`

---

## 🔗 Links Úteis

**Interspeech:**
- [Interspeech 2024](https://interspeech2024.org/)
- [ISCA Archive](https://www.isca-archive.org/)

**Papers:**
- [Acoustic Feature Mixup (arXiv)](https://arxiv.org/abs/2406.15723)
- [Wav2Vec Cochlear Implants (ISCA)](https://www.isca-archive.org/interspeech_2024/lee24e_interspeech.pdf)

---

## ✅ Status

- ✅ Papers baixados (2/2)
- ✅ Convertidos para Markdown (2/2)
- ✅ Análise completa realizada
- ✅ Código de exemplo criado
- ⏳ Implementação pendente

**Data de Download:** 23 de Janeiro de 2025

