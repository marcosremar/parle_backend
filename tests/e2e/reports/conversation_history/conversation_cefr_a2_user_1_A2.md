# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:05
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Oi! Eu descansei muito. Eu assisti filmes quando choveu. Assisti um filme de comédia. Era engraçado. Sim, eu gosto, porque é divertido.
```

## Nível CEFR Identificado
**Nível:** A2  
**Confiança:** 51%  
**Nível Esperado:** A2  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: A2 (confiança: 90%)
- Justificativa: O texto apresenta características que o classificam como A2. Ele possui frases com comprimento médio de 6-10 palavras, utiliza o pretérito perfeito ('descansei', 'assisti', 'choveu', 'era') e inclui o...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 2.8 palavras
- Subordinação: 25.0% das frases
- Type-Token Ratio: 0.734
- Comprimento médio de palavras: 4.8 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 45.17
- MATTR: 0.864
- Zipf-TTR: 0.587
- Hapax legomena: 89.5%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 10.67
- Frazier depth: 1.67
- T-units: 6
- Subordination index: 0.333

**Features de Fala (Phase 2):**
- Mean Word Span: 1.58
- Repetition rate: 0.091
- Disfluency rate: 0.091

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- A2: 87%
- A1: 76%
- B2: 44%
- C1: 34%
- C2: 29%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.447
- Complexity trend: 0.000
- Number of windows: 1

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
AKT sugere A1 mas texto indica A2 (níveis adjacentes)
- AKT nível estimado: A1
- Progresso no nível A2: 0%
- Convergência: low
- Ajuste de confiança: -5.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: A2 (confiança: 51%)


## Análise Detalhada

### Análise Sintática

*Análise sintática não disponível*

### Análise Lexical

*Análise lexical não disponível*

### Análise Discursiva

*Análise discursiva não disponível*

## Indicadores-Chave do Nível Identificado

*Nenhum indicador-chave fornecido*


## Metodologia

Esta análise foi realizada usando o modelo de linguagem **Gemini Flash 2.5** (via OpenRouter), que foi instruído com critérios CEFR baseados nos seguintes papers acadêmicos:

1. **Leal, S. E., et al. (2022).** NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese.
2. **Vajjala, S., & Rama, T. (2021).** Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs.
3. **Arnold, T., et al. (2018).** Predicting CEFRL levels in learner English on the basis of metrics and full texts.
4. **Ribeiro, E., et al. (2024).** Avaliação automática do nível de complexidade de textos em português europeu.

O LLM analisa o texto considerando:
- Complexidade sintática (subordinação, tempos verbais, estruturas complexas)
- Riqueza lexical (vocabulário, diversidade, comprimento de palavras)
- Coesão discursiva (conectores, marcadores, progressão textual)

Para mais detalhes sobre os critérios, consulte: `docs/CEFR_COMPLEXITY_MARKERS_IMPLEMENTATION.md`
