# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:32:00
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Professor, tenho ponderado bastante sobre a polarização política contemporânea e suas implicações para a democracia representativa. Particularmente, questiono como a fragmentação do discurso impede o consenso sobre problemas urgentes, como as mudanças climáticas, por exemplo. Acredito que a ascensão das redes sociais exacerbou essa fragmentação, criando bolhas ideológicas praticamente impenetráveis, em grande medida. Paradoxalmente, essa 'conexão' parece ter nos distanciado da capacidade de dialogar construtivamente com quem pensa divergentemente. Como poderíamos, então, fomentar um espaço de deliberação genuína, onde o respeito preceda a intransigência ideológica? Talvez a educação cívica, focada na escuta ativa e na empatia, possa desempenhar um papel fundamental desde as fases iniciais da formação. Não obstante as dificuldades inerentes, acredito na capacidade humana de transcender essas divisões e buscar soluções colaborativas.
```

## Nível CEFR Identificado
**Nível:** C1  
**Confiança:** 40%  
**Nível Esperado:** C1  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: C1 (confiança: 95%)
- Justificativa: O texto demonstra um nível C1 devido à sua complexidade sintática e lexical, uso sofisticado de conectores e marcadores discursivos, e a abordagem de temas abstratos com profundidade. As frases são lo...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 13.6 palavras
- Subordinação: 0.0% das frases
- Type-Token Ratio: 0.693
- Comprimento médio de palavras: 6.5 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 187.19
- MATTR: 0.917
- Zipf-TTR: 0.374
- Hapax legomena: 86.1%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 122.57
- Frazier depth: 4.86
- T-units: 6
- Subordination index: 0.833

**Features de Fala (Phase 2):**
- Mean Word Span: 1.48
- Repetition rate: 0.024
- Disfluency rate: 0.000

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- C2: 76%
- B2: 71%
- A1: 71%
- A2: 65%
- C1: 62%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.741
- Complexity trend: -0.001
- Number of windows: 75

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
⚠️ Inconsistência: AKT sugere A1 mas texto indica C1
- AKT nível estimado: A1
- Progresso no nível C1: 0%
- Convergência: very_low
- Ajuste de confiança: -25.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: C1 (confiança: 40%)


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
