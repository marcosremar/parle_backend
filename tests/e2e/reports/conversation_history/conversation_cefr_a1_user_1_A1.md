# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:30:55
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Olá. Eu... uhm... [nome]. Meu nome é... [nome]. Eu sou... uhm... [país]. Sim. E você?
```

## Nível CEFR Identificado
**Nível:** A1  
**Confiança:** 69%  
**Nível Esperado:** A1  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: A1 (confiança: 100%)
- Justificativa: O texto se encaixa perfeitamente nos critérios de A1. As frases são extremamente curtas e simples, com estrutura SVO básica. O vocabulário é limitado às palavras mais frequentes para autoapresentação ...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 1.1 palavras
- Subordinação: 0.0% das frases
- Type-Token Ratio: 0.623
- Comprimento médio de palavras: 2.9 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 15.00
- MATTR: 0.733
- Zipf-TTR: 0.516
- Hapax legomena: 72.7%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 15.71
- Frazier depth: 1.57
- T-units: 7
- Subordination index: 0.000

**Features de Fala (Phase 2):**
- Mean Word Span: 1.56
- Repetition rate: 0.133
- Disfluency rate: 0.133

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- A1: 82%
- A2: 62%
- B1: 50%
- B2: 26%
- C1: 21%
- C2: 18%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.173
- Complexity trend: 0.000
- Number of windows: 1

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
AKT confirma nível A1 mas progresso baixo (0%)
- AKT nível estimado: A1
- Progresso no nível A1: 0%
- Convergência: moderate
- Ajuste de confiança: +5.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: A1 (confiança: 70%)


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
