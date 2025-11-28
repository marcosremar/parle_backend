# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:00
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Olá. Eu uhm, fome. Sim. Eu quero... comida. Uhm, arroz. E carne. Água. Sem gás. Por favor.
```

## Nível CEFR Identificado
**Nível:** A1  
**Confiança:** 68%  
**Nível Esperado:** A1  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: A1 (confiança: 95%)
- Justificativa: O texto se encaixa perfeitamente nos critérios de A1. As frases são extremamente curtas, a estrutura sintática é mínima e fragmentada, o vocabulário é básico e essencial para necessidades imediatas, e...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 1.3 palavras
- Subordinação: 0.0% das frases
- Type-Token Ratio: 0.750
- Comprimento médio de palavras: 3.5 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 40.46
- MATTR: 0.882
- Zipf-TTR: 0.689
- Hapax legomena: 86.7%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 4.00
- Frazier depth: 1.22
- T-units: 9
- Subordination index: 0.000

**Features de Fala (Phase 2):**
- Mean Word Span: 1.46
- Repetition rate: 0.118
- Disfluency rate: 0.000

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- A1: 83%
- A2: 62%
- B1: 51%
- B2: 27%
- C1: 24%
- C2: 21%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.230
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
- Nível final: A1 (confiança: 68%)


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
