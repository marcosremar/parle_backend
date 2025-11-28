# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:22
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Olá, professor(a)! Estou bem, obrigado. Pode perguntar. Gosto muito de falar sobre a cultura do Brasil. Sim, já participei de uma festa junina ano passado. Foi muito divertido, embora eu não entendesse tudo. A comida era deliciosa, especialmente o bolo de milho. E as danças eram muito animadas, mas eu não consegui dançar direito. As pessoas estavam muito felizes. Sim, eu acho que sim. As pessoas cantavam e riam muito. Se eu tivesse mais tempo, gostaria de participar de outra festa junina. Sim, eu concordo. Foi uma experiência muito boa para entender mais sobre o Brasil. Quero conhecer outros festivais também.
```

## Nível CEFR Identificado
**Nível:** B1  
**Confiança:** 47%  
**Nível Esperado:** B1  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: B1 (confiança: 90%)
- Justificativa: O texto demonstra características consistentes com o nível B1. Apresenta um comprimento médio de frase que se alinha com B1, uso de diferentes tempos verbais (presente, pretérito perfeito, imperfeito)...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 5.2 palavras
- Subordinação: 15.0% das frases
- Type-Token Ratio: 0.592
- Comprimento médio de palavras: 4.8 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 89.06
- MATTR: 0.787
- Zipf-TTR: 0.310
- Hapax legomena: 76.1%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 28.27
- Frazier depth: 2.53
- T-units: 14
- Subordination index: 0.286

**Features de Fala (Phase 2):**
- Mean Word Span: 1.42
- Repetition rate: 0.020
- Disfluency rate: 0.000

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- A2: 74%
- A1: 67%
- B2: 63%
- C1: 43%
- C2: 32%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.866
- Complexity trend: -0.000
- Number of windows: 53

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
⚠️ Inconsistência: AKT sugere A1 mas texto indica B1
- AKT nível estimado: A1
- Progresso no nível B1: 0%
- Convergência: very_low
- Ajuste de confiança: -15.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: B1 (confiança: 47%)


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
