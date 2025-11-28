# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:16
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Olá, professor! Foi bom, obrigado. Eu vi um filme novo, mas eu não gostei muito, para ser honesto. Vi um filme de ação. Achei a história um pouco confusa, embora os efeitos especiais fossem bons. Faltou emoção. Sim, eu gosto mais de drama ou comédia. Eu gosto quando os personagens têm histórias interessantes, sabe? Se o filme me faz pensar, é melhor. Difícil escolher um. Mas eu gosto muito de 'A Vida é Bela', porque ele mostra a esperança mesmo em tempos difíceis. É um filme muito emocionante. Obrigado! Eu acho que é importante ver filmes com mensagens positivas, mesmo que sejam tristes às vezes.
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
- Justificativa: O texto apresenta características consistentes com o nível B1. Há uma variedade de tempos verbais (presente, pretérito perfeito), subordinação presente com conectores como 'embora', 'quando', 'porque'...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 5.8 palavras
- Subordinação: 32.2% das frases
- Type-Token Ratio: 0.615
- Comprimento médio de palavras: 4.5 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 95.42
- MATTR: 0.826
- Zipf-TTR: 0.318
- Hapax legomena: 81.6%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 41.43
- Frazier depth: 2.71
- T-units: 13
- Subordination index: 0.462

**Features de Fala (Phase 2):**
- Mean Word Span: 1.65
- Repetition rate: 0.010
- Disfluency rate: 0.057

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- A2: 100%
- B1: 100%
- A1: 74%
- B2: 65%
- C2: 60%
- C1: 46%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.867
- Complexity trend: 0.000
- Number of windows: 56

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
