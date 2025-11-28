# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:46
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
De fato, professor. Mergulhei na *Fenomenologia do Espírito* e, embora a prosa de Hegel seja notoriamente densa, a arquitetura conceitual de sua progressão dialética revela uma inteligência assombrosa. A 'Aufhebung' transcende a mera aniquilação, não é mesmo? É uma preservação que eleva, uma abolição que simultaneamente resgata. A meu ver, a síntese kantiana opera predominantemente no reino epistêmico, um arranjo da experiência sensível pela estrutura inerente do entendimento. A 'Aufhebung' hegeliana, contudo, é uma dinâmica ontológica, uma força motriz imanente à própria realidade que impulsiona o desenvolvimento da consciência e do ser em direção à autoconsciência absoluta. Absolutamente. A maneira como teorias científicas são derrubadas e, no entanto, seus elementos válidos são incorporados e transformados em um novo paradigma, parece-me uma manifestação empírica da 'Aufhebung'. Não é uma substituição pura, mas uma reconfiguração que transcende as limitações anteriores, um processo notavelmente dialético.
```

## Nível CEFR Identificado
**Nível:** C2  
**Confiança:** 47%  
**Nível Esperado:** C2  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: C2 (confiança: 98%)
- Justificativa: O texto exibe características robustas de um nível C2. A sintaxe é extremamente complexa, com múltiplas subordinações encadeadas, uso sofisticado de tempos verbais e subjuntivo em contextos complexos....

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 12.1 palavras
- Subordinação: 16.7% das frases
- Type-Token Ratio: 0.618
- Comprimento médio de palavras: 5.9 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 146.81
- MATTR: 0.850
- Zipf-TTR: 0.264
- Hapax legomena: 84.6%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 125.89
- Frazier depth: 4.00
- T-units: 7
- Subordination index: 0.571

**Features de Fala (Phase 2):**
- Mean Word Span: 1.60
- Repetition rate: 0.049
- Disfluency rate: 0.035

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- B2: 81%
- C2: 77%
- A2: 68%
- C1: 64%
- A1: 54%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.743
- Complexity trend: -0.001
- Number of windows: 94

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
⚠️ Inconsistência: AKT sugere A1 mas texto indica C2
- AKT nível estimado: A1
- Progresso no nível C2: 0%
- Convergência: very_low
- Ajuste de confiança: -25.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: C2 (confiança: 47%)


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
