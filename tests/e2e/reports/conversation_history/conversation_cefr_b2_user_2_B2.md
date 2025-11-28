# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:37
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Professor, tenho lido bastante sobre a gestão de resíduos sólidos urbanos no Brasil e, honestamente, parece um desafio gigantesco. O problema é que, apesar de existirem leis como a Política Nacional de Resíduos Sólidos, a implementação ainda é muito precária em muitas cidades, não é? Exatamente! Eu observei que, enquanto algumas capitais possuem programas de coleta seletiva mais estruturados, em outras regiões, o descarte irregular e os lixões a céu aberto ainda são uma triste realidade. Isso me faz pensar: qual seria uma estratégia eficaz para abranger todo o território nacional? Considerando que a população brasileira é vasta e diversificada, a educação ambiental enfrenta barreiras culturais significativas. No entanto, acredito que campanhas governamentais de grande alcance, usando mídias sociais e televisão, poderiam ter um impacto substancial na mudança de hábitos. Apesar de todos os desafios, vejo um potencial enorme para o Brasil se tornar um exemplo na gestão de resíduos. Investir em economia circular, por exemplo, não só resolveria parte do problema ambiental, mas também geraria novos empregos e oportunidades econômicas para muitas comunidades.
```

## Nível CEFR Identificado
**Nível:** B2  
**Confiança:** 37%  
**Nível Esperado:** B2  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: B2 (confiança: 95%)
- Justificativa: O texto demonstra características consistentes com o nível B2. As frases são longas e complexas, com uso variado de subordinação e coordenação. O vocabulário é amplo e aborda temas abstratos como 'ges...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 14.9 palavras
- Subordinação: 25.0% das frases
- Type-Token Ratio: 0.610
- Comprimento médio de palavras: 5.6 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 174.00
- MATTR: 0.887
- Zipf-TTR: 0.269
- Hapax legomena: 80.0%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 218.33
- Frazier depth: 4.78
- T-units: 8
- Subordination index: 0.625

**Features de Fala (Phase 2):**
- Mean Word Span: 1.54
- Repetition rate: 0.011
- Disfluency rate: 0.029

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- C2: 100%
- B2: 77%
- C1: 69%
- A2: 68%
- A1: 58%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.858
- Complexity trend: -0.000
- Number of windows: 125

**Pairwise Classification (Phase 3):**
- Predicted level: C2
- Confidence: 33%
- Vote distribution: C2: 33%, C1: 27%, B2: 20%

**Validação Cruzada com AKT (Opção 1):**
⚠️ Inconsistência: AKT sugere A1 mas texto indica B2
- AKT nível estimado: A1
- Progresso no nível B2: 0%
- Convergência: very_low
- Ajuste de confiança: -25.0%

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: B2 (confiança: 37%)


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
