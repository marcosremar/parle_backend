# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** 2025-11-23 08:31:30
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
Olá, Professor. Sim, tenho refletido bastante sobre isso. Estou dividido entre seguir uma carreira mais tradicional na minha área de formação, que é engenharia de software, ou explorar algo mais voltado para o impacto social, como trabalhar em uma startup de tecnologia verde. Exatamente. A segurança financeira e as oportunidades de crescimento em uma empresa estabelecida são inegáveis. No entanto, sinto que minha contribuição poderia ser mais significativa em um projeto que realmente fizesse a diferença para o meio ambiente ou a sociedade. Sim, tenho pensado nisso. Em uma empresa grande, eu desenvolveria sistemas complexos, talvez otimizando processos internos ou criando novas funcionalidades. Já na tecnologia verde, eu poderia estar projetando soluções para monitoramento ambiental ou eficiência energética, o que me parece mais recompensador pessoalmente. Sim, tenho plena consciência dos riscos envolvidos. Apesar de a instabilidade ser um fator, acredito que o aprendizado e a autonomia em um ambiente de startup seriam imensamente valiosos, talvez até superando os benefícios monetários a longo prazo. Essa é uma excelente sugestão, Professor. Não havia explorado essa via com tanta profundidade. Talvez eu devesse pesquisar mais sobre empresas que investem em inovação sustentável dentro de suas estruturas maiores. Isso poderia oferecer um equilíbrio interessante.
```

## Nível CEFR Identificado
**Nível:** B2  
**Confiança:** 33%  
**Nível Esperado:** B2  
**Status:** ✅ CONFORME

## Explicação Geral

**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: B2 (confiança: 90%)
- Justificativa: O texto demonstra um nível B2 devido à presença consistente de subordinação variada, uso de diferentes tempos verbais (presente, pretérito perfeito, futuro implícito), vocabulário que aborda temas abs...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: 10.2 palavras
- Subordinação: 5.0% das frases
- Type-Token Ratio: 0.578
- Comprimento médio de palavras: 5.8 caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: 155.44
- MATTR: 0.888
- Zipf-TTR: 0.260
- Hapax legomena: 78.7%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: 98.93
- Frazier depth: 3.67
- T-units: 14
- Subordination index: 0.500

**Features de Fala (Phase 2):**
- Mean Word Span: 1.55
- Repetition rate: 0.005
- Disfluency rate: 0.010

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
- B1: 100%
- B2: 70%
- A2: 68%
- C1: 61%
- A1: 55%
- C2: 44%

**Complexity Contours (Phase 3):**
- Mean complexity: 0.780
- Complexity trend: 0.001
- Number of windows: 151

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
- Nível final: B2 (confiança: 33%)


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
