# Relatórios de Análise CEFR

Este diretório contém relatórios gerados automaticamente pelos testes E2E que analisam o nível CEFR (A1-C2) das respostas geradas pelo sistema.

## Tipos de Relatórios

### 1. Relatórios Individuais
- **Formato**: `cefr_analysis_{level}_{timestamp}.md`
- **Conteúdo**: Análise detalhada de uma resposta específica
- **Inclui**:
  - Texto analisado
  - Nível CEFR identificado
  - Confiança da identificação
  - Métricas sintáticas, lexicais e discursivas
  - Justificativa baseada nos critérios dos papers acadêmicos
  - Scores por nível CEFR

### 2. Relatórios Comparativos
- **Formato**: `cefr_comparison_{test_name}_{timestamp}.md`
- **Conteúdo**: Comparação entre múltiplas respostas (ex: A1 vs C1)
- **Inclui**:
  - Comparação lado a lado
  - Tabela de métricas
  - Análises individuais
  - Conclusão sobre conformidade

## Critérios Utilizados

Os relatórios utilizam critérios baseados nos seguintes papers acadêmicos:

1. **Leal, S. E., et al. (2022)**. NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese.
2. **Vajjala, S., & Rama, T. (2021)**. Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs.
3. **Arnold, T., et al. (2018)**. Predicting CEFRL levels in learner English on the basis of metrics and full texts.
4. **Ribeiro, E., et al. (2024)**. Avaliação automática do nível de complexidade de textos em português europeu.

## Métricas Analisadas

### Sintáticas
- Comprimento médio de sentença
- Presença de subordinação
- Presença de coordenação
- Uso de voz passiva
- Uso de subjuntivo
- Orações relativas
- Estruturas complexas

### Lexicais
- Tipo/token ratio
- Comprimento médio de palavras
- Diversidade lexical
- Expressões idiomáticas
- Vocabulário técnico

### Discursivas
- Densidade de conectores
- Marcadores discursivos
- Referências anafóricas

## Como Gerar Relatórios

Os relatórios são gerados automaticamente ao executar:

```bash
pytest tests/e2e/test_cefr_adaptation.py::test_cefr_comparison_a1_vs_c1 -v -s
```

Os relatórios serão salvos neste diretório com timestamps únicos.

## Interpretação dos Relatórios

### Nível Identificado
O sistema identifica o nível CEFR baseado em:
- Correspondência com critérios sintáticos
- Correspondência com critérios lexicais
- Correspondência com critérios discursivos
- Score calculado para cada nível

### Confiança
A confiança (0-100%) indica quão bem o texto corresponde aos critérios do nível identificado:
- **Alta (>80%)**: Forte correspondência com os critérios
- **Média (50-80%)**: Correspondência moderada
- **Baixa (<50%)**: Correspondência fraca, pode haver ambiguidade

### Status de Conformidade
- **✅ CONFORME**: Nível identificado corresponde ao nível esperado
- **⚠️ NÃO CONFORME**: Nível identificado não corresponde ao nível esperado

## Exemplo de Uso Programático

```python
from tests.e2e.cefr_level_analyzer import identify_cefr_level, generate_report, save_report

# Analisar texto
text = "Olá! Eu vou bem, obrigado!"
analysis = identify_cefr_level(text)

print(f"Nível identificado: {analysis['identified_level']}")
print(f"Confiança: {analysis['confidence']:.0%}")

# Gerar e salvar relatório
report = generate_report(text, target_level="A1")
save_report(report, filename="meu_relatorio.md")
```

## Limitações

- A análise é baseada em padrões heurísticos e pode não capturar todas as nuances
- Expressões idiomáticas e vocabulário técnico são detectados por padrões limitados
- A análise sintática profunda requer processamento de linguagem natural mais avançado

## Melhorias Futuras

- Integração com parser sintático para análise mais precisa
- Base de dados de frequência de palavras para análise lexical
- Machine learning para classificação de nível CEFR
- Validação com corpus anotado por especialistas

