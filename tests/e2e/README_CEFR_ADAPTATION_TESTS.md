# Testes E2E para Adaptação CEFR

## 📋 Visão Geral

Este documento descreve os testes end-to-end criados para validar a adaptação automática de linguagem baseada no nível CEFR do aluno.

## 🎯 Objetivo

Verificar que o sistema:
1. **Compõe prompts apropriados** para cada nível CEFR (A1-C2)
2. **Adapta a linguagem do LLM** baseado no nível do aluno
3. **Aplica restrições gramaticais e vocabulares** corretamente
4. **Integra todas as camadas** do prompt composer com adaptação CEFR

## 🧪 Testes Implementados

### 1. `test_cefr_prompt_composition_a1`
**Objetivo**: Verificar que prompts A1 contêm instruções apropriadas

**Validações**:
- ✅ Prompt menciona nível A1
- ✅ Instruções incluem presente do indicativo
- ✅ Especifica frases curtas (3-5 palavras)
- ✅ Não permite estruturas avançadas (subjuntivo, voz passiva)

### 2. `test_cefr_prompt_composition_c1`
**Objetivo**: Verificar que prompts C1 contêm instruções apropriadas

**Validações**:
- ✅ Prompt menciona nível C1
- ✅ Permite estruturas complexas (subjuntivo)
- ✅ Permite expressões idiomáticas
- ✅ Instruções apropriadas para nível avançado

### 3. `test_cefr_adaptation_all_levels`
**Objetivo**: Testar composição de prompts para todos os níveis CEFR

**Validações**:
- ✅ Prompts são gerados para todos os níveis (A1, A2, B1, B2, C1, C2)
- ✅ Cada prompt menciona o nível correspondente
- ✅ Cada prompt contém restrições gramaticais
- ✅ Cada prompt contém instruções de validação

### 4. `test_cefr_llm_response_adaptation_a1`
**Objetivo**: Verificar que respostas do LLM são adaptadas ao nível A1

**Fluxo**:
1. Compõe prompt para A1
2. Chama LLM com o prompt
3. Analisa complexidade da resposta
4. Verifica conformidade com restrições A1

**Validações**:
- ✅ Frases têm comprimento apropriado (≤8 palavras em média)
- ✅ Não usa estruturas avançadas (subjuntivo, voz passiva)
- ✅ Resposta está em conformidade com restrições A1

### 5. `test_cefr_llm_response_adaptation_c1`
**Objetivo**: Verificar que respostas do LLM são adaptadas ao nível C1

**Fluxo**:
1. Compõe prompt para C1
2. Chama LLM com o prompt
3. Analisa complexidade da resposta
4. Verifica que estruturas complexas são permitidas

**Validações**:
- ✅ Frases podem ser mais longas (≥8 palavras em média)
- ✅ Estruturas complexas são permitidas
- ✅ Resposta reflete nível avançado

### 6. `test_cefr_comparison_a1_vs_c1`
**Objetivo**: Comparar respostas A1 vs C1 para verificar adaptação

**Validações**:
- ✅ A1 tem frases mais curtas que C1
- ✅ A1 tem score de complexidade menor que C1
- ✅ Diferença clara entre níveis

### 7. `test_cefr_full_flow_with_adaptation`
**Objetivo**: Testar fluxo completo através do orchestrator com adaptação CEFR

**Fluxo**:
1. Cria sessão e cenário
2. Define nível CEFR do aluno (A1)
3. Processa turno através do orchestrator
4. Verifica que resposta está adaptada ao nível

**Validações**:
- ✅ Orchestrator coordena todos os serviços
- ✅ Resposta final está adaptada ao nível CEFR
- ✅ Restrições gramaticais são respeitadas

### 8. `test_cefr_prompt_layers_integration`
**Objetivo**: Verificar que todas as camadas do prompt incluem adaptação CEFR

**Validações**:
- ✅ BaseLayer inclui instruções de adaptação
- ✅ StudentStateLayer inclui restrições gramaticais/vocabulares
- ✅ StrategyLayer inclui restrições linguísticas
- ✅ Instrução final de validação está presente

## 🔍 Análise de Complexidade com LLM (Sonnet 4.5)

Os testes usam **Claude Sonnet 4.5** para análise automática de complexidade e verificação de conformidade CEFR. O LLM analisa:

### Por que usar LLM?
- ✅ **Análise contextual mais precisa** do que heurísticas
- ✅ **Detecção de nuances linguísticas** que regex não captura
- ✅ **Avaliação de conformidade CEFR** baseada em conhecimento linguístico
- ✅ **Explicações detalhadas** das violações e conformidade

### Análise de Complexidade

O LLM analisa automaticamente:

### Métricas Analisadas pelo LLM
O LLM retorna um JSON estruturado com:
- **avg_words_per_sentence**: Média de palavras por frase
- **max_words_per_sentence**: Máximo de palavras em uma frase
- **total_words**: Total de palavras no texto
- **has_subjunctive**: Detecta subjuntivo (`que seja`, `se fosse`)
- **has_passive**: Detecta voz passiva (`é feito`, `foi dito`)
- **has_relative_clauses**: Detecta orações relativas (`que é`, `onde está`)
- **has_conditional**: Detecta condicional (`se seria`, `caso que`)
- **complexity_score**: Score de 0 a 100 calculado pelo LLM
- **explanation**: Explicação textual da análise

### Verificação de Conformidade CEFR

O LLM verifica se o texto está em conformidade com as restrições do nível CEFR:
- **is_compliant**: `true` se está em conformidade, `false` caso contrário
- **violations**: Lista de violações encontradas (vazia se `is_compliant = true`)
- **explanation**: Explicação geral da conformidade
- **detected_structures**: Estruturas detectadas no texto

### Restrições por Nível

#### A1/A2
- ❌ Não permite subjuntivo
- ❌ Não permite voz passiva
- ❌ Não permite orações relativas complexas
- ✅ Frases curtas (A1: 3-5 palavras, A2: 5-8 palavras)

#### B1/B2
- ✅ Permite subjuntivo básico
- ✅ Permite voz passiva simples
- ✅ Permite orações subordinadas

#### C1/C2
- ✅ Permite todas as estruturas
- ✅ Permite expressões idiomáticas
- ✅ Permite variações estilísticas

## 🚀 Como Executar

### Pré-requisitos

1. **Iniciar todos os serviços ITS**:
```bash
# Terminal 1: Student Model
cd src/services/student_model
python app_complete.py

# Terminal 2: Pedagogical Policy
cd src/services/pedagogical_policy
python app_complete.py

# Terminal 3: Diagnostic Module
cd src/services/diagnostic_module
python app_complete.py

# Terminal 4: Orchestrator
cd src/services/orchestrator
python app_complete.py

# Terminal 5: LLM Service
cd src/services/llm
python app_complete.py
```

### Executar Testes

#### Todos os testes de adaptação CEFR:
```bash
pytest tests/e2e/test_cefr_adaptation.py -v -s
```

#### Teste específico:
```bash
# Teste de composição de prompt A1
pytest tests/e2e/test_cefr_adaptation.py::test_cefr_prompt_composition_a1 -v -s

# Teste de adaptação de resposta LLM
pytest tests/e2e/test_cefr_adaptation.py::test_cefr_llm_response_adaptation_a1 -v -s

# Teste de comparação A1 vs C1
pytest tests/e2e/test_cefr_adaptation.py::test_cefr_comparison_a1_vs_c1 -v -s

# Teste de fluxo completo
pytest tests/e2e/test_cefr_adaptation.py::test_cefr_full_flow_with_adaptation -v -s
```

## ✅ Cobertura de Funcionalidades

### ✅ Composição de Prompt
- [x] Prompts para todos os níveis CEFR (A1-C2)
- [x] Instruções específicas por nível
- [x] Restrições gramaticais detalhadas
- [x] Restrições vocabulares detalhadas
- [x] Instrução final de validação

### ✅ Adaptação de Linguagem
- [x] LLM adapta respostas ao nível A1
- [x] LLM adapta respostas ao nível C1
- [x] Diferença clara entre níveis
- [x] Conformidade com restrições gramaticais

### ✅ Integração de Camadas
- [x] BaseLayer inclui adaptação CEFR
- [x] StudentStateLayer inclui restrições
- [x] StrategyLayer inclui restrições linguísticas
- [x] Validação global no final do prompt

### ✅ Fluxo Completo
- [x] Orchestrator coordena adaptação CEFR
- [x] Fluxo end-to-end com adaptação
- [x] Resposta final adaptada ao nível

## 📊 Resultados Esperados

### Teste de Composição de Prompt
- ✅ Prompts contêm instruções apropriadas para cada nível
- ✅ Restrições gramaticais estão presentes
- ✅ Instruções de validação estão presentes

### Teste de Adaptação de Resposta
- ✅ Respostas A1 são simples e curtas
- ✅ Respostas C1 podem ser complexas
- ✅ Respostas respeitam restrições do nível

### Teste de Comparação
- ✅ Diferença clara entre A1 e C1
- ✅ A1 tem menor complexidade que C1
- ✅ Adaptação funciona corretamente

## 🔧 Troubleshooting

### Serviços não disponíveis
Se os testes pularem com "service not available":
1. Verifique se todos os serviços estão rodando
2. Verifique as URLs nos arquivos `.env` ou variáveis de ambiente
3. Verifique se as portas estão corretas

### LLM não retorna texto
Se o LLM não retornar texto:
1. Verifique se o serviço LLM está configurado corretamente
2. Verifique se há tokens/credits disponíveis
3. Verifique os logs do serviço LLM

### Respostas não adaptadas
Se as respostas não estiverem adaptadas:
1. Verifique se o prompt contém as instruções CEFR
2. Verifique se o nível CEFR está sendo passado corretamente
3. Verifique os logs do Pedagogical Policy service

## 📝 Notas

- Os testes podem ser executados mesmo se alguns serviços não estiverem disponíveis (eles serão pulados)
- Os testes de LLM podem falhar se o modelo não seguir as instruções perfeitamente (isso é esperado em alguns casos)
- A análise de complexidade é heurística e pode não capturar todas as nuances

