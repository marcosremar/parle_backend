# Testes E2E para Intelligent Tutoring System (ITS)

## 📋 Visão Geral

Este documento descreve os testes end-to-end criados para validar todas as funcionalidades do Intelligent Tutoring System (ITS).

## 🧪 Testes Implementados

### 1. `test_its_complete_flow.py` - Testes Completos do Fluxo ITS

Testes abrangentes que verificam todas as funcionalidades principais:

#### Teste 1: `test_complete_flow_single_turn`
**Objetivo**: Testar o fluxo completo de um único turno

**Fluxo testado**:
1. STT → Transcreve fala do usuário
2. Diagnostic Module → Analisa turno (erros, skills, features)
3. Student Model → Atualiza conhecimento (AKT)
4. Pedagogical Policy → Compõe prompt adaptado
5. LLM → Gera resposta adaptada
6. TTS → Gera áudio de resposta

**Validações**:
- ✅ Análise de turno retorna erros e skills corretas
- ✅ Features linguísticas são extraídas
- ✅ Conhecimento é atualizado no Student Model
- ✅ Prompt pedagógico é composto com contexto correto
- ✅ Estratégia pedagógica é selecionada apropriadamente

#### Teste 2: `test_multi_turn_session_analysis`
**Objetivo**: Testar análise agregada de múltiplos turnos

**Fluxo testado**:
1. Múltiplos turnos são analisados
2. Padrões de sessão são identificados
3. Padrões históricos são agregados
4. Adaptação pedagógica baseada em análise de sessão

**Validações**:
- ✅ Análise de sessão identifica tendências
- ✅ Skills problemáticas são identificadas
- ✅ Padrões linguísticos são detectados
- ✅ Análise histórica agrega dados de 90 dias

#### Teste 3: `test_linguistic_features_integration`
**Objetivo**: Testar extração e integração de features linguísticas

**Fluxo testado**:
1. Features linguísticas são extraídas do texto
2. Features são armazenadas no histórico
3. Features são usadas em adaptação de padrões AKT
4. Features são agregadas para análise histórica

**Validações**:
- ✅ Features são extraídas (tense, person, number, register, domain)
- ✅ Features são armazenadas em `InteractionHistory`
- ✅ Features são usadas em similaridade semântica AKT
- ✅ Agregação histórica identifica padrões por feature

#### Teste 4: `test_akt_vs_bkt_comparison`
**Objetivo**: Comparar AKT vs BKT e verificar que AKT é usado por padrão

**Fluxo testado**:
1. Sequência de interações (erros → acertos)
2. AKT rastreia conhecimento com atenção histórica
3. BKT é usado como fallback se AKT falhar
4. Progressão de aprendizado é verificada

**Validações**:
- ✅ AKT é usado por padrão
- ✅ Mastery probability melhora com acertos
- ✅ Progressão reflete padrão de aprendizado

#### Teste 5: `test_complete_orchestrator_flow`
**Objetivo**: Testar fluxo completo do orchestrator

**Fluxo testado**:
1. Orchestrator coordena todos os serviços
2. Fluxo completo: STT → Diagnostic → Student Model → Pedagogical Policy → LLM → TTS
3. Dados fluem corretamente entre serviços

**Validações**:
- ✅ Orchestrator coordena serviços
- ✅ Fluxo completo funciona end-to-end

#### Teste 6: `test_cefr_progress_tracking`
**Objetivo**: Testar rastreamento de progresso CEFR

**Fluxo testado**:
1. Múltiplas skills são praticadas
2. Nível CEFR é calculado a partir de masteries
3. Progresso é rastreado ao longo do tempo
4. Breakdown por dimensão é disponível

**Validações**:
- ✅ Nível CEFR é calculado corretamente
- ✅ Progresso por dimensão está disponível
- ✅ Estado interpretável inclui recomendações
- ✅ Skills fortes/fracas são identificadas

### 2. `test_intelligent_tutoring.py` - Testes Unitários ITS

Testes unitários que testam algoritmos diretamente (sem serviços HTTP):

- ✅ Teste do algoritmo AKT
- ✅ Teste do mecanismo de atenção AKT
- ✅ Teste de parâmetros adaptativos AKT
- ✅ Teste de decay temporal AKT
- ✅ Teste de features contextuais AKT
- ✅ Teste de predição AKT
- ✅ Teste do algoritmo BKT (fallback)
- ✅ Teste de ciclo completo de aprendizado
- ✅ Teste de múltiplas skills simultaneamente
- ✅ Teste de impacto de estado emocional
- ✅ Teste de adaptação de nível CEFR

### 3. `test_learning_progression.py` - Testes de Progressão

Testes que verificam se o sistema aprende ao longo do tempo:

- ✅ Teste de progressão de aprendizado (10 iterações)
- ✅ Teste de melhoria rápida
- ✅ Teste de aprendizado com esquecimento
- ✅ Teste de múltiplas skills simultaneamente
- ✅ Teste de mecanismo de atenção AKT

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

# Terminal 4: Learning Path
cd src/services/learning_path
python app_complete.py

# Terminal 5: Orchestrator
cd src/services/orchestrator
python app_complete.py
```

### Executar Testes

#### Todos os testes ITS:
```bash
pytest tests/e2e/test_its_complete_flow.py -v -s
```

#### Teste específico:
```bash
# Teste de fluxo completo
pytest tests/e2e/test_its_complete_flow.py::test_complete_flow_single_turn -v -s

# Teste de análise de sessão
pytest tests/e2e/test_its_complete_flow.py::test_multi_turn_session_analysis -v -s

# Teste de features linguísticas
pytest tests/e2e/test_its_complete_flow.py::test_linguistic_features_integration -v -s

# Teste de progressão CEFR
pytest tests/e2e/test_its_complete_flow.py::test_cefr_progress_tracking -v -s
```

#### Testes unitários:
```bash
pytest tests/e2e/test_intelligent_tutoring.py -v
```

#### Testes de progressão:
```bash
pytest tests/e2e/test_learning_progression.py -v -s
```

## ✅ Cobertura de Funcionalidades

Os testes e2e cobrem:

### ✅ Fluxo Completo
- [x] STT → Diagnostic → Student Model → Pedagogical Policy → LLM → TTS
- [x] Coordenação de serviços pelo Orchestrator
- [x] Fluxo de dados entre serviços

### ✅ Análise de Turno
- [x] Identificação de erros
- [x] Identificação de skills corretas
- [x] Extração de features linguísticas
- [x] Mapeamento semântico (SINKT)

### ✅ Análise de Sessão
- [x] Agregação de múltiplos turnos
- [x] Identificação de tendências
- [x] Padrões recorrentes
- [x] Skills problemáticas na sessão

### ✅ Análise Histórica
- [x] Agregação de dados de 90 dias
- [x] Padrões de erro por feature linguística
- [x] Features problemáticas vs dominadas
- [x] Recomendações baseadas em padrões

### ✅ Student Model
- [x] Atualização de conhecimento (AKT)
- [x] Rastreamento de progresso CEFR
- [x] Estado interpretável
- [x] Breakdown por dimensão
- [x] Skills fortes/fracas

### ✅ Pedagogical Policy
- [x] Composição de prompt adaptado
- [x] Seleção de estratégia (TEACH/REINFORCE/CHALLENGE)
- [x] Ajuste de scaffolding
- [x] Integração de análise de turno
- [x] Integração de análise de sessão

### ✅ AKT (Attentive Knowledge Tracing)
- [x] Mecanismo de atenção
- [x] Similaridade semântica
- [x] Adaptação de parâmetros
- [x] Decay temporal
- [x] Features contextuais
- [x] Predição de performance

### ✅ Features Linguísticas
- [x] Extração pelo LLM
- [x] Armazenamento no histórico
- [x] Uso em AKT (pattern adaptation)
- [x] Agregação histórica
- [x] Identificação de padrões problemáticos

### ✅ CEFR Progress
- [x] Cálculo de nível CEFR
- [x] Progresso por dimensão
- [x] Agregação de skills
- [x] Recomendações didáticas

## 📊 Resultados Esperados

### Teste de Fluxo Completo
- ✅ Todos os serviços respondem
- ✅ Análise de turno identifica erros e skills
- ✅ Conhecimento é atualizado
- ✅ Prompt é composto com contexto correto

### Teste de Análise de Sessão
- ✅ Padrões são identificados (ex: sempre erra na 3ª pessoa)
- ✅ Tendências são detectadas (melhorando/piorando)
- ✅ Skills problemáticas são identificadas

### Teste de Features Linguísticas
- ✅ Features são extraídas (tense, person, number, etc.)
- ✅ Features são armazenadas
- ✅ Padrões são identificados (ex: "person:3rd tem 75% de erros")

### Teste de Progressão CEFR
- ✅ Nível CEFR é calculado corretamente
- ✅ Progresso por dimensão está disponível
- ✅ Recomendações são geradas

## 🔧 Troubleshooting

### Serviços não disponíveis
Se um serviço não estiver disponível, o teste será pulado automaticamente:
```
⚠️  Diagnostic Module not available (skipping)
```

### Timeout em chamadas LLM
Alguns testes podem demorar devido a chamadas LLM. Timeouts são configurados para 30 segundos.

### Banco de dados
Os testes criam usuários únicos para isolamento. Se necessário, limpe o banco:
```bash
rm data/student_model.db
```

## 📝 Notas

- Os testes são **não-destrutivos**: cada teste cria usuários únicos
- Os testes são **independentes**: podem ser executados em qualquer ordem
- Os testes são **tolerantes a falhas**: pulam serviços não disponíveis
- Os testes são **verbosos**: usam `-s` para mostrar prints detalhados

## 🎯 Próximos Passos

- [ ] Adicionar testes de performance
- [ ] Adicionar testes de carga
- [ ] Adicionar testes de integração com WebSocket
- [ ] Adicionar testes de integração com API Gateway
- [ ] Adicionar testes de fallback (BKT quando AKT falha)

