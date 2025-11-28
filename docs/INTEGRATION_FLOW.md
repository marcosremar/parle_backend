# Fluxo de Integração Completo - Scenario → Analysis → Adaptive Pedagogy

## 🎯 Objetivo do Sistema

O estudante fala dentro de um **cenário específico** definido. O sistema:
1. **Analisa** o que o estudante disse (identifica o que está bom e o que está ruim)
   - **Por turno**: Analisa cada frase individualmente
   - **Por sessão**: Agrega dados de toda a conversa para identificar padrões
   - **Por histórico**: Usa dados dos últimos 90 dias para conclusões de longo prazo
2. **Adapta** a pedagogia baseado na análise
3. **Gera resposta** adaptada ao estado do estudante

## 🔄 Fluxo Completo Integrado

### Fluxo Atual (Corrigido)

```
1. User Speech (Audio)
   ↓
2. STT → Get Transcript
   ↓
3. Diagnostic Module → Analyze Turn (FRASE INDIVIDUAL)
   - Identifica erros na frase atual
   - Extrai features linguísticas da frase atual
   - Mapeia skills (SINKT) da frase atual
   - Identifica skills usadas corretamente na frase atual
   ↓
4. Student Model → Update Knowledge
   - Atualiza mastery probabilities (AKT)
   - AKT carrega últimas 30 interações do banco (HISTÓRICO)
   - AKT usa attention + similaridade semântica (compara com histórico)
   - Armazena linguistic features no banco
   ↓
5. Student Model → Get Interpretable Knowledge State
   - Agrega dados dos últimos 90 dias (HISTÓRICO)
   - _analyze_linguistic_error_patterns() identifica padrões de longo prazo
   - Retorna: problematic_features, mastered_features, recommendations
   ↓
6. Orchestrator → Build Session Analysis
   - Usa interpretable_knowledge_state (padrões históricos de 90 dias)
   - Usa conversation_history (últimas 10 mensagens da sessão)
   - Combina: análise do turno atual + padrões históricos + contexto da sessão
   ↓
7. Pedagogical Policy → Compose Prompt
   - Usa scenario (contexto)
   - Usa CEFR progress
   - Usa interpretable knowledge state (padrões históricos)
   - **USA análise do turno atual** (erros, features, skills da frase)
   - **USA análise da sessão** (padrões históricos de 90 dias)
   ↓
8. LLM → Generate Response
   - Recebe prompt adaptado com:
     * Scenario context
     * Student state (CEFR, mastery)
     * Current turn analysis (o que está bom/ruim NA FRASE ATUAL)
     * Session analysis (padrões históricos de 90 dias)
     * Conversation history (últimas 10 mensagens)
   - Gera resposta baseada em análise multi-nível
   ↓
9. TTS → Generate Audio
   ↓
10. Return Response to User
```

## 📋 Componentes Integrados

### 1. Scenario (Cenário)
- **Localização**: `scenarios` service
- **Uso**: Define contexto da conversa (ex: "restaurante", "aeroporto")
- **Integração**: Passado para `Pedagogical Policy` via `PromptContext.scenario`
- **Renderização**: `BaseLayer` no prompt pedagógico

### 2. Diagnostic Module (Análise Multi-Nível)
- **Localização**: `diagnostic_module` service
- **Função**: Analisa `user_transcript` ANTES de gerar resposta
- **Análise por Turno**: `analyze_turn()` - Analisa frase individual
  - Identifica erros na frase atual
  - Extrai features linguísticas da frase atual
  - Mapeia skills (SINKT) da frase atual
- **Análise de Sessão**: `analyze_session()` - Agrega múltiplos turnos (NOVO)
  - Agrega dados de todos os turnos da sessão
  - Identifica tendências (melhorando/piorando)
  - Identifica padrões recorrentes na sessão
- **Retorna**:
  - `errors`: Lista de erros encontrados
  - `correct_skills`: Skills usadas corretamente
  - `linguistic_features`: Features extraídas (tense, person, register, etc.)
  - `semantic_skill_mapping`: Mapeamento semântico (SINKT)
- **Integração**: Passado para `Pedagogical Policy` via `PromptContext.current_turn_analysis`
- **Renderização**: `TurnAnalysisLayer` no prompt pedagógico

### 3. Student Model (Atualização de Conhecimento + Agregação Histórica)
- **Localização**: `student_model` service
- **Função**: Atualiza conhecimento ANTES de compor prompt
- **Atualiza**:
  - Mastery probabilities (AKT)
  - Linguistic features (armazenadas)
  - Padrões de erro (feature_patterns)
- **AKT usa histórico**:
  - Carrega últimas 30 interações do banco
  - Usa attention weights (interações recentes têm mais peso)
  - Usa similaridade semântica (compara com histórico)
  - Adapta parâmetros baseado em padrões históricos
- **Agregação histórica**:
  - `_analyze_linguistic_error_patterns()` agrega últimos 90 dias
  - Identifica padrões de longo prazo (ex: sempre erra na 3ª pessoa)
  - Calcula taxas de erro por feature linguística
- **Integração**: Estado atualizado usado em `interpretable_knowledge_state`

### 4. Pedagogical Policy (Adaptação)
- **Localização**: `pedagogical_policy` service
- **Função**: Compõe prompt adaptado baseado em:
  - Scenario
  - Student state (CEFR, mastery)
  - **Current turn analysis** (o que está bom/ruim)
  - Interpretable knowledge state
- **Camadas do Prompt**:
  1. `BaseLayer`: Scenario context
  2. `StudentStateLayer`: CEFR level, progress, recommendations
  3. `StrategyLayer`: TEACH/REINFORCE/CHALLENGE
  4. `FocusLayer`: Target skill
  5. **`TurnAnalysisLayer`**: Análise do turno atual (NOVO)
  6. `AffectiveLayer`: Emotional modulation

### 5. Learning Path (Próxima Skill)
- **Localização**: `learning_path` service
- **Função**: Recomenda próxima skill para praticar
- **Integração**: Passado como `target_skill` no prompt

## 🔧 Mudanças Implementadas

### 1. Análise ANTES da Resposta ✅
- **Antes**: Análise acontecia em background DEPOIS da resposta
- **Agora**: Análise acontece ANTES de gerar resposta
- **Localização**: `orchestrator_engine.py` → STEP 2.5

### 2. Atualização de Conhecimento ANTES do Prompt ✅
- **Antes**: Conhecimento atualizado em background
- **Agora**: Conhecimento atualizado ANTES de compor prompt
- **Benefício**: Prompt usa estado atualizado do estudante

### 3. Análise do Turno no Prompt ✅
- **Nova Layer**: `TurnAnalysisLayer`
- **Renderiza**:
  - Erros identificados
  - Skills usadas corretamente
  - Features linguísticas
  - Semantic skill mapping
  - Instruções pedagógicas baseadas na análise

### 4. Scenario Integrado ✅
- **BaseLayer**: Renderiza contexto do scenario
- **Prompt**: Inclui instruções do scenario

## 📊 Exemplo de Prompt Gerado

```
[System Role]
Você é um professor em um cenário de restaurante.

[Student State]
Nível CEFR Global: A2
Descrição: Iniciante-Intermediário. Consegue comunicar em situações simples do dia a dia.
Língua nativa: pt

[Análise Detalhada do Progresso]
Progresso por dimensão:
  - grammar: 65%
  - vocabulary: 72%
Skills que precisam de prática:
  - person 3rd: 25% (75% de erros)

[Pedagogical Strategy]
Estratégia: REFORÇO COM PRÁTICA
Mastery: 45%
Abordagem: Prática guiada com correção sutil

[Focus Instruction]
Habilidade em foco: verb_conjugation_past (verb_conjugation_past)
Domínio atual: 45%
Objetivo: Reforçar e praticar 'verb_conjugation_past'.

[Current Turn Analysis]
Erros identificados (1):
  1. grammar em 'verb_conjugation_past' (medium)
     → Erro na conjugação do verbo no passado

✅ Skills usadas corretamente (2):
  - vocabulary_basic
  - article_definite

📊 Features linguísticas identificadas:
  Tempo: past, Pessoa: 3rd, Número: singular

💡 Instruções pedagógicas:
  - Foque nos erros identificados ao responder
  - Use correção apropriada (implicit/explicit) baseada na estratégia
  - Reforce as skills usadas corretamente

[Affective Modulation]
Tom: Encorajador e paciente
```

## ✅ Checklist de Integração

- [x] Scenario carregado e passado para prompt
- [x] Transcript obtido ANTES de análise
- [x] Análise do turno feita ANTES de gerar resposta
- [x] Conhecimento atualizado ANTES de compor prompt
- [x] Análise do turno incluída no prompt
- [x] Prompt pedagógico adaptado baseado em análise
- [x] Resposta LLM gerada com contexto completo
- [x] Background task removida (análise agora é síncrona)

## 🎯 Resultado Final

O sistema agora:
1. ✅ Analisa o que o estudante disse (erros, features, skills)
2. ✅ Identifica o que está bom e o que está ruim
3. ✅ Adapta a pedagogia baseado na análise
4. ✅ Gera resposta adaptada ao estado do estudante
5. ✅ Usa o scenario como contexto da conversa

**Tudo integrado e funcionando em tempo real!**

