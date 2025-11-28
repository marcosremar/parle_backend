# Parle Backend

Backend do projeto Parle - Sistema de conversação multimodal com arquitetura monolítica modular.

## 📋 Visão Geral

O Parle Backend é uma plataforma completa de conversação speech-to-speech que processa áudio de entrada, transcreve para texto, gera respostas usando modelos de linguagem e converte de volta para áudio. O sistema é construído com arquitetura monolítica modular, simplificando deploy e manutenção.

### Características Principais

- 🎤 **Speech-to-Speech Completo**: Pipeline end-to-end de áudio para áudio
- 🏗️ **Arquitetura Monolítica Modular**: Todos os serviços em um único processo com módulos internos
- 🔄 **Tempo Real**: Suporte a WebSocket (processo separado) e WebRTC para comunicação em tempo real
- 🤖 **IA Integrada**: STT (Whisper), LLM (GPT-2/LiteLLM), TTS (ElevenLabs/HuggingFace)
- 📊 **Orquestração Inteligente**: Orchestrator central que gerencia o fluxo completo
- 🔐 **Autenticação JWT**: API Gateway com autenticação e autorização
- 💾 **Persistência**: Múltiplos módulos de armazenamento (conversação, usuários, arquivos)
- ⚡ **Performance**: Chamadas diretas Python entre módulos (sem overhead HTTP)

## 🐍 Ambiente de Desenvolvimento

Este projeto usa **Conda** como ambiente padrão, com Python 3.11. Todas as dependências são gerenciadas através de um único ambiente Conda.

## 🚀 Início Rápido

O projeto inclui um script principal `main.sh` que facilita todas as operações:

```bash
# Configurar ambiente Conda (primeira vez)
./main.sh setup

# Ativar ambiente conda manualmente
./main.sh conda-activate

# Testar instalação
./main.sh test

# Iniciar API Principal (monolito modular) + WebSocket
./main.sh start --all

# Iniciar serviços individualmente
./main.sh start api        # API Principal (monolito com todos os módulos)
./main.sh start websocket  # WebSocket Service (processo separado)

# Ver status dos serviços
./main.sh status

# Abrir interface de demonstração
./main.sh demo

# Executar teste de demonstração
./main.sh test:demo:simple

# Abrir dashboard de monitoramento
./main.sh monitor

# Executar benchmark de performance
./main.sh benchmark

# Abrir shell com conda ativado
./main.sh shell
```

Para ver todos os comandos disponíveis: `./main.sh help`

## 🎤 Sistema Speech-to-Speech

O Parle Backend inclui um sistema completo de conversação multimodal:

### Funcionalidades

- **🎙️ STT (Speech-to-Text)**: Transcreve áudio usando OpenAI Whisper e Groq
- **🤖 LLM (Language Model)**: Gera respostas usando GPT-2, LiteLLM e múltiplos provedores
- **🔊 TTS (Text-to-Speech)**: Converte texto em áudio usando Eleven Labs e HuggingFace
- **🎯 Orchestrator**: Pipeline completo STT → LLM → TTS com gerenciamento de estado
- **💬 Conversação**: Histórico persistente e gerenciamento de contexto
- **🌐 Comunicação**: WebSocket e WebRTC para comunicação em tempo real

### Interface Web

- **Demonstração**: `./main.sh demo` - Interface completa com gravação
- **Monitoramento**: `./main.sh monitor` - Dashboard de status dos serviços

### Workflow Completo

```bash
# 1. Setup inicial
./main.sh setup

# 2. Iniciar todos os serviços
./main.sh start --all

# 3. Abrir demonstração
./main.sh demo

# 4. Testar pipeline completo
./main.sh test:demo:simple

# 5. Testar performance
./main.sh benchmark
```

## 🏗️ Arquitetura Monolítica Modular

O sistema é composto por **2 processos principais**:

1. **API Principal** (porta 8000): Monolito modular com todos os serviços como módulos internos
2. **WebSocket Service** (porta 8022): Processo separado para comunicação em tempo real

### Módulos Internos da API Principal

Os seguintes módulos estão integrados na API Principal (chamadas diretas Python):

### 🤖 Módulos de IA
- **STT**: Speech-to-Text (Whisper, Groq)
- **LLM**: Language Model (GPT-2, LiteLLM, múltiplos provedores)
- **TTS**: Text-to-Speech (ElevenLabs, HuggingFace)
- **Neural Codec**: Codec neural para compressão de áudio

### 🎯 Módulos de Conversação
- **Orchestrator**: Orquestrador central do pipeline speech-to-speech
- **Session**: Gerenciamento de sessões de conversação
- **Scenarios**: Gerenciamento de cenários e fluxos

### 💾 Módulos de Armazenamento
- **Conversation History**: Histórico de conversações
- **Conversation Store**: Armazenamento estruturado de conversas
- **User**: Gerenciamento de usuários e autenticação
- **Database**: Banco de dados genérico
- **File Storage**: Armazenamento de arquivos

### 🎓 Módulos de Tutoria Inteligente (ITS)
- **Student Model**: Rastreamento de conhecimento usando AKT (Attentive Knowledge Tracing) e BKT (fallback)
- **Pedagogical Policy**: Engine de política pedagógica com composição dinâmica de prompts
- **Diagnostic Module**: Análise de erros e complexidade linguística com SINKT
- **Learning Path**: Navegação de caminhos de aprendizado com spaced repetition

### 🔄 Processo Separado
- **WebSocket** (8022): Servidor Socket.IO para comunicação em tempo real (comunica com API via HTTP)

## 🎓 Intelligent Tutoring System (ITS)

O Parle Backend inclui um sistema completo de tutoria inteligente adaptativa que personaliza o aprendizado de idiomas baseado no progresso individual do estudante.

### Características Principais

- **🧠 Knowledge Tracing Avançado**: AKT (Attentive Knowledge Tracing) com ajuste IRT (Item Response Theory)
- **📊 Rastreamento CEFR**: Progresso detalhado por níveis CEFR (A1-C2) com agregação de skills
- **🔍 Diagnóstico Inteligente**: Análise de erros com SINKT (semantic skill tagging) e features linguísticas
- **📈 Estado Interpretável**: Breakdown detalhado por dimensão (grammar, vocabulary, pronunciation)
- **🎯 Política Pedagógica Adaptativa**: Composição dinâmica de prompts baseada no estado do estudante
- **🛤️ Navegação de Caminhos**: Recomendação de próximas skills baseada em ZPD e spaced repetition

### Conceitos Avançados Implementados

#### 1. Rasch Model-based Embeddings (IRT Integration)
- Cada skill possui um parâmetro de **dificuldade IRT** (0.0-1.0) baseado no nível CEFR
- O AKT ajusta probabilidades de acerto (`p_G`, `p_S`) baseado na dificuldade do skill
- Mapeamento automático: A1=0.2, A2=0.4, B1=0.6, B2=0.75, C1=0.85, C2=0.95

#### 2. FoLiBi (Forgetting-aware Linear Bias)
- **Desacopla esquecimento da correlação entre questões** usando bias linear de posição
- Combina decay temporal exponencial com bias linear para modelar esquecimento mais preciso
- Configurável via `folibi_enabled` flag e `linear_decay_factor`

#### 3. SINKT (LLM-based Semantic Encoding)
- **Mapeamento semântico** de `user_text` para skills relevantes usando LLM
- Retorna top-3 skills mais relevantes com confidence scores (0.0-1.0)
– Integrado no `speech_grader` para identificar skills mesmo sem erros explícitos

#### 4. Context-aware Representations
- **Attention window aumentado** de 10 para 30 interações
- **Similaridade semântica** entre interações baseada em features contextuais
- Considera padrões temporais (streaks, intervalos) para predições mais precisas

#### 5. Feature-rich Sub-skills
- Extração de **features linguísticas** como metadata (não skills separadas):
  - **Verbos**: tense, person, number, mood, aspect
  - **Vocabulário**: register (formal/informal), domain (travel/family/emotions)
  - **Artigos/Preposições**: tipo e categoria gramatical
- Features armazenadas no contexto do AKT para análise de padrões

#### 6. Interpretable Knowledge State
- **Breakdown por dimensão**: Progresso separado para grammar, vocabulary, pronunciation
- **Top skills fortes/fracas**: Identificação automática das 3 principais skills em cada categoria
- **Recomendações human-readable**: Sugestões didáticas baseadas no estado atual
- Endpoint: `/api/student/{user_id}/interpretable_knowledge_state`

#### 7. Análise de Padrões de Erro por Feature Linguística
- **LLM analisa fala do usuário**: Extrai features linguísticas (tense, person, number, register, domain) de cada turno
- **Armazenamento histórico**: Features salvas em `InteractionHistory.semantic_features` para análise temporal
- **Agregação de padrões**: Sistema identifica quais features causam mais erros agregando dados dos últimos 90 dias
- **Identificação automática**: Detecta features problemáticas (error_rate > 50%) e dominadas (error_rate < 20%)
- **Feedback didático**: Gera recomendações personalizadas baseadas em padrões de erro
- **Adaptação em tempo real**: AKT ajusta parâmetros baseado em padrões de erro por feature
- Endpoints:
  - `GET /api/student/{user_id}/linguistic_error_patterns` - Análise completa de padrões
  - `GET /api/student/{user_id}/interpretable_knowledge_state` - Inclui padrões e recomendações

#### 8. Filtragem Inteligente de Skills por Nível CEFR

O sistema implementa um **filtro inteligente de skills** que adapta dinamicamente quais habilidades linguísticas são analisadas baseado no nível CEFR do aluno. Isso otimiza performance, reduz custos e melhora a qualidade da análise.

##### 🎯 Objetivo

Evitar que o sistema analise skills muito avançadas para alunos iniciantes, reduzindo:
- **Latência**: Menos skills = prompt menor = resposta mais rápida
- **Custo**: Menos tokens processados = menor custo de API
- **Ruído**: Análise focada em skills relevantes = melhor qualidade

##### 🔒 Hard Cap (Proteção de Iniciantes)

**Alunos iniciantes (A1, A2, B1) NUNCA recebem skills C1/C2:**
- A1: Apenas skills A1-A2 (máximo até B2)
- A2: Skills A1-A2-B1 (máximo até B2)
- B1: Skills A2-B1-B2 (máximo até B2)

**Alunos avançados podem acessar skills superiores:**
- B2: Skills B1-B2-C1 (pode acessar C1)
- C1: Skills B2-C1-C2
- C2: Skills C1-C2

##### 📊 Janela de Níveis

O sistema usa uma **janela centrada no nível atual** (nível-1, nível, nível+1):

| Nível Aluno | Janela de Skills | Skills Retornadas |
|-------------|------------------|-------------------|
| A1 | A1, A2 | ~37 skills |
| A2 | A1, A2, B1 | ~59 skills |
| B1 | A2, B1, B2 | ~58 skills |
| B2 | B1, B2, C1 | ~48 skills |
| C1 | B2, C1, C2 | ~34 skills |
| C2 | C1, C2 | ~18 skills |

**Antes (sem filtro):** 189 skills sempre enviadas  
**Depois (com filtro):** 18-59 skills (68-90% de redução)

##### 🎓 Core vs Exploratório

O sistema prioriza skills do nível atual (core) mas também inclui skills exploratórias do próximo nível:

- **Core Skills**: Skills do nível atual e anteriores (prioridade alta)
- **Exploratory Skills**: Skills do próximo nível (prioridade baixa, para detectar talentos)

**Exemplo (Nível B1):**
- Core: A2 (20 skills), B1 (7 skills)
- Exploratório: B2 (3 skills)

##### 🛡️ Tratamento de Nível Desconhecido

Quando o nível do aluno é desconhecido ou inválido, o sistema assume **A1-A2** como padrão seguro:
- Níveis inválidos: `UNKNOWN`, `X1`, `""`, `INVALID`, etc.
- Retorna apenas skills A1-A2 (37 skills)
- Nenhuma skill C1/C2 incluída

##### ⚡ Impacto de Performance

**Redução de Skills Enviadas ao LLM:**
- A1: **80% de redução** (37 vs 189 skills)
- B1: **69% de redução** (58 vs 189 skills)
- C2: **90% de redução** (18 vs 189 skills)

**Benefícios:**
- ✅ Latência reduzida (prompt menor)
- ✅ Custo reduzido (menos tokens)
- ✅ Melhor qualidade (análise focada)
- ✅ Menos ruído (skills relevantes apenas)

##### 🔧 Implementação

A filtragem é implementada em `skill_registry.py` através da função `get_relevant_skills_for_context()`:

```python
from src.services.student_model.skill_registry import get_relevant_skills_for_context

# Obter skills relevantes para um aluno
valid_skills = get_relevant_skills_for_context(
    user_text="Eu fui ao banco ontem",
    cefr_level="A1",  # Nível do aluno
    context_type="production",  # production, comprehension, interaction
    max_skills=50  # Limite máximo
)
```

O orchestrator usa essa função automaticamente antes de chamar o Diagnostic Module, garantindo que apenas skills relevantes sejam analisadas.

##### ✅ Validação

Todos os aspectos da filtragem foram validados através de testes E2E (`tests/e2e/test_skill_filtering_by_level.py`):
- ✅ Hard cap funcionando (A1/A2/B1 sem C1/C2)
- ✅ Janela de níveis correta para todos os níveis
- ✅ Níveis desconhecidos tratados como A1-A2
- ✅ Priorização core/exploratório funcionando

Para mais detalhes, consulte `docs/TEST_REPORT_SKILL_FILTERING.md`.

### Fluxo de Funcionamento

#### Fluxo Principal (End-to-End)

```
1. User Speech (Audio)
   ↓
2. STT → Get Transcript
   ↓
3. Diagnostic Module → Analyze Turn
   - Identifica erros (o que está ruim)
   - Identifica skills corretas (o que está bom)
   - Extrai features linguísticas
   ↓
4. Student Model → Update Knowledge
   - Atualiza mastery probabilities
   ↓
5. Pedagogical Policy → Compose Prompt
   - Usa scenario (contexto)
   - Usa análise do turno atual (o que está bom/ruim)
   - Usa estado do estudante
   ↓
6. LLM → Generate Response
   - Resposta adaptada baseada em tudo acima
   ↓
7. TTS → Return to User
```

#### Fluxo Detalhado com Componentes Avançados

```
1. Usuário fala → STT transcreve
2. Diagnostic Module analisa:
   - Identifica erros gramaticais/vocabulário
   - Extrai features linguísticas (tense, person, register, etc.)
   - Mapeia semanticamente para skills (SINKT)
   - Identifica skills usadas corretamente
3. Student Model atualiza conhecimento:
   - AKT calcula mastery probability com IRT difficulty
   - FoLiBi modela esquecimento baseado em posição temporal
   - Features linguísticas armazenadas para análise de padrões
   - AKT usa histórico de 30 interações para attention e similaridade
4. Pedagogical Policy compõe prompt:
   - Usa CEFR progress e interpretable knowledge state
   - Usa análise do turno atual (erros, skills, features)
   - Usa padrões históricos (últimos 90 dias)
   - Seleciona estratégia (TEACH, REINFORCE, CHALLENGE)
   - Ajusta scaffolding baseado em mastery probability
5. Learning Path recomenda próxima skill:
   - Considera ZPD (Zone of Proximal Development)
   - Aplica spaced repetition para revisão
   - Prioriza skills com baixo domínio
```

### Análise Multi-Nível: Turno, Sessão e Histórico

O sistema **não analisa apenas uma frase individual**, mas agrega dados em **3 níveis** para tirar conclusões:

1. **Análise por Turno**: Cada frase é analisada individualmente (erros, skills, features)
2. **Análise de Sessão**: Agrega dados de todos os turnos da sessão atual (tendências, padrões recorrentes)
3. **Análise Histórica**: Agrega dados dos últimos 90 dias (padrões de longo prazo, progresso CEFR)

**Como funciona**:
- **AKT usa histórico**: Carrega últimas 30 interações para attention e similaridade semântica
- **Agregação temporal**: `_analyze_linguistic_error_patterns()` agrega últimos 90 dias
- **Session Analyzer**: Novo componente que agrega turnos da sessão atual
- **Prompt recebe tudo**: Análise do turno + padrões de sessão + histórico

Para detalhes completos, consulte `docs/SESSION_ANALYSIS_EXPLAINED.md`.

### Análise de Padrões de Erro por Feature Linguística

O sistema usa **LLM para analisar a fala do usuário** e identificar quais **features linguísticas** estão causando mais erros. Este fluxo está totalmente integrado no sistema:

#### 🔄 Fluxo Completo

```
User Speech
    ↓
LLM Analysis (diagnostic_module)
    ↓
Extract linguistic_features (tense, person, number, register, domain)
    ↓
Store in InteractionHistory.semantic_features
    ↓
AKT tracks patterns in real-time (feature_patterns)
    ↓
_analyze_linguistic_error_patterns() aggregates historical data (90 days)
    ↓
Identifica: "person:3rd tem 75% de erros"
    ↓
get_interpretable_knowledge_state() inclui padrões
    ↓
Pedagogical Policy recebe padrões no prompt
    ↓
StudentStateLayer renderiza: "Skills que precisam de prática: person 3rd (75% de erros)"
    ↓
LLM gera feedback focado: "Vamos praticar 3ª pessoa"
```

#### 📊 Features Extraídas pelo LLM

O LLM analisa cada turno e extrai:

- **Verbos**: `tense` (present/past/future), `person` (1st/2nd/3rd), `number` (singular/plural), `mood` (indicative/subjunctive/conditional), `aspect` (simple/progressive/perfect)
- **Vocabulário**: `register` (formal/informal/neutral), `domain` (family/travel/emotions/daily_life/general)
- **Artigos**: `article_type` (definite/indefinite)
- **Preposições**: `preposition_type` (location/basic)

#### 🎯 Análise de Padrões

A função `_analyze_linguistic_error_patterns()` agrega interações dos últimos 90 dias e calcula:

- **Taxa de erro por feature**: `error_rate = error_count / total_attempts`
- **Features problemáticas**: `error_rate > 50%` e `>= 5 tentativas`
- **Features dominadas**: `error_rate < 20%` e `>= 5 tentativas`

**Exemplo de resultado**:
```json
{
  "total_interactions_analyzed": 45,
  "features_analyzed": 12,
  "problematic_features": [
    {
      "feature_key": "person:3rd",
      "feature_type": "person",
      "feature_value": "3rd",
      "total_attempts": 8,
      "error_count": 6,
      "error_rate": 0.75,
      "success_rate": 0.25
    }
  ],
  "mastered_features": [
    {
      "feature_key": "tense:present",
      "feature_type": "tense",
      "feature_value": "present",
      "total_attempts": 10,
      "error_count": 1,
      "error_rate": 0.1,
      "success_rate": 0.9
    }
  ],
  "summary": "Maior dificuldade: person 3rd (75% de erros em 8 tentativas). Melhor domínio: tense present (90% de acertos em 10 tentativas)"
}
```

#### 🔗 Integração no Sistema

1. **Diagnostic Module**: LLM extrai features de cada turno
2. **Student Model**: Armazena features e agrega padrões históricos
3. **Interpretable Knowledge State**: Inclui análise de padrões e gera recomendações
4. **Pedagogical Policy**: Recebe padrões no prompt para feedback didático
5. **AKT**: Adapta parâmetros em tempo real baseado em padrões de erro

#### 📝 Exemplo Prático

**Cenário**: Estudante sempre erra na 3ª pessoa

1. **Turno 1**: "Ele foi ao mercado" → LLM extrai `person:3rd`, `tense:past` → Erro detectado
2. **Turno 2**: "Ela comprou pão" → LLM extrai `person:3rd`, `tense:past` → Erro detectado
3. **Após 5+ erros**: Sistema identifica padrão `person:3rd` com `error_rate: 75%`
4. **Recomendação gerada**: "Você está tendo dificuldades com person 3rd (75% de erros)"
5. **Próximo prompt**: LLM recebe contexto e foca em praticar 3ª pessoa
6. **AKT ajusta**: Aumenta `p_T` para essa feature específica (precisa de mais prática)

Para mais detalhes, consulte `docs/LINGUISTIC_FEATURES_ANALYSIS.md`.

### Endpoints Principais

#### Student Model Service (8900)
- `POST /api/student/{user_id}/assess` - Avaliar resposta e atualizar conhecimento
- `GET /api/student/{user_id}/profile` - Obter perfil completo do estudante
- `GET /api/student/{user_id}/skills` - Listar todas as skills do estudante
- `GET /api/student/{user_id}/cefr_progress` - Progresso detalhado por nível CEFR
- `GET /api/student/{user_id}/interpretable_knowledge_state` - Estado interpretável com recomendações e padrões de erro
- `GET /api/student/{user_id}/linguistic_error_patterns` - Análise de padrões de erro por feature linguística
- `GET /api/student/{user_id}/focus_areas` - Top áreas de foco recomendadas

#### Diagnostic Module Service (8960)
- `POST /api/diagnostic/analyze_turn` - Análise completa de um turno (erros, features, SINKT)
- `POST /api/diagnostic/estimate_level` - Estimar nível CEFR baseado em texto

#### Pedagogical Policy Service (8950)
- `POST /api/prompt/compose` - Compor prompt pedagógico baseado em contexto

#### Learning Path Service (8970)
- `GET /api/learning_path/next_skill` - Recomendar próxima skill para praticar

### Exemplo de Uso

```python
# 1. Analisar turno do estudante
analysis = await diagnostic_module.analyze_turn(
    user_text="Eu fui ao mercado ontem",
    ai_text="Ótimo! Você usou o passado corretamente.",
    valid_skills=["verb_conjugation_past", "vocabulary_basic", ...]
)

# Resultado inclui:
# - errors: Lista de erros encontrados
# - correct_skills: Skills usadas corretamente
# - linguistic_features: {tense: "past", person: "1st", number: "singular"}
# - semantic_skill_mapping: {"verb_conjugation_past": 0.85, ...}

# 2. Atualizar conhecimento do estudante
result = await student_model.assess(
    user_id="user123",
    skill_id="verb_conjugation_past",
    correct=True,
    difficulty=0.4,  # IRT difficulty (A2 level)
    linguistic_features={"tense": "past", "person": "1st"}
)

# 3. Obter estado interpretável
knowledge_state = await student_model.get_interpretable_knowledge_state("user123")

# Retorna:
# - dimension_progress: {"grammar": 0.65, "vocabulary": 0.72, "pronunciation": 0.58}
# - strong_skills: [{"skill_id": "...", "mastery": 0.85, ...}, ...]
# - weak_skills: [{"skill_id": "...", "mastery": 0.25, ...}, ...]
# - recommendations: ["Você precisa praticar mais pronunciation...", ...]
```

### Configuração

As skills padrão e seus parâmetros são definidos em `src/services/student_model/skill_registry.py`:

- **SKILL_CEFR_MAP**: Mapeamento de skills para níveis CEFR
- **CEFR_DIFFICULTY_MAP**: Dificuldades IRT por nível CEFR
- **DEFAULT_SKILLS**: Catálogo de skills padrão do sistema

### Referências Acadêmicas

O sistema implementa conceitos de papers recentes:
- **AKT (Attentive Knowledge Tracing)**: Modelo de knowledge tracing com attention mechanisms
- **FoLiBi**: Forgetting-aware Linear Bias para AKT
- **SINKT**: Structure-Aware Inductive Knowledge Tracing com LLMs
- **IRT (Item Response Theory)**: Framework probabilístico para avaliação de proficiência
- **CEFR (Common European Framework of Reference)**: Padrão internacional de proficiência linguística

## 📁 Estrutura do Projeto

Este projeto utiliza uma estrutura organizada:

```
parle_backend/
├── src/                    # Código fonte
│   ├── core/               # Biblioteca core compartilhada
│   │   └── shared/         # Modelos e utilitários compartilhados
│   └── services/           # Microserviços (16 serviços)
│       ├── api_gateway/    # Gateway principal
│       ├── orchestrator/   # Orquestrador
│       ├── stt/            # Speech-to-Text
│       ├── llm/            # Language Model
│       ├── tts/            # Text-to-Speech
│       └── ...             # Outros serviços
├── deploy/                 # Configurações de deploy
│   └── nomad/              # Arquivos Nomad para deploy
├── docs/                   # Documentação do projeto
├── scripts/                # Scripts de automação e utilitários
├── tests/                  # Testes end-to-end e fixtures
│   ├── e2e/                # Testes end-to-end
│   └── fixtures/           # Dados de teste
├── config/                 # Configurações globais
├── data/                   # Bancos de dados locais
└── vendor/                 # Submódulos e dependências
```

## ⚙️ Configuração Inicial

### 1. Clonagem com Submódulos

```bash
git clone <url-do-repositorio>
cd parle_backend
git submodule update --init --recursive
```

### 2. Instalação de Dependências

O script `main.sh setup` instala automaticamente o Miniconda e cria um ambiente conda otimizado para M1:

```bash
# Executar o script de setup (instala Miniconda e cria ambiente)
./main.sh setup

# Ativar ambiente conda
./main.sh conda-activate

# Ou manualmente
export PATH="$HOME/miniconda3/bin:$PATH"
conda activate parle_backend
```

**Nota**: O Miniconda é otimizado para Apple Silicon com pacotes pré-compilados, garantindo melhor performance e compatibilidade.

### 3. Verificar Instalação

Após o setup, verifique se tudo está funcionando:

```bash
./main.sh test
```

### 4. Configuração dos Submódulos

#### Skypilot

O submódulo skypilot já está configurado. Para atualizar:

```bash
git submodule update --remote vendor/skypilot
```

## 🔧 Desenvolvimento

### Trabalhando com Serviços

```bash
# Iniciar um serviço específico
./main.sh start orchestrator

# Ver logs de um serviço
./main.sh logs orchestrator

# Parar um serviço
./main.sh stop orchestrator

# Ver status de todos os serviços
./main.sh status
```

### Trabalhando com Conda

```bash
# Ativar ambiente
./main.sh conda-activate

# Ou manualmente
export PATH="$HOME/miniconda3/bin:$PATH"
conda activate parle_backend

# Instalar novas dependências
conda install -c conda-forge <pacote>

# Ver ambiente ativo
conda info --envs

# Desativar ambiente
conda deactivate
```

### Trabalhando com Submódulos

- **Atualizar submódulos**: `git submodule update --remote`
- **Commit de mudanças em submódulos**: Faça commit no submódulo primeiro, depois no projeto principal
- **Adicionar novo submódulo**: `git submodule add <url> vendor/<nome>`

## 🧪 Testes

### Testar Instalação

Execute o script de teste para verificar se tudo está configurado corretamente:

```bash
./scripts/test_installation.sh
```

Este script verifica:
- ✅ Python 3.11 instalado
- ✅ Ambiente virtual criado
- ✅ Dependências instaladas
- ✅ Estrutura de diretórios
- ✅ Imports Python funcionando
- ✅ Nomad instalado (opcional)

### Testes End-to-End

```bash
# Executar todos os testes
./main.sh test-all

# Teste de demonstração simples (speech-to-speech)
./main.sh test:demo:simple

# Testar health checks de todos os serviços
./main.sh test-services

# Executar testes com pytest
pytest tests/e2e/ -v
```

## 🚀 Deploy e Gerenciamento

### Iniciar Serviços

```bash
# Iniciar API Principal (monolito modular)
./main.sh start api

# Iniciar WebSocket (processo separado)
./main.sh start websocket

# Iniciar ambos
./main.sh start --all

# Ver status
./main.sh status

# Parar serviços
./main.sh stop api
./main.sh stop websocket
./main.sh stop --all
```

### Modo de Operação

- **Monolito Modular**: Todos os módulos rodam no mesmo processo Python
- **Chamadas Diretas**: Comunicação entre módulos via chamadas Python (sem HTTP)
- **WebSocket Separado**: Processo isolado que comunica com API via HTTP (localhost:8000)
- **Performance**: Sem overhead de serialização HTTP entre módulos

## 📝 Logs e Monitoramento

O projeto utiliza uma abordagem nativa e eficiente para logs, sem necessidade de bibliotecas adicionais complexas.

### Como funciona

1. **Aplicação (Python)**: 
   - Utilizamos a biblioteca `loguru` em todos os serviços.
   - Os logs são enviados para `stdout` (saída padrão) e `stderr` (erro padrão).
   - Não há necessidade de configurar arquivos de log manualmente na aplicação.

2. **Infraestrutura (Nomad)**:
   - O Nomad captura automaticamente os streams `stdout` e `stderr`.
   - Os logs são rotacionados automaticamente conforme configuração nos arquivos `.nomad`:
     ```hcl
     logs {
       max_files     = 10  # Mantém os últimos 10 arquivos
       max_file_size = 10  # Tamanho máximo de 10MB por arquivo
     }
     ```

### Visualizando Logs

Você pode visualizar os logs de qualquer serviço em tempo real:

```bash
# Ver logs de uma alocação específica
nomad alloc logs -f <alloc-id>

# Ver logs pelo nome do job (mais fácil)
nomad alloc logs -job api-gateway
nomad alloc logs -job user-service

# Ver logs de erro (stderr)
nomad alloc logs -stderr -job api-gateway
```

### Monitoramento

Para monitorar o status dos serviços:
```bash
./main.sh monitor
```

## 🔐 Segurança

- **Autenticação JWT**: API Gateway com tokens JWT para autenticação
- **Isolamento de Dados**: Cada usuário tem acesso apenas aos seus próprios dados
- **Validação de Entrada**: Validação rigorosa de dados de entrada usando Pydantic
- **Configuração Segura**: Variáveis de ambiente para segredos (não versionadas)

## 📊 Arquivos Ignorados

O arquivo `.gitignore` está configurado para ignorar:
- Arquivos Python compilados (`__pycache__/`, `*.pyc`)
- Ambientes conda (`miniconda3/`, `envs/`)
- Ambientes virtuais (`venv/`, `.env`)
- Logs e arquivos temporários (`*.log`, `tmp/`)
- Executável do Nomad (`vendor/nomad`)
- Arquivos de banco de dados locais (`*.db`, `*.sqlite`)
- Arquivos de configuração com segredos
- Modelos de ML (`models/`, `*.safetensors`)
- Arquivos de teste temporários (`tests/output/*.mp3`)

## 🤝 Contribuição

1. Crie uma branch para sua feature: `git checkout -b feature/nome-da-feature`
2. Faça commit das mudanças: `git commit -am 'Adiciona nova feature'`
3. Push para a branch: `git push origin feature/nome-da-feature`
4. Abra um Pull Request

## 📄 Licença

Ver arquivo LICENSE.txt

## 🔗 Links Úteis

- **Documentação Nomad**: `docs/NOMAD_GUIDELINES.md`
- **Service Discovery**: `SERVICE_DISCOVERY_IMPLEMENTATION.md`
- **Scripts**: `scripts/README.md`
- **Testes**: `tests/README.md`
