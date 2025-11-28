# Parle Backend - Visão Geral para Claude

## 📋 Resumo Executivo

**Parle Backend** é uma plataforma de conversação speech-to-speech construída com arquitetura de microserviços. O sistema processa áudio de entrada, transcreve para texto, gera respostas usando modelos de linguagem e converte de volta para áudio, tudo em tempo real.

## 🏗️ Arquitetura

### Padrão de Arquitetura
- **Microserviços**: 16 serviços independentes e escaláveis
- **Comunicação**: REST API, WebSocket (Socket.IO), WebRTC
- **Orquestração**: Orchestrator central que gerencia o pipeline completo
- **Deploy**: Suporte a Nomad para orquestração de containers

### Stack Tecnológico
- **Linguagem**: Python 3.11+ (Miniconda)
- **Framework Web**: FastAPI
- **IA/ML**: 
  - STT: OpenAI Whisper, Groq
  - LLM: GPT-2, LiteLLM (múltiplos provedores)
  - TTS: ElevenLabs, HuggingFace
- **Banco de Dados**: SQLite (local), suporte a outros
- **Comunicação Real-time**: Socket.IO, WebRTC
- **Autenticação**: JWT
- **Logging**: Loguru
- **Deploy**: HashiCorp Nomad

## 🔧 Serviços (16 Total)

### Gateway e Autenticação (4)
1. **api_gateway** (8000): Gateway principal, autenticação JWT, roteamento
2. **websocket** (8022): Servidor Socket.IO para comunicação em tempo real
3. **webrtc** (10100): Serviço WebRTC para comunicação peer-to-peer
4. **webrtc_signaling** (10101): Servidor de sinalização WebRTC

### Serviços de IA (4)
5. **stt** (8099): Speech-to-Text (Whisper, Groq)
6. **llm** (8110): Language Model (GPT-2, LiteLLM, múltiplos provedores)
7. **tts** (8103): Text-to-Speech (ElevenLabs, HuggingFace)
8. **neural_codec**: Codec neural para compressão de áudio

### Orquestração (3)
9. **orchestrator** (8500): Orquestrador central do pipeline speech-to-speech
10. **session** (8200): Gerenciamento de sessões de conversação
11. **scenarios** (8700): Gerenciamento de cenários e fluxos

### Armazenamento (5)
12. **conversation_history** (8501): Histórico de conversações
13. **conversation_store** (8800): Armazenamento estruturado de conversas
14. **user** (8201): Gerenciamento de usuários e autenticação
15. **database** (8400): Banco de dados genérico
16. **file_storage** (8300): Armazenamento de arquivos

### Comunicação (1)
17. **rest_polling** (8701): Polling REST para integração com sistemas externos

## 🔄 Fluxo de Dados (Speech-to-Speech)

```
Áudio de Entrada
    ↓
[WebSocket/WebRTC] → Recebe áudio
    ↓
[Orchestrator] → Coordena o pipeline
    ↓
[STT] → Transcreve áudio para texto
    ↓
[LLM] → Gera resposta baseada no texto
    ↓
[TTS] → Converte resposta em áudio
    ↓
[Orchestrator] → Retorna áudio processado
    ↓
[WebSocket/WebRTC] → Envia áudio de resposta
    ↓
Cliente recebe áudio
```

## 📁 Estrutura de Diretórios

```
parle_backend/
├── src/
│   ├── core/              # Biblioteca core compartilhada
│   └── services/          # 16 microserviços
├── deploy/nomad/          # Configurações Nomad
├── docs/                  # Documentação
├── scripts/               # Scripts de automação
├── tests/                 # Testes E2E
│   ├── e2e/              # Testes end-to-end
│   └── fixtures/         # Dados de teste
├── config/               # Configurações globais
├── data/                 # Bancos de dados locais
└── vendor/               # Submódulos
```

## 🚀 Comandos Principais

### Setup e Inicialização
```bash
./main.sh setup              # Configurar ambiente
./main.sh start --all        # Iniciar todos os serviços
./main.sh status             # Ver status
./main.sh test:demo:simple   # Teste completo
```

### Gerenciamento de Serviços
```bash
./main.sh start <servico>    # Iniciar serviço
./main.sh stop <servico>     # Parar serviço
./main.sh logs <servico>     # Ver logs
```

## 🔑 Conceitos Importantes

### Orchestrator
- **Função**: Coordena todo o pipeline speech-to-speech
- **Responsabilidades**:
  - Recebe requisições de áudio
  - Chama STT → LLM → TTS em sequência
  - Gerencia estado e contexto
  - Retorna áudio processado

### Service Discovery
- Comunicação entre serviços via HTTP
- URLs configuráveis via variáveis de ambiente
- Fallback automático em caso de falha

### Autenticação
- JWT tokens via API Gateway
- Isolamento de dados por usuário
- Validação de entrada rigorosa

### Persistência
- **Conversation History**: Histórico completo de conversas
- **Conversation Store**: Armazenamento estruturado
- **User Service**: Dados de usuários
- **File Storage**: Arquivos de áudio e mídia

## 🧪 Testes

- **E2E Tests**: Testes end-to-end completos
- **Health Checks**: Verificação de saúde dos serviços
- **Demo Test**: Teste completo do pipeline speech-to-speech
- **Coverage**: 20+ testes cobrindo principais funcionalidades

## 📊 Status Atual

- ✅ **16/16 serviços** funcionando
- ✅ **Pipeline completo** testado e validado
- ✅ **Estrutura reorganizada** e limpa
- ✅ **Documentação atualizada**
- ✅ **Testes passando** (20+ testes)

## 🔍 Pontos de Entrada Principais

1. **API Gateway** (8000): Ponto de entrada HTTP principal
2. **WebSocket** (8022): Comunicação em tempo real
3. **Orchestrator** (8500): Pipeline speech-to-speech
4. **WebRTC** (10100): Comunicação peer-to-peer

## 💡 Notas para Desenvolvimento

- Todos os serviços usam FastAPI
- Logging via Loguru (stdout/stderr)
- Configuração via variáveis de ambiente
- Suporte a múltiplos provedores de IA (LLM, TTS, STT)
- Arquitetura extensível e modular
- Testes automatizados com pytest

## 📚 Documentação Adicional

- `README.md`: Documentação completa do projeto
- `docs/NOMAD_GUIDELINES.md`: Guia de deploy com Nomad
- `SERVICE_DISCOVERY_IMPLEMENTATION.md`: Detalhes de service discovery
- `scripts/README.md`: Documentação dos scripts
- `tests/README.md`: Guia de testes

