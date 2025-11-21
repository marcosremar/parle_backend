# Testes E2E - Parle Backend

Testes end-to-end para todos os serviços essenciais do sistema de speech-to-speech.

## Estrutura

```
tests/
├── e2e/
│   ├── test_stt.py                  # Testes E2E para STT (Speech-to-Text)
│   ├── test_llm.py                  # Testes E2E para LLM (Language Model)
│   ├── test_tts.py                  # Testes E2E para TTS (Text-to-Speech)
│   ├── test_orchestrator.py         # Testes E2E para Orchestrator
│   ├── test_websocket.py            # Testes E2E para WebSocket
│   ├── test_conversation_store.py   # Testes E2E para Conversation Store
│   ├── test_conversation_history.py # Testes E2E para Conversation History
│   ├── test_session.py              # Testes E2E para Session
│   ├── test_scenarios.py            # Testes E2E para Scenarios
│   ├── test_user.py                 # Testes E2E para User
│   ├── test_database.py             # Testes E2E para Database
│   ├── test_file_storage.py         # Testes E2E para File Storage
│   ├── test_rest_polling.py         # Testes E2E para Rest Polling
│   ├── test_api_gateway.py          # Testes E2E para API Gateway
│   └── conftest.py                  # Configuração do pytest
└── fixtures/
    └── test_audio_real_speech.wav  # Áudio de teste
```

## Executando os Testes

### Pré-requisitos

1. Instalar dependências de teste:
```bash
pip install pytest pytest-asyncio requests python-socketio
```

2. Iniciar todos os serviços:
```bash
./start_services.sh
```

### Executar todos os testes E2E

```bash
pytest tests/e2e/ -v
```

### Executar testes de um serviço específico

```bash
# STT
pytest tests/e2e/test_stt.py -v

# LLM
pytest tests/e2e/test_llm.py -v

# TTS
pytest tests/e2e/test_tts.py -v

# Orchestrator
pytest tests/e2e/test_orchestrator.py -v

# WebSocket
pytest tests/e2e/test_websocket.py -v

# Conversation Store
pytest tests/e2e/test_conversation_store.py -v

# Conversation History
pytest tests/e2e/test_conversation_history.py -v

# Session
pytest tests/e2e/test_session.py -v

# Scenarios
pytest tests/e2e/test_scenarios.py -v

# User
pytest tests/e2e/test_user.py -v

# Database
pytest tests/e2e/test_database.py -v

# File Storage
pytest tests/e2e/test_file_storage.py -v

# Rest Polling
pytest tests/e2e/test_rest_polling.py -v

# API Gateway
pytest tests/e2e/test_api_gateway.py -v
```

## Variáveis de Ambiente

Os testes usam URLs padrão, mas podem ser sobrescritas:

```bash
export STT_SERVICE_URL="http://localhost:8099"
export LLM_SERVICE_URL="http://localhost:8110"
export TTS_SERVICE_URL="http://localhost:8103"
export ORCHESTRATOR_URL="http://localhost:8500"
export WEBSOCKET_URL="http://localhost:8022"
export CONVERSATION_STORE_URL="http://localhost:8800"
export CONVERSATION_HISTORY_URL="http://localhost:8501"
export SESSION_SERVICE_URL="http://localhost:8200"
export SCENARIOS_SERVICE_URL="http://localhost:8700"
export USER_SERVICE_URL="http://localhost:8201"
export DATABASE_SERVICE_URL="http://localhost:8400"
export FILE_STORAGE_SERVICE_URL="http://localhost:8300"
export REST_POLLING_SERVICE_URL="http://localhost:8701"
export API_GATEWAY_URL="http://localhost:8000"
```

## Serviços Testados

### Serviços Core
- **STT**: Health check, transcrição de arquivo, transcrição base64
- **LLM**: Health check, geração de texto, chat completion, listagem de modelos
- **TTS**: Health check, síntese de fala (Eleven Labs e HuggingFace), listagem de vozes
- **Orchestrator**: Health check, pipeline completo speech-to-speech
- **WebSocket**: Health check, conexão WebSocket, speech-to-speech via WebSocket

### Serviços de Dados e Armazenamento
- **Conversation Store**: Health check, criação de conversas, adicionar turnos, recuperar mensagens
- **Conversation History**: Health check, criar/listar conversas, salvar/recuperar mensagens, busca semântica, estatísticas
- **Database**: Health check, set/get/delete de dados, busca
- **File Storage**: Health check, upload/download/delete de arquivos, listagem, estatísticas

### Serviços de Gerenciamento
- **Session**: Health check, criar/listar/atualizar sessões
- **User**: Health check, registro, login, perfil de usuário
- **Scenarios**: Health check, listar/obter cenários
- **Rest Polling**: Health check, criar/listar/parar polls
- **API Gateway**: Health check, proxy para serviços, endpoints de conversação

