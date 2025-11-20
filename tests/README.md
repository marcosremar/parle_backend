# Testes E2E - Parle Backend

Testes end-to-end para todos os serviços essenciais do sistema de speech-to-speech.

## Estrutura

```
tests/
├── e2e/
│   ├── test_stt.py          # Testes E2E para STT (Speech-to-Text)
│   ├── test_llm.py          # Testes E2E para LLM (Language Model)
│   ├── test_tts.py          # Testes E2E para TTS (Text-to-Speech)
│   ├── test_orchestrator.py # Testes E2E para Orchestrator
│   ├── test_websocket.py    # Testes E2E para WebSocket
│   └── conftest.py          # Configuração do pytest
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
```

## Variáveis de Ambiente

Os testes usam URLs padrão, mas podem ser sobrescritas:

```bash
export STT_SERVICE_URL="http://localhost:8099"
export LLM_SERVICE_URL="http://localhost:8110"
export TTS_SERVICE_URL="http://localhost:8103"
export ORCHESTRATOR_URL="http://localhost:8080"
export WEBSOCKET_URL="http://localhost:8022"
```

## Serviços Testados

- **STT**: Health check, transcrição de arquivo, transcrição base64
- **LLM**: Health check, geração de texto, chat completion, listagem de modelos
- **TTS**: Health check, síntese de fala (Eleven Labs e HuggingFace), listagem de vozes
- **Orchestrator**: Health check, pipeline completo speech-to-speech
- **WebSocket**: Health check, conexão WebSocket, speech-to-speech via WebSocket

