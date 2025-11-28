# Teste de Integração E2E - Monolith Speech-to-Speech

## 📋 Descrição

Este teste valida o fluxo completo **Speech-to-Speech** no modo monolith:

1. **STT (Speech-to-Text)**: Envia áudio → recebe transcrição
2. **LLM (Language Model)**: Envia texto → recebe resposta
3. **TTS (Text-to-Speech)**: Envia texto → recebe áudio
4. **Pipeline Completo**: Testa o orchestrator end-to-end

## 🚀 Como Executar

### 1. Iniciar a API

```bash
# Opção 1: Usando main.sh
./main.sh start api

# Opção 2: Diretamente com Python
export MONOLITH_MODE=true
python src/api/main.py

# Opção 3: Com Docker Compose
docker-compose up api
```

### 2. Executar o Teste

```bash
# Teste completo
pytest tests/e2e/test_monolith_speech_to_speech.py -v -s

# Apenas health check
pytest tests/e2e/test_monolith_speech_to_speech.py::test_api_health -v -s

# Teste completo com logs
pytest tests/e2e/test_monolith_speech_to_speech.py::test_speech_to_speech_complete_flow -v -s

# Executar diretamente
python tests/e2e/test_monolith_speech_to_speech.py
```

## 📦 Dependências

O teste requer:
- `gtts` - Para gerar áudio de teste
- `httpx` - Para fazer requisições HTTP
- `pytest` - Framework de testes
- `pytest-asyncio` - Suporte a testes assíncronos

Instalar:
```bash
pip install gtts httpx pytest pytest-asyncio
```

## ✅ O que o Teste Valida

### 1. Health Check
- API está respondendo
- Endpoint `/health` ou `/api/health` funciona

### 2. STT (Speech-to-Text)
- Gera áudio de teste com gTTS
- Envia áudio para `/api/speech/stt/transcribe`
- Recebe e valida transcrição

### 3. LLM (Language Model)
- Envia texto para `/api/llm/chat`
- Recebe e valida resposta do LLM

### 4. TTS (Text-to-Speech)
- Envia texto para `/api/speech/tts/synthesize`
- Recebe e valida áudio de resposta
- Salva áudio em `tests/output/`

### 5. Pipeline Completo
- Testa `/api/conversation/process` (orchestrator)
- Valida fluxo end-to-end

## 📁 Arquivos Gerados

O teste salva:
- `tests/output/test_response_<timestamp>.wav` - Áudio de resposta do TTS

## 🔧 Configuração

Variáveis de ambiente:
```bash
export API_URL=http://localhost:8000  # URL da API
export TIMEOUT=60.0                    # Timeout das requisições
```

## 🐛 Troubleshooting

### API não está respondendo
```bash
# Verificar se a API está rodando
curl http://localhost:8000/health

# Ver logs
tail -f /tmp/api.log  # Se usando main.sh
```

### Erro de importação
```bash
# Instalar dependências
pip install -r requirements.txt
```

### gTTS não instalado
O teste funciona sem gTTS, mas usa texto direto em vez de áudio real.

### Timeout
Aumentar `TIMEOUT` se os serviços estiverem lentos:
```python
TIMEOUT = 120.0  # 2 minutos
```

## 📊 Exemplo de Saída

```
🧪 Executando teste de integração E2E...
   API URL: http://localhost:8000

✅ Health check passou
🎤 Gerando áudio com gTTS: 'Olá, como você está? Este é um teste de integração.'
✅ Áudio gerado: 15234 bytes
📤 Enviando áudio para transcrição...
✅ STT transcreveu: 'Olá, como você está? Este é um teste de integração.'
🤖 Testando LLM (Language Model)...
✅ LLM respondeu: 'Olá! Estou bem, obrigado por perguntar...'
🔊 Testando TTS (Text-to-Speech)...
✅ TTS gerou áudio (binário direto): 45678 bytes
💾 Áudio de resposta salvo: tests/output/test_response_1234567890.wav
✅ Teste Speech-to-Speech completo passou!
```

