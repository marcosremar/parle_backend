# Guia de Troubleshooting - Parle Backend

Este documento lista problemas comuns e suas soluções.

## 📋 Índice

- [Problemas de Inicialização](#problemas-de-inicialização)
- [Problemas de Conexão](#problemas-de-conexão)
- [Problemas de Performance](#problemas-de-performance)
- [Problemas de Autenticação](#problemas-de-autenticação)
- [Problemas de Módulos](#problemas-de-módulos)
- [Problemas de Docker](#problemas-de-docker)

## Problemas de Inicialização

### Erro: "Module not found"

**Sintoma**: `ModuleNotFoundError: No module named 'src'`

**Solução**:
```bash
# Certifique-se de estar no diretório raiz do projeto
cd /path/to/parle_backend

# Verifique se o PYTHONPATH está configurado
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou execute com python -m
python -m src.api.main
```

### Erro: "Failed to initialize modules"

**Sintoma**: Logs mostram `❌ Failed to initialize modules`

**Solução**:
1. Verifique variáveis de ambiente:
```bash
echo $JWT_SECRET_KEY
echo $DATABASE_URL
```

2. Verifique logs detalhados:
```bash
# Aumentar nível de log
export LOG_LEVEL=DEBUG
python src/api/main.py
```

3. Verifique dependências:
```bash
pip install -r requirements.txt
```

### Erro: "Port already in use"

**Sintoma**: `Address already in use: 8000`

**Solução**:
```bash
# Encontrar processo usando a porta
lsof -i :8000
# ou
netstat -tulpn | grep 8000

# Matar processo
kill -9 <PID>

# Ou usar outra porta
export PORT=8001
```

## Problemas de Conexão

### Erro: "Connection refused" ao conectar ao Redis

**Sintoma**: `redis.exceptions.ConnectionError: Error connecting to Redis`

**Solução**:
1. Verifique se Redis está rodando:
```bash
redis-cli ping
# Deve retornar PONG
```

2. Verifique variáveis de ambiente:
```bash
echo $REDIS_HOST
echo $REDIS_PORT
```

3. Teste conexão:
```bash
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
```

### Erro: "Database connection failed"

**Sintoma**: Erros ao acessar banco de dados

**Solução**:
1. Verifique URL de conexão:
```bash
echo $DATABASE_URL
```

2. Teste conexão:
```python
# Python
from sqlalchemy import create_engine
engine = create_engine($DATABASE_URL)
engine.connect()
```

3. Verifique permissões do banco de dados

## Problemas de Performance

### API lenta ou timeout

**Sintoma**: Requisições demoram muito ou dão timeout

**Solução**:
1. Verifique logs para identificar gargalos:
```bash
tail -f logs/api.log | grep -i "slow\|timeout\|error"
```

2. Verifique recursos do servidor:
```bash
# CPU
top
# Memória
free -h
# Disco
df -h
```

3. Ajuste workers:
```python
# Em src/api/main.py ou via variável de ambiente
workers = 4  # Ajuste conforme CPU cores
```

4. Verifique queries lentas no banco de dados

### Alto uso de memória

**Sintoma**: Processo consome muita memória

**Solução**:
1. Limite workers:
```bash
export WORKERS=2  # Reduzir número de workers
```

2. Verifique vazamentos de memória:
```bash
# Monitorar memória
watch -n 1 'ps aux | grep python'
```

3. Reinicie periodicamente (usar systemd com restart automático)

## Problemas de Autenticação

### Erro: "Invalid token"

**Sintoma**: `401 Unauthorized` mesmo com token válido

**Solução**:
1. Verifique JWT_SECRET_KEY:
```bash
# Deve ser o mesmo usado para gerar o token
echo $JWT_SECRET_KEY
```

2. Verifique expiração do token:
```python
import jwt
token = "seu-token"
secret = os.getenv("JWT_SECRET_KEY")
payload = jwt.decode(token, secret, algorithms=["HS256"])
print(payload)  # Verificar expiração
```

3. Verifique formato do header:
```bash
# Deve ser: Authorization: Bearer <token>
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/auth/me
```

### Erro: "Rate limit exceeded"

**Sintoma**: `429 Too Many Requests`

**Solução**:
1. Aguarde o período de rate limit
2. Verifique limites configurados em `src/core/constants.py`
3. Use autenticação para limites mais altos

## Problemas de Módulos

### Módulo STT/TTS não funciona

**Sintoma**: Erros ao transcrever ou sintetizar

**Solução**:
1. Verifique API keys:
```bash
echo $OPENAI_API_KEY
echo $ELEVENLABS_API_KEY
```

2. Verifique formato de áudio:
   - STT: WAV, MP3, formato suportado
   - TTS: Texto válido

3. Verifique logs do módulo:
```bash
grep -i "stt\|tts" logs/api.log
```

### Módulos de Tutoring desabilitados

**Sintoma**: `503 Tutoring modules are disabled`

**Solução**:
```bash
# Habilitar módulos de tutoring
export ENABLE_TUTORING_MODULES=true
```

## Problemas de Docker

### Erro: "Cannot connect to Docker daemon"

**Sintoma**: `Cannot connect to the Docker daemon`

**Solução**:
```bash
# Iniciar Docker
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
# Fazer logout/login
```

### Erro: "Out of memory" no Docker

**Sintoma**: Container para ou falha por falta de memória

**Solução**:
1. Aumente memória do Docker (Docker Desktop: Settings > Resources)
2. Limite recursos:
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### Erro: "Volume mount failed"

**Sintoma**: Erros ao montar volumes

**Solução**:
1. Verifique permissões:
```bash
sudo chown -R $USER:$USER /path/to/volume
```

2. Use caminhos absolutos no `docker/docker-compose.yml`
3. Certifique-se de executar da raiz do projeto:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Logs e Debugging

### Habilitar logs detalhados

```bash
# Via variável de ambiente
export LOG_LEVEL=DEBUG

# Ou no código
from loguru import logger
logger.add("logs/debug.log", level="DEBUG")
```

### Verificar logs específicos

```bash
# Erros
grep -i error logs/api.log

# Requisições lentas
grep -i "duration" logs/api.log | awk '$NF > 1.0'

# Módulos específicos
grep "stt\|tts\|llm" logs/api.log
```

## Obter Ajuda

Se o problema persistir:

1. Colete informações:
   - Versão do Python: `python --version`
   - Versão do projeto: `git rev-parse HEAD`
   - Logs relevantes
   - Stack trace completo

2. Abra uma issue no GitHub com:
   - Descrição do problema
   - Passos para reproduzir
   - Logs e stack traces
   - Ambiente (OS, Python version, etc.)
