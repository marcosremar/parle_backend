# Guia de Deployment

**Última atualização:** 28/11/2025  
**Status:** Migração completa para módulos

---

## 📋 Visão Geral

O sistema suporta dois modos de deployment:

1. **Modo Monolith** (Padrão) - Todos os módulos em um único processo
2. **Modo Microservices** - Serviços HTTP independentes

---

## 🚀 Modo Monolith (Recomendado)

### Características
- ✅ Todos os módulos em um único processo Python
- ✅ Chamadas diretas (sem HTTP)
- ✅ Melhor performance
- ✅ Mais simples de gerenciar

### Estrutura
```
src/
├── modules/          # Módulos principais (usados em modo monolith)
│   ├── speech/
│   ├── conversation/
│   ├── storage/
│   └── ...
└── services/         # Serviços HTTP standalone (não usados em monolith)
    ├── api_gateway/
    ├── webrtc/
    └── ...
```

### Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
export MONOLITH_MODE=true
export ENABLE_TUTORING_MODULES=true  # Opcional

# 3. Executar API principal
python src/api/main.py
```

### Endpoints
Todos os endpoints estão em `src/api/routers/api.py`:
- `/health` - Health check
- `/speech/stt/transcribe` - Speech-to-text
- `/speech/tts/synthesize` - Text-to-speech
- `/llm/generate` - LLM generation
- `/conversation` - Conversação
- `/auth/*` - Autenticação
- `/tutoring/*` - Tutoring (se habilitado)
- `/storage/*` - Storage

---

## 🔧 Modo Microservices

### Características
- ⚠️ Cada serviço roda como processo HTTP independente
- ⚠️ Comunicação via HTTP REST
- ⚠️ Mais complexo de gerenciar
- ⚠️ Útil para escalabilidade horizontal

### Serviços Disponíveis

#### Serviços Principais (Migrados)
Estes serviços têm módulos equivalentes, mas podem rodar como HTTP:

- `orchestrator` - Porta 8001
- `session` - Porta 8002
- `stt` - Porta 8003
- `tts` - Porta 8004
- `llm` - Porta 8005
- `user` - Porta 8006
- `conversation_store` - Porta 8007
- `file_storage` - Porta 8008
- `database` - Porta 8009

#### Serviços Standalone
Estes só existem como serviços HTTP:

- `api_gateway` - Porta 8010 (Gateway principal)
- `webrtc` - Porta 8011
- `webrtc_signaling` - Porta 8012
- `websocket` - Porta 8013

### Como Executar

```bash
# Executar cada serviço individualmente
python src/services/orchestrator/app_complete.py
python src/services/session/app_complete.py
python src/services/api_gateway/app_complete.py
# ... etc
```

Ou usar um gerenciador de processos como:
- `supervisord`
- `systemd`
- `docker/docker-compose.yml`

---

## 🔄 Migração de Microservices para Monolith

### Passo 1: Validar Migração
```bash
python scripts/validate_migration.py
```

### Passo 2: Configurar Modo Monolith
```bash
export MONOLITH_MODE=true
```

### Passo 3: Executar API Principal
```bash
python src/api/main.py
```

### Passo 4: Verificar
```bash
curl http://localhost:8000/health
```

---

## 📦 Docker Deployment

### Dockerfile (Monolith)
Localizado em `docker/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MONOLITH_MODE=true
ENV PYTHONPATH=/app

CMD ["python", "src/api/main.py"]
```

### docker-compose.yml
Localizado em `docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONOLITH_MODE=true
    command: python src/api/main.py

  orchestrator:
    build: .
    ports:
      - "8001:8001"
    environment:
      - MONOLITH_MODE=false
    command: python src/services/orchestrator/app_complete.py
```

---

## 🔐 Variáveis de Ambiente

### Obrigatórias
- `MONOLITH_MODE` - `true` para modo monolith, `false` para microservices

### Opcionais
- `ENABLE_TUTORING_MODULES` - `true` para habilitar módulos de tutoring
- `JWT_SECRET_KEY` - Chave secreta para JWT
- `DATABASE_URL` - URL do banco de dados
- `REDIS_URL` - URL do Redis

### Por Serviço (Microservices)
- `ORCHESTRATOR_PORT` - Porta do orchestrator
- `SESSION_PORT` - Porta do session
- `API_GATEWAY_PORT` - Porta do API Gateway
- ... etc

---

## 📊 Monitoramento

### Health Checks
```bash
# Monolith
curl http://localhost:8000/health

# Microservices
curl http://localhost:8001/health  # Orchestrator
curl http://localhost:8002/health  # Session
```

### Logs
Logs são gerenciados via `loguru`:
- Console (stdout)
- Arquivo: `logs/app.log`
- Rotação automática

---

## 🐛 Troubleshooting

### Módulo não encontrado
```bash
# Verificar se módulo existe
python -c "from src.modules import module_factory; module_factory.create('module_name')"
```

### Import errors
```bash
# Validar imports
python scripts/validate_migration.py
```

### Porta já em uso
```bash
# Verificar portas
lsof -i :8000
# Mudar porta via variável de ambiente
export API_PORT=8001
```

---

## 📝 Notas Importantes

1. **Modo Monolith é Recomendado:**
   - Melhor performance
   - Mais simples de gerenciar
   - Menos overhead de rede

2. **Microservices para Escalabilidade:**
   - Use apenas se precisar escalar serviços individualmente
   - Adiciona complexidade de rede e gerenciamento

3. **Compatibilidade:**
   - Ambos os modos são suportados
   - Migração é reversível
   - Fallbacks garantem compatibilidade

---

**Última atualização:** 28/11/2025
