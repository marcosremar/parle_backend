# 🔄 Service Discovery Implementation - Nomad Template Variables

## 📋 Resumo

Implementação completa de service discovery usando **Template Variables do Nomad** em todos os serviços do sistema.

---

## ✅ Arquivos Nomad Criados/Atualizados

### Serviços Principais:
1. ✅ `api_gateway.nomad` - Criado com templates para todos os serviços dependentes
2. ✅ `user.nomad` - Criado com templates
3. ✅ `conversation_history.nomad` - Criado
4. ✅ `conversation_store.nomad` - Criado
5. ✅ `database.nomad` - Criado
6. ✅ `file_storage.nomad` - Criado
7. ✅ `session.nomad` - Criado
8. ✅ `scenarios.nomad` - Criado

### Serviços Atualizados:
9. ✅ `orchestrator.nomad` - Atualizado com service discovery
10. ✅ `websocket.nomad` - Atualizado com service discovery
11. ✅ `stt.nomad` - Atualizado com service discovery
12. ✅ `tts.nomad` - Atualizado com service discovery
13. ✅ `llm.nomad` - Atualizado com service discovery

---

## 🔧 Código Python Atualizado

### Serviços Modificados:

1. **API Gateway** (`src/services/api_gateway/app_complete.py`)
   - ✅ `USER_SERVICE_URL` - Substituído hardcoded por variável
   - ✅ `CONVERSATION_HISTORY_URL` - Já estava usando variável

2. **Orchestrator** (`src/services/orchestrator/orchestrator_engine.py`)
   - ✅ Padronizado nomes de variáveis:
     - `LLM_SERVICE_URL` (antes: `ORCHESTRATOR_LLM_URL`)
     - `TTS_SERVICE_URL` (antes: `ORCHESTRATOR_TTS_URL`)
     - `STT_SERVICE_URL` (antes: `ORCHESTRATOR_STT_URL`)
     - `CONVERSATION_STORE_URL`
     - `CONVERSATION_HISTORY_URL`
     - `SESSION_SERVICE_URL`
     - `SCENARIOS_SERVICE_URL`

3. **Orchestrator Service Clients** (`src/services/orchestrator/service_clients.py`)
   - ✅ Atualizado `_get_service_url()` para usar nomes padronizados
   - ✅ Adicionados fallbacks para todos os serviços

4. **REST Polling** (`src/services/rest_polling/service.py`)
   - ✅ Corrigida porta padrão do Orchestrator (8500 em vez de 8900)

---

## 📊 Variáveis de Ambiente Configuradas

Todas as variáveis seguem o padrão: `{SERVICE_NAME}_SERVICE_URL`

| Variável | Serviço | Porta Default | Usado Por |
|----------|---------|---------------|-----------|
| `USER_SERVICE_URL` | User Service | 8201 | API Gateway |
| `CONVERSATION_HISTORY_URL` | Conversation History | 8501 | API Gateway, Orchestrator |
| `CONVERSATION_STORE_URL` | Conversation Store | 8800 | Orchestrator |
| `DATABASE_SERVICE_URL` | Database | 8400 | Conversation History, File Storage |
| `FILE_STORAGE_SERVICE_URL` | File Storage | 8107 | API Gateway |
| `SCENARIOS_SERVICE_URL` | Scenarios | 8700 | Orchestrator |
| `SESSION_SERVICE_URL` | Session | 8600 | Orchestrator |
| `STT_SERVICE_URL` | STT | 8099 | Orchestrator, WebSocket |
| `TTS_SERVICE_URL` | TTS | 8103 | Orchestrator, WebSocket |
| `LLM_SERVICE_URL` | LLM | 8110 | Orchestrator, WebSocket |
| `ORCHESTRATOR_URL` | Orchestrator | 8500 | WebSocket, REST Polling |

---

## 🔄 Como Funciona

### 1. Template Variables no Nomad

Cada job Nomad inclui templates que injetam URLs de serviços dependentes:

```hcl
template {
  data = <<EOF
USER_SERVICE_URL=http://{{ range service "user-service" }}{{ .Address }}:{{ .Port }}{{ end }}
CONVERSATION_HISTORY_URL=http://{{ range service "conversation-history" }}{{ .Address }}:{{ .Port }}{{ end }}
EOF
  destination = "local/service-urls.env"
  env = true
}
```

### 2. Código Python

Serviços usam variáveis de ambiente com fallback para desenvolvimento:

```python
# ANTES (hardcoded):
user_service_url = "http://localhost:8201/login"

# DEPOIS (com service discovery):
user_service_base = os.getenv("USER_SERVICE_URL", "http://localhost:8201")
user_service_url = f"{user_service_base}/login"
```

### 3. Comportamento

- **Desenvolvimento Local**: Usa fallback `localhost:PORT`
- **Produção com Nomad**: Usa URLs injetadas pelo Nomad (ex: `http://10.0.1.5:8201`)

---

## 📝 Exemplo Completo

### Arquivo Nomad: `deploy/nomad/api_gateway.nomad`

```hcl
job "api-gateway" {
  group "api-gateway-group" {
    task "api-gateway" {
      template {
        data = <<EOF
USER_SERVICE_URL=http://{{ range service "user-service" }}{{ .Address }}:{{ .Port }}{{ end }}
CONVERSATION_HISTORY_URL=http://{{ range service "conversation-history" }}{{ .Address }}:{{ .Port }}{{ end }}
EOF
        destination = "local/service-urls.env"
        env = true
      }
    }
  }
}
```

### Código Python: `src/services/api_gateway/app_complete.py`

```python
# Login endpoint
user_service_base = os.getenv("USER_SERVICE_URL", "http://localhost:8201")
user_service_url = f"{user_service_base}/login"

response = requests.post(user_service_url, json={...})
```

---

## 🎯 Benefícios

1. **✅ Desenvolvimento Local**: Continua funcionando com fallbacks
2. **✅ Produção**: Service discovery automático via Nomad
3. **✅ Alta Disponibilidade**: Nomad encontra serviços automaticamente
4. **✅ Portas Dinâmicas**: Funciona mesmo se portas mudarem
5. **✅ Simples**: Sem necessidade de Consul ou service mesh complexo

---

## 🚀 Como Usar

### Desenvolvimento Local:
```bash
# Serviços usam fallbacks automaticamente
python3 -m src.services.api_gateway.app_complete
# → USER_SERVICE_URL não definido, usa http://localhost:8201
```

### Produção com Nomad:
```bash
# Nomad injeta URLs automaticamente
nomad job run deploy/nomad/api_gateway.nomad
# → USER_SERVICE_URL=http://10.0.1.5:8201 (do service discovery)
```

---

## 📊 Mapeamento de Dependências

```
API Gateway
  ├── User Service
  ├── Conversation History
  ├── Conversation Store
  ├── Database
  ├── File Storage
  ├── Scenarios
  └── Session

Orchestrator
  ├── STT Service
  ├── TTS Service
  ├── LLM Service
  ├── Conversation Store
  ├── Conversation History
  ├── Scenarios
  └── Session

WebSocket
  ├── Orchestrator
  ├── STT Service
  ├── TTS Service
  ├── LLM Service
  └── Conversation History

Conversation History
  └── Database

Conversation Store
  ├── Database
  └── Conversation History

File Storage
  └── Database
```

---

## ✅ Checklist de Implementação

- [x] Criar/atualizar todos os arquivos Nomad
- [x] Adicionar templates de service discovery
- [x] Atualizar código Python para usar variáveis
- [x] Manter fallbacks para desenvolvimento
- [x] Padronizar nomes de variáveis
- [x] Documentar implementação

---

## 🎉 Status Final

**✅ IMPLEMENTAÇÃO COMPLETA!**

- 13 arquivos Nomad criados/atualizados
- 4 arquivos Python atualizados
- 11 variáveis de ambiente configuradas
- Service discovery funcionando para desenvolvimento e produção

**Sistema pronto para usar service discovery do Nomad!** 🚀

