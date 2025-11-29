# Configuração Centralizada

## 📋 Resumo

Todas as configurações do sistema foram centralizadas em um único arquivo YAML: `config/settings.yaml`.

## 🎯 Objetivo

- **Centralização**: Todas as configurações em um único lugar
- **Manutenibilidade**: Fácil de encontrar e modificar valores
- **Flexibilidade**: Suporte a variáveis de ambiente para override
- **Documentação**: YAML serve como documentação viva das configurações

## 📁 Estrutura

### Arquivo Principal
- **`config/settings.yaml`**: Arquivo centralizado com todas as configurações

### Código Atualizado
- **`src/core/config.py`**: Carrega configurações do YAML
- **`src/modules/conversation/orchestrator/constants.py`**: Carrega constantes do YAML

## 🔄 Ordem de Prioridade

As configurações são carregadas na seguinte ordem (maior para menor prioridade):

1. **Variáveis de Ambiente** (maior prioridade)
   - Exemplo: `LLM_API_KEY`, `DB_URL`, `SERVER_PORT`
   - Sempre sobrescrevem valores do YAML

2. **`config/settings.yaml`**
   - Valores padrão para desenvolvimento
   - Pode ser versionado no Git (sem secrets)

3. **Valores Padrão no Código**
   - Fallback se YAML não existir ou valor não estiver definido

## 📝 Seções do YAML

### `app`
- Informações do projeto (nome, versão)
- Ambiente (development, production)
- Debug mode

### `server`
- Host, porta, workers
- Log level
- CORS origins
- Upload size limits

### `database`
- URL de conexão
- Pool size
- Configurações de conexão

### `redis`
- URL, host, porta
- Configurações de conexão

### `llm`, `stt`, `tts`
- Providers
- Models
- API keys (via env vars)
- Timeouts
- Configurações específicas

### `auth`
- JWT configuration
- Password policies

### `monolith`
- Mode (monolith/microservices)
- Direct calls
- Shared resources

### `orchestrator`
- Cache configuration
- Audio settings
- Confidence thresholds
- Client defaults
- Heuristic analysis constants
- System prompts
- Service URLs

## 🔧 Como Usar

### Modificar Configurações

1. **Editar `config/settings.yaml`**:
   ```yaml
   server:
     port: 8080
     log_level: "DEBUG"
   ```

2. **Ou usar variáveis de ambiente** (recomendado para produção):
   ```bash
   export SERVER_PORT=8080
   export SERVER_LOG_LEVEL=DEBUG
   ```

### Acessar Configurações no Código

```python
from src.core.config import get_config

config = get_config()

# Acessar valores
print(config.server.port)
print(config.llm.provider)
print(config.database.url)
```

### Acessar Constantes do Orchestrator

```python
from src.modules.conversation.orchestrator.constants import (
    DEFAULT_SAMPLE_RATE,
    HEURISTIC_SHORT_RESPONSE_THRESHOLD,
    DEFAULT_MAX_RETRIES,
)

# Valores são carregados automaticamente do YAML
print(DEFAULT_SAMPLE_RATE)  # 16000 (ou valor do YAML)
```

## ✅ Vantagens

1. **Centralização**: Todas as configurações em um único arquivo
2. **Versionamento**: YAML pode ser versionado (sem secrets)
3. **Flexibilidade**: Variáveis de ambiente para override
4. **Documentação**: YAML serve como documentação
5. **Type Safety**: Pydantic valida tipos e valores
6. **Manutenibilidade**: Fácil de encontrar e modificar

## 🔒 Segurança

- **Secrets**: Nunca commitar API keys ou secrets no YAML
- **Produção**: Usar variáveis de ambiente para valores sensíveis
- **Validação**: Pydantic valida configurações em produção

## 📚 Exemplos

### Exemplo 1: Mudar porta do servidor

**Opção 1: YAML**
```yaml
server:
  port: 9000
```

**Opção 2: Variável de ambiente**
```bash
export SERVER_PORT=9000
```

### Exemplo 2: Configurar LLM

**YAML:**
```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.8
```

**API Key (sempre via env var):**
```bash
export LLM_API_KEY=sk-...
```

### Exemplo 3: Ajustar heurísticas

**YAML:**
```yaml
orchestrator:
  heuristics:
    short_response_threshold: 60
    long_response_threshold: 150
    analysis_confidence: 0.9
```

## 🚀 Próximos Passos

- [ ] Adicionar validação de schema YAML
- [ ] Criar `settings.example.yaml` para documentação
- [ ] Adicionar comentários explicativos no YAML
- [ ] Migrar outras configurações espalhadas para o YAML
