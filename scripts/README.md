# Scripts Auxiliares

Este diretório contém scripts para facilitar o desenvolvimento e deploy do Parle Backend.

## Scripts Disponíveis

### `test_installation.sh`

Testa se a instalação está correta e todas as dependências estão configuradas.

```bash
./scripts/test_installation.sh
```

**O que verifica:**
- ✅ Python 3.12 instalado
- ✅ Ambiente virtual criado e ativo
- ✅ Dependências Python instaladas (FastAPI, Uvicorn, etc.)
- ✅ Estrutura de diretórios (`src/core`, `src/services`, `deploy/nomad`)
- ✅ Arquivos importantes existem
- ✅ Imports Python funcionando
- ✅ Nomad instalado (opcional)

### `nomad.sh`

Script principal para gerenciar serviços no Nomad.

#### Comandos Disponíveis

**Listar serviços:**
```bash
./scripts/nomad.sh list
```

**Iniciar um serviço:**
```bash
./scripts/nomad.sh start api-gateway
./scripts/nomad.sh start user-service
```

**Iniciar TODOS os serviços:**
```bash
./scripts/nomad.sh start-all
```

**Ver status:**
```bash
./scripts/nomad.sh status
```

**Ver logs (seguir):**
```bash
./scripts/nomad.sh logs api-gateway
```

**Parar um serviço:**
```bash
./scripts/nomad.sh stop api-gateway
```

**Parar todos os serviços:**
```bash
./scripts/nomad.sh stop-all
```

**Ajuda:**
```bash
./scripts/nomad.sh help
```

## Exemplos de Uso

### Iniciar o API Gateway

```bash
# 1. Verificar instalação
./scripts/test_installation.sh

# 2. Iniciar Nomad (em outro terminal)
nomad agent -dev -bind=0.0.0.0

# 3. Iniciar API Gateway
./scripts/nomad.sh start api-gateway

# 4. Verificar status
./scripts/nomad.sh status

# 5. Ver logs
./scripts/nomad.sh logs api-gateway
```

### Iniciar Todos os Serviços

```bash
# Iniciar todos de uma vez
./scripts/nomad.sh start-all

# Ver status de todos
./scripts/nomad.sh status
```

### Workflow Completo

```bash
# 1. Setup inicial (primeira vez)
./setup.sh
source venv/bin/activate

# 2. Testar instalação
./scripts/test_installation.sh

# 3. Iniciar Nomad (terminal separado)
nomad agent -dev -bind=0.0.0.0

# 4. Iniciar serviços
./scripts/nomad.sh start-all

# 5. Monitorar
./scripts/nomad.sh status
./scripts/nomad.sh logs api-gateway

# 6. Parar tudo quando terminar
./scripts/nomad.sh stop-all
```

## Requisitos

- Python 3.12
- Ambiente virtual criado (`./setup.sh`)
- Nomad instalado (para comandos de deploy)

## Troubleshooting

### Nomad não encontrado

```bash
# macOS
brew install nomad

# Ou baixar de:
# https://developer.hashicorp.com/nomad/downloads
```

### Nomad não está rodando

O script `nomad.sh` detecta automaticamente se o Nomad não está rodando e oferece iniciar.

Ou inicie manualmente:
```bash
nomad agent -dev -bind=0.0.0.0
```

### Serviço não encontrado

Use `./scripts/nomad.sh list` para ver todos os serviços disponíveis.

### Erros de import

Certifique-se de que:
1. O ambiente virtual está ativo: `source venv/bin/activate`
2. As dependências estão instaladas: `pip install -r requirements.txt`
3. O PYTHONPATH está configurado: `export PYTHONPATH=src`

