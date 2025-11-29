# Scripts Auxiliares

Este diretório contém scripts para facilitar o desenvolvimento e deploy do Parle Backend.

## Scripts Disponíveis

### `test_installation.sh`

Testa se a instalação está correta e todas as dependências estão configuradas.

```bash
./scripts/test_installation.sh
```

**O que verifica:**
- ✅ Python 3.11+ instalado
- ✅ Ambiente conda criado e ativo
- ✅ Dependências Python instaladas (FastAPI, Uvicorn, etc.)
- ✅ Estrutura de diretórios (`src/core`, `src/modules`, `src/api`)
- ✅ Arquivos importantes existem
- ✅ Imports Python funcionando
- ✅ Módulos principais disponíveis

## Exemplos de Uso

### Iniciar a API Principal

```bash
# 1. Verificar instalação
./scripts/test_installation.sh

# 2. Iniciar API Principal (monolito modular)
./main.sh start api

# 3. Verificar status
./main.sh status

# 4. Ver logs
./main.sh logs api
```

### Iniciar Todos os Serviços

```bash
# 1. Iniciar todos os serviços
./main.sh start --all

# 2. Verificar status
./main.sh status

# 3. Ver logs
./main.sh logs api
./main.sh logs websocket

# 4. Parar todos
./main.sh stop --all
```

### Workflow Completo

```bash
# 1. Setup inicial (primeira vez)
./main.sh setup

# 2. Ativar ambiente conda
./main.sh conda-activate

# 3. Testar instalação
./scripts/test_installation.sh

# 4. Iniciar API Principal
./main.sh start api

# 5. Iniciar WebSocket (opcional)
./main.sh start websocket

# 6. Monitorar
./main.sh status
./main.sh logs api

# 7. Parar quando terminar
./main.sh stop api
./main.sh stop websocket
```

## Requisitos

- Python 3.11+
- Conda instalado (recomendado) ou ambiente virtual
- Dependências instaladas (`requirements.txt`)

## Troubleshooting

### Erros de import

Certifique-se de que:
1. O ambiente conda está ativo: `./main.sh conda-activate`
2. As dependências estão instaladas: `pip install -r requirements.txt`
3. O PYTHONPATH está configurado: `export PYTHONPATH=src`

### Serviço não inicia

1. Verifique se a porta está disponível: `lsof -i :8000`
2. Verifique os logs: `./main.sh logs api`
3. Verifique se o ambiente está ativo: `conda info --envs`

### Módulos não encontrados

1. Verifique se `src/modules/` existe
2. Verifique se `module_factory.py` está presente
3. Execute: `python -c "from src.modules import module_factory; print(module_factory)"`
