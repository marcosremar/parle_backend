# Parle Backend

Backend do projeto Parle - Sistema de conversação multimodal.

## 🐍 Ambiente de Desenvolvimento

Este projeto usa **Miniconda** como ambiente padrão, otimizado para MacBook M1 com pacotes pré-compilados para Apple Silicon.

## 🚀 Início Rápido

O projeto inclui um script principal `main.sh` que facilita todas as operações:

```bash
# Configurar ambiente Miniconda (primeira vez)
./main.sh setup

# Ativar ambiente conda manualmente
./main.sh conda-activate

# Testar instalação
./main.sh test

# Iniciar todos os serviços speech-to-speech
./main.sh start --all

# Iniciar serviços individuais
./main.sh start llm        # Serviço LLM (Python/Conda)
./main.sh start stt        # Serviço STT (Python/Conda)
./main.sh start tts        # Serviço TTS (Python/Conda)
./main.sh start orchestrator  # Orchestrator (Python/Conda)

# Ver status de todos os serviços
./main.sh status

# Abrir interface de demonstração
./main.sh demo

# Abrir dashboard de monitoramento
./main.sh monitor

# Executar benchmark de performance
./main.sh benchmark

# Abrir shell com conda ativado
./main.sh shell
```

Para ver todos os comandos disponíveis: `./main.sh help`

## 🎤 Sistema Speech-to-Speech

O Parle Backend inclui um sistema completo de conversação multimodal:

### Funcionalidades
- **🎙️ STT**: Speech-to-Text com OpenAI Whisper
- **🤖 LLM**: Language Model com GPT-2
- **🔊 TTS**: Text-to-Speech com Eleven Labs e HuggingFace
- **🎯 Orchestrator**: Pipeline completo STT → LLM → TTS

### Interface Web
- **Demonstração**: `./main.sh demo` - Interface completa com gravação
- **Monitoramento**: `./main.sh monitor` - Dashboard de status dos serviços

### Workflow Completo
```bash
# 1. Setup inicial
./main.sh setup

# 2. Iniciar todos os serviços
./main.sh start --all

# 3. Abrir demonstração
./main.sh demo

# 4. Testar performance
./main.sh benchmark
```

## Estrutura do Projeto

Este projeto utiliza uma estrutura organizada:

- `src/` - Código fonte
  - `src/core/` - Biblioteca core compartilhada
  - `src/services/` - Microserviços
- `deploy/nomad/` - Arquivos de configuração do Nomad para deploy
- `docs/` - Documentação do projeto
- `scripts/` - Scripts de automação e utilitários
- `tests/` - Testes end-to-end e fixtures
- `vendor/` - Submódulos e dependências
  - `vendor/skypilot/` - Submódulo Git para gerenciamento de recursos na nuvem
  - `vendor/nomad` - Executável do Nomad (não versionado, instalado via script)

## Configuração Inicial

### 1. Clonagem com Submódulos

```bash
git clone <url-do-repositorio>
git submodule update --init --recursive
```

### 2. Instalação de Dependências

O script `main.sh setup` instala automaticamente o Miniconda e cria um ambiente conda otimizado para M1:

```bash
# Executar o script de setup (instala Miniconda e cria ambiente)
./main.sh setup

# Ativar ambiente conda
./main.sh conda-activate

# Ou manualmente
export PATH="$HOME/miniconda3/bin:$PATH"
conda activate parle_backend
```

**Nota**: O Miniconda é otimizado para Apple Silicon com pacotes pré-compilados, garantindo melhor performance e compatibilidade.

### 3. Instalação do Nomad

O Nomad não é versionado neste repositório. Para instalar:

1. Baixe o binário do Nomad em: https://developer.hashicorp.com/nomad/downloads
2. Coloque o executável em `vendor/nomad` (ou em algum lugar no PATH)

### 4. Configuração dos Submódulos

#### Skypilot

O submódulo skypilot já está configurado. Para atualizar:

```bash
git submodule update --remote vendor/skypilot
```

## Desenvolvimento

### Trabalhando com Submódulos

- **Atualizar submódulos**: `git submodule update --remote`
- **Commit de mudanças em submódulos**: Faça commit no submódulo primeiro, depois no projeto principal
- **Adicionar novo submódulo**: `git submodule add <url> vendor/<nome>`

### Trabalhando com Conda

```bash
# Ativar ambiente
./main.sh conda-activate

# Ou manualmente
export PATH="$HOME/miniconda3/bin:$PATH"
conda activate parle_backend

# Instalar novas dependências
conda install -c conda-forge <pacote>

# Ver ambiente ativo
conda info --envs

# Desativar ambiente
conda deactivate
```

### Arquivos Ignorados

O arquivo `.gitignore` está configurado para ignorar:
- Arquivos Python compilados (`__pycache__/`, `*.pyc`)
- Ambientes conda (`miniconda3/`, `envs/`)
- Ambientes virtuais (`venv/`, `.env`)
- Logs e arquivos temporários (`*.log`)
- Executável do Nomad (`vendor/nomad`)
- Arquivos de banco de dados locais
- Arquivos de configuração com segredos
- Modelos de ML (`models/`, `*.safetensors`)

## Verificação e Testes

### Testar Instalação

Execute o script de teste para verificar se tudo está configurado corretamente:

```bash
./scripts/test_installation.sh
```

Este script verifica:
- ✅ Python 3.12 instalado
- ✅ Ambiente virtual criado
- ✅ Dependências instaladas
- ✅ Estrutura de diretórios
- ✅ Imports Python funcionando
- ✅ Nomad instalado (opcional)

## Deploy e Gerenciamento de Serviços

### Serviços Python (Miniconda)

Os serviços Python principais usam Miniconda e são gerenciados pelo `main.sh`:

```bash
# Iniciar serviço Python
./main.sh start llm
./main.sh start api_gateway
./main.sh start user

# Ver status (mistura todos os serviços)
./main.sh status

# Parar serviço Python
./main.sh stop llm
```

### Serviços Nomad

Os serviços Nomad (outras tecnologias) são gerenciados pelo `scripts/nomad.sh`:

```bash
# Listar serviços disponíveis
./scripts/nomad.sh list

# Iniciar um serviço Nomad
./scripts/nomad.sh start external-stt
./scripts/nomad.sh start scenarios

# Iniciar TODOS os serviços Nomad
./scripts/nomad.sh start-all

# Ver status dos serviços Nomad
./scripts/nomad.sh status

# Ver logs de um serviço Nomad
./scripts/nomad.sh logs external-stt

# Parar serviço Nomad
./scripts/nomad.sh stop external-stt
```

### Deploy Manual

1. Iniciar o agente Nomad em modo desenvolvimento:
   ```bash
   nomad agent -dev -bind=0.0.0.0
   ```

2. Deploy dos serviços (execute a partir da raiz do projeto):
   ```bash
   nomad job run deploy/nomad/api-gateway.nomad
   nomad job run deploy/nomad/user-service.nomad
   # ... etc
   ```

3. Verificar status:
   ```bash
   nomad job status
   ```

Para mais detalhes sobre os serviços disponíveis, consulte `deploy/nomad/README.md`.

## 📝 Logs e Monitoramento

O projeto utiliza uma abordagem nativa e eficiente para logs, sem necessidade de bibliotecas adicionais complexas.

### Como funciona

1. **Aplicação (Python)**: 
   - Utilizamos a biblioteca `loguru` em todos os serviços.
   - Os logs são enviados para `stdout` (saída padrão) e `stderr` (erro padrão).
   - Não há necessidade de configurar arquivos de log manualmente na aplicação.

2. **Infraestrutura (Nomad)**:
   - O Nomad captura automaticamente os streams `stdout` e `stderr`.
   - Os logs são rotacionados automaticamente conforme configuração nos arquivos `.nomad`:
     ```hcl
     logs {
       max_files     = 10  # Mantém os últimos 10 arquivos
       max_file_size = 10  # Tamanho máximo de 10MB por arquivo
     }
     ```

### Visualizando Logs

Você pode visualizar os logs de qualquer serviço em tempo real:

```bash
# Ver logs de uma alocação específica
nomad alloc logs -f <alloc-id>

# Ver logs pelo nome do job (mais fácil)
nomad alloc logs -job api-gateway
nomad alloc logs -job user-service

# Ver logs de erro (stderr)
nomad alloc logs -stderr -job api-gateway
```

### Monitoramento

Para monitorar o status dos serviços:
```bash
./main.sh monitor
```

## Contribuição

1. Crie uma branch para sua feature: `git checkout -b feature/nome-da-feature`
2. Faça commit das mudanças: `git commit -am 'Adiciona nova feature'`
3. Push para a branch: `git push origin feature/nome-da-feature`
4. Abra um Pull Request

## Licença

Ver arquivo LICENSE.txt
