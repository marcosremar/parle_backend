# Parle Backend

Backend do projeto Parle - Sistema de conversação multimodal.

## Estrutura do Projeto

Este projeto utiliza uma estrutura organizada com submódulos em `vendor/`:

- `vendor/skypilot/` - Submódulo Git para gerenciamento de recursos na nuvem
- `vendor/nomad` - Executável do Nomad (não versionado, instalado via script)

## Configuração Inicial

### 1. Clonagem com Submódulos

```bash
git clone <url-do-repositorio>
git submodule update --init --recursive
```

### 2. Instalação de Dependências

```bash
# Instalar dependências Python
pip install -r requirements.txt

# Ou usar o script de setup
./setup.sh
```

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

### Arquivos Ignorados

O arquivo `.gitignore` está configurado para ignorar:
- Arquivos Python compilados (`__pycache__/`, `*.pyc`)
- Ambientes virtuais (`venv/`, `.env`)
- Logs e arquivos temporários
- Executável do Nomad (`vendor/nomad`)
- Arquivos de banco de dados locais
- Arquivos de configuração com segredos

## Deploy com Nomad

1. Iniciar o agente Nomad em modo desenvolvimento:
   ```bash
   nomad agent -dev -bind=0.0.0.0
   ```

2. Deploy dos serviços:
   ```bash
   nomad job run <servico>.nomad
   ```

3. Verificar status:
   ```bash
   nomad job status
   ```

## Serviços Disponíveis

- `user-service.nomad` - Serviço de usuários
- `conversation-history.nomad` - Histórico de conversas
- `database-service.nomad` - Serviço de banco de dados
- `external-llm.nomad` - LLM externo
- `external-stt.nomad` - STT externo
- `external-tts.nomad` - TTS externo
- `file-storage.nomad` - Armazenamento de arquivos
- `neural-codec.nomad` - Codec neural
- `scenarios.nomad` - Cenários
- `websocket.nomad` - WebSocket

## Contribuição

1. Crie uma branch para sua feature: `git checkout -b feature/nome-da-feature`
2. Faça commit das mudanças: `git commit -am 'Adiciona nova feature'`
3. Push para a branch: `git push origin feature/nome-da-feature`
4. Abra um Pull Request

## Licença

Ver arquivo LICENSE.txt
