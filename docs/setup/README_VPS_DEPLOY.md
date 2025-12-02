# 🖥️ Deploy na VPS - Guia Completo

## ✅ Sim! Já está implementado

O `docker-manager` já tem suporte completo para deploy na VPS usando SSH + Docker.

## 🚀 Como Usar

### Opção 1: Comando Integrado (Recomendado)

```bash
./main.sh deploy:vps
```

### Opção 2: Script Direto

```bash
./deploy-vps.sh
```

### Opção 3: Python API

```python
from vendor.docker-manager.docker_api import DockerAPI

api = DockerAPI(
    platform="vps",
    vps_host="54.37.225.188",
    vps_user="ubuntu",
    ssh_key_path="~/.ssh/id_rsa",
    container_name="parle-backend"
)

# Sincronizar arquivos
api.sync_files(Path("."))

# Garantir container rodando
api.ensure_container_running()
```

## 📋 Pré-requisitos

### 1. Acesso SSH à VPS

```bash
# Testar conexão
ssh usuario@seu-vps.com

# Ou com chave específica
ssh -i ~/.ssh/id_rsa usuario@seu-vps.com
```

### 2. Docker Instalado na VPS

```bash
# Na VPS, verificar Docker
ssh usuario@seu-vps.com "docker --version"

# Se não tiver, instalar:
ssh usuario@seu-vps.com "curl -fsSL https://get.docker.com | sh"
```

### 3. Container Criado na VPS

**Primeira vez - Setup inicial:**

```bash
# Na VPS, construir imagem
ssh usuario@seu-vps.com "cd /caminho/do/projeto && docker build -f docker/Dockerfile -t parle-backend:latest ."

# Criar container
ssh usuario@seu-vps.com "docker create --name parle-backend \
    -v /tmp/parle-backend-sync:/tmp/parle-backend-sync \
    -v pip-cache:/root/.cache/pip \
    parle-backend:latest"

# Iniciar container
ssh usuario@seu-vps.com "docker start parle-backend"
```

**Ou usar script automático (se existir):**

```bash
./setup_vps.sh
```

## ⚙️ Configuração

### Variáveis de Ambiente

```bash
# Configurar antes de executar
export VPS_HOST="54.37.225.188"
export VPS_USER="ubuntu"
export SSH_KEY_PATH="~/.ssh/id_rsa"
export CONTAINER_NAME="parle-backend"
export WORKSPACE_PATH="/workspace"
```

### Arquivo de Configuração

Criar `vendor/docker-manager/config.yaml`:

```yaml
ssh:
  host: "54.37.225.188"
  user: "ubuntu"
  key_path: "~/.ssh/id_rsa"

docker:
  container_name: "parle-backend"
  workspace_path: "/workspace"
  remote_sync_path: "/tmp/parle-backend-sync"
```

## 🔄 O que o Deploy Faz

1. **Testa conexão SSH** com a VPS
2. **Verifica se container está rodando** (inicia se necessário)
3. **Sincroniza arquivos** do projeto para VPS (rsync)
4. **Copia arquivos** para dentro do container
5. **Instala dependências** (se necessário)
6. **Mostra status** final do container

## 📊 Comparação: VPS vs GCP

| Aspecto | VPS | GCP Cloud Run |
|---------|-----|---------------|
| **Custo** | Fixo (~$5-20/mês) | Por uso (~$0.01-0.15/build) |
| **Velocidade deploy** | ~30s-2min | ~5-20 min (build) |
| **Setup inicial** | Médio | Fácil |
| **Manutenção** | Você gerencia | Gerenciado |
| **Escalabilidade** | Manual | Automática |
| **Melhor para** | Desenvolvimento, testes | Produção, CI/CD |

## 💡 Casos de Uso

### Desenvolvimento e Testes

**VPS é ideal para:**
- ✅ Testes rápidos (deploy em ~30s)
- ✅ Desenvolvimento ativo
- ✅ Ambientes de staging
- ✅ Custo fixo previsível

**Como usar:**
```bash
# Desenvolvimento ativo
./main.sh deploy:vps  # Deploy rápido na VPS

# Quando pronto para produção
./main.sh deploy:gcp:fast  # Deploy no GCP
```

### Produção

**GCP é ideal para:**
- ✅ Escalabilidade automática
- ✅ Alta disponibilidade
- ✅ CI/CD automatizado
- ✅ Sem manutenção de servidor

## 🛠️ Troubleshooting

### Erro: "Não foi possível conectar à VPS"

**Soluções:**
1. Verificar se VPS está online:
   ```bash
   ping seu-vps.com
   ```

2. Verificar credenciais SSH:
   ```bash
   ssh -v usuario@seu-vps.com
   ```

3. Verificar firewall (porta 22 deve estar aberta)

### Erro: "Container não está rodando"

**Soluções:**
1. Verificar se container existe:
   ```bash
   ssh usuario@seu-vps.com "docker ps -a | grep parle-backend"
   ```

2. Criar container se não existir:
   ```bash
   ssh usuario@seu-vps.com "docker create --name parle-backend ..."
   ```

3. Iniciar container:
   ```bash
   ssh usuario@seu-vps.com "docker start parle-backend"
   ```

### Erro: "Falha ao sincronizar arquivos"

**Soluções:**
1. Verificar permissões SSH
2. Verificar espaço em disco na VPS
3. Verificar se rsync está instalado na VPS

## 📝 Exemplos Práticos

### Deploy Rápido

```bash
# Configurar uma vez
export VPS_HOST="54.37.225.188"
export VPS_USER="ubuntu"

# Deploy
./main.sh deploy:vps
```

### Deploy com Configuração Customizada

```bash
VPS_HOST="meu-vps.com" \
VPS_USER="deploy" \
SSH_KEY_PATH="~/.ssh/deploy_key" \
CONTAINER_NAME="parle-backend-prod" \
./deploy-vps.sh
```

### Deploy via Python

```python
from pathlib import Path
from vendor.docker-manager.docker_api import DockerAPI

api = DockerAPI(
    platform="vps",
    vps_host="54.37.225.188",
    vps_user="ubuntu",
    container_name="parle-backend"
)

# Deploy completo
api.ensure_container_running()
api.sync_files(Path("."))
api.strategy.copy_to_container()
api.strategy.install_dependencies()

# Executar testes
result = api.run_tests("tests/", "-v")
print(result)
```

## 🎯 Workflow Recomendado

### Desenvolvimento

```bash
# 1. Desenvolver localmente
vim src/api/main.py

# 2. Deploy rápido na VPS para testar
./main.sh deploy:vps  # ~30s-2min

# 3. Testar na VPS
ssh usuario@vps "docker exec parle-backend pytest tests/ -v"

# 4. Quando pronto, deploy em produção
./main.sh deploy:gcp:fast  # ~5-8 min
```

### CI/CD

```yaml
# .github/workflows/deploy.yml
on:
  push:
    branches: [ main ]

jobs:
  deploy-vps:
    # Deploy em staging (VPS)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to VPS
        run: ./deploy-vps.sh
  
  deploy-gcp:
    # Deploy em produção (GCP)
    needs: deploy-vps
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to GCP
        run: ./main.sh deploy:gcp:fast
```

## ✅ Checklist

- [ ] VPS configurada e acessível via SSH
- [ ] Docker instalado na VPS
- [ ] Container criado na VPS
- [ ] Chave SSH configurada
- [ ] Variáveis de ambiente configuradas (ou config.yaml)
- [ ] Testado deploy: `./main.sh deploy:vps`

## 🎉 Conclusão

**Sim, você pode fazer deploy na VPS!**

- ✅ **Já implementado** no docker-manager
- ✅ **Comando simples**: `./main.sh deploy:vps`
- ✅ **Deploy rápido**: ~30s-2min
- ✅ **Ideal para desenvolvimento e testes**

**Use VPS para desenvolvimento rápido, GCP para produção!**
