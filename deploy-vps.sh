#!/bin/bash
# Script para deploy na VPS usando docker-manager

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:/opt/homebrew/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}🚀 Deploy na VPS - Parle Backend${NC}"
echo "================================================================================"
echo ""

# Verificar se docker-manager existe
DOCKER_MANAGER_DIR="$PROJECT_DIR/vendor/docker-manager"
if [ ! -d "$DOCKER_MANAGER_DIR" ]; then
    echo -e "${RED}❌ docker-manager não encontrado em: $DOCKER_MANAGER_DIR${NC}"
    exit 1
fi

# Configurações padrão (podem ser sobrescritas por variáveis de ambiente)
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"
WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"

echo -e "${BLUE}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Container: $CONTAINER_NAME"
echo "   SSH Key: $SSH_KEY_PATH"
echo ""

# Verificar se chave SSH existe
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo -e "${YELLOW}⚠️  Chave SSH não encontrada: $SSH_KEY_PATH${NC}"
    echo ""
    echo -e "${CYAN}💡 Opções:${NC}"
    echo "   1. Criar chave SSH: ssh-keygen -t rsa"
    echo "   2. Copiar chave para VPS: ssh-copy-id $VPS_USER@$VPS_HOST"
    echo "   3. Definir SSH_KEY_PATH: export SSH_KEY_PATH=/caminho/para/chave"
    echo ""
    read -p "Continuar mesmo assim? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verificar conectividade SSH
echo -e "${BLUE}🔌 Testando conexão SSH...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conexão SSH OK${NC}"
else
    echo -e "${RED}❌ Não foi possível conectar à VPS${NC}"
    echo ""
    echo -e "${YELLOW}💡 Verifique:${NC}"
    echo "   - Host está correto: $VPS_HOST"
    echo "   - Usuário está correto: $VPS_USER"
    echo "   - Chave SSH está correta: $SSH_KEY_PATH"
    echo "   - Firewall permite conexão SSH (porta 22)"
    exit 1
fi

echo ""
echo "================================================================================"
echo -e "${BLUE}🚀 Iniciando deploy...${NC}"
echo "================================================================================"
echo ""

# Mudar para diretório docker-manager
cd "$DOCKER_MANAGER_DIR"

# Criar script Python temporário para deploy
cat > /tmp/deploy_vps.py << EOF
#!/usr/bin/env python3
"""Script para deploy na VPS."""

import sys
from pathlib import Path

# Adicionar diretório ao path
docker_manager_path = Path(__file__).parent.parent
sys.path.insert(0, str(docker_manager_path))

from docker_api import DockerAPI

# Configurações
VPS_HOST = "$VPS_HOST"
VPS_USER = "$VPS_USER"
SSH_KEY_PATH = "$SSH_KEY_PATH"
CONTAINER_NAME = "$CONTAINER_NAME"
WORKSPACE_PATH = "$WORKSPACE_PATH"
PROJECT_ROOT = "$PROJECT_DIR"

print("🚀 Deploy na VPS")
print("=" * 80)
print()

# Criar API
api = DockerAPI(
    platform="vps",
    vps_host=VPS_HOST,
    vps_user=VPS_USER,
    ssh_key_path=SSH_KEY_PATH,
    container_name=CONTAINER_NAME,
    workspace_path=WORKSPACE_PATH,
    project_root=PROJECT_ROOT,
)

print("📋 Configuração:")
print(f"   Host: {VPS_HOST}")
print(f"   User: {VPS_USER}")
print(f"   Container: {CONTAINER_NAME}")
print()

# 1. Garantir que container está rodando
print("📦 Verificando container...")
if not api.ensure_container_running():
    print("❌ Container não está rodando")
    print("💡 Execute primeiro: ./setup_vps.sh")
    sys.exit(1)

print("✅ Container rodando")
print()

# 2. Sincronizar arquivos
print("📤 Sincronizando arquivos...")
try:
    metrics = api.sync_files(Path(PROJECT_ROOT))
    print(f"✅ Sincronizado: {metrics.total_files} arquivos em {metrics.duration_seconds:.2f}s")
except Exception as e:
    print(f"❌ Erro ao sincronizar: {e}")
    sys.exit(1)

print()

# 3. Copiar para container
print("📋 Copiando para container...")
if hasattr(api.strategy, 'copy_to_container'):
    api.strategy.copy_to_container()
    print("✅ Código copiado")
else:
    print("⚠️  Método copy_to_container não disponível, pulando...")

print()

# 4. Instalar dependências (se necessário)
print("📦 Verificando dependências...")
if hasattr(api.strategy, 'install_dependencies'):
    api.strategy.install_dependencies()
    print("✅ Dependências verificadas")
else:
    print("⚠️  Método install_dependencies não disponível, pulando...")

print()

# 5. Status final
print("📊 Status do container:")
status = api.container_status()
print(f"   Status: {status.get('status', 'unknown')}")
print(f"   Detalhes: {status.get('details', 'N/A')}")
print()

print("=" * 80)
print("✅ Deploy concluído!")
print()
print("💡 Comandos úteis:")
print(f"   Ver logs: ssh {VPS_USER}@{VPS_HOST} 'sudo docker logs {CONTAINER_NAME}'")
print(f"   Status: ssh {VPS_USER}@{VPS_HOST} 'sudo docker ps -a | grep {CONTAINER_NAME}'")
print(f"   Executar comando: ssh {VPS_USER}@{VPS_HOST} 'sudo docker exec {CONTAINER_NAME} <comando>'")
print()
EOF

chmod +x /tmp/deploy_vps.py

# Executar deploy
python3 /tmp/deploy_vps.py

DEPLOY_EXIT_CODE=$?

# Limpar script temporário
rm -f /tmp/deploy_vps.py

if [ $DEPLOY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ Deploy concluído com sucesso!${NC}"
    echo "================================================================================"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}❌ Deploy falhou${NC}"
    echo "================================================================================"
    echo ""
    exit 1
fi
