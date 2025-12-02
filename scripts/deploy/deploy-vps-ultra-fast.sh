#!/bin/bash
# Deploy ultra-rápido - apenas sincronização de código, sem rebuild

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}⚡ Deploy Ultra-Rápido - Parle Backend${NC}"
echo "================================================================================"
echo ""

# Configurações
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"

echo -e "${BLUE}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Container: $CONTAINER_NAME"
echo ""

# 1. Verificar se container está rodando
echo -e "${BLUE}🔍 Verificando container...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "sudo docker ps | grep $CONTAINER_NAME" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Container rodando${NC}"
else
    echo -e "${RED}❌ Container não está rodando${NC}"
    echo -e "${YELLOW}💡 Execute primeiro: ./setup_vps.sh${NC}"
    exit 1
fi

# 2. Sincronizar apenas código fonte (excluindo tudo o que não precisa)
echo ""
echo -e "${BLUE}📤 Sincronizando código fonte ultra-rápido...${NC}"

# Lista de arquivos/pastas essenciais para sincronizar
ESSENTIAL_FILES=(
    "src/"
    "requirements.txt"
    "environment.yml"
    "pyproject.toml"
    "main.sh"
    "docker/"
    "config/"
)

# Criar rsync com excludes agressivos
rsync_cmd=(
    "rsync" "-avz" "--delete"
    "--exclude" ".git"
    "--exclude" "__pycache__"
    "--exclude" "*.pyc"
    "--exclude" "*.pyo"
    "--exclude" ".pytest_cache"
    "--exclude" ".mypy_cache"
    "--exclude" ".ruff_cache"
    "--exclude" "*.egg-info"
    "--exclude" "node_modules"
    "--exclude" ".venv"
    "--exclude" "venv"
    "--exclude" "temp"
    "--exclude" "*.log"
    "--exclude" ".env"
    "--exclude" ".DS_Store"
    "--exclude" "*.swp"
    "--exclude" ".idea"
    "--exclude" ".vscode"
    "--exclude" "dist"
    "--exclude" "build"
    "--exclude" ".coverage"
    "--exclude" "htmlcov"
    "--exclude" "logs"
    "--exclude" "references"
    "--exclude" "myproject"
    "--exclude" "debug_*.json"
    "--exclude" "file*.txt"
    "--exclude" "test_*.txt"
    "--exclude" "hello.py"
    "--exclude" "blaxel-sandbox.yaml"
    "--exclude" ".docs/similar-projects"
    "--exclude" ".agent/chromadb/"
    # Excludes agressivos do parle_backend
    "--exclude" "docs/"
    "--exclude" "data/"
    "--exclude" "*.md"
    "--exclude" "BUILD_*.md"
    "--exclude" "CI_CD_*.md"
    "--exclude" "COST_*.md"
    "--exclude" "DEPLOY_*.md"
    "--exclude" "FAST_BUILD_*.md"
    "--exclude" "GCP_*.md"
    "--exclude" "README_*.md"
    "--exclude" "REGISTRY_*.md"
    "--exclude" "STARTUP_*.md"
    "--exclude" "TEST_*.md"
    "--exclude" "VPS_*.md"
    "--exclude" "VPS_VS_*.md"
    "--exclude" "htmlcov/"
    "--exclude" ".coverage"
    "--exclude" "tmp/"
    "--exclude" "vm/"
    "--exclude" "tests/output/"
    "--exclude" "*.html"
    "--exclude" "vendor/"
    "--exclude" "scripts/"
    "--exclude" "tests/"
    "-e" "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no"
    "$PROJECT_DIR/"
    "$VPS_USER@$VPS_HOST:/tmp/parle-sync/"
)

echo -e "${CYAN}→${NC} Executando rsync ultra-rápido..."
"${rsync_cmd[@]}"

echo -e "${GREEN}✅ Sincronizado${NC}"

# 3. Copiar código para container (sem instalar dependências)
echo ""
echo -e "${BLUE}📋 Copiando código para container...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
sudo docker cp /tmp/parle-sync/src $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/requirements.txt $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/environment.yml $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/pyproject.toml $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/main.sh $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/docker $CONTAINER_NAME:/workspace/
sudo docker cp /tmp/parle-sync/config $CONTAINER_NAME:/workspace/
"

echo -e "${GREEN}✅ Código copiado${NC}"

# 4. Reiniciar serviço (se já tiver dependências)
echo ""
echo -e "${BLUE}🔄 Reiniciando serviço...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
sudo docker exec $CONTAINER_NAME pkill -f uvicorn || true
sudo docker exec $CONTAINER_NAME bash -c 'cd /workspace && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000' &
"

echo -e "${GREEN}✅ Serviço reiniciado${NC}"

# 5. Status final
echo ""
echo -e "${BLUE}📊 Status final:${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
echo 'Container:'
sudo docker ps | grep $CONTAINER_NAME
echo ''
echo 'Logs (últimas 5 linhas):'
sudo docker logs --tail 5 $CONTAINER_NAME
"

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Deploy Ultra-Rápido concluído!${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}💡 Se precisar instalar dependências, execute:${NC}"
echo "   ssh $VPS_USER@$VPS_HOST 'sudo docker exec $CONTAINER_NAME pip install -r /workspace/requirements.txt'"
echo ""
echo -e "${YELLOW}🌐 Acesse:${NC}"
echo "   API: http://$VPS_HOST:8000"
echo "   Docs: http://$VPS_HOST:8000/docs"
echo ""
