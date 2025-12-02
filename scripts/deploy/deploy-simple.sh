#!/bin/bash
# Versão simplificada do deploy para testar apenas criação de container

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}🚀 Deploy Simplificado - Parle Backend${NC}"
echo "========================================"

# Configurações
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"
WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.11-slim}"
FORCE_NEW_CONTAINER="${FORCE_NEW_CONTAINER:-false}"

echo -e "${BLUE}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   Container: $CONTAINER_NAME"
echo "   Image: $DOCKER_IMAGE"
echo "   Force: $FORCE_NEW_CONTAINER"
echo ""

# Testar SSH
echo -e "${YELLOW}1. Testando SSH...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH OK${NC}"
else
    echo -e "${RED}❌ SSH falhou${NC}"
    exit 1
fi

# Forçar novo container se solicitado
if [ "$FORCE_NEW_CONTAINER" = "true" ]; then
    echo ""
    echo -e "${YELLOW}2. Removendo container antigo...${NC}"
    ssh -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_HOST" "sudo docker stop $CONTAINER_NAME 2>/dev/null; sudo docker rm $CONTAINER_NAME 2>/dev/null" || true
    echo -e "${GREEN}✅ Container removido${NC}"
fi

# Criar container
echo ""
echo -e "${YELLOW}3. Criando container...${NC}"
ssh -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_HOST" "sudo mkdir -p /tmp/lca-sync"
CREATE_CMD="sudo docker create --name $CONTAINER_NAME -v /tmp/lca-sync:$WORKSPACE_PATH --restart unless-stopped $DOCKER_IMAGE sleep 3600"
echo "Executando: $CREATE_CMD"
if ssh -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_HOST" "$CREATE_CMD"; then
    echo -e "${GREEN}✅ Container criado${NC}"
else
    echo -e "${RED}❌ Falha na criação${NC}"
    exit 1
fi

# Iniciar container
echo ""
echo -e "${YELLOW}4. Iniciando container...${NC}"
if ssh -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_HOST" "sudo docker start $CONTAINER_NAME"; then
    echo -e "${GREEN}✅ Container iniciado${NC}"
else
    echo -e "${RED}❌ Falha ao iniciar${NC}"
    exit 1
fi

# Verificar status
echo ""
echo -e "${YELLOW}5. Verificando status...${NC}"
STATUS=$(ssh -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_HOST" "sudo docker ps --filter name=$CONTAINER_NAME --format '{{.Names}}:{{.Status}}'")
if echo "$STATUS" | grep -q "$CONTAINER_NAME"; then
    echo -e "${GREEN}✅ Container funcionando: $STATUS${NC}"
else
    echo -e "${RED}❌ Container não está funcionando${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}🎉 Deploy simplificado concluído!${NC}"
echo ""
echo "Container criado e funcionando na VPS!"
