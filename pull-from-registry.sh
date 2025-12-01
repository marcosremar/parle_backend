#!/bin/bash
# Script para pull rápido da imagem do registry na VPS

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Configurações
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
REGISTRY="${REGISTRY:-docker.io}"
REGISTRY_USER="${REGISTRY_USER:-}"
IMAGE_NAME="${IMAGE_NAME:-parle-backend}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"

echo -e "${BLUE}⚡ Pull Rápido do Registry${NC}"
echo "================================================================================"
echo ""

# Construir nome completo da imagem
if [ "$REGISTRY" = "docker.io" ]; then
    if [ -z "$REGISTRY_USER" ]; then
        echo -e "${RED}❌ REGISTRY_USER não definido${NC}"
        echo ""
        echo -e "${YELLOW}💡 Configure:${NC}"
        echo "   export REGISTRY_USER=\"seu-usuario-dockerhub\""
        exit 1
    fi
    FULL_IMAGE_NAME="$REGISTRY_USER/$IMAGE_NAME:$IMAGE_TAG"
elif [ "$REGISTRY" = "gcr.io" ]; then
    GCP_PROJECT="${GCP_PROJECT:-avian-computer-477918-j9}"
    FULL_IMAGE_NAME="gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG"
else
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
fi

echo -e "${YELLOW}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Image: $FULL_IMAGE_NAME"
echo "   Container: $CONTAINER_NAME"
echo ""

# Verificar conexão SSH
echo -e "${BLUE}🔌 Testando conexão SSH...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conexão SSH OK${NC}"
else
    echo -e "${RED}❌ Não foi possível conectar à VPS${NC}"
    exit 1
fi
echo ""

# Limpar container anterior (se existir)
echo -e "${YELLOW}🧹 Limpando container anterior...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
" > /dev/null 2>&1
echo -e "${GREEN}✅ Limpeza concluída${NC}"
echo ""

# Pull
echo "================================================================================"
echo -e "${BLUE}📥 Pulling imagem do registry...${NC}"
echo "================================================================================"
echo ""

start_pull=$(date +%s.%N)

ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker pull $FULL_IMAGE_NAME" 2>&1 | \
    tee /tmp/docker_pull.log | \
    grep -E "(Pulling|Downloading|Extracting|Pull complete|ERROR)" || true

pull_exit=${PIPESTATUS[0]}
end_pull=$(date +%s.%N)
pull_duration=$(echo "$end_pull - $start_pull" | bc)

if [ $pull_exit -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Pull falhou${NC}"
    echo ""
    echo -e "${YELLOW}💡 Verifique:${NC}"
    echo "   - Imagem existe no registry: docker search $REGISTRY_USER/$IMAGE_NAME"
    echo "   - Autenticação: docker login (se registry privado)"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Pull concluído: ${pull_duration}s${NC}"
echo ""

# Criar container
echo -e "${BLUE}📦 Criando container...${NC}"
start_create=$(date +%s.%N)

ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker create --name $CONTAINER_NAME \
        -v /tmp/parle-backend-sync:/tmp/parle-backend-sync \
        -v pip-cache:/root/.cache/pip \
        $FULL_IMAGE_NAME
" > /dev/null 2>&1

create_exit=$?
end_create=$(date +%s.%N)
create_duration=$(echo "$end_create - $start_create" | bc)

if [ $create_exit -ne 0 ]; then
    echo -e "${RED}❌ Criação do container falhou${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container criado: ${create_duration}s${NC}"
echo ""

# Iniciar container
echo -e "${BLUE}🚀 Iniciando container...${NC}"
start_start=$(date +%s.%N)

ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker start $CONTAINER_NAME" > /dev/null 2>&1

start_exit=$?
end_start=$(date +%s.%N)
start_duration=$(echo "$end_start - $start_start" | bc)

if [ $start_exit -ne 0 ]; then
    echo -e "${RED}❌ Inicialização do container falhou${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container iniciado: ${start_duration}s${NC}"
echo ""

# Resumo
total_time=$(echo "$pull_duration + $create_duration + $start_duration" | bc)
total_minutes=$(echo "scale=2; $total_time / 60" | bc)

echo "================================================================================"
echo -e "${GREEN}✅ DEPLOY CONCLUÍDO${NC}"
echo "================================================================================"
echo ""
echo -e "${BLUE}📊 TEMPOS:${NC}"
echo "   Pull: ${pull_duration}s"
echo "   Create: ${create_duration}s"
echo "   Start: ${start_duration}s"
echo -e "${GREEN}   Total: ${total_time}s (${total_minutes} min)${NC}"
echo ""
echo -e "${GREEN}💡 Comandos úteis:${NC}"
echo "   Ver logs: ssh $VPS_USER@$VPS_HOST 'docker logs $CONTAINER_NAME'"
echo "   Status: ssh $VPS_USER@$VPS_HOST 'docker ps -a | grep $CONTAINER_NAME'"
echo ""
