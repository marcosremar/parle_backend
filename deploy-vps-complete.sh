#!/bin/bash
# Script completo para deploy do parle_backend na VPS

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
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"
IMAGE_NAME="${IMAGE_NAME:-gcr.io/avian-computer-477918-j9/parle-backend:latest}"
PORT="${PORT:-8080}"

echo -e "${BLUE}🚀 Deploy Completo - parle_backend na VPS${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Container: $CONTAINER_NAME"
echo "   Image: $IMAGE_NAME"
echo "   Port: $PORT"
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

# Limpar container anterior
echo -e "${YELLOW}🧹 Limpando container anterior...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
" > /dev/null 2>&1
echo -e "${GREEN}✅ Limpeza concluída${NC}"
echo ""

# Garantir que imagem está disponível
echo -e "${BLUE}📥 Verificando imagem...${NC}"
if ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker images | grep -q parle-backend"; then
    echo -e "${GREEN}✅ Imagem já existe${NC}"
else
    echo -e "${YELLOW}📥 Fazendo pull da imagem...${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker pull $IMAGE_NAME" 2>&1 | tail -5
    echo -e "${GREEN}✅ Pull concluído${NC}"
fi
echo ""

# Criar container com configurações corretas
echo -e "${BLUE}📦 Criando container...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
docker create --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -v /tmp/parle-backend-sync:/tmp/parle-backend-sync \
    -v pip-cache:/root/.cache/pip \
    -e PYTHONPATH=/home/parle/.local/lib/python3.11/site-packages:/app \
    -e PATH=/home/parle/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    $IMAGE_NAME \
    sh -c 'cd /app && export PATH=/home/parle/.local/bin:\$PATH && export PYTHONPATH=/home/parle/.local/lib/python3.11/site-packages:\$PYTHONPATH && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080'
" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container criado${NC}"
else
    echo -e "${RED}❌ Falha ao criar container${NC}"
    exit 1
fi
echo ""

# Iniciar container
echo -e "${BLUE}🚀 Iniciando container...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker start $CONTAINER_NAME" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container iniciado${NC}"
else
    echo -e "${RED}❌ Falha ao iniciar container${NC}"
    exit 1
fi
echo ""

# Aguardar aplicação iniciar
echo -e "${BLUE}⏳ Aguardando aplicação iniciar (10 segundos)...${NC}"
sleep 10
echo ""

# Verificar status
echo -e "${BLUE}📊 Verificando status...${NC}"
STATUS=$(ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker ps -a --filter name=$CONTAINER_NAME --format '{{.Status}}'")

if echo "$STATUS" | grep -q "Up"; then
    echo -e "${GREEN}✅ Container está rodando!${NC}"
    echo ""
    echo -e "${BLUE}📋 Status:${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker ps | grep $CONTAINER_NAME"
    echo ""
    echo -e "${BLUE}📋 Logs (últimas 10 linhas):${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker logs --tail 10 $CONTAINER_NAME" 2>&1 | tail -15
    echo ""
    echo -e "${GREEN}🧪 Testando health check...${NC}"
    HEALTH=$(ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker exec $CONTAINER_NAME curl -s http://localhost:8080/health 2>&1 || echo 'ERROR'")
    if echo "$HEALTH" | grep -q "ok\|OK\|healthy"; then
        echo -e "${GREEN}✅ Health check OK!${NC}"
        echo "   $HEALTH"
    else
        echo -e "${YELLOW}⚠️  Health check ainda não respondeu${NC}"
        echo "   (Aplicação pode estar iniciando ainda)"
    fi
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ DEPLOY CONCLUÍDO COM SUCESSO!${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${CYAN}💡 Comandos úteis:${NC}"
    echo "   Ver logs: ssh $VPS_USER@$VPS_HOST 'docker logs -f $CONTAINER_NAME'"
    echo "   Status: ssh $VPS_USER@$VPS_HOST 'docker ps | grep $CONTAINER_NAME'"
    echo "   Testar: ssh $VPS_USER@$VPS_HOST 'docker exec $CONTAINER_NAME curl http://localhost:8080/health'"
    echo ""
else
    echo -e "${RED}❌ Container não está rodando${NC}"
    echo ""
    echo -e "${YELLOW}📋 Status:${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker ps -a | grep $CONTAINER_NAME"
    echo ""
    echo -e "${YELLOW}📋 Logs (últimas 20 linhas):${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker logs --tail 20 $CONTAINER_NAME" 2>&1
    echo ""
    echo -e "${RED}❌ Deploy falhou. Verifique os logs acima.${NC}"
    exit 1
fi
