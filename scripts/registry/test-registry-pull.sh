#!/bin/bash
# Script completo para testar pull do registry na VPS

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
GCP_PROJECT="${GCP_PROJECT:-avian-computer-477918-j9}"
IMAGE_NAME="gcr.io/$GCP_PROJECT/parle-backend:latest"
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend-registry}"

echo -e "${BLUE}⚡ Teste Completo: Pull do Registry${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}📋 Configuração:${NC}"
echo "   Image: $IMAGE_NAME"
echo "   VPS: $VPS_USER@$VPS_HOST"
echo "   Container: $CONTAINER_NAME"
echo ""

# Verificar conexão SSH
echo -e "${BLUE}🔌 Testando conexão SSH...${NC}"
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conexão SSH OK${NC}"
else
    echo -e "${RED}❌ Não foi possível conectar à VPS${NC}"
    exit 1
fi
echo ""

# Limpar container anterior
echo -e "${YELLOW}🧹 Limpando container anterior...${NC}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
" > /dev/null 2>&1
echo -e "${GREEN}✅ Limpeza concluída${NC}"
echo ""

# Pull
echo "================================================================================"
echo -e "${BLUE}📥 PULLING IMAGEM DO REGISTRY${NC}"
echo "================================================================================"
echo ""

start_pull=$(date +%s.%N)

# Tentar pull direto
PULL_OUTPUT=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker pull $IMAGE_NAME 2>&1" || echo "PULL_FAILED")

end_pull=$(date +%s.%N)
pull_duration=$(echo "$end_pull - $start_pull" | bc)
pull_minutes=$(echo "scale=2; $pull_duration / 60" | bc)

echo "$PULL_OUTPUT" | grep -E "(Pulling|Downloading|Extracting|Pull complete|unauthorized|denied|ERROR|PULL_FAILED)" || echo "$PULL_OUTPUT" | tail -10

if echo "$PULL_OUTPUT" | grep -q "Pull complete\|already exists"; then
    echo ""
    echo -e "${GREEN}✅ Pull concluído: ${pull_duration}s (${pull_minutes} min)${NC}"
    PULL_SUCCESS=true
elif echo "$PULL_OUTPUT" | grep -q "unauthorized\|denied"; then
    echo ""
    echo -e "${YELLOW}⚠️  Pull requer autenticação${NC}"
    echo ""
    echo -e "${CYAN}💡 Opções:${NC}"
    echo "   1. Usar Docker Hub (público, mais fácil)"
    echo "   2. Configurar autenticação GCR na VPS"
    echo "   3. Usar build direto na VPS"
    PULL_SUCCESS=false
else
    echo ""
    echo -e "${RED}❌ Pull falhou${NC}"
    PULL_SUCCESS=false
fi
echo ""

if [ "$PULL_SUCCESS" = true ]; then
    # Verificar imagem
    echo -e "${BLUE}🖼️  Verificando imagem...${NC}"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker images | grep parle-backend" || true
    echo ""
    
    # Criar container
    echo -e "${BLUE}📦 Criando container...${NC}"
    start_create=$(date +%s.%N)
    
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
        docker create --name $CONTAINER_NAME \
            -v /tmp/parle-backend-sync:/tmp/parle-backend-sync \
            -v pip-cache:/root/.cache/pip \
            $IMAGE_NAME
    " > /dev/null 2>&1
    
    create_exit=$?
    end_create=$(date +%s.%N)
    create_duration=$(echo "$end_create - $start_create" | bc)
    
    if [ $create_exit -eq 0 ]; then
        echo -e "${GREEN}✅ Container criado: ${create_duration}s${NC}"
    else
        echo -e "${RED}❌ Criação do container falhou${NC}"
        exit 1
    fi
    echo ""
    
    # Iniciar container
    echo -e "${BLUE}🚀 Iniciando container...${NC}"
    start_start=$(date +%s.%N)
    
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker start $CONTAINER_NAME" > /dev/null 2>&1
    
    start_exit=$?
    end_start=$(date +%s.%N)
    start_duration=$(echo "$end_start - $start_start" | bc)
    
    if [ $start_exit -eq 0 ]; then
        echo -e "${GREEN}✅ Container iniciado: ${start_duration}s${NC}"
    else
        echo -e "${RED}❌ Inicialização do container falhou${NC}"
        exit 1
    fi
    echo ""
    
    # Resumo
    total_time=$(echo "$pull_duration + $create_duration + $start_duration" | bc)
    total_minutes=$(echo "scale=2; $total_time / 60" | bc)
    
    echo "================================================================================"
    echo -e "${GREEN}✅ TESTE CONCLUÍDO${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${BLUE}📊 TEMPOS:${NC}"
    echo "   Pull: ${pull_duration}s (${pull_minutes} min)"
    echo "   Create: ${create_duration}s"
    echo "   Start: ${start_duration}s"
    echo -e "${GREEN}   Total: ${total_time}s (${total_minutes} min)${NC}"
    echo ""
    
    # Comparação
    echo -e "${YELLOW}📈 COMPARAÇÃO:${NC}"
    echo "   Build direto na VPS: ~1-2 min"
    echo "   Pull do registry: ${total_time}s"
    if (( $(echo "$total_time < 60" | bc -l) )); then
        speedup=$(echo "scale=1; 120 / $total_time" | bc)
        echo -e "${GREEN}   ⚡ ${speedup}x mais rápido!${NC}"
    fi
    echo ""
else
    echo "================================================================================"
    echo -e "${YELLOW}⚠️  Pull requer autenticação${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${CYAN}💡 Para usar pull do registry:${NC}"
    echo ""
    echo "   Opção 1: Docker Hub (mais fácil)"
    echo "   1. docker login"
    echo "   2. docker tag parle-backend:latest seu-usuario/parle-backend:latest"
    echo "   3. docker push seu-usuario/parle-backend:latest"
    echo "   4. Na VPS: docker pull seu-usuario/parle-backend:latest"
    echo ""
    echo "   Opção 2: GCR com autenticação"
    echo "   1. Copiar service account key para VPS"
    echo "   2. Na VPS: cat key.json | docker login -u _json_key --password-stdin https://gcr.io"
    echo "   3. docker pull $IMAGE_NAME"
    echo ""
    echo "   Opção 3: Continuar usando build direto"
    echo "   ./build-vps-fast.sh"
    echo ""
fi
