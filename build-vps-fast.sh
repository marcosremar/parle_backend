#!/bin/bash
# Script para build rápido na VPS usando Dockerfile otimizado

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
IMAGE_NAME="${IMAGE_NAME:-parle-backend:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile.vps-fast}"
USE_CACHE="${USE_CACHE:-true}"
BUILD_MODE="${BUILD_MODE:-fast}"  # fast, ultra-fast, base-optimized

echo -e "${BLUE}⚡ Build Rápido na VPS${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Image: $IMAGE_NAME"
echo "   Dockerfile: $DOCKERFILE_PATH"
echo "   Cache: $USE_CACHE"
echo ""

# Selecionar Dockerfile baseado no modo
case "$BUILD_MODE" in
    ultra-fast)
        DOCKERFILE_PATH="docker/Dockerfile.vps-ultra-fast"
        echo -e "${CYAN}⚡ Modo: Ultra-Fast (uma camada)${NC}"
        ;;
    base-optimized)
        DOCKERFILE_PATH="docker/Dockerfile.vps-base-optimized"
        echo -e "${CYAN}⚡ Modo: Base Otimizada (imagem PyTorch) - Mais Rápido${NC}"
        ;;
    fast|*)
        DOCKERFILE_PATH="docker/Dockerfile.vps-fast"
        echo -e "${CYAN}⚡ Modo: Fast (camadas otimizadas)${NC}"
        ;;
esac

# Verificar se Dockerfile existe
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${YELLOW}⚠️  $DOCKERFILE_PATH não encontrado, usando Dockerfile padrão${NC}"
    DOCKERFILE_PATH="docker/Dockerfile"
fi

# Verificar conexão SSH
echo -e "${BLUE}🔌 Testando conexão SSH...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conexão SSH OK${NC}"
else
    echo -e "${RED}❌ Não foi possível conectar à VPS${NC}"
    exit 1
fi
echo ""

# Preparar ambiente na VPS
echo -e "${BLUE}📁 Preparando ambiente na VPS...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    mkdir -p /tmp/parle-backend-build
    rm -rf /tmp/parle-backend-build/*
" > /dev/null 2>&1
echo -e "${GREEN}✅ Ambiente preparado${NC}"
echo ""

# Sincronizar arquivos
echo -e "${BLUE}📤 Sincronizando arquivos...${NC}"
start_sync=$(date +%s.%N)

rsync -avz --progress \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no" \
    "$PROJECT_DIR/" "$VPS_USER@$VPS_HOST:/tmp/parle-backend-build/" > /tmp/rsync_build.log 2>&1

end_sync=$(date +%s.%N)
sync_duration=$(echo "$end_sync - $start_sync" | bc)
echo -e "${GREEN}✅ Sincronização concluída: ${sync_duration}s${NC}"
echo ""

# Build na VPS
echo "================================================================================"
echo -e "${BLUE}🏗️  INICIANDO BUILD OTIMIZADO${NC}"
echo "================================================================================"
echo ""

start_build=$(date +%s.%N)

# Construir comando de build
BUILD_CMD="cd /tmp/parle-backend-build && docker build"

# Adicionar cache se solicitado
if [ "$USE_CACHE" = "true" ]; then
    # Verificar se imagem existe
    if ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "docker images -q $IMAGE_NAME" | grep -q .; then
        BUILD_CMD="$BUILD_CMD --cache-from $IMAGE_NAME"
        echo -e "${CYAN}💡 Usando cache da imagem existente${NC}"
    else
        echo -e "${YELLOW}⚠️  Imagem não encontrada, build sem cache${NC}"
    fi
fi

BUILD_CMD="$BUILD_CMD -f $DOCKERFILE_PATH -t $IMAGE_NAME ."

echo -e "${BLUE}⏳ Executando build...${NC}"
case "$BUILD_MODE" in
    base-optimized)
        echo "   (Tempo estimado: 1-2 minutos - usando imagem base PyTorch)"
        ;;
    ultra-fast)
        echo "   (Tempo estimado: 1.5-3 minutos - uma camada)"
        ;;
    *)
        echo "   (Tempo estimado: 2-5 minutos com otimizações)"
        ;;
esac
echo ""

# Executar build e mostrar progresso
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "$BUILD_CMD" 2>&1 | \
    tee /tmp/docker_build.log | \
    grep -E "(Step|RUN|COPY|Successfully|ERROR)" || true

BUILD_EXIT_CODE=${PIPESTATUS[0]}

end_build=$(date +%s.%N)
build_duration=$(echo "$end_build - $start_build" | bc)
build_minutes=$(echo "scale=2; $build_duration / 60" | bc)

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ BUILD CONCLUÍDO${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${BLUE}📊 TEMPOS:${NC}"
    echo "   Sincronização: ${sync_duration}s"
    echo "   Build: ${build_duration}s (${build_minutes} min)"
    echo ""
    total_time=$(echo "$sync_duration + $build_duration" | bc)
    total_minutes=$(echo "scale=2; $total_time / 60" | bc)
    echo -e "${GREEN}⏱️  TEMPO TOTAL: ${total_time}s (${total_minutes} min)${NC}"
    echo ""
    
    # Verificar imagem
    echo -e "${BLUE}🖼️  Imagem criada:${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" \
        "docker images $IMAGE_NAME --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'"
    echo ""
    
    echo -e "${GREEN}💡 Próximos passos:${NC}"
    echo "   Criar container: docker create --name parle-backend $IMAGE_NAME"
    echo "   Iniciar container: docker start parle-backend"
    echo "   Ou usar: ./main.sh deploy:vps (se container já existe)"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}❌ BUILD FALHOU${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${YELLOW}💡 Verifique os logs:${NC}"
    echo "   tail -50 /tmp/docker_build.log"
    echo ""
    exit 1
fi
