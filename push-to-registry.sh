#!/bin/bash
# Script para build e push da imagem para registry

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
REGISTRY="${REGISTRY:-docker.io}"  # docker.io, gcr.io, etc
REGISTRY_USER="${REGISTRY_USER:-}"
IMAGE_NAME="${IMAGE_NAME:-parle-backend}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile.vps-base-optimized}"
BUILD_MODE="${BUILD_MODE:-base-optimized}"

echo -e "${BLUE}📤 Build e Push para Registry${NC}"
echo "================================================================================"
echo ""

# Selecionar Dockerfile
case "$BUILD_MODE" in
    base-optimized)
        DOCKERFILE_PATH="docker/Dockerfile.vps-base-optimized"
        ;;
    ultra-fast)
        DOCKERFILE_PATH="docker/Dockerfile.vps-ultra-fast"
        ;;
    fast|*)
        DOCKERFILE_PATH="docker/Dockerfile.vps-fast"
        ;;
esac

# Construir nome completo da imagem
if [ "$REGISTRY" = "docker.io" ]; then
    if [ -z "$REGISTRY_USER" ]; then
        echo -e "${RED}❌ REGISTRY_USER não definido${NC}"
        echo ""
        echo -e "${YELLOW}💡 Configure:${NC}"
        echo "   export REGISTRY_USER=\"seu-usuario-dockerhub\""
        echo "   ou use: REGISTRY_USER=usuario ./push-to-registry.sh"
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
echo "   Registry: $REGISTRY"
echo "   Image: $FULL_IMAGE_NAME"
echo "   Dockerfile: $DOCKERFILE_PATH"
echo "   Mode: $BUILD_MODE"
echo ""

# Verificar se Dockerfile existe
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}❌ Dockerfile não encontrado: $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Build
echo "================================================================================"
echo -e "${BLUE}🏗️  Building imagem...${NC}"
echo "================================================================================"
echo ""

start_build=$(date +%s.%N)

docker build -f "$DOCKERFILE_PATH" -t "$FULL_IMAGE_NAME" .

build_exit=$?
end_build=$(date +%s.%N)
build_duration=$(echo "$end_build - $start_build" | bc)
build_minutes=$(echo "scale=2; $build_duration / 60" | bc)

if [ $build_exit -ne 0 ]; then
    echo -e "${RED}❌ Build falhou${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Build concluído: ${build_duration}s (${build_minutes} min)${NC}"
echo ""

# Login no registry (se necessário)
if [ "$REGISTRY" = "docker.io" ]; then
    echo -e "${BLUE}🔐 Verificando login Docker Hub...${NC}"
    if ! docker info | grep -q "Username"; then
        echo -e "${YELLOW}⚠️  Não autenticado no Docker Hub${NC}"
        echo "   Executando: docker login"
        docker login
    else
        echo -e "${GREEN}✅ Já autenticado${NC}"
    fi
    echo ""
elif [ "$REGISTRY" = "gcr.io" ]; then
    echo -e "${BLUE}🔐 Configurando GCR...${NC}"
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI não encontrado${NC}"
        exit 1
    fi
    gcloud auth configure-docker gcr.io --quiet
    echo -e "${GREEN}✅ GCR configurado${NC}"
    echo ""
fi

# Push
echo "================================================================================"
echo -e "${BLUE}📤 Pushing para registry...${NC}"
echo "================================================================================"
echo ""

start_push=$(date +%s.%N)

docker push "$FULL_IMAGE_NAME"

push_exit=$?
end_push=$(date +%s.%N)
push_duration=$(echo "$end_push - $start_push" | bc)
push_minutes=$(echo "scale=2; $push_duration / 60" | bc)

if [ $push_exit -ne 0 ]; then
    echo -e "${RED}❌ Push falhou${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Push concluído: ${push_duration}s (${push_minutes} min)${NC}"
echo ""

# Resumo
total_time=$(echo "$build_duration + $push_duration" | bc)
total_minutes=$(echo "scale=2; $total_time / 60" | bc)

echo "================================================================================"
echo -e "${GREEN}✅ CONCLUÍDO${NC}"
echo "================================================================================"
echo ""
echo -e "${BLUE}📊 TEMPOS:${NC}"
echo "   Build: ${build_duration}s (${build_minutes} min)"
echo "   Push: ${push_duration}s (${push_minutes} min)"
echo -e "${GREEN}   Total: ${total_time}s (${total_minutes} min)${NC}"
echo ""
echo -e "${BLUE}🖼️  Imagem disponível em:${NC}"
echo "   $FULL_IMAGE_NAME"
echo ""
echo -e "${GREEN}💡 Para usar na VPS:${NC}"
echo "   docker pull $FULL_IMAGE_NAME"
echo "   docker create --name parle-backend $FULL_IMAGE_NAME"
echo "   docker start parle-backend"
echo ""
echo -e "${CYAN}⏱️  Tempo na VPS será: 10-30s (apenas pull)${NC}"
echo ""
