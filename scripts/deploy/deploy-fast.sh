#!/bin/bash
# Script para deploy rápido usando máquina maior e Dockerfile otimizado

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:/opt/homebrew/bin:$PATH"

PROJECT_ID="avian-computer-477918-j9"
REGION="us-central1"
SERVICE_NAME="parle-backend"

echo -e "${BLUE}⚡ Deploy Rápido - Parle Backend${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}Usando:${NC}"
echo "  - Máquina: E2_HIGHCPU_8 (8 vCPUs)"
echo "  - Dockerfile: docker/Dockerfile.fast (cache otimizado)"
echo "  - Tempo estimado: 5-8 minutos (vs 15-20 padrão)"
echo ""

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI não encontrado!"
    echo "   Execute: ./setup_gcloud.sh"
    exit 1
fi

# Verificar se está autenticado
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Não autenticado no GCP!"
    echo "   Execute: gcloud auth login"
    exit 1
fi

# Verificar se cloudbuild-fast.yaml existe
if [ ! -f "docker/cloudbuild-fast.yaml" ]; then
    echo "❌ docker/cloudbuild-fast.yaml não encontrado!"
    exit 1
fi

# Verificar se Dockerfile.fast existe
if [ ! -f "docker/Dockerfile.fast" ]; then
    echo "❌ docker/Dockerfile.fast não encontrado!"
    exit 1
fi

echo -e "${GREEN}✅ Pré-requisitos OK${NC}"
echo ""

# Obter diretório do projeto
PROJECT_ROOT=$(pwd)
echo "📁 Diretório do projeto: $PROJECT_ROOT"
echo ""

# Verificar tamanho do contexto
echo "📊 Analisando contexto Docker..."
CONTEXT_SIZE=$(du -sh . | cut -f1)
echo "   Tamanho: $CONTEXT_SIZE"
echo ""

# Build e deploy
echo -e "${BLUE}🚀 Iniciando build rápido...${NC}"
echo ""

START_TIME=$(date +%s)

gcloud builds submit \
  --config docker/cloudbuild-fast.yaml \
  --project "$PROJECT_ID" \
  --substitutions "_SERVICE_NAME=$SERVICE_NAME,_REGION=$REGION,_DOCKERFILE_PATH=docker/Dockerfile.fast" \
  "$PROJECT_ROOT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Build concluído!${NC}"
echo ""
echo "⏱️  Tempo total: ${DURATION_MIN}m ${DURATION_SEC}s"
echo ""
echo "🌐 Serviço disponível em:"
echo "   https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME?project=$PROJECT_ID"
echo ""

# Obter URL do serviço
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$SERVICE_URL" ]; then
    echo "🔗 URL do serviço:"
    echo "   $SERVICE_URL"
    echo ""
    echo "🧪 Testar:"
    echo "   curl $SERVICE_URL/health"
    echo ""
fi
