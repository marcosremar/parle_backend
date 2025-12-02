#!/bin/bash
# Script rápido para medir tempo de inicialização do Cloud Run

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configurações
GCP_PROJECT_ID="avian-computer-477918-j9"
GCP_REGION="us-central1"
SERVICE_NAME="parle-backend-test"

# Adicionar gcloud ao PATH
if [ -f "/opt/homebrew/share/google-cloud-sdk/bin/gcloud" ]; then
    export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
fi
if [ -f "/opt/homebrew/bin/gcloud" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
fi

echo -e "${BLUE}⏱️  Medindo Tempo de Inicialização do Cloud Run${NC}"
echo "================================================================================"
echo ""

# Verificar se serviço existe
echo "🔍 Verificando serviço..."
URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT_ID" \
    --format="value(status.url)" 2>/dev/null || echo "")

if [ -z "$URL" ]; then
    echo -e "${YELLOW}⚠️  Serviço não encontrado. Fazendo deploy primeiro...${NC}"
    echo ""
    
    cd vendor/docker-manager
    
    # Fazer deploy simples com Dockerfile.cloudrun
    echo "📦 Fazendo build..."
    if [ -f "Dockerfile.cloudrun" ]; then
        if [ -f "cloudbuild.yaml" ]; then
            gcloud builds submit \
                --config cloudbuild.yaml \
                --project "$GCP_PROJECT_ID" \
                . 2>&1 | grep -E "(SUCCESS|FAILURE|ERROR)" || true
        else
            echo "❌ cloudbuild.yaml não encontrado"
            exit 1
        fi
    else
        echo "❌ Dockerfile.cloudrun não encontrado"
        exit 1
    fi
    
    echo "🚀 Fazendo deploy..."
    gcloud run deploy "$SERVICE_NAME" \
        --image "gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME" \
        --region "$GCP_REGION" \
        --platform managed \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --port 8080 \
        --project "$GCP_PROJECT_ID" \
        --quiet > /dev/null 2>&1 || {
        echo "❌ Erro no deploy"
        exit 1
    }
    
    # Obter URL
    URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$GCP_REGION" \
        --project "$GCP_PROJECT_ID" \
        --format="value(status.url)" 2>/dev/null)
    
    if [ -z "$URL" ]; then
        echo "❌ Não foi possível obter URL do serviço"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Deploy concluído${NC}"
    echo ""
fi

echo -e "${GREEN}✅ Serviço encontrado${NC}"
echo "   URL: $URL"
echo ""

# Configurar para cold start
echo "🛑 Configurando para cold start..."
gcloud run services update "$SERVICE_NAME" \
    --region "$GCP_REGION" \
    --min-instances 0 \
    --project "$GCP_PROJECT_ID" \
    --quiet > /dev/null 2>&1

echo "   Aguardando 15 segundos para garantir shutdown..."
sleep 15
echo ""

# Medir tempo
echo "🚀 Iniciando medição..."
echo "   Fazendo requisição HTTP para forçar cold start..."
echo ""

START_TIME=$(date +%s.%N)
START_DATETIME=$(date '+%H:%M:%S')

echo "⏱️  Iniciando requisição em: $START_DATETIME"
echo "   ⏳ Aguardando resposta (pode demorar até 5 minutos no cold start)..."
echo ""

# Fazer requisição
RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" \
    --max-time 300 \
    "$URL" 2>&1) || {
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo ""
    echo "❌ Erro na requisição ou timeout"
    echo "   Tempo decorrido: ${ELAPSED}s"
    exit 1
}

# Extrair informações
HTTP_CODE=$(echo "$RESPONSE" | tail -2 | head -1)
TIME_TOTAL=$(echo "$RESPONSE" | tail -1)

END_TIME=$(date +%s.%N)
END_DATETIME=$(date '+%H:%M:%S')
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
ELAPSED_MIN=$(echo "scale=2; $ELAPSED / 60" | bc)

echo ""
echo "⏱️  Resposta recebida em: $END_DATETIME"
echo ""
echo "================================================================================"
echo -e "${GREEN}📊 RESULTADOS${NC}"
echo "================================================================================"
echo ""
echo "   ⏱️  Tempo total: ${ELAPSED}s"
echo "   ⏱️  Tempo total: ${ELAPSED_MIN} minutos"
echo "   📡 Status HTTP: $HTTP_CODE"
echo "   ⚡ Tempo da requisição: ${TIME_TOTAL}s"
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Medição concluída!${NC}"
echo "================================================================================"
echo ""
echo -e "${BLUE}⏱️  TEMPO DE INICIALIZAÇÃO: ${ELAPSED}s (${ELAPSED_MIN} minutos)${NC}"
echo ""
