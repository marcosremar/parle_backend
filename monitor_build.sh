#!/bin/bash
# Script para monitorar build do Cloud Run

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:/opt/homebrew/bin:$PATH"

PROJECT_ID="avian-computer-477918-j9"

echo -e "${BLUE}⏱️  Monitorando Build do Cloud Run${NC}"
echo "================================================================================"
echo ""

# Obter último build
BUILD_ID=$(gcloud builds list --project "$PROJECT_ID" --limit 1 --format="value(id)" 2>/dev/null)

if [ -z "$BUILD_ID" ]; then
    echo "❌ Nenhum build encontrado"
    exit 1
fi

echo "📋 Build ID: $BUILD_ID"
echo ""

# Obter status e tempo
BUILD_INFO=$(gcloud builds describe "$BUILD_ID" --project "$PROJECT_ID" --format="json" 2>/dev/null)

if [ -z "$BUILD_INFO" ]; then
    echo "❌ Não foi possível obter informações do build"
    exit 1
fi

# Extrair informações
STATUS=$(echo "$BUILD_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'UNKNOWN'))")
CREATE_TIME=$(echo "$BUILD_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('createTime', ''))")

echo -e "${BLUE}Status:${NC} $STATUS"

if [ -n "$CREATE_TIME" ]; then
    # Calcular tempo decorrido
    ELAPSED=$(python3 << EOF
from datetime import datetime, timezone
create_dt = datetime.fromisoformat("$CREATE_TIME".replace("Z", "+00:00"))
now_dt = datetime.now(timezone.utc)
elapsed = (now_dt - create_dt).total_seconds()
print(f"{elapsed / 60:.1f}")
EOF
)
    
    echo -e "${BLUE}⏱️  Tempo decorrido:${NC} ${ELAPSED} minutos"
    echo -e "${BLUE}🕐 Iniciado em:${NC} $CREATE_TIME"
    echo ""
    
    # Verificar se está demorando muito
    if (( $(echo "$ELAPSED > 10" | bc -l) )); then
        echo -e "${YELLOW}⚠️  Build está demorando mais que o esperado (>10 minutos)${NC}"
        echo ""
        echo "💡 Por que está demorando:"
        echo "   1. Instalação de 49 dependências Python"
        echo "   2. Compilação de pacotes nativos:"
        echo "      - numpy (C/Fortran)"
        echo "      - cryptography (Rust)"
        echo "      - scikit-learn (C++)"
        echo "      - outros pacotes com extensões C"
        echo "   3. Upload do contexto Docker (~650MB)"
        echo "   4. Build multi-stage (builder + production)"
        echo ""
        echo "⏳ Isso é normal para builds Python com muitas dependências!"
        echo "   Builds típicos: 10-20 minutos"
        echo ""
    fi
fi

# Mostrar progresso atual
echo -e "${BLUE}📋 Progresso atual:${NC}"
echo ""

gcloud builds log "$BUILD_ID" --project "$PROJECT_ID" 2>&1 | tail -10 | while IFS= read -r line; do
    if [[ "$line" == *"Step"* ]] || [[ "$line" == *"RUN"* ]] || [[ "$line" == *"COPY"* ]]; then
        echo "   $line"
    elif [[ "$line" == *"Installing"* ]] || [[ "$line" == *"Building"* ]]; then
        echo "   $line"
    fi
done

echo ""
echo "================================================================================"
echo -e "${GREEN}💡 Dica:${NC} O build continuará em background."
echo "   Você pode verificar o progresso em:"
echo "   https://console.cloud.google.com/cloud-build/builds/$BUILD_ID?project=$PROJECT_ID"
echo ""
