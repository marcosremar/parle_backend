#!/bin/bash
# Script para executar avaliação CEFR completa com conversas geradas e validação AKT

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=" | tr -d '\n' | head -c 80 && echo
echo "🧪 AVALIAÇÃO CEFR - CONVERSAS GERADAS COM VALIDAÇÃO AKT"
echo "=" | tr -d '\n' | head -c 80 && echo
echo

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verificar serviços
check_service() {
    local url=$1
    local name=$2
    
    if curl -s -f "$url/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $name está rodando"
        return 0
    else
        echo -e "${RED}❌${NC} $name não está rodando"
        return 1
    fi
}

echo "🔍 Verificando serviços necessários..."
echo

LLM_OK=false
STUDENT_MODEL_OK=false
LINGUISTIC_OK=false

if check_service "http://localhost:8006" "LLM Service"; then
    LLM_OK=true
fi

if check_service "http://localhost:8900" "Student Model Service"; then
    STUDENT_MODEL_OK=true
fi

if check_service "http://localhost:8901" "Linguistic Analysis Service"; then
    LINGUISTIC_OK=true
fi

echo

if [ "$LLM_OK" = false ] || [ "$STUDENT_MODEL_OK" = false ] || [ "$LINGUISTIC_OK" = false ]; then
    echo -e "${YELLOW}⚠️  Alguns serviços não estão rodando${NC}"
    echo
    echo "Para iniciar os serviços, execute em terminais separados:"
    echo
    [ "$LLM_OK" = false ] && echo -e "${BLUE}Terminal 1:${NC} python3 -m uvicorn src.services.llm.app_complete:app --host 0.0.0.0 --port 8006"
    [ "$STUDENT_MODEL_OK" = false ] && echo -e "${BLUE}Terminal 2:${NC} python3 -m uvicorn src.services.student_model.app_complete:app --host 0.0.0.0 --port 8900"
    [ "$LINGUISTIC_OK" = false ] && echo -e "${BLUE}Terminal 3:${NC} ./main.sh start:linguistic"
    echo
    read -p "Continuar mesmo assim? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Avaliação cancelada"
        exit 1
    fi
fi

echo
echo "🚀 Iniciando avaliação CEFR..."
echo

# Executar teste
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://localhost:8006}"
export STUDENT_MODEL_URL="${STUDENT_MODEL_URL:-http://localhost:8900}"

python3 -m pytest tests/e2e/test_cefr_conversation_history.py -v -s

echo
echo "=" | tr -d '\n' | head -c 80 && echo
echo "📊 Resultados salvos em:"
echo "   - tests/e2e/reports/conversation_history/conversation_history_summary.json"
echo "   - tests/e2e/reports/conversation_history/*.md (relatórios individuais)"
echo "=" | tr -d '\n' | head -c 80 && echo
