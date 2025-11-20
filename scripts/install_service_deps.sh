#!/bin/bash
# Install dependencies for services
# Supports both shared venv and service-specific requirements

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Instalador de Dependências por Serviço              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Venv não encontrado em $VENV_DIR${NC}"
    echo -e "${YELLOW}   Execute ./setup.sh primeiro${NC}"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Function to install requirements from a file
install_requirements() {
    local req_file=$1
    local service_name=$2
    
    if [ ! -f "$req_file" ]; then
        echo -e "${YELLOW}⚠️  $req_file não encontrado${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📦 Instalando dependências de $service_name...${NC}"
    echo -e "   Arquivo: $req_file"
    
    if pip install -r "$req_file" > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Dependências instaladas${NC}"
        return 0
    else
        echo -e "${RED}   ❌ Erro ao instalar dependências${NC}"
        return 1
    fi
}

# Install base requirements
echo -e "${BLUE}1️⃣  Instalando dependências principais...${NC}"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    install_requirements "$PROJECT_DIR/requirements.txt" "projeto principal"
else
    echo -e "${YELLOW}⚠️  requirements.txt não encontrado na raiz${NC}"
fi
echo ""

# Find and install service-specific requirements
echo -e "${BLUE}2️⃣  Instalando dependências por serviço...${NC}"
echo ""

SERVICES_DIR="$PROJECT_DIR/src/services"
INSTALLED=0
FAILED=0

# Find all requirements files in services
while IFS= read -r req_file; do
    # Get service name from path
    service_name=$(echo "$req_file" | sed -n 's|.*/services/\([^/]*\)/.*|\1|p')
    
    if [ -n "$service_name" ]; then
        install_requirements "$req_file" "$service_name"
        if [ $? -eq 0 ]; then
            INSTALLED=$((INSTALLED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        echo ""
    fi
done < <(find "$SERVICES_DIR" -name "requirements*.txt" -type f | sort)

# Find and install core module requirements
echo -e "${BLUE}3️⃣  Instalando dependências de módulos core...${NC}"
echo ""

CORE_DIR="$PROJECT_DIR/src/core"
while IFS= read -r req_file; do
    module_name=$(echo "$req_file" | sed -n 's|.*/core/\([^/]*\)/.*|\1|p')
    
    if [ -n "$module_name" ]; then
        install_requirements "$req_file" "core.$module_name"
        if [ $? -eq 0 ]; then
            INSTALLED=$((INSTALLED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        echo ""
    fi
done < <(find "$CORE_DIR" -name "requirements*.txt" -type f | sort)

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Resumo da Instalação                                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Instalados: $INSTALLED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ Falhados: $FAILED${NC}"
fi
echo ""

# Verify installation
echo -e "${BLUE}4️⃣  Verificando instalação...${NC}"
python -c "import fastapi, uvicorn, loguru; print('✅ Dependências principais OK')" 2>/dev/null && \
python -c "import transformers, torch; print('✅ Dependências ML OK')" 2>/dev/null || \
echo -e "${YELLOW}⚠️  Algumas dependências podem não estar instaladas${NC}"

echo ""
echo -e "${GREEN}✅ Instalação concluída!${NC}"
echo ""

