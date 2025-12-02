#!/bin/bash
# Wrapper script para deploy otimizado
# Instala dependências e roda deploy ultra-rápido

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}🚀 Deploy Otimizado - Parle Backend${NC}"
echo "================================================================================"
echo ""

# 1. Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado${NC}"
    exit 1
fi

# 2. Verificar/install Fabric
echo -e "${BLUE}📦 Verificando Fabric...${NC}"
if ! python3 -c "import fabric" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Fabric não encontrado, instalando...${NC}"
    pip3 install fabric paramiko pyyaml
    echo -e "${GREEN}✅ Fabric instalado${NC}"
else
    echo -e "${GREEN}✅ Fabric já instalado${NC}"
fi

# 3. Verificar arquivo de deploy
if [ ! -f "$PROJECT_DIR/deploy_optimized.py" ]; then
    echo -e "${RED}❌ deploy_optimized.py não encontrado${NC}"
    exit 1
fi

# 4. Verificar config.yaml
if [ ! -f "$PROJECT_DIR/config.yaml" ]; then
    echo -e "${YELLOW}⚠️  config.yaml não encontrado, usando excludes padrão${NC}"
fi

# 5. Rodar deploy otimizado
echo ""
echo -e "${BLUE}⚡ Iniciando deploy ultra-rápido...${NC}"
echo ""

python3 deploy_optimized.py

DEPLOY_EXIT_CODE=$?

if [ $DEPLOY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ Deploy otimizado concluído com sucesso!${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${YELLOW}💡 Tempo total esperado: 30-60 segundos (vs 20+ minutos)${NC}"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}❌ Deploy otimizado falhou${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${YELLOW}💡 Verifique:${NC}"
    echo "   - Conexão SSH com a VPS"
    echo "   - Container está rodando"
    echo "   - Permissões SSH"
    echo ""
    exit 1
fi
