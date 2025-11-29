#!/bin/bash
# Script de otimização para MacBook M1 8GB
# Uso: ./scripts/optimize_m1.sh

set -e

echo "🔧 Otimizando MacBook M1 8GB para desenvolvimento..."
echo ""

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Limpar cache Python
echo -e "${GREEN}🧹 Limpando cache Python...${NC}"
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
echo "✅ Cache Python limpo"

# 2. Limpar cache pip
echo -e "${GREEN}🧹 Limpando cache pip...${NC}"
pip cache purge 2>/dev/null || echo "⚠️  pip não encontrado ou cache já limpo"
echo "✅ Cache pip limpo"

# 3. Limpar cache conda (se disponível)
if command -v conda &> /dev/null; then
    echo -e "${GREEN}🧹 Limpando cache conda...${NC}"
    conda clean --all -y 2>/dev/null || true
    echo "✅ Cache conda limpo"
fi

# 4. Verificar processos pesados
echo ""
echo -e "${YELLOW}📊 Top 5 processos usando mais memória:${NC}"
ps aux | sort -nrk 4 | head -6 | tail -5 || true

# 5. Verificar swap
echo ""
echo -e "${YELLOW}💾 Uso de swap:${NC}"
sysctl vm.swapusage 2>/dev/null || echo "⚠️  Não foi possível verificar swap"

# 6. Verificar uso de disco
echo ""
echo -e "${YELLOW}💿 Uso de disco:${NC}"
df -h / | tail -1

# 7. Verificar variáveis de ambiente importantes
echo ""
echo -e "${YELLOW}🔍 Variáveis de ambiente de otimização:${NC}"
if [ -n "$OMP_NUM_THREADS" ]; then
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo -e "  ${RED}⚠️  OMP_NUM_THREADS não definido${NC}"
fi

if [ -n "$TORCH_NUM_THREADS" ]; then
    echo "  TORCH_NUM_THREADS=$TORCH_NUM_THREADS"
else
    echo -e "  ${RED}⚠️  TORCH_NUM_THREADS não definido${NC}"
fi

# 8. Verificar arquivo .env
echo ""
if [ -f ".env" ]; then
    echo -e "${YELLOW}📝 Verificando configurações no .env:${NC}"
    if grep -q "SERVER_WORKERS" .env; then
        grep "SERVER_WORKERS" .env
    else
        echo -e "  ${RED}⚠️  SERVER_WORKERS não definido no .env${NC}"
    fi
    
    if grep -q "DB_POOL_SIZE" .env; then
        grep "DB_POOL_SIZE" .env
    else
        echo -e "  ${YELLOW}ℹ️  DB_POOL_SIZE não definido (usando padrão)${NC}"
    fi
else
    echo -e "${YELLOW}ℹ️  Arquivo .env não encontrado${NC}"
fi

# 9. Verificar processos Python
echo ""
echo -e "${YELLOW}🐍 Processos Python em execução:${NC}"
PYTHON_PROCS=$(ps aux | grep -i python | grep -v grep | wc -l | tr -d ' ')
if [ "$PYTHON_PROCS" -gt 0 ]; then
    echo "  $PYTHON_PROCS processo(s) Python encontrado(s):"
    ps aux | grep -i python | grep -v grep | head -5
else
    echo "  Nenhum processo Python em execução"
fi

echo ""
echo -e "${GREEN}✅ Otimização concluída!${NC}"
echo ""
echo "💡 Dicas adicionais:"
echo "  - Feche aplicações desnecessárias (navegador com muitas abas, etc.)"
echo "  - Configure variáveis de ambiente no .env ou ~/.zshrc"
echo "  - Use modelos ML menores (MiniLM ao invés de modelos grandes)"
echo "  - Monitore memória com: python scripts/monitor_memory.py"
