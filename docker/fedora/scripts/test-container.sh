#!/bin/bash
# Script de teste para container Fedora ARM64

set -e

echo "🧪 Testando Container Fedora ARM64"
echo "===================================="
echo ""

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}📊 Informações do Sistema:${NC}"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

echo -e "${YELLOW}💾 Memória:${NC}"
free -h
echo ""

echo -e "${YELLOW}💻 CPU:${NC}"
echo "Cores: $(nproc)"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

echo -e "${YELLOW}🐍 Python:${NC}"
python3 --version
echo ""

echo -e "${YELLOW}📦 Pacotes Instalados:${NC}"
dnf list installed | wc -l | xargs echo "Total:"
echo ""

echo -e "${GREEN}✅ Teste concluído!${NC}"
echo ""
echo "💡 Dicas:"
echo "  - Use 'htop' para monitorar recursos"
echo "  - Use 'free -h' para ver memória"
echo "  - Use 'df -h' para ver espaço em disco"
