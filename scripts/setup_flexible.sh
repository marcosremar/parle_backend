#!/bin/bash
# Setup flexível - funciona com qualquer Python 3.x disponível
# Não requer Python 3.12 específico

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Setup Flexível - Parle Backend                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Função para encontrar Python disponível
find_python() {
    # Ordem de preferência
    PYTHON_CANDIDATES=(
        "/Users/marcos/bin/python3"
        "python3.12"
        "python3.11"
        "python3.10"
        "python3.9"
        "python3"
        "python"
    )
    
    for py_cmd in "${PYTHON_CANDIDATES[@]}"; do
        if command -v "$py_cmd" &> /dev/null; then
            # Testar se funciona
            if "$py_cmd" --version &> /dev/null 2>&1; then
                PYTHON_CMD="$py_cmd"
                PYTHON_VERSION=$("$py_cmd" --version 2>&1 | awk '{print $2}')
                return 0
            fi
        fi
    done
    
    return 1
}

# Encontrar Python
echo -e "${BLUE}🔍 Procurando Python disponível...${NC}"
if find_python; then
    echo -e "${GREEN}✅ Python encontrado: $PYTHON_CMD${NC}"
    echo -e "   Versão: $PYTHON_VERSION"
    
    # Verificar se é Python 3
    MAJOR_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" != "3" ]; then
        echo -e "${RED}❌ Python 3 é necessário (encontrado: $PYTHON_VERSION)${NC}"
        exit 1
    fi
    
    if [ "$MINOR_VERSION" -lt 9 ]; then
        echo -e "${YELLOW}⚠️  Python 3.9+ recomendado (encontrado: $PYTHON_VERSION)${NC}"
        echo -e "${YELLOW}   Continuando, mas algumas dependências podem não funcionar${NC}"
    fi
    
    echo ""
else
    echo -e "${RED}❌ Python não encontrado${NC}"
    exit 1
fi

# Diretório do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${BLUE}📁 Diretório do projeto: $PROJECT_DIR${NC}"
echo ""

# Criar venv
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Venv já existe em $VENV_DIR${NC}"
    read -p "Deseja recriar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${YELLOW}🗑️  Removendo venv existente...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}✅ Usando venv existente${NC}"
        echo ""
        echo -e "${GREEN}✅ Setup concluído!${NC}"
        echo ""
        echo -e "${BLUE}Para ativar o venv:${NC}"
        echo -e "  source venv/bin/activate"
        exit 0
    fi
fi

echo -e "${BLUE}🔨 Criando ambiente virtual...${NC}"
"$PYTHON_CMD" -m venv "$VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Falha ao criar venv${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Venv criado${NC}"
echo ""

# Ativar venv
source "$VENV_DIR/bin/activate"

# Atualizar pip
echo -e "${BLUE}📦 Atualizando pip...${NC}"
pip install --upgrade pip setuptools wheel

# Instalar dependências
echo ""
echo -e "${BLUE}📦 Instalando dependências...${NC}"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo -e "${BLUE}   Instalando requirements.txt...${NC}"
    pip install -r "$PROJECT_DIR/requirements.txt"
fi

# Instalar dependências do LLM (se necessário)
if [ -f "$PROJECT_DIR/src/services/llm/requirements_light.txt" ]; then
    echo -e "${BLUE}   Instalando dependências do LLM...${NC}"
    pip install -r "$PROJECT_DIR/src/services/llm/requirements_light.txt"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ✅ Setup Concluído com Sucesso!                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 Resumo:${NC}"
echo -e "   Python usado: $PYTHON_CMD ($PYTHON_VERSION)"
echo -e "   Venv criado em: $VENV_DIR"
echo ""
echo -e "${BLUE}🚀 Próximos passos:${NC}"
echo -e "   1. Ativar o venv:"
echo -e "      ${GREEN}source venv/bin/activate${NC}"
echo ""
echo -e "   2. Testar instalação:"
echo -e "      ${GREEN}./main.sh test${NC}"
echo ""
echo -e "   3. Iniciar um serviço:"
echo -e "      ${GREEN}./main.sh start llm${NC}"
echo ""

