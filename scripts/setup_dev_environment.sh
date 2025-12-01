#!/bin/bash
# Script para configurar ambiente de desenvolvimento

set -e

echo "🔧 Configurando ambiente de desenvolvimento..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION detectado"

# Instalar dependências
echo "📦 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Instalar pre-commit hooks
echo "🔗 Instalando pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks instalados"
else
    echo "⚠️  pre-commit não encontrado. Instalando..."
    pip install pre-commit
    pre-commit install
fi

# Executar pre-commit em todos os arquivos (opcional)
read -p "Executar pre-commit em todos os arquivos agora? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pre-commit run --all-files
fi

# Auditar dependências
echo "🔍 Auditing dependências..."
if command -v pip-audit &> /dev/null; then
    pip-audit -r requirements.txt || echo "⚠️  Algumas vulnerabilidades encontradas. Revise o output acima."
else
    echo "⚠️  pip-audit não encontrado. Instalando..."
    pip install pip-audit
    pip-audit -r requirements.txt || echo "⚠️  Algumas vulnerabilidades encontradas. Revise o output acima."
fi

echo ""
echo "✅ Ambiente de desenvolvimento configurado!"
echo ""
echo "Próximos passos:"
echo "  1. Configure variáveis de ambiente: cp .env.example .env"
echo "  2. Execute testes: pytest"
echo "  3. Inicie o servidor: python src/api/main.py"
