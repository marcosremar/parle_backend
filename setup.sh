#!/bin/bash

set -e

echo "🚀 Configurando ambiente Nomad para User Service"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Por favor, instale o Python 3.12 primeiro."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI não encontrado. Instalando dependências..."
    pip3 install -q fastapi uvicorn loguru pydantic pyyaml
    echo "✅ Dependências instaladas"
else
    echo "✅ Dependências já instaladas"
fi

echo ""
echo "✅ Setup concluído!"
echo ""
echo "Próximos passos:"
echo "  1. Instalar Nomad (se não estiver instalado):"
echo "     - Baixar do https://developer.hashicorp.com/nomad/downloads"
echo "     - Ou usar o script de instalação completo"
echo "  2. Iniciar Nomad em modo dev: nomad agent -dev -bind=0.0.0.0"
echo "  3. Em outro terminal, fazer deploy: nomad job run user-service.nomad"
echo "  4. Verificar status: nomad job status user-service"
echo "  5. Ver logs: nomad alloc logs <allocation-id>"
