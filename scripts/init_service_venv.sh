#!/bin/bash
# Script para inicializar venv específico de um serviço com Python 3.11

SERVICE_NAME=$1
SERVICE_DIR="src/services/${SERVICE_NAME}"
VENV_DIR="${SERVICE_DIR}/venv"

if [ -z "$SERVICE_NAME" ]; then
    echo "❌ Erro: Nome do serviço não fornecido"
    exit 1
fi

# Verificar se o diretório do serviço existe
if [ ! -d "$SERVICE_DIR" ]; then
    echo "❌ Erro: Diretório do serviço não encontrado: $SERVICE_DIR"
    exit 1
fi

# Verificar qual Python usar (prioridade: python3.11 > python3)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Erro: Python não encontrado."
    exit 1
fi

# Criar venv se não existir
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Criando venv para $SERVICE_NAME com $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Erro: Falha ao criar venv."
        exit 1
    fi
fi

# Ativar venv e instalar dependências se houver requirements.txt
if [ -f "${SERVICE_DIR}/requirements.txt" ]; then
    echo "📥 Instalando dependências para $SERVICE_NAME..."
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r "${SERVICE_DIR}/requirements.txt"
    deactivate
fi

echo "✅ Venv do serviço $SERVICE_NAME está pronto: $VENV_DIR"
exit 0

