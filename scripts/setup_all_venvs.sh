#!/bin/bash
# Script para criar venvs isolados para todos os serviços com Python 3.11

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🐍 Configurando ambientes virtuais Python 3.11 para todos os serviços"
echo "======================================================================"
echo ""

# Verificar qual Python usar (prioridade: python3.11 > python3)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    PYTHON_VERSION=$(python3.11 --version)
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version)
    echo "⚠️  Python 3.11 não encontrado, usando: $PYTHON_VERSION"
    echo "   (Recomendado: instalar Python 3.11 para compatibilidade)"
    echo ""
else
    echo "❌ Erro: Python não encontrado."
    exit 1
fi

echo "✅ Python encontrado: $PYTHON_VERSION"
echo ""

# Lista de serviços
SERVICES=(
    "api_gateway"
    "user"
    "conversation_history"
    "conversation_store"
    "database"
    "file_storage"
    "scenarios"
    "session"
    "orchestrator"
    "websocket"
    "llm"
    "stt"
    "tts"
    "webrtc_signaling"
    "neural_codec"
    "rest_polling"
)

SUCCESS=0
FAILED=0

for service in "${SERVICES[@]}"; do
    SERVICE_DIR="src/services/${service}"
    VENV_DIR="${SERVICE_DIR}/venv"
    
    if [ ! -d "$SERVICE_DIR" ]; then
        echo "⚠️  $service: Diretório não encontrado, pulando..."
        ((FAILED++))
        continue
    fi
    
    echo -n "📦 $service: "
    
    # Criar venv se não existir
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -n "venv criado"
        else
            echo "❌ Falha ao criar venv"
            ((FAILED++))
            continue
        fi
    else
        echo -n "venv já existe"
    fi
    
    # Instalar dependências se houver requirements.txt
    if [ -f "${SERVICE_DIR}/requirements.txt" ]; then
        echo -n ", instalando dependências..."
        source "${VENV_DIR}/bin/activate"
        pip install --quiet --upgrade pip > /dev/null 2>&1
        pip install --quiet -r "${SERVICE_DIR}/requirements.txt" > /dev/null 2>&1
        deactivate
        echo " ✅"
    else
        echo " ✅"
    fi
    
    ((SUCCESS++))
done

echo ""
echo "======================================================================"
echo "📊 Resumo:"
echo "   ✅ Sucesso: $SUCCESS serviços"
if [ $FAILED -gt 0 ]; then
    echo "   ❌ Falhas: $FAILED serviços"
fi
echo ""
echo "✅ Configuração concluída!"
echo ""
echo "💡 Cada serviço agora tem seu próprio venv isolado em:"
echo "   src/services/{service_name}/venv/"
echo ""

