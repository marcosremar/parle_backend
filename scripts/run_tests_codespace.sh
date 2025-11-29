#!/bin/bash
# Script para executar testes no GitHub Codespace
# Uso: ./scripts/run_tests_codespace.sh

set -e

echo "🚀 Iniciando execução de testes no Codespace..."
echo "📦 Verificando dependências..."

# Verificar se estamos no Codespace
if [ -n "$CODESPACE_NAME" ] || [ -n "$GITHUB_CODESPACE_TOKEN" ]; then
    echo "✅ Executando no GitHub Codespace"
else
    echo "ℹ️  Executando localmente (não é Codespace)"
fi

# Ativar ambiente conda se existir
if [ -d "$CONDA_PREFIX" ] || command -v conda &> /dev/null; then
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate base 2>/dev/null || true
    fi
fi

# Verificar Python
if ! command -v python &> /dev/null; then
    echo "❌ Python não encontrado"
    exit 1
fi

echo "🐍 Python: $(python --version)"
echo "📍 Diretório: $(pwd)"
echo ""

# Instalar dependências de teste se necessário (silencioso)
echo "📦 Verificando dependências de teste..."
pip install -q pytest pytest-asyncio pytest-cov pytest-timeout 2>/dev/null || echo "⚠️  Algumas dependências podem estar faltando"

echo ""
echo "🧪 Executando bateria completa de testes..."
echo ""

# Executar testes com relatório resumido
python -m pytest \
    tests/unit/ \
    tests/integration/ \
    tests/performance/ \
    tests/robustness/ \
    tests/quality/ \
    tests/regression/ \
    -v \
    --tb=short \
    --maxfail=5 \
    -q \
    --durations=10

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Todos os testes passaram!"
else
    echo "⚠️  Alguns testes falharam (código: $EXIT_CODE)"
fi

echo ""
echo "📊 Resumo final:"
python -m pytest \
    tests/unit/ \
    tests/integration/ \
    tests/performance/ \
    tests/robustness/ \
    tests/quality/ \
    tests/regression/ \
    --tb=no \
    -q \
    2>&1 | grep -E "(passed|failed|skipped|warnings)" | tail -1

exit $EXIT_CODE
