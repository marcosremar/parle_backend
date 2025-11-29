#!/bin/bash
# Script para executar testes via GitHub CLI no Codespace
# Uso: ./scripts/run_tests_gh_codespace.sh

set -e

echo "🔍 Verificando Codespaces disponíveis..."

# Obter nome do Codespace
CODESPACE_NAME=$(gh codespace list --json name,state --jq '.[0].name' --limit 1 2>/dev/null)

if [ -z "$CODESPACE_NAME" ]; then
    echo "❌ Nenhum Codespace encontrado"
    echo "💡 Crie um Codespace primeiro: gh codespace create"
    exit 1
fi

echo "📦 Codespace encontrado: $CODESPACE_NAME"

# Verificar status
STATUS=$(gh codespace list --json name,state --jq ".[0] | select(.name==\"$CODESPACE_NAME\") | .state" --limit 1)
echo "📊 Status atual: $STATUS"

if [ "$STATUS" != "Available" ]; then
    echo "⏳ Iniciando Codespace (isso pode levar 1-2 minutos)..."
    # Iniciar Codespace em background
    gh codespace code -c "$CODESPACE_NAME" > /dev/null 2>&1 &
    CODESPACE_PID=$!
    
    echo "⏳ Aguardando Codespace ficar disponível..."
    MAX_WAIT=120
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        sleep 5
        WAITED=$((WAITED + 5))
        STATUS=$(gh codespace list --json name,state --jq ".[0] | select(.name==\"$CODESPACE_NAME\") | .state" --limit 1 2>/dev/null || echo "Unknown")
        echo "   Aguardando... ($WAITED/$MAX_WAIT segundos) - Status: $STATUS"
        
        if [ "$STATUS" = "Available" ]; then
            echo "✅ Codespace está disponível!"
            break
        fi
    done
    
    if [ "$STATUS" != "Available" ]; then
        echo "❌ Codespace não ficou disponível a tempo (Status: $STATUS)"
        echo "💡 Tente iniciar manualmente: gh codespace code -c $CODESPACE_NAME"
        exit 1
    fi
fi

echo ""
echo "🧪 Executando testes no Codespace..."
echo ""

# Executar testes via SSH
gh codespace ssh -c "$CODESPACE_NAME" << 'TESTSCRIPT'
# Encontrar diretório do projeto
cd /workspaces/parle_backend 2>/dev/null || \
cd ~/parle_backend 2>/dev/null || \
cd /home/codespace/parle_backend 2>/dev/null || \
(cd ~ && find . -name 'parle_backend' -type d -maxdepth 3 2>/dev/null | head -1 | xargs -I {} sh -c 'cd {} && pwd') || \
pwd

echo "📍 Diretório: $(pwd)"
echo "🐍 Python: $(python --version 2>&1)"
echo ""

# Dar permissão aos scripts
chmod +x scripts/run_tests_codespace.sh 2>/dev/null || true
chmod +x scripts/run_tests_quick.sh 2>/dev/null || true

# Executar testes
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
TESTSCRIPT

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Testes concluídos com sucesso!"
else
    echo "⚠️  Alguns testes falharam (código: $EXIT_CODE)"
fi

exit $EXIT_CODE
