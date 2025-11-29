#!/bin/bash
# Script para executar testes de estresse com servidor rodando

echo "=========================================="
echo "STRESS TEST - Parle Backend"
echo "=========================================="
echo ""

# Verificar se servidor está rodando
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Servidor não está rodando em http://localhost:8000"
    echo ""
    echo "Iniciando servidor em background..."
    cd "$(dirname "$0")/.."
    python -m src.api.main > /tmp/parle_server.log 2>&1 &
    SERVER_PID=$!
    echo "Servidor iniciado (PID: $SERVER_PID)"
    echo "Aguardando servidor iniciar..."
    sleep 5
    
    # Verificar novamente
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "❌ Servidor não iniciou corretamente"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    echo "✅ Servidor está rodando"
    echo ""
    SERVER_STARTED=true
else
    echo "✅ Servidor já está rodando"
    echo ""
    SERVER_STARTED=false
fi

# Executar testes
echo "Executando testes de estresse..."
cd "$(dirname "$0")/.."
python scripts/comprehensive_stress_test.py

# Gerar relatório
echo ""
echo "Gerando relatório..."
python scripts/generate_stress_report.py

# Limpar se iniciou servidor
if [ "$SERVER_STARTED" = true ]; then
    echo ""
    echo "Parando servidor..."
    kill $SERVER_PID 2>/dev/null
    echo "✅ Servidor parado"
fi

echo ""
echo "✅ Testes concluídos!"
echo "📄 Relatório: tmp/RELATORIO_OTIMIZACAO.md"
