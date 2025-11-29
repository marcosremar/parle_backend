#!/bin/bash
# Script rápido para executar apenas testes críticos
# Uso: ./scripts/run_tests_quick.sh

set -e

echo "⚡ Executando testes rápidos (apenas unitários e integração básica)..."

python -m pytest \
    tests/unit/speech/ \
    tests/unit/orchestrator/ \
    tests/integration/test_pipeline_basic.py \
    -v \
    --tb=line \
    --maxfail=3 \
    -q

echo "✅ Testes rápidos concluídos!"
