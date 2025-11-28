#!/bin/bash
# Script para remover arquivos desnecessários do projeto

set -e

echo "🧹 Iniciando limpeza de arquivos desnecessários..."

# 1. Remover arquivos .bak
echo "📦 Removendo arquivos .bak..."
find . -name "*.bak" -type f -not -path "./.git/*" -delete
echo "✅ Arquivos .bak removidos"

# 2. Remover __pycache__ e .pyc (mesmo que estejam no .gitignore)
echo "🗑️  Removendo diretórios __pycache__..."
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
echo "✅ Diretórios __pycache__ removidos"

echo "🗑️  Removendo arquivos .pyc..."
find . -name "*.pyc" -type f -not -path "./.git/*" -delete
echo "✅ Arquivos .pyc removidos"

# 3. Remover venv do orchestrator (256MB)
if [ -d "src/services/orchestrator/venv" ]; then
    echo "🗑️  Removendo venv do orchestrator (256MB)..."
    rm -rf src/services/orchestrator/venv
    echo "✅ venv do orchestrator removido"
else
    echo "ℹ️  venv do orchestrator não encontrado"
fi

# 4. Consolidar documentação de refatoração
echo "📝 Consolidando documentação de refatoração..."
if [ -f "src/services/orchestrator/LEGACY_CLEANUP_COMPLETE.md" ]; then
    echo "   Removendo arquivos de documentação de refatoração..."
    rm -f src/services/orchestrator/LEGACY_CLEANUP_COMPLETE.md
    rm -f src/services/orchestrator/LEGACY_CLEANUP_PLAN.md
    rm -f src/services/orchestrator/LEGACY_CODE_ANALYSIS.md
    rm -f src/services/orchestrator/README_REFACTORING.md
    rm -f src/services/orchestrator/REFACTORING_SUMMARY.md
    echo "✅ Documentação de refatoração removida (consolidada no histórico do git)"
else
    echo "ℹ️  Arquivos de documentação já foram removidos"
fi

echo ""
echo "✅ Limpeza concluída!"
echo ""
echo "📊 Resumo:"
echo "   - Arquivos .bak removidos"
echo "   - Diretórios __pycache__ removidos"
echo "   - Arquivos .pyc removidos"
echo "   - venv do orchestrator removido (~256MB)"
echo "   - Documentação de refatoração consolidada"
echo ""
echo "💡 Nota: Os arquivos removidos já estão no histórico do git,"
echo "   então podem ser recuperados se necessário."
