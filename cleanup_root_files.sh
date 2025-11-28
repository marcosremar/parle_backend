#!/bin/bash
# Script para remover arquivos desnecessários da raiz do projeto

set -e

echo "🧹 Removendo arquivos desnecessários da raiz do projeto..."

# 1. Remover arquivos de documentação de limpeza temporários
echo "📝 Removendo arquivos de documentação de limpeza..."
rm -f CLEANUP_REPORT.md
rm -f CLEANUP_SUMMARY.md
rm -f CLEANUP_PHASE2.md
rm -f CLEANUP_FINAL_REPORT.md
echo "✅ Arquivos de documentação de limpeza removidos"

# 2. Remover script de limpeza (já executado, pode ser mantido ou removido)
# Descomente a linha abaixo se quiser remover o script também:
# rm -f cleanup_unnecessary_files.sh
echo "ℹ️  Script cleanup_unnecessary_files.sh mantido (pode ser útil no futuro)"

# 3. Verificar e remover outros arquivos temporários
echo "🗑️  Verificando outros arquivos temporários..."
find . -maxdepth 1 -type f \( -name "*.bak" -o -name "*.tmp" -o -name "*.log" -o -name "*.old" \) -delete 2>/dev/null || true
echo "✅ Arquivos temporários removidos"

echo ""
echo "✅ Limpeza da raiz concluída!"
echo ""
echo "💡 Nota: Os arquivos removidos são temporários e de documentação."
echo "   O histórico completo está no git."
