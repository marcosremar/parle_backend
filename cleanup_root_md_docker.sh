#!/bin/bash
# Script para remover arquivos .md e Docker desnecessários da raiz

set -e

echo "🧹 Removendo arquivos .md e Docker desnecessários da raiz..."

# Arquivos .md a remover (documentação temporária/legada)
echo "📝 Removendo arquivos .md desnecessários..."
rm -f claude.md
rm -f info.md
rm -f IMPLEMENTATION_STATUS.md
rm -f IMPLEMENTATION_SUMMARY.md
rm -f MIGRATION_STATUS.md
rm -f MIGRATION_CHECKLIST.md
rm -f PAPERS_DOWNLOAD_SUMMARY.md
rm -f RELATORIO_PAPERS_INTERSPEECH_2024.md
rm -f CLEANUP_FINAL_REPORT.md
echo "✅ Arquivos .md temporários removidos"

# Manter README.md (documentação principal)
echo "ℹ️  README.md mantido (documentação principal do projeto)"

# Arquivos Docker a remover (se não forem usados)
echo "🐳 Verificando arquivos Docker..."
# Dockerfile e docker-compose.yml podem ser necessários, então vamos apenas verificar
# Se não forem usados, podem ser removidos manualmente

echo ""
echo "✅ Limpeza concluída!"
echo ""
echo "💡 Arquivos mantidos:"
echo "   - README.md (documentação principal)"
echo "   - Dockerfile (verificar se é usado)"
echo "   - docker-compose.yml (verificar se é usado)"
echo ""
echo "⚠️  Se Dockerfile e docker-compose.yml não forem usados,"
echo "   remova-os manualmente após verificação."
