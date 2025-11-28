#!/bin/bash
# Script para continuar a limpeza do projeto

set -e

echo "🧹 Continuando limpeza do projeto..."

# 1. Remover arquivos .pyc, .pyo, .pyd restantes
echo "🗑️  Removendo arquivos Python compilados..."
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos Python compilados removidos"

# 2. Remover diretórios __pycache__
echo "🗑️  Removendo diretórios __pycache__..."
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
echo "✅ Diretórios __pycache__ removidos"

# 3. Remover arquivos .bak, .backup, .old, .orig, .rej
echo "🗑️  Removendo arquivos de backup..."
find . -type f \( -name "*.bak" -o -name "*.backup" -o -name "*.old" -o -name "*.orig" -o -name "*.rej" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos de backup removidos"

# 4. Remover arquivos .log (exceto se importantes)
echo "🗑️  Removendo arquivos .log..."
find . -type f -name "*.log" -not -path "./.git/*" -not -path "./venv/*" -not -path "*/venv/*" -delete 2>/dev/null || true
echo "✅ Arquivos .log removidos"

# 5. Remover arquivos .tmp, .temp
echo "🗑️  Removendo arquivos temporários..."
find . -type f \( -name "*.tmp" -o -name "*.temp" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos temporários removidos"

# 6. Remover arquivos do sistema operacional
echo "🗑️  Removendo arquivos do sistema operacional..."
find . -type f \( -name ".DS_Store" -o -name "Thumbs.db" -o -name "desktop.ini" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos do sistema removidos"

# 7. Remover arquivos de editor
echo "🗑️  Removendo arquivos de editor..."
find . -type f \( -name "*.swp" -o -name "*.swo" -o -name "*~" -o -name ".#*" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos de editor removidos"

# 8. Remover diretórios .egg-info
echo "🗑️  Removendo diretórios .egg-info..."
find . -type d -name "*.egg-info" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
echo "✅ Diretórios .egg-info removidos"

# 9. Remover diretórios build e dist
echo "🗑️  Removendo diretórios build e dist..."
find . -type d \( -name "build" -o -name "dist" \) -not -path "./.git/*" -not -path "./venv/*" -not -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true
echo "✅ Diretórios build e dist removidos"

# 10. Remover arquivos .coverage (serão regenerados)
echo "🗑️  Removendo arquivos .coverage..."
find . -type f -name ".coverage" -o -name ".coverage.*" -not -path "./.git/*" -delete 2>/dev/null || true
echo "✅ Arquivos .coverage removidos"

echo ""
echo "✅ Limpeza continuada concluída!"
echo ""
echo "💡 Nota: Arquivos de banco de dados (.db, .sqlite) foram mantidos"
echo "   pois podem conter dados importantes."
