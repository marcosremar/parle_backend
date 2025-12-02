#!/bin/bash
# Script mínimo para testar deploy

set -e

echo "🚀 Deploy Mínimo - Parle Backend"
echo "================================="

# Configurações
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend-simple}"

echo "📋 Configuração:"
echo "   Host: $VPS_HOST"
echo "   Container: $CONTAINER_NAME"
echo ""

# Testar SSH
echo "🔌 Testando SSH..."
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 ubuntu@$VPS_HOST "echo 'SSH OK'" >/dev/null 2>&1; then
    echo "✅ SSH funcionando"
else
    echo "❌ SSH falhou"
    exit 1
fi

# Verificar container
echo ""
echo "🐳 Verificando container..."
if ssh -i "$SSH_KEY_PATH" ubuntu@$VPS_HOST "sudo docker ps --filter name=$CONTAINER_NAME --format '{{.Names}}'" | grep -q "$CONTAINER_NAME"; then
    echo "✅ Container rodando"
    
    # Testar execução de comando
    echo ""
    echo "⚡ Testando execução..."
    if ssh -i "$SSH_KEY_PATH" ubuntu@$VPS_HOST "sudo docker exec $CONTAINER_NAME echo 'Docker exec OK'" >/dev/null 2>&1; then
        echo "✅ Docker exec funcionando"
        echo ""
        echo "🎉 Deploy mínimo bem-sucedido!"
    else
        echo "❌ Docker exec falhou"
        exit 1
    fi
else
    echo "❌ Container não está rodando"
    exit 1
fi
