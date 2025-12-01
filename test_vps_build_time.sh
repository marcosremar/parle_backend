#!/bin/bash
# Script para testar tempo de build completo na VPS

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Configurações
VPS_HOST="${VPS_HOST:-54.37.225.188}"
VPS_USER="${VPS_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend-build-test}"
IMAGE_NAME="${IMAGE_NAME:-parle-backend:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile}"

echo -e "${BLUE}⏱️  Teste de Tempo - Build Completo na VPS${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}📋 Configuração:${NC}"
echo "   Host: $VPS_HOST"
echo "   User: $VPS_USER"
echo "   Container: $CONTAINER_NAME"
echo "   Image: $IMAGE_NAME"
echo "   Dockerfile: $DOCKERFILE_PATH"
echo ""

# Verificar conexão SSH
echo -e "${BLUE}🔌 Testando conexão SSH...${NC}"
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conexão SSH OK${NC}"
else
    echo -e "${RED}❌ Não foi possível conectar à VPS${NC}"
    exit 1
fi
echo ""

# Função para medir tempo
measure_time() {
    local start_time=$(date +%s.%N)
    "$@"
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    echo "$duration"
}

# Função para executar comando remoto e medir tempo
remote_exec_time() {
    local description="$1"
    local command="$2"
    
    echo -e "${BLUE}⏳ $description...${NC}"
    start_time=$(date +%s.%N)
    
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "$command" > /tmp/vps_build_output.log 2>&1
    exit_code=$?
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $description concluído: ${duration}s${NC}"
    else
        echo -e "${RED}❌ $description falhou: ${duration}s${NC}"
        echo "   Log:"
        tail -20 /tmp/vps_build_output.log | sed 's/^/   /'
        return 1
    fi
    
    echo "$duration"
}

# Limpar container e imagem anteriores (se existirem)
echo -e "${YELLOW}🧹 Limpando containers/imagens anteriores...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME 2>/dev/null || true
" > /dev/null 2>&1
echo -e "${GREEN}✅ Limpeza concluída${NC}"
echo ""

# Criar diretório temporário na VPS
echo -e "${BLUE}📁 Preparando ambiente na VPS...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    mkdir -p /tmp/parle-backend-build
    rm -rf /tmp/parle-backend-build/*
" > /dev/null 2>&1
echo -e "${GREEN}✅ Ambiente preparado${NC}"
echo ""

# Sincronizar arquivos do projeto para VPS
echo -e "${BLUE}📤 Sincronizando arquivos do projeto...${NC}"
start_sync=$(date +%s.%N)

rsync -avz --progress \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no" \
    "$PROJECT_DIR/" "$VPS_USER@$VPS_HOST:/tmp/parle-backend-build/" > /tmp/rsync_output.log 2>&1

end_sync=$(date +%s.%N)
sync_duration=$(echo "$end_sync - $start_sync" | bc)
echo -e "${GREEN}✅ Sincronização concluída: ${sync_duration}s${NC}"
echo ""

# Medir tempo de build
echo "================================================================================"
echo -e "${BLUE}🏗️  INICIANDO BUILD DO DOCKER${NC}"
echo "================================================================================"
echo ""

build_duration=$(remote_exec_time "Construindo imagem Docker" "
    cd /tmp/parle-backend-build && \
    docker build -f $DOCKERFILE_PATH -t $IMAGE_NAME . 2>&1
")

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Build falhou${NC}"
    exit 1
fi

echo ""

# Criar container
create_duration=$(remote_exec_time "Criando container" "
    docker create --name $CONTAINER_NAME \
        -v /tmp/parle-backend-sync:/tmp/parle-backend-sync \
        -v pip-cache:/root/.cache/pip \
        $IMAGE_NAME 2>&1
")

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Criação do container falhou${NC}"
    exit 1
fi

echo ""

# Iniciar container
start_duration=$(remote_exec_time "Iniciando container" "
    docker start $CONTAINER_NAME 2>&1
")

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Inicialização do container falhou${NC}"
    exit 1
fi

echo ""

# Verificar status
echo -e "${BLUE}📊 Verificando status do container...${NC}"
status=$(ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
    docker ps -a --filter name=$CONTAINER_NAME --format '{{.Status}}'
")
echo -e "${GREEN}✅ Container: $status${NC}"
echo ""

# Calcular totais
total_duration=$(echo "$sync_duration + $build_duration + $create_duration + $start_duration" | bc)
total_minutes=$(echo "scale=2; $total_duration / 60" | bc)

echo "================================================================================"
echo -e "${GREEN}✅ BUILD COMPLETO CONCLUÍDO${NC}"
echo "================================================================================"
echo ""
echo -e "${BLUE}📊 RESUMO DE TEMPOS:${NC}"
echo ""
printf "   %-35s %10s\n" "Etapa" "Tempo"
echo "   " $(printf '=%.0s' {1..50})
printf "   %-35s %10.2fs\n" "Sincronização de arquivos" "$sync_duration"
printf "   %-35s %10.2fs\n" "Build da imagem Docker" "$build_duration"
printf "   %-35s %10.2fs\n" "Criação do container" "$create_duration"
printf "   %-35s %10.2fs\n" "Inicialização do container" "$start_duration"
echo "   " $(printf '=%.0s' {1..50})
printf "   %-35s %10.2fs\n" "TOTAL" "$total_duration"
printf "   %-35s %10.2f min\n" "" "$total_minutes"
echo ""

# Comparação com modo sincronização
echo -e "${YELLOW}📈 COMPARAÇÃO:${NC}"
echo ""
echo "   Modo Sincronização (já testado):"
echo "   - Tempo: ~6-10s"
echo "   - Requer: Container já existe"
echo ""
echo "   Modo Build (este teste):"
echo "   - Tempo: ${total_duration}s (${total_minutes} min)"
echo "   - Requer: Build completo da imagem"
echo ""
echo -e "${GREEN}💡 Diferença:${NC}"
diff_ratio=$(echo "scale=1; $total_duration / 6" | bc)
echo "   Build é ~${diff_ratio}x mais lento que sincronização"
echo "   (mas só precisa fazer uma vez ou quando Dockerfile muda)"
echo ""

# Limpar (opcional)
read -p "Deseja manter container/imagem na VPS? (S/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo -e "${YELLOW}🧹 Limpando...${NC}"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        rm -rf /tmp/parle-backend-build
    " > /dev/null 2>&1
    echo -e "${GREEN}✅ Limpeza concluída${NC}"
else
    echo -e "${CYAN}💡 Container e imagem mantidos na VPS${NC}"
    echo "   Container: $CONTAINER_NAME"
    echo "   Image: $IMAGE_NAME"
fi

echo ""
