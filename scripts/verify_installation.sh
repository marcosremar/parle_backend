#!/bin/bash
# Script para verificar se a instalação está completa e funcionando

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔍 Verificação Completa de Instalação${NC}"
echo "========================================"
echo ""

# Configurações
CONTAINER_NAME="${CONTAINER_NAME:-parle-backend}"
WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"

echo -e "${BLUE}📋 Verificando container: ${CONTAINER_NAME}${NC}"
echo ""

# 1. Verificar se container existe e está rodando
echo -e "${YELLOW}1. Status do Container${NC}"
if docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
    echo -e "   ✅ Container rodando"
else
    echo -e "   ❌ Container não está rodando"
    echo -e "   💡 Execute: docker start ${CONTAINER_NAME}"
    exit 1
fi

# 2. Verificar arquivos essenciais
echo ""
echo -e "${YELLOW}2. Arquivos Essenciais${NC}"

checks=(
    "src/api/main.py:API principal"
    "src/core/config.py:Configuração"
    "requirements.txt:Dependências"
    "src/modules/:Módulos"
    "src/services/:Serviços"
)

for check in "${checks[@]}"; do
    file=$(echo $check | cut -d: -f1)
    desc=$(echo $check | cut -d: -f2)

    if docker exec "${CONTAINER_NAME}" test -e "${WORKSPACE_PATH}/${file}" 2>/dev/null; then
        echo -e "   ✅ ${desc} (${file})"
    else
        echo -e "   ❌ ${desc} (${file}) ausente"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    echo ""
    echo -e "${RED}❌ Arquivos essenciais estão faltando!${NC}"
    echo -e "${YELLOW}💡 Execute o deploy novamente para sincronizar arquivos${NC}"
    exit 1
fi

# 3. Verificar ambiente Python
echo ""
echo -e "${YELLOW}3. Ambiente Python${NC}"

python_version=$(docker exec "${CONTAINER_NAME}" python3 --version 2>&1 | head -1)
if echo "$python_version" | grep -q "Python 3"; then
    echo -e "   ✅ Python: ${python_version}"
else
    echo -e "   ❌ Python não encontrado"
    exit 1
fi

# Verificar pip
if docker exec "${CONTAINER_NAME}" pip --version >/dev/null 2>&1; then
    echo -e "   ✅ Pip instalado"
else
    echo -e "   ❌ Pip não encontrado"
fi

# 4. Verificar dependências
echo ""
echo -e "${YELLOW}4. Dependências Python${NC}"

# Verificar algumas dependências críticas
deps=(
    "fastapi:FastAPI"
    "uvicorn:Uvicorn"
    "pydantic:Pydantic"
    "loguru:Loguru"
)

for dep in "${deps[@]}"; do
    package=$(echo $dep | cut -d: -f1)
    desc=$(echo $dep | cut -d: -f2)

    if docker exec "${CONTAINER_NAME}" python3 -c "import ${package}" 2>/dev/null; then
        echo -e "   ✅ ${desc} (${package})"
    else
        echo -e "   ❌ ${desc} (${package}) não instalado"
        missing_deps=true
    fi
done

if [ "$missing_deps" = true ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Algumas dependências estão faltando${NC}"
    echo -e "${CYAN}💡 Instalando dependências...${NC}"

    docker exec "${CONTAINER_NAME}" pip install -r "${WORKSPACE_PATH}/requirements.txt" 2>&1 | head -20

    # Verificar novamente após instalação
    echo ""
    echo -e "${YELLOW}Reverficando dependências...${NC}"
    for dep in "${deps[@]}"; do
        package=$(echo $dep | cut -d: -f1)
        desc=$(echo $dep | cut -d: -f2)

        if docker exec "${CONTAINER_NAME}" python3 -c "import ${package}" 2>/dev/null; then
            echo -e "   ✅ ${desc} (${package})"
        else
            echo -e "   ❌ ${desc} (${package}) ainda faltando"
        fi
    done
fi

# 5. Testar API
echo ""
echo -e "${YELLOW}5. Teste da API${NC}"

# Verificar se API está rodando no container
api_process=$(docker exec "${CONTAINER_NAME}" ps aux | grep -E "(uvicorn|python.*main:app)" | grep -v grep || true)

if [ -n "$api_process" ]; then
    echo -e "   ✅ API rodando no container"
    api_running=true
else
    echo -e "   ⚠️  API não está rodando no container"
    echo -e "   💡 Iniciando API para teste..."

    # Tentar iniciar API
    docker exec -d "${CONTAINER_NAME}" bash -c "cd ${WORKSPACE_PATH} && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000" 2>/dev/null || true

    sleep 3

    # Verificar novamente
    api_process=$(docker exec "${CONTAINER_NAME}" ps aux | grep -E "(uvicorn|python.*main:app)" | grep -v grep || true)
    if [ -n "$api_process" ]; then
        echo -e "   ✅ API iniciada com sucesso"
        api_running=true
    else
        echo -e "   ❌ Falha ao iniciar API"
        api_running=false
    fi
fi

# Testar health check
if [ "$api_running" = true ]; then
    echo -e "   🔍 Testando health check..."

    health_response=$(docker exec "${CONTAINER_NAME}" curl -s -f http://localhost:8000/health 2>/dev/null || echo "failed")

    if [ "$health_response" != "failed" ]; then
        echo -e "   ✅ Health check OK"
        api_healthy=true
    else
        echo -e "   ❌ Health check falhou"
        api_healthy=false
    fi
fi

# 6. Resultado final
echo ""
echo "========================================"
if [ "$api_healthy" = true ]; then
    echo -e "${GREEN}🎉 Instalação verificada com sucesso!${NC}"
    echo ""
    echo -e "${CYAN}📊 Status:${NC}"
    echo "   • Container: ✅ Rodando"
    echo "   • Arquivos: ✅ Presentes"
    echo "   • Python: ✅ Funcionando"
    echo "   • Dependências: ✅ Instaladas"
    echo "   • API: ✅ Respondendo"
    echo ""
    echo -e "${GREEN}🚀 Sistema pronto para uso!${NC}"
    exit 0
else
    echo -e "${RED}❌ Problemas encontrados na instalação${NC}"
    echo ""
    echo -e "${YELLOW}🔧 Ações recomendadas:${NC}"
    echo "   1. Verificar logs do container:"
    echo "      docker logs ${CONTAINER_NAME}"
    echo "   2. Acessar shell do container:"
    echo "      docker exec -it ${CONTAINER_NAME} bash"
    echo "   3. Verificar arquivos em ${WORKSPACE_PATH}"
    echo "   4. Executar novamente: ./main.sh deploy:vps --force"
    exit 1
fi
