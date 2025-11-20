#!/bin/bash

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOMAD_DIR="$PROJECT_DIR/deploy/nomad"

# Verificar se Nomad está instalado
check_nomad() {
    if ! command -v nomad &> /dev/null; then
        echo -e "${RED}❌ Nomad não encontrado${NC}"
        echo ""
        echo "Por favor, instale o Nomad:"
        echo "  macOS: brew install nomad"
        echo "  Ou baixe de: https://developer.hashicorp.com/nomad/downloads"
        exit 1
    fi
}

# Verificar se Nomad está rodando
check_nomad_running() {
    if ! nomad node status &> /dev/null; then
        echo -e "${YELLOW}⚠️  Nomad não está rodando${NC}"
        echo ""
        echo "Inicie o Nomad em outro terminal:"
        echo -e "  ${CYAN}nomad agent -dev -bind=0.0.0.0${NC}"
        echo ""
        read -p "Deseja tentar iniciar o Nomad agora? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            echo "Iniciando Nomad em modo desenvolvimento..."
            nomad agent -dev -bind=0.0.0.0 &
            sleep 3
            if nomad node status &> /dev/null; then
                echo -e "${GREEN}✅ Nomad iniciado${NC}"
            else
                echo -e "${RED}❌ Falha ao iniciar Nomad${NC}"
                exit 1
            fi
        else
            exit 1
        fi
    fi
}

# Listar serviços disponíveis
list_services() {
    echo -e "${BLUE}📋 Serviços disponíveis:${NC}"
    echo ""
    
    if [ ! -d "$NOMAD_DIR" ]; then
        echo -e "${RED}❌ Diretório deploy/nomad não encontrado${NC}"
        return 1
    fi
    
    local count=0
    for file in "$NOMAD_DIR"/*.nomad; do
        if [ -f "$file" ]; then
            local name=$(basename "$file" .nomad)
            echo -e "  ${CYAN}•${NC} $name"
            ((count++))
        fi
    done
    
    echo ""
    echo -e "Total: ${GREEN}$count serviços${NC}"
}

# Iniciar um serviço
start_service() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: $0 start <servico>"
        echo ""
        list_services
        exit 1
    fi
    
    local nomad_file="$NOMAD_DIR/$service.nomad"
    
    if [ ! -f "$nomad_file" ]; then
        echo -e "${RED}❌ Arquivo não encontrado: $nomad_file${NC}"
        echo ""
        list_services
        exit 1
    fi
    
    check_nomad
    check_nomad_running
    
    echo -e "${BLUE}🚀 Iniciando serviço: ${CYAN}$service${NC}"
    echo ""
    
    cd "$PROJECT_DIR"
    nomad job run "$nomad_file"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Serviço $service iniciado${NC}"
        echo ""
        echo "Comandos úteis:"
        echo "  • Ver status: ${CYAN}nomad job status $service${NC}"
        echo "  • Ver logs: ${CYAN}nomad alloc logs -f \$(nomad job status $service | grep running | head -1 | awk '{print \$1}')${NC}"
        echo "  • Parar: ${CYAN}nomad job stop $service${NC}"
    else
        echo -e "${RED}❌ Falha ao iniciar serviço${NC}"
        exit 1
    fi
}

# Parar um serviço
stop_service() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: $0 stop <servico>"
        exit 1
    fi
    
    check_nomad
    check_nomad_running
    
    echo -e "${BLUE}🛑 Parando serviço: ${CYAN}$service${NC}"
    nomad job stop "$service"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Serviço $service parado${NC}"
    else
        echo -e "${RED}❌ Falha ao parar serviço${NC}"
        exit 1
    fi
}

# Iniciar todos os serviços
start_all() {
    check_nomad
    check_nomad_running
    
    echo -e "${BLUE}🚀 Iniciando todos os serviços...${NC}"
    echo ""
    
    if [ ! -d "$NOMAD_DIR" ]; then
        echo -e "${RED}❌ Diretório deploy/nomad não encontrado${NC}"
        exit 1
    fi
    
    local count=0
    local failed=0
    
    cd "$PROJECT_DIR"
    
    for file in "$NOMAD_DIR"/*.nomad; do
        if [ -f "$file" ]; then
            local name=$(basename "$file" .nomad)
            echo -e "  ${CYAN}→${NC} Iniciando $name..."
            
            if nomad job run "$file" > /dev/null 2>&1; then
                echo -e "    ${GREEN}✅${NC}"
                ((count++))
            else
                echo -e "    ${RED}❌${NC}"
                ((failed++))
            fi
        fi
    done
    
    echo ""
    echo -e "${GREEN}✅ $count serviços iniciados${NC}"
    if [ $failed -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $failed serviços falharam${NC}"
    fi
    echo ""
    echo "Ver status: ${CYAN}nomad job status${NC}"
}

# Parar todos os serviços
stop_all() {
    check_nomad
    check_nomad_running
    
    echo -e "${BLUE}🛑 Parando todos os serviços...${NC}"
    echo ""
    
    local jobs=$(nomad job status -short 2>/dev/null | grep -v "^ID" | awk '{print $1}' | grep -v "^$")
    
    if [ -z "$jobs" ]; then
        echo -e "${YELLOW}⚠️  Nenhum job rodando${NC}"
        return 0
    fi
    
    local count=0
    for job in $jobs; do
        echo -e "  ${CYAN}→${NC} Parando $job..."
        nomad job stop "$job" > /dev/null 2>&1
        ((count++))
    done
    
    echo ""
    echo -e "${GREEN}✅ $count serviços parados${NC}"
}

# Ver status de todos os serviços
status() {
    check_nomad
    check_nomad_running
    
    echo -e "${BLUE}📊 Status dos serviços:${NC}"
    echo ""
    nomad job status
}

# Ver logs de um serviço
logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: $0 logs <servico>"
        exit 1
    fi
    
    check_nomad
    check_nomad_running
    
    local alloc=$(nomad job status "$service" 2>/dev/null | grep -E "running|pending" | head -1 | awk '{print $1}')
    
    if [ -z "$alloc" ]; then
        echo -e "${YELLOW}⚠️  Nenhuma alocação encontrada para $service${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}📋 Logs do serviço: ${CYAN}$service${NC} (allocation: $alloc)"
    echo ""
    nomad alloc logs -f "$alloc"
}

# Mostrar ajuda
show_help() {
    echo -e "${BLUE}📖 Uso: $0 <comando> [opções]${NC}"
    echo ""
    echo "Comandos disponíveis:"
    echo ""
    echo -e "  ${CYAN}list${NC}              Listar todos os serviços disponíveis"
    echo -e "  ${CYAN}start <servico>${NC}    Iniciar um serviço específico"
    echo -e "  ${CYAN}stop <servico>${NC}     Parar um serviço específico"
    echo -e "  ${CYAN}start-all${NC}         Iniciar todos os serviços"
    echo -e "  ${CYAN}stop-all${NC}          Parar todos os serviços"
    echo -e "  ${CYAN}status${NC}            Ver status de todos os serviços"
    echo -e "  ${CYAN}logs <servico>${NC}    Ver logs de um serviço (seguir)"
    echo -e "  ${CYAN}help${NC}              Mostrar esta ajuda"
    echo ""
    echo "Exemplos:"
    echo ""
    echo "  $0 list"
    echo "  $0 start api-gateway"
    echo "  $0 start-all"
    echo "  $0 status"
    echo "  $0 logs api-gateway"
    echo "  $0 stop api-gateway"
    echo "  $0 stop-all"
    echo ""
}

# Main
case "${1:-help}" in
    list)
        list_services
        ;;
    start)
        start_service "$2"
        ;;
    stop)
        stop_service "$2"
        ;;
    start-all)
        start_all
        ;;
    stop-all)
        stop_all
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Comando desconhecido: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

