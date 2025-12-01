#!/bin/bash

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Banner
show_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                  ${CYAN}Parle Backend${BLUE}                          ║"
    echo "║              Sistema de Conversação Multimodal              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Mostrar ajuda
show_help() {
    show_banner
    echo -e "${CYAN}📖 Uso:${NC} ${MAGENTA}main.sh${NC} <comando> [opções]"
    echo ""
    echo -e "${BLUE}Comandos disponíveis:${NC}"
    echo ""
    echo -e "  ${CYAN}setup${NC}                    Configurar ambiente Miniconda (Python 3.11)"
    echo -e "  ${CYAN}test${NC}                     Testar instalação"
    echo -e "  ${CYAN}test-all${NC}                 Executar todos os testes"
    echo -e "  ${CYAN}test-services${NC}             Testar health checks de todos os serviços"
    echo -e "  ${CYAN}test:demo:simple${NC}          Teste de demonstração simples (speech-to-speech)"
    echo ""
    echo -e "  ${CYAN}start api${NC}               Iniciar API Principal (monolito modular)"
    echo -e "  ${CYAN}start websocket${NC}          Iniciar WebSocket Service (processo separado)"
    echo -e "  ${CYAN}start --all${NC}              Iniciar API + WebSocket"
    echo -e "  ${CYAN}start <servico>${NC}          Iniciar serviço individual (legacy)"
    echo -e "  ${CYAN}start:linguistic${NC}         Iniciar serviço de análise linguística (porta 8901)"
    echo -e "  ${CYAN}start:acoustic${NC}           Iniciar serviço de features acústicas (porta 8970)"
    echo -e "  ${CYAN}stop <servico>${NC}           Parar um serviço específico"
    echo -e "  ${CYAN}stop --all${NC}               Parar todos os serviços"
    echo -e "  ${CYAN}restart <servico>${NC}        Reiniciar um serviço"
    echo ""
    echo -e "  ${CYAN}list${NC}                     Listar todos os serviços disponíveis"
    echo -e "  ${CYAN}status${NC}                   Ver status de todos os serviços"
    echo -e "  ${CYAN}logs <servico>${NC}           Ver logs de um serviço (seguir)"
    echo ""
    echo -e "  ${CYAN}shell${NC}                    Abrir shell com ambiente conda ativado"
    echo -e "  ${CYAN}conda-activate${NC}           Ativar ambiente conda manualmente"
    echo -e "  ${CYAN}conda-deactivate${NC}         Desativar ambiente conda"
    echo ""
    echo -e "  ${CYAN}demo${NC}                     Abrir interface de demonstração web"
    echo -e "  ${CYAN}monitor${NC}                  Abrir dashboard de monitoramento"
    echo -e "  ${CYAN}benchmark${NC}                Executar testes de performance"
    echo -e "  ${CYAN}deploy${NC}                   Configurar deploy para produção"
    echo -e "  ${CYAN}deploy:gcp${NC}               Fazer deploy no Google Cloud (Cloud Run)"
    echo -e "  ${CYAN}deploy:gcp:fast${NC}           Deploy rápido no GCP (máquina maior, 5-8 min)"
    echo -e "  ${CYAN}clean${NC}                    Limpar arquivos temporários"
    echo ""
    echo -e "  ${CYAN}docker-up${NC}                Iniciar projeto com Docker (produção)"
    echo -e "  ${CYAN}docker-up:dev${NC}             Iniciar projeto com Docker (desenvolvimento)"
    echo -e "  ${CYAN}docker-up:logging${NC}         Iniciar projeto com Docker + logging"
    echo -e "  ${CYAN}docker-down${NC}              Parar containers Docker"
    echo -e "  ${CYAN}docker-logs${NC}               Ver logs dos containers Docker"
    echo -e "  ${CYAN}docker-status${NC}            Ver status dos containers Docker"
    echo -e "  ${CYAN}docker-build${NC}             Build das imagens Docker"
    echo ""
    echo -e "  ${CYAN}help${NC}                     Mostrar esta ajuda"
    echo ""
    echo -e "${BLUE}Exemplos:${NC}"
    echo ""
    echo -e "  ${CYAN}main.sh setup${NC}"
    echo -e "  ${CYAN}main.sh test${NC}"
    echo -e "  ${CYAN}main.sh start api-gateway${NC}"
    echo -e "  ${CYAN}main.sh start --all${NC}"
    echo -e "  ${CYAN}main.sh status${NC}"
    echo -e "  ${CYAN}main.sh logs api-gateway${NC}"
    echo -e "  ${CYAN}main.sh stop --all${NC}"
    echo -e "  ${CYAN}main.sh docker-up${NC}"
    echo -e "  ${CYAN}main.sh docker-up:dev${NC}"
    echo -e "  ${CYAN}main.sh deploy:gcp${NC}"
    echo ""
}

# Setup
cmd_setup() {
    show_banner
    echo -e "${BLUE}🔧 Configurando ambiente Conda...${NC}"
    echo ""

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${YELLOW}⚠️  Conda não encontrado${NC}"
        echo -e "${CYAN}   Instalando Miniconda...${NC}"
        
        if [ ! -f "$PROJECT_DIR/setup_miniconda.sh" ]; then
            echo -e "${RED}❌ setup_miniconda.sh não encontrado${NC}"
            exit 1
        fi

        chmod +x "$PROJECT_DIR/setup_miniconda.sh"
        "$PROJECT_DIR/setup_miniconda.sh"
    fi

    # Create conda environment from environment.yml
    if [ -f "$PROJECT_DIR/environment.yml" ]; then
        echo -e "${CYAN}📦 Criando ambiente Conda do environment.yml...${NC}"
        conda env create -f "$PROJECT_DIR/environment.yml" || {
            echo -e "${YELLOW}⚠️  Ambiente já existe, atualizando...${NC}"
            conda env update -f "$PROJECT_DIR/environment.yml" --prune
        }
        echo -e "${GREEN}✅ Ambiente Conda configurado!${NC}"
        echo -e "${CYAN}💡 Para ativar: conda activate parle_backend${NC}"
    else
        echo -e "${RED}❌ environment.yml não encontrado${NC}"
        exit 1
    fi
}

# Helper: Activate conda
_activate_conda() {
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        if [ -f "$HOME/miniconda3/bin/conda" ]; then
            export PATH="$HOME/miniconda3/bin:$PATH"
        else
            echo -e "${RED}❌ Conda não encontrado. Execute: main.sh setup${NC}"
            exit 1
        fi
    fi
    
    # Initialize conda
    eval "$(conda shell.bash hook 2>/dev/null)" || {
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
    }
    
    # Activate environment
    conda activate parle_backend 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Ambiente parle_backend não encontrado${NC}"
        echo -e "${CYAN}   Execute: main.sh setup${NC}"
        exit 1
    }
}

# Start API (Monolith)
cmd_start_api() {
    show_banner
    echo -e "${BLUE}🚀 Iniciando API Principal (Monolito Modular)...${NC}"
    echo ""
    
    _activate_conda
    
    export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"
    export MONOLITH_MODE="true"
    
    local port="${PORT:-8000}"
    local script_path="src/api/main.py"
    
    if [ ! -f "$PROJECT_DIR/$script_path" ]; then
        echo -e "${RED}❌ Arquivo não encontrado: $script_path${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Iniciando na porta $port..."
    echo -e "  ${CYAN}→${NC} Modo: MONOLITH (chamadas diretas Python)"
    echo ""
    
    python "$PROJECT_DIR/$script_path" > "/tmp/api.log" 2>&1 &
    local pid=$!
    
    echo -e "  ${GREEN}✅${NC} API iniciada (PID: $pid)"
    echo -e "  ${CYAN}📋${NC} Log: /tmp/api.log"
    echo ""
    echo -e "${GREEN}✅ API Principal iniciada com sucesso!${NC}"
    echo -e "${CYAN}💡 Acesse: http://localhost:$port${NC}"
}

# Start WebSocket (Separate process)
cmd_start_websocket() {
    show_banner
    echo -e "${BLUE}🚀 Iniciando WebSocket Service...${NC}"
    echo ""
    
    _activate_conda
    
    export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"
    
    local port=8022
    local script_path="src/services/websocket/app_complete.py"
    
    if [ ! -f "$PROJECT_DIR/$script_path" ]; then
        echo -e "${RED}❌ Arquivo não encontrado: $script_path${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Iniciando na porta $port..."
    echo -e "  ${CYAN}→${NC} Comunica com API via HTTP (localhost:8000)"
    echo ""
    
    python "$PROJECT_DIR/$script_path" > "/tmp/websocket.log" 2>&1 &
    local pid=$!
    
    echo -e "  ${GREEN}✅${NC} WebSocket iniciado (PID: $pid)"
    echo -e "  ${CYAN}📋${NC} Log: /tmp/websocket.log"
    echo ""
    echo -e "${GREEN}✅ WebSocket Service iniciado com sucesso!${NC}"
}

# Test
cmd_test() {
    show_banner
    echo -e "${BLUE}🧪 Testando instalação...${NC}"
    echo ""
    
    if [ ! -f "$PROJECT_DIR/scripts/test_installation.sh" ]; then
        echo -e "${RED}❌ scripts/test_installation.sh não encontrado${NC}"
        exit 1
    fi
    
    "$PROJECT_DIR/scripts/test_installation.sh"
}

# Start service
cmd_start() {
    local service="$1"

    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: main.sh start <servico>"
        echo "     main.sh start --all"
        echo ""
        echo "Serviços disponíveis:"
        cmd_list_services
        exit 1
    fi

    # Se for --all, iniciar todos os serviços Python
    if [ "$service" = "--all" ] || [ "$service" = "all" ]; then
        cmd_start_all
        return
    fi

    # Casos especiais: api e websocket
    if [ "$service" = "api" ]; then
        cmd_start_api
        return
    fi
    
    if [ "$service" = "websocket" ]; then
        cmd_start_websocket
        return
    fi

    # Iniciar serviço individual (legacy - para compatibilidade)
    show_banner
    echo -e "${BLUE}🚀 Iniciando serviço: ${CYAN}$service${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Modo legacy - considere usar 'start api' para monolito${NC}"
    echo ""

    # Ativar ambiente conda
    _activate_conda

    # Configurar PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"

    # Mapear nome do serviço para script
    local script_path=""
    local port=""
    
    case "$service" in
        stt)
            script_path="src/services/stt/app_complete.py"
            port=8099
            ;;
        tts)
            script_path="src/services/tts/app_complete.py"
            port=8103
            ;;
        llm)
            script_path="src/services/llm/app_complete.py"
            port=8110
            ;;
        orchestrator)
            script_path="src/services/orchestrator/app_complete.py"
            port=8500
            ;;
        scenarios)
            script_path="src/services/scenarios/app_complete.py"
            port=8700
            ;;
        session)
            script_path="src/services/session/app_complete.py"
            port=8200
            ;;
        user)
            script_path="src/services/user/app_complete.py"
            port=8201
            ;;
        conversation_store)
            script_path="src/services/conversation_store/app_complete.py"
            port=8800
            ;;
        rest_polling)
            script_path="src/services/rest_polling/app_complete.py"
            port=8701
            ;;
        webrtc)
            script_path="src/services/webrtc/app_complete.py"
            port=10100
            ;;
        webrtc_signaling)
            script_path="src/services/webrtc_signaling/app_complete.py"
            port=10101
            ;;
        api_gateway)
            script_path="src/services/api_gateway/app_complete.py"
            port=8000
            ;;
        file_storage)
            script_path="src/services/file_storage/app_complete.py"
            port=8300
            ;;
        database)
            script_path="src/services/database/app_complete.py"
            port=8400
            ;;
        conversation_history)
            script_path="src/services/conversation_history/app_complete.py"
            port=8501
            ;;
        diagnostic_module|speech_grader|diagnostic)
            script_path="src/services/diagnostic_module/app_complete.py"
            port=8960
            ;;
        *)
            echo -e "${RED}❌ Serviço desconhecido: $service${NC}"
            echo ""
            cmd_list_services
            exit 1
            ;;
    esac

    if [ ! -f "$PROJECT_DIR/$script_path" ]; then
        echo -e "${RED}❌ Arquivo não encontrado: $script_path${NC}"
        exit 1
    fi

    # Iniciar serviço em background
    echo -e "  ${CYAN}→${NC} Iniciando na porta $port..."
    python "$PROJECT_DIR/$script_path" > "/tmp/${service}.log" 2>&1 &
    local pid=$!
    
    echo -e "  ${GREEN}✅${NC} Serviço iniciado (PID: $pid)"
    echo -e "  ${CYAN}📋${NC} Log: /tmp/${service}.log"
    echo ""
    echo -e "${GREEN}✅ Serviço $service iniciado com sucesso!${NC}"
}

# Start all services (API + WebSocket)
cmd_start_all() {
    show_banner
    echo -e "${BLUE}🚀 Iniciando todos os serviços (API + WebSocket)...${NC}"
    echo ""

    # Start API
    cmd_start_api
    sleep 2
    
    # Start WebSocket
    cmd_start_websocket
    sleep 2
    
    echo ""
    echo -e "${BLUE}⏳ Aguardando serviços iniciarem (5 segundos)...${NC}"
    sleep 5
    
    echo ""
    echo -e "${BLUE}🧪 Testando health checks...${NC}"
    echo ""
    
    # Test health checks
    local PASSED=0
    local FAILED=0
    
    test_health() {
        local service_name=$1
        local port=$2
        
        echo -n "  Testando $service_name (port $port)... "
        
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ OK${NC}"
            ((PASSED++))
            return 0
        else
            echo -e "${RED}❌ FAILED${NC}"
            ((FAILED++))
            return 1
        fi
    }
    
    test_health "api" 8000
    test_health "websocket" 8022
    
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}RESULTADOS${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}✅ Passou: $PASSED${NC}"
    echo -e "${RED}❌ Falhou: $FAILED${NC}"
    echo -e "${CYAN}Total: $((PASSED + FAILED))${NC}"
    echo ""
    echo -e "${CYAN}💡 Para parar todos: main.sh stop --all${NC}"
    echo -e "${CYAN}💡 Para ver logs: tail -f /tmp/api.log ou /tmp/websocket.log${NC}"
    
    # Old implementation (commented out for reference)
    # export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"

    # Array to store PIDs
    declare -a PIDS=()

    # Function to start a service
    start_service() {
        local service_name=$1
        local port=$2
        local script_path=$3
        
        echo -e "  ${CYAN}→${NC} Iniciando ${CYAN}$service_name${NC} na porta $port..."
        
        python3 "$PROJECT_DIR/$script_path" > "/tmp/${service_name}.log" 2>&1 &
        local pid=$!
        PIDS+=($pid)
        
        echo -e "    ${GREEN}✅${NC} PID: $pid"
        sleep 1
    }

    # Start all services
    start_service "stt" 8099 "src/services/stt/app_complete.py"
    start_service "tts" 8103 "src/services/tts/app_complete.py"
    start_service "llm" 8110 "src/services/llm/app_complete.py"
    start_service "websocket" 8022 "src/services/websocket/app_complete.py"
    start_service "orchestrator" 8500 "src/services/orchestrator/app_complete.py"
    start_service "scenarios" 8700 "src/services/scenarios/app_complete.py"
    start_service "session" 8200 "src/services/session/app_complete.py"
    start_service "user" 8201 "src/services/user/app_complete.py"
    start_service "conversation_store" 8800 "src/services/conversation_store/app_complete.py"
    start_service "rest_polling" 8701 "src/services/rest_polling/app_complete.py"
    start_service "webrtc" 10100 "src/services/webrtc/app_complete.py"
    start_service "webrtc_signaling" 10101 "src/services/webrtc_signaling/app_complete.py"
    start_service "api_gateway" 8000 "src/services/api_gateway/app_complete.py"
    start_service "file_storage" 8300 "src/services/file_storage/app_complete.py"
    start_service "database" 8400 "src/services/database/app_complete.py"
    start_service "conversation_history" 8501 "src/services/conversation_history/app_complete.py"

    echo ""
    echo -e "${BLUE}⏳ Aguardando serviços iniciarem (10 segundos)...${NC}"
    sleep 10

    echo ""
    echo -e "${BLUE}🧪 Testando health checks...${NC}"
    echo ""

    # Test health checks
    local PASSED=0
    local FAILED=0

    test_health() {
        local service_name=$1
        local port=$2
        
        echo -n "  Testando $service_name (port $port)... "
        
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ OK${NC}"
            ((PASSED++))
        return 0
    else
            echo -e "${RED}❌ FAILED${NC}"
            ((FAILED++))
        return 1
    fi
}

    test_health "stt" 8099
    test_health "tts" 8103
    test_health "llm" 8110
    test_health "websocket" 8022
    test_health "orchestrator" 8500
    test_health "scenarios" 8700
    test_health "session" 8200
    test_health "user" 8201
    test_health "conversation_store" 8800
    test_health "rest_polling" 8701
    test_health "webrtc" 10100
    test_health "webrtc_signaling" 10101
    test_health "api_gateway" 8000
    test_health "file_storage" 8300
    test_health "database" 8400
    test_health "conversation_history" 8501

    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}RESULTADOS${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}✅ Passou: $PASSED${NC}"
    echo -e "${RED}❌ Falhou: $FAILED${NC}"
    echo -e "${CYAN}Total: $((PASSED + FAILED))${NC}"
    echo ""
    echo -e "${CYAN}💡 PIDs dos serviços: ${PIDS[*]}${NC}"
    echo -e "${CYAN}💡 Para parar todos: main.sh stop --all${NC}"
    echo -e "${CYAN}💡 Para ver logs: tail -f /tmp/<service_name>.log${NC}"
}

# List services
cmd_list_services() {
    echo -e "${CYAN}Serviços disponíveis:${NC}"
    echo "  • stt (8099)"
    echo "  • tts (8103)"
    echo "  • llm (8110)"
    echo "  • websocket (8022)"
    echo "  • orchestrator (8500)"
    echo "  • scenarios (8700)"
    echo "  • session (8200)"
    echo "  • user (8201)"
    echo "  • conversation_store (8800)"
    echo "  • rest_polling (8701)"
    echo "  • webrtc (10100)"
    echo "  • webrtc_signaling (10101)"
    echo "  • api_gateway (8000)"
    echo "  • file_storage (8300)"
    echo "  • database (8400)"
    echo "  • conversation_history (8501)"
}

# Stop service
cmd_stop() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: main.sh stop <servico>"
        echo "     main.sh stop --all"
        exit 1
    fi
    
    if [ "$service" = "--all" ] || [ "$service" = "all" ]; then
        show_banner
        echo -e "${BLUE}🛑 Parando todos os serviços...${NC}"
        echo ""
        
        # Encontrar e parar todos os processos Python dos serviços
        local services=("stt" "tts" "llm" "websocket" "orchestrator" "scenarios" "session" "user" "conversation_store" "rest_polling" "webrtc" "webrtc_signaling" "api_gateway" "file_storage" "database" "conversation_history")
        
        for svc in "${services[@]}"; do
            local pids=$(pgrep -f "app_complete.py.*${svc}" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                echo -e "  ${CYAN}→${NC} Parando $svc..."
                echo "$pids" | xargs kill 2>/dev/null || true
            fi
        done
        
        echo ""
        echo -e "${GREEN}✅ Todos os serviços parados${NC}"
    else
        show_banner
        echo -e "${BLUE}🛑 Parando serviço: ${CYAN}$service${NC}"
        echo ""
        
        local pids=$(pgrep -f "app_complete.py.*${service}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            echo -e "${GREEN}✅ Serviço $service parado${NC}"
        else
            echo -e "${YELLOW}⚠️  Serviço $service não encontrado rodando${NC}"
        fi
    fi
}

# Restart service
cmd_restart() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: main.sh restart <servico>"
        exit 1
    fi
    
    show_banner
    echo -e "${BLUE}🔄 Reiniciando serviço: ${CYAN}$service${NC}"
    echo ""
    
    # Parar primeiro
    ./main.sh stop "$service" 2>/dev/null || true
    sleep 2
    
    # Iniciar novamente
    ./main.sh start "$service"
}

# List services
cmd_list() {
    show_banner
    cmd_list_services
}

# Status
cmd_status() {
    show_banner
    echo -e "${BLUE}📊 Status dos serviços...${NC}"
    echo ""

    test_health() {
        local service_name=$1
        local port=$2
        
        echo -n "  $service_name (port $port)... "
        
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ ONLINE${NC}"
            return 0
        else
            echo -e "${RED}❌ OFFLINE${NC}"
            return 1
        fi
    }

    test_health "stt" 8099
    test_health "tts" 8103
    test_health "llm" 8110
    test_health "websocket" 8022
    test_health "orchestrator" 8500
    test_health "scenarios" 8700
    test_health "session" 8200
    test_health "user" 8201
    test_health "conversation_store" 8800
    test_health "rest_polling" 8701
    test_health "webrtc" 10100
    test_health "webrtc_signaling" 10101
    test_health "api_gateway" 8000
    test_health "file_storage" 8300
    test_health "database" 8400
    test_health "conversation_history" 8501
}

# Logs
cmd_logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Nome do serviço não fornecido${NC}"
        echo ""
        echo "Uso: main.sh logs <servico>"
        echo ""
        echo "Serviços disponíveis:"
        echo "  • api - API Principal (monolito modular)"
        echo "  • websocket - WebSocket Service"
        exit 1
    fi
    
    show_banner
    
    # Logs são gerenciados diretamente (sem Nomad)
    case "$service" in
        api)
            if [ -f "/tmp/parle_api.log" ]; then
                tail -f /tmp/parle_api.log
            else
                echo -e "${YELLOW}⚠️  Arquivo de log não encontrado: /tmp/parle_api.log${NC}"
                echo "Inicie o serviço primeiro: ./main.sh start api"
            fi
            ;;
        websocket)
            if [ -f "/tmp/parle_websocket.log" ]; then
                tail -f /tmp/parle_websocket.log
            else
                echo -e "${YELLOW}⚠️  Arquivo de log não encontrado: /tmp/parle_websocket.log${NC}"
                echo "Inicie o serviço primeiro: ./main.sh start websocket"
            fi
            ;;
        *)
            echo -e "${YELLOW}⚠️  Serviço '$service' não reconhecido${NC}"
            echo "Use: ./main.sh logs api ou ./main.sh logs websocket"
            ;;
    esac
}

# Shell
cmd_shell() {
    show_banner
    echo -e "${BLUE}🐚 Abrindo shell com ambiente conda ativado...${NC}"
    echo ""

    # Verificar se miniconda está instalado
    if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
        echo -e "${YELLOW}⚠️  Miniconda não encontrado${NC}"
        echo -e "${YELLOW}   Execute: main.sh setup${NC}"
        exit 1
    fi

    # Configurar PATH e ativar ambiente
    export PATH="$HOME/miniconda3/bin:$PATH"
    source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true

    # Verificar se ambiente existe
    if ! conda env list 2>/dev/null | grep -q parle_backend; then
        echo -e "${YELLOW}⚠️  Ambiente conda 'parle_backend' não encontrado${NC}"
        echo -e "${YELLOW}   Execute: main.sh setup${NC}"
        exit 1
    fi

    # Ativar ambiente
    conda activate parle_backend
    export PYTHONPATH="$PROJECT_DIR/src"

    echo -e "${GREEN}✅ Ambiente conda ativado${NC}"
    echo -e "${GREEN}✅ PYTHONPATH=$PYTHONPATH${NC}"
    echo ""
    echo -e "${CYAN}💡 Dica: Digite 'exit' para sair${NC}"
    echo ""

    # Iniciar shell interativo
    exec "$SHELL"
}

# Conda activate
cmd_conda_activate() {
    show_banner
    echo -e "${BLUE}🔄 Ativando ambiente conda...${NC}"
    echo ""

    # Verificar se miniconda está instalado
    if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
        echo -e "${RED}❌ Miniconda não encontrado${NC}"
        echo -e "${RED}   Execute: main.sh setup${NC}"
        exit 1
    fi

    # Configurar PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true

    # Verificar se ambiente existe
    if ! conda env list 2>/dev/null | grep -q parle_backend; then
        echo -e "${RED}❌ Ambiente conda 'parle_backend' não encontrado${NC}"
        echo -e "${RED}   Execute: main.sh setup${NC}"
        exit 1
    fi

    # Ativar ambiente
    conda activate parle_backend
    export PYTHONPATH="$PROJECT_DIR/src"

    echo -e "${GREEN}✅ Ambiente conda 'parle_backend' ativado${NC}"
    echo -e "${GREEN}✅ PYTHONPATH=$PYTHONPATH${NC}"
    echo ""
    echo -e "${CYAN}💡 Ambiente pronto para desenvolvimento!${NC}"
}

# Conda deactivate
cmd_conda_deactivate() {
    show_banner
    echo -e "${BLUE}🔄 Desativando ambiente conda...${NC}"
    echo ""

    conda deactivate 2>/dev/null || true
    echo -e "${GREEN}✅ Ambiente conda desativado${NC}"
}

# Clean
cmd_clean() {
    show_banner
    echo -e "${BLUE}🧹 Limpando arquivos temporários...${NC}"
    echo ""
    
    # Limpar __pycache__
    echo -e "  ${CYAN}→${NC} Removendo __pycache__..."
    find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
    echo -e "    ${GREEN}✅${NC}"
    
    # Limpar .pytest_cache
    echo -e "  ${CYAN}→${NC} Removendo .pytest_cache..."
    find "$PROJECT_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    echo -e "    ${GREEN}✅${NC}"
    
    # Limpar .mypy_cache
    echo -e "  ${CYAN}→${NC} Removendo .mypy_cache..."
    find "$PROJECT_DIR" -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    echo -e "    ${GREEN}✅${NC}"
    
    # Limpar arquivos .log
    echo -e "  ${CYAN}→${NC} Removendo arquivos .log..."
    find "$PROJECT_DIR" -type f -name "*.log" -delete 2>/dev/null || true
    echo -e "    ${GREEN}✅${NC}"
    
    echo ""
    echo -e "${GREEN}✅ Limpeza concluída${NC}"
}

# Test demo simple
cmd_test_demo_simple() {
    show_banner
    echo -e "${BLUE}🧪 Executando teste de demonstração simples...${NC}"
    echo ""
    
    # Verificar se os serviços estão rodando
    echo -e "  ${CYAN}→${NC} Verificando se serviços estão rodando..."
    if ! curl -s -f "http://localhost:8022/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  WebSocket service não está rodando${NC}"
        echo -e "${YELLOW}   Execute: main.sh start websocket${NC}"
        echo -e "${YELLOW}   Ou: main.sh start --all${NC}"
        exit 1
    fi
    
    if ! curl -s -f "http://localhost:8500/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Orchestrator service não está rodando${NC}"
        echo -e "${YELLOW}   Execute: main.sh start orchestrator${NC}"
        echo -e "${YELLOW}   Ou: main.sh start --all${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Serviços estão rodando${NC}"
    echo ""
    
    # Verificar se gTTS está instalado
    echo -e "  ${CYAN}→${NC} Verificando dependências..."
    if ! python3 -c "import gtts" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  gTTS não está instalado${NC}"
        echo -e "${CYAN}   Instalando gTTS...${NC}"
        pip install gtts 2>/dev/null || {
            echo -e "${RED}❌ Falha ao instalar gTTS${NC}"
            echo -e "${YELLOW}   Execute manualmente: pip install gtts${NC}"
            exit 1
        }
    fi
    
    if ! python3 -c "import socketio" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  python-socketio não está instalado${NC}"
        echo -e "${CYAN}   Instalando python-socketio...${NC}"
        pip install python-socketio 2>/dev/null || {
            echo -e "${RED}❌ Falha ao instalar python-socketio${NC}"
            echo -e "${YELLOW}   Execute manualmente: pip install python-socketio${NC}"
            exit 1
        }
    fi
    
    echo -e "${GREEN}✅ Dependências OK${NC}"
    echo ""
    
    # Executar teste
    echo -e "  ${CYAN}→${NC} Executando teste..."
    echo ""
    
    local test_script="$PROJECT_DIR/tests/e2e/test_demo_simple.py"
    
    if [ ! -f "$test_script" ]; then
        echo -e "${RED}❌ Script de teste não encontrado: $test_script${NC}"
        exit 1
    fi
    
    # Criar diretório de output se não existir
    mkdir -p "$PROJECT_DIR/tests/output"
    
    # Executar teste
    export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"
    python3 "$test_script"
    
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Teste concluído com sucesso!${NC}"
        echo ""
        echo -e "${CYAN}📁 Arquivos salvos em:${NC}"
        echo -e "   $PROJECT_DIR/tests/output/"
        echo ""
        ls -lh "$PROJECT_DIR/tests/output/" | tail -5 | sed 's/^/   /'
    else
        echo -e "${RED}❌ Teste falhou${NC}"
        exit $exit_code
    fi
}

# Test services health checks
cmd_test_services() {
    show_banner
    echo -e "${BLUE}🧪 Testando health checks de todos os serviços...${NC}"
    echo ""

    local PASSED=0
    local FAILED=0

    test_health() {
        local service_name=$1
        local port=$2
        
        echo -n "  $service_name (port $port)... "
        
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ OK${NC}"
            ((PASSED++))
            return 0
        else
            echo -e "${RED}❌ FAILED${NC}"
            ((FAILED++))
            return 1
        fi
    }

    test_health "stt" 8099
    test_health "tts" 8103
    test_health "llm" 8110
    test_health "websocket" 8022
    test_health "orchestrator" 8500
    test_health "scenarios" 8700
    test_health "session" 8200
    test_health "user" 8201
    test_health "conversation_store" 8800
    test_health "rest_polling" 8701
    test_health "webrtc" 10100
    test_health "webrtc_signaling" 10101
    test_health "api_gateway" 8000
    test_health "file_storage" 8300
    test_health "database" 8400
    test_health "conversation_history" 8501

    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}✅ Passou: $PASSED${NC}"
    echo -e "${RED}❌ Falhou: $FAILED${NC}"
    echo -e "${CYAN}Total: $((PASSED + FAILED))${NC}"
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ Todos os serviços estão funcionando!${NC}"
    else
        echo -e "${YELLOW}⚠️  Alguns serviços não estão respondendo${NC}"
        echo -e "${YELLOW}   Use: main.sh start --all${NC}"
    fi
}

# Abrir interface de demonstração
cmd_demo() {
    show_banner
    echo -e "${BLUE}🎤 Abrindo interface de demonstração...${NC}"
    echo ""

    local demo_file="$PROJECT_DIR/speech_to_speech.html"

    if [ ! -f "$demo_file" ]; then
        echo -e "${RED}❌ Arquivo speech_to_speech.html não encontrado${NC}"
        echo -e "${YELLOW}   Execute: python -m http.server 8000${NC}"
        echo -e "${YELLOW}   E acesse: http://localhost:8000/speech_to_speech.html${NC}"
        return 1
    fi

    # Verificar se temos um comando para abrir navegador
    if command -v open >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo no navegador padrão..."
        open "$demo_file"
    elif command -v xdg-open >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo no navegador padrão (Linux)..."
        xdg-open "$demo_file"
    elif command -v start >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo no navegador padrão (Windows)..."
        start "$demo_file"
    else
        echo -e "${YELLOW}⚠️  Não foi possível detectar comando para abrir navegador${NC}"
        echo -e "${YELLOW}   Abra manualmente: $demo_file${NC}"
        echo ""
        echo -e "${CYAN}💡 Alternativa: Use um servidor web local${NC}"
        echo -e "   python -m http.server 8000"
        echo -e "   Acesse: http://localhost:8000/speech_to_speech.html"
        return 1
    fi

    echo ""
    echo -e "${GREEN}✅ Interface de demonstração aberta!${NC}"
    echo ""
    echo -e "${CYAN}🌐 Funcionalidades disponíveis:${NC}"
    echo -e "   • Gravação de áudio via microfone"
    echo -e "   • Upload de arquivos de áudio"
    echo -e "   • Histórico de conversas"
    echo -e "   • Configurações avançadas"
    echo -e "   • Métricas de performance"
    echo ""
    echo -e "${YELLOW}💡 Dica: Certifique-se de que os serviços estão rodando${NC}"
    echo -e "   Use: main.sh start --all"
}

# Abrir dashboard de monitoramento
cmd_monitor() {
    show_banner
    echo -e "${BLUE}📊 Abrindo dashboard de monitoramento...${NC}"
    echo ""

    local monitor_file="$PROJECT_DIR/service_monitor.html"

    if [ ! -f "$monitor_file" ]; then
        echo -e "${RED}❌ Arquivo service_monitor.html não encontrado${NC}"
        return 1
    fi

    # Verificar se temos um comando para abrir navegador
    if command -v open >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo dashboard no navegador..."
        open "$monitor_file"
    elif command -v xdg-open >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo dashboard no navegador (Linux)..."
        xdg-open "$monitor_file"
    elif command -v start >/dev/null 2>&1; then
        echo -e "  ${CYAN}→${NC} Abrindo dashboard no navegador (Windows)..."
        start "$monitor_file"
    else
        echo -e "${YELLOW}⚠️  Não foi possível abrir automaticamente${NC}"
        echo -e "${YELLOW}   Abra manualmente: $monitor_file${NC}"
        echo ""
        echo -e "${CYAN}💡 Alternativa: Use um servidor web local${NC}"
        echo -e "   python -m http.server 8000"
        echo -e "   Acesse: http://localhost:8000/service_monitor.html"
        return 1
    fi

    echo ""
    echo -e "${GREEN}✅ Dashboard de monitoramento aberto!${NC}"
    echo ""
    echo -e "${CYAN}📊 Recursos disponíveis:${NC}"
    echo -e "   • Status em tempo real dos serviços"
    echo -e "   • Health checks automáticos"
    echo -e "   • Links para APIs e documentação"
    echo -e "   • Auto-refresh a cada 10 segundos"
}

# Executar benchmark de performance
cmd_benchmark() {
    show_banner
    echo -e "${BLUE}⚡ Executando benchmark de performance...${NC}"
    echo ""

    local benchmark_script="$PROJECT_DIR/benchmark_speech_services.py"

    if [ ! -f "$benchmark_script" ]; then
        echo -e "${RED}❌ Script de benchmark não encontrado${NC}"
        return 1
    fi

    # Verificar se os serviços estão rodando
    echo -e "  ${CYAN}→${NC} Verificando se serviços estão rodando..."
    if ! curl -s http://localhost:8080/api/health >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Serviços não parecem estar rodando${NC}"
        echo -e "${YELLOW}   Recomendação: main.sh start --all${NC}"
        echo ""
        read -p "Continuar mesmo assim? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}❌ Benchmark cancelado${NC}"
            return 1
        fi
    fi

    # Ativar ambiente conda
    echo -e "  ${CYAN}→${NC} Ativando ambiente conda..."
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate parle_backend 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Ambiente conda não encontrado, executando sem isolamento${NC}"
        }
    fi

    # Executar benchmark
    echo -e "  ${CYAN}→${NC} Executando testes de performance..."
    echo ""
    PYTHONPATH="$PROJECT_DIR/src" python "$benchmark_script"

    echo ""
    echo -e "${GREEN}✅ Benchmark concluído!${NC}"
    echo ""
    echo -e "${CYAN}📊 Resultados salvos em:${NC}"
    echo -e "   benchmark_report_*.json"
    echo ""
    echo -e "${YELLOW}💡 Dica: Analise os resultados para otimizar configurações${NC}"
}

# Docker commands
cmd_docker_up() {
    show_banner
    echo -e "${BLUE}🐳 Iniciando projeto com Docker (produção)...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Iniciando containers..."
    docker-compose -f "$compose_file" up -d
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Containers iniciados com sucesso!${NC}"
        echo ""
        echo -e "${CYAN}💡 Acesse: http://localhost:8000${NC}"
        echo -e "${CYAN}💡 Ver logs: main.sh docker-logs${NC}"
        echo -e "${CYAN}💡 Status: main.sh docker-status${NC}"
        echo -e "${CYAN}💡 Parar: main.sh docker-down${NC}"
    else
        echo -e "${RED}❌ Erro ao iniciar containers${NC}"
        exit 1
    fi
}

cmd_docker_up_dev() {
    show_banner
    echo -e "${BLUE}🐳 Iniciando projeto com Docker (desenvolvimento)...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Iniciando containers em modo desenvolvimento (hot-reload)..."
    docker-compose -f "$compose_file" --profile dev up -d api-dev
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Containers iniciados com sucesso!${NC}"
        echo ""
        echo -e "${CYAN}💡 Acesse: http://localhost:8000${NC}"
        echo -e "${CYAN}💡 Modo: Desenvolvimento (hot-reload ativado)${NC}"
        echo -e "${CYAN}💡 Ver logs: main.sh docker-logs${NC}"
        echo -e "${CYAN}💡 Status: main.sh docker-status${NC}"
        echo -e "${CYAN}💡 Parar: main.sh docker-down${NC}"
    else
        echo -e "${RED}❌ Erro ao iniciar containers${NC}"
        exit 1
    fi
}

cmd_docker_up_logging() {
    show_banner
    echo -e "${BLUE}🐳 Iniciando projeto com Docker + logging (Loki + Grafana)...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    local logging_file="$PROJECT_DIR/docker/docker-compose.logging.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    if [ ! -f "$logging_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.logging.yml não encontrado: $logging_file${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Iniciando containers com stack de logging..."
    docker-compose \
        -f "$compose_file" \
        -f "$logging_file" \
        up -d
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Containers iniciados com sucesso!${NC}"
        echo ""
        echo -e "${CYAN}💡 API: http://localhost:8000${NC}"
        echo -e "${CYAN}💡 Grafana: http://localhost:3000${NC}"
        echo -e "${CYAN}💡 Ver logs: main.sh docker-logs${NC}"
        echo -e "${CYAN}💡 Status: main.sh docker-status${NC}"
        echo -e "${CYAN}💡 Parar: main.sh docker-down${NC}"
    else
        echo -e "${RED}❌ Erro ao iniciar containers${NC}"
        exit 1
    fi
}

cmd_docker_down() {
    show_banner
    echo -e "${BLUE}🛑 Parando containers Docker...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    docker-compose -f "$compose_file" down
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Containers parados com sucesso!${NC}"
    else
        echo -e "${RED}❌ Erro ao parar containers${NC}"
        exit 1
    fi
}

cmd_docker_logs() {
    local service="${1:-api}"
    
    show_banner
    echo -e "${BLUE}📋 Logs do container: ${CYAN}$service${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    docker-compose -f "$compose_file" logs -f "$service"
}

cmd_docker_status() {
    show_banner
    echo -e "${BLUE}📊 Status dos containers Docker...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    docker-compose -f "$compose_file" ps
}

cmd_docker_build() {
    show_banner
    echo -e "${BLUE}🔨 Build das imagens Docker...${NC}"
    echo ""
    
    local compose_file="$PROJECT_DIR/docker/docker-compose.yml"
    
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Arquivo docker-compose.yml não encontrado: $compose_file${NC}"
        exit 1
    fi
    
    echo -e "  ${CYAN}→${NC} Fazendo build das imagens..."
    docker-compose -f "$compose_file" build
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Build concluído com sucesso!${NC}"
        echo -e "${CYAN}💡 Para iniciar: main.sh docker-up${NC}"
    else
        echo -e "${RED}❌ Erro ao fazer build${NC}"
        exit 1
    fi
}

# Configurar deploy para produção

# Iniciar serviço de análise linguística
cmd_start_linguistic() {
    show_banner
    echo -e "${BLUE}🚀 Iniciando Linguistic Analysis Service...${NC}"
    echo -e "${BLUE}   Service will run on port 8901${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Make sure SpaCy Portuguese model is installed:${NC}"
    echo -e "   ${CYAN}python -m spacy download pt_core_news_lg${NC}"
    echo ""
    
    python3 -m uvicorn src.services.linguistic_analysis.app_complete:app --host 0.0.0.0 --port 8901 --reload
}

cmd_start_acoustic() {
    show_banner
    echo -e "${BLUE}🚀 Iniciando Acoustic Features Service...${NC}"
    echo -e "${BLUE}   Service will run on port 8970${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Make sure PyTorch and transformers are installed:${NC}"
    echo -e "   ${CYAN}pip install torch torchaudio transformers soundfile${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Device: Will auto-detect (MPS for M1, CUDA for GPU, CPU otherwise)${NC}"
    echo ""
    
    export PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR"
    
    python3 -m uvicorn src.services.acoustic_features.app_complete:app --host 0.0.0.0 --port 8970 --reload
}
cmd_deploy() {
    show_banner
    echo -e "${BLUE}🏭 Configurando deploy para produção...${NC}"
    echo ""

    local deploy_script="$PROJECT_DIR/deploy_production.sh"

    if [ ! -f "$deploy_script" ]; then
        echo -e "${RED}❌ Script de deploy não encontrado${NC}"
        return 1
    fi

    echo -e "${YELLOW}⚠️  Este comando irá configurar o ambiente para produção${NC}"
    echo -e "${YELLOW}   Isso inclui:${NC}"
    echo -e "   • Criar ambiente conda isolado"
    echo -e "   • Gerar configurações de produção"
    echo -e "   • Criar arquivos de serviço systemd (Linux)"
    echo -e "   • Configurar secrets seguros"
    echo ""
    read -p "Continuar com deploy para produção? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}❌ Deploy cancelado${NC}"
        return 1
    fi

    # Executar deploy
    bash "$deploy_script"

    echo ""
    echo -e "${GREEN}✅ Deploy para produção configurado!${NC}"
    echo ""
    echo -e "${CYAN}🚀 Para iniciar em produção:${NC}"
    echo -e "   ./start_production.sh"
    echo ""
    echo -e "${CYAN}📁 Arquivos criados:${NC}"
    echo -e "   • .env.production - Configurações de produção"
    echo -e "   • start_production.sh - Script de inicialização"
    echo -e "   • stop_production.sh - Script de parada"
    echo -e "   • environment.yml - Ambiente conda"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "   • /tmp/parle-*.service - Serviços systemd"
    fi
}

# Deploy no Google Cloud usando docker-manager
cmd_deploy_gcp() {
    show_banner
    echo -e "${BLUE}☁️  Deploy no Google Cloud Platform${NC}"
    echo "================================================================================"
    echo ""
    
    # Verificar se docker-manager existe
    local docker_manager_dir="$PROJECT_DIR/vendor/docker-manager"
    if [ ! -d "$docker_manager_dir" ]; then
        echo -e "${RED}❌ docker-manager não encontrado em: $docker_manager_dir${NC}"
        echo ""
        echo -e "${YELLOW}💡 O docker-manager deve estar em vendor/docker-manager${NC}"
        return 1
    fi
    
    # Verificar se gcloud está instalado
    if ! command -v gcloud &> /dev/null; then
        # Tentar adicionar ao PATH
        if [ -f "/opt/homebrew/share/google-cloud-sdk/bin/gcloud" ]; then
            export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
        elif [ -f "/opt/homebrew/bin/gcloud" ]; then
            export PATH="/opt/homebrew/bin:$PATH"
        fi
    fi
    
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI não encontrado${NC}"
        echo ""
        echo -e "${YELLOW}💡 Instale o Google Cloud SDK:${NC}"
        echo "   ./setup_gcp_test.sh  (instala automaticamente)"
        echo "   ou: brew install --cask google-cloud-sdk"
        return 1
    fi
    
    # Verificar credenciais
    local credentials_path="${HOME}/Downloads/avian-computer-477918-j9-54b778b99398.json"
    if [ ! -f "$credentials_path" ]; then
        echo -e "${YELLOW}⚠️  Arquivo de credenciais não encontrado:${NC}"
        echo "   $credentials_path"
        echo ""
        echo -e "${CYAN}💡 Coloque o arquivo JSON de credenciais GCP em:${NC}"
        echo "   $credentials_path"
        return 1
    fi
    
    # Verificar permissões
    echo -e "${BLUE}🔒 Verificando permissões GCP...${NC}"
    if [ -f "$PROJECT_DIR/setup_gcp_permissions.sh" ]; then
        echo -e "${CYAN}   Executando setup de permissões...${NC}"
        bash "$PROJECT_DIR/setup_gcp_permissions.sh" > /dev/null 2>&1 || {
            echo -e "${YELLOW}⚠️  Algumas permissões podem estar faltando${NC}"
        }
    fi
    echo ""
    
    # Configurações
    local gcp_project_id="avian-computer-477918-j9"
    local gcp_region="us-central1"
    local service_name="parle-backend"
    
    # Usar SERVICE_NAME do test_gcp.py se disponível, senão usar padrão
    if [ -f "$docker_manager_dir/test_gcp.py" ]; then
        # Tentar extrair SERVICE_NAME do arquivo
        local extracted_name=$(grep -E "^SERVICE_NAME\s*=" "$docker_manager_dir/test_gcp.py" | head -1 | sed 's/.*=.*"\(.*\)".*/\1/' || echo "")
        if [ -n "$extracted_name" ]; then
            service_name="$extracted_name"
        fi
    fi
    
    echo -e "${BLUE}📋 Configuração:${NC}"
    echo "   Project: $gcp_project_id"
    echo "   Region: $gcp_region"
    echo "   Service: $service_name"
    echo ""
    
    # Confirmar deploy
    read -p "Continuar com deploy no Google Cloud? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}❌ Deploy cancelado${NC}"
        return 1
    fi
    
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}🚀 Iniciando deploy...${NC}"
    echo "================================================================================"
    echo ""
    
    # Verificar Dockerfile do projeto
    local project_dockerfile="$PROJECT_DIR/docker/Dockerfile"
    if [ ! -f "$project_dockerfile" ]; then
        echo -e "${YELLOW}⚠️  Dockerfile do projeto não encontrado, usando do docker-manager${NC}"
        project_dockerfile="$docker_manager_dir/Dockerfile"
    fi
    
    # Mudar para diretório docker-manager temporariamente (para usar scripts)
    local original_dir=$(pwd)
    cd "$docker_manager_dir"
    
    # Usar o script test_gcp.py para fazer deploy
    if [ -f "test_gcp.py" ]; then
        echo -e "${CYAN}📦 Fazendo build e deploy usando docker-manager...${NC}"
        echo ""
        echo -e "${YELLOW}💡 O deploy irá:${NC}"
        echo "   1. Autenticar no GCP"
        echo "   2. Habilitar APIs necessárias"
        echo "   3. Fazer build da imagem Docker do projeto"
        echo "   4. Fazer deploy no Cloud Run"
        echo "   5. Configurar o serviço"
        echo ""
        
        # Modificar temporariamente o test_gcp.py para usar Dockerfile do projeto
        # Ou passar como variável de ambiente
        export PARLE_BACKEND_ROOT="$PROJECT_DIR"
        export PARLE_BACKEND_DOCKERFILE="$project_dockerfile"
        export USE_FAST_BUILD="false"  # Padrão: build normal
        
        # Executar deploy
        python3 test_gcp.py 2>&1 | tee /tmp/gcp_deploy.log
        
        local deploy_exit_code=${PIPESTATUS[0]}
        
        if [ $deploy_exit_code -eq 0 ]; then
            echo ""
            echo "================================================================================"
            echo -e "${GREEN}✅ Deploy concluído com sucesso!${NC}"
            echo "================================================================================"
            echo ""
            
            # Obter URL do serviço
            local service_url=$(gcloud run services describe "$service_name" \
                --region "$gcp_region" \
                --project "$gcp_project_id" \
                --format="value(status.url)" 2>/dev/null)
            
            if [ -n "$service_url" ]; then
                echo -e "${GREEN}🌐 Serviço disponível em:${NC}"
                echo "   $service_url"
                echo ""
            fi
            
            echo -e "${CYAN}📋 Comandos úteis:${NC}"
            echo "   Ver logs: gcloud run services logs read $service_name --region $gcp_region"
            echo "   Ver status: gcloud run services describe $service_name --region $gcp_region"
            echo "   Medir startup: ./measure_startup.sh"
            echo ""
        else
            echo ""
            echo "================================================================================"
            echo -e "${RED}❌ Deploy falhou${NC}"
            echo "================================================================================"
            echo ""
            echo -e "${YELLOW}💡 Verifique os logs em: /tmp/gcp_deploy.log${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ test_gcp.py não encontrado em docker-manager${NC}"
        return 1
    fi
    
    # Voltar para diretório original
    cd "$PROJECT_DIR"
}

# Main
main() {
    local command="${1:-help}"
    
    case "$command" in
        setup)
            cmd_setup
            ;;
        test)
            cmd_test
            ;;
        start)
            cmd_start "$2"
            ;;
        stop)
            cmd_stop "$2"
            ;;
        restart)
            cmd_restart "$2"
            ;;
        list)
            cmd_list
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs "$2"
            ;;
        shell)
            cmd_shell
            ;;
        conda-activate)
            cmd_conda_activate
            ;;
        conda-deactivate)
            cmd_conda_deactivate
            ;;
        clean)
            cmd_clean
            ;;
    test-all)
        if [ -f "$PROJECT_DIR/scripts/test_all.sh" ]; then
            "$PROJECT_DIR/scripts/test_all.sh"
        else
            echo -e "${RED}❌ scripts/test_all.sh não encontrado${NC}"
            exit 1
        fi
        ;;
    test-services)
        cmd_test_services
        ;;
    test:demo:simple)
        cmd_test_demo_simple
        ;;
    demo)
        cmd_demo
        ;;
    monitor)
        cmd_monitor
        ;;
    benchmark)
        cmd_benchmark
        ;;
    deploy)
        cmd_deploy
        ;;
    deploy:gcp)
        cmd_deploy_gcp
        ;;
    deploy:gcp:fast)
        cmd_deploy_gcp_fast
        ;;
    start:linguistic)
        cmd_start_linguistic
        ;;
    start:acoustic)
        cmd_start_acoustic
        ;;
    docker-up)
        cmd_docker_up
        ;;
    docker-up:dev)
        cmd_docker_up_dev
        ;;
    docker-up:logging)
        cmd_docker_up_logging
        ;;
    docker-down)
        cmd_docker_down
        ;;
    docker-logs)
        cmd_docker_logs "$2"
        ;;
    docker-status)
        cmd_docker_status
        ;;
    docker-build)
        cmd_docker_build
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Comando desconhecido: $command${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
}

# Executar
main "$@"

