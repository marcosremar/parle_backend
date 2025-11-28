#!/bin/bash

# Parle Backend - Simple Deploy Script
# Deploys monolith to a VPS via SSH + Docker Compose

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SSH_HOST="${DEPLOY_HOST:-}"
SSH_USER="${DEPLOY_USER:-root}"
SSH_PORT="${DEPLOY_PORT:-22}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/parle_backend}"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Functions
show_help() {
    echo -e "${CYAN}Parle Backend - Deploy Script${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST        SSH host (required)"
    echo "  -u, --user USER        SSH user (default: root)"
    echo "  -p, --port PORT        SSH port (default: 22)"
    echo "  -d, --path PATH        Deploy path (default: /opt/parle_backend)"
    echo "  --help                 Show this help"
    echo ""
    echo "Environment variables:"
    echo "  DEPLOY_HOST           SSH host"
    echo "  DEPLOY_USER           SSH user"
    echo "  DEPLOY_PORT           SSH port"
    echo "  DEPLOY_PATH           Deploy path"
    echo ""
    echo "Example:"
    echo "  $0 -h myserver.com -u deploy -p 2222"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            SSH_HOST="$2"
            shift 2
            ;;
        -u|--user)
            SSH_USER="$2"
            shift 2
            ;;
        -p|--port)
            SSH_PORT="$2"
            shift 2
            ;;
        -d|--path)
            DEPLOY_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$SSH_HOST" ]; then
    echo -e "${RED}❌ SSH host is required${NC}"
    echo "   Use -h/--host or set DEPLOY_HOST environment variable"
    show_help
    exit 1
fi

echo -e "${CYAN}🚀 Deploying Parle Backend to ${SSH_USER}@${SSH_HOST}:${SSH_PORT}${NC}"
echo -e "   Path: ${DEPLOY_PATH}"
echo ""

# Check if Docker and docker-compose are installed on remote
echo -e "${CYAN}📋 Checking remote requirements...${NC}"
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "command -v docker >/dev/null 2>&1 || { echo 'Docker not found'; exit 1; }"
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "command -v docker-compose >/dev/null 2>&1 || { echo 'docker-compose not found'; exit 1; }"
echo -e "${GREEN}✅ Remote requirements OK${NC}"
echo ""

# Create deploy directory
echo -e "${CYAN}📁 Creating deploy directory...${NC}"
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "mkdir -p ${DEPLOY_PATH}"
echo -e "${GREEN}✅ Directory created${NC}"
echo ""

# Copy files
echo -e "${CYAN}📦 Copying files...${NC}"
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.env' --exclude 'data' --exclude '*.log' \
    -e "ssh -p ${SSH_PORT}" \
    ./ "${SSH_USER}@${SSH_HOST}:${DEPLOY_PATH}/"
echo -e "${GREEN}✅ Files copied${NC}"
echo ""

# Build and start services
echo -e "${CYAN}🔨 Building and starting services...${NC}"
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" << EOF
    cd ${DEPLOY_PATH}
    docker-compose down || true
    docker-compose build
    docker-compose up -d
EOF

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo -e "${CYAN}📊 Services:${NC}"
echo -e "   API: http://${SSH_HOST}:8000"
echo -e "   WebSocket: ws://${SSH_HOST}:8022"
echo ""
echo -e "${CYAN}💡 Useful commands:${NC}"
echo -e "   ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} 'cd ${DEPLOY_PATH} && docker-compose logs -f'"
echo -e "   ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} 'cd ${DEPLOY_PATH} && docker-compose restart'"
echo -e "   ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} 'cd ${DEPLOY_PATH} && docker-compose down'"
