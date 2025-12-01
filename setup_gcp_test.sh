#!/bin/bash
# Script de instalação e teste GCP para Parle Backend

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
GCP_CREDENTIALS_PATH="${HOME}/Downloads/avian-computer-477918-j9-54b778b99398.json"
GCP_PROJECT_ID="avian-computer-477918-j9"
DOCKER_MANAGER_DIR="vendor/docker-manager"

echo -e "${BLUE}🚀 Setup e Teste GCP para Parle Backend${NC}"
echo "================================================================================"
echo ""

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para instalar gcloud diretamente
install_gcloud_direct() {
    echo "   Baixando Google Cloud SDK..."
    
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
        ARCH_TYPE="darwin-arm64"
    else
        ARCH_TYPE="darwin-x86_64"
    fi
    
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    curl -O "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${ARCH_TYPE}.tar.gz" || {
        echo -e "${RED}❌ Falha ao baixar Google Cloud SDK${NC}"
        echo "   Instale manualmente: https://cloud.google.com/sdk/docs/install"
        exit 1
    }
    
    tar -xzf "google-cloud-cli-${ARCH_TYPE}.tar.gz"
    ./google-cloud-sdk/install.sh --quiet --path-update=true
    
    # Adicionar ao PATH para esta sessão
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    
    # Adicionar ao .zshrc ou .bashrc
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "google-cloud-sdk/bin" "$HOME/.zshrc"; then
            echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> "$HOME/.zshrc"
        fi
    elif [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "google-cloud-sdk/bin" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> "$HOME/.bashrc"
        fi
    fi
    
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    
    # Verificar se foi instalado
    if [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
        export PATH="$HOME/google-cloud-sdk/bin:$PATH"
        echo -e "${GREEN}✅ Google Cloud SDK instalado em ~/google-cloud-sdk${NC}"
    else
        echo -e "${RED}❌ Falha na instalação${NC}"
        exit 1
    fi
}

# Adicionar possíveis locais do gcloud ao PATH antes de verificar
if [ -f "/opt/homebrew/share/google-cloud-sdk/bin/gcloud" ]; then
    export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
fi
if [ -f "/opt/homebrew/bin/gcloud" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
fi
if [ -f "/usr/local/share/google-cloud-sdk/bin/gcloud" ]; then
    export PATH="/usr/local/share/google-cloud-sdk/bin:$PATH"
fi
if [ -f "/usr/local/bin/gcloud" ]; then
    export PATH="/usr/local/bin:$PATH"
fi
if [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# 1. Verificar se gcloud está instalado
echo -e "${BLUE}📋 Verificando gcloud CLI...${NC}"
if command_exists gcloud; then
    echo -e "${GREEN}✅ gcloud já está instalado${NC}"
    gcloud --version | head -1
else
    echo -e "${YELLOW}⚠️  gcloud não encontrado. Instalando...${NC}"
    
    # Verificar sistema operacional
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - tentar encontrar brew em locais comuns
        BREW_PATH=""
        if command_exists brew; then
            BREW_PATH="brew"
        elif [ -f "/opt/homebrew/bin/brew" ]; then
            BREW_PATH="/opt/homebrew/bin/brew"
        elif [ -f "/usr/local/bin/brew" ]; then
            BREW_PATH="/usr/local/bin/brew"
        fi
        
        if [ -n "$BREW_PATH" ]; then
            echo "   Instalando via Homebrew..."
            $BREW_PATH install --cask google-cloud-sdk || {
                echo -e "${YELLOW}⚠️  Instalação via Homebrew falhou. Tentando método alternativo...${NC}"
                # Método alternativo: download direto
                install_gcloud_direct
            }
        else
            echo -e "${YELLOW}⚠️  Homebrew não encontrado.${NC}"
            echo "   Tentando instalação direta..."
            install_gcloud_direct
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "   Instalando via script oficial..."
        curl https://sdk.cloud.google.com | bash
        exec -l $SHELL
    else
        echo -e "${RED}❌ Sistema operacional não suportado automaticamente.${NC}"
        echo "   Instale manualmente: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Verificar instalação e adicionar ao PATH
    # Homebrew instala em /opt/homebrew/share/google-cloud-sdk/bin (M1/M2)
    # ou /usr/local/share/google-cloud-sdk/bin (Intel)
    if [ -f "/opt/homebrew/share/google-cloud-sdk/bin/gcloud" ]; then
        export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
    elif [ -f "/usr/local/share/google-cloud-sdk/bin/gcloud" ]; then
        export PATH="/usr/local/share/google-cloud-sdk/bin:$PATH"
    elif [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
        export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    fi
    
    # Também verificar em /opt/homebrew/bin (onde Homebrew cria symlinks)
    if [ -f "/opt/homebrew/bin/gcloud" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
    elif [ -f "/usr/local/bin/gcloud" ]; then
        export PATH="/usr/local/bin:$PATH"
    fi
    
    if ! command_exists gcloud; then
        echo -e "${RED}❌ Falha na instalação do gcloud${NC}"
        echo "   Tente reiniciar o terminal ou adicione ao PATH:"
        echo "   export PATH=\"/opt/homebrew/share/google-cloud-sdk/bin:\$PATH\""
        exit 1
    fi
    
    echo -e "${GREEN}✅ gcloud instalado com sucesso${NC}"
fi

echo ""

# 2. Verificar arquivo de credenciais
echo -e "${BLUE}🔐 Verificando credenciais GCP...${NC}"
if [ ! -f "$GCP_CREDENTIALS_PATH" ]; then
    echo -e "${RED}❌ Arquivo de credenciais não encontrado:${NC}"
    echo "   $GCP_CREDENTIALS_PATH"
    echo ""
    echo -e "${YELLOW}💡 Coloque o arquivo JSON de credenciais em:${NC}"
    echo "   $GCP_CREDENTIALS_PATH"
    exit 1
fi

echo -e "${GREEN}✅ Credenciais encontradas${NC}"
echo ""

# 3. Autenticar no GCP
echo -e "${BLUE}🔑 Autenticando no GCP...${NC}"
export GOOGLE_APPLICATION_CREDENTIALS="$GCP_CREDENTIALS_PATH"

# Tentar autenticar
gcloud auth activate-service-account \
    --key-file="$GCP_CREDENTIALS_PATH" \
    2>/dev/null || {
    # Se já estiver autenticado, apenas avisar
    echo -e "${YELLOW}⚠️  Service account já autenticado ou erro (continuando...)${NC}"
}

echo -e "${GREEN}✅ Autenticação configurada${NC}"
echo ""

# 4. Configurar projeto
echo -e "${BLUE}⚙️  Configurando projeto GCP...${NC}"
gcloud config set project "$GCP_PROJECT_ID" 2>/dev/null || true
echo -e "${GREEN}✅ Projeto configurado: $GCP_PROJECT_ID${NC}"
echo ""

# 5. Verificar se diretório docker-manager existe
echo -e "${BLUE}📁 Verificando docker-manager...${NC}"
if [ ! -d "$DOCKER_MANAGER_DIR" ]; then
    echo -e "${RED}❌ Diretório não encontrado: $DOCKER_MANAGER_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Diretório encontrado${NC}"
echo ""

# 6. Verificar permissões necessárias
echo -e "${BLUE}🔒 Verificando permissões do service account...${NC}"
echo -e "${YELLOW}💡 O service account precisa das seguintes permissões:${NC}"
echo "   - Cloud Build Editor (roles/cloudbuild.builds.editor)"
echo "   - Cloud Run Admin (roles/run.admin)"
echo "   - Service Account User (roles/iam.serviceAccountUser)"
echo "   - Storage Admin (roles/storage.admin) - para Cloud Build"
echo ""
echo -e "${YELLOW}   Para conceder permissões, execute no GCP Console ou via gcloud:${NC}"
echo "   gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \\"
echo "     --member='serviceAccount:39962934747-compute@developer.gserviceaccount.com' \\"
echo "     --role='roles/cloudbuild.builds.editor'"
echo ""

# 6. Executar teste
echo -e "${BLUE}🧪 Executando teste GCP...${NC}"
echo "================================================================================"
echo ""

cd "$DOCKER_MANAGER_DIR"

# Verificar se Python está disponível
if ! command_exists python3; then
    echo -e "${RED}❌ python3 não encontrado${NC}"
    exit 1
fi

# Executar script de teste
python3 test_gcp.py

EXIT_CODE=$?

# 7. Medir tempo de inicialização (se deploy foi bem-sucedido)
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 1 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}⏱️  Medindo tempo de inicialização...${NC}"
    echo "================================================================================"
    echo ""
    
    # Verificar se requests está instalado
    python3 -c "import requests" 2>/dev/null || {
        echo "   Instalando biblioteca 'requests'..."
        pip3 install requests --quiet 2>/dev/null || true
    }
    
    python3 test_startup_time.py
    STARTUP_EXIT_CODE=$?
    
    if [ $STARTUP_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Medição de tempo concluída${NC}"
    else
        echo -e "${YELLOW}⚠️  Medição de tempo falhou (mas não é crítico)${NC}"
    fi
fi

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Teste concluído com sucesso!${NC}"
else
    echo -e "${RED}❌ Teste falhou com código: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
