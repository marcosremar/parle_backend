#!/bin/bash
# Script para conceder permissões necessárias ao service account no GCP

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

echo -e "${BLUE}🔒 Configurando Permissões GCP${NC}"
echo "================================================================================"
echo ""

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Adicionar possíveis locais do gcloud ao PATH
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

# 1. Verificar gcloud
echo -e "${BLUE}📋 Verificando gcloud CLI...${NC}"
if ! command_exists gcloud; then
    echo -e "${RED}❌ gcloud não encontrado${NC}"
    echo "   Instale: brew install --cask google-cloud-sdk"
    exit 1
fi

echo -e "${GREEN}✅ gcloud encontrado${NC}"
gcloud --version | head -1
echo ""

# 2. Verificar arquivo de credenciais
echo -e "${BLUE}🔐 Verificando credenciais...${NC}"
if [ ! -f "$GCP_CREDENTIALS_PATH" ]; then
    echo -e "${RED}❌ Arquivo de credenciais não encontrado:${NC}"
    echo "   $GCP_CREDENTIALS_PATH"
    exit 1
fi

# Extrair service account email do JSON
SERVICE_ACCOUNT=$(python3 -c "
import json
with open('$GCP_CREDENTIALS_PATH', 'r') as f:
    data = json.load(f)
    print(data.get('client_email', ''))
" 2>/dev/null)

if [ -z "$SERVICE_ACCOUNT" ]; then
    echo -e "${RED}❌ Não foi possível extrair service account do JSON${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Service account encontrado:${NC}"
echo "   $SERVICE_ACCOUNT"
echo ""

# 3. Autenticar
echo -e "${BLUE}🔑 Autenticando...${NC}"
export GOOGLE_APPLICATION_CREDENTIALS="$GCP_CREDENTIALS_PATH"

gcloud auth activate-service-account \
    --key-file="$GCP_CREDENTIALS_PATH" \
    --quiet 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Service account já autenticado ou erro (continuando...)${NC}"
}

gcloud config set project "$GCP_PROJECT_ID" --quiet 2>/dev/null || true

echo -e "${GREEN}✅ Autenticação configurada${NC}"
echo ""

# 4. Verificar permissões atuais
echo -e "${BLUE}🔍 Verificando permissões atuais...${NC}"
echo ""

# 5. Conceder permissões
echo -e "${BLUE}🔒 Concedendo permissões necessárias...${NC}"
echo ""

PERMISSIONS_GRANTED=0
PERMISSIONS_FAILED=0

# Lista de roles necessárias
declare -a ROLES=(
    "roles/cloudbuild.builds.editor:Cloud Build Editor"
    "roles/run.admin:Cloud Run Admin"
    "roles/iam.serviceAccountUser:Service Account User"
    "roles/storage.admin:Storage Admin"
    "roles/artifactregistry.writer:Artifact Registry Writer"
    "roles/source.reader:Source Repository Reader"
)

for ROLE_INFO in "${ROLES[@]}"; do
    IFS=':' read -r ROLE ROLE_NAME <<< "$ROLE_INFO"
    
    echo -n "   Concedendo $ROLE_NAME ($ROLE)... "
    
    result=$(gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$ROLE" \
        --condition=None \
        2>&1)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅${NC}"
        ((PERMISSIONS_GRANTED++))
    else
        # Verificar se já tem a permissão
        if echo "$result" | grep -q "already has"; then
            echo -e "${YELLOW}⚠️  Já possui${NC}"
            ((PERMISSIONS_GRANTED++))
        else
            echo -e "${RED}❌${NC}"
            echo "      Erro: $result" | head -1
            ((PERMISSIONS_FAILED++))
        fi
    fi
done

echo ""

# 6. Verificar permissões concedidas
echo -e "${BLUE}📋 Verificando permissões concedidas...${NC}"
echo ""

gcloud projects get-iam-policy "$GCP_PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:$SERVICE_ACCOUNT" \
    --format="table(bindings.role)" 2>/dev/null | grep -E "roles/(cloudbuild|run|iam|storage|artifactregistry|source)" || {
    echo "   ⚠️  Não foi possível listar permissões (mas podem ter sido concedidas)"
}

echo ""

# 7. Resumo
echo "================================================================================"
echo -e "${BLUE}📊 RESUMO${NC}"
echo "================================================================================"
echo ""
echo "   Service Account: $SERVICE_ACCOUNT"
echo "   Project: $GCP_PROJECT_ID"
echo "   Permissões concedidas: $PERMISSIONS_GRANTED"
echo "   Permissões com erro: $PERMISSIONS_FAILED"
echo ""

if [ $PERMISSIONS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Todas as permissões foram configuradas com sucesso!${NC}"
    echo ""
    echo "🚀 Agora você pode executar:"
    echo "   ./setup_gcp_test.sh"
    exit 0
else
    echo -e "${YELLOW}⚠️  Algumas permissões falharam${NC}"
    echo ""
    echo "💡 Possíveis causas:"
    echo "   - Você não tem permissão para conceder IAM policies"
    echo "   - O service account não existe no projeto"
    echo "   - Verifique no GCP Console: IAM & Admin > IAM"
    exit 1
fi
