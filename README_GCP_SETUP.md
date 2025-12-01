# 🚀 Setup e Teste GCP - Parle Backend

Script automatizado para instalar gcloud CLI, autenticar no GCP e executar testes de Docker no Cloud Run.

## 📋 Pré-requisitos

1. **Arquivo de credenciais GCP** em:
   ```
   ~/Downloads/avian-computer-477918-j9-54b778b99398.json
   ```

2. **Homebrew** (macOS) ou acesso à internet para download

## 🎯 Uso

Execute na raiz do projeto:

```bash
./setup_gcp_test.sh
```

O script irá:

1. ✅ Verificar/instalar gcloud CLI
2. ✅ Autenticar no GCP usando o arquivo JSON
3. ✅ Configurar projeto GCP
4. ✅ Habilitar APIs necessárias
5. ✅ Fazer build e deploy no Cloud Run
6. ✅ Testar usando a API Python

## 🔒 Permissões Necessárias

O service account precisa das seguintes permissões no GCP:

- **Cloud Build Editor** (`roles/cloudbuild.builds.editor`)
- **Cloud Run Admin** (`roles/run.admin`)
- **Service Account User** (`roles/iam.serviceAccountUser`)
- **Storage Admin** (`roles/storage.admin`) - para Cloud Build

### Conceder Permissões

No GCP Console ou via gcloud:

```bash
PROJECT_ID="avian-computer-477918-j9"
SERVICE_ACCOUNT="39962934747-compute@developer.gserviceaccount.com"

# Cloud Build Editor
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/cloudbuild.builds.editor"

# Cloud Run Admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/run.admin"

# Service Account User
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/iam.serviceAccountUser"

# Storage Admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.admin"
```

## ⚙️ Configuração

Para alterar as configurações, edite as variáveis no início do script:

```bash
GCP_CREDENTIALS_PATH="${HOME}/Downloads/avian-computer-477918-j9-54b778b99398.json"
GCP_PROJECT_ID="avian-computer-477918-j9"
DOCKER_MANAGER_DIR="vendor/docker-manager"
```

## 🐛 Troubleshooting

### gcloud não encontrado após instalação

O script adiciona automaticamente ao PATH, mas se não funcionar:

```bash
# macOS Homebrew (M1/M2)
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

# macOS Homebrew (Intel)
export PATH="/usr/local/share/google-cloud-sdk/bin:$PATH"

# Instalação manual
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
```

### Erro de permissão no Cloud Build

Verifique se o service account tem as permissões necessárias (veja seção acima).

### Arquivo de credenciais não encontrado

Coloque o arquivo JSON em:
```
~/Downloads/avian-computer-477918-j9-54b778b99398.json
```

Ou altere `GCP_CREDENTIALS_PATH` no script.

## 📚 Recursos

- [Google Cloud SDK Docs](https://cloud.google.com/sdk/docs)
- [Cloud Run Docs](https://cloud.google.com/run/docs)
- [Cloud Build Docs](https://cloud.google.com/build/docs)
