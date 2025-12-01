# Guia de Deploy em Produção - Parle Backend

Este documento descreve como fazer deploy do Parle Backend em um ambiente de produção.

## 📋 Pré-requisitos

- Servidor Linux (Ubuntu 20.04+ recomendado)
- Docker e Docker Compose instalados
- Python 3.11+ (se não usar Docker)
- Redis (para cache e sessões)
- Banco de dados (SQLite para desenvolvimento, PostgreSQL recomendado para produção)
- Certificado SSL (Let's Encrypt recomendado)

## 🐳 Deploy com Docker (Recomendado)

### 1. Preparação

```bash
# Clone o repositório
git clone https://github.com/parle/parle_backend.git
cd parle_backend

# Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações de produção
```

### 2. Variáveis de Ambiente Críticas

```bash
# .env
ENVIRONMENT=production
JWT_SECRET_KEY=<gerar-chave-secreta-forte>
DATABASE_URL=postgresql://user:pass@host:5432/parle
REDIS_HOST=redis
REDIS_PORT=6379

# API Keys
OPENAI_API_KEY=<sua-chave>
ELEVENLABS_API_KEY=<sua-chave>
# ... outras chaves necessárias
```

**⚠️ IMPORTANTE**: Nunca commite o arquivo `.env` no repositório!

### 3. Build e Deploy

```bash
# Build da imagem (da raiz do projeto)
docker build -f docker/Dockerfile -t parle-backend:latest .

# Ou usando docker-compose (da raiz do projeto)
docker-compose -f docker/docker-compose.yml build

# Iniciar serviços
docker-compose -f docker/docker-compose.yml up -d

# Verificar logs
docker-compose -f docker/docker-compose.yml logs -f api
```

### 4. Health Check

```bash
# Verificar se a API está respondendo
curl http://localhost:8000/health

# Verificar métricas
curl http://localhost:8000/metrics
```

## 🚀 Deploy Manual (Sem Docker)

### 1. Instalação de Dependências

```bash
# Atualizar sistema
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip ffmpeg libsndfile1

# Criar usuário
sudo useradd -m -s /bin/bash parle

# Clonar repositório
sudo -u parle git clone https://github.com/parle/parle_backend.git /opt/parle
cd /opt/parle

# Criar ambiente virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuração

```bash
# Criar diretórios
sudo mkdir -p /var/log/parle /var/lib/parle/data
sudo chown -R parle:parle /var/log/parle /var/lib/parle

# Configurar variáveis de ambiente
sudo -u parle cp .env.example .env
sudo -u parle nano .env  # Editar com configurações
```

### 3. Systemd Service

Crie `/etc/systemd/system/parle-backend.service`:

```ini
[Unit]
Description=Parle Backend API
After=network.target redis.service

[Service]
Type=notify
User=parle
Group=parle
WorkingDirectory=/opt/parle
Environment="PATH=/opt/parle/venv/bin"
EnvironmentFile=/opt/parle/.env
ExecStart=/opt/parle/venv/bin/uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config /opt/parle/logging.conf
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Recarregar systemd
sudo systemctl daemon-reload

# Iniciar serviço
sudo systemctl enable parle-backend
sudo systemctl start parle-backend

# Verificar status
sudo systemctl status parle-backend
```

## 🔒 Segurança

### 1. Firewall

```bash
# Permitir apenas portas necessárias
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. SSL/TLS

Use Nginx como reverse proxy com Let's Encrypt:

```nginx
# /etc/nginx/sites-available/parle
server {
    listen 80;
    server_name api.parle.ai;
    
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name api.parle.ai;
    
    ssl_certificate /etc/letsencrypt/live/api.parle.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.parle.ai/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Variáveis Sensíveis

- Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
- Nunca commite credenciais
- Rotacione chaves regularmente

## 📊 Monitoramento

### 1. Logs

```bash
# Docker
docker-compose -f docker/docker-compose.yml logs -f api

# Systemd
sudo journalctl -u parle-backend -f

# Arquivos de log
tail -f /var/log/parle/api.log
```

### 2. Métricas

- Acesse `/metrics` para Prometheus
- Configure Grafana dashboards
- Configure alertas para métricas críticas

### 3. Health Checks

Configure health checks no seu load balancer:

```bash
# Endpoint de health check
GET /health

# Resposta esperada
{
  "status": "healthy",
  "mode": "monolith",
  "modules_initialized": true
}
```

## 🔄 Atualizações

### Processo de Deploy

1. **Backup**: Faça backup do banco de dados e configurações
2. **Teste**: Teste em ambiente de staging primeiro
3. **Deploy**: Aplique atualizações
4. **Verificação**: Verifique health checks e logs
5. **Rollback**: Tenha plano de rollback pronto

### Zero-Downtime Deploy

```bash
# 1. Build nova imagem (da raiz do projeto)
docker build -f docker/Dockerfile -t parle-backend:v1.1.0 .

# 2. Atualizar docker/docker-compose.yml com nova tag
# 3. Deploy gradual (blue-green)
docker-compose -f docker/docker-compose.yml up -d --scale api=2 --no-deps api
# Aguardar health checks
docker-compose -f docker/docker-compose.yml up -d --scale api=1 --no-deps api
```

## 🐛 Troubleshooting

Veja [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) para problemas comuns.

## 📚 Recursos Adicionais

- [Documentação da API](./API.md)
- [Guia de Configuração](./CONFIGURATION.md)
- [Monitoramento](./MONITORING.md)
