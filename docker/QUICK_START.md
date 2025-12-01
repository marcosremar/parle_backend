# Quick Start - Docker

Guia rápido para executar o Parle Backend com Docker.

## 🚀 Execução Rápida

### Produção

```bash
# Da raiz do projeto
docker-compose -f docker/docker-compose.yml up -d
```

### Desenvolvimento

```bash
# Modo desenvolvimento com hot-reload
docker-compose -f docker/docker-compose.yml --profile dev up -d api-dev
```

### Com Logging (Loki + Grafana)

```bash
# Stack completo com logging
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.logging.yml \
  up -d
```

## 📋 Comandos Úteis

```bash
# Ver logs
docker-compose -f docker/docker-compose.yml logs -f api

# Parar serviços
docker-compose -f docker/docker-compose.yml down

# Rebuild
docker-compose -f docker/docker-compose.yml build

# Status
docker-compose -f docker/docker-compose.yml ps
```

## 🔧 Build Manual

```bash
# Produção
docker build -f docker/Dockerfile -t parle-backend:latest .

# Desenvolvimento
docker build -f docker/Dockerfile.dev -t parle-backend:dev .
```

## 📝 Notas

- Todos os arquivos Docker estão em `docker/`
- Execute comandos da **raiz do projeto**
- Use `-f docker/docker-compose.yml` para especificar o arquivo
