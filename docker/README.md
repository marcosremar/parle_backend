# Docker - Parle Backend

Este diretório contém todos os arquivos relacionados ao Docker.

## 📁 Arquivos

- `Dockerfile` - Imagem de produção otimizada
- `Dockerfile.dev` - Imagem de desenvolvimento com hot-reload
- `docker-compose.yml` - Orquestração completa (API + Redis)
- `docker-compose.logging.yml` - Stack de logging (Loki + Grafana)
- `.dockerignore` - Arquivos excluídos do build

## 🚀 Uso

### Produção

```bash
# Build e executar
cd docker
docker-compose up -d

# Ou da raiz do projeto
docker-compose -f docker/docker-compose.yml up -d
```

### Desenvolvimento

```bash
# Executar em modo desenvolvimento
docker-compose -f docker/docker-compose.yml --profile dev up -d api-dev
```

### Com Logging

```bash
# Executar com stack de logging
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.logging.yml \
  up -d
```

## 🔧 Build Manual

### Produção

```bash
docker build -f docker/Dockerfile -t parle-backend:latest ..
```

### Desenvolvimento

```bash
docker build -f docker/Dockerfile.dev -t parle-backend:dev ..
```

## 📝 Notas

- Todos os arquivos Docker estão organizados neste diretório
- Os `docker-compose.yml` usam `context: ..` para acessar a raiz do projeto
- Volumes são mapeados relativos à raiz do projeto (`../logs`, `../data`)
