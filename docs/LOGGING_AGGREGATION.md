# Agregação de Logs - ELK/Loki Stack

Este documento descreve como configurar agregação de logs usando ELK Stack (Elasticsearch, Logstash, Kibana) ou Loki.

## 📋 Visão Geral

O Parle Backend já está configurado para logging estruturado em JSON. Este guia mostra como integrar com sistemas de agregação de logs.

## 🔧 Configuração Atual

O sistema já suporta:
- ✅ Logging estruturado em JSON
- ✅ Correlation IDs em todos os logs
- ✅ Níveis de log configuráveis por módulo
- ✅ Rotação e compressão de logs

## 📊 Opção 1: Loki + Grafana (Recomendado)

Loki é mais leve e integra bem com Grafana (já usado para métricas).

### Instalação com Docker Compose

```yaml
# docker/docker-compose.logging.yml
version: '3.8'

services:
  loki:
    image: grafana/loki:latest
    container_name: parle-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./logs:/var/log/parle
    networks:
      - parle-network

  promtail:
    image: grafana/promtail:latest
    container_name: parle-promtail
    volumes:
      - ./logs:/var/log/parle:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    networks:
      - parle-network
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:latest
    container_name: parle-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - parle-network
    depends_on:
      - loki
```

### Configuração do Promtail

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: parle-backend
    static_configs:
      - targets:
          - localhost
        labels:
          job: parle-backend
          __path__: /var/log/parle/*.log
    pipeline_stages:
      - json:
          expressions:
            timestamp: time
            level: level
            message: message
            service: service
            correlation_id: correlation_id
      - labels:
          level:
          service:
          correlation_id:
      - timestamp:
          source: timestamp
          format: RFC3339
```

### Queries no Grafana

```logql
# Logs de erro
{job="parle-backend"} |= "error"

# Logs por correlation ID
{job="parle-backend"} | json | correlation_id="abc123"

# Logs por serviço
{job="parle-backend", service="orchestrator"}

# Rate de erros
rate({job="parle-backend"} |= "error" [5m])
```

## 📊 Opção 2: ELK Stack (Elasticsearch, Logstash, Kibana)

### Instalação com Docker Compose

```yaml
# docker/docker-compose.elk.yml (exemplo)
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: parle-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - parle-network

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: parle-logstash
    volumes:
      - ./logs:/var/log/parle:ro
      - ./logstash-config.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    networks:
      - parle-network
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: parle-kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - parle-network
    depends_on:
      - elasticsearch
```

### Configuração do Logstash

```ruby
# logstash-config.conf
input {
  file {
    path => "/var/log/parle/*.log"
    codec => "json"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}

filter {
  json {
    source => "message"
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
  
  mutate {
    add_field => {
      "[@metadata][index]" => "parle-backend-%{+YYYY.MM.dd}"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index]}"
  }
}
```

## 🔄 Configuração do Parle Backend

### Habilitar JSON Logging

O sistema já está configurado. Certifique-se de que logs JSON estão sendo gerados:

```python
# Em produção, configure logging JSON
import json
from loguru import logger

# Configurar formato JSON
logger.add(
    "logs/api.json.log",
    format="{message}",
    serialize=True,  # JSON format
    rotation="500 MB",
    retention="30 days"
)
```

### Adicionar Correlation ID

Correlation IDs já estão sendo adicionados automaticamente via middleware.

## 📈 Dashboards

### Grafana (Loki)

1. Acesse Grafana: `http://localhost:3000`
2. Adicione Loki como data source: `http://loki:3100`
3. Crie dashboards com queries LogQL

### Kibana (ELK)

1. Acesse Kibana: `http://localhost:5601`
2. Configure index pattern: `parle-backend-*`
3. Crie visualizações e dashboards

## 🔍 Queries Úteis

### Erros por Serviço

```logql
# Loki
sum by (service) (count_over_time({job="parle-backend"} |= "error" [5m]))

# Elasticsearch
GET parle-backend-*/_search
{
  "query": {
    "match": {
      "level": "ERROR"
    }
  },
  "aggs": {
    "by_service": {
      "terms": {
        "field": "service.keyword"
      }
    }
  }
}
```

### Rastrear Request por Correlation ID

```logql
# Loki
{job="parle-backend"} | json | correlation_id="abc123"

# Elasticsearch
GET parle-backend-*/_search
{
  "query": {
    "match": {
      "correlation_id": "abc123"
    }
  },
  "sort": [
    {
      "timestamp": {
        "order": "asc"
      }
    }
  ]
}
```

## 🚀 Deploy em Produção

### Opções de Deploy

1. **Self-hosted**: Use `docker/docker-compose.yml` em servidor dedicado
2. **Cloud Managed**:
   - AWS: CloudWatch Logs + Elasticsearch Service
   - GCP: Cloud Logging + BigQuery
   - Azure: Application Insights

### Recomendações

- **Desenvolvimento/Staging**: Loki (mais leve)
- **Produção**: ELK Stack ou serviço gerenciado
- **Alta escala**: Considere serviço gerenciado (AWS, GCP, Azure)

## 📚 Referências

- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [ELK Stack Documentation](https://www.elastic.co/guide/index.html)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
