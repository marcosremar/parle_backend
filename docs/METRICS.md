# Métricas e Monitoramento - Parle Backend

Este documento descreve as métricas disponíveis no Parle Backend e como monitorá-las.

## 📊 Endpoint de Métricas

O Parle Backend expõe métricas Prometheus no endpoint `/metrics`:

```bash
curl http://localhost:8000/metrics
```

## 🔢 Métricas Disponíveis

### Métricas de Sistema

#### `http_requests_total`
Contador total de requisições HTTP.

**Labels**:
- `method`: Método HTTP (GET, POST, etc.)
- `endpoint`: Caminho do endpoint
- `status`: Código de status HTTP

**Exemplo**:
```
http_requests_total{method="POST",endpoint="/api/v1/conversation",status="200"} 1234
```

#### `http_request_duration_seconds`
Duração das requisições HTTP em segundos.

**Labels**:
- `method`: Método HTTP
- `endpoint`: Caminho do endpoint
- `status`: Código de status HTTP

**Tipo**: Histogram

**Buckets**: 0.1, 0.5, 1.0, 2.5, 5.0, 10.0

#### `http_request_size_bytes`
Tamanho das requisições HTTP em bytes.

**Tipo**: Histogram

#### `http_response_size_bytes`
Tamanho das respostas HTTP em bytes.

**Tipo**: Histogram

### Métricas de Negócio

#### `conversations_total`
Total de conversações processadas.

**Labels**:
- `status`: Status (success, error)
- `type`: Tipo (text, audio)

**Exemplo**:
```
conversations_total{status="success",type="audio"} 5678
```

#### `stt_transcriptions_total`
Total de transcrições STT.

**Labels**:
- `provider`: Provedor STT (whisper, groq, etc.)
- `status`: Status (success, error)
- `language`: Idioma detectado

#### `tts_syntheses_total`
Total de sínteses TTS.

**Labels**:
- `provider`: Provedor TTS (elevenlabs, huggingface, etc.)
- `status`: Status (success, error)
- `voice_id`: ID da voz usada

#### `llm_generations_total`
Total de gerações LLM.

**Labels**:
- `provider`: Provedor LLM (openai, litellm, etc.)
- `model`: Modelo usado
- `status`: Status (success, error)

#### `stt_errors_total`
Total de erros STT.

**Labels**:
- `error_type`: Tipo de erro (timeout, api_error, etc.)
- `provider`: Provedor que falhou

#### `tts_errors_total`
Total de erros TTS.

**Labels**:
- `error_type`: Tipo de erro
- `provider`: Provedor que falhou

#### `llm_errors_total`
Total de erros LLM.

**Labels**:
- `error_type`: Tipo de erro
- `provider`: Provedor que falhou

### Métricas de Performance

#### `stt_latency_seconds`
Latência de transcrição STT.

**Tipo**: Histogram

**Labels**:
- `provider`: Provedor STT

#### `tts_latency_seconds`
Latência de síntese TTS.

**Tipo**: Histogram

**Labels**:
- `provider`: Provedor TTS

#### `llm_latency_seconds`
Latência de geração LLM.

**Tipo**: Histogram

**Labels**:
- `provider`: Provedor LLM
- `model`: Modelo usado

#### `pipeline_duration_seconds`
Duração total do pipeline STT→LLM→TTS.

**Tipo**: Histogram

### Métricas de Autenticação

#### `auth_logins_total`
Total de tentativas de login.

**Labels**:
- `status`: Status (success, failed)

#### `auth_registrations_total`
Total de registros.

**Labels**:
- `status`: Status (success, failed)

#### `auth_token_validations_total`
Total de validações de token.

**Labels**:
- `status`: Status (valid, invalid, expired)

### Métricas de Rate Limiting

#### `rate_limit_hits_total`
Total de requisições bloqueadas por rate limit.

**Labels**:
- `endpoint`: Endpoint bloqueado
- `limit_type`: Tipo de limite (ip, user)

## 📈 Dashboards Grafana

### Métricas Recomendadas

1. **Throughput**:
   - `rate(http_requests_total[5m])`
   - `rate(conversations_total[5m])`

2. **Latência**:
   - `histogram_quantile(0.95, http_request_duration_seconds)`
   - `histogram_quantile(0.99, pipeline_duration_seconds)`

3. **Taxa de Erro**:
   - `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
   - `rate(stt_errors_total[5m]) / rate(stt_transcriptions_total[5m])`

4. **Uso de Provedores**:
   - `rate(stt_transcriptions_total[5m]) by (provider)`
   - `rate(tts_syntheses_total[5m]) by (provider)`

## 🚨 Alertas Recomendados

### Alta Taxa de Erro

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "Alta taxa de erro HTTP (>5%)"
```

### Alta Latência

```yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
  for: 5m
  annotations:
    summary: "Latência P95 acima de 2s"
```

### Falhas de Provedor

```yaml
- alert: ProviderFailures
  expr: rate(stt_errors_total[5m]) > 0.1
  for: 5m
  annotations:
    summary: "Alta taxa de erros STT"
```

## 🔍 Consultas Úteis

### Top Endpoints por Requisições

```promql
topk(10, rate(http_requests_total[5m]))
```

### Taxa de Sucesso por Endpoint

```promql
sum(rate(http_requests_total{status=~"2.."}[5m])) by (endpoint) 
/ 
sum(rate(http_requests_total[5m])) by (endpoint)
```

### Latência P95 por Endpoint

```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))
```

### Uso de Provedores

```promql
sum(rate(stt_transcriptions_total[5m])) by (provider)
```

## 📝 Adicionar Novas Métricas

Para adicionar novas métricas:

```python
from src.core.metrics import increment_counter, record_histogram

# Contador
increment_counter("my_service", "my_metric_total", labels={"label1": "value1"})

# Histograma
record_histogram("my_service", "my_duration_seconds", value=1.5, labels={"operation": "process"})
```

## 🔗 Integração

### Prometheus

Configure Prometheus para coletar métricas:

```yaml
scrape_configs:
  - job_name: 'parle-backend'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana

Importe dashboards ou crie seus próprios usando as métricas acima.

## 📚 Referências

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)
