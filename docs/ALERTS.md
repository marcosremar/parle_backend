# Alertas Prometheus - Parle Backend

Este documento descreve os alertas recomendados para monitoramento do Parle Backend.

## 🚨 Alertas Críticos

### Alta Taxa de Erro HTTP

```yaml
- alert: HighHTTPErrorRate
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[5m])) by (endpoint)
      /
      sum(rate(http_requests_total[5m])) by (endpoint)
    ) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Alta taxa de erro HTTP (>5%) em {{ $labels.endpoint }}"
    description: "Taxa de erro de {{ $value | humanizePercentage }} em {{ $labels.endpoint }}"
```

### Alta Latência

```yaml
- alert: HighLatency
  expr: |
    histogram_quantile(0.95, 
      sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
    ) > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Latência P95 acima de 2s em {{ $labels.endpoint }}"
    description: "Latência P95 de {{ $value }}s em {{ $labels.endpoint }}"
```

### Falhas de Provedor STT

```yaml
- alert: STTProviderFailures
  expr: |
    rate(stt_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Alta taxa de erros STT"
    description: "Taxa de erro STT: {{ $value }} erros/segundo"
```

### Falhas de Provedor TTS

```yaml
- alert: TTSProviderFailures
  expr: |
    rate(tts_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Alta taxa de erros TTS"
    description: "Taxa de erro TTS: {{ $value }} erros/segundo"
```

### Falhas de Provedor LLM

```yaml
- alert: LLMProviderFailures
  expr: |
    rate(llm_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Alta taxa de erros LLM"
    description: "Taxa de erro LLM: {{ $value }} erros/segundo"
```

## ⚠️ Alertas de Aviso

### Taxa de Sucesso Baixa

```yaml
- alert: LowSuccessRate
  expr: |
    (
      sum(rate(http_requests_total{status=~"2.."}[5m]))
      /
      sum(rate(http_requests_total[5m]))
    ) < 0.95
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Taxa de sucesso abaixo de 95%"
    description: "Taxa de sucesso atual: {{ $value | humanizePercentage }}"
```

### Pipeline Lento

```yaml
- alert: SlowPipeline
  expr: |
    histogram_quantile(0.95, 
      sum(rate(pipeline_duration_seconds_bucket[5m])) by (le)
    ) > 5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Pipeline STT→LLM→TTS lento"
    description: "Duração P95 do pipeline: {{ $value }}s"
```

### Alta Taxa de Rate Limit

```yaml
- alert: HighRateLimitHits
  expr: |
    rate(rate_limit_hits_total[5m]) > 10
  for: 5m
  labels:
    severity: info
  annotations:
    summary: "Muitas requisições bloqueadas por rate limit"
    description: "{{ $value }} requisições bloqueadas por segundo"
```

## 📊 Alertas de Informação

### Pico de Tráfego

```yaml
- alert: TrafficSpike
  expr: |
    increase(http_requests_total[5m]) > 1000
  for: 1m
  labels:
    severity: info
  annotations:
    summary: "Pico de tráfego detectado"
    description: "{{ $value }} requisições nos últimos 5 minutos"
```

## 🔧 Configuração no Prometheus

Adicione os alertas ao arquivo `prometheus.yml`:

```yaml
rule_files:
  - "alerts/parle-backend.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 📧 Configuração de Notificações

### Slack

```yaml
receivers:
  - name: 'parle-team'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#parle-alerts'
        title: 'Parle Backend Alert'
        text: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

### Email

```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'team@parle.ai'
        from: 'alerts@parle.ai'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alerts@parle.ai'
        auth_password: 'password'
```

### PagerDuty

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
```

## 🎯 Priorização de Alertas

### Crítico (Pager)
- Alta taxa de erro HTTP (>5%)
- Falhas de todos os provedores
- API completamente indisponível

### Aviso (Notificação)
- Alta latência
- Falhas de provedor individual
- Taxa de sucesso baixa

### Info (Log)
- Picos de tráfego
- Rate limit hits
- Mudanças de padrão

## 📚 Referências

- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
