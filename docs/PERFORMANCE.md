# Performance - Parle Backend

Este documento descreve considerações de performance e otimizações do Parle Backend.

## 📊 Métricas de Performance

### Benchmarks Esperados

- **Latência P95**: < 2s para pipeline completo (STT→LLM→TTS)
- **Throughput**: > 100 requisições/segundo
- **Tempo de resposta HTTP**: < 500ms para endpoints simples
- **Uso de memória**: < 2GB por worker

## ⚡ Otimizações Implementadas

### 1. Arquitetura Monolítica Modular

- **Benefício**: Chamadas diretas Python (sem overhead HTTP)
- **Ganho**: ~10-100x mais rápido que comunicação HTTP entre serviços

### 2. Lazy Loading de Módulos

- Módulos são carregados apenas quando necessário
- Reduz tempo de inicialização
- Economiza memória

### 3. Connection Pooling

- Pool de conexões para banco de dados
- Pool de conexões HTTP (aiohttp, httpx)
- Reutilização de conexões reduz latência

### 4. Caching

- Redis para cache de dados frequentes
- Cache de resultados de LLM (quando apropriado)
- Cache de sessões de usuário

### 5. Async/Await

- Todas as operações I/O são assíncronas
- Permite processamento concorrente
- Melhor utilização de recursos

## 🔍 Profiling

### Ferramentas Recomendadas

1. **cProfile**: Profiling de código Python
```bash
python -m cProfile -o profile.stats src/api/main.py
```

2. **py-spy**: Profiling em tempo real
```bash
pip install py-spy
py-spy record -o profile.svg -- python src/api/main.py
```

3. **memory_profiler**: Análise de memória
```bash
pip install memory-profiler
python -m memory_profiler src/api/main.py
```

### Endpoints Críticos para Profiling

- `/api/v1/conversation` - Pipeline completo
- `/api/v1/speech/stt/transcribe` - STT
- `/api/v1/speech/tts/synthesize` - TTS
- `/api/v1/llm/generate` - LLM

## 🐌 Gargalos Comuns

### 1. Chamadas a APIs Externas

**Problema**: Latência de APIs externas (OpenAI, ElevenLabs, etc.)

**Soluções**:
- Timeouts adequados
- Circuit breakers
- Retry com backoff exponencial
- Cache quando apropriado

### 2. Processamento de Áudio

**Problema**: Processamento de áudio pode ser CPU-intensivo

**Soluções**:
- Processamento assíncrono
- Workers dedicados se necessário
- Otimização de formatos de áudio

### 3. Queries de Banco de Dados

**Problema**: Queries lentas ou N+1 queries

**Soluções**:
- Índices adequados
- Eager loading quando necessário
- Paginação
- Cache de queries frequentes

### 4. Serialização JSON

**Problema**: Serialização de grandes objetos JSON

**Soluções**:
- Usar `orjson` (já implementado)
- Limitar tamanho de respostas
- Streaming para grandes dados

## 📈 Escalabilidade

### Horizontal Scaling

- Múltiplos workers com uvicorn
- Load balancer na frente
- Stateless design (sessões em Redis)

### Vertical Scaling

- Aumentar workers conforme CPU cores
- Aumentar memória se necessário
- GPU para processamento de áudio (se aplicável)

### Configuração de Workers

```python
# Recomendação: workers = CPU cores
workers = 4  # Para servidor com 4 cores
```

## 🔧 Configurações de Performance

### Uvicorn

```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,  # Ajustar conforme CPU
    loop="uvloop",  # Mais rápido que asyncio padrão
    log_level="info"
)
```

### Timeouts

- **STT**: 30s
- **TTS**: 30s
- **LLM**: 60s
- **HTTP requests**: 10s

### Connection Limits

- **Database**: Pool size = 20
- **Redis**: Connection pool = 10
- **HTTP clients**: Max connections = 100

## 📊 Monitoramento

### Métricas Importantes

1. **Latência**:
   - `histogram_quantile(0.95, http_request_duration_seconds)`
   - `histogram_quantile(0.99, pipeline_duration_seconds)`

2. **Throughput**:
   - `rate(http_requests_total[5m])`
   - `rate(conversations_total[5m])`

3. **Recursos**:
   - CPU usage
   - Memory usage
   - Network I/O

### Alertas

- Latência P95 > 2s
- CPU usage > 80%
- Memory usage > 90%
- Error rate > 5%

## 🚀 Melhorias Futuras

1. **Caching mais agressivo**
   - Cache de respostas LLM
   - Cache de transcrições STT

2. **Otimização de queries**
   - Análise de queries lentas
   - Otimização de índices

3. **CDN para arquivos estáticos**
   - Servir arquivos de áudio via CDN

4. **Compressão**
   - GZip já implementado
   - Considerar Brotli

5. **Database read replicas**
   - Para alta carga de leitura

## 📚 Referências

- [FastAPI Performance](https://fastapi.tiangolo.com/advanced/concurrency/)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Async Python Best Practices](https://docs.python.org/3/library/asyncio-dev.html)
