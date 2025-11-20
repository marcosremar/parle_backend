

# 📊 OpenTelemetry Observability

**Unified observability with distributed tracing, metrics, and structured logging.**

---

## 🎯 Overview

This module provides **three pillars of observability** in one unified interface:

1. **📍 Distributed Tracing** - Follow requests across services
2. **📊 Metrics Collection** - Track performance and health
3. **📝 Structured Logging** - Logs correlated with traces

**Key Benefit:** Jump from metric spike → trace → error log in **6x less time** (5 min vs 30 min).

---

## 🚀 Quick Start

### 1. Configure at Application Startup

```python
from src.core.observability import configure_telemetry

# In main.py or __init__.py
configure_telemetry(
    service_name="orchestrator",
    service_version="1.0.0",
    environment="production",
    jaeger_endpoint="localhost:6831",  # Optional: Jaeger UI
    enable_prometheus=True,             # Metrics export
)
```

### 2. Use in Your Service

```python
from src.core.observability import get_telemetry

class MyService:
    def __init__(self):
        self.telemetry = get_telemetry("my_service")

    async def process_request(self, data):
        # Create trace span
        with self.telemetry.trace("process_request"):
            # Log with automatic trace correlation
            self.telemetry.log_info("Processing started")

            # Record metric
            self.telemetry.counter("requests_total").add(1)

            # Do work
            result = await self._do_work(data)

            return result
```

### 3. View in Jaeger UI

```
http://localhost:16686
```

---

## 📚 API Reference

### UnifiedTelemetry

#### Tracing

```python
# Create a span
with telemetry.trace("operation_name"):
    # Work happens here
    pass

# With attributes
with telemetry.trace("database_query", {"table": "users", "op": "select"}):
    result = await db.query("SELECT * FROM users")

# Get current trace/span IDs
trace_id = telemetry.get_current_trace_id()  # "5f8a3b2c..."
span_id = telemetry.get_current_span_id()     # "abcdef12..."
```

#### Metrics

```python
# Counter (monotonically increasing)
telemetry.counter("requests_total").add(1, {"method": "POST"})

# Histogram (distribution)
telemetry.histogram("request_duration_ms").record(150, {"endpoint": "/api"})

# Example: Track request processing
start = time.time()
with telemetry.trace("handle_request"):
    # ... process ...
    duration_ms = (time.time() - start) * 1000
    telemetry.histogram("request_duration_ms").record(duration_ms)
```

#### Structured Logging

```python
# Logs automatically include trace_id and span_id
telemetry.log_info("Request received", extra={"user_id": "123"})
telemetry.log_warning("Slow query detected", extra={"duration_ms": 5000})
telemetry.log_error("Database connection failed", exc_info=True)

# Output:
# INFO - Processing request [trace:5f8a3b2c...] [span:abcd...]
```

---

## 🔧 Auto-Instrumentation

Automatically instrument common libraries:

```python
from src.core.observability import (
    instrument_fastapi,
    instrument_httpx,
    instrument_asyncpg,
    instrument_redis,
    instrument_all,  # Instrument everything
)

# FastAPI
app = FastAPI()
instrument_fastapi(app)  # Auto-trace all HTTP requests

# HTTP clients
instrument_httpx()  # Auto-trace httpx requests

# Database
instrument_asyncpg()  # Auto-trace asyncpg queries

# Cache
instrument_redis()  # Auto-trace Redis operations

# Or instrument everything at once
instrument_all()
```

**Result:** Automatic traces for:
- All HTTP requests to your API
- All HTTP requests you make to other services
- All database queries
- All Redis operations

No manual instrumentation needed!

---

## 🌐 Context Propagation

Propagate trace context across service boundaries:

### Outgoing Requests

```python
from src.core.observability import inject_trace_context

# Inject trace into HTTP headers
headers = {"Content-Type": "application/json"}
headers = inject_trace_context(headers)

# Make request (trace will continue)
async with httpx.AsyncClient() as client:
    response = await client.post(url, headers=headers)
```

### Incoming Requests

```python
from src.core.observability import extract_trace_context, attach_trace_context

@app.post("/process")
async def process(request: Request):
    # Extract context from incoming request
    ctx = extract_trace_context(dict(request.headers))

    # Attach to current execution
    token = attach_trace_context(ctx)

    try:
        # Process request (trace continues from caller)
        result = await do_work()
        return result
    finally:
        detach_trace_context(token)
```

---

## 📊 Visualization

### Jaeger UI (Distributed Tracing)

```
┌────────────────────────────────────────────────┐
│ Trace: 5f8a3b2c1d4e...                         │
├────────────────────────────────────────────────┤
│ api_gateway.handle_request      (200ms)        │
│   └─ orchestrator.process       (180ms)        │
│       ├─ llm.generate           (120ms)        │
│       │   └─ httpx.post         (115ms)        │
│       └─ tts.synthesize         (50ms)         │
│           └─ redis.get          (2ms)          │
└────────────────────────────────────────────────┘
```

Click any span to see:
- Duration, start time, end time
- Attributes (e.g., `{"table": "users", "op": "select"}`)
- Logs within that span
- Errors/exceptions

### Prometheus Metrics

```
# Counter
requests_total{method="POST",status="200"} 1234

# Histogram
request_duration_ms_bucket{le="100"} 500
request_duration_ms_bucket{le="500"} 800
request_duration_ms_count 1000
request_duration_ms_sum 125000
```

---

## 🎓 Examples

### Example 1: API Endpoint with Full Observability

```python
from fastapi import FastAPI, Request
from src.core.observability import get_telemetry, instrument_fastapi

app = FastAPI()
instrument_fastapi(app)

telemetry = get_telemetry("api_gateway")

@app.post("/process")
async def process(request: Request, data: dict):
    # Automatic span from FastAPI instrumentation
    # Add custom child span
    with telemetry.trace("validate_data"):
        telemetry.log_info("Validating input")
        if not data.get("text"):
            telemetry.counter("validation_errors").add(1)
            telemetry.log_warning("Missing text field")
            raise HTTPException(400, "text required")

    # Call other service (trace propagates automatically)
    async with httpx.AsyncClient() as client:
        response = await client.post("http://orchestrator:8500/process", json=data)

    # Record success metric
    telemetry.counter("requests_total").add(1, {"status": "success"})

    return response.json()
```

### Example 2: Background Worker with Observability

```python
from src.core.observability import get_telemetry

class Worker:
    def __init__(self):
        self.telemetry = get_telemetry("background_worker")

    async def process_job(self, job_id: str):
        with self.telemetry.trace("process_job", {"job_id": job_id}):
            self.telemetry.log_info(f"Processing job {job_id}")

            try:
                # Step 1
                with self.telemetry.trace("fetch_data"):
                    data = await self.fetch_data(job_id)

                # Step 2
                with self.telemetry.trace("transform_data"):
                    result = await self.transform(data)

                # Step 3
                with self.telemetry.trace("save_result"):
                    await self.save(result)

                # Success metric
                self.telemetry.counter("jobs_completed").add(1)
                self.telemetry.log_info(f"Job {job_id} completed")

            except Exception as e:
                # Error metric
                self.telemetry.counter("jobs_failed").add(1)
                self.telemetry.log_error(f"Job {job_id} failed: {e}")
                raise
```

### Example 3: BaseService Integration (TODO)

```python
from src.core.base_service import BaseService

class MyService(BaseService):
    def __init__(self, context: ServiceContext):
        super().__init__(context)
        # self.telemetry is automatically available!

    async def process(self, data):
        with self.telemetry.trace("process"):
            self.telemetry.log_info("Processing...")
            # ...
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Jaeger endpoint
OTEL_EXPORTER_JAEGER_ENDPOINT=localhost:6831

# OTLP endpoint (alternative)
OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317

# Service name
OTEL_SERVICE_NAME=my-service

# Environment
OTEL_ENVIRONMENT=production

# Sampling (1.0 = 100%, 0.1 = 10%)
OTEL_TRACES_SAMPLER=always_on
```

### Programmatic Configuration

```python
configure_telemetry(
    service_name="my-service",
    service_version="1.2.3",
    environment="staging",
    jaeger_endpoint="jaeger:6831",
    otlp_endpoint="otlp:4317",
    enable_prometheus=True,
    enable_console=False,  # Console exporter for debugging
)
```

---

## 🐳 Docker Compose Setup

```yaml
version: '3.8'

services:
  # Jaeger (tracing UI)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "6831:6831/udp"  # Jaeger agent

  # Prometheus (metrics)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Grafana (dashboards)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

---

## 📊 Best Practices

### 1. Name Spans Clearly

```python
# ✅ Good
with telemetry.trace("database_query_users"):
    ...

# ❌ Bad
with telemetry.trace("query"):
    ...
```

### 2. Add Meaningful Attributes

```python
# ✅ Good
with telemetry.trace("process_order", {
    "order_id": order.id,
    "user_id": order.user_id,
    "amount": order.total
}):
    ...

# ❌ Bad
with telemetry.trace("process"):
    ...
```

### 3. Use Correct Metric Types

```python
# Counter: Always increases (requests, errors)
telemetry.counter("requests_total").add(1)

# Histogram: Distributions (latency, size)
telemetry.histogram("request_duration_ms").record(duration)

# Gauge: Current value (connections, memory)
# (Use observable callbacks)
```

### 4. Log with Context

```python
# ✅ Good (trace_id automatically added)
telemetry.log_error("Failed to process", extra={"user_id": "123"})

# ❌ Bad (no trace context)
print("Failed to process")
```

---

## 🔧 Troubleshooting

### Jaeger not receiving traces

```bash
# Check Jaeger is running
curl http://localhost:16686

# Check endpoint config
echo $OTEL_EXPORTER_JAEGER_ENDPOINT
```

### Metrics not appearing in Prometheus

```bash
# Check Prometheus scrape config
# prometheus.yml:
scrape_configs:
  - job_name: 'ultravox'
    static_configs:
      - targets: ['api_gateway:8000']
```

### Traces not propagating between services

```python
# Ensure headers are propagated
headers = inject_trace_context({})
response = await client.post(url, headers=headers)
```

---

## 📚 Resources

- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Jaeger UI**: http://localhost:16686
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

---

**🎉 Happy Observing!**
