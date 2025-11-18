# Service Manager

**Central Orchestration and Management System for Ultravox Pipeline Services**

Version: 2.0 (Modular Architecture)
Last Updated: 2025-10-14

---

## 🎉 v2.0 - Modular Architecture (NEW!)

**Status:** ✅ Refactored (Phase 4.2 - October 2025-10-14)

The Service Manager has been split into focused, testable modules:

```
src/core/service_manager/
├── main.py (~200 lines)       # Orchestration only ✅ NEW
├── loader.py                   # Service loading & DI ✅ NEW
├── lifecycle.py                # Start/stop/restart ✅ NEW
├── router.py                   # FastAPI route mounting ✅ NEW
├── monitoring/                 # Health checks & metrics
├── discovery/                  # Service discovery
└── ... (other modules)
```

**Improvement:**
- **-85% code per file** (2825 lines → ~200-400 lines per module)
- **Better separation of concerns** (each module = one responsibility)
- **Improved testability** (test modules independently)

**See full details in:** [v2.0 Modular Architecture Guide](#v20-modular-architecture-guide) (section below)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Service Installation](#service-installation)
6. [API Reference](#api-reference)
7. [Remote Deployment](#remote-deployment)
8. [Health Monitoring](#health-monitoring)
9. [Performance Metrics](#performance-metrics)
10. [Configuration](#configuration)
11. [Integration with Machine Manager](#integration-with-machine-manager)
12. [SystemD Integration](#systemd-integration)
13. [Task Scheduler](#task-scheduler)
14. [Service Discovery & Lifecycle Management](#service-discovery--lifecycle-management)
15. [Troubleshooting](#troubleshooting)
16. [Development Guide](#development-guide)

---

## 📋 Overview

The **Service Manager** is the central orchestration system for the Ultravox Pipeline. It provides unified management for all microservices including LLM (Ultravox), STT (Whisper), TTS (Kokoro), and auxiliary services.

### Key Capabilities

- ✅ **Service Discovery** - Automatic service registration and discovery
- ✅ **Remote Launching** - Deploy services on RunPod pods with one command
- ✅ **Health Monitoring** - Continuous health checks with auto-restart
- ✅ **Automated Installation** - One-command setup for all services
- ✅ **Performance Metrics** - Real-time tracking of latency, throughput, and resources
- ✅ **Activity Tracking** - Monitor service usage and lifecycle events
- ✅ **Profile Management** - Pre-configured service profiles for different scenarios
- ✅ **SystemD Integration** - Production-ready systemd service management
- ✅ **Task Scheduling** - Cron-like task scheduling with celery integration

### Managed Services

| Service | Type | Port | Purpose | GPU Required |
|---------|------|------|---------|--------------|
| **LLM** | AI Model | 8100 | Speech-to-Speech (Ultravox) | Yes (8GB+ VRAM) |
| **STT** | AI Model | 8101 | Speech-to-Text (Whisper) | Optional |
| **TTS** | AI Model | 8102 | Text-to-Speech (Kokoro) | No |
| **API Gateway** | API | 8000 | Main HTTP API endpoint | No |
| **WebSocket** | Communication | 8001 | Real-time WebSocket server | No |
| **File Storage** | Storage | 8003 | File upload/download service | No |
| **Session Manager** | State | 8004 | Session state management | No |
| **Orchestrator** | Coordination | 8005 | Service orchestration | No |
| **Metrics Testing** | Testing | 8006 | Performance testing | No |

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Service Manager (Port 8888)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│  │ Service Launcher  │  │  Remote Launcher  │  │  Registry   │ │
│  │ - Start/Stop      │  │  - Pod Mgmt       │  │  - Services │ │
│  │ - Local Services  │  │  - SSH Tunneling  │  │  - Discovery│ │
│  └───────────────────┘  └───────────────────┘  └─────────────┘ │
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│  │  Health Monitor   │  │  Metrics Tracker  │  │  Activity   │ │
│  │  - HTTP Checks    │  │  - Latency        │  │  - Usage    │ │
│  │  - Auto-restart   │  │  - Throughput     │  │  - Lifecycle│ │
│  └───────────────────┘  └───────────────────┘  └─────────────┘ │
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│  │ Profile Manager   │  │  Task Scheduler   │  │  SystemD    │ │
│  │  - Presets        │  │  - Cron Tasks     │  │  - Manager  │ │
│  │  - Benchmarks     │  │  - Celery         │  │  - Watchdog │ │
│  └───────────────────┘  └───────────────────┘  └─────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────┐
        │          Managed Services (Local/Remote)     │
        ├─────────────────────────────────────────────┤
        │  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────┐ │
        │  │ LLM  │  │ STT  │  │ TTS  │  │ Gateway │ │
        │  │:8100 │  │:8101 │  │:8102 │  │ :8000  │ │
        │  └──────┘  └──────┘  └──────┘  └────────┘ │
        │                                             │
        │  ┌────────┐  ┌──────────┐  ┌───────────┐  │
        │  │WebSocket│  │  Storage │  │  Session  │  │
        │  │ :8001  │  │  :8003   │  │  :8004    │  │
        │  └────────┘  └──────────┘  └───────────┘  │
        └─────────────────────────────────────────────┘
```

### Component Communication Flow

```
┌──────────┐         ┌──────────────────┐         ┌──────────┐
│  Client  │────────▶│  Service Manager │────────▶│ Service  │
│  (curl)  │  HTTP   │   (Port 8888)    │   SSH   │  (Pod)   │
└──────────┘         └──────────────────┘         └──────────┘
                              │
                              │ Health Checks
                              ▼
                     ┌──────────────────┐
                     │  Metrics Storage │
                     │  (In-Memory)     │
                     └──────────────────┘
```

---

## 🚀 Quick Start

### 1. Start Service Manager

```bash
# Basic start (default port 8888)
python3 -m src.core.service_manager.main

# With custom port
python3 -m src.core.service_manager.main --port 9000

# With debug logging
LOG_LEVEL=DEBUG python3 -m src.core.service_manager.main

# Production mode (systemd)
sudo systemctl start ultravox-service-manager
```

### 2. Verify Service Manager

```bash
# Health check
curl http://localhost:8888/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0",
  "uptime_seconds": 123.45,
  "services_registered": 9
}
```

### 3. List Available Services

```bash
curl http://localhost:8888/services

# Response:
{
  "services": [
    {
      "id": "llm",
      "status": "running",
      "port": 8100,
      "health": "healthy",
      "uptime": "2h 15m"
    },
    ...
  ]
}
```

### 4. Install Services

```bash
# Install all AI services
python3 src/services/llm/install.py
python3 src/services/stt/install.py
python3 src/services/tts/install.py

# Or use the quick install script
chmod +x install_all_services.sh
./install_all_services.sh
```

**For detailed installation:** See [SERVICE_INSTALLATION.md](./SERVICE_INSTALLATION.md)

---

## 🔧 Core Components

### 1. Service Launcher (`service_launcher.py`)

Manages local service processes.

**Features:**
- Start/stop services locally
- Process management
- Port allocation
- Environment setup
- Log file management

**Usage:**
```python
from src.core.service_manager.service_launcher import ServiceLauncher

launcher = ServiceLauncher()
await launcher.start_service(
    service_id="llm",
    command="python3 -m src.services.llm.service",
    port=8100,
    env={"GPU_MEMORY_UTILIZATION": "0.85"}
)
```

**Key Methods:**
- `start_service(service_id, command, port, env)` - Start a service
- `stop_service(service_id)` - Stop a service
- `restart_service(service_id)` - Restart a service
- `get_service_status(service_id)` - Get service status

---

### 2. Remote Launcher (`remote_launcher.py`)

Deploys services to remote RunPod pods.

**Features:**
- Pod provisioning via Machine Manager
- SSH tunnel management
- Automated code sync (git pull)
- Dependency installation
- Service startup
- Health verification
- Auto-benchmark

**Usage:**
```python
from src.core.service_manager.remote_launcher import get_remote_launcher

launcher = await get_remote_launcher()

result = await launcher.launch_service(service_info)

if result.success:
    print(f"✅ Service: {result.service_url}")
    print(f"📡 Pod ID: {result.service_id}")
else:
    print(f"❌ Error: {result.error_message}")
```

**Deployment Flow:**
1. ✅ Ensure pod is running
2. ✅ Wait for SSH access
3. ✅ Sync code (git commit local → push → pull on pod)
4. ✅ Create virtualenv
5. ✅ Install dependencies
6. ✅ Run service install script
7. ✅ Start service process
8. ✅ Health check
9. ✅ Run benchmark (optional)

**Configuration:** `config/runpod_services.yaml`

---

### 3. Service Registry (`service_registry.py`)

Central registry for all services.

**Features:**
- Service registration
- Discovery endpoints
- Health status tracking
- Metadata storage

**Usage:**
```python
from src.core.service_manager.service_registry import ServiceRegistry

registry = ServiceRegistry()

# Register service
registry.register_service(
    service_id="llm",
    host="localhost",
    port=8100,
    metadata={"type": "ai_model", "gpu": True}
)

# Discover service
service = registry.get_service("llm")
print(f"LLM at {service.host}:{service.port}")

# List all services
services = registry.list_services()
```

**Data Structure:**
```python
{
    "service_id": "llm",
    "host": "localhost",
    "port": 8100,
    "status": "running",
    "health": "healthy",
    "registered_at": "2025-10-07T10:30:00Z",
    "metadata": {
        "type": "ai_model",
        "gpu": True,
        "vram_gb": 12
    }
}
```

---

### 4. Health Monitor (`main.py` - integrated)

Continuous health monitoring for all services.

**Features:**
- HTTP health checks (configurable interval)
- Auto-restart on failures
- Health history tracking
- Alerting (planned)

**Configuration:**
```python
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 10   # seconds
MAX_FAILURES_BEFORE_RESTART = 3
```

**Health Check Flow:**
```
┌─────────┐
│ Service │
└─────────┘
     │
     │ Every 30s
     ▼
┌──────────────┐      ┌────────────┐
│ Health Check │─────▶│  /health   │
│   Monitor    │      │  endpoint  │
└──────────────┘      └────────────┘
     │
     │ If fails 3x
     ▼
┌──────────────┐
│ Auto-Restart │
│   Service    │
└──────────────┘
```

---

### 5. Metrics Tracker (`metrics_tracker.py`)

Real-time performance metrics collection.

**Metrics Collected:**
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Response time distribution
- CPU/Memory usage (optional)

**Usage:**
```python
from src.core.service_manager.metrics_tracker import MetricsTracker

tracker = MetricsTracker()

# Record request
tracker.record_request(
    service_id="llm",
    latency_ms=123.45,
    success=True
)

# Get metrics
metrics = tracker.get_metrics("llm")
print(f"P95 latency: {metrics.latency_p95}ms")
print(f"RPS: {metrics.requests_per_second}")
print(f"Error rate: {metrics.error_rate}%")
```

**Metrics API:**
```bash
# Get service metrics
curl http://localhost:8888/metrics/llm

# Response:
{
  "service_id": "llm",
  "latency_p50": 98.2,
  "latency_p95": 156.7,
  "latency_p99": 234.5,
  "requests_per_second": 12.3,
  "error_rate": 0.01,
  "total_requests": 1234,
  "uptime_seconds": 7200
}
```

---

### 6. Activity Tracker (`activity_tracker.py`)

Tracks service usage and lifecycle events.

**Events Tracked:**
- Service start/stop
- Request count
- Error occurrences
- Health check results
- Configuration changes

**Usage:**
```python
from src.core.service_manager.activity_tracker import ActivityTracker

tracker = ActivityTracker()

# Log activity
tracker.log_activity(
    service_id="llm",
    activity_type="request",
    details={"endpoint": "/v1/completions", "duration_ms": 123}
)

# Get activity history
history = tracker.get_activity("llm", limit=100)
```

---

### 7. Profile Manager (`profile_endpoints.py`)

Pre-configured service profiles for different scenarios.

**Built-in Profiles:**
- `production` - Optimized for production workloads
- `development` - Fast iteration, debug logging
- `benchmark` - Performance testing configuration
- `low_memory` - Reduced memory footprint

**Usage:**
```bash
# List profiles
curl http://localhost:8888/profiles

# Get profile details
curl http://localhost:8888/profiles/production

# Apply profile
curl -X POST http://localhost:8888/profiles/production/apply
```

**Profile Structure:**
```yaml
production:
  llm:
    gpu_memory_utilization: 0.90
    max_workers: 4
    timeout_seconds: 30
  stt:
    model: "whisper-large-v3-turbo"
    batch_size: 8
```

---

### 8. SystemD Integration (`systemd_manager.py`, `systemd_watchdog.py`)

Production-ready systemd service management.

**Features:**
- Automatic service file generation
- Systemd watchdog support
- Auto-restart on crashes
- Journal logging integration

**Generate Service Files:**
```bash
python3 -m src.core.service_manager.systemd_integration_example
```

**Service File Generated:** `/etc/systemd/system/ultravox-service-manager.service`

```ini
[Unit]
Description=Ultravox Service Manager
After=network.target

[Service]
Type=notify
WatchdogSec=30
User=ultravox
WorkingDirectory=/opt/ultravox-pipeline
ExecStart=/usr/bin/python3 -m src.core.service_manager.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Usage:**
```bash
# Enable and start
sudo systemctl enable ultravox-service-manager
sudo systemctl start ultravox-service-manager

# Check status
sudo systemctl status ultravox-service-manager

# View logs
sudo journalctl -u ultravox-service-manager -f
```

---

### 9. Task Scheduler (`task_integration.py`)

Celery-based task scheduling for background jobs.

**Features:**
- Cron-like scheduling
- Periodic tasks
- Background job processing
- Task result tracking

**Usage:**
```python
from src.core.service_manager.task_integration import schedule_task

# Schedule daily model update
schedule_task(
    task_name="update_models",
    schedule="0 0 * * *",  # Daily at midnight
    task_function=update_ultravox_models
)

# One-time task
schedule_task(
    task_name="benchmark",
    schedule="once",
    task_function=run_benchmark,
    args=["llm"]
)
```

---

## 📦 Service Installation

**Complete installation guide:** [SERVICE_INSTALLATION.md](./SERVICE_INSTALLATION.md)

### Quick Reference

```bash
# LLM Service (Ultravox)
python3 src/services/llm/install.py
# Requires: GPU 8GB+, 25GB disk, vLLM

# STT Service (Whisper)
python3 src/services/stt/install.py
# Requires: 10GB disk, PyTorch, Whisper

# TTS Service (Kokoro)
python3 src/services/tts/install.py
# Requires: 5GB disk, Kokoro package
```

### Installation Matrix

| Service | Disk | GPU | Install Time | Validation |
|---------|------|-----|--------------|------------|
| LLM | 25GB | Yes (8GB+) | 10-15 min | `curl localhost:8100/health` |
| STT | 10GB | Optional | 5-10 min | `curl localhost:8101/health` |
| TTS | 5GB | No | 5-10 min | `curl localhost:8102/health` |

---

## 🌐 API Reference

### Service Manager Endpoints

#### Health & Status

```bash
# Service Manager health
GET /health
Response: {"status": "healthy", "uptime_seconds": 123}

# List all services
GET /services
Response: {"services": [...]}

# Get service details
GET /services/{service_id}
Response: {"id": "llm", "status": "running", ...}
```

#### Service Control

```bash
# Start service
POST /services/{service_id}/start
Body: {"config": {...}}
Response: {"status": "started", "pid": 1234}

# Stop service
POST /services/{service_id}/stop
Response: {"status": "stopped"}

# Restart service
POST /services/{service_id}/restart
Response: {"status": "restarted"}
```

#### Discovery

```bash
# Register service
POST /discovery/register
Body: {
  "service_id": "llm",
  "host": "localhost",
  "port": 8100
}

# Discover services
GET /discovery/services
Response: {"services": [...]}

# Health check specific service
GET /discovery/health/{service_id}
```

#### Metrics

```bash
# Get service metrics
GET /metrics/{service_id}
Response: {
  "latency_p50": 98.2,
  "latency_p95": 156.7,
  "requests_per_second": 12.3
}

# Get all metrics
GET /metrics
Response: {"services": {"llm": {...}, "stt": {...}}}
```

#### Profiles

```bash
# List profiles
GET /profiles
Response: {"profiles": ["production", "development", ...]}

# Get profile
GET /profiles/{profile_name}
Response: {...}

# Apply profile
POST /profiles/{profile_name}/apply
Response: {"status": "applied"}

# Run benchmark
POST /profiles/{profile_name}/run
Response: {"metrics": {...}}
```

---

## 🚀 Remote Deployment

### Deploy Service to RunPod

```python
from src.core.service_manager.remote_launcher import get_remote_launcher
from src.config.service_execution_config import ServiceExecutionInfo, ExecutionMode

# Configure service
service_info = ServiceExecutionInfo(
    service_id="llm",
    execution_mode=ExecutionMode.REMOTE,
    remote_host="pod-ip",
    remote_port=8100,
    gpu_memory_utilization=0.85,
    auto_scale=True,
    idle_timeout_seconds=300
)

# Deploy
launcher = await get_remote_launcher()
result = await launcher.launch_service(service_info)

if result.success:
    print(f"✅ Service URL: {result.service_url}")
else:
    print(f"❌ Error: {result.error_message}")
```

### Remote Deployment Configuration

**File:** `config/runpod_services.yaml`

```yaml
services:
  llm:
    enabled: true
    port: 8100
    gpu_memory_utilization: 0.85
    model: "fixie-ai/ultravox-v0_8"
    post_start_benchmark:
      enabled: true
      iterations: 10
      wait_for_service_seconds: 15

  stt:
    enabled: true
    port: 8101
    provider: "local"  # or "groq"
    model: "whisper-large-v3-turbo"

  tts:
    enabled: true
    port: 8102
    provider: "kokoro"  # or "azure"
```

### Automated Setup Commands

The Remote Launcher executes these commands on the pod:

```bash
# 1. Base Setup
cd ~/ultravox-pipeline
mkdir -p logs
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Service-Specific (LLM example)
pip install vllm
python3 src/services/llm/install.py

# 3. Start Service
nohup python -m src.services.llm.service --port 8100 > logs/llm.log 2>&1 &

# 4. Health Check
curl http://localhost:8100/health
```

---

## 🔍 Health Monitoring

### Health Check Configuration

```python
# main.py
HEALTH_CHECK_CONFIG = {
    "interval_seconds": 30,
    "timeout_seconds": 10,
    "max_failures": 3,
    "auto_restart": True
}
```

### Health Check Endpoints

Each service must implement:

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "llm",
        "model": "ultravox-v0_8",
        "gpu_available": True,
        "uptime_seconds": 7200
    }
```

### Monitoring Dashboard

```bash
# View health status
curl http://localhost:8888/services

# Response:
{
  "services": [
    {
      "id": "llm",
      "status": "running",
      "health": "healthy",
      "last_check": "2025-10-07T10:30:15Z",
      "consecutive_failures": 0,
      "uptime": "2h 15m"
    }
  ]
}
```

---

## 📊 Performance Metrics

### Metrics Collection

```python
from src.core.service_manager.metrics_tracker import MetricsTracker

tracker = MetricsTracker()

# Record request
tracker.record_request(
    service_id="llm",
    latency_ms=123.45,
    success=True,
    metadata={"endpoint": "/v1/completions"}
)

# Get real-time metrics
metrics = tracker.get_metrics("llm")
```

### Available Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `latency_p50` | Median latency | milliseconds |
| `latency_p95` | 95th percentile latency | milliseconds |
| `latency_p99` | 99th percentile latency | milliseconds |
| `requests_per_second` | Throughput | requests/sec |
| `error_rate` | Percentage of failed requests | percentage |
| `total_requests` | Total request count | count |
| `uptime_seconds` | Service uptime | seconds |

### Metrics API

```bash
# Get service metrics
curl http://localhost:8888/metrics/llm | jq

# Output:
{
  "service_id": "llm",
  "latency_p50": 98.2,
  "latency_p95": 156.7,
  "latency_p99": 234.5,
  "requests_per_second": 12.3,
  "error_rate": 0.01,
  "total_requests": 1234,
  "uptime_seconds": 7200,
  "last_updated": "2025-10-07T10:30:00Z"
}
```

---

## ⚙️ Configuration

### Service Manager Config

**File:** `config/service_manager.yaml` (planned)

```yaml
service_manager:
  host: "0.0.0.0"
  port: 8888
  log_level: "INFO"

  health_checks:
    enabled: true
    interval_seconds: 30
    timeout_seconds: 10
    max_failures: 3
    auto_restart: true

  metrics:
    enabled: true
    retention_hours: 24

  discovery:
    enabled: true
    announce_interval_seconds: 60
```

### Service Execution Config

**File:** `src/config/service_execution_config.py`

```python
from dataclasses import dataclass
from enum import Enum

class ExecutionMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"

@dataclass
class ServiceExecutionInfo:
    service_id: str
    execution_mode: ExecutionMode
    remote_host: str = None
    remote_port: int = None
    gpu_memory_utilization: float = 0.85
    auto_scale: bool = False
    idle_timeout_seconds: int = 300
```

---

## 🔗 Integration with Machine Manager

The Service Manager integrates with the [Machine Manager](../../services/machine_manager/README.md) for pod provisioning.

### Workflow

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────┐
│ Service Manager  │────────▶│ Machine Manager  │────────▶│  RunPod  │
│                  │  Deploy │                  │ Provision│          │
└──────────────────┘         └──────────────────┘         └──────────┘
        │                              │
        │ 2. SSH Tunnel                │ 3. Pod Ready
        │◀─────────────────────────────┘
        │
        │ 4. Install & Start Service
        ▼
┌──────────────────┐
│   Service (Pod)  │
│   :8100          │
└──────────────────┘
```

### Usage Example

```python
from src.services.machine_manager.manager import MachineManager
from src.core.service_manager.remote_launcher import RemoteServiceLauncher

# 1. Provision pod via Machine Manager
machine_mgr = MachineManager()
pod = await machine_mgr.provision_gpu_instance(
    gpu_type="RTX A4000",
    image="ultravox-base:latest"
)

# 2. Deploy service via Remote Launcher
remote_launcher = RemoteServiceLauncher()
result = await remote_launcher.launch_service(service_info)
```

---

## 🖥️ SystemD Integration

### Generate SystemD Service Files

```bash
python3 -m src.core.service_manager.systemd_integration_example
```

**Generated Files:**
- `/etc/systemd/system/ultravox-service-manager.service`
- `/etc/systemd/system/ultravox-llm.service`
- `/etc/systemd/system/ultravox-stt.service`
- `/etc/systemd/system/ultravox-tts.service`

### Production Deployment

```bash
# Install services
sudo cp *.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable ultravox-service-manager
sudo systemctl enable ultravox-llm

# Start services
sudo systemctl start ultravox-service-manager
sudo systemctl start ultravox-llm

# View logs
sudo journalctl -u ultravox-service-manager -f
```

### Watchdog Support

The Service Manager supports systemd watchdog for automatic health monitoring:

```python
# systemd_watchdog.py
import systemd.daemon

def notify_watchdog():
    """Notify systemd that service is alive"""
    systemd.daemon.notify("WATCHDOG=1")

# In main loop
while True:
    await asyncio.sleep(15)  # Half of WatchdogSec
    notify_watchdog()
```

---

## 📅 Task Scheduler

### Celery Integration

**File:** `task_integration.py`

```python
from celery import Celery
from celery.schedules import crontab

app = Celery('ultravox', broker='redis://localhost:6379/0')

@app.task
def update_models():
    """Daily model update task"""
    # Download latest models
    pass

@app.task
def cleanup_old_logs():
    """Weekly log cleanup"""
    # Remove logs older than 30 days
    pass

# Schedule tasks
app.conf.beat_schedule = {
    'update-models-daily': {
        'task': 'tasks.update_models',
        'schedule': crontab(hour=0, minute=0),
    },
    'cleanup-logs-weekly': {
        'task': 'tasks.cleanup_old_logs',
        'schedule': crontab(day_of_week='sunday', hour=3, minute=0),
    },
}
```

### Start Celery Worker

```bash
# Start worker
celery -A src.core.service_manager.task_integration worker --loglevel=info

# Start beat scheduler
celery -A src.core.service_manager.task_integration beat --loglevel=info
```

---

## 🔍 Service Discovery & Lifecycle Management

**Automated service discovery, installation, and lifecycle testing for all Ultravox Pipeline services.**

### Overview

The Service Manager includes three powerful modules for managing the complete lifecycle of services:

1. **Service Discovery** (`service_discovery.py`) - Automatically discovers all 22 services
2. **Bulk Installer** (`bulk_installer.py`) - Installs, uninstalls, and validates services
3. **Lifecycle Tester** (`test_service_lifecycle.py`) - 3-cycle automated testing

### Discovered Services (22 total)

| Service | Type | Has install.py? | Port | Description |
|---------|------|----------------|------|-------------|
| **llm** | AI Model | ✅ Yes | 8100 | Speech-to-Speech (Ultravox) |
| **stt** | AI Model | ✅ Yes | 8101 | Speech-to-Text (Whisper) |
| **tts** | AI Model | ✅ Yes | 8102 | Text-to-Speech (Kokoro) |
| **api_gateway** | Infrastructure | ❌ No | 8000 | Main HTTP API Gateway |
| **websocket** | Infrastructure | ❌ No | 8001 | WebSocket Server |
| **orchestrator** | Infrastructure | ❌ No | 8005 | Service Orchestration |
| **file_storage** | Utility | ❌ No | 8003 | File Upload/Download |
| **session** | Utility | ❌ No | 8004 | Session State Management |
| *...and 14 more services* | | | | |

### Quick Commands

**Discover all services:**
```bash
python3 src/core/service_manager/service_discovery.py
```

**Validate a service:**
```bash
python3 src/core/service_manager/bulk_installer.py validate --service api_gateway
```

**Validate ALL services:**
```bash
python3 src/core/service_manager/bulk_installer.py validate-all
```

**Install a service:**
```bash
python3 src/core/service_manager/bulk_installer.py install --service llm
```

**3-cycle lifecycle test (install → validate → uninstall → repeat):**
```bash
# Test single service (2 cycles)
python3 src/core/service_manager/test_service_lifecycle.py --service api_gateway --cycles 2

# Test all infrastructure services (default - fast test)
python3 src/core/service_manager/test_service_lifecycle.py

# Test ALL services including AI models (WARNING: takes hours!)
python3 src/core/service_manager/test_service_lifecycle.py --all
```

### Service Discovery Module

**Features:**
- ✅ Scans `src/services/` and detects all service directories
- ✅ Classifies services by type (AI Model, Infrastructure, Utility, External, Test)
- ✅ Checks installation status (Available, Installed, Healthy, Stopped, etc.)
- ✅ Detects install scripts, service files, and configuration
- ✅ Provides health check status for running services

**Example output:**
```
🔍 Discovering services in /workspace/ultravox-pipeline/src/services...
   ⏹️ 🏗️ api_gateway               - stopped
   📦 🔧 conversation_store        - installed
   ⏹️ 🤖 llm                       - stopped
   ✅ 🤖 stt                       - healthy

✅ Discovered 22 services

📊 Services by Type:
   🤖 AI Model            : 3
   🏗️ Infrastructure      : 6
   🔧 Utility             : 7
```

### Bulk Installer Module

**Operations:**
- `install_service(service_id)` - Install single service
- `uninstall_service(service_id)` - Uninstall single service
- `validate_service(service_id)` - Validate installation
- `install_all_services()` - Bulk install all services
- `validate_all_services()` - Bulk validate all services

**Validation checks:**
- ✅ Service directory exists
- ✅ `service.py` file exists
- ✅ Service module is importable
- ✅ Dependencies are installed (for AI models)
- ✅ Health endpoint responding (if service is running)

### Lifecycle Testing (3-Cycle)

**What it does:**
Tests complete service lifecycle through 3 full cycles:
- **Cycle 1**: Install → Validate → Uninstall
- **Cycle 2**: Reinstall → Validate → Uninstall
- **Cycle 3**: Reinstall → Validate (leave installed)

This ensures services can be reliably installed/uninstalled multiple times without issues.

**Example test:**
```bash
python3 src/core/service_manager/test_service_lifecycle.py --service api_gateway --cycles 3
```

**Output:**
```
====================================================================================================
🔄 LIFECYCLE TEST: api_gateway
   Type: infrastructure
   Cycles: 3
====================================================================================================

----------------------------------------------------------------------------------------------------
🔄 CYCLE 1/3
----------------------------------------------------------------------------------------------------
📦 [1.1] Installing api_gateway...
   ✅ INSTALL: Dependencies installed from requirements.txt (3.2s)

✅ [1.2] Validating api_gateway...
   ✅ VALIDATE: All validation checks passed (4 checks) (0.0s)

🗑️  [1.3] Uninstalling api_gateway...
   ✅ UNINSTALL: Infrastructure service - dependencies kept installed (0.0s)

✅ CYCLE 1 SUMMARY:
   Duration: 3.2s
   Result:    SUCCESS

... (cycles 2 and 3) ...

====================================================================================================
✅ LIFECYCLE TEST SUMMARY: api_gateway
====================================================================================================
Service Type:      infrastructure
Cycles Completed:  3
All Cycles Success: True
Total Duration:    9.5s (0.2 min)
====================================================================================================
```

### Export Test Results

All lifecycle tests can export results to JSON:

```bash
python3 src/core/service_manager/test_service_lifecycle.py --all --export /tmp/results.json
```

**Result format:**
```json
{
  "api_gateway": {
    "service_type": "infrastructure",
    "all_cycles_success": true,
    "total_duration": 9.5,
    "num_cycles": 3,
    "error_messages": [],
    "cycles": [
      {
        "cycle_number": 1,
        "success": true,
        "duration": 3.2,
        "install_status": "success",
        "validate_status": "success",
        "uninstall_status": "success"
      }
    ]
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
tail -f logs/service_manager.log

# Check port availability
lsof -i :8888

# Check process
ps aux | grep service_manager
```

#### 2. Health Checks Failing

```bash
# Test endpoint manually
curl -v http://localhost:8100/health

# Check service logs
tail -f logs/llm.log

# Restart service
curl -X POST http://localhost:8888/services/llm/restart
```

#### 3. Remote Deployment Fails

```bash
# Check SSH connectivity
ssh root@pod-ip

# Check git sync
cd ~/ultravox-pipeline && git status

# Check dependencies
source venv/bin/activate && pip list
```

### Debug Mode

```bash
# Start with debug logging
LOG_LEVEL=DEBUG python3 -m src.core.service_manager.main

# Check all services
curl http://localhost:8888/services | jq

# View metrics
curl http://localhost:8888/metrics | jq
```

---

## 🔄 Refactoring Progress

**Status:** 🚧 Phase 2 in progress (Phases 1 & 3 complete)
**Goal:** Reduce main.py from 5,295 lines to ~500 lines (90% reduction)

### Current Status

| Phase | Status | Description | Progress |
|-------|--------|-------------|----------|
| **Phase 1** | ✅ Complete | Modular directory structure created | 100% |
| **Phase 2** | 🚧 In Progress | Extract modules from main.py | 0% |
| **Phase 3** | ✅ Complete | Pytest test suite (35 tests passing) | 100% |
| **Phase 4** | ⏭️ Pending | Update main.py to use extracted modules | 0% |

### Phase 2: Module Extraction Plan

Seven modules to extract from main.py (~4,500 lines total):

1. **core/health_monitor.py** (~300 lines) - ⏭️ Next
2. **core/service_controller.py** (~600 lines) - Pending
3. **core/orchestrator.py** (~800 lines) - Pending
4. **endpoints/services.py** (~1200 lines) - Pending
5. **endpoints/monitoring.py** (~400 lines) - Pending
6. **endpoints/venv.py** (~500 lines) - Pending
7. **venv/venv_manager.py** (~700 lines) - Pending

### Detailed Documentation

For comprehensive implementation guides:
- **[PHASE_2_GUIDE.md](./PHASE_2_GUIDE.md)** - Technical guide with code examples (535 lines)
- **[NEXT_STEPS.md](./NEXT_STEPS.md)** - Practical roadmap for next steps (329 lines)
- **[REFACTORING.md](./REFACTORING.md)** - Overall refactoring progress tracker

### Test Coverage

**35 tests passing** in 0.25s:
- `test_service_discovery.py` - 27 tests
- `test_bulk_installer.py` - 8 tests

---

## 👨‍💻 Development Guide

### Project Structure

```
src/core/service_manager/
├── main.py                          # Main entry point (5,295 lines → target ~500)
├── core/                            # Core orchestration modules
│   ├── __init__.py                  # ✅ Created
│   ├── health_monitor.py            # ⏭️ Next to create
│   ├── service_controller.py        # ⏭️ To create
│   └── orchestrator.py              # ⏭️ To create
├── discovery/                       # Service discovery
│   ├── __init__.py                  # ✅ Created
│   ├── service_discovery.py         # ✅ Moved
│   ├── bulk_installer.py            # ✅ Moved
│   └── lifecycle_tester.py          # ✅ Moved
├── deployment/                      # Remote deployment
│   ├── __init__.py                  # ✅ Created
│   └── remote_launcher.py           # ✅ Moved
├── monitoring/                      # Metrics & monitoring
│   ├── __init__.py                  # ✅ Created
│   ├── metrics_tracker.py           # ✅ Moved
│   └── activity_tracker.py          # ✅ Moved
├── venv/                            # Virtual environment management
│   ├── __init__.py                  # ✅ Created
│   └── venv_manager.py              # ⏭️ To extract from main.py
├── integration/                     # External integrations
│   ├── __init__.py                  # ✅ Created
│   ├── systemd_manager.py           # ✅ Moved
│   └── task_integration.py          # ✅ Moved
├── registry/                        # Service registry
│   ├── __init__.py                  # ✅ Created
│   └── service_registry.py          # ✅ Moved
├── endpoints/                       # API endpoints
│   ├── __init__.py                  # ✅ Created
│   ├── discovery.py                 # ✅ Existing
│   ├── services.py                  # ⏭️ To extract from main.py
│   ├── monitoring.py                # ⏭️ To extract from main.py
│   ├── venv.py                      # ⏭️ To extract from main.py
│   └── profile_endpoints.py         # ✅ Existing
├── tests/                           # Pytest test suite
│   ├── __init__.py                  # ✅ Created
│   ├── conftest.py                  # ✅ Fixtures (310 lines)
│   ├── test_service_discovery.py    # ✅ 27 tests
│   └── test_bulk_installer.py       # ✅ 8 tests
├── models.py                        # Data models
├── README.md                        # This file
├── SERVICE_INSTALLATION.md          # Installation guide
├── PHASE_2_GUIDE.md                 # ✅ Implementation guide
├── NEXT_STEPS.md                    # ✅ Next steps roadmap
└── REFACTORING.md                   # ✅ Progress tracker
```

### Adding a New Service

1. **Create Service Module**
   ```bash
   mkdir src/services/my_service
   touch src/services/my_service/service.py
   touch src/services/my_service/install.py
   ```

2. **Register in Config**
   ```yaml
   # config/service_execution_config.py
   my_service:
     port: 8110
     execution_mode: "local"
   ```

3. **Implement Health Endpoint**
   ```python
   # service.py
   @app.get("/health")
   async def health():
       return {"status": "healthy", "service": "my_service"}
   ```

4. **Register with Service Manager**
   ```bash
   curl -X POST http://localhost:8888/discovery/register \
     -H "Content-Type: application/json" \
     -d '{"service_id": "my_service", "host": "localhost", "port": 8110}'
   ```

### Running Tests

```bash
# Unit tests
pytest src/core/service_manager/tests/

# Integration tests
pytest src/core/service_manager/tests/integration/

# E2E tests
pytest src/services/machine_manager/tests/test_e2e_full_deployment.py -v -s
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linters: `black`, `isort`, `flake8`
5. Submit pull request

---

## 📚 Related Documentation

- [SERVICE_INSTALLATION.md](./SERVICE_INSTALLATION.md) - Complete installation guide
- [Machine Manager](../../services/machine_manager/README.md) - Pod provisioning
- [Machine Manager Tests](../../services/machine_manager/tests/README.md) - Automated E2E tests
- [RunPod CLI Provider](../../services/machine_manager/providers/runpod_cli.py) - RunPod operations

---

## 📝 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 5000+ | Main Service Manager application |
| `remote_launcher.py` | 485 | Remote service deployment |
| `service_launcher.py` | 300+ | Local service management |
| `service_registry.py` | 350+ | Service discovery |
| `metrics_tracker.py` | 350+ | Performance tracking |
| `activity_tracker.py` | 270+ | Usage tracking |
| `profile_endpoints.py` | 380+ | Profile management |
| `systemd_manager.py` | 370+ | SystemD integration |
| `task_integration.py` | 510+ | Celery tasks |

---

## 🎯 Quick Reference Card

```bash
# Service Manager
python3 -m src.core.service_manager.main

# Health Checks
curl http://localhost:8888/health
curl http://localhost:8888/services
curl http://localhost:8888/metrics/llm

# Service Control
curl -X POST http://localhost:8888/services/llm/start
curl -X POST http://localhost:8888/services/llm/stop
curl -X POST http://localhost:8888/services/llm/restart

# Installation
python3 src/services/llm/install.py
python3 src/services/stt/install.py
python3 src/services/tts/install.py

# Validation
curl http://localhost:8100/health  # LLM
curl http://localhost:8101/health  # STT
curl http://localhost:8102/health  # TTS

# SystemD
sudo systemctl start ultravox-service-manager
sudo journalctl -u ultravox-service-manager -f
```

---

**Version:** 1.0
**Last Updated:** 2025-10-07
**Maintainer:** Ultravox Pipeline Team
