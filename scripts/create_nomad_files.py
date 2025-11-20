#!/usr/bin/env python3
"""
Script para criar arquivos .nomad para serviços migrados
"""
from pathlib import Path

PARLE_DIR = Path("/Users/marcos/Documents/projects/backend/parle_backend")

# Portas e configurações para cada serviço
SERVICE_CONFIGS = {
    "orchestrator": {"port": 8500, "memory": 512},
    "session": {"port": 8600, "memory": 256},
    "rest_polling": {"port": 8700, "memory": 256},
    "conversation_store": {"port": 8800, "memory": 512},
    "tts": {"port": 8900, "memory": 512},
    "stt": {"port": 9000, "memory": 512},
    "diarization": {"port": 9100, "memory": 512},
    "vad_service": {"port": 9200, "memory": 256},
    "sentiment_analysis": {"port": 9300, "memory": 256},
    "broadcaster": {"port": 9400, "memory": 256},
    "communication_strategy": {"port": 9500, "memory": 512},
    "group_orchestrator": {"port": 9600, "memory": 512},
    "group_session": {"port": 9700, "memory": 256},
    "metrics_testing": {"port": 9800, "memory": 512},
    "runpod_llm": {"port": 9900, "memory": 1024},
    "streaming_orchestrator": {"port": 10000, "memory": 512},
    "webrtc": {"port": 10100, "memory": 256},
    "webrtc_signaling": {"port": 10200, "memory": 256},
    "discord_voice": {"port": 10300, "memory": 256},
    "viber_gateway": {"port": 10400, "memory": 256},
    "whatsapp_gateway": {"port": 10500, "memory": 256},
}

def create_nomad_file(service_name: str, config: dict):
    """Cria arquivo .nomad para um serviço"""
    nomad_dir = PARLE_DIR / "deploy" / "nomad"
    nomad_file = nomad_dir / f"{service_name.replace('_', '-')}.nomad"
    
    if nomad_file.exists():
        print(f"  ⚠️  {nomad_file.name} já existe - pulando")
        return False
    
    port = config["port"]
    memory = config["memory"]
    
    # Nome do job (kebab-case)
    job_name = service_name.replace("_", "-")
    
    template = f'''job "{job_name}" {{
  datacenters = ["dc1"]
  type        = "service"

  group "{job_name}-group" {{
    count = 1

    task "{job_name}" {{
      driver = "raw_exec"

      config {{
        command = "venv/bin/python"
        args    = ["src/services/{service_name}/app_complete.py"]
      }}

      env {{
        PYTHONPATH = "src"
        PORT       = "{port}"
      }}

      resources {{
        cpu    = 500
        memory = {memory}
      }}

      service {{
        name = "{job_name}"
        port = "http"

        check {{
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "2s"
        }}
      }}
    }}

    network {{
      port "http" {{
        to = {port}
      }}
    }}
  }}
}}
'''
    
    nomad_file.write_text(template, encoding='utf-8')
    return True

def main():
    """Cria arquivos .nomad para todos os serviços migrados"""
    print(f"🚀 Criando arquivos .nomad para {len(SERVICE_CONFIGS)} serviços")
    
    created = 0
    skipped = 0
    
    for service, config in SERVICE_CONFIGS.items():
        print(f"\n📦 {service}")
        if create_nomad_file(service, config):
            created += 1
            print(f"  ✅ {service.replace('_', '-')}.nomad criado")
        else:
            skipped += 1
    
    print(f"\n📊 Resumo:")
    print(f"   ✅ Criados: {created}")
    print(f"   ⏭️  Pulados: {skipped}")

if __name__ == "__main__":
    main()

