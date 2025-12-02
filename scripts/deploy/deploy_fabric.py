#!/usr/bin/env python3
"""
Deploy ultra-rápido usando Fabric para Python
Executa deploy em segundos em vez de minutos
"""

from fabric import Connection
from pathlib import Path
import time
import sys

# Configurações
VPS_HOST = "54.37.225.188"
VPS_USER = "ubuntu"
SSH_KEY_PATH = Path.home() / ".ssh" / "id_rsa"
CONTAINER_NAME = "parle-backend"
PROJECT_ROOT = Path(__file__).parent

def ultra_fast_deploy():
    """Deploy ultra-rápido usando Fabric"""
    
    print("⚡ Deploy Ultra-Rápido com Fabric")
    print("=" * 50)
    
    # Conexão SSH
    try:
        c = Connection(
            host=VPS_HOST,
            user=VPS_USER,
            key_filename=str(SSH_KEY_PATH),
            connect_timeout=10
        )
        print("✅ Conexão SSH OK")
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False
    
    # 1. Verificar container
    print("\n🔍 Verificando container...")
    result = c.run(f"sudo docker ps | grep {CONTAINER_NAME}", hide=True)
    if not result.ok:
        print("❌ Container não está rodando")
        return False
    print("✅ Container rodando")
    
    # 2. Sincronizar apenas arquivos essenciais
    print("\n📤 Sincronizando arquivos essenciais...")
    
    # Arquivos que precisam ser sincronizados
    essential_files = [
        "src/",
        "requirements.txt", 
        "environment.yml",
        "pyproject.toml",
        "main.sh",
        "docker/",
        "config/"
    ]
    
    start_time = time.time()
    
    # Usar rsync com excludes agressivos via Fabric
    rsync_cmd = (
        f"rsync -avz --delete "
        f"--exclude='.git' --exclude='__pycache__' --exclude='*.pyc' "
        f"--exclude='*.md' --exclude='docs/' --exclude='tests/' "
        f"--exclude='vendor/' --exclude='scripts/' --exclude='*.html' "
        f"--exclude='.pytest_cache' --exclude='.mypy_cache' "
        f"-e 'ssh -i {SSH_KEY_PATH} -o StrictHostKeyChecking=no' "
        f"{PROJECT_ROOT}/ {VPS_USER}@{VPS_HOST}:/tmp/parle-sync/"
    )
    
    print("🚀 Executando rsync otimizado...")
    result = c.local(rsync_cmd, hide=True)
    
    sync_time = time.time() - start_time
    print(f"✅ Sincronizado em {sync_time:.1f}s")
    
    # 3. Copiar para container (sem instalar dependências)
    print("\n📋 Copiando código para container...")
    
    copy_commands = [
        f"sudo docker cp /tmp/parle-sync/src {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/requirements.txt {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/environment.yml {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/pyproject.toml {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/main.sh {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/docker {CONTAINER_NAME}:/workspace/",
        f"sudo docker cp /tmp/parle-sync/config {CONTAINER_NAME}:/workspace/"
    ]
    
    for cmd in copy_commands:
        c.run(cmd, hide=True)
    
    print("✅ Código copiado")
    
    # 4. Reiniciar serviço (se já tiver dependências)
    print("\n🔄 Reiniciando serviço...")
    
    # Matar processo antigo e iniciar novo
    c.run(f"sudo docker exec {CONTAINER_NAME} pkill -f uvicorn || true", hide=True)
    c.run(f"sudo docker exec -d {CONTAINER_NAME} bash -c 'cd /workspace && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000'", hide=True)
    
    print("✅ Serviço reiniciado")
    
    # 5. Verificar status
    print("\n📊 Status final:")
    
    # Verificar container
    result = c.run(f"sudo docker ps | grep {CONTAINER_NAME}", hide=True)
    print("Container:", result.stdout.strip())
    
    # Verificar logs
    result = c.run(f"sudo docker logs --tail 3 {CONTAINER_NAME}", hide=True)
    print("Logs recentes:", result.stdout.strip())
    
    total_time = time.time() - start_time
    print(f"\n⚡ Deploy concluído em {total_time:.1f}s!")
    
    print(f"\n🌐 Acesse:")
    print(f"   API: http://{VPS_HOST}:8000")
    print(f"   Docs: http://{VPS_HOST}:8000/docs")
    
    print(f"\n💡 Se precisar instalar dependências:")
    print(f"   ssh {VPS_USER}@{VPS_HOST} 'sudo docker exec {CONTAINER_NAME} pip install -r /workspace/requirements.txt'")
    
    return True

if __name__ == "__main__":
    try:
        success = ultra_fast_deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Deploy cancelado")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro no deploy: {e}")
        sys.exit(1)
