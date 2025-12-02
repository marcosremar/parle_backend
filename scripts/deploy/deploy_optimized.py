#!/usr/bin/env python3
"""
Deploy otimizado com Fabric + rsync para Parle Backend
Combina o melhor de Fabric com excludes agressivos do config.yaml
"""

from fabric import Connection
from pathlib import Path
import time
import sys
import yaml

# Configurações
VPS_HOST = "54.37.225.188"
VPS_USER = "ubuntu"
SSH_KEY_PATH = Path.home() / ".ssh" / "id_rsa"
CONTAINER_NAME = "parle-backend"
PROJECT_ROOT = Path(__file__).parent

def load_config():
    """Carrega excludes do config.yaml se existir"""
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('sync', {}).get('excludes', [])
    return []

def build_rsync_excludes():
    """Constrói lista de excludes para rsync"""
    # Carrega excludes do config.yaml
    yaml_excludes = load_config()
    
    # Excludes padrão para rsync (formato corrigido)
    default_excludes = [
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.egg-info",
        "node_modules",
        ".venv",
        "venv",
        "temp",
        "*.log",
        ".env",
        ".DS_Store",
        "*.swp",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".coverage",
        "htmlcov",
        "logs",
        "references",
        "myproject",
        "debug_*.json",
        "file*.txt",
        "test_*.txt",
        "hello.py",
        "blaxel-sandbox.yaml",
        ".docs/similar-projects",
        ".agent/chromadb/",
        # Excludes agressivos do parle_backend
        "docs/",
        "data/",
        "*.md",
        "BUILD_*.md",
        "CI_CD_*.md",
        "COST_*.md",
        "DEPLOY_*.md",
        "FAST_BUILD_*.md",
        "GCP_*.md",
        "README_*.md",
        "REGISTRY_*.md",
        "STARTUP_*.md",
        "TEST_*.md",
        "VPS_*.md",
        "VPS_VS_*.md",
        "htmlcov/",
        "tmp/",
        "vm/",
        "tests/output/",
        "*.html",
        "vendor/",
        "scripts/",
        "tests/"
    ]
    
    # Combinar e remover duplicados
    all_excludes = list(set(default_excludes + yaml_excludes))
    return all_excludes

def check_container_health(c):
    """Verifica saúde do container e serviço"""
    try:
        # Verificar se container está rodando
        result = c.run(f"sudo docker ps | grep {CONTAINER_NAME}", hide=True)
        if not result.ok:
            return False, "Container não está rodando"
        
        # Verificar se API responde
        try:
            health_check = c.run(f"curl -s -f http://localhost:8000/health || echo 'FAILED'", hide=True)
            if "FAILED" in health_check.stdout:
                return False, "API não responde"
        except:
            return False, "Health check falhou"
            
        return True, "Container saudável"
    except Exception as e:
        return False, f"Erro: {str(e)}"

def ultra_fast_deploy():
    """Deploy ultra-rápido otimizado"""
    
    print("⚡ Deploy Ultra-Rápido com Fabric + rsync")
    print("=" * 60)
    
    start_total = time.time()
    
    # 1. Conexão SSH
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
    
    # 2. Verificar container
    print("\n🔍 Verificando container...")
    healthy, status_msg = check_container_health(c)
    if not healthy:
        print(f"❌ {status_msg}")
        print("💡 Execute primeiro: ./setup_vps.sh")
        return False
    print(f"✅ {status_msg}")
    
    # 3. Sincronização otimizada
    print("\n📤 Sincronizando arquivos essenciais...")
    
    # Construir excludes
    excludes = build_rsync_excludes()
    exclude_args = " ".join([f"--exclude='{exc}'" for exc in excludes])
    
    # Comando rsync otimizado
    rsync_cmd = (
        f"rsync -avz --delete --progress "
        f"{exclude_args} "
        f"-e 'ssh -i {SSH_KEY_PATH} -o StrictHostKeyChecking=no -o Compression=no' "
        f"{PROJECT_ROOT}/ {VPS_USER}@{VPS_HOST}:/tmp/parle-sync/"
    )
    
    print(f"🚀 rsync com {len(excludes)} excludes...")
    start_sync = time.time()
    
    try:
        result = c.local(rsync_cmd, hide=True)
        sync_time = time.time() - start_sync
        print(f"✅ Sincronizado em {sync_time:.1f}s")
    except Exception as e:
        print(f"❌ Erro no rsync: {e}")
        return False
    
    # 4. Copiar para container
    print("\n📋 Copiando código para container...")
    start_copy = time.time()
    
    copy_commands = [
        (f"sudo docker cp /tmp/parle-sync/src {CONTAINER_NAME}:/workspace/", "src/"),
        (f"sudo docker cp /tmp/parle-sync/requirements.txt {CONTAINER_NAME}:/workspace/", "requirements.txt"),
        (f"sudo docker cp /tmp/parle-sync/environment.yml {CONTAINER_NAME}:/workspace/", "environment.yml"),
        (f"sudo docker cp /tmp/parle-sync/pyproject.toml {CONTAINER_NAME}:/workspace/", "pyproject.toml"),
        (f"sudo docker cp /tmp/parle-sync/main.sh {CONTAINER_NAME}:/workspace/", "main.sh"),
        (f"sudo docker cp /tmp/parle-sync/docker {CONTAINER_NAME}:/workspace/", "docker/"),
        (f"sudo docker cp /tmp/parle-sync/config {CONTAINER_NAME}:/workspace/", "config/")
    ]
    
    for cmd, desc in copy_commands:
        try:
            c.run(cmd, hide=True, timeout=30)
            print(f"  ✅ {desc}")
        except Exception as e:
            print(f"  ❌ {desc}: {e}")
            return False
    
    copy_time = time.time() - start_copy
    print(f"✅ Código copiado em {copy_time:.1f}s")
    
    # 5. Reiniciar serviço (smooth restart)
    print("\n🔄 Reiniciando serviço...")
    start_restart = time.time()
    
    try:
        # Graceful restart
        c.run(f"sudo docker exec {CONTAINER_NAME} bash -c 'cd /workspace && python -c \"import uvicorn; uvicorn.run(\\\"src.api.main:app\\\", host=\\\"0.0.0.0\\\", port=8000, reload=False)\"' &", hide=True)
        
        # Esperar um pouco
        time.sleep(2)
        
        # Verificar se está rodando
        result = c.run(f"sudo docker ps | grep {CONTAINER_NAME}", hide=True)
        if result.ok:
            print("✅ Serviço reiniciado")
        else:
            print("⚠️  Container pode ter reiniciado")
            
    except Exception as e:
        print(f"⚠️  Restart com warnings: {e}")
    
    restart_time = time.time() - start_restart
    
    # 6. Verificação final
    print("\n📊 Status final:")
    
    # Container status
    result = c.run(f"sudo docker ps | grep {CONTAINER_NAME}", hide=True)
    print("📦 Container:", result.stdout.strip() if result.stdout else "Não encontrado")
    
    # Health check
    try:
        health = c.run(f"curl -s http://localhost:8000/health || echo 'API não responde'", hide=True)
        print("🏥 Health:", health.stdout.strip())
    except:
        print("🏥 Health: Falha no check")
    
    # Logs recentes
    try:
        logs = c.run(f"sudo docker logs --tail 3 {CONTAINER_NAME}", hide=True)
        print("📋 Logs recentes:", logs.stdout.strip() if logs.stdout else "Sem logs")
    except:
        print("📋 Logs recentes: Erro ao obter")
    
    total_time = time.time() - start_total
    
    print("\n" + "=" * 60)
    print(f"⚡ Deploy concluído em {total_time:.1f}s!")
    print(f"   - Sync: {sync_time:.1f}s")
    print(f"   - Copy: {copy_time:.1f}s") 
    print(f"   - Restart: {restart_time:.1f}s")
    print("=" * 60)
    
    print(f"\n🌐 Acesse:")
    print(f"   API: http://{VPS_HOST}:8000")
    print(f"   Docs: http://{VPS_HOST}:8000/docs")
    
    print(f"\n💡 Comandos úteis:")
    print(f"   Logs: ssh {VPS_USER}@{VPS_HOST} 'sudo docker logs -f {CONTAINER_NAME}'")
    print(f"   Shell: ssh {VPS_USER}@{VPS_HOST} 'sudo docker exec -it {CONTAINER_NAME} bash'")
    
    return True

def install_dependencies_if_needed():
    """Instala dependências apenas se necessário"""
    print("\n📦 Verificando dependências...")
    
    try:
        c = Connection(
            host=VPS_HOST,
            user=VPS_USER,
            key_filename=str(SSH_KEY_PATH),
            connect_timeout=10
        )
        
        # Verificar se requirements.txt mudou
        result = c.run(f"sudo docker exec {CONTAINER_NAME} bash -c 'cd /workspace && md5sum requirements.txt 2>/dev/null || echo \"missing\"'", hide=True)
        
        if "missing" in result.stdout:
            print("📦 Instalando dependências...")
            c.run(f"sudo docker exec {CONTAINER_NAME} pip install -r /workspace/requirements.txt", hide=True, timeout=300)
            print("✅ Dependências instaladas")
        else:
            print("✅ Dependências OK")
            
    except Exception as e:
        print(f"⚠️  Erro ao verificar dependências: {e}")

if __name__ == "__main__":
    print("🚀 Deploy Otimizado - Parle Backend")
    print("🔧 Usa Fabric + rsync + excludes agressivos")
    print("")
    
    try:
        success = ultra_fast_deploy()
        if success:
            print("\n🎉 Deploy bem-sucedido!")
        else:
            print("\n❌ Deploy falhou")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Deploy cancelado")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro no deploy: {e}")
        sys.exit(1)
