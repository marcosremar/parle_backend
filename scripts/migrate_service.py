#!/usr/bin/env python3
"""
Script para migrar serviços do ultravox-pipeline para parle_backend
"""
import os
import shutil
import re
from pathlib import Path

ULTRAVOX_DIR = Path("/Users/marcos/Documents/projects/backend/ultravox-pipeline")
PARLE_DIR = Path("/Users/marcos/Documents/projects/backend/parle_backend")

def fix_imports(file_path: Path):
    """Ajusta imports de core para src.core"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Substituir imports
        replacements = [
            (r'from core\.', 'from src.core.'),
            (r'import core\.', 'import src.core.'),
            (r'from \.core\.', 'from src.core.'),
            (r'^import core$', 'import src.core'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        file_path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"  ⚠️  Erro ao ajustar imports em {file_path}: {e}")
        return False

def migrate_service(service_name: str, dry_run: bool = False):
    """Migra um serviço do ultravox-pipeline para parle_backend"""
    print(f"\n📦 Migrando serviço: {service_name}")
    
    source_dir = ULTRAVOX_DIR / "src" / "services" / service_name
    target_dir = PARLE_DIR / "src" / "services" / service_name
    
    if not source_dir.exists():
        print(f"  ❌ Serviço não encontrado em: {source_dir}")
        return False
    
    if target_dir.exists():
        print(f"  ⚠️  Serviço já existe em: {target_dir}")
        response = input("  Deseja sobrescrever? (s/N): ")
        if response.lower() != 's':
            print("  ⏭️  Pulando...")
            return False
    
    if dry_run:
        print(f"  [DRY RUN] Copiaria de {source_dir} para {target_dir}")
        return True
    
    try:
        # Copiar diretório
        print(f"  📁 Copiando arquivos...")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        
        # Ajustar imports em arquivos Python
        print(f"  🔧 Ajustando imports...")
        py_files = list(target_dir.rglob("*.py"))
        fixed = 0
        for py_file in py_files:
            if fix_imports(py_file):
                fixed += 1
        
        print(f"  ✅ {fixed}/{len(py_files)} arquivos ajustados")
        print(f"  ✅ Serviço migrado com sucesso!")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao migrar: {e}")
        return False

def main():
    """Migra todos os serviços não migrados"""
    services_to_migrate = [
        "orchestrator",
        "session",
        "rest_polling",
        "conversation_store",
        "tts",
        "stt",
        "diarization",
        "vad_service",
        "sentiment_analysis",
        "broadcaster",
        "communication_strategy",
        "group_orchestrator",
        "group_session",
        "metrics_testing",
        "runpod_llm",
        "streaming_orchestrator",
        "webrtc",
        "webrtc_signaling",
        "discord_voice",
        "viber_gateway",
        "whatsapp_gateway",
    ]
    
    print(f"🚀 Migração de Serviços")
    print(f"   De: {ULTRAVOX_DIR}")
    print(f"   Para: {PARLE_DIR}")
    print(f"   Total: {len(services_to_migrate)} serviços")
    
    # Verificar se é dry run
    dry_run = '--dry-run' in os.sys.argv
    
    if dry_run:
        print("\n⚠️  MODO DRY RUN - Nenhuma alteração será feita")
    
    migrated = 0
    failed = 0
    
    for service in services_to_migrate:
        if migrate_service(service, dry_run=dry_run):
            migrated += 1
        else:
            failed += 1
    
    print(f"\n📊 Resumo:")
    print(f"   ✅ Migrados: {migrated}")
    print(f"   ❌ Falharam: {failed}")
    print(f"   📋 Total: {len(services_to_migrate)}")

if __name__ == "__main__":
    main()

