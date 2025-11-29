#!/usr/bin/env python3
"""
Script para verificar o status de migração dos serviços de services.backup para modules
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mapeamento de serviços para módulos
SERVICE_TO_MODULE_MAP = {
    "acoustic_features": None,  # Serviço standalone (opcional, usado para análise acústica)
    "api_gateway": None,  # Serviço standalone (API Gateway)
    "conversation_history": "storage/conversation_history",
    "conversation_store": "storage/conversation_store",
    "database": "storage/database",
    "diagnostic_module": "tutoring/diagnostic",
    "file_storage": "storage/file_storage",
    "learning_path": "tutoring/learning_path",
    "linguistic_analysis": None,  # Serviço standalone (opcional, usado para análise linguística)
    "llm": "llm",
    "neural_codec": "speech/neural_codec",
    "orchestrator": "conversation/orchestrator",
    "pedagogical_policy": "tutoring/pedagogical_policy",
    "rest_polling": "realtime/rest_polling",
    "scenarios": "conversation/scenarios",
    "session": "conversation/session",
    "stt": "speech/stt",
    "student_model": "tutoring/student_model",
    "tts": "speech/tts",
    "user": "auth",
    "webrtc": None,  # Serviço standalone (WebRTC)
    "webrtc_signaling": None,  # Serviço standalone (WebRTC signaling)
    "websocket": None,  # Serviço standalone (WebSocket)
}

# Serviços standalone (não precisam migração para módulos)
STANDALONE_SERVICES = {
    "api_gateway",
    "webrtc",
    "webrtc_signaling",
    "websocket",
    "acoustic_features",  # Opcional - análise acústica
    "linguistic_analysis",  # Opcional - análise linguística
}

def check_migration_status():
    """Verifica o status de migração de cada serviço"""
    services_backup = project_root / "src" / "services.backup"
    modules = project_root / "src" / "modules"
    
    results = {
        "migrated": [],
        "not_migrated": [],
        "standalone": [],
        "missing": []
    }
    
    for service_name, module_path in SERVICE_TO_MODULE_MAP.items():
        service_backup_path = services_backup / service_name
        module_path_full = modules / module_path if module_path else None
        
        # Verificar se existe em services.backup
        if not service_backup_path.exists():
            results["missing"].append(service_name)
            continue
        
        # Se não tem mapeamento, é standalone ou não migrado
        if module_path is None:
            # Verificar se é um serviço standalone (não precisa migração)
            if service_name in STANDALONE_SERVICES:
                results["standalone"].append(service_name)
            else:
                results["not_migrated"].append(service_name)
        else:
            # Verificar se o módulo existe
            if module_path_full and module_path_full.exists():
                results["migrated"].append((service_name, module_path))
            else:
                results["not_migrated"].append(service_name)
    
    return results

def main():
    """Gera relatório de status de migração"""
    print("=" * 80)
    print("RELATÓRIO DE STATUS DE MIGRAÇÃO")
    print("=" * 80)
    print()
    
    results = check_migration_status()
    
    print(f"✅ SERVIÇOS MIGRADOS ({len(results['migrated'])}):")
    print("-" * 80)
    for service_name, module_path in results["migrated"]:
        print(f"  ✓ {service_name:30} → modules/{module_path}")
    print()
    
    if results["standalone"]:
        print(f"🔧 SERVIÇOS STANDALONE (não precisam migração) ({len(results['standalone'])}):")
        print("-" * 80)
        for service_name in results["standalone"]:
            print(f"  • {service_name}")
        print()
    
    if results["not_migrated"]:
        print(f"⚠️  SERVIÇOS NÃO MIGRADOS ({len(results['not_migrated'])}):")
        print("-" * 80)
        for service_name in results["not_migrated"]:
            print(f"  ✗ {service_name}")
        print()
    
    if results["missing"]:
        print(f"❌ SERVIÇOS NÃO ENCONTRADOS EM services.backup ({len(results['missing'])}):")
        print("-" * 80)
        for service_name in results["missing"]:
            print(f"  ? {service_name}")
        print()
    
    print("=" * 80)
    print(f"RESUMO:")
    print(f"  Total de serviços em services.backup: {len(SERVICE_TO_MODULE_MAP)}")
    print(f"  ✅ Migrados: {len(results['migrated'])}")
    print(f"  🔧 Standalone: {len(results['standalone'])}")
    print(f"  ⚠️  Não migrados: {len(results['not_migrated'])}")
    print(f"  ❌ Não encontrados: {len(results['missing'])}")
    print("=" * 80)
    
    # Verificar se todos os migráveis foram migrados
    migrable_count = len(results["migrated"]) + len(results["not_migrated"])
    if len(results["not_migrated"]) == 0:
        print("\n🎉 TODOS OS SERVIÇOS MIGRÁVEIS FORAM MIGRADOS!")
    else:
        print(f"\n⚠️  Ainda há {len(results['not_migrated'])} serviços não migrados")

if __name__ == "__main__":
    main()
