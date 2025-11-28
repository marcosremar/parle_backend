#!/usr/bin/env python3
"""
Script para identificar código duplicado entre services/ e modules/
Ajuda a identificar o que pode ser removido após validação
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mapeamento de serviços que foram migrados para módulos
MIGRATED_SERVICES = {
    "stt": "src/modules/speech/stt",
    "tts": "src/modules/speech/tts",
    "llm": "src/modules/llm",
    "user": "src/modules/auth",
    "orchestrator": "src/modules/conversation/orchestrator",
    "session": "src/modules/conversation/session",
    "scenarios": "src/modules/conversation/scenarios",
    "conversation_store": "src/modules/storage/conversation_store",
    "file_storage": "src/modules/storage/file_storage",
    "database": "src/modules/storage/database",
    "student_model": "src/modules/tutoring/student_model",
    "diagnostic_module": "src/modules/tutoring/diagnostic",
    "pedagogical_policy": "src/modules/tutoring/pedagogical_policy",
    "learning_path": "src/modules/tutoring/learning_path",
    "rest_polling": "src/modules/realtime/rest_polling",
}

# Arquivos que podem ser removidos (após validação)
POTENTIALLY_REMOVABLE = {
    "stt": ["app_complete.py"],  # Código migrado para modules
    "tts": ["app_complete.py"],  # Código migrado para modules
    "orchestrator": ["orchestrator_engine.py"],  # Migrado para modules/conversation/orchestrator/engine.py
}

# Arquivos que devem ser mantidos (ainda usados)
KEEP_FILES = {
    "orchestrator": ["service.py", "routes.py", "clients/"],  # Ainda usado como serviço HTTP
    "session": ["service.py", "routes.py"],  # Ainda usado como serviço HTTP
    "stt": ["service.py", "routes.py"],  # Ainda usado como serviço HTTP
    "tts": ["service.py", "routes.py"],  # Ainda usado como serviço HTTP
}


def check_service_files(service_name: str) -> Dict:
    """Verifica arquivos de um serviço"""
    service_dir = project_root / "src" / "services" / service_name
    module_dir = project_root / MIGRATED_SERVICES.get(service_name)
    
    result = {
        "service_name": service_name,
        "service_dir_exists": service_dir.exists(),
        "module_dir_exists": module_dir.exists() if module_dir else False,
        "files": [],
        "potentially_removable": [],
        "should_keep": [],
    }
    
    if not service_dir.exists():
        return result
    
    # Listar arquivos Python no serviço
    for py_file in service_dir.rglob("*.py"):
        rel_path = py_file.relative_to(service_dir)
        result["files"].append(str(rel_path))
        
        # Verificar se pode ser removido
        if service_name in POTENTIALLY_REMOVABLE:
            for removable in POTENTIALLY_REMOVABLE[service_name]:
                if removable in str(rel_path):
                    result["potentially_removable"].append(str(rel_path))
        
        # Verificar se deve ser mantido
        if service_name in KEEP_FILES:
            for keep_pattern in KEEP_FILES[service_name]:
                if keep_pattern in str(rel_path):
                    result["should_keep"].append(str(rel_path))
    
    return result


def main():
    """Identifica código duplicado"""
    print("🔍 Identificando código duplicado entre services/ e modules/\n")
    
    results = []
    for service_name in MIGRATED_SERVICES.keys():
        result = check_service_files(service_name)
        results.append(result)
    
    # Relatório
    print("=" * 80)
    print("RELATÓRIO DE CÓDIGO DUPLICADO")
    print("=" * 80)
    
    for result in results:
        if not result["service_dir_exists"]:
            continue
        
        print(f"\n📦 {result['service_name']}")
        print(f"   Service dir: {'✅' if result['service_dir_exists'] else '❌'}")
        print(f"   Module dir: {'✅' if result['module_dir_exists'] else '❌'}")
        
        if result["potentially_removable"]:
            print(f"   ⚠️  Potencialmente removível (após validação):")
            for file in result["potentially_removable"]:
                print(f"      - {file}")
        
        if result["should_keep"]:
            print(f"   ✅ Deve ser mantido (ainda usado):")
            for file in result["should_keep"]:
                print(f"      - {file}")
    
    print("\n" + "=" * 80)
    print("RECOMENDAÇÕES:")
    print("=" * 80)
    print("""
1. ⚠️  NÃO REMOVER arquivos ainda - aguardar validação completa
2. ✅ Arquivos em 'should_keep' são necessários para modo HTTP
3. 🔍 Validar que módulos funcionam antes de remover duplicados
4. 📝 Manter services/ para compatibilidade com modo microservices
    """)


if __name__ == "__main__":
    main()
