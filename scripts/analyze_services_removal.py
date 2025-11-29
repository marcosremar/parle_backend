#!/usr/bin/env python3
"""
Script para analisar se src/services pode ser removido

Verifica:
1. Quais serviços ainda são necessários
2. Quais imports ainda referenciam services
3. Se serviços standalone podem ser migrados
4. Se fallbacks podem ser removidos
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Serviços standalone (só existem em services, não têm módulos)
STANDALONE_SERVICES = {
    "api_gateway",
    "webrtc",
    "webrtc_signaling",
    "websocket",
    "acoustic_features",
    "linguistic_analysis",
    "neural_codec",
}

# Serviços migrados (têm módulos equivalentes)
MIGRATED_SERVICES = {
    "stt", "tts", "llm", "user", "orchestrator", "session",
    "scenarios", "conversation_store", "conversation_history",
    "file_storage", "database", "student_model", "diagnostic_module",
    "pedagogical_policy", "learning_path", "rest_polling"
}


def find_imports_in_file(file_path: Path) -> List[str]:
    """Encontra imports de src.services em um arquivo"""
    imports = []
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.search(r'from\s+src\.services\.|import.*src\.services\.', line):
                # Verificar se é fallback (try/except)
                is_fallback = False
                for j in range(max(0, i-10), min(len(lines), i+5)):
                    if 'try:' in lines[j] or 'except ImportError:' in lines[j] or 'except:' in lines[j]:
                        is_fallback = True
                        break
                
                imports.append({
                    "line": i,
                    "content": line.strip(),
                    "is_fallback": is_fallback,
                    "file": str(file_path.relative_to(project_root))
                })
    except Exception as e:
        pass
    
    return imports


def analyze_services_usage():
    """Analisa uso de src/services"""
    print("=" * 80)
    print("ANÁLISE: Pode remover src/services?")
    print("=" * 80)
    
    services_dir = project_root / "src" / "services"
    
    if not services_dir.exists():
        print("✅ src/services não existe - nada a remover")
        return
    
    # Encontrar todos os imports de src.services
    all_imports = []
    critical_imports = []
    fallback_imports = []
    
    # Buscar em arquivos Python (fora de services/)
    for py_file in project_root.rglob("*.py"):
        if "__pycache__" in str(py_file) or ".pyc" in str(py_file):
            continue
        if "src/services" in str(py_file):
            continue  # Ignorar arquivos dentro de services
        if "scripts" in str(py_file) and "analyze" in str(py_file):
            continue  # Ignorar este script
        if "tests" in str(py_file):
            continue  # Ignorar testes por enquanto
        
        imports = find_imports_in_file(py_file)
        all_imports.extend(imports)
        
        for imp in imports:
            if imp["is_fallback"]:
                fallback_imports.append(imp)
            else:
                critical_imports.append(imp)
    
    # Análise
    print(f"\n📊 Estatísticas:")
    print(f"   Total de imports: {len(all_imports)}")
    print(f"   Imports críticos (sem fallback): {len(critical_imports)}")
    print(f"   Imports com fallback: {len(fallback_imports)}")
    
    # Serviços standalone
    print(f"\n🔍 Serviços Standalone (só em services):")
    standalone_found = []
    for service in STANDALONE_SERVICES:
        service_dir = services_dir / service
        if service_dir.exists():
            standalone_found.append(service)
            print(f"   ✅ {service} - Existe em services/")
    
    # Serviços migrados
    print(f"\n✅ Serviços Migrados (têm módulos):")
    for service in MIGRATED_SERVICES:
        service_dir = services_dir / service
        module_path = project_root / "src" / "modules"
        # Verificar se módulo existe
        module_exists = False
        for mod_dir in module_path.rglob("*"):
            if mod_dir.is_dir() and service.replace("_", "") in str(mod_dir):
                module_exists = True
                break
        
        if service_dir.exists():
            status = "✅" if module_exists else "⚠️"
            print(f"   {status} {service} - Services: {'existe' if service_dir.exists() else 'não existe'}, Module: {'existe' if module_exists else 'não existe'}")
    
    # Imports críticos
    if critical_imports:
        print(f"\n❌ Imports Críticos (SEM fallback) - {len(critical_imports)} encontrados:")
        print("   Estes precisam ser corrigidos antes de remover services:")
        for imp in critical_imports[:10]:  # Mostrar apenas os primeiros 10
            print(f"      - {imp['file']}:{imp['line']} - {imp['content'][:60]}")
        if len(critical_imports) > 10:
            print(f"      ... e mais {len(critical_imports) - 10} imports")
    
    # Conclusão
    print("\n" + "=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)
    
    can_remove = len(critical_imports) == 0
    
    if can_remove and len(standalone_found) == 0:
        print("✅ SIM - Pode remover src/services completamente")
        print("   - Nenhum import crítico encontrado")
        print("   - Nenhum serviço standalone necessário")
    elif can_remove and len(standalone_found) > 0:
        print("⚠️  PARCIALMENTE - Pode remover serviços migrados")
        print(f"   - {len(standalone_found)} serviços standalone devem ser mantidos:")
        for service in standalone_found:
            print(f"      - {service}")
        print("\n   Ação recomendada:")
        print("   1. Manter apenas serviços standalone em services/")
        print("   2. Remover serviços migrados (já em modules/)")
    else:
        print("❌ NÃO - Não pode remover ainda")
        print(f"   - {len(critical_imports)} imports críticos precisam ser corrigidos")
        print("\n   Ação recomendada:")
        print("   1. Corrigir imports críticos (adicionar fallbacks)")
        print("   2. Migrar serviços standalone (se necessário)")
        print("   3. Depois tentar remover novamente")
    
    return {
        "can_remove": can_remove,
        "critical_imports": len(critical_imports),
        "fallback_imports": len(fallback_imports),
        "standalone_services": standalone_found
    }


if __name__ == "__main__":
    result = analyze_services_usage()
    sys.exit(0 if result["can_remove"] else 1)
