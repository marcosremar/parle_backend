#!/usr/bin/env python3
"""
Script para verificar arquivos duplicados que podem ser removidos com segurança

Verifica:
1. Se arquivo em services/ tem equivalente em modules/
2. Se arquivo ainda é usado por algum import
3. Se arquivo é necessário para modo HTTP/microservices
"""

import sys
import re
from pathlib import Path
from typing import Set, List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Arquivos que podem ser removidos (após validação)
POTENTIALLY_REMOVABLE = {
    "stt/app_complete.py": {
        "reason": "Código migrado para modules/speech/stt/",
        "module_path": "src/modules/speech/stt/module.py",
        "keep_if_http": True  # Manter se usado como serviço HTTP
    },
    "tts/app_complete.py": {
        "reason": "Código migrado para modules/speech/tts/",
        "module_path": "src/modules/speech/tts/module.py",
        "keep_if_http": True
    },
    "orchestrator/orchestrator_engine.py": {
        "reason": "Código migrado para modules/conversation/orchestrator/engine.py",
        "module_path": "src/modules/conversation/orchestrator/engine.py",
        "keep_if_http": True  # service.py ainda pode usar
    },
}

# Arquivos que devem ser mantidos (ainda usados)
KEEP_FILES = {
    "orchestrator/service.py": "Usado como serviço HTTP standalone",
    "orchestrator/routes.py": "Usado como serviço HTTP standalone",
    "session/service.py": "Usado como serviço HTTP standalone",
    "session/routes.py": "Usado como serviço HTTP standalone",
    "stt/service.py": "Usado como serviço HTTP standalone",
    "tts/service.py": "Usado como serviço HTTP standalone",
}


def find_imports(file_path: Path) -> Set[str]:
    """Encontra todos os imports de um arquivo"""
    imports = set()
    
    try:
        content = file_path.read_text()
        # Padrão para imports
        patterns = [
            r'from\s+([\w.]+)\s+import',
            r'import\s+([\w.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                imports.add(match)
    except Exception:
        pass
    
    return imports


def check_file_usage(file_path: Path) -> Dict:
    """Verifica se arquivo é usado em algum lugar"""
    rel_path = file_path.relative_to(project_root)
    file_name = file_path.name
    module_name = rel_path.parent.name
    
    # Verificar imports em todo o projeto
    usage_count = 0
    usage_locations = []
    
    # Padrões de import possíveis
    import_patterns = [
        f"from src.services.{module_name}",
        f"from src.services.{module_name}.{file_path.stem}",
        f"import.*{module_name}",
    ]
    
    # Buscar em arquivos Python
    for py_file in project_root.rglob("*.py"):
        if "__pycache__" in str(py_file) or py_file == file_path:
            continue
        
        try:
            content = py_file.read_text()
            for pattern in import_patterns:
                if re.search(pattern, content):
                    usage_count += 1
                    usage_locations.append(str(py_file.relative_to(project_root)))
                    break
        except Exception:
            pass
    
    return {
        "used": usage_count > 0,
        "usage_count": usage_count,
        "locations": usage_locations[:5]  # Limitar a 5
    }


def check_duplicate_files():
    """Verifica arquivos duplicados"""
    print("=" * 80)
    print("VERIFICAÇÃO DE ARQUIVOS DUPLICADOS")
    print("=" * 80)
    
    removable = []
    keep = []
    
    services_dir = project_root / "src" / "services"
    
    for file_pattern, info in POTENTIALLY_REMOVABLE.items():
        file_path = services_dir / file_pattern
        
        if not file_path.exists():
            continue
        
        print(f"\n📄 {file_pattern}")
        print(f"   Razão: {info['reason']}")
        
        # Verificar se módulo equivalente existe
        module_path = project_root / info['module_path']
        module_exists = module_path.exists()
        print(f"   Módulo equivalente: {'✅ Existe' if module_exists else '❌ Não existe'}")
        
        # Verificar uso
        usage = check_file_usage(file_path)
        print(f"   Uso: {usage['usage_count']} referências")
        
        if usage['usage_count'] > 0:
            print(f"   Localizações:")
            for loc in usage['locations']:
                print(f"      - {loc}")
        
        # Decisão
        if module_exists and usage['usage_count'] == 0:
            removable.append({
                "file": file_pattern,
                "path": file_path,
                "reason": info['reason'],
                "module": info['module_path']
            })
            print(f"   Status: ✅ PODE SER REMOVIDO")
        elif info['keep_if_http']:
            keep.append({
                "file": file_pattern,
                "reason": "Mantido para compatibilidade HTTP/microservices"
            })
            print(f"   Status: ⚠️  MANTER (compatibilidade HTTP)")
        else:
            keep.append({
                "file": file_pattern,
                "reason": "Ainda em uso"
            })
            print(f"   Status: ⚠️  MANTER (ainda usado)")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    
    if removable:
        print(f"\n✅ Arquivos que PODEM ser removidos ({len(removable)}):")
        for item in removable:
            print(f"   - {item['file']}")
            print(f"     Razão: {item['reason']}")
    
    if keep:
        print(f"\n⚠️  Arquivos que DEVEM ser mantidos ({len(keep)}):")
        for item in keep:
            print(f"   - {item['file']}")
            print(f"     Razão: {item['reason']}")
    
    return removable, keep


def main():
    """Executa verificação"""
    removable, keep = check_duplicate_files()
    
    print("\n" + "=" * 80)
    print("RECOMENDAÇÕES")
    print("=" * 80)
    print("""
1. ⚠️  NÃO REMOVER arquivos ainda sem validação completa
2. ✅ Executar testes antes de remover qualquer arquivo
3. 🔍 Validar que módulos funcionam corretamente
4. 📝 Manter backup antes de remover
    """)
    
    if removable:
        print("\n💡 Para remover arquivos seguros, execute:")
        print("   python3 scripts/remove_duplicate_files.py --dry-run")
        print("   python3 scripts/remove_duplicate_files.py  # Remove de fato")


if __name__ == "__main__":
    main()
