#!/usr/bin/env python3
"""
Script para verificar se arquivos ainda são usados antes de remover
"""

import sys
from pathlib import Path
import re

project_root = Path(__file__).parent.parent

FILES_TO_CHECK = [
    "src/services/stt/app_complete.py",
    "src/services/tts/app_complete.py",
    "src/services/orchestrator/orchestrator_engine.py",
]

def find_imports(file_path: Path) -> list:
    """Encontra todos os imports de um arquivo"""
    imports = []
    try:
        content = file_path.read_text()
        # Find imports
        patterns = [
            r'from\s+([^\s]+)\s+import',
            r'import\s+([^\s]+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return imports

def check_file_usage(file_path: Path) -> dict:
    """Verifica se um arquivo é usado em outros lugares"""
    relative_path = file_path.relative_to(project_root)
    module_name = str(relative_path).replace('/', '.').replace('.py', '')
    
    # Find all Python files
    python_files = list(project_root.rglob("*.py"))
    
    usages = []
    for py_file in python_files:
        if py_file == file_path:
            continue
        try:
            content = py_file.read_text()
            # Check for imports
            if module_name in content or str(relative_path) in content:
                # More specific check
                if f"from {module_name}" in content or f"import {module_name}" in content:
                    usages.append(str(py_file.relative_to(project_root)))
        except:
            pass
    
    return {
        "file": str(relative_path),
        "module_name": module_name,
        "usages": usages,
        "usage_count": len(usages)
    }

def main():
    """Verifica uso de arquivos"""
    print("🔍 Verificando uso de arquivos antes de remover...\n")
    
    for file_path_str in FILES_TO_CHECK:
        file_path = project_root / file_path_str
        if not file_path.exists():
            print(f"⚠️  {file_path_str} não existe")
            continue
        
        result = check_file_usage(file_path)
        print(f"\n📄 {result['file']}")
        print(f"   Module: {result['module_name']}")
        print(f"   Usos encontrados: {result['usage_count']}")
        
        if result['usages']:
            print("   ⚠️  Ainda usado em:")
            for usage in result['usages'][:10]:  # Mostrar apenas os primeiros 10
                print(f"      - {usage}")
            if len(result['usages']) > 10:
                print(f"      ... e mais {len(result['usages']) - 10} arquivos")
        else:
            print("   ✅ Não usado - pode ser removido com segurança")

if __name__ == "__main__":
    main()
