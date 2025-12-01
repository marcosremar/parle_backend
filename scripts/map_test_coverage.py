#!/usr/bin/env python3
"""
Script para mapear módulos sem testes unitários
Identifica arquivos Python que não têm testes correspondentes
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_python_files(directory: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Find all Python files in directory"""
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".venv", "venv", "tests", "migrations", "scripts"]
    
    python_files = []
    for py_file in directory.rglob("*.py"):
        # Skip excluded patterns
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue
        
        # Only src directory
        if "src" not in str(py_file):
            continue
        
        python_files.append(py_file)
    
    return python_files


def find_test_files(directory: Path) -> Set[str]:
    """Find all test files and map to source files"""
    test_files = set()
    
    for test_file in directory.rglob("test_*.py"):
        # Extract module name from test file
        # test_api_main.py -> api/main.py
        rel_path = test_file.relative_to(directory)
        module_name = str(rel_path).replace("test_", "").replace(".py", "")
        test_files.add(module_name)
    
    # Also check for files in test directories that test specific modules
    for test_file in directory.rglob("tests/**/*.py"):
        if "test_" in test_file.name or "_test" in test_file.name:
            # Try to infer source module from test file structure
            rel_path = test_file.relative_to(directory)
            parts = rel_path.parts
            if len(parts) > 1:
                # tests/unit/api/test_main.py -> src/api/main.py
                module_name = "/".join(parts[2:]).replace("test_", "").replace("_test", "").replace(".py", "")
                test_files.add(module_name)
    
    return test_files


def map_source_to_tests(source_files: List[Path], test_files: Set[str]) -> Dict[str, Dict]:
    """Map source files to their test status"""
    mapping = {}
    
    for source_file in source_files:
        rel_path = source_file.relative_to(project_root / "src")
        module_path = str(rel_path).replace(".py", "")
        
        # Check if test exists
        has_test = False
        test_candidates = [
            f"test_{module_path}",
            f"{module_path}_test",
            module_path.replace("/", "_"),
        ]
        
        for candidate in test_candidates:
            if candidate in test_files or any(candidate in tf for tf in test_files):
                has_test = True
                break
        
        # Also check if there's a test file in tests/ directory
        test_dir = project_root / "tests"
        possible_test_files = [
            test_dir / "unit" / f"test_{rel_path.name}",
            test_dir / "unit" / rel_path.parent / f"test_{rel_path.name}",
            test_dir / "integration" / f"test_{rel_path.name}",
        ]
        
        for test_file in possible_test_files:
            if test_file.exists():
                has_test = True
                break
        
        mapping[str(source_file)] = {
            "module": module_path,
            "has_test": has_test,
            "is_critical": any(critical in str(source_file) for critical in [
                "src/api/",
                "src/core/",
                "src/modules/conversation/orchestrator/",
            ])
        }
    
    return mapping


def generate_report(mapping: Dict[str, Dict]) -> str:
    """Generate markdown report"""
    no_tests = [m for m in mapping.values() if not m["has_test"]]
    has_tests = [m for m in mapping.values() if m["has_test"]]
    
    critical_no_tests = [m for m in no_tests if m["is_critical"]]
    
    report = f"""# Mapa de Cobertura de Testes

**Data**: {Path(__file__).stat().st_mtime}

## 📊 Resumo

- **Módulos sem testes**: {len(no_tests)}
- **Módulos com testes**: {len(has_tests)}
- **Módulos críticos sem testes**: {len(critical_no_tests)}
- **Total de módulos**: {len(mapping)}

## 🔴 Módulos Críticos sem Testes

"""
    
    if critical_no_tests:
        for module in sorted(critical_no_tests, key=lambda x: x["module"]):
            report += f"- `{module['module']}`\n"
    else:
        report += "✅ Todos os módulos críticos têm testes!\n"
    
    report += f"\n## ⚠️ Outros Módulos sem Testes\n\n"
    
    non_critical_no_tests = [m for m in no_tests if not m["is_critical"]]
    for module in sorted(non_critical_no_tests, key=lambda x: x["module"])[:30]:  # Top 30
        report += f"- `{module['module']}`\n"
    
    if len(non_critical_no_tests) > 30:
        report += f"\n... e mais {len(non_critical_no_tests) - 30} módulos\n"
    
    report += f"""
## ✅ Módulos com Testes

Total: {len(has_tests)} módulos

## 🎯 Recomendações

1. **Priorizar criação de testes para módulos críticos sem testes**
2. **Focar em**:
   - Módulos em `src/api/`
   - Módulos em `src/core/`
   - Módulos em `src/modules/conversation/orchestrator/`

3. **Usar estrutura de testes existente** em `tests/unit/` e `tests/integration/`
"""
    
    return report


def main():
    """Main function"""
    print("🔍 Mapeando módulos sem testes...")
    
    # Find source files
    src_dir = project_root / "src"
    source_files = find_python_files(src_dir)
    print(f"📁 Encontrados {len(source_files)} arquivos Python em src/")
    
    # Find test files
    test_dir = project_root / "tests"
    test_files = find_test_files(test_dir)
    print(f"🧪 Encontrados {len(test_files)} arquivos de teste")
    
    # Map
    mapping = map_source_to_tests(source_files, test_files)
    
    # Generate report
    report = generate_report(mapping)
    
    # Save report
    report_file = project_root / "docs" / "TEST_COVERAGE_MAP.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    
    print(f"\n✅ Relatório gerado: {report_file}")
    
    no_tests = [m for m in mapping.values() if not m["has_test"]]
    critical_no_tests = [m for m in no_tests if m["is_critical"]]
    
    print(f"\n📊 Estatísticas:")
    print(f"   - Módulos sem testes: {len(no_tests)}")
    print(f"   - Módulos críticos sem testes: {len(critical_no_tests)}")
    
    if critical_no_tests:
        print(f"\n⚠️  Módulos críticos sem testes:")
        for module in sorted(critical_no_tests, key=lambda x: x["module"])[:10]:
            print(f"   - {module['module']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
