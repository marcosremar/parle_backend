#!/usr/bin/env python3
"""
Script para analisar cobertura de testes por módulo
Identifica módulos com baixa cobertura e gera relatório
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_coverage() -> Dict:
    """Run pytest with coverage and return JSON report"""
    print("🔍 Executando testes com cobertura...")
    
    result = subprocess.run(
        [
            "python", "-m", "pytest",
            "--cov=src",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-q"
        ],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        print("⚠️  Alguns testes falharam, mas continuando análise...")
    
    # Read coverage JSON
    coverage_file = project_root / "coverage.json"
    if not coverage_file.exists():
        print("❌ Arquivo coverage.json não encontrado")
        return {}
    
    with open(coverage_file) as f:
        return json.load(f)


def analyze_module_coverage(coverage_data: Dict, threshold: float = 80.0) -> Tuple[List, List]:
    """Analyze coverage by module and identify low coverage modules"""
    files = coverage_data.get("files", {})
    
    low_coverage = []
    good_coverage = []
    critical_modules = [
        "src/api/",
        "src/core/",
        "src/modules/conversation/orchestrator/",
        "src/modules/speech/",
    ]
    
    for file_path, file_data in files.items():
        # Skip test files
        if "test" in file_path or "tests" in file_path:
            continue
        
        # Skip __pycache__ and __init__
        if "__pycache__" in file_path or "__init__.py" in file_path:
            continue
        
        summary = file_data.get("summary", {})
        percent_covered = summary.get("percent_covered", 0)
        
        is_critical = any(critical in file_path for critical in critical_modules)
        
        module_info = {
            "file": file_path,
            "coverage": percent_covered,
            "lines_covered": summary.get("covered_lines", 0),
            "lines_total": summary.get("num_statements", 0),
            "is_critical": is_critical
        }
        
        if percent_covered < threshold:
            low_coverage.append(module_info)
        else:
            good_coverage.append(module_info)
    
    # Sort by coverage (lowest first)
    low_coverage.sort(key=lambda x: x["coverage"])
    good_coverage.sort(key=lambda x: x["coverage"], reverse=True)
    
    return low_coverage, good_coverage


def generate_report(low_coverage: List, good_coverage: List, threshold: float):
    """Generate markdown report"""
    report = f"""# Relatório de Cobertura de Testes

**Threshold**: {threshold}%
**Data**: {Path(__file__).stat().st_mtime}

## 📊 Resumo

- **Módulos com baixa cobertura**: {len(low_coverage)}
- **Módulos com boa cobertura**: {len(good_coverage)}
- **Total de módulos analisados**: {len(low_coverage) + len(good_coverage)}

## ⚠️ Módulos com Baixa Cobertura (< {threshold}%)

"""
    
    # Critical modules first
    critical_low = [m for m in low_coverage if m["is_critical"]]
    non_critical_low = [m for m in low_coverage if not m["is_critical"]]
    
    if critical_low:
        report += "### 🔴 Módulos Críticos com Baixa Cobertura\n\n"
        for module in critical_low:
            report += f"- **{module['file']}**: {module['coverage']:.1f}% "
            report += f"({module['lines_covered']}/{module['lines_total']} linhas)\n"
        report += "\n"
    
    if non_critical_low:
        report += "### ⚠️ Outros Módulos com Baixa Cobertura\n\n"
        for module in non_critical_low[:20]:  # Top 20
            report += f"- **{module['file']}**: {module['coverage']:.1f}% "
            report += f"({module['lines_covered']}/{module['lines_total']} linhas)\n"
        report += "\n"
    
    report += f"""## ✅ Módulos com Boa Cobertura (≥ {threshold}%)

Total: {len(good_coverage)} módulos

## 🎯 Recomendações

1. **Priorizar módulos críticos** com baixa cobertura
2. **Focar em**:
   - `src/api/` - Endpoints da API
   - `src/core/` - Funcionalidades core
   - `src/modules/conversation/orchestrator/` - Orquestração

3. **Criar testes para**:
"""
    
    for module in critical_low[:5]:  # Top 5 críticos
        report += f"   - {module['file']}\n"
    
    return report


def main():
    """Main function"""
    threshold = 80.0
    
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print(f"⚠️  Threshold inválido, usando padrão: {threshold}%")
    
    print(f"📊 Analisando cobertura de testes (threshold: {threshold}%)...")
    
    # Run coverage
    coverage_data = run_coverage()
    
    if not coverage_data:
        print("❌ Não foi possível obter dados de cobertura")
        return 1
    
    # Analyze
    low_coverage, good_coverage = analyze_module_coverage(coverage_data, threshold)
    
    # Generate report
    report = generate_report(low_coverage, good_coverage, threshold)
    
    # Save report
    report_file = project_root / "docs" / "COVERAGE_REPORT.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    
    print(f"\n✅ Relatório gerado: {report_file}")
    print(f"\n📊 Estatísticas:")
    print(f"   - Módulos com baixa cobertura: {len(low_coverage)}")
    print(f"   - Módulos críticos com baixa cobertura: {sum(1 for m in low_coverage if m['is_critical'])}")
    print(f"   - Módulos com boa cobertura: {len(good_coverage)}")
    
    if low_coverage:
        print(f"\n⚠️  Top 5 módulos críticos com baixa cobertura:")
        critical_low = sorted([m for m in low_coverage if m["is_critical"]], 
                            key=lambda x: x["coverage"])[:5]
        for module in critical_low:
            print(f"   - {module['file']}: {module['coverage']:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
