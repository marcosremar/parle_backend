#!/usr/bin/env python3
"""
Script de auditoria de segurança
Verifica proteções contra SQL Injection, XSS, sanitização de inputs
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SecurityAuditor(ast.NodeVisitor):
    """AST visitor for security audit"""
    
    def __init__(self):
        self.issues = []
        self.current_file = None
    
    def visit_file(self, file_path: Path):
        """Visit a Python file"""
        self.current_file = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append({
                "file": str(file_path),
                "type": "syntax_error",
                "severity": "low",
                "message": f"Syntax error: {e}",
                "line": e.lineno
            })
    
    def visit_Call(self, node):
        """Check for unsafe function calls"""
        # Check for raw SQL queries
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ["execute", "executemany"]:
                # Check if it's SQLAlchemy (safe) or raw SQL (unsafe)
                if isinstance(node.args) and len(node.args) > 0:
                    if isinstance(node.args[0], ast.Str) or isinstance(node.args[0], ast.Constant):
                        self.issues.append({
                            "file": str(self.current_file),
                            "type": "sql_injection_risk",
                            "severity": "high",
                            "message": f"Potential SQL injection: raw SQL string in {func_name}()",
                            "line": node.lineno
                        })
        
        # Check for eval, exec
        if isinstance(node.func, ast.Name):
            if node.func.id in ["eval", "exec", "__import__"]:
                self.issues.append({
                    "file": str(self.current_file),
                    "type": "code_injection_risk",
                    "severity": "high",
                    "message": f"Use of {node.func.id}() is dangerous",
                    "line": node.lineno
                })
        
        self.generic_visit(node)
    
    def visit_Str(self, node):
        """Check for hardcoded secrets in strings"""
        value = node.s
        # Check for potential secrets
        secret_patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', "hardcoded_password"),
            (r'api[_-]?key\s*=\s*["\']([^"\']+)["\']', "hardcoded_api_key"),
            (r'secret\s*=\s*["\']([^"\']+)["\']', "hardcoded_secret"),
        ]
        
        for pattern, issue_type in secret_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.issues.append({
                    "file": str(self.current_file),
                    "type": issue_type,
                    "severity": "critical",
                    "message": f"Potential hardcoded secret found",
                    "line": node.lineno
                })
        
        self.generic_visit(node)


def check_sqlalchemy_usage(file_path: Path) -> List[Dict]:
    """Check if file uses SQLAlchemy ORM (safe)"""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check for SQLAlchemy imports
        has_sqlalchemy = "from sqlalchemy" in content or "import sqlalchemy" in content
        
        # Check for raw SQL
        raw_sql_patterns = [
            r'\.execute\s*\(\s*["\']SELECT',
            r'\.execute\s*\(\s*["\']INSERT',
            r'\.execute\s*\(\s*["\']UPDATE',
            r'\.execute\s*\(\s*["\']DELETE',
        ]
        
        for pattern in raw_sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                if not has_sqlalchemy:
                    issues.append({
                        "file": str(file_path),
                        "type": "sql_injection_risk",
                        "severity": "high",
                        "message": "Raw SQL detected without SQLAlchemy ORM",
                        "line": 0
                    })
    except Exception as e:
        pass
    
    return issues


def check_xss_protection(file_path: Path) -> List[Dict]:
    """Check for XSS protection in responses"""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # FastAPI automatically escapes JSON, but check for HTML responses
        if "HTMLResponse" in content or "Response" in content:
            # Check if content is being returned directly without escaping
            if re.search(r'return\s+.*\.content', content) or re.search(r'Response\(.*content=', content):
                # This might be okay if it's not user input, but flag for review
                issues.append({
                    "file": str(file_path),
                    "type": "xss_review_needed",
                    "severity": "medium",
                    "message": "HTML response detected - verify XSS protection",
                    "line": 0
                })
    except Exception as e:
        pass
    
    return issues


def check_input_sanitization(file_path: Path) -> List[Dict]:
    """Check for input sanitization"""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check for user input handling
        input_patterns = [
            r'request\.(body|json|form|query_params)',
            r'UploadFile',
            r'Form\(',
        ]
        
        has_user_input = any(re.search(pattern, content) for pattern in input_patterns)
        
        if has_user_input:
            # Check for sanitization
            sanitization_patterns = [
                r'sanitize',
                r'validate',
                r'escape',
                r'html\.escape',
            ]
            
            has_sanitization = any(re.search(pattern, content, re.IGNORECASE) for pattern in sanitization_patterns)
            
            if not has_sanitization:
                # FastAPI/Pydantic validate automatically, but flag for review
                issues.append({
                    "file": str(file_path),
                    "type": "input_sanitization_review",
                    "severity": "low",
                    "message": "User input detected - verify sanitization (Pydantic may handle this)",
                    "line": 0
                })
    except Exception as e:
        pass
    
    return issues


def audit_file(file_path: Path) -> List[Dict]:
    """Audit a single file"""
    issues = []
    
    # Skip test files
    if "test" in str(file_path) or "tests" in str(file_path):
        return issues
    
    # AST-based checks
    auditor = SecurityAuditor()
    auditor.visit_file(file_path)
    issues.extend(auditor.issues)
    
    # Additional checks
    issues.extend(check_sqlalchemy_usage(file_path))
    issues.extend(check_xss_protection(file_path))
    issues.extend(check_input_sanitization(file_path))
    
    return issues


def generate_report(all_issues: List[Dict]) -> str:
    """Generate security audit report"""
    # Group by severity
    critical = [i for i in all_issues if i["severity"] == "critical"]
    high = [i for i in all_issues if i["severity"] == "high"]
    medium = [i for i in all_issues if i["severity"] == "medium"]
    low = [i for i in all_issues if i["severity"] == "low"]
    
    report = f"""# Relatório de Auditoria de Segurança

**Data**: {Path(__file__).stat().st_mtime}

## 📊 Resumo

- **Issues Críticas**: {len(critical)}
- **Issues Altas**: {len(high)}
- **Issues Médias**: {len(medium)}
- **Issues Baixas**: {len(low)}
- **Total**: {len(all_issues)}

## 🔴 Issues Críticas

"""
    
    if critical:
        for issue in critical:
            report += f"- **{issue['type']}** em `{issue['file']}` (linha {issue['line']})\n"
            report += f"  - {issue['message']}\n\n"
    else:
        report += "✅ Nenhuma issue crítica encontrada!\n\n"
    
    report += "## ⚠️ Issues Altas\n\n"
    if high:
        for issue in high[:20]:  # Top 20
            report += f"- **{issue['type']}** em `{issue['file']}` (linha {issue['line']})\n"
            report += f"  - {issue['message']}\n\n"
    else:
        report += "✅ Nenhuma issue alta encontrada!\n\n"
    
    report += f"""## 📝 Issues Médias e Baixas

Total: {len(medium) + len(low)} issues (ver relatório completo)

## ✅ Verificações Realizadas

1. **SQL Injection**: Verificado uso de SQLAlchemy ORM
2. **XSS Protection**: Verificado escaping de respostas
3. **Input Sanitization**: Verificado sanitização de inputs
4. **Hardcoded Secrets**: Verificado strings suspeitas
5. **Code Injection**: Verificado uso de eval/exec

## 🎯 Recomendações

1. **Revisar issues críticas e altas imediatamente**
2. **Usar SQLAlchemy ORM** para todas as queries
3. **Validar inputs** com Pydantic
4. **Nunca hardcodar secrets** - usar variáveis de ambiente
5. **Evitar eval/exec** a menos que absolutamente necessário
"""
    
    return report


def main():
    """Main function"""
    print("🔒 Executando auditoria de segurança...")
    
    src_dir = project_root / "src"
    all_issues = []
    
    # Audit all Python files
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        issues = audit_file(py_file)
        all_issues.extend(issues)
    
    # Generate report
    report = generate_report(all_issues)
    
    # Save report
    report_file = project_root / "docs" / "SECURITY_AUDIT_REPORT.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    
    print(f"\n✅ Relatório gerado: {report_file}")
    
    critical = [i for i in all_issues if i["severity"] == "critical"]
    high = [i for i in all_issues if i["severity"] == "high"]
    
    print(f"\n📊 Estatísticas:")
    print(f"   - Issues críticas: {len(critical)}")
    print(f"   - Issues altas: {len(high)}")
    print(f"   - Total de issues: {len(all_issues)}")
    
    if critical:
        print(f"\n🔴 Issues críticas encontradas:")
        for issue in critical[:5]:
            print(f"   - {issue['file']}:{issue['line']} - {issue['type']}")
    
    return 0 if len(critical) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
