#!/usr/bin/env python3
"""
Script para auditar permissões de endpoints
Mapeia todos os endpoints e suas permissões atuais
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EndpointVisitor(ast.NodeVisitor):
    """AST visitor to extract endpoint information"""
    
    def __init__(self):
        self.endpoints = []
        self.current_decorators = []
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        """Visit function definitions"""
        self.current_function = node.name
        self.current_decorators = []
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    self.current_decorators.append(decorator.func.attr)
                elif isinstance(decorator.func, ast.Name):
                    self.current_decorators.append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                self.current_decorators.append(decorator.attr)
            elif isinstance(decorator, ast.Name):
                self.current_decorators.append(decorator.id)
        
        # Check if it's an endpoint (has router decorator)
        is_endpoint = any(d in ["get", "post", "put", "delete", "patch"] for d in self.current_decorators)
        
        if is_endpoint:
            # Extract route path
            route_path = None
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if len(decorator.args) > 0:
                        if isinstance(decorator.args[0], ast.Constant):
                            route_path = decorator.args[0].value
                        elif isinstance(decorator.args[0], ast.Str):
                            route_path = decorator.args[0].s
            
            # Check for authentication
            requires_auth = any(d in ["Depends", "get_current_user", "security"] for d in self.current_decorators)
            
            # Check function parameters for auth dependencies
            for arg in node.args.args:
                if isinstance(arg.annotation, ast.Call):
                    if isinstance(arg.annotation.func, ast.Name):
                        if arg.annotation.func.id == "Depends":
                            requires_auth = True
            
            # Extract HTTP method
            http_method = None
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr in ["get", "post", "put", "delete", "patch"]:
                            http_method = decorator.func.attr.upper()
            
            self.endpoints.append({
                "function": node.name,
                "route": route_path or "/",
                "method": http_method or "GET",
                "requires_auth": requires_auth,
                "decorators": self.current_decorators.copy(),
                "line": node.lineno
            })
        
        self.generic_visit(node)


def audit_endpoints(file_path: Path) -> List[Dict]:
    """Audit endpoints in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
        
        visitor = EndpointVisitor()
        visitor.visit(tree)
        
        # Add file path to each endpoint
        for endpoint in visitor.endpoints:
            endpoint["file"] = str(file_path.relative_to(project_root))
        
        return visitor.endpoints
    except Exception as e:
        return []


def generate_report(all_endpoints: List[Dict]) -> str:
    """Generate permissions matrix report"""
    # Group by authentication requirement
    requires_auth = [e for e in all_endpoints if e["requires_auth"]]
    public = [e for e in all_endpoints if not e["requires_auth"]]
    
    # Group by method
    by_method = {}
    for endpoint in all_endpoints:
        method = endpoint["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(endpoint)
    
    report = f"""# Matriz de Permissões de Endpoints

**Data**: {Path(__file__).stat().st_mtime}

## 📊 Resumo

- **Total de endpoints**: {len(all_endpoints)}
- **Endpoints protegidos (requerem auth)**: {len(requires_auth)}
- **Endpoints públicos**: {len(public)}

## 🔐 Endpoints Protegidos (Requerem Autenticação)

"""
    
    for endpoint in sorted(requires_auth, key=lambda x: x["route"]):
        report += f"- **{endpoint['method']}** `{endpoint['route']}`\n"
        report += f"  - Função: `{endpoint['function']}`\n"
        report += f"  - Arquivo: `{endpoint['file']}`\n\n"
    
    report += "## 🌐 Endpoints Públicos\n\n"
    
    # Critical public endpoints that might need protection
    critical_public = [
        e for e in public 
        if any(keyword in e["route"].lower() for keyword in ["auth", "login", "register", "admin", "user", "conversation"])
    ]
    
    if critical_public:
        report += "### ⚠️ Endpoints Públicos que Podem Precisar de Proteção\n\n"
        for endpoint in sorted(critical_public, key=lambda x: x["route"]):
            report += f"- **{endpoint['method']}** `{endpoint['route']}`\n"
            report += f"  - Função: `{endpoint['function']}`\n"
            report += f"  - Arquivo: `{endpoint['file']}`\n\n"
    
    report += "### Outros Endpoints Públicos\n\n"
    other_public = [e for e in public if e not in critical_public]
    for endpoint in sorted(other_public, key=lambda x: x["route"])[:20]:
        report += f"- **{endpoint['method']}** `{endpoint['route']}`\n"
    
    if len(other_public) > 20:
        report += f"\n... e mais {len(other_public) - 20} endpoints públicos\n"
    
    report += f"""
## 📋 Endpoints por Método HTTP

"""
    
    for method in sorted(by_method.keys()):
        endpoints = by_method[method]
        report += f"### {method} ({len(endpoints)} endpoints)\n\n"
        for endpoint in sorted(endpoints, key=lambda x: x["route"])[:10]:
            auth_status = "🔐 Protegido" if endpoint["requires_auth"] else "🌐 Público"
            report += f"- {auth_status} `{endpoint['route']}`\n"
        if len(endpoints) > 10:
            report += f"... e mais {len(endpoints) - 10} endpoints\n"
        report += "\n"
    
    report += """
## 🎯 Recomendações

1. **Revisar endpoints públicos críticos**:
   - Endpoints de autenticação podem ser públicos (login, register)
   - Endpoints de dados sensíveis devem ser protegidos

2. **Verificar rate limiting**:
   - Endpoints públicos devem ter rate limiting mais restritivo
   - Endpoints protegidos podem ter limites mais altos

3. **Considerar proteção adicional**:
   - CSRF tokens para endpoints que modificam dados
   - Validação de permissões específicas por usuário
   - Audit logs para ações sensíveis
"""
    
    return report


def main():
    """Main function"""
    print("🔍 Auditando permissões de endpoints...")
    
    # Find router files
    router_file = project_root / "src" / "api" / "routers" / "api.py"
    all_endpoints = []
    
    if router_file.exists():
        endpoints = audit_endpoints(router_file)
        all_endpoints.extend(endpoints)
        print(f"📁 Analisado: {router_file}")
        print(f"   Encontrados {len(endpoints)} endpoints")
    
    # Also check main.py for root endpoints
    main_file = project_root / "src" / "api" / "main.py"
    if main_file.exists():
        endpoints = audit_endpoints(main_file)
        all_endpoints.extend(endpoints)
        print(f"📁 Analisado: {main_file}")
        print(f"   Encontrados {len(endpoints)} endpoints")
    
    # Generate report
    report = generate_report(all_endpoints)
    
    # Save report
    report_file = project_root / "docs" / "ENDPOINT_PERMISSIONS.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    
    print(f"\n✅ Relatório gerado: {report_file}")
    
    requires_auth = [e for e in all_endpoints if e["requires_auth"]]
    public = [e for e in all_endpoints if not e["requires_auth"]]
    
    print(f"\n📊 Estatísticas:")
    print(f"   - Total de endpoints: {len(all_endpoints)}")
    print(f"   - Endpoints protegidos: {len(requires_auth)}")
    print(f"   - Endpoints públicos: {len(public)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
