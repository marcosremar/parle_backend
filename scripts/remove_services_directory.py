#!/usr/bin/env python3
"""
Script para remover diretório src/services após verificação

IMPORTANTE: Execute validação completa antes de usar este script
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_modules_work():
    """Verifica se todos os módulos funcionam sem services"""
    print("=" * 80)
    print("VERIFICAÇÃO: Módulos funcionam sem services?")
    print("=" * 80)
    
    from src.modules import module_factory
    
    modules = [
        'stt', 'tts', 'llm', 'user', 'orchestrator', 'session',
        'scenarios', 'conversation_store', 'file_storage', 'database',
        'rest_polling'
    ]
    
    success = 0
    failed = 0
    
    for module_name in modules:
        try:
            module = module_factory.create(module_name)
            module_type = type(module).__name__
            print(f"✅ {module_name:25s} -> {module_type}")
            success += 1
        except Exception as e:
            failed += 1
            print(f"❌ {module_name:25s} -> {str(e)[:60]}")
    
    print("=" * 80)
    print(f"Resultado: {success} sucessos, {failed} falhas")
    
    return failed == 0


def check_imports():
    """Verifica se há imports críticos de services"""
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO: Imports de services")
    print("=" * 80)
    
    import re
    
    services_dir = project_root / "src" / "services"
    critical_imports = []
    
    # Buscar imports em modules e api
    for py_file in project_root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if "src/services" in str(py_file):
            continue
        if "scripts" in str(py_file):
            continue
        if "tests" in str(py_file):
            continue
        
        try:
            content = py_file.read_text()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if re.search(r'from\s+src\.services\.|import.*src\.services\.', line):
                    # Verificar se é fallback
                    is_fallback = False
                    for j in range(max(0, i-10), min(len(lines), i+5)):
                        if 'try:' in lines[j] or 'except ImportError:' in lines[j]:
                            is_fallback = True
                            break
                    
                    if not is_fallback:
                        rel_path = py_file.relative_to(project_root)
                        critical_imports.append(f"{rel_path}:{i} - {line.strip()}")
        except Exception:
            pass
    
    if critical_imports:
        print(f"⚠️  {len(critical_imports)} imports críticos encontrados:")
        for imp in critical_imports[:10]:
            print(f"   - {imp}")
        if len(critical_imports) > 10:
            print(f"   ... e mais {len(critical_imports) - 10} imports")
        return False
    else:
        print("✅ Nenhum import crítico encontrado (todos têm fallback ou são opcionais)")
        return True


def remove_services_directory(dry_run=True):
    """Remove diretório src/services"""
    services_dir = project_root / "src" / "services"
    
    if not services_dir.exists():
        print("✅ src/services não existe - nada a remover")
        return True
    
    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Simulação de remoção")
        print("=" * 80)
        print(f"📁 Diretório a remover: {services_dir}")
        print(f"   Tamanho: {sum(f.stat().st_size for f in services_dir.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")
        print(f"   Arquivos: {len(list(services_dir.rglob('*.py')))} arquivos Python")
        print("\n⚠️  Use --force para remover de fato")
        return True
    else:
        print("\n" + "=" * 80)
        print("REMOÇÃO: src/services")
        print("=" * 80)
        
        # Criar backup
        backup_dir = project_root / "src" / "services.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        print(f"💾 Criando backup em {backup_dir}...")
        shutil.copytree(services_dir, backup_dir)
        print("✅ Backup criado")
        
        # Remover
        print(f"🗑️  Removendo {services_dir}...")
        shutil.rmtree(services_dir)
        print("✅ Diretório removido")
        
        return True


def main():
    """Executa verificação e remoção"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove src/services directory")
    parser.add_argument('--force', action='store_true', help='Remove de fato (não apenas simula)')
    parser.add_argument('--skip-verify', action='store_true', help='Pular verificação')
    
    args = parser.parse_args()
    
    # Verificações
    if not args.skip_verify:
        print("🔍 Executando verificações...\n")
        
        # Verificar módulos
        modules_ok = verify_modules_work()
        if not modules_ok:
            print("\n❌ Módulos não funcionam corretamente. Corrija antes de remover services/")
            return 1
        
        # Verificar imports
        imports_ok = check_imports()
        if not imports_ok:
            print("\n❌ Imports críticos encontrados. Corrija antes de remover services/")
            return 1
    
    # Remover
    remove_services_directory(dry_run=not args.force)
    
    if not args.force:
        print("\n✅ Verificações passaram! Use --force para remover de fato")
        return 0
    else:
        print("\n✅ src/services removido com sucesso!")
        print("💾 Backup disponível em src/services.backup")
        return 0


if __name__ == "__main__":
    sys.exit(main())
