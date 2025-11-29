#!/usr/bin/env python3
"""
Script para remover arquivos duplicados com segurança

IMPORTANTE: Execute check_duplicate_files.py primeiro para verificar
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.check_duplicate_files import check_duplicate_files


def remove_files(files: list, dry_run: bool = True):
    """Remove arquivos duplicados"""
    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Nenhum arquivo será removido")
        print("=" * 80)
    
    removed = []
    failed = []
    
    for item in files:
        file_path = item['path']
        
        if dry_run:
            print(f"📄 [DRY RUN] Removeria: {file_path}")
            removed.append(file_path)
        else:
            try:
                # Criar backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    print(f"💾 Backup criado: {backup_path}")
                
                # Remover arquivo
                file_path.unlink()
                print(f"✅ Removido: {file_path}")
                removed.append(file_path)
            except Exception as e:
                print(f"❌ Erro ao remover {file_path}: {e}")
                failed.append(file_path)
    
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Removidos: {len(removed)}")
    print(f"Falhas: {len(failed)}")
    
    if failed:
        print("\n⚠️  Arquivos que falharam:")
        for f in failed:
            print(f"   - {f}")


def main():
    """Executa remoção"""
    parser = argparse.ArgumentParser(description="Remove arquivos duplicados")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Apenas simula remoção (padrão)')
    parser.add_argument('--force', action='store_true',
                       help='Remove arquivos de fato (não apenas simula)')
    
    args = parser.parse_args()
    
    # Verificar arquivos
    removable, keep = check_duplicate_files()
    
    if not removable:
        print("\n✅ Nenhum arquivo duplicado encontrado para remover")
        return
    
    # Confirmar
    if not args.force:
        print("\n⚠️  Use --force para remover arquivos de fato")
        print("   (Por padrão, apenas simula - dry-run)")
    
    # Remover
    remove_files(removable, dry_run=not args.force)


if __name__ == "__main__":
    main()
