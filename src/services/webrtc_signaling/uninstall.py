#!/usr/bin/env python3
"""
Webrtc Signaling Service Uninstallation Script
Removes virtual environment and cleans tmp directory
"""

import sys
import shutil
from pathlib import Path

def remove_venv():
    """Remove virtual environment"""
    service_dir = Path(__file__).parent
    venv_dir = service_dir / "tmp" / "venv"

    print(f"🗑️  Removendo ambiente virtual...")

    if venv_dir.exists():
        try:
            shutil.rmtree(venv_dir)
            print(f"   ✅ Venv removido: {venv_dir}")
            return True
        except Exception as e:
            print(f"   ❌ Erro ao remover venv: {e}")
            return False
    else:
        print(f"   ℹ️  Venv não encontrado (já removido)")
        return True

def clear_tmp_directory():
    """Clear entire tmp directory"""
    service_dir = Path(__file__).parent
    tmp_dir = service_dir / "tmp"

    print(f"\n🗑️  Limpando diretório tmp...")

    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            print(f"   ✅ Diretório tmp removido: {tmp_dir}")
            return True
        except Exception as e:
            print(f"   ❌ Erro ao remover tmp: {e}")
            return False
    else:
        print(f"   ℹ️  Diretório tmp não encontrado (já removido)")
        return True

def main():
    """Main uninstallation function"""
    print("🚀 Webrtc Signaling Service - Desinstalação")
    print("=" * 60)

    # Step 1: Remove venv
    if not remove_venv():
        print("\n⚠️ Falha ao remover venv, mas continuando...")

    # Step 2: Clear tmp directory
    if not clear_tmp_directory():
        print("\n❌ Falha ao limpar diretório tmp")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Desinstalação concluída com sucesso!")
    print("\n📋 O que foi removido:")
    print("   - Ambiente virtual (venv)")
    print("   - Diretório tmp completo")
    print("\n💡 Para reinstalar:")
    print("   python3 install.py")

if __name__ == "__main__":
    main()
