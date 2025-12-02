#!/usr/bin/env python3
"""
Universal Service Installation Script

Uses VenvManager to create isolated venvs with automatic:
- Python version detection from requirements.txt
- GPU detection and appropriate package installation
- Isolated dependencies per service
"""

import sys
from pathlib import Path

# Add project root to path (services are at src/services/<service>/install.py)
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent  # Go up 4 levels
sys.path.insert(0, str(project_root))

from src.utils.venv_manager import get_venv_manager


def main():
    """Install service with isolated venv"""
    # Detect service name from script location
    script_path = Path(__file__).resolve()
    service_name = script_path.parent.name

    print(f"=" * 80)
    print(f"{service_name.upper()} SERVICE INSTALLATION")
    print(f"=" * 80)
    print()

    # Get VenvManager singleton
    venv_manager = get_venv_manager()

    # Setup isolated venv with automatic dependency installation
    print(f"📦 Setting up isolated venv for '{service_name}' service...")
    success = venv_manager.setup_service_venv(service_name, install_deps=True)

    if success:
        venv_path = venv_manager.get_venv_path(service_name)
        print()
        print(f"✅ Service '{service_name}' installed successfully!")
        print(f"   Venv location: {venv_path}")
        return 0
    else:
        print()
        print(f"❌ Failed to install '{service_name}' service")
        return 1


if __name__ == "__main__":
    sys.exit(main())
