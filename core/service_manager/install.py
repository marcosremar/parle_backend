#!/usr/bin/env python3
"""
Service Manager Installation Script

Installs dependencies required for service discovery, installation, and testing.
"""

import subprocess
import sys
import shutil
from pathlib import Path

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
            timeout=5
        )
        return True
    except Exception as e:
        print(f"⚠️  pip check failed: {e}")
        return False

def install_with_apt():
    """Install dependencies using apt (Ubuntu/Debian)"""
    print("\n📦 Installing dependencies using apt...")

    packages = [
        "python3-pytest",
        "python3-pytest-asyncio",
        "python3-yaml",
        "python3-requests"
    ]

    try:
        subprocess.run(
            ["sudo", "apt-get", "install", "-y"] + packages,
            check=True,
            timeout=300  # 5 minutes for apt install
        )
        print("✅ Dependencies installed successfully via apt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ apt installation failed: {e}")
        return False

def install_with_pip(requirements_file):
    """Install dependencies using pip"""
    print(f"\n📦 Installing from {requirements_file}")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            timeout=300  # 5 minutes for pip install
        )
        print("✅ Dependencies installed successfully via pip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ pip installation failed: {e}")
        return False

def main():
    """Install Service Manager dependencies"""
    print("=" * 80)
    print("INSTALLING SERVICE MANAGER DEPENDENCIES")
    print("=" * 80)

    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return 1

    # Check if pip is available
    has_pip = check_pip()
    has_apt = shutil.which("apt-get") is not None

    success = False

    if has_pip:
        print("\n✓ pip detected - using pip for installation")
        success = install_with_pip(requirements_file)
    elif has_apt:
        print("\n⚠️  pip not available - falling back to apt")
        success = install_with_apt()
    else:
        print("\n❌ Neither pip nor apt-get available")
        print("\nPlease install dependencies manually:")
        print("  - pyyaml")
        print("  - requests")
        print("  - pytest")
        print("  - pytest-asyncio")
        return 1

    if success:
        print("\nInstalled packages:")
        print("  ✓ pyyaml")
        print("  ✓ requests")
        print("  ✓ pytest")
        print("  ✓ pytest-asyncio")
        print("  ✓ pytest-cov (optional)")
        print("  ✓ pytest-mock (optional)")

        print("\n" + "=" * 80)
        print("INSTALLATION COMPLETE")
        print("=" * 80)
        print("\nYou can now run tests with:")
        print("  python3 -m pytest src/core/service_manager/tests/ -v")
        return 0
    else:
        print("\n❌ Installation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
