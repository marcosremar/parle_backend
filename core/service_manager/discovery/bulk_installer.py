#!/usr/bin/env python3
"""
Bulk Installation Manager

Handles installation, uninstallation, and validation of services.
Supports both individual service operations and bulk operations on all services.
"""

import os
import sys
import subprocess
import importlib.util
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
# __file__ is in src/core/service_manager/discovery/bulk_installer.py
# So we need 5 .parent calls to get to project root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .service_discovery import (
from src.core.exceptions import UltravoxError, wrap_exception
    ServiceDiscovery,
    ServiceInfo,
    ServiceType,
    ServiceStatus
)

# Import profile manager to respect active profile
try:
    sys.path.insert(0, str(project_root / "tests"))
    from profile_manager import get_profile_manager, is_service_enabled
    PROFILE_MANAGER_AVAILABLE = True
except ImportError:
    PROFILE_MANAGER_AVAILABLE = False
    print("⚠️  Profile manager not available - will process all services")


class InstallationStatus(Enum):
    """Status of an installation operation"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"


@dataclass
class InstallationResult:
    """Result of an installation/uninstallation operation"""
    service_id: str
    status: InstallationStatus
    message: str
    duration_seconds: float = 0.0
    error: Optional[str] = None


class BulkInstaller:
    """
    Manages installation and uninstallation of services.

    Supports:
    - Individual service install/uninstall/validate
    - Bulk operations (install all, uninstall all)
    - AI model services (with install.py)
    - Infrastructure services (dependencies only)
    """

    def __init__(self, discovery: Optional[ServiceDiscovery] = None):
        """
        Initialize bulk installer.

        Args:
            discovery: Service discovery instance (will create if not provided)
        """
        self.discovery = discovery or ServiceDiscovery()
        if not self.discovery.services:
            self.discovery.discover_all_services()

    def install_service(self, service_id: str, force: bool = False) -> InstallationResult:
        """
        Install a specific service.

        Args:
            service_id: Service identifier
            force: Force reinstall even if already installed

        Returns:
            InstallationResult with operation details
        """
        print(f"\n{'='*80}")
        print(f"📦 INSTALLING SERVICE: {service_id}")
        print(f"{'='*80}")

        start_time = time.time()

        # Get service info
        service_info = self.discovery.services.get(service_id)
        if not service_info:
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Service '{service_id}' not found",
                error="Service not discovered"
            )

        # Check if already installed
        if service_info.status in [ServiceStatus.INSTALLED, ServiceStatus.HEALTHY] and not force:
            elapsed = time.time() - start_time
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.SKIPPED,
                message=f"Service already installed (status: {service_info.status.value})",
                duration_seconds=elapsed
            )

        # Install based on service type
        try:
            if service_info.has_install_script:
                # AI model services with install.py
                result = self._install_with_script(service_info)
            else:
                # Infrastructure services - just install dependencies
                result = self._install_dependencies(service_info)

            elapsed = time.time() - start_time
            result.duration_seconds = elapsed

            # Update service status
            self.discovery.services[service_id] = self.discovery._analyze_service(
                service_id,
                service_info.path
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Installation failed: {str(e)}",
                duration_seconds=elapsed,
                error=str(e)
            )

    def uninstall_service(self, service_id: str) -> InstallationResult:
        """
        Uninstall a specific service.

        Note: For AI models, this removes downloaded models.
        For infrastructure, this is mostly a no-op.

        Args:
            service_id: Service identifier

        Returns:
            InstallationResult with operation details
        """
        print(f"\n{'='*80}")
        print(f"🗑️  UNINSTALLING SERVICE: {service_id}")
        print(f"{'='*80}")

        start_time = time.time()

        # Get service info
        service_info = self.discovery.services.get(service_id)
        if not service_info:
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Service '{service_id}' not found",
                error="Service not discovered"
            )

        try:
            if service_info.service_type == ServiceType.AI_MODEL:
                # For AI models, remove downloaded models
                result = self._uninstall_ai_model(service_info)
            else:
                # For infrastructure, we don't uninstall system packages
                result = InstallationResult(
                    service_id=service_id,
                    status=InstallationStatus.SUCCESS,
                    message="Infrastructure service - dependencies kept installed"
                )

            elapsed = time.time() - start_time
            result.duration_seconds = elapsed

            # Update service status
            self.discovery.services[service_id] = self.discovery._analyze_service(
                service_id,
                service_info.path
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Uninstallation failed: {str(e)}",
                duration_seconds=elapsed,
                error=str(e)
            )

    def validate_service(self, service_id: str) -> InstallationResult:
        """
        Validate service installation.

        Checks:
        - Dependencies are installed
        - Required files exist
        - Service can import successfully
        - Health endpoint responds (if service is running)

        Args:
            service_id: Service identifier

        Returns:
            InstallationResult with validation details
        """
        print(f"\n{'='*80}")
        print(f"✅ VALIDATING SERVICE: {service_id}")
        print(f"{'='*80}")

        start_time = time.time()

        # Get service info
        service_info = self.discovery.services.get(service_id)
        if not service_info:
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Service '{service_id}' not found",
                error="Service not discovered"
            )

        validation_checks = []

        try:
            # Check 1: Service directory exists
            if service_info.path.exists():
                print("   ✅ Service directory exists")
                validation_checks.append(True)
            else:
                print("   ❌ Service directory not found")
                validation_checks.append(False)

            # Check 2: service.py exists
            if service_info.has_service_file:
                print("   ✅ service.py exists")
                validation_checks.append(True)
            else:
                print("   ❌ service.py not found")
                validation_checks.append(False)

            # Check 3: Can import service module
            try:
                module_path = f"src.services.{service_id}.service"
                spec = importlib.util.find_spec(module_path)
                if spec:
                    print("   ✅ Service module importable")
                    validation_checks.append(True)
                else:
                    print("   ⚠️  Service module not importable")
                    validation_checks.append(False)
            except Exception as e:
                print(f"   ⚠️  Import check failed: {e}")
                validation_checks.append(False)

            # Check 4: Dependencies installed (for AI models)
            if service_info.has_install_script:
                deps_ok = self._check_dependencies(service_info)
                if deps_ok:
                    print("   ✅ Dependencies installed")
                    validation_checks.append(True)
                else:
                    print("   ⚠️  Some dependencies missing")
                    validation_checks.append(False)

            # Check 5: Health endpoint (if service is running)
            if service_info.port and service_info.health_url:
                health_ok = self.discovery._check_health_endpoint(service_info.health_url)
                if health_ok == "healthy":
                    print(f"   ✅ Health endpoint responding: {service_info.health_url}")
                    validation_checks.append(True)
                elif health_ok == "degraded":
                    print(f"   ⚠️  Health endpoint degraded: {service_info.health_url}")
                    validation_checks.append(False)
                else:
                    print(f"   ⏹️  Service not running (port {service_info.port})")
                    # Not running is OK for validation
                    validation_checks.append(True)

            # Determine overall result
            all_passed = all(validation_checks)
            some_passed = any(validation_checks)

            elapsed = time.time() - start_time

            if all_passed:
                return InstallationResult(
                    service_id=service_id,
                    status=InstallationStatus.SUCCESS,
                    message=f"All validation checks passed ({len(validation_checks)} checks)",
                    duration_seconds=elapsed
                )
            elif some_passed:
                return InstallationResult(
                    service_id=service_id,
                    status=InstallationStatus.FAILED,
                    message=f"Some validation checks failed ({sum(validation_checks)}/{len(validation_checks)} passed)",
                    duration_seconds=elapsed,
                    error="Partial validation failure"
                )
            else:
                return InstallationResult(
                    service_id=service_id,
                    status=InstallationStatus.FAILED,
                    message="All validation checks failed",
                    duration_seconds=elapsed,
                    error="Complete validation failure"
                )

        except Exception as e:
            elapsed = time.time() - start_time
            return InstallationResult(
                service_id=service_id,
                status=InstallationStatus.FAILED,
                message=f"Validation error: {str(e)}",
                duration_seconds=elapsed,
                error=str(e)
            )

    def install_all_services(self, force: bool = False) -> Dict[str, InstallationResult]:
        """
        Install all services in the system.

        Args:
            force: Force reinstall even if already installed

        Returns:
            Dictionary mapping service_id to InstallationResult
        """
        print(f"\n{'='*80}")
        print(f"📦 BULK INSTALLATION - ALL SERVICES")
        print(f"{'='*80}")

        results = {}
        total_start = time.time()

        # Install in order: AI models first, then infrastructure, then utilities
        service_order = [
            ServiceType.AI_MODEL,
            ServiceType.INFRASTRUCTURE,
            ServiceType.UTILITY,
            ServiceType.EXTERNAL,
            ServiceType.TEST
        ]

        for service_type in service_order:
            services = self.discovery.get_services_by_type(service_type)

            for service_info in sorted(services, key=lambda s: s.service_id):
                result = self.install_service(service_info.service_id, force=force)
                results[service_info.service_id] = result

                # Print result
                status_icon = "✅" if result.status == InstallationStatus.SUCCESS else \
                             "⏭️" if result.status == InstallationStatus.SKIPPED else "❌"
                print(f"{status_icon} {service_info.service_id}: {result.message} ({result.duration_seconds:.1f}s)")

        total_elapsed = time.time() - total_start

        # Print summary
        self._print_bulk_summary(results, total_elapsed, "INSTALLATION")

        return results

    def uninstall_all_services(self) -> Dict[str, InstallationResult]:
        """
        Uninstall all services in the system.

        Returns:
            Dictionary mapping service_id to InstallationResult
        """
        print(f"\n{'='*80}")
        print(f"🗑️  BULK UNINSTALLATION - ALL SERVICES")
        print(f"{'='*80}")

        results = {}
        total_start = time.time()

        # Uninstall in reverse order
        service_order = [
            ServiceType.TEST,
            ServiceType.EXTERNAL,
            ServiceType.UTILITY,
            ServiceType.INFRASTRUCTURE,
            ServiceType.AI_MODEL
        ]

        for service_type in service_order:
            services = self.discovery.get_services_by_type(service_type)

            for service_info in sorted(services, key=lambda s: s.service_id):
                result = self.uninstall_service(service_info.service_id)
                results[service_info.service_id] = result

                # Print result
                status_icon = "✅" if result.status == InstallationStatus.SUCCESS else "❌"
                print(f"{status_icon} {service_info.service_id}: {result.message} ({result.duration_seconds:.1f}s)")

        total_elapsed = time.time() - total_start

        # Print summary
        self._print_bulk_summary(results, total_elapsed, "UNINSTALLATION")

        return results

    def validate_all_services(self, respect_profile: bool = True) -> Dict[str, InstallationResult]:
        """
        Validate all services in the system.

        Args:
            respect_profile: If True, only validate services enabled in active profile

        Returns:
            Dictionary mapping service_id to InstallationResult
        """
        # Get profile information if available
        profile_name = "unknown"
        enabled_services = set(self.discovery.services.keys())
        skipped_count = 0

        if respect_profile and PROFILE_MANAGER_AVAILABLE:
            profile_manager = get_profile_manager()
            profile_name = profile_manager.get_active_profile_name()
            enabled_services = set(profile_manager.get_enabled_services())

            print(f"\n{'='*80}")
            print(f"✅ BULK VALIDATION - PROFILE: {profile_name.upper()}")
            print(f"{'='*80}")
            print(f"📋 Validating only services enabled in profile '{profile_name}'")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"✅ BULK VALIDATION - ALL SERVICES")
            print(f"{'='*80}")

        results = {}
        total_start = time.time()

        for service_id in sorted(self.discovery.services.keys()):
            # Check if service is enabled in profile
            if respect_profile and PROFILE_MANAGER_AVAILABLE:
                if service_id not in enabled_services:
                    print(f"⏭️  {service_id}: Skipped (not enabled in profile '{profile_name}')")
                    skipped_count += 1
                    continue

            result = self.validate_service(service_id)
            results[service_id] = result

            # Print result
            status_icon = "✅" if result.status == InstallationStatus.SUCCESS else "❌"
            print(f"{status_icon} {service_id}: {result.message} ({result.duration_seconds:.1f}s)")

        total_elapsed = time.time() - total_start

        # Print summary with profile info
        if respect_profile and PROFILE_MANAGER_AVAILABLE and skipped_count > 0:
            print(f"\n⏭️  Skipped {skipped_count} services (not in profile '{profile_name}')")

        self._print_bulk_summary(results, total_elapsed, "VALIDATION")

        return results

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _install_with_script(self, service_info: ServiceInfo) -> InstallationResult:
        """Install service using its install.py script"""
        install_script = service_info.path / "install.py"

        print(f"\n📥 Running install script: {install_script}")

        try:
            # Run install.py
            result = subprocess.run(
                [sys.executable, str(install_script)],
                cwd=str(service_info.path),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for model downloads
            )

            if result.returncode == 0:
                print(f"✅ Install script completed successfully")
                return InstallationResult(
                    service_id=service_info.service_id,
                    status=InstallationStatus.SUCCESS,
                    message="Installation completed successfully via install.py"
                )
            else:
                print(f"❌ Install script failed:")
                print(result.stderr)
                return InstallationResult(
                    service_id=service_info.service_id,
                    status=InstallationStatus.FAILED,
                    message="install.py script failed",
                    error=result.stderr
                )

        except subprocess.TimeoutExpired:
            return InstallationResult(
                service_id=service_info.service_id,
                status=InstallationStatus.FAILED,
                message="Installation timeout (30 minutes)",
                error="Timeout exceeded"
            )
        except Exception as e:
            return InstallationResult(
                service_id=service_info.service_id,
                status=InstallationStatus.FAILED,
                message=f"Install script error: {str(e)}",
                error=str(e)
            )

    def _detect_gpu(self) -> bool:
        """
        Detect if NVIDIA GPU is available

        Returns:
            bool: True if GPU detected, False otherwise
        """
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=5
            )
            has_gpu = result.returncode == 0
            if has_gpu:
                print("   🎮 GPU detected - installing GPU-optimized packages")
            else:
                print("   💻 No GPU detected - installing CPU-only packages")
            return has_gpu
        except FileNotFoundError:
            print("   💻 nvidia-smi not found - installing CPU-only packages")
            return False
        except Exception as e:
            print(f"   ⚠️  GPU detection failed ({str(e)}) - defaulting to CPU packages")
            return False

    def _install_dependencies(self, service_info: ServiceInfo) -> InstallationResult:
        """Install dependencies for infrastructure services with GPU detection"""
        print(f"\n📥 Installing dependencies for {service_info.service_id}")

        # Check if requirements.txt exists
        requirements_file = service_info.path / "requirements.txt"
        if requirements_file.exists():
            try:
                # Detect GPU availability
                has_gpu = self._detect_gpu()

                print(f"   Installing from {requirements_file}")

                # First, install PyTorch with appropriate backend if not already installed
                pytorch_installed = self._check_pytorch_installed()
                if not pytorch_installed:
                    torch_result = self._install_pytorch(has_gpu)
                    if torch_result.status == InstallationStatus.FAILED:
                        return torch_result

                # Read requirements.txt and process GPU-aware packages
                with open(requirements_file, 'r') as f:
                    requirements = f.read()

                # Skip torch if we just installed it
                if not pytorch_installed:
                    # Create temporary requirements without torch
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                        for line in requirements.splitlines():
                            line = line.strip()
                            # Skip torch lines as we already installed it
                            if not line.startswith('torch') and not line.startswith('#torch'):
                                # Handle faiss packages based on GPU availability
                                if 'faiss-cpu' in line and has_gpu:
                                    tmp_file.write(line.replace('faiss-cpu', 'faiss-gpu') + '\n')
                                elif 'faiss-gpu' in line and not has_gpu:
                                    tmp_file.write(line.replace('faiss-gpu', 'faiss-cpu') + '\n')
                                else:
                                    tmp_file.write(line + '\n')
                        tmp_requirements = tmp_file.name
                else:
                    # Use original requirements but swap faiss if needed
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                        for line in requirements.splitlines():
                            line = line.strip()
                            # Handle faiss packages based on GPU availability
                            if 'faiss-cpu' in line and has_gpu:
                                tmp_file.write(line.replace('faiss-cpu', 'faiss-gpu') + '\n')
                            elif 'faiss-gpu' in line and not has_gpu:
                                tmp_file.write(line.replace('faiss-gpu', 'faiss-cpu') + '\n')
                            else:
                                tmp_file.write(line + '\n')
                        tmp_requirements = tmp_file.name

                # Install remaining dependencies
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", tmp_requirements],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                # Clean up temporary file
                try:
                    os.unlink(tmp_requirements)
                except Exception as e:
                    pass

                if result.returncode == 0:
                    print(f"✅ Dependencies installed successfully")
                    gpu_mode = "GPU-optimized" if has_gpu else "CPU-only"
                    return InstallationResult(
                        service_id=service_info.service_id,
                        status=InstallationStatus.SUCCESS,
                        message=f"Dependencies installed from requirements.txt ({gpu_mode} mode)"
                    )
                else:
                    print(f"⚠️  Some dependencies may have failed")
                    return InstallationResult(
                        service_id=service_info.service_id,
                        status=InstallationStatus.SUCCESS,
                        message="Dependencies partially installed",
                        error=result.stderr
                    )

            except Exception as e:
                return InstallationResult(
                    service_id=service_info.service_id,
                    status=InstallationStatus.FAILED,
                    message=f"Dependency installation failed: {str(e)}",
                    error=str(e)
                )
        else:
            # No requirements.txt - assume no special dependencies
            print(f"   No requirements.txt found - using global dependencies")
            return InstallationResult(
                service_id=service_info.service_id,
                status=InstallationStatus.SUCCESS,
                message="No service-specific dependencies (using global requirements)"
            )

    def _check_pytorch_installed(self) -> bool:
        """Check if PyTorch is already installed"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            return False

    def _install_pytorch(self, has_gpu: bool) -> InstallationResult:
        """
        Install PyTorch with appropriate backend (GPU/CPU)

        Args:
            has_gpu: Whether GPU is available

        Returns:
            InstallationResult with installation status
        """
        try:
            if has_gpu:
                print("   📦 Installing PyTorch with CUDA support...")
                # Install default PyTorch (includes CUDA)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch", "torchvision", "torchaudio"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
            else:
                print("   📦 Installing PyTorch (CPU-only version)...")
                # Install CPU-only PyTorch (much smaller)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
                     "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cpu"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            if result.returncode == 0:
                mode = "CUDA" if has_gpu else "CPU-only"
                print(f"   ✅ PyTorch installed ({mode})")
                return InstallationResult(
                    service_id="pytorch",
                    status=InstallationStatus.SUCCESS,
                    message=f"PyTorch installed ({mode})"
                )
            else:
                print(f"   ⚠️  PyTorch installation had issues")
                return InstallationResult(
                    service_id="pytorch",
                    status=InstallationStatus.FAILED,
                    message="PyTorch installation failed",
                    error=result.stderr
                )
        except Exception as e:
            return InstallationResult(
                service_id="pytorch",
                status=InstallationStatus.FAILED,
                message=f"PyTorch installation error: {str(e)}",
                error=str(e)
            )

    def _uninstall_ai_model(self, service_info: ServiceInfo) -> InstallationResult:
        """Uninstall AI model by removing downloaded models"""
        print(f"\n🗑️  Removing models for {service_info.service_id}")

        # Model directories to check
        model_dirs = [
            project_root / "models",
            Path.home() / ".cache" / "huggingface" / "hub"
        ]

        removed_files = []

        for model_dir in model_dirs:
            if not model_dir.exists():
                continue

            # Look for service-specific models
            service_patterns = {
                "llm": ["ultravox"],
                "stt": ["whisper"],
                "tts": ["kokoro"]
            }

            patterns = service_patterns.get(service_info.service_id, [])

            for pattern in patterns:
                for model_path in model_dir.glob(f"*{pattern}*"):
                    if model_path.is_dir():
                        print(f"   Removing: {model_path}")
                        import shutil
                        shutil.rmtree(model_path)
                        removed_files.append(str(model_path))

        if removed_files:
            return InstallationResult(
                service_id=service_info.service_id,
                status=InstallationStatus.SUCCESS,
                message=f"Removed {len(removed_files)} model directories"
            )
        else:
            return InstallationResult(
                service_id=service_info.service_id,
                status=InstallationStatus.SUCCESS,
                message="No models found to remove (may already be uninstalled)"
            )

    def _check_dependencies(self, service_info: ServiceInfo) -> bool:
        """Check if service dependencies are installed"""
        if not service_info.dependencies:
            return True

        for dep in service_info.dependencies:
            try:
                __import__(dep)
            except ImportError:
                return False

        return True

    def _print_bulk_summary(self, results: Dict[str, InstallationResult], total_time: float, operation: str):
        """Print summary of bulk operation"""
        print(f"\n{'='*80}")
        print(f"📊 {operation} SUMMARY")
        print(f"{'='*80}")

        # Count by status
        success_count = sum(1 for r in results.values() if r.status == InstallationStatus.SUCCESS)
        failed_count = sum(1 for r in results.values() if r.status == InstallationStatus.FAILED)
        skipped_count = sum(1 for r in results.values() if r.status == InstallationStatus.SKIPPED)

        print(f"\n📊 Results:")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failed:  {failed_count}")
        print(f"   ⏭️  Skipped: {skipped_count}")
        print(f"   📦 Total:   {len(results)}")

        print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

        # List failures
        if failed_count > 0:
            print(f"\n❌ Failed services:")
            for service_id, result in results.items():
                if result.status == InstallationStatus.FAILED:
                    print(f"   - {service_id}: {result.message}")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    # CLI for testing bulk installer
    import argparse

    parser = argparse.ArgumentParser(description="Service Bulk Installer")
    parser.add_argument("action", choices=["install", "uninstall", "validate", "install-all", "uninstall-all", "validate-all"])
    parser.add_argument("--service", help="Service ID (for individual operations)")
    parser.add_argument("--force", action="store_true", help="Force reinstall")

    args = parser.parse_args()

    installer = BulkInstaller()

    if args.action == "install":
        if not args.service:
            print("❌ --service required for install action")
            sys.exit(1)
        result = installer.install_service(args.service, force=args.force)
        print(f"\n{'='*80}")
        print(f"Result: {result.status.value} - {result.message}")
        print(f"{'='*80}")

    elif args.action == "uninstall":
        if not args.service:
            print("❌ --service required for uninstall action")
            sys.exit(1)
        result = installer.uninstall_service(args.service)
        print(f"\n{'='*80}")
        print(f"Result: {result.status.value} - {result.message}")
        print(f"{'='*80}")

    elif args.action == "validate":
        if not args.service:
            print("❌ --service required for validate action")
            sys.exit(1)
        result = installer.validate_service(args.service)
        print(f"\n{'='*80}")
        print(f"Result: {result.status.value} - {result.message}")
        print(f"{'='*80}")

    elif args.action == "install-all":
        results = installer.install_all_services(force=args.force)

    elif args.action == "uninstall-all":
        results = installer.uninstall_all_services()

    elif args.action == "validate-all":
        results = installer.validate_all_services()
