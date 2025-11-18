"""
Service Validator - Validates service installation and configuration
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ServiceValidator:
    """
    Validates service installation, venv, dependencies, and configuration
    """

    def __init__(
        self,
        service_name: str,
        project_root: str = os.getenv("ULTRAVOX_HOME", str(Path(__file__).parent.parent.parent.parent)),
    ):
        self.service_name = service_name
        self.project_root = Path(project_root)
        self.service_dir = self.project_root / "src" / "services" / service_name
        # Use same pattern as VenvManager: .venvs/{service_name}_service
        self.venv_dir = self.project_root / ".venvs" / f"{service_name}_service"

    def validate_all(self) -> Dict[str, Any]:
        """Run all validations"""
        results = {
            "service_name": self.service_name,
            "validations": {},
            "overall_status": "passed",
        }

        # 1. Service directory
        results["validations"]["service_directory"] = self._validate_directory()

        # 2. Requirements
        results["validations"]["requirements"] = self._validate_requirements()

        # 3. Venv
        results["validations"]["venv"] = self._validate_venv()

        # 4. Dependencies
        results["validations"]["dependencies"] = self._validate_dependencies()

        # 5. GPU requirements
        results["validations"]["gpu"] = self._validate_gpu_requirements()

        # Calculate overall status
        for validation in results["validations"].values():
            if validation["status"] == "failed":
                results["overall_status"] = "failed"
                break
            elif validation["status"] == "warning":
                results["overall_status"] = "warning"

        return results

    def _validate_directory(self) -> Dict[str, Any]:
        """Validate service directory exists"""
        if self.service_dir.exists():
            return {
                "status": "passed",
                "message": f"Service directory exists: {self.service_dir}",
            }
        else:
            return {
                "status": "failed",
                "message": f"Service directory not found: {self.service_dir}",
            }

    def _validate_requirements(self) -> Dict[str, Any]:
        """Validate requirements.txt"""
        req_file = self.service_dir / "requirements.txt"

        if not req_file.exists():
            return {
                "status": "warning",
                "message": "requirements.txt not found (may have no dependencies)",
            }

        try:
            with open(req_file) as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            return {
                "status": "passed",
                "message": f"requirements.txt valid ({len(lines)} dependencies)",
                "dependency_count": len(lines),
            }

        except Exception as e:
            return {"status": "failed", "message": f"Error reading requirements.txt: {e}"}

    def _validate_venv(self) -> Dict[str, Any]:
        """Validate venv exists"""
        if self.venv_dir.exists():
            return {
                "status": "passed",
                "message": f"Venv exists: {self.venv_dir}",
            }
        else:
            return {
                "status": "warning",
                "message": f"Venv not found: {self.venv_dir} (may use root venv)",
            }

    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies are installed"""
        # Check if venv exists
        if not self.venv_dir.exists():
            return {
                "status": "skipped",
                "message": "No venv to check (service may use root venv)",
            }

        # Count installed packages
        try:
            import subprocess

            python_exe = self.venv_dir / "bin" / "python3"

            if not python_exe.exists():
                return {
                    "status": "failed",
                    "message": f"Python executable not found in venv: {python_exe}",
                }

            result = subprocess.run(
                [str(python_exe), "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                packages = [
                    line for line in result.stdout.split("\n") if line.strip()
                ]
                return {
                    "status": "passed",
                    "message": f"Dependencies installed ({len(packages)} packages)",
                    "package_count": len(packages),
                }
            else:
                return {
                    "status": "failed",
                    "message": f"pip list failed: {result.stderr}",
                }

        except Exception as e:
            return {
                "status": "warning",
                "message": f"Could not check dependencies: {e}",
            }

    def _validate_gpu_requirements(self) -> Dict[str, Any]:
        """Validate GPU requirements"""
        try:
            from src.core.gpu_memory_manager import get_gpu_manager

            gpu_manager = get_gpu_manager()
            gpu_info = gpu_manager.get_gpu_info()

            if not gpu_info:
                return {
                    "status": "skipped",
                    "message": "No GPU available (service may not need GPU)",
                }

            return {
                "status": "passed",
                "message": f"GPU available: {gpu_info.free_mb}MB free / {gpu_info.total_mb}MB total",
                "gpu_info": {
                    "free_mb": gpu_info.free_mb,
                    "total_mb": gpu_info.total_mb,
                    "used_mb": gpu_info.used_mb,
                },
            }

        except ImportError:
            return {
                "status": "skipped",
                "message": "GPU manager not available",
            }
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Could not check GPU: {e}",
            }

    def format_text(self, results: Dict[str, Any]) -> str:
        """Format validation results as text"""
        lines = []

        lines.append("=" * 80)
        lines.append(f"🔍 SERVICE VALIDATION: {results['service_name']}")
        lines.append("=" * 80)
        lines.append("")

        for name, result in results["validations"].items():
            status = result["status"]
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "warning": "⚠️",
                "skipped": "⊘",
            }.get(status, "❓")

            lines.append(f"{status_icon} {name.replace('_', ' ').title()}")
            lines.append(f"   {result['message']}")
            lines.append("")

        # Overall
        lines.append("━" * 80)
        overall_icon = {
            "passed": "✅",
            "failed": "❌",
            "warning": "⚠️",
        }.get(results["overall_status"], "❓")

        lines.append(f"{overall_icon} Overall Status: {results['overall_status'].upper()}")
        lines.append("=" * 80)

        return "\n".join(lines)
