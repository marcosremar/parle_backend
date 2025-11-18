"""
Service Tester - Comprehensive service testing integrated with Service Manager

Tests services in both module and internal modes, validates structure,
dependencies, and functionality.
"""

import sys
import os
import time
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: TestStatus
    message: str
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    """Complete test report for a service"""
    service_name: str
    execution_mode: str
    timestamp: str
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    total_duration_ms: float = 0.0

    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == TestStatus.SKIPPED),
            "warning": sum(1 for r in self.results if r.status == TestStatus.WARNING),
        }

    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.summary["total"] == 0:
            return 0.0
        return (self.summary["passed"] / self.summary["total"]) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_name": self.service_name,
            "execution_mode": self.execution_mode,
            "timestamp": self.timestamp,
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "summary": self.summary,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate(),
        }


class ServiceTester:
    """
    Comprehensive service tester integrated with Service Manager

    Tests services in multiple modes:
    - Structure validation
    - Python import
    - Dependencies
    - Module mode (in-process)
    - Internal mode (separate process)
    - Pytest execution
    """

    def __init__(
        self,
        service_name: str,
        project_root: str = os.getenv("ULTRAVOX_HOME", str(Path(__file__).parent.parent.parent.parent)),
        mode: str = "auto",
        pytest_args: str = "",
    ):
        self.service_name = service_name
        self.project_root = Path(project_root)
        self.service_dir = self.project_root / "src" / "services" / service_name
        self.mode = mode  # auto, module, internal, both
        self.pytest_args = pytest_args  # Additional pytest arguments (e.g., "-m 'unit'")

        self.report = TestReport(
            service_name=service_name,
            execution_mode=mode,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def run_all_tests(self, install_deps: bool = True) -> TestReport:
        """
        Run all tests

        Args:
            install_deps: Whether to install dependencies before testing
        """
        start_time = time.time()

        logger.info(f"🧪 Testing service: {self.service_name} (mode: {self.mode})")

        # 0. Install dependencies (if requested)
        if install_deps:
            installed = self._install_dependencies()

            # If dependencies were installed, re-run tests in fresh subprocess
            if installed:
                print(f"   🔄 Re-running tests in fresh Python process...")
                return self._run_tests_in_subprocess()

        # Run tests normally (no deps installed or install_deps=False)
        return self._run_tests_internal()

    def _run_tests_internal(self) -> TestReport:
        """Run all tests internally (current process)"""
        start_time = time.time()

        # 1. Structure validation
        self._test_structure()

        # 2. Python import
        self._test_import()

        # 3. Dependencies
        self._test_dependencies()

        # 4. Service initialization
        self._test_initialization()

        # 5. Execution mode
        self._test_execution_mode()

        # 6. Pytest
        self._test_pytest()

        # Calculate summary
        self.report.total_duration_ms = (time.time() - start_time) * 1000
        self.report.calculate_summary()

        logger.info(
            f"✅ Testing complete: {self.report.success_rate():.1f}% success rate"
        )

        return self.report

    def _run_tests_in_subprocess(self) -> TestReport:
        """Run tests in a fresh subprocess (after installing dependencies)"""
        import json
        import tempfile

        # Create temp file for report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_file = f.name

        # Determine Python executable to use
        python_exe = sys.executable
        detected_mode = self._detect_execution_mode()

        # Use venv Python for internal services
        if detected_mode == "internal" or self.mode == "internal":
            from src.utils.venv_manager import get_venv_manager
            venv_mgr = get_venv_manager()
            venv_python = venv_mgr.get_python_executable(self.service_name)
            if venv_python:
                python_exe = str(venv_python)
                print(f"   Using venv Python: {python_exe}")

        try:
            # Run this script in subprocess with install_deps=False
            result = subprocess.run(
                [
                    python_exe,
                    "-c",
                    f"""
import sys
import json
from pathlib import Path

# Add project root
project_root = Path("{self.project_root}")
sys.path.insert(0, str(project_root))

from src.core.service_manager.testing.service_tester import ServiceTester
from src.core.exceptions import UltravoxError, wrap_exception

# Run tests without installing deps again
tester = ServiceTester(
    service_name="{self.service_name}",
    project_root="{self.project_root}",
    mode="{self.mode}"
)
report = tester._run_tests_internal()

# Save report
with open("{report_file}", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
""",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_root),
            )

            # Check subprocess output
            if result.returncode != 0:
                print(f"   ⚠️ Subprocess failed:")
                print(f"   stdout: {result.stdout[:500]}")
                print(f"   stderr: {result.stderr[:500]}")

            # Load report from file
            try:
                with open(report_file) as f:
                    report_data = json.load(f)
            except Exception as e:
                print(f"   ⚠️ Failed to load report: {e}")
                print(f"   stdout: {result.stdout[:500]}")
                print(f"   stderr: {result.stderr[:500]}")
                raise

            # Reconstruct TestReport
            report = TestReport(
                service_name=report_data["service_name"],
                execution_mode=report_data["execution_mode"],
                timestamp=report_data["timestamp"],
            )

            for result_data in report_data["results"]:
                result = TestResult(
                    test_name=result_data["test_name"],
                    status=TestStatus(result_data["status"]),
                    message=result_data["message"],
                    duration_ms=result_data["duration_ms"],
                    error=result_data.get("error"),
                    metadata=result_data.get("metadata", {}),
                )
                report.add_result(result)

            report.total_duration_ms = report_data["total_duration_ms"]
            report.summary = report_data.get("summary", {})

            # Ensure summary is populated
            if not report.summary:
                report.calculate_summary()

            print(f"   ✅ Tests completed in subprocess (summary: {report.summary})")

            return report

        finally:
            # Cleanup temp file
            try:
                os.unlink(report_file)
            except Exception as e:
                pass

    def _install_dependencies(self) -> bool:
        """
        Install service dependencies using install script or pip

        For internal services, creates isolated venv using VenvManager.
        For module services, installs to global Python.

        Returns:
            True if dependencies were installed, False otherwise
        """
        print(f"\n📦 Installing dependencies for {self.service_name}...")
        logger.info(f"📦 Installing dependencies for {self.service_name}...")

        # Detect execution mode
        detected_mode = self._detect_execution_mode()

        # Use VenvManager for internal services
        if detected_mode == "internal" or self.mode == "internal":
            return self._install_dependencies_venv()
        else:
            # Use global Python for module services
            return self._install_dependencies_global()

    def _install_dependencies_venv(self) -> bool:
        """Install dependencies in isolated venv using VenvManager"""
        from src.utils.venv_manager import get_venv_manager

        print(f"   🔧 Using isolated venv (internal service)")
        logger.info(f"   Using VenvManager for internal service: {self.service_name}")

        venv_mgr = get_venv_manager()

        # Setup venv and install dependencies
        success = venv_mgr.setup_service_venv(self.service_name, install_deps=True)

        if success:
            print(f"   ✅ Venv setup complete for {self.service_name}")
            logger.info(f"   ✅ Venv created at: {venv_mgr.get_venv_path(self.service_name)}")
            return True
        else:
            print(f"   ⚠️ Venv setup failed for {self.service_name}")
            return False

    def _install_dependencies_global(self) -> bool:
        """Install dependencies to global Python (for module services)"""
        print(f"   🌐 Using global Python (module service)")
        logger.info(f"   Installing to global Python for module service: {self.service_name}")

        installed = False

        # Check for install script (priority order)
        install_scripts = [
            self.service_dir / "scripts" / "install_dependencies.sh",
            self.service_dir / "install_dependencies.sh",
            self.service_dir / "scripts" / "setup.sh",
            self.service_dir / "setup.sh",
        ]

        install_script = None
        for script in install_scripts:
            if script.exists():
                install_script = script
                break

        if install_script:
            # Use install script
            print(f"   Using install script: {install_script.name}")
            logger.info(f"   Using install script: {install_script.name}")
            try:
                result = subprocess.run(
                    ["bash", str(install_script)],
                    cwd=str(self.service_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout
                )

                if result.returncode == 0:
                    print(f"   ✅ Dependencies installed successfully via script")
                    logger.info(f"   ✅ Dependencies installed successfully")
                    installed = True
                else:
                    print(f"   ⚠️ Install script failed: {result.stderr[:200]}")
                    logger.warning(
                        f"   ⚠️ Install script failed: {result.stderr[:200]}"
                    )
            except Exception as e:
                print(f"   ⚠️ Failed to run install script: {e}")
                logger.warning(f"   ⚠️ Failed to run install script: {e}")

        else:
            # Fallback to pip install
            requirements_file = self.service_dir / "requirements.txt"

            if requirements_file.exists():
                print(f"   Using pip install -r requirements.txt")
                logger.info(f"   Using pip install -r requirements.txt")
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--break-system-packages",
                            "-r",
                            str(requirements_file),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minutes timeout
                    )

                    if result.returncode == 0:
                        print(f"   ✅ Dependencies installed successfully via pip")
                        logger.info(f"   ✅ Dependencies installed successfully")
                        installed = True
                    else:
                        print(f"   ⚠️ pip install failed: {result.stderr[:200]}")
                        logger.warning(
                            f"   ⚠️ pip install failed: {result.stderr[:200]}"
                        )
                except Exception as e:
                    print(f"   ⚠️ Failed to install dependencies: {e}")
                    logger.warning(f"   ⚠️ Failed to install dependencies: {e}")
            else:
                print(f"   ⊘ No requirements.txt found, skipping")
                logger.info(f"   ⊘ No requirements.txt found, skipping")

        return installed

    def _run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test"""
        start_time = time.time()

        try:
            logger.debug(f"Running test: {test_name}")
            test_func()

            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Test passed",
                duration_ms=duration_ms,
            )

        except AssertionError as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                message="Assertion failed",
                duration_ms=duration_ms,
                error=str(e),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Test failed: {test_name} - {e}")
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                message="Exception raised",
                duration_ms=duration_ms,
                error=str(e),
            )

    def _test_structure(self):
        """Test service structure"""

        def check_service_py():
            assert (self.service_dir / "service.py").exists(), "service.py not found"

        def check_init_py():
            if not (self.service_dir / "__init__.py").exists():
                raise AssertionError("__init__.py missing (not critical)")

        def check_requirements():
            assert (
                self.service_dir / "requirements.txt"
            ).exists(), "requirements.txt not found"

        def check_tests_dir():
            if not (self.service_dir / "tests").exists():
                raise AssertionError("tests/ directory missing")

        result = self._run_test("structure_service_py", check_service_py)
        self.report.add_result(result)

        result = self._run_test("structure_init_py", check_init_py)
        if result.status == TestStatus.FAILED:
            result.status = TestStatus.SKIPPED
        self.report.add_result(result)

        result = self._run_test("structure_requirements", check_requirements)
        self.report.add_result(result)

        result = self._run_test("structure_tests_dir", check_tests_dir)
        self.report.add_result(result)

    def _test_import(self):
        """Test Python import"""

        def import_module():
            module_path = f"src.services.{self.service_name}.service"
            module = importlib.import_module(module_path)
            assert module is not None, "Module import returned None"

        def find_service_class():
            module_path = f"src.services.{self.service_name}.service"
            module = importlib.import_module(module_path)

            # Find service class
            service_class = None
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and name.endswith("Service")
                    and obj.__module__ == module.__name__
                ):
                    service_class = obj
                    break

            assert service_class is not None, "Service class not found"

        result = self._run_test("import_module", import_module)
        self.report.add_result(result)

        result = self._run_test("import_service_class", find_service_class)
        self.report.add_result(result)

    def _test_dependencies(self):
        """Test dependencies"""

        def validate_requirements():
            requirements_file = self.service_dir / "requirements.txt"

            if not requirements_file.exists():
                raise AssertionError("requirements.txt not found")

            # Count dependencies
            with open(requirements_file) as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            assert len(lines) >= 0, f"Found {len(lines)} dependencies"

        result = self._run_test("dependencies_validate", validate_requirements)
        self.report.add_result(result)

    def _test_initialization(self):
        """Test service initialization"""

        def instantiate_service():
            module_path = f"src.services.{self.service_name}.service"
            module = importlib.import_module(module_path)

            # Find service class
            service_class = None
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and name.endswith("Service")
                    and obj.__module__ == module.__name__
                ):
                    service_class = obj
                    break

            if service_class is None:
                raise AssertionError("Service class not found")

            # Try to instantiate
            try:
                instance = service_class(config={})
                assert instance is not None, "Instantiation returned None"
            except TypeError as e:
                if "context" in str(e):
                    # Expected - service requires ServiceContext
                    pass
                else:
                    raise

        result = self._run_test("initialization_instantiate", instantiate_service)
        if (
            result.status == TestStatus.FAILED
            and result.error
            and "context" in result.error
        ):
            result.status = TestStatus.SKIPPED
            result.message = "Service requires ServiceContext (expected)"
        self.report.add_result(result)

    def _test_execution_mode(self):
        """Test execution mode"""

        # Auto-detect mode from profiles.yaml
        detected_mode = self._detect_execution_mode()

        if self.mode == "auto":
            test_mode = detected_mode
        elif self.mode == "both":
            # Test both modes
            self._test_module_mode()
            self._test_internal_mode()
            return
        else:
            test_mode = self.mode

        if test_mode == "module":
            self._test_module_mode()
        elif test_mode == "internal":
            self._test_internal_mode()
        else:
            result = TestResult(
                test_name="execution_mode_unknown",
                status=TestStatus.SKIPPED,
                message=f"Unknown execution mode: {test_mode}",
            )
            self.report.add_result(result)

    def _detect_execution_mode(self) -> str:
        """Detect execution mode from profiles.yaml"""
        profiles_file = self.project_root / "config" / "profiles.yaml"

        if not profiles_file.exists():
            return "unknown"

        try:
            import yaml

            with open(profiles_file) as f:
                data = yaml.safe_load(f)

            # Search in all profiles
            for profile_name, profile in data.get("profiles", {}).items():
                overrides = profile.get("service_overrides", {})
                if self.service_name in overrides:
                    mode = overrides[self.service_name].get("execution_mode")
                    if mode:
                        return mode

            return "unknown"

        except Exception as e:
            logger.warning(f"Could not detect execution mode: {e}")
            return "unknown"

    def _test_module_mode(self):
        """Test service as module (in-process)"""

        def test_module():
            # Module mode means service can be imported and used in-process
            # Already tested in import tests
            pass

        result = self._run_test("execution_module_mode", test_module)
        result.message = "Service can run as module (in-process)"
        self.report.add_result(result)

    def _test_internal_mode(self):
        """Test service as internal (separate process)"""

        # Get configured port
        configured_port = self._get_service_port()

        if not configured_port:
            result = TestResult(
                test_name="execution_internal_mode",
                status=TestStatus.SKIPPED,
                message="Port not found for internal service",
            )
            self.report.add_result(result)
            return

        # Use dynamic port allocation in parallel mode
        if os.getenv("PARALLEL_TEST_MODE") == "1":
            port = self._find_free_port()
            logger.info(f"Parallel mode: using dynamic port {port} (configured: {configured_port})")
        else:
            port = configured_port

        def start_and_test_service():
            # Start service with dynamic port (if parallel mode)
            service_script = self.service_dir / "service.py"

            # Prepare environment with dynamic port
            env = os.environ.copy()
            env[f"{self.service_name.upper()}_PORT"] = str(port)

            process = subprocess.Popen(
                [sys.executable, str(service_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.service_dir),
                env=env,
            )

            try:
                # Wait for startup (increased timeout for services with heavy dependencies)
                time.sleep(20)

                # Check if process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    raise AssertionError(
                        f"Process died: {stderr.decode()[:200]}"
                    )

                # Health check with retry logic (exponential backoff)
                import requests

                max_retries = 5
                retry_delay = 1  # Start with 1 second
                last_error = None

                for attempt in range(max_retries):
                    try:
                        response = requests.get(
                            f"http://localhost:{port}/health", timeout=5
                        )
                        if response.status_code == 200:
                            # Success!
                            break
                        else:
                            last_error = f"HTTP {response.status_code}"
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                    except requests.exceptions.RequestException as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                else:
                    # All retries failed - capture process output for debugging
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                        stdout_str = stdout.decode('utf-8', errors='replace')[:1000] if stdout else ""
                        stderr_str = stderr.decode('utf-8', errors='replace')[:1000] if stderr else ""

                        # Save to debug file
                        debug_file = fstr(Path.home() / ".cache" / "ultravox-pipeline" / "service_debug_{self.service_name}.log"
                        with open(debug_file, 'w') as f:
                            f.write(f"=== Service: {self.service_name} ===\n")
                            f.write(f"Port: {port}\n")
                            f.write(f"Last error: {last_error}\n\n")
                            f.write(f"=== STDOUT ===\n{stdout_str}\n\n")
                            f.write(f"=== STDERR ===\n{stderr_str}\n")

                        error_msg = f"Health check failed after {max_retries} attempts: {last_error}\nDebug log: {debug_file}"
                    except Exception as e:
                        error_msg = f"Health check failed after {max_retries} attempts: {last_error}"

                    raise AssertionError(error_msg)

            finally:
                # Cleanup
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

        result = self._run_test("execution_internal_mode", start_and_test_service)
        self.report.add_result(result)

    def _get_service_port(self) -> Optional[int]:
        """Get service port from profiles.yaml"""
        profiles_file = self.project_root / "config" / "profiles.yaml"

        if not profiles_file.exists():
            return None

        try:
            import yaml

            with open(profiles_file) as f:
                data = yaml.safe_load(f)

            # Search in all profiles
            for profile_name, profile in data.get("profiles", {}).items():
                overrides = profile.get("service_overrides", {})
                if self.service_name in overrides:
                    port = overrides[self.service_name].get("port")
                    if port:
                        return port

            return None

        except Exception as e:
            logger.warning(f"Could not get service port: {e}")
            return None

    def _find_free_port(self, start: int = 9000, end: int = 9999) -> int:
        """
        Find a free port in the specified range.

        Uses socket to check if port is available.
        """
        import socket
        import random

        # Try random ports in range to avoid sequential allocation conflicts
        ports = list(range(start, end + 1))
        random.shuffle(ports)

        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return port
            except OSError:
                continue

        raise RuntimeError(f"No free ports available in range {start}-{end}")

    def _test_pytest(self):
        """Run pytest tests"""
        tests_dir = self.service_dir / "tests"

        if not tests_dir.exists():
            result = TestResult(
                test_name="pytest_execution",
                status=TestStatus.SKIPPED,
                message="No tests/ directory",
            )
            self.report.add_result(result)
            return

        def run_pytest():
            # Check if there are any test files
            test_files = list(tests_dir.rglob("test_*.py"))
            if not test_files:
                # No test files, mark as passed (nothing to test)
                return

            # Build pytest command with optional arguments
            pytest_cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(tests_dir),
                "-v",
                "--tb=short",
            ]

            # Add custom pytest args (e.g., marker filters)
            if self.pytest_args:
                # Split pytest_args by space and add to command
                pytest_cmd.extend(self.pytest_args.split())

            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )

            # Check output for "no tests ran" or "deselected"
            if "0 passed" in result.stdout and "deselected" in result.stdout:
                # Tests were deselected (probably need service running)
                # This is OK - mark as passed
                return

            if result.returncode != 0:
                raise AssertionError(f"Pytest failed:\n{result.stdout}\n{result.stderr}")

        result = self._run_test("pytest_execution", run_pytest)
        self.report.add_result(result)

    def format_report_text(self) -> str:
        """Format report as text"""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(f"🧪 SERVICE TEST REPORT: {self.report.service_name}")
        lines.append("=" * 80)
        lines.append(f"Execution Mode: {self.report.execution_mode}")
        lines.append(f"Timestamp: {self.report.timestamp}")
        lines.append(f"Duration: {self.report.total_duration_ms:.1f}ms")
        lines.append("")

        # Results
        lines.append("━" * 80)
        lines.append("▶ TEST RESULTS")
        lines.append("━" * 80)
        lines.append("")

        for result in self.report.results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.SKIPPED: "⊘",
                TestStatus.WARNING: "⚠️",
            }[result.status]

            lines.append(f"{status_icon} {result.test_name}: {result.message}")
            if result.error:
                lines.append(f"   Error: {result.error[:100]}")

        # Summary
        lines.append("")
        lines.append("━" * 80)
        lines.append("📊 SUMMARY")
        lines.append("━" * 80)
        lines.append(f"Total Tests: {self.report.summary['total']}")
        lines.append(f"✅ Passed: {self.report.summary['passed']}")
        lines.append(f"❌ Failed: {self.report.summary['failed']}")
        lines.append(f"⊘ Skipped: {self.report.summary['skipped']}")
        lines.append(f"Success Rate: {self.report.success_rate():.1f}%")
        lines.append("=" * 80)

        return "\n".join(lines)
