#!/usr/bin/env python3
"""
Generic Validation Runner for Services

Provides a base ValidationRunner that services can use for their /validate endpoint.
Services can extend this class to add service-specific tests.

Features:
- Runs pytest tests automatically if available
- Falls back to basic health checks
- Provides structured JSON output
"""

import asyncio
import aiohttp
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Generic validation runner for service health and basic functionality checks

    Usage:
        # In service routes.py
        from src.core.validation_runner import ValidationRunner

        @router.get("/validate")
        async def validate_service():
            base_url = "http://localhost:8888/api/my_service"
            runner = ValidationRunner(base_url=base_url, service_name="my_service")
            return await runner.run_all_tests()
    """

    def __init__(
        self,
        base_url: str,
        service_name: Optional[str] = None,
        timeout: int = 10
    ):
        """
        Initialize validation runner

        Args:
            base_url: Base URL for service endpoints (e.g., http://localhost:8888/api/session)
            service_name: Service name for logging (optional)
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip("/")
        self.service_name = service_name or "unknown"
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all validation tests

        Returns:
            Dict with test results and summary
        """
        logger.info(f"🧪 Running validation tests for {self.service_name}")

        # Reset results
        self.results = []

        # 1. Try to run pytest tests first
        pytest_results = await self.run_pytest_tests()

        # 2. Run basic health checks
        await self.test_health_endpoint()
        await self.test_service_discovery()

        # Calculate summary
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        total = len(self.results)

        summary = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "pytest_executed": pytest_results is not None
        }

        result = {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary,
            "tests": self.results
        }

        # Add pytest details if available
        if pytest_results:
            result["pytest"] = pytest_results

        return result

    async def run_pytest_tests(self) -> Optional[Dict[str, Any]]:
        """
        Run pytest tests for the service if they exist

        Returns:
            Dict with pytest results or None if no tests found
        """
        # Find service directory
        service_dir = Path(f"src/services/{self.service_name}")
        tests_dir = service_dir / "tests"

        # Check if tests exist
        if not tests_dir.exists():
            logger.info(f"ℹ️  No tests directory found for {self.service_name}")
            self._add_result(
                "Pytest Tests",
                True,
                f"No tests directory found (src/services/{self.service_name}/tests)",
                {"note": "This is OK - service may not need tests"}
            )
            return None

        # Check if directory has test files
        test_files = list(tests_dir.glob("test_*.py")) + list(tests_dir.glob("**/test_*.py"))
        if not test_files:
            logger.info(f"ℹ️  No test files found in {tests_dir}")
            self._add_result(
                "Pytest Tests",
                True,
                f"Tests directory exists but no test_*.py files found",
                {"note": "Consider adding tests for better validation"}
            )
            return None

        # Run pytest
        try:
            logger.info(f"🧪 Running pytest for {self.service_name}...")

            # Run pytest with JSON output
            cmd = [
                "python3", "-m", "pytest",
                str(tests_dir),
                "-v",
                "--tb=short",
                "--maxfail=5",  # Stop after 5 failures
                "-x"  # Stop on first failure for faster feedback
            ]

            # Execute pytest (run in background to not block)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                self._add_result(
                    "Pytest Tests",
                    False,
                    "Pytest timed out after 30 seconds",
                    {"timeout": 30}
                )
                return {"status": "timeout", "timeout_seconds": 30}

            # Parse output
            output = stdout.decode("utf-8")
            errors = stderr.decode("utf-8")

            # Check return code
            passed = process.returncode == 0

            # Extract summary from pytest output
            summary_info = self._parse_pytest_output(output)

            # Add result
            self._add_result(
                "Pytest Tests",
                passed,
                f"Pytest {'passed' if passed else 'failed'} - {summary_info.get('summary', 'No summary')}",
                {
                    "return_code": process.returncode,
                    "test_count": summary_info.get("total", 0),
                    "passed_count": summary_info.get("passed", 0),
                    "failed_count": summary_info.get("failed", 0)
                }
            )

            return {
                "status": "passed" if passed else "failed",
                "return_code": process.returncode,
                **summary_info,
                "output_preview": output[-500:] if output else ""  # Last 500 chars
            }

        except Exception as e:
            logger.error(f"❌ Error running pytest: {e}")
            self._add_result(
                "Pytest Tests",
                False,
                f"Error running pytest: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract summary"""
        import re

        # Look for summary line like "5 passed in 2.50s" or "3 passed, 2 failed in 5.00s"
        summary_pattern = r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error"
        matches = re.findall(summary_pattern, output)

        passed = 0
        failed = 0
        errors = 0

        for match in matches:
            if match[0]:  # passed
                passed = int(match[0])
            if match[1]:  # failed
                failed = int(match[1])
            if match[2]:  # errors
                errors = int(match[2])

        total = passed + failed + errors

        # Extract last line for summary
        lines = output.strip().split("\n")
        summary_line = lines[-1] if lines else "No output"

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "summary": summary_line
        }

    async def test_health_endpoint(self) -> Any:
        """Test if health endpoint is accessible"""
        test_name = "Health Endpoint"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._add_result(test_name, True, "Health endpoint OK", data)
                    else:
                        self._add_result(
                            test_name,
                            False,
                            f"Health endpoint returned {response.status}",
                            {"status_code": response.status}
                        )
        except Exception as e:
            self._add_result(test_name, False, f"Health endpoint error: {str(e)}")

    async def test_service_discovery(self) -> Any:
        """Test if service is registered in service discovery"""
        test_name = "Service Discovery"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:8888/discovery/service/{self.service_name}",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._add_result(
                            test_name,
                            True,
                            "Service registered in discovery",
                            {"service_id": data.get("service_id")}
                        )
                    else:
                        self._add_result(
                            test_name,
                            False,
                            f"Service not found in discovery (HTTP {response.status})"
                        )
        except Exception as e:
            self._add_result(test_name, False, f"Discovery check error: {str(e)}")

    def _add_result(
        self,
        test_name: str,
        passed: bool,
        message: str,
        data: Optional[Dict] = None
    ):
        """Add test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }

        if data:
            result["data"] = data

        self.results.append(result)

        # Log result
        if passed:
            logger.info(f"  ✅ {test_name}: {message}")
        else:
            logger.warning(f"  ❌ {test_name}: {message}")


class ExtendedValidationRunner(ValidationRunner):
    """
    Extended validation runner with additional common tests

    Services can extend this to add more specific tests
    """

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests including extended ones"""
        # Run base tests
        await super().run_all_tests()

        # Run extended tests
        await self.test_response_time()

        # Recalculate summary
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        total = len(self.results)

        return {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "passed": passed,
                "failed": failed,
                "total": total,
                "success_rate": (passed / total * 100) if total > 0 else 0
            },
            "tests": self.results
        }

    async def test_response_time(self) -> Any:
        """Test if health endpoint responds within acceptable time"""
        test_name = "Response Time"

        try:
            start = asyncio.get_event_loop().time()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    end = asyncio.get_event_loop().time()
                    response_time_ms = (end - start) * 1000

                    # Consider <500ms as good response time
                    if response_time_ms < 500:
                        self._add_result(
                            test_name,
                            True,
                            f"Response time OK ({response_time_ms:.0f}ms)",
                            {"response_time_ms": response_time_ms}
                        )
                    else:
                        self._add_result(
                            test_name,
                            False,
                            f"Slow response time ({response_time_ms:.0f}ms > 500ms)",
                            {"response_time_ms": response_time_ms}
                        )
        except Exception as e:
            self._add_result(test_name, False, f"Response time test error: {str(e)}")
