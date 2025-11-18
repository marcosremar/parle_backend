"""
Service Validation Framework

Provides comprehensive validation for services including:
- Health checks
- Endpoint validation
- Performance tests
- Resource monitoring
- Integration tests
"""

import asyncio
import time
import httpx
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation test status"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of a validation test"""

    test_name: str
    status: ValidationStatus
    message: str
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a service"""

    service_name: str
    timestamp: str
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    total_duration_ms: float = 0.0

    def add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == ValidationStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == ValidationStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == ValidationStatus.SKIPPED),
            "warning": sum(1 for r in self.results if r.status == ValidationStatus.WARNING),
        }

    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.summary["total"] == 0:
            return 0.0
        return (self.summary["passed"] / self.summary["total"]) * 100


class ServiceValidator:
    """
    Service validation framework

    Usage:
        validator = ServiceValidator(
            service_name="llm",
            base_url="http://localhost:8100",
            timeout=30.0
        )

        report = await validator.run_all_validations()
        print(f"Success rate: {report.success_rate():.1f}%")
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout: float = 30.0,
        enable_resource_monitoring: bool = False,
    ):
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enable_resource_monitoring = enable_resource_monitoring

        self.report = ValidationReport(
            service_name=service_name, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ==========================================================================
    # Test Execution
    # ==========================================================================

    async def run_test(
        self, test_name: str, test_func: Callable, *args, **kwargs
    ) -> ValidationResult:
        """
        Run a single validation test

        Args:
            test_name: Name of the test
            test_func: Async function to execute
            *args, **kwargs: Arguments for test_func

        Returns:
            ValidationResult
        """
        start_time = time.time()

        try:
            logger.info(f"Running test: {test_name}")

            # Execute test
            result = await test_func(*args, **kwargs)

            duration_ms = (time.time() - start_time) * 1000

            # Test passed
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.PASSED,
                message="Test passed",
                duration_ms=duration_ms,
                metadata=result if isinstance(result, dict) else {},
            )

        except AssertionError as e:
            # Test failed (assertion)
            duration_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAILED,
                message="Assertion failed",
                duration_ms=duration_ms,
                error=str(e),
            )

        except Exception as e:
            # Test failed (exception)
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Test failed: {test_name} - {e}")
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAILED,
                message="Exception raised",
                duration_ms=duration_ms,
                error=str(e),
            )

    # ==========================================================================
    # Health Checks
    # ==========================================================================

    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test /health endpoint"""
        response = await self.client.get(f"{self.base_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "status" in data, "Missing 'status' field in health response"
        assert data["status"] in [
            "healthy",
            "ok",
            "up",
        ], f"Unexpected status: {data['status']}"

        return data

    async def test_info_endpoint(self) -> Dict[str, Any]:
        """Test /info endpoint"""
        response = await self.client.get(f"{self.base_url}/info")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "service" in data or "name" in data, "Missing service name in info"

        return data

    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test /metrics endpoint"""
        response = await self.client.get(f"{self.base_url}/metrics")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        return data

    # ==========================================================================
    # Performance Tests
    # ==========================================================================

    async def test_response_time(self, max_ms: float = 1000.0) -> Dict[str, Any]:
        """Test health endpoint response time"""
        start = time.time()
        response = await self.client.get(f"{self.base_url}/health")
        duration_ms = (time.time() - start) * 1000

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert (
            duration_ms < max_ms
        ), f"Response time {duration_ms:.1f}ms exceeds limit {max_ms}ms"

        return {"response_time_ms": duration_ms, "limit_ms": max_ms}

    async def test_concurrent_requests(
        self, num_requests: int = 10
    ) -> Dict[str, Any]:
        """Test concurrent request handling"""

        async def single_request():
            start = time.time()
            response = await self.client.get(f"{self.base_url}/health")
            duration_ms = (time.time() - start) * 1000
            return {
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }

        # Send concurrent requests
        tasks = [single_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        # Calculate statistics
        durations = [r["duration_ms"] for r in results]
        success_count = sum(1 for r in results if r["status_code"] == 200)

        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        assert success_count == num_requests, f"Only {success_count}/{num_requests} succeeded"

        return {
            "num_requests": num_requests,
            "success_rate": (success_count / num_requests) * 100,
            "avg_duration_ms": avg_duration,
            "max_duration_ms": max_duration,
            "min_duration_ms": min_duration,
        }

    # ==========================================================================
    # Resource Monitoring
    # ==========================================================================

    async def test_memory_usage(self, max_mb: float = 1000.0) -> Dict[str, Any]:
        """Test service memory usage"""
        if not self.enable_resource_monitoring:
            raise AssertionError("Resource monitoring not enabled")

        # Get current process (simplified - would need PID tracking)
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        return {
            "memory_mb": memory_mb,
            "limit_mb": max_mb,
        }

    async def test_cpu_usage(self, max_percent: float = 80.0) -> Dict[str, Any]:
        """Test service CPU usage"""
        if not self.enable_resource_monitoring:
            raise AssertionError("Resource monitoring not enabled")

        cpu_percent = psutil.cpu_percent(interval=1.0)

        assert (
            cpu_percent < max_percent
        ), f"CPU usage {cpu_percent:.1f}% exceeds limit {max_percent}%"

        return {
            "cpu_percent": cpu_percent,
            "limit_percent": max_percent,
        }

    # ==========================================================================
    # Custom Tests
    # ==========================================================================

    async def test_custom_endpoint(
        self, endpoint: str, method: str = "GET", expected_status: int = 200, **kwargs
    ) -> Dict[str, Any]:
        """Test custom endpoint"""
        url = f"{self.base_url}{endpoint}"

        if method == "GET":
            response = await self.client.get(url, **kwargs)
        elif method == "POST":
            response = await self.client.post(url, **kwargs)
        elif method == "PUT":
            response = await self.client.put(url, **kwargs)
        elif method == "DELETE":
            response = await self.client.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        assert (
            response.status_code == expected_status
        ), f"Expected {expected_status}, got {response.status_code}"

        return {"status_code": response.status_code, "response": response.json()}

    # ==========================================================================
    # Full Validation Suite
    # ==========================================================================

    async def run_all_validations(
        self, include_performance: bool = True, include_resource: bool = False
    ) -> ValidationReport:
        """
        Run all validation tests

        Args:
            include_performance: Include performance tests
            include_resource: Include resource monitoring tests

        Returns:
            ValidationReport with all results
        """
        logger.info(f"Starting validation for service: {self.service_name}")
        start_time = time.time()

        # Health checks
        result = await self.run_test("health_endpoint", self.test_health_endpoint)
        self.report.add_result(result)

        result = await self.run_test("info_endpoint", self.test_info_endpoint)
        self.report.add_result(result)

        result = await self.run_test("metrics_endpoint", self.test_metrics_endpoint)
        self.report.add_result(result)

        # Performance tests
        if include_performance:
            result = await self.run_test(
                "response_time", self.test_response_time, max_ms=1000.0
            )
            self.report.add_result(result)

            result = await self.run_test(
                "concurrent_requests", self.test_concurrent_requests, num_requests=10
            )
            self.report.add_result(result)

        # Resource tests
        if include_resource:
            result = await self.run_test(
                "memory_usage", self.test_memory_usage, max_mb=1000.0
            )
            self.report.add_result(result)

            result = await self.run_test(
                "cpu_usage", self.test_cpu_usage, max_percent=80.0
            )
            self.report.add_result(result)

        # Calculate summary
        self.report.total_duration_ms = (time.time() - start_time) * 1000
        self.report.calculate_summary()

        logger.info(
            f"Validation complete: {self.report.success_rate():.1f}% success rate"
        )

        return self.report

    # ==========================================================================
    # Report Formatting
    # ==========================================================================

    def format_report_text(self) -> str:
        """Format report as text"""
        lines = []
        lines.append(f"Service: {self.report.service_name}")
        lines.append(f"Timestamp: {self.report.timestamp}")
        lines.append(f"Total Duration: {self.report.total_duration_ms:.1f}ms")
        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Total: {self.report.summary['total']}")
        lines.append(f"  Passed: {self.report.summary['passed']}")
        lines.append(f"  Failed: {self.report.summary['failed']}")
        lines.append(f"  Skipped: {self.report.summary['skipped']}")
        lines.append(f"  Success Rate: {self.report.success_rate():.1f}%")
        lines.append("")
        lines.append("Results:")

        for result in self.report.results:
            status_icon = {
                ValidationStatus.PASSED: "✅",
                ValidationStatus.FAILED: "❌",
                ValidationStatus.SKIPPED: "⊘",
                ValidationStatus.WARNING: "⚠️",
            }[result.status]

            lines.append(
                f"  {status_icon} {result.test_name} ({result.duration_ms:.1f}ms)"
            )
            if result.error:
                lines.append(f"     Error: {result.error}")

        return "\n".join(lines)

    def format_report_markdown(self) -> str:
        """Format report as markdown"""
        lines = []
        lines.append(f"# Service Validation Report: {self.report.service_name}")
        lines.append("")
        lines.append(f"**Timestamp**: {self.report.timestamp}")
        lines.append(f"**Total Duration**: {self.report.total_duration_ms:.1f}ms")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tests | {self.report.summary['total']} |")
        lines.append(f"| Passed | {self.report.summary['passed']} |")
        lines.append(f"| Failed | {self.report.summary['failed']} |")
        lines.append(f"| Skipped | {self.report.summary['skipped']} |")
        lines.append(f"| Success Rate | {self.report.success_rate():.1f}% |")
        lines.append("")
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| Test | Status | Duration | Details |")
        lines.append("|------|--------|----------|---------|")

        for result in self.report.results:
            status_icon = {
                ValidationStatus.PASSED: "✅",
                ValidationStatus.FAILED: "❌",
                ValidationStatus.SKIPPED: "⊘",
                ValidationStatus.WARNING: "⚠️",
            }[result.status]

            error = result.error[:50] if result.error else "-"
            lines.append(
                f"| {result.test_name} | {status_icon} | {result.duration_ms:.1f}ms | {error} |"
            )

        return "\n".join(lines)


# ==============================================================================
# Convenience Functions
# ==============================================================================


async def validate_service(
    service_name: str,
    base_url: str,
    include_performance: bool = True,
    include_resource: bool = False,
    timeout: float = 30.0,
) -> ValidationReport:
    """
    Convenience function to validate a service

    Args:
        service_name: Service name
        base_url: Service base URL
        include_performance: Include performance tests
        include_resource: Include resource tests
        timeout: Request timeout

    Returns:
        ValidationReport
    """
    async with ServiceValidator(
        service_name=service_name,
        base_url=base_url,
        timeout=timeout,
        enable_resource_monitoring=include_resource,
    ) as validator:
        report = await validator.run_all_validations(
            include_performance=include_performance, include_resource=include_resource
        )
        return report
