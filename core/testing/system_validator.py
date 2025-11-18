import os
#!/usr/bin/env python3
"""
System-Wide Validation Framework

Executa todos os validadores (validate.py) de todos os serviços e agrega os resultados.
Fornece relatório unificado com estatísticas detalhadas.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized ValidationManager
from src.lib.validation_manager import ValidationManager

from loguru import logger


class SystemValidator:
    """
    Valida todos os serviços do sistema executando seus ValidationRunners
    """

    def __init__(self, base_url: str = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")):
        """
        Initialize system validator

        Args:
            base_url: Base URL for Service Manager
        """
        self.base_url = base_url
        self.results: Dict[str, Any] = {}
        self.services_config = self._load_services_config()

    def _load_services_config(self) -> Dict[str, Any]:
        """Load services configuration from services_config.yaml"""
        import yaml

        config_path = project_root / "services_config.yaml"
        if not config_path.exists():
            logger.warning(f"services_config.yaml not found at {config_path}")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config.get("services", {})

    def _find_validators(self) -> List[Dict[str, str]]:
        """
        Find all services that might have validators

        Discovers services by:
        1. Services in services_config.yaml (with port)
        2. Services with tests/ directory (auto-discovery)

        Returns:
            List of dicts with service_name, module_path, port
        """
        validators = []
        validators_dict = {}  # To avoid duplicates
        services_dir = project_root / "src" / "services"

        if not services_dir.exists():
            logger.error(f"Services directory not found: {services_dir}")
            return []

        # Method 1: Get services from services_config.yaml
        for service_name, service_config in self.services_config.items():
            # Skip if no port (can't validate)
            port = service_config.get("port")
            if port is None:
                logger.debug(f"No port found for {service_name}, skipping")
                continue

            # Check if service directory exists
            service_dir = services_dir / service_name
            if not service_dir.exists():
                logger.debug(f"Service directory not found: {service_dir}, skipping")
                continue

            # Check for validate.py or assume HTTP endpoint exists
            validate_file = service_dir / "validate.py"
            has_validate_py = validate_file.exists()

            validators_dict[service_name] = {
                "service_name": service_name,
                "module_path": f"src.services.{service_name}.validate",
                "port": port,
                "type": service_config.get("type", "service"),
                "has_validate_py": has_validate_py
            }
            logger.debug(f"Found service {service_name} (port {port}, validate.py: {has_validate_py})")

        # Method 2: Auto-discover services with tests/ directory
        # This catches services not in services_config.yaml
        for service_dir in services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            service_name = service_dir.name

            # Skip if already found in config
            if service_name in validators_dict:
                continue

            # Check if has tests/ directory
            tests_dir = service_dir / "tests"
            if not tests_dir.exists():
                continue

            # Check if tests directory has test files
            test_files = list(tests_dir.rglob("test_*.py"))
            if not test_files:
                continue

            # Auto-discovered service (no port, will use ValidationManager only)
            validators_dict[service_name] = {
                "service_name": service_name,
                "module_path": f"src.services.{service_name}.validate",
                "port": 0,  # No port (ValidationManager doesn't need it)
                "type": "service",
                "has_validate_py": False
            }
            logger.debug(f"Auto-discovered service {service_name} (tests/ found, {len(test_files)} test files)")

        return list(validators_dict.values())

    async def _run_validator(self, validator_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Run a single service validator using communication_manager with Service Discovery

        Uses Service Discovery to find the service's actual port, then calls /validate endpoint.
        Each service executes its own tests using ValidationManager and returns results.

        Args:
            validator_info: Dict with service_name, module_path, port (port is fallback)

        Returns:
            Validation results with summary, passed/failed counts, etc.
        """
        import time
        service_name = validator_info["service_name"]
        fallback_port = validator_info.get("port", 0)

        logger.info(f"Validating {service_name}...")
        start_time = time.time()

        # Try Method 1: Call /validate via communication_manager with Service Discovery (PRIORITY)
        try:
            # Import communication manager (uses Service Discovery automatically)
            from src.core.managers.communication_manager import get_communication_manager

            comm_manager = get_communication_manager()
            await comm_manager.initialize()

            # Communication manager will:
            # 1. Try Service Discovery first (asks Service Manager where the service is)
            # 2. Falls back to default ports if discovery fails
            # 3. Handles internal vs external routing automatically
            response = await comm_manager.call_service(
                service_name=service_name,
                endpoint_path="/validate",
                method="GET",
                timeout=5,  # Reduced from 120s to 5s - fail fast if service not available
                enable_resilience=False  # Disable retry for validation (run once)
            )

            if response and isinstance(response, dict):
                results = response

                summary = results.get('summary', {})
                logger.success(
                    f"{service_name}: {summary.get('passed', 0)}/{summary.get('total', 0)} tests passed (via Service Discovery)"
                )

                elapsed_time = time.time() - start_time
                return {
                    "service": service_name,
                    "status": results.get("status", "completed"),
                    "type": validator_info["type"],
                    "method": "service_discovery",
                    "elapsed_time_seconds": round(elapsed_time, 2),
                    **results
                }
        except Exception as e:
            logger.debug(f"{service_name}: Service Discovery communication failed - {e}")

        # Try Method 2: Direct ValidationManager (fallback for services without HTTP)
        try:
            validation_manager = ValidationManager(service_name=service_name)

            if validation_manager.has_tests():
                results = validation_manager.run_pytest_tests(
                    verbose=False,
                    skip_slow=False,  # Run ALL tests including slow ones
                    max_failures=None,  # Don't stop on failures
                    timeout=120  # 2 minutes timeout per service
                )

                logger.success(
                    f"{service_name}: {results['summary']['passed']}/{results['summary']['total']} tests passed (ValidationManager direct)"
                )

                elapsed_time = time.time() - start_time
                return {
                    "service": service_name,
                    "status": results.get("status", "completed"),
                    "port": fallback_port,
                    "type": validator_info["type"],
                    "method": "validation_manager_direct",
                    "elapsed_time_seconds": round(elapsed_time, 2),
                    **results
                }
        except Exception as e:
            logger.debug(f"{service_name}: ValidationManager direct failed - {e}")

        # Method 3: Skip (no tests or service not available)
        logger.warning(f"{service_name}: No validation method available")
        elapsed_time = time.time() - start_time
        return {
            "service": service_name,
            "status": "skipped",
            "error": "Service not available or no tests found",
            "elapsed_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat()
        }

    async def validate_all(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Validate all services in the system

        Args:
            parallel: Run validations in parallel (default: True)

        Returns:
            Aggregated validation results
        """
        validators = self._find_validators()

        if not validators:
            logger.warning("No validators found")
            return {
                "status": "error",
                "error": "No validators found in services",
                "timestamp": datetime.now().isoformat()
            }

        logger.info(f"Found {len(validators)} validators")

        # Run validators in batches with INTELLIGENT LOAD BALANCING
        # Mix slow and fast services in each batch to minimize total time
        if parallel:
            logger.info("Running validations with INTELLIGENT BATCHING (mixing slow + fast)...")

            # Historical timing data (from previous runs) - in seconds
            service_times = {
                "scenarios": 150, "websocket": 150, "external_llm": 145, "session": 145,
                "external_stt": 140, "stt": 140, "database": 135, "orchestrator": 130,
                "webrtc": 75, "external_tts": 70, "conversation_store": 65,
                "file_storage": 65, "api_gateway": 65,
                "machine_manager": 20, "deployment": 20,
                "service_manager": 20, "metrics_testing": 20, "user": 20,
            }

            # Sort validators by estimated time (slow to fast)
            def get_time_estimate(validator_info):
                service_name = validator_info["service_name"]
                return service_times.get(service_name, 30)  # default 30s for unknown

            sorted_validators = sorted(validators, key=get_time_estimate, reverse=True)

            # Create balanced batches (6 services per batch, mixing slow+fast)
            # Strategy: Round-robin assignment to distribute load evenly
            batch_size = 6
            num_batches = (len(sorted_validators) + batch_size - 1) // batch_size
            batches = [[] for _ in range(num_batches)]

            # Distribute validators round-robin: batch 0, 1, 2, 3, 0, 1, 2, 3...
            for idx, validator in enumerate(sorted_validators):
                batch_idx = idx % num_batches
                batches[batch_idx].append(validator)

            logger.info(f"Created {num_batches} balanced batches:")
            for i, batch in enumerate(batches):
                batch_names = [v["service_name"] for v in batch]
                estimated_time = max([get_time_estimate(v) for v in batch])
                logger.info(f"  Batch {i+1}: {len(batch)} services, est. {estimated_time}s - {batch_names}")

            results = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{num_batches} ({len(batch)} services)...")

                tasks = [self._run_validator(v) for v in batch]
                try:
                    # Add timeout to prevent hanging (10 minutes per batch)
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=600
                    )
                    results.extend(batch_results)
                except asyncio.TimeoutError:
                    logger.error(f"Batch {i+1} timed out after 600 seconds")
                    # Add timeout errors for all tasks in batch
                    for v in batch:
                        results.append({
                            "service": v["service_name"],
                            "status": "error",
                            "error": "Batch timeout after 600 seconds"
                        })
        else:
            logger.info("Running validations sequentially...")
            results = []
            for v in validators:
                result = await self._run_validator(v)
                results.append(result)

        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "service": "unknown",
                    "status": "error",
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)

        # Calculate summary
        total_services = len(processed_results)
        # Count services with status "passed" or "failed" as completed
        completed = sum(1 for r in processed_results if r["status"] in ["passed", "failed", "completed"])
        errors = sum(1 for r in processed_results if r["status"] == "error")
        skipped = sum(1 for r in processed_results if r["status"] == "skipped")

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for result in processed_results:
            # Count tests from services that ran (passed, failed, or completed)
            if result["status"] in ["passed", "failed", "completed"] and "summary" in result:
                total_tests += result["summary"]["total"]
                total_passed += result["summary"]["passed"]
                total_failed += result["summary"]["failed"]

        overall_success_rate = (
            f"{(total_passed / total_tests * 100):.1f}%"
            if total_tests > 0
            else "0%"
        )

        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": total_services,
                "completed": completed,
                "errors": errors,
                "skipped": skipped,
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_success_rate": overall_success_rate,
                "all_services_passed": errors == 0 and total_failed == 0
            },
            "services": processed_results
        }

    def print_summary(self, results: Dict[str, Any]):
        """
        Print formatted summary of validation results

        Args:
            results: Validation results from validate_all()
        """
        summary = results.get("summary", {})
        services = results.get("services", [])

        print()
        print("=" * 80)
        print("🔍 SYSTEM-WIDE VALIDATION RESULTS")
        print("=" * 80)
        print()

        # Summary
        print(f"Total services:        {summary.get('total_services', 0)}")
        print(f"✅ Completed:          {summary.get('completed', 0)}")
        print(f"❌ Errors:             {summary.get('errors', 0)}")
        print(f"⏭️  Skipped:            {summary.get('skipped', 0)}")
        print()
        print(f"Total tests:           {summary.get('total_tests', 0)}")
        print(f"✅ Passed:             {summary.get('total_passed', 0)}")
        print(f"❌ Failed:             {summary.get('total_failed', 0)}")
        print(f"Overall success rate:  {summary.get('overall_success_rate', '0%')}")
        print()
        print("=" * 80)
        print("📋 INDIVIDUAL SERVICES")
        print("=" * 80)
        print()

        # Sort by status (completed/passed/failed first, then errors, then skipped)
        status_order = {"passed": 0, "completed": 0, "failed": 0, "error": 1, "skipped": 2}
        sorted_services = sorted(
            services,
            key=lambda s: (status_order.get(s["status"], 99), s["service"])
        )

        for service in sorted_services:
            service_name = service.get("service", "unknown")
            status = service.get("status", "unknown")

            if status in ["completed", "passed", "failed"]:
                svc_summary = service.get("summary", {})
                passed = svc_summary.get("passed", 0)
                total = svc_summary.get("total", 0)
                success_rate = svc_summary.get("success_rate", "0%")

                if passed == total:
                    icon = "✅"
                    status_text = f"PASSED - {passed}/{total} tests ({success_rate})"
                else:
                    icon = "⚠️ "
                    status_text = f"PARTIAL - {passed}/{total} tests ({success_rate})"

                print(f"{icon} {service_name:25s} - {status_text}")

                # Show failed tests if any
                if passed < total:
                    tests = service.get("tests", [])
                    failed_tests = [t for t in tests if not t.get("passed", False)]
                    for test in failed_tests[:3]:  # Show first 3 failures
                        test_name = test.get("test", "unknown")
                        message = test.get("message", "")
                        print(f"   └─ ❌ {test_name}: {message}")

            elif status == "error":
                icon = "❌"
                error = service.get("error", "Unknown error")
                print(f"{icon} {service_name:25s} - ERROR: {error}")

            elif status == "skipped":
                icon = "⏭️ "
                reason = service.get("error", "No reason provided")
                print(f"{icon} {service_name:25s} - SKIPPED: {reason}")

        print()
        print("=" * 80)

        # Performance Analysis - Show slowest services
        print("⏱️  PERFORMANCE ANALYSIS (SLOWEST SERVICES)")
        print("=" * 80)
        print()

        # Get services with elapsed_time and sort by time (descending)
        services_with_time = [s for s in services if s.get("elapsed_time_seconds")]
        slowest_services = sorted(
            services_with_time,
            key=lambda s: s.get("elapsed_time_seconds", 0),
            reverse=True
        )[:10]  # Top 10 slowest

        for i, service in enumerate(slowest_services, 1):
            elapsed = service.get("elapsed_time_seconds", 0)
            service_name = service.get("service", "unknown")
            svc_summary = service.get("summary", {})
            total_tests = svc_summary.get("total", 0)

            print(f"{i:2d}. {service_name:25s} - {elapsed:6.2f}s ({total_tests} tests)")

        print()
        print("=" * 80)

        # Final verdict
        if summary.get("all_services_passed", False):
            print()
            print("✅ ALL VALIDATIONS PASSED!")
            print()
        else:
            print()
            print("⚠️  SOME VALIDATIONS FAILED OR HAD ERRORS")
            print()

        print("=" * 80)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="System-wide validation - runs all service validators"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888"),
        help="Base URL for Service Manager (default: http://localhost:8888)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run validations sequentially instead of parallel"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to file"
    )

    args = parser.parse_args()

    # Create validator
    validator = SystemValidator(base_url=args.base_url)

    # Run validations
    results = await validator.validate_all(parallel=not args.sequential)

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        validator.print_summary(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        logger.success(f"Results saved to {output_path}")

    # Exit code based on success
    summary = results.get("summary", {})
    if summary.get("all_services_passed", False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
