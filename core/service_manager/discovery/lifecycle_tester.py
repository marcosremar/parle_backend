#!/usr/bin/env python3
"""
Service Lifecycle Test - 3-Cycle Validation

Tests the complete lifecycle of services through 3 full cycles:
- Cycle 1: Install → Validate → Uninstall
- Cycle 2: Reinstall → Validate → Uninstall
- Cycle 3: Reinstall → Validate (leave installed)

This verifies that services can be installed, validated, and uninstalled
reliably multiple times without issues.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .service_discovery import (
    ServiceDiscovery,
    ServiceType,
    ServiceStatus
)
from .bulk_installer import (
    BulkInstaller,
    InstallationStatus,
    InstallationResult
)


@dataclass
class CycleResult:
    """Result of one complete cycle (install/validate/uninstall)"""
    cycle_number: int
    install_result: InstallationResult
    validate_result: InstallationResult
    uninstall_result: InstallationResult
    total_duration: float
    success: bool


@dataclass
class ServiceLifecycleResult:
    """Complete lifecycle test result for one service"""
    service_id: str
    service_type: ServiceType
    cycles: List[CycleResult]
    total_duration: float
    all_cycles_success: bool
    error_messages: List[str]


class ServiceLifecycleTester:
    """
    Tests complete service lifecycle through multiple install/uninstall cycles.

    For AI model services (llm, stt, tts):
    - Tests full installation (including model downloads)
    - Tests validation (dependencies, health checks)
    - Tests uninstallation (model cleanup)

    For infrastructure services:
    - Tests dependency installation
    - Tests validation (imports, module checks)
    - Tests graceful handling (no uninstall needed)
    """

    def __init__(self, num_cycles: int = 3):
        """
        Initialize lifecycle tester.

        Args:
            num_cycles: Number of install/uninstall cycles to run (default: 3)
        """
        self.num_cycles = num_cycles
        self.installer = BulkInstaller()
        self.discovery = self.installer.discovery
        self.results: Dict[str, ServiceLifecycleResult] = {}

    def test_service_lifecycle(self, service_id: str) -> ServiceLifecycleResult:
        """
        Test complete lifecycle of a single service through all cycles.

        Args:
            service_id: Service identifier

        Returns:
            ServiceLifecycleResult with detailed results
        """
        service_info = self.discovery.services.get(service_id)
        if not service_info:
            raise ValueError(f"Service '{service_id}' not found")

        print(f"\n{'='*100}")
        print(f"🔄 LIFECYCLE TEST: {service_id}")
        print(f"   Type: {service_info.service_type.value}")
        print(f"   Cycles: {self.num_cycles}")
        print(f"{'='*100}")

        total_start = time.time()
        cycles = []
        error_messages = []

        for cycle_num in range(1, self.num_cycles + 1):
            print(f"\n{'-'*100}")
            print(f"🔄 CYCLE {cycle_num}/{self.num_cycles}")
            print(f"{'-'*100}")

            cycle_start = time.time()

            # Step 1: Install
            print(f"\n📦 [{cycle_num}.1] Installing {service_id}...")
            install_result = self.installer.install_service(service_id, force=True)
            self._print_result(install_result, "INSTALL")

            # Step 2: Validate
            print(f"\n✅ [{cycle_num}.2] Validating {service_id}...")
            validate_result = self.installer.validate_service(service_id)
            self._print_result(validate_result, "VALIDATE")

            # Step 3: Uninstall (except on last cycle)
            if cycle_num < self.num_cycles:
                print(f"\n🗑️  [{cycle_num}.3] Uninstalling {service_id}...")
                uninstall_result = self.installer.uninstall_service(service_id)
                self._print_result(uninstall_result, "UNINSTALL")
            else:
                print(f"\n✅ [{cycle_num}.3] Skipping uninstall (final cycle - leaving installed)")
                uninstall_result = InstallationResult(
                    service_id=service_id,
                    status=InstallationStatus.SKIPPED,
                    message="Final cycle - left installed"
                )

            cycle_duration = time.time() - cycle_start

            # Determine cycle success
            cycle_success = (
                install_result.status in [InstallationStatus.SUCCESS, InstallationStatus.SKIPPED] and
                validate_result.status == InstallationStatus.SUCCESS
            )

            # Collect errors
            if install_result.error:
                error_messages.append(f"Cycle {cycle_num} Install: {install_result.error}")
            if validate_result.error:
                error_messages.append(f"Cycle {cycle_num} Validate: {validate_result.error}")
            if uninstall_result.error:
                error_messages.append(f"Cycle {cycle_num} Uninstall: {uninstall_result.error}")

            # Store cycle result
            cycle_result = CycleResult(
                cycle_number=cycle_num,
                install_result=install_result,
                validate_result=validate_result,
                uninstall_result=uninstall_result,
                total_duration=cycle_duration,
                success=cycle_success
            )
            cycles.append(cycle_result)

            # Print cycle summary
            self._print_cycle_summary(cycle_result)

        total_duration = time.time() - total_start
        all_cycles_success = all(c.success for c in cycles)

        # Create final result
        result = ServiceLifecycleResult(
            service_id=service_id,
            service_type=service_info.service_type,
            cycles=cycles,
            total_duration=total_duration,
            all_cycles_success=all_cycles_success,
            error_messages=error_messages
        )

        self.results[service_id] = result

        # Print service summary
        self._print_service_summary(result)

        return result

    def test_all_services(self, service_types: List[ServiceType] = None) -> Dict[str, ServiceLifecycleResult]:
        """
        Test lifecycle of all services (or filtered by type).

        Args:
            service_types: Optional list of service types to test (default: all)

        Returns:
            Dictionary mapping service_id to ServiceLifecycleResult
        """
        print(f"\n{'='*100}")
        print(f"🚀 BULK LIFECYCLE TEST - ALL SERVICES")
        print(f"   Cycles per service: {self.num_cycles}")
        print(f"{'='*100}")

        total_start = time.time()

        # Get services to test
        if service_types:
            services_to_test = []
            for service_type in service_types:
                services_to_test.extend(self.discovery.get_services_by_type(service_type))
        else:
            services_to_test = list(self.discovery.services.values())

        # Sort by service ID
        services_to_test = sorted(services_to_test, key=lambda s: s.service_id)

        print(f"\n📊 Testing {len(services_to_test)} services:")
        for service in services_to_test:
            print(f"   - {service.service_id} ({service.service_type.value})")

        # Test each service
        for service_info in services_to_test:
            try:
                self.test_service_lifecycle(service_info.service_id)
            except Exception as e:
                print(f"\n❌ ERROR testing {service_info.service_id}: {e}")
                # Create failed result
                self.results[service_info.service_id] = ServiceLifecycleResult(
                    service_id=service_info.service_id,
                    service_type=service_info.service_type,
                    cycles=[],
                    total_duration=0,
                    all_cycles_success=False,
                    error_messages=[str(e)]
                )

        total_duration = time.time() - total_start

        # Print final summary
        self._print_final_summary(total_duration)

        return self.results

    def export_results(self, output_file: Path = None):
        """
        Export test results to JSON file.

        Args:
            output_file: Path to output file (default: lifecycle_test_results.json)
        """
        if output_file is None:
            output_file = Path(str(Path.home() / ".cache" / "ultravox-pipeline" / "service_lifecycle_results.json")

        # Convert results to dict
        results_dict = {}
        for service_id, result in self.results.items():
            results_dict[service_id] = {
                "service_type": result.service_type.value,
                "all_cycles_success": result.all_cycles_success,
                "total_duration": result.total_duration,
                "num_cycles": len(result.cycles),
                "error_messages": result.error_messages,
                "cycles": [
                    {
                        "cycle_number": c.cycle_number,
                        "success": c.success,
                        "duration": c.total_duration,
                        "install_status": c.install_result.status.value,
                        "validate_status": c.validate_result.status.value,
                        "uninstall_status": c.uninstall_result.status.value,
                    }
                    for c in result.cycles
                ]
            }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n📄 Results exported to: {output_file}")

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _print_result(self, result: InstallationResult, operation: str):
        """Print operation result"""
        icon = "✅" if result.status == InstallationStatus.SUCCESS else \
               "⏭️" if result.status == InstallationStatus.SKIPPED else "❌"
        print(f"   {icon} {operation}: {result.message} ({result.duration_seconds:.1f}s)")

    def _print_cycle_summary(self, cycle: CycleResult):
        """Print summary of one cycle"""
        icon = "✅" if cycle.success else "❌"
        print(f"\n{icon} CYCLE {cycle.cycle_number} SUMMARY:")
        print(f"   Duration: {cycle.total_duration:.1f}s")
        print(f"   Install:   {cycle.install_result.status.value}")
        print(f"   Validate:  {cycle.validate_result.status.value}")
        print(f"   Uninstall: {cycle.uninstall_result.status.value}")
        print(f"   Result:    {'SUCCESS' if cycle.success else 'FAILED'}")

    def _print_service_summary(self, result: ServiceLifecycleResult):
        """Print summary of complete service lifecycle test"""
        icon = "✅" if result.all_cycles_success else "❌"
        print(f"\n{'='*100}")
        print(f"{icon} LIFECYCLE TEST SUMMARY: {result.service_id}")
        print(f"{'='*100}")
        print(f"Service Type:      {result.service_type.value}")
        print(f"Cycles Completed:  {len(result.cycles)}")
        print(f"All Cycles Success: {result.all_cycles_success}")
        print(f"Total Duration:    {result.total_duration:.1f}s ({result.total_duration/60:.1f} min)")

        if result.error_messages:
            print(f"\n⚠️  Errors:")
            for error in result.error_messages:
                print(f"   - {error}")

        print(f"{'='*100}\n")

    def _print_final_summary(self, total_duration: float):
        """Print final summary of all tests"""
        print(f"\n{'='*100}")
        print(f"📊 FINAL LIFECYCLE TEST SUMMARY")
        print(f"{'='*100}")

        total_services = len(self.results)
        successful_services = sum(1 for r in self.results.values() if r.all_cycles_success)
        failed_services = total_services - successful_services

        print(f"\n📦 Services Tested: {total_services}")
        print(f"   ✅ Success: {successful_services}")
        print(f"   ❌ Failed:  {failed_services}")

        print(f"\n⏱️  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

        if failed_services > 0:
            print(f"\n❌ Failed Services:")
            for service_id, result in self.results.items():
                if not result.all_cycles_success:
                    print(f"   - {service_id} ({result.service_type.value})")
                    for error in result.error_messages:
                        print(f"      • {error}")

        print(f"\n{'='*100}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Service Lifecycle Tester (3-Cycle Install/Validate/Uninstall)")
    parser.add_argument("--service", help="Test specific service")
    parser.add_argument("--type", choices=["ai_model", "infrastructure", "utility", "external", "test"],
                       help="Test all services of specific type")
    parser.add_argument("--all", action="store_true", help="Test ALL services (WARNING: AI models take hours!)")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run (default: 3)")
    parser.add_argument("--export", help="Export results to JSON file")
    parser.add_argument("--skip-ai-models", action="store_true", help="Skip AI model services (llm, stt, tts)")

    args = parser.parse_args()

    # Create tester
    tester = ServiceLifecycleTester(num_cycles=args.cycles)

    # Run tests
    if args.service:
        # Test single service
        result = tester.test_service_lifecycle(args.service)
        if args.export:
            tester.export_results(Path(args.export))

    elif args.type:
        # Test all services of specific type
        service_type = ServiceType(args.type)
        results = tester.test_all_services(service_types=[service_type])
        if args.export:
            tester.export_results(Path(args.export))

    elif args.all:
        # Test ALL services
        if args.skip_ai_models:
            # Skip AI models
            service_types = [ServiceType.INFRASTRUCTURE, ServiceType.UTILITY, ServiceType.EXTERNAL, ServiceType.TEST]
            results = tester.test_all_services(service_types=service_types)
        else:
            results = tester.test_all_services()

        if args.export:
            tester.export_results(Path(args.export))

    else:
        # Default: Test infrastructure services only (fast test)
        print("\n⚡ Running QUICK TEST (infrastructure services only)")
        print("   Use --all to test ALL services (WARNING: takes hours!)")
        print("   Use --service <name> to test specific service")
        print("   Use --type <type> to test all services of specific type")

        results = tester.test_all_services(service_types=[ServiceType.INFRASTRUCTURE])

        if args.export:
            tester.export_results(Path(args.export))
        else:
            tester.export_results()  # Auto-export to /tmp
