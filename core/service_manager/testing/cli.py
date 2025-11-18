"""
CLI Commands for Service Testing

Provides command-line interface for testing and validating services.
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .service_tester import ServiceTester
from .validator import ServiceValidator


def run_test(service_name: str, mode: str = "auto", level: str = "standard", confirm_real: bool = False) -> int:
    """
    Test a single service

    Args:
        service_name: Name of the service to test
        mode: Execution mode (auto, module, internal, both)
        level: Test level (unit, integration, standard, real, expensive, all)
        confirm_real: Confirm running expensive tests

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    # Map level to pytest marker
    marker_map = {
        "unit": "unit",
        "integration": "integration",
        "standard": "not real and not expensive",
        "real": "real",
        "expensive": "expensive",
        "all": ""  # No filter, run everything
    }

    # Check if expensive/all tests require confirmation
    if level in ["expensive", "all"] and not confirm_real:
        print("❌ Expensive/all tests require --confirm-real flag")
        print("   These tests may cost money (RunPod, VastAI, GPU clusters)")
        print(f"   Usage: ./main.sh test {service_name} --level {level} --confirm-real")
        return 1

    marker = marker_map.get(level, "not real and not expensive")

    # Display test level info
    level_info = {
        "unit": "🔬 Unit tests only (fast, isolated)",
        "integration": "🔗 Integration tests with mocks",
        "standard": "✅ Standard tests (unit + integration, no real/expensive)",
        "real": "🌐 Real tests (dry-run validation with external APIs)",
        "expensive": "💰 Expensive tests (WILL COST MONEY - clusters, GPUs)",
        "all": "⚠️  ALL tests (including expensive!)"
    }

    print(f"\n🧪 Testing service: {service_name}")
    print(f"📊 Level: {level_info.get(level, level)}")
    print("━" * 80)

    try:
        tester = ServiceTester(service_name=service_name, mode=mode, pytest_args=f"-m '{marker}'" if marker else "")
        report = tester.run_all_tests()

        # Print report using the returned report
        tester.report = report  # Update tester's report
        print(tester.format_report_text())

        # Save JSON report
        report_file = Path(fstr(Path.home() / ".cache" / "ultravox-pipeline" / "test_report_{service_name}.json")
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\n📄 JSON report saved to: {report_file}")

        # Return exit code
        if report.summary["failed"] > 0:
            return 1
        else:
            return 0

    except Exception as e:
        print(f"\n❌ Error testing service: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_test_all(profile: Optional[str] = None, level: str = "standard", confirm_real: bool = False) -> int:
    """
    Test all services IN PARALLEL

    Args:
        profile: Profile name (optional, uses active profile from config)
        level: Test level (unit, integration, standard, real, expensive, all)
        confirm_real: Confirm running expensive tests

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    import concurrent.futures
    from datetime import datetime

    # Map level to pytest marker
    marker_map = {
        "unit": "unit",
        "integration": "integration",
        "standard": "not real and not expensive",
        "real": "real",
        "expensive": "expensive",
        "all": ""  # No filter
    }

    # Check confirmation for expensive tests
    if level in ["expensive", "all"] and not confirm_real:
        print("❌ Expensive/all tests require --confirm-real flag")
        print("   These tests may cost money!")
        print(f"   Usage: ./main.sh test --all --level {level} --confirm-real")
        return 1

    marker = marker_map.get(level, "not real and not expensive")

    level_info = {
        "unit": "🔬 Unit tests only",
        "integration": "🔗 Integration tests with mocks",
        "standard": "✅ Standard (unit + integration)",
        "real": "🌐 Real tests (dry-run)",
        "expensive": "💰 Expensive tests (COSTS MONEY!)",
        "all": "⚠️  ALL tests (including expensive!)"
    }

    print(f"\n🚀 Testing all services (PARALLEL MODE)")
    print(f"📊 Level: {level_info.get(level, level)}")
    print("━" * 80)
    print()

    # Get list of services
    services_dir = project_root / "src" / "services"

    if not services_dir.exists():
        print(f"❌ Services directory not found: {services_dir}")
        return 1

    all_services = sorted([d.name for d in services_dir.iterdir() if d.is_dir()])

    # Filter by profile using ProfileManager
    try:
        # Import ProfileManager
        tests_dir = project_root / "tests"
        sys.path.insert(0, str(tests_dir.parent))
        from tests.profile_manager import (
            get_active_profile,
            is_service_enabled,
            get_disabled_reason,
        )

        active_profile = get_active_profile()
        print(f"📋 Active profile: {active_profile}")
        print()

        # Filter services based on profile
        services = []
        skipped = []
        for service in all_services:
            if is_service_enabled(service):
                services.append(service)
            else:
                reason = get_disabled_reason(service) or "Not in active profile"
                skipped.append((service, reason))

        # Show skipped services
        if skipped:
            print(f"⊘ Skipped {len(skipped)} services (disabled in {active_profile} profile):")
            for service, reason in skipped:
                print(f"   ⊘ {service:25s} - {reason}")
            print()

        print(f"✅ Testing {len(services)} enabled services")

    except Exception as e:
        print(f"⚠️  Could not load profile configuration: {e}")
        print(f"📋 Testing all services ({len(all_services)} services)")
        services = all_services

    print(f"⏱️  Running {len(services)} tests concurrently (max 8 workers)...")
    print()

    start_time = datetime.now()

    def test_service(service_name):
        """Test a single service (runs in parallel)"""
        try:
            # Set environment variable to indicate parallel mode
            import os
            os.environ["PARALLEL_TEST_MODE"] = "1"

            pytest_args = f"-m '{marker}'" if marker else ""
            tester = ServiceTester(service_name=service_name, mode="auto", pytest_args=pytest_args)
            report = tester.run_all_tests()
            return {
                "service": service_name,
                "success": True,
                "report": report.to_dict()
            }
        except Exception as e:
            return {
                "service": service_name,
                "success": False,
                "error": str(e)
            }

    # Test services in parallel
    results = {}
    passed = 0
    failed = 0
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_service = {executor.submit(test_service, svc): svc for svc in services}

        for future in concurrent.futures.as_completed(future_to_service, timeout=600):
            try:
                result = future.result(timeout=300)  # 5 minutes per service
                svc = result["service"]
                completed += 1

                if result["success"]:
                    report = result["report"]
                    results[svc] = report

                    if report["summary"]["failed"] > 0:
                        print(f"[{completed}/{len(services)}] ❌ {svc}: {report['success_rate']:.0f}% ({report['summary']['passed']}/{report['summary']['failed']}/{report['summary']['skipped']})")
                        failed += 1
                    else:
                        print(f"[{completed}/{len(services)}] ✅ {svc}: {report['success_rate']:.0f}% ({report['summary']['passed']}/{report['summary']['failed']}/{report['summary']['skipped']})")
                        passed += 1
                else:
                    print(f"[{completed}/{len(services)}] ❌ {svc}: ERROR")
                    failed += 1
                    results[svc] = {"error": result["error"]}
            except concurrent.futures.TimeoutError:
                svc = future_to_service[future]
                print(f"[{completed+1}/{len(services)}] ⏱️  {svc}: TIMEOUT (exceeded 5 minutes)")
                failed += 1
                completed += 1
                results[svc] = {"error": "Test timeout after 300 seconds"}

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary
    print()
    print("━" * 80)
    print("📊 PARALLEL TEST SUMMARY")
    print("━" * 80)
    print(f"Services tested: {len(services)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    success_rate = (passed / len(services)) * 100 if services else 0
    print(f"Success rate: {success_rate:.1f}%")
    print(f"⏱️  Total time: {duration:.1f}s (parallel)")
    print()

    # Save full report
    report_file = Path(str(Path.home() / ".cache" / "ultravox-pipeline" / "test_all_services.json")
    with open(report_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": len(services),
                    "passed": passed,
                    "failed": failed,
                    "success_rate": success_rate,
                    "duration_seconds": duration,
                },
                "services": results,
            },
            f,
            indent=2,
        )

    print(f"📄 Full report saved to: {report_file}")
    print("=" * 80)

    return 0 if failed == 0 else 1


def run_validate(service_name: str) -> int:
    """
    Validate a service installation

    Args:
        service_name: Name of the service to validate

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print(f"\n🔍 Validating service: {service_name}")
    print("━" * 80)
    print()

    try:
        validator = ServiceValidator(service_name=service_name)
        results = validator.validate_all()

        # Print results
        print(validator.format_text(results))

        # Save JSON report
        report_file = Path(fstr(Path.home() / ".cache" / "ultravox-pipeline" / "validate_{service_name}.json")
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 JSON report saved to: {report_file}")

        # Return exit code
        if results["overall_status"] == "failed":
            return 1
        else:
            return 0

    except Exception as e:
        print(f"\n❌ Error validating service: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.core.service_manager.testing.cli test <service_name> [options]")
        print("  python -m src.core.service_manager.testing.cli test_all [options]")
        print("  python -m src.core.service_manager.testing.cli validate <service_name>")
        print("")
        print("Options:")
        print("  --level <level>     Test level: unit, integration, standard, real, expensive, all")
        print("  --confirm-real      Confirm running expensive tests (required for expensive/all)")
        print("")
        print("Examples:")
        print("  ./main.sh test skypilot                           # Standard tests (default)")
        print("  ./main.sh test skypilot --level unit              # Unit tests only")
        print("  ./main.sh test skypilot --level real              # Real dry-run tests")
        print("  ./main.sh test skypilot --level expensive --confirm-real  # Expensive tests")
        print("  ./main.sh test --all                              # All services (standard)")
        print("  ./main.sh test --all --level real                 # All services (real tests)")
        sys.exit(1)

    command = sys.argv[1]

    # Parse common options
    level = "standard"
    confirm_real = False

    for i, arg in enumerate(sys.argv):
        if arg == "--level" and i + 1 < len(sys.argv):
            level = sys.argv[i + 1]
        elif arg == "--confirm-real":
            confirm_real = True

    if command == "test":
        if len(sys.argv) < 3 or sys.argv[2].startswith("--"):
            print("Error: service_name required")
            sys.exit(1)

        service_name = sys.argv[2]
        mode = "auto"

        # Check for mode (backward compatibility)
        for arg in sys.argv[3:]:
            if not arg.startswith("--") and arg not in ["unit", "integration", "standard", "real", "expensive", "all"]:
                mode = arg
                break

        exit_code = run_test(service_name, mode, level, confirm_real)
        sys.exit(exit_code)

    elif command == "test_all":
        profile = None
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                profile = arg
                break

        exit_code = run_test_all(profile, level, confirm_real)
        sys.exit(exit_code)

    elif command == "validate":
        if len(sys.argv) < 3:
            print("Error: service_name required")
            sys.exit(1)

        service_name = sys.argv[2]
        exit_code = run_validate(service_name)
        sys.exit(exit_code)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
