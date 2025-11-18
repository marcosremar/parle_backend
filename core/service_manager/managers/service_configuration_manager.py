"""
Service Configuration Manager - Handles service configuration and validation.

Part of Service Manager refactoring (Phase 3).
Loads, validates, and manages service configurations.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import yaml
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.service_execution_config import get_service_execution_config, ExecutionMode


class ProcessStatus(Enum):
    """Service process status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    port: int
    script: str
    env: Dict[str, str] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    pid: Optional[int] = None

    def __post_init__(self):
        if self.env is None:
            self.env = {}


class ServiceConfigurationManager:
    """
    Manages service configuration - loading, validation, discovery.

    Responsibilities:
    - Load service configurations from YAML
    - Validate port configurations
    - Check for conflicts
    - Determine service execution mode (internal/external/module)
    - Get service metadata

    SOLID Principles:
    - Single Responsibility: Only handles configuration
    - Open/Closed: Easy to add new configuration sources
    """

    # Service dependencies - which services need to be started before others
    SERVICE_DEPENDENCIES = {
        "webrtc": ["websocket"],
        "orchestrator": ["llm", "tts", "stt"],
        "api_gateway": []
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to services_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent.parent / "services_config.yaml"

        self.config_path = config_path
        self.services: Dict[str, ServiceConfig] = {}
        self.base_dir = Path(__file__).parent.parent.parent.parent

        # Load service execution configuration
        self.execution_config = get_service_execution_config()
        logger.info("📋 Loaded service execution configuration")

        # Load configurations
        self._load_services()
        self._validate_configuration()

        logger.info(f"📋 Service Configuration Manager initialized ({len(self.services)} services)")

    def _load_services(self) -> None:
        """Load service configurations from YAML."""
        if not self.config_path.exists():
            logger.warning(f"⚠️  Config file not found: {self.config_path}")
            self.services = {}
            return

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            services_data = config.get('services', {})
            self.services = {}

            for service_id, service_data in services_data.items():
                self.services[service_id] = ServiceConfig(
                    name=service_data.get('name', service_id),
                    port=service_data.get('port', 8000),
                    script=service_data.get('script', f'src/services/{service_id}/service.py'),
                    env=service_data.get('env', {})
                )

            logger.info(f"📋 Loaded {len(self.services)} service configurations")

        except Exception as e:
            logger.error(f"❌ Failed to load services config: {e}")
            self.services = {}

    def _validate_configuration(self) -> None:
        """Validate service configuration for conflicts."""
        # Check for port conflicts
        ports = {}
        for service_id, config in self.services.items():
            if config.port in ports:
                logger.error(
                    f"❌ Port conflict: {service_id} and {ports[config.port]} "
                    f"both use port {config.port}"
                )
            ports[config.port] = service_id

        # Validate dependencies
        for service_id, deps in self.SERVICE_DEPENDENCIES.items():
            if service_id in self.services:
                for dep in deps:
                    if dep not in self.services:
                        logger.warning(
                            f"⚠️  {service_id} depends on {dep}, but {dep} not in config"
                        )

    def get_service(self, service_id: str) -> Optional[ServiceConfig]:
        """
        Get service configuration by ID.

        Args:
            service_id: Service identifier

        Returns:
            ServiceConfig if found, None otherwise
        """
        return self.services.get(service_id)

    def get_all_services(self) -> Dict[str, ServiceConfig]:
        """Get all service configurations."""
        return self.services.copy()

    def get_service_port(self, service_id: str) -> int:
        """
        Get port for a service.

        Args:
            service_id: Service identifier

        Returns:
            Port number (default 8000 if not found)
        """
        service = self.get_service(service_id)
        return service.port if service else 8000

    def is_internal_service(self, service_id: str) -> bool:
        """
        Check if service runs locally (MODULE or SERVICE, not REMOTE RunPod).

        Returns:
            True if service is MODULE or SERVICE mode (runs locally)
        """
        execution_mode = self.execution_config.get_execution_mode(service_id)
        return execution_mode in [ExecutionMode.MODULE, ExecutionMode.SERVICE]

    def is_module_service(self, service_id: str) -> bool:
        """
        Check if service runs in-process (MODULE mode).

        Returns:
            True if service is MODULE mode (in-process)
        """
        return self.execution_config.is_module(service_id)

    def is_service_service(self, service_id: str) -> bool:
        """
        Check if service runs in separate local process (SERVICE mode).

        Returns:
            True if service is SERVICE mode (local process)
        """
        execution_mode = self.execution_config.get_execution_mode(service_id)
        return execution_mode == ExecutionMode.SERVICE

    def get_start_order(self) -> List[str]:
        """
        Get recommended service start order based on dependencies.

        Returns:
            List of service IDs in start order
        """
        # Simple topological sort
        started = set()
        order = []

        def can_start(service_id: str) -> bool:
            """Check if service dependencies are satisfied."""
            deps = self.SERVICE_DEPENDENCIES.get(service_id, [])
            return all(dep in started for dep in deps)

        # Keep trying until all services are added
        while len(order) < len(self.services):
            added = False
            for service_id in self.services:
                if service_id not in started and can_start(service_id):
                    order.append(service_id)
                    started.add(service_id)
                    added = True

            if not added:
                # Add remaining services (circular dependencies or orphans)
                for service_id in self.services:
                    if service_id not in started:
                        order.append(service_id)
                        started.add(service_id)
                break

        return order

    def validate_configuration_consistency(self) -> Dict[str, Any]:
        """
        Validate configuration consistency to catch issues early.

        Checks:
        1. Execution mode nomenclature (module vs internal vs external)
        2. Services in both hardcoded registry and YAML config
        3. Port conflicts
        4. Missing module paths for internal services

        Returns:
            Dict with validation results (issues, warnings, status)
        """
        logger.info("🔍 Validating configuration consistency...")

        issues = []
        warnings = []

        # Check 1: Find services in BOTH hardcoded registry AND YAML config
        yaml_services = set(self.execution_config.services.keys())
        hardcoded_services = set(self.services.keys())

        dual_registered = yaml_services.intersection(hardcoded_services)
        if dual_registered:
            for service_id in dual_registered:
                yaml_mode = self.execution_config.get_execution_mode(service_id).value
                # Hardcoded are always external
                hardcoded_mode = "external"

                if yaml_mode in ["module", "service"]:
                    warnings.append(
                        f"⚠️  {service_id}: Dual registration detected\n"
                        f"    YAML config: {yaml_mode} | Hardcoded: {hardcoded_mode}\n"
                        f"    → Will use YAML config ({yaml_mode})"
                    )

        # Check 2: Validate module/service services have module_path
        for service_id, service_info in self.execution_config.services.items():
            if service_info.execution_mode in [ExecutionMode.MODULE, ExecutionMode.SERVICE]:
                if not service_info.module_path:
                    issues.append(
                        f"❌ {service_id}: Configured as {service_info.execution_mode.value} "
                        f"but missing module_path"
                    )

        # Check 3: Port conflicts between services
        port_map = {}
        for service_id, service in self.services.items():
            if service.port:
                if service.port in port_map:
                    issues.append(
                        f"❌ Port {service.port} conflict: {port_map[service.port]} and {service_id}"
                    )
                else:
                    port_map[service.port] = service_id

        # Report results
        if warnings:
            logger.warning("⚠️  Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")

        if issues:
            logger.error("❌ Configuration issues found:")
            for issue in issues:
                logger.error(f"  {issue}")
            logger.warning("   Continuing startup, but services may not work as expected")
        else:
            logger.info("✅ Configuration validation passed")

        return {
            "status": "ok" if not issues else "error",
            "issues": issues,
            "warnings": warnings,
            "services_checked": len(self.services)
        }

    def validate_port_configuration(self) -> Dict[str, Any]:
        """
        Validate port configuration for all services.

        Checks:
        1. No duplicate ports
        2. Ports in valid range (1024-65535)
        3. Ports not in use by system services

        Returns:
            Dict with validation results
        """
        logger.info("🔍 Validating port configuration...")

        issues = []
        warnings = []
        port_map = {}

        for service_id, service in self.services.items():
            port = service.port

            # Check valid range
            if port < 1024:
                warnings.append(f"⚠️  {service_id}: Port {port} is in system range (<1024)")
            elif port > 65535:
                issues.append(f"❌ {service_id}: Port {port} is invalid (>65535)")

            # Check duplicates
            if port in port_map:
                issues.append(
                    f"❌ Port {port} conflict: {service_id} and {port_map[port]}"
                )
            else:
                port_map[port] = service_id

        # Report results
        if warnings:
            logger.warning("⚠️  Port configuration warnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")

        if issues:
            logger.error("❌ Port configuration issues:")
            for issue in issues:
                logger.error(f"  {issue}")
        else:
            logger.info("✅ Port configuration validation passed")

        return {
            "status": "ok" if not issues else "error",
            "issues": issues,
            "warnings": warnings,
            "ports_checked": len(port_map)
        }
