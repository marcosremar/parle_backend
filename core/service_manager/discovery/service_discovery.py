#!/usr/bin/env python3
"""
Service Discovery Module

Automatically discovers all services in src/services/ directory,
classifies them by type, and determines their installation status.
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import requests

# Add project root to path
# __file__ is in src/core/service_manager/discovery/service_discovery.py
# So we need 5 .parent calls to get to project root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import health check timeout constant
from src.core.service_manager.core import HEALTH_CHECK_TIMEOUT


class ServiceType(Enum):
    """Types of services in the Ultravox Pipeline"""
    AI_MODEL = "ai_model"  # LLM, STT, TTS - require model downloads
    INFRASTRUCTURE = "infrastructure"  # API Gateway, WebSocket, Orchestrator
    EXTERNAL = "external"  # External service integrations
    UTILITY = "utility"  # File storage, session management, metrics
    TEST = "test"  # Test/demo services


class ServiceStatus(Enum):
    """Installation and runtime status of a service"""
    AVAILABLE = "available"  # Directory exists with service.py
    INSTALLED = "installed"  # Dependencies installed
    HEALTHY = "healthy"  # Running and responding to /health
    DEGRADED = "degraded"  # Running but with issues
    NOT_INSTALLED = "not_installed"  # Missing dependencies
    STOPPED = "stopped"  # Not currently running
    UNKNOWN = "unknown"  # Cannot determine status


@dataclass
class ServiceInfo:
    """Complete information about a discovered service"""
    service_id: str
    service_type: ServiceType
    status: ServiceStatus
    path: Path
    has_install_script: bool = False
    has_service_file: bool = False
    has_config: bool = False
    port: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    health_url: Optional[str] = None
    description: str = ""
    install_time_estimate: str = ""  # e.g., "5-10 min"
    disk_space_required: str = ""  # e.g., "10GB"


class ServiceDiscovery:
    """
    Discovers and classifies all services in the Ultravox Pipeline.

    Scans src/services/ directory and builds a complete catalog of:
    - Service types (AI models, infrastructure, external, utilities)
    - Installation status (available, installed, healthy, etc.)
    - Dependencies and requirements
    - Health check endpoints
    """

    def __init__(self, services_dir: Optional[Path] = None):
        """
        Initialize service discovery.

        Args:
            services_dir: Path to services directory (default: src/services)
        """
        self.services_dir = services_dir or project_root / "src" / "services"
        self.services: Dict[str, ServiceInfo] = {}

        # Service type classification rules
        self.ai_model_services = {"llm", "stt", "tts"}
        self.external_services = {"external_llm", "external_stt", "external_ultravox"}
        self.infrastructure_services = {
            "api_gateway", "websocket", "orchestrator", "webrtc",
            "webrtc_signaling", "machine_manager"
        }
        self.utility_services = {
            "file_storage", "session", "conversation_store",
            "user", "metrics_testing", "rest_polling", "runpod_llm"
        }
        self.test_services = {"simple_test", "simple_test2", "scenarios"}

        # Default port mappings (from service_config.py)
        self.default_ports = {
            "llm": 8100,
            "stt": 8101,
            "tts": 8102,
            "api_gateway": 8000,
            "websocket": 8001,
            "file_storage": 8003,
            "session": 8004,
            "orchestrator": 8005,
            "metrics_testing": 8006,
        }

    def discover_all_services(self) -> Dict[str, ServiceInfo]:
        """
        Discover all services in src/services/ directory.

        Returns:
            Dictionary mapping service_id to ServiceInfo
        """
        print(f"\n🔍 Discovering services in {self.services_dir}...")

        if not self.services_dir.exists():
            print(f"❌ Services directory not found: {self.services_dir}")
            return {}

        # Scan all subdirectories
        for service_dir in sorted(self.services_dir.iterdir()):
            if not service_dir.is_dir():
                continue

            service_id = service_dir.name

            # Skip __pycache__ and other non-service directories
            if service_id.startswith("_") or service_id.startswith("."):
                continue

            # Create service info
            service_info = self._analyze_service(service_id, service_dir)
            self.services[service_id] = service_info

            # Print discovery result
            status_icon = self._get_status_icon(service_info.status)
            type_icon = self._get_type_icon(service_info.service_type)
            print(f"   {status_icon} {type_icon} {service_id:25s} - {service_info.status.value}")

        print(f"\n✅ Discovered {len(self.services)} services")
        return self.services

    def _analyze_service(self, service_id: str, service_dir: Path) -> ServiceInfo:
        """
        Analyze a service directory to determine its properties.

        Args:
            service_id: Service identifier
            service_dir: Path to service directory

        Returns:
            ServiceInfo with complete service details
        """
        # Determine service type
        service_type = self._classify_service_type(service_id)

        # Check for key files
        has_install_script = (service_dir / "install.py").exists()
        has_service_file = (service_dir / "service.py").exists()
        has_config = (service_dir / "config.py").exists() or (service_dir / "config.yaml").exists()

        # Get port
        port = self.default_ports.get(service_id)

        # Determine health URL
        health_url = f"http://localhost:{port}/health" if port else None

        # Check installation status
        status = self._check_service_status(service_id, service_dir, port, health_url)

        # Get dependencies (if install.py exists)
        dependencies = self._extract_dependencies(service_dir) if has_install_script else []

        # Get description
        description = self._get_service_description(service_id, service_type)

        # Get install estimates (for AI models)
        install_time, disk_space = self._get_install_estimates(service_id, service_type)

        return ServiceInfo(
            service_id=service_id,
            service_type=service_type,
            status=status,
            path=service_dir,
            has_install_script=has_install_script,
            has_service_file=has_service_file,
            has_config=has_config,
            port=port,
            dependencies=dependencies,
            health_url=health_url,
            description=description,
            install_time_estimate=install_time,
            disk_space_required=disk_space
        )

    def _classify_service_type(self, service_id: str) -> ServiceType:
        """Classify service by type based on service_id"""
        if service_id in self.ai_model_services:
            return ServiceType.AI_MODEL
        elif service_id in self.external_services:
            return ServiceType.EXTERNAL
        elif service_id in self.infrastructure_services:
            return ServiceType.INFRASTRUCTURE
        elif service_id in self.test_services:
            return ServiceType.TEST
        elif service_id in self.utility_services:
            return ServiceType.UTILITY
        else:
            return ServiceType.UTILITY  # Default to utility

    def _check_service_status(
        self,
        service_id: str,
        service_dir: Path,
        port: Optional[int],
        health_url: Optional[str]
    ) -> ServiceStatus:
        """
        Check the current status of a service.

        Checks:
        1. Is service.py available?
        2. Are dependencies installed?
        3. Is service running on its port?
        4. Does it respond to health check?
        """
        # Check if service.py exists
        if not (service_dir / "service.py").exists():
            return ServiceStatus.AVAILABLE

        # Check dependencies (for AI models with install.py)
        if (service_dir / "install.py").exists():
            if not self._check_dependencies_installed(service_dir):
                return ServiceStatus.NOT_INSTALLED

        # Check if running (port check)
        if port:
            if not self._check_port_listening(port):
                return ServiceStatus.STOPPED

            # Check health endpoint
            if health_url:
                health_status = self._check_health_endpoint(health_url)
                if health_status == "healthy":
                    return ServiceStatus.HEALTHY
                elif health_status == "degraded":
                    return ServiceStatus.DEGRADED
                else:
                    return ServiceStatus.STOPPED

            # Port is open but no health endpoint
            return ServiceStatus.INSTALLED

        # No port defined, assume installed if dependencies OK
        return ServiceStatus.INSTALLED

    def _check_dependencies_installed(self, service_dir: Path) -> bool:
        """Check if service dependencies are installed"""
        install_script = service_dir / "install.py"
        if not install_script.exists():
            return True  # No dependencies to check

        # Try to parse install.py for required packages
        try:
            with open(install_script) as f:
                content = f.read()

            # Look for required_packages list
            if "required_packages" in content:
                # This is a basic check - full check would import and run
                return True  # Assume installed for now

            return True
        except (IOError, OSError) as e:
            # Log file read failures but don't crash
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not read install script {install_script}: {e}")
            return True

    def _check_port_listening(self, port: int) -> bool:
        """Check if a port is listening"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except (socket.error, socket.timeout, OSError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Port check failed for port {port}: {e}")
            return False

    def _check_health_endpoint(self, health_url: str) -> str:
        """
        Check service health endpoint.

        Returns:
            'healthy', 'degraded', or 'unknown'
        """
        try:
            response = requests.get(health_url, timeout=HEALTH_CHECK_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "").lower()
                if status == "healthy":
                    return "healthy"
                else:
                    return "degraded"
            else:
                return "degraded"
        except (requests.RequestException, requests.Timeout, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Health check failed for {health_url}: {e}")
            return "unknown"

    def _extract_dependencies(self, service_dir: Path) -> List[str]:
        """Extract list of dependencies from install.py"""
        install_script = service_dir / "install.py"
        dependencies = []

        try:
            with open(install_script) as f:
                content = f.read()

            # Look for required_packages list
            if "required_packages" in content:
                # Basic parsing - look for list
                import re
                match = re.search(r'required_packages\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    packages_str = match.group(1)
                    # Extract quoted strings
                    packages = re.findall(r"['\"]([^'\"]+)['\"]", packages_str)
                    dependencies.extend(packages)
        except (IOError, OSError, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not extract dependencies from {install_script}: {e}")

        return dependencies

    def _get_service_description(self, service_id: str, service_type: ServiceType) -> str:
        """Get human-readable description of service"""
        descriptions = {
            # AI Models
            "llm": "Speech-to-Speech LLM (Ultravox)",
            "stt": "Speech-to-Text (Whisper)",
            "tts": "Text-to-Speech (Kokoro)",

            # Infrastructure
            "api_gateway": "Main HTTP API Gateway",
            "websocket": "WebSocket Server",
            "orchestrator": "Service Orchestration",
            "webrtc": "WebRTC Communication",
            "webrtc_signaling": "WebRTC Signaling Server",
            "machine_manager": "Pod & Machine Management",

            # External
            "external_llm": "External LLM Integration",
            "external_stt": "External STT Integration",
            "external_ultravox": "External Ultravox Integration",

            # Utilities
            "file_storage": "File Upload/Download Service",
            "session": "Session State Management",
            "conversation_store": "Conversation History Storage",
            "user": "User Management",
            "metrics_testing": "Performance Metrics Testing",
            "rest_polling": "REST API Polling",
            "runpod_llm": "RunPod LLM Integration",

            # Tests
            "scenarios": "Test Scenarios",
            "simple_test": "Simple Test Service",
            "simple_test2": "Simple Test Service 2",
        }

        return descriptions.get(service_id, f"{service_type.value.title()} Service")

    def _get_install_estimates(self, service_id: str, service_type: ServiceType) -> tuple:
        """Get installation time and disk space estimates"""
        estimates = {
            "llm": ("10-15 min", "25GB"),
            "stt": ("5-10 min", "10GB"),
            "tts": ("5-10 min", "5GB"),
        }

        return estimates.get(service_id, ("< 1 min", "< 100MB"))

    def _get_status_icon(self, status: ServiceStatus) -> str:
        """Get emoji icon for service status"""
        icons = {
            ServiceStatus.HEALTHY: "✅",
            ServiceStatus.INSTALLED: "📦",
            ServiceStatus.AVAILABLE: "📁",
            ServiceStatus.DEGRADED: "⚠️",
            ServiceStatus.NOT_INSTALLED: "❌",
            ServiceStatus.STOPPED: "⏹️",
            ServiceStatus.UNKNOWN: "❓",
        }
        return icons.get(status, "❓")

    def _get_type_icon(self, service_type: ServiceType) -> str:
        """Get emoji icon for service type"""
        icons = {
            ServiceType.AI_MODEL: "🤖",
            ServiceType.INFRASTRUCTURE: "🏗️",
            ServiceType.EXTERNAL: "🔌",
            ServiceType.UTILITY: "🔧",
            ServiceType.TEST: "🧪",
        }
        return icons.get(service_type, "📦")

    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all services of a specific type"""
        return [s for s in self.services.values() if s.service_type == service_type]

    def get_services_by_status(self, status: ServiceStatus) -> List[ServiceInfo]:
        """Get all services with a specific status"""
        return [s for s in self.services.values() if s.status == status]

    def get_installable_services(self) -> List[ServiceInfo]:
        """Get all services that have install scripts"""
        return [s for s in self.services.values() if s.has_install_script]

    def print_summary(self):
        """Print a summary of discovered services"""
        print("\n" + "=" * 80)
        print("📊 SERVICE DISCOVERY SUMMARY")
        print("=" * 80)

        # Count by type
        print("\n📦 Services by Type:")
        for service_type in ServiceType:
            services = self.get_services_by_type(service_type)
            if services:
                print(f"   {self._get_type_icon(service_type)} {service_type.value.title():20s}: {len(services)}")

        # Count by status
        print("\n📊 Services by Status:")
        for status in ServiceStatus:
            services = self.get_services_by_status(status)
            if services:
                print(f"   {self._get_status_icon(status)} {status.value.title():20s}: {len(services)}")

        # Installable services
        installable = self.get_installable_services()
        print(f"\n✅ Services with install.py: {len(installable)}")
        for service in installable:
            print(f"   - {service.service_id}")

        print("\n" + "=" * 80)


def discover_services() -> Dict[str, ServiceInfo]:
    """
    Convenience function to discover all services.

    Returns:
        Dictionary mapping service_id to ServiceInfo
    """
    discovery = ServiceDiscovery()
    services = discovery.discover_all_services()
    discovery.print_summary()
    return services


if __name__ == "__main__":
    # Run discovery when executed directly
    services = discover_services()

    print(f"\n✅ Total services discovered: {len(services)}")
