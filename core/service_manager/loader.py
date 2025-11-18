#!/usr/bin/env python3
"""
Service Loader Module

Handles loading and initialization of services (internal, external, composite).
Extracted from ServiceManager to improve modularity.

Responsibilities:
- Load internal service classes dynamically
- Initialize service instances with DI
- Handle service configuration
- Validate service dependencies
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ServiceLoader:
    """
    Service Loader - Dynamically loads service classes and creates instances

    Handles:
    - Internal services (loaded from Python modules)
    - External services (launched as separate processes)
    - Composite services (groups of related services)
    """

    def __init__(self, base_dir: Path):
        """
        Initialize Service Loader

        Args:
            base_dir: Base directory for services (usually src/core)
        """
        self.base_dir = base_dir
        self.loaded_classes = {}  # Cache of loaded service classes

        logger.info("📦 Service Loader initialized")

    def is_internal_service(self, service_id: str, execution_config) -> bool:
        """
        Check if service is internal (should run in Service Manager process)

        Args:
            service_id: Service identifier
            execution_config: Execution configuration

        Returns:
            True if service should run internally (module or internal)
        """
        exec_mode = execution_config.get_execution_mode(service_id)
        return exec_mode in ["module", "internal"]

    def load_service_class(self, service_id: str, services_config: Dict) -> Optional[type]:
        """
        Dynamically load service class from module path

        Args:
            service_id: Service identifier (e.g., "session", "orchestrator")
            services_config: Services configuration from services_config.yaml

        Returns:
            Service class or None if loading fails

        Example:
            session_class = loader.load_service_class("session", config)
            # Returns: SessionService class from src.services.session.service
        """
        try:
            # Check cache first
            if service_id in self.loaded_classes:
                logger.debug(f"📦 Using cached class for {service_id}")
                return self.loaded_classes[service_id]

            # Get service configuration
            service_config = services_config.get('services', {}).get(service_id)
            if not service_config:
                logger.error(f"❌ No configuration found for service: {service_id}")
                return None

            # Get DI configuration
            di_config = service_config.get('di')
            if not di_config:
                logger.error(f"❌ No DI config for service: {service_id}")
                return None

            module_path = di_config.get('module_path')
            class_name = di_config.get('class_name')

            if not module_path or not class_name:
                logger.error(f"❌ Missing module_path or class_name for {service_id}")
                return None

            logger.info(f"📦 Loading {service_id} from {module_path}.{class_name}")

            # Add project root to Python path if not already there
            project_root = self.base_dir.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Import module
            module = importlib.import_module(module_path)

            # Get class
            service_class = getattr(module, class_name)

            # Cache the class
            self.loaded_classes[service_id] = service_class

            logger.info(f"✅ Loaded {service_id}: {service_class.__name__}")
            return service_class

        except ImportError as e:
            logger.error(f"❌ Failed to import module for {service_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        except AttributeError as e:
            logger.error(f"❌ Class not found in module for {service_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error loading {service_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_service_instance(
        self,
        service_id: str,
        service_class: type,
        context: Any,
        config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Create service instance with dependency injection

        Args:
            service_id: Service identifier
            service_class: Service class to instantiate
            context: ServiceContext for dependency injection
            config: Service configuration

        Returns:
            Service instance or None if creation fails
        """
        try:
            logger.info(f"🏗️  Creating instance of {service_id}")

            # Create service with context (DI v4.0)
            service_instance = service_class(context=context, config=config)

            logger.info(f"✅ Created instance: {service_id}")
            return service_instance

        except Exception as e:
            logger.error(f"❌ Failed to create instance for {service_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_service_dependencies(
        self,
        service_id: str,
        dependencies: Dict[str, list],
        loaded_services: Dict[str, Any]
    ) -> bool:
        """
        Validate that service dependencies are loaded

        Args:
            service_id: Service to validate
            dependencies: Dependency map (e.g., {"orchestrator": ["llm", "tts", "stt"]})
            loaded_services: Currently loaded services

        Returns:
            True if all dependencies are satisfied
        """
        required_deps = dependencies.get(service_id, [])

        if not required_deps:
            logger.debug(f"✅ {service_id} has no dependencies")
            return True

        missing_deps = [dep for dep in required_deps if dep not in loaded_services]

        if missing_deps:
            logger.warning(
                f"⚠️  {service_id} missing dependencies: {missing_deps}",
                required=required_deps,
                loaded=list(loaded_services.keys())
            )
            return False

        logger.debug(f"✅ {service_id} dependencies satisfied: {required_deps}")
        return True

    def get_load_order(
        self,
        service_ids: list,
        dependencies: Dict[str, list]
    ) -> list:
        """
        Calculate optimal service load order based on dependencies

        Uses topological sort to ensure dependencies are loaded first.

        Args:
            service_ids: List of services to load
            dependencies: Dependency map

        Returns:
            Ordered list of service IDs

        Example:
            >>> loader.get_load_order(
            ...     ["orchestrator", "llm", "tts"],
            ...     {"orchestrator": ["llm", "tts"]}
            ... )
            ["llm", "tts", "orchestrator"]
        """
        # Simple topological sort
        ordered = []
        remaining = set(service_ids)
        loaded = set()

        while remaining:
            # Find services with all dependencies loaded
            ready = [
                svc for svc in remaining
                if all(dep in loaded or dep not in service_ids for dep in dependencies.get(svc, []))
            ]

            if not ready:
                # Circular dependency or missing dependency
                logger.warning(f"⚠️  Cannot resolve dependencies, loading remaining in order: {remaining}")
                ordered.extend(sorted(remaining))
                break

            # Load services that are ready
            for svc in ready:
                ordered.append(svc)
                loaded.add(svc)
                remaining.remove(svc)

        return ordered

    def reload_service_class(self, service_id: str) -> bool:
        """
        Reload service class (for hot reload)

        Args:
            service_id: Service to reload

        Returns:
            True if reload successful
        """
        try:
            if service_id not in self.loaded_classes:
                logger.warning(f"⚠️  Service {service_id} not loaded, cannot reload")
                return False

            # Get module path from cached class
            service_class = self.loaded_classes[service_id]
            module_name = service_class.__module__

            # Reload module
            if module_name in sys.modules:
                logger.info(f"🔄 Reloading module: {module_name}")
                importlib.reload(sys.modules[module_name])

                # Update cached class
                module = sys.modules[module_name]
                class_name = service_class.__name__
                reloaded_class = getattr(module, class_name)
                self.loaded_classes[service_id] = reloaded_class

                logger.info(f"✅ Reloaded {service_id} successfully")
                return True
            else:
                logger.error(f"❌ Module {module_name} not in sys.modules")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to reload {service_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clear_cache(self):
        """Clear loaded class cache (for testing)"""
        self.loaded_classes.clear()
        logger.debug("🗑️  Cleared service class cache")

    def get_loaded_services(self) -> list:
        """Get list of loaded service IDs"""
        return list(self.loaded_classes.keys())
