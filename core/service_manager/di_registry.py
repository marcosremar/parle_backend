#!/usr/bin/env python3
"""
ServiceRegistryWithDI - Service Registry with Dependency Injection (v4.0)

Integrates:
- services_config.yaml (installation + execution config)
- DIContainer (dependency injection)
- ServiceContext (unified single-layer context)

Provides transparent dependency injection for all services (modules and services).

Changes in v4.0:
- Uses unified ServiceContext instead of 3-layer system
- Simpler DI with single context injection
- Lazy-loaded GPU and Metrics managers
"""

from typing import Dict, Type, Any, Optional
from pathlib import Path
import yaml
from loguru import logger
import importlib

from src.core.container import DIContainer, ServiceLifetime
from src.core.unified_context import ServiceContext


class ServiceRegistryWithDI:
    """
    Service Registry with integrated Dependency Injection (v4.0)

    Responsibilities:
    1. Load services_config.yaml
    2. Initialize Communication Manager (shared singleton)
    3. Register services in DIContainer
    4. Create ServiceContext for each service with DI

    Changes in v4.0:
    - No more GlobalContext/ProcessContext (unified into ServiceContext)
    - Communication Manager created once and shared
    - GPU/Metrics lazy-loaded per service as needed
    """

    def __init__(self, config_path: Path):
        """
        Initialize ServiceRegistryWithDI

        Args:
            config_path: Path to services_config.yaml
        """
        self.config_path = config_path
        self.services_config = {}
        self.di_container = DIContainer()

        # Shared Communication Manager (created once, injected into all services)
        self.communication_manager = None
        self.profile_name = "dev-local"

        logger.info(f"🔧 ServiceRegistryWithDI created (config: {config_path})")

    async def initialize(self, profile_name: str = "dev-local"):
        """
        Initialize registry + DI (v4.0 simplified)

        Steps:
        1. Load services_config.yaml
        2. Create shared Communication Manager
        3. Register all services in DIContainer

        Args:
            profile_name: Profile to load (dev-local, main-dev, gpu-prod, etc.)
        """
        logger.info(f"🚀 Initializing ServiceRegistryWithDI (profile: {profile_name})...")

        self.profile_name = profile_name

        # 1. Load services_config.yaml
        await self._load_services_config()

        # 2. Initialize shared Communication Manager
        await self._initialize_communication_manager()

        # 3. Register all services in DI container
        registered_count = 0
        for service_id, service_info in self.services_config.get('services', {}).items():
            di_config = service_info.get('di')
            if di_config:
                await self._register_service(service_id, service_info, di_config)
                registered_count += 1
            else:
                logger.debug(f"⏭️  Skipping {service_id} (no DI config)")

        logger.info(f"✅ ServiceRegistryWithDI initialized: {registered_count} services registered")

    async def _load_services_config(self):
        """Load services_config.yaml"""
        try:
            with open(self.config_path) as f:
                self.services_config = yaml.safe_load(f)

            services_count = len(self.services_config.get('services', {}))
            logger.info(f"📋 Loaded services_config.yaml: {services_count} services")

        except FileNotFoundError:
            logger.error(f"❌ services_config.yaml not found: {self.config_path}")
            self.services_config = {'services': {}}
        except Exception as e:
            logger.error(f"❌ Failed to load services_config.yaml: {e}")
            self.services_config = {'services': {}}

    async def _initialize_communication_manager(self):
        """
        Initialize shared Communication Manager

        This manager is created once and shared across all services.
        """
        try:
            from src.core.managers.communication_manager import ServiceCommunicationManager

            self.communication_manager = ServiceCommunicationManager()
            await self.communication_manager.initialize()

            logger.info("✅ Communication Manager initialized (shared)")

        except ImportError as e:
            logger.warning(f"⚠️ Communication Manager not available: {e}")
            self.communication_manager = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Communication Manager: {e}")
            self.communication_manager = None

    async def _register_service(self, service_id: str, service_info: Dict, di_config: Dict):
        """
        Register service in DIContainer (v4.0)

        Args:
            service_id: Service identifier (e.g., "communication", "user")
            service_info: Service configuration from YAML
            di_config: DI configuration section
        """
        try:
            # Extract DI config
            lifetime_str = di_config.get('lifetime', 'singleton')
            context_required = di_config.get('context_required', True)
            module_path = di_config.get('module_path')
            class_name = di_config.get('class_name')

            if not module_path or not class_name:
                logger.warning(f"⚠️  Skipping {service_id}: missing module_path or class_name")
                return

            # Map lifetime string to enum
            lifetime_map = {
                'singleton': ServiceLifetime.SINGLETON,
                'transient': ServiceLifetime.TRANSIENT,
                'scoped': ServiceLifetime.SCOPED
            }
            lifetime = lifetime_map.get(lifetime_str, ServiceLifetime.SINGLETON)

            # Create factory function for service creation
            def create_service_with_context():
                """Factory that creates service with ServiceContext injected (v4.0)"""
                # 1. Import service class dynamically
                module = importlib.import_module(module_path)
                service_class = getattr(module, class_name)

                # 2. Create ServiceContext for this service (v4.0 - unified single-layer)
                if context_required and self.communication_manager:
                    service_context = ServiceContext.create(
                        service_name=service_id,
                        comm=self.communication_manager,
                        profile=self.profile_name,
                        execution_mode="module"
                    )
                else:
                    service_context = None

                # 3. Extract only the 'config' section (not full service_info)
                service_config = service_info.get('config', {})

                # 4. Create service instance with context
                if context_required and service_context:
                    # v4.0: Pass unified ServiceContext
                    return service_class(context=service_context, config=service_config)
                else:
                    # LEGACY: Old-style without context (deprecated)
                    logger.warning(f"⚠️ {service_id} created without context (legacy mode - deprecated)")
                    return service_class(config=service_config)

            # Register in DI container based on lifetime
            if lifetime == ServiceLifetime.SINGLETON:
                self.di_container.register_singleton(
                    interface=service_id,
                    factory=create_service_with_context
                )
            elif lifetime == ServiceLifetime.TRANSIENT:
                self.di_container.register_transient(
                    interface=service_id,
                    implementation=lambda: create_service_with_context()
                )
            elif lifetime == ServiceLifetime.SCOPED:
                self.di_container.register_scoped(
                    interface=service_id,
                    implementation=lambda: create_service_with_context()
                )

            logger.info(f"📝 Registered: {service_id} ({lifetime_str}, context={context_required})")

        except Exception as e:
            logger.error(f"❌ Failed to register {service_id}: {e}")
            import traceback
            traceback.print_exc()

    def resolve_service(self, service_id: str, scope_id: Optional[str] = None):
        """
        Resolve service from DI container (synchronous)

        Args:
            service_id: Service identifier
            scope_id: Optional scope ID for scoped services

        Returns:
            Service instance with ServiceContext injected
        """
        try:
            return self.di_container.resolve(service_id, scope_id=scope_id)
        except ValueError as e:
            logger.error(f"❌ Cannot resolve {service_id}: {e}")
            raise

    async def resolve_service_async(self, service_id: str, scope_id: Optional[str] = None):
        """
        Resolve service from DI container (asynchronous)

        Args:
            service_id: Service identifier
            scope_id: Optional scope ID for scoped services

        Returns:
            Service instance with ServiceContext injected
        """
        try:
            return await self.di_container.resolve_async(service_id, scope_id=scope_id)
        except ValueError as e:
            logger.error(f"❌ Cannot resolve {service_id}: {e}")
            raise

    async def shutdown(self):
        """Shutdown registry and cleanup contexts (v4.0)"""
        logger.info("🛑 Shutting down ServiceRegistryWithDI...")

        # Cleanup all services in DI container
        await self.di_container.cleanup_all()

        # Shutdown Communication Manager
        if self.communication_manager and hasattr(self.communication_manager, 'shutdown'):
            try:
                await self.communication_manager.shutdown()
                logger.info("   ✅ Communication Manager shutdown")
            except Exception as e:
                logger.error(f"   ❌ Communication Manager shutdown error: {e}")

        # Shutdown all ServiceContexts
        from src.core.unified_context import shutdown_all_contexts
        await shutdown_all_contexts()

        logger.info("✅ ServiceRegistryWithDI shutdown complete")

    def get_registered_services(self) -> Dict[str, str]:
        """
        Get all registered services

        Returns:
            Dictionary of service_id -> lifetime
        """
        return self.di_container.get_all_registered()

    def get_status(self) -> dict:
        """Get registry status (v4.0)"""
        from src.core.unified_context import get_service_registry

        return {
            "initialized": self.communication_manager is not None,
            "profile": self.profile_name,
            "services_registered": len(self.di_container.get_all_registered()),
            "services_active": len(get_service_registry()),
            "communication_manager": "initialized" if self.communication_manager else "not initialized"
        }
