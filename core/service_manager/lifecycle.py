#!/usr/bin/env python3
"""
Service Lifecycle Module

Handles service lifecycle operations (start, stop, restart, reload).
Extracted from ServiceManager to improve modularity.

Responsibilities:
- Start services (internal, external, composite)
- Stop services gracefully
- Restart services
- Hot reload (for internal services)
- Monitor service health during lifecycle
"""

import asyncio
import time
from typing import Dict, Any, Optional
from loguru import logger
from fastapi import FastAPI


class LifecycleManager:
    """
    Service Lifecycle Manager - Manages service start/stop/restart

    Handles:
    - Internal services (loaded in-process)
    - External services (separate processes)
    - Composite services (groups)
    - Health monitoring during lifecycle
    """

    def __init__(self):
        """Initialize Lifecycle Manager"""
        self.starting_services = set()  # Track services currently starting
        self.stopping_services = set()  # Track services currently stopping

        logger.info("🔄 Lifecycle Manager initialized")

    async def start_internal_service(
        self,
        service_id: str,
        service_instance: Any,
        app: Optional[FastAPI] = None
    ) -> bool:
        """
        Start internal service (loaded in Service Manager process)

        Args:
            service_id: Service identifier
            service_instance: Service instance (already created)
            app: FastAPI app instance (for route mounting)

        Returns:
            True if service started successfully
        """
        try:
            if service_id in self.starting_services:
                logger.warning(f"⚠️  {service_id} is already starting")
                return False

            self.starting_services.add(service_id)
            logger.info(f"🚀 Starting internal service: {service_id}")

            # 1. Initialize service
            if hasattr(service_instance, 'initialize'):
                success = await service_instance.initialize()
                if not success:
                    logger.error(f"❌ {service_id} initialization failed")
                    self.starting_services.discard(service_id)
                    return False

            # 2. Mount routes (if FastAPI app provided)
            if app and hasattr(service_instance, 'get_router'):
                router = service_instance.get_router()
                prefix = f"/api/{service_id}"
                app.include_router(router, prefix=prefix, tags=[service_id])
                logger.info(f"📡 Mounted {service_id} routes at {prefix}")

            # 3. Start service (if has start method)
            if hasattr(service_instance, 'start'):
                success = await service_instance.start()
                if not success:
                    logger.error(f"❌ {service_id} start failed")
                    self.starting_services.discard(service_id)
                    return False

            logger.info(f"✅ {service_id} started successfully")
            self.starting_services.discard(service_id)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start {service_id}: {e}")
            import traceback
            traceback.print_exc()
            self.starting_services.discard(service_id)
            return False

    async def stop_internal_service(
        self,
        service_id: str,
        service_instance: Any
    ) -> bool:
        """
        Stop internal service gracefully

        Args:
            service_id: Service identifier
            service_instance: Service instance to stop

        Returns:
            True if service stopped successfully
        """
        try:
            if service_id in self.stopping_services:
                logger.warning(f"⚠️  {service_id} is already stopping")
                return False

            self.stopping_services.add(service_id)
            logger.info(f"🛑 Stopping internal service: {service_id}")

            # 1. Stop service (if has stop method)
            if hasattr(service_instance, 'stop'):
                await service_instance.stop()

            # 2. Shutdown service (if has shutdown method)
            if hasattr(service_instance, 'shutdown'):
                await service_instance.shutdown()

            logger.info(f"✅ {service_id} stopped successfully")
            self.stopping_services.discard(service_id)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to stop {service_id}: {e}")
            import traceback
            traceback.print_exc()
            self.stopping_services.discard(service_id)
            return False

    async def restart_internal_service(
        self,
        service_id: str,
        service_instance: Any,
        app: Optional[FastAPI] = None
    ) -> bool:
        """
        Restart internal service (stop + start)

        Args:
            service_id: Service identifier
            service_instance: Service instance to restart
            app: FastAPI app instance

        Returns:
            True if restart successful
        """
        logger.info(f"🔄 Restarting {service_id}")

        # Stop
        stop_success = await self.stop_internal_service(service_id, service_instance)
        if not stop_success:
            logger.error(f"❌ Failed to stop {service_id} for restart")
            return False

        # Wait a bit for cleanup
        await asyncio.sleep(1)

        # Start
        start_success = await self.start_internal_service(service_id, service_instance, app)
        if not start_success:
            logger.error(f"❌ Failed to start {service_id} after restart")
            return False

        logger.info(f"✅ {service_id} restarted successfully")
        return True

    async def reload_internal_service(
        self,
        service_id: str,
        service_instance: Any,
        loader: Any,
        app: Optional[FastAPI] = None
    ) -> tuple[bool, Optional[Any]]:
        """
        Hot reload internal service (reload code without full restart)

        Args:
            service_id: Service identifier
            service_instance: Current service instance
            loader: ServiceLoader instance (for reloading classes)
            app: FastAPI app instance

        Returns:
            (success, new_service_instance)
        """
        try:
            logger.info(f"♻️  Hot reloading {service_id}")

            # 1. Reload service class
            reload_success = loader.reload_service_class(service_id)
            if not reload_success:
                logger.error(f"❌ Failed to reload class for {service_id}")
                return False, None

            # 2. Stop old instance
            stop_success = await self.stop_internal_service(service_id, service_instance)
            if not stop_success:
                logger.warning(f"⚠️  Failed to stop old instance of {service_id}")

            # 3. Create new instance (caller must handle this with new class)
            logger.info(f"✅ {service_id} reloaded (new instance required)")
            return True, None

        except Exception as e:
            logger.error(f"❌ Failed to reload {service_id}: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    async def start_all_services(
        self,
        services: Dict[str, Any],
        service_instances: Dict[str, Any],
        app: Optional[FastAPI] = None,
        start_order: Optional[list] = None
    ) -> Dict[str, bool]:
        """
        Start all services in dependency order

        Args:
            services: Service configurations
            service_instances: Created service instances
            app: FastAPI app instance
            start_order: Optional custom start order

        Returns:
            Dict mapping service_id -> success status
        """
        results = {}

        # Use provided start order or default to service keys
        order = start_order or list(services.keys())

        logger.info(f"🚀 Starting {len(order)} services in order: {order}")

        for service_id in order:
            if service_id not in service_instances:
                logger.warning(f"⚠️  Skipping {service_id} (not loaded)")
                results[service_id] = False
                continue

            service_instance = service_instances[service_id]

            # Start service
            success = await self.start_internal_service(
                service_id,
                service_instance,
                app
            )

            results[service_id] = success

            if success:
                logger.info(f"✅ [{len([r for r in results.values() if r])}/{len(order)}] {service_id} started")
            else:
                logger.error(f"❌ [{len([r for r in results.values() if r])}/{len(order)}] {service_id} failed")

            # Small delay between starts
            await asyncio.sleep(0.5)

        success_count = sum(1 for r in results.values() if r)
        logger.info(f"🎯 Started {success_count}/{len(order)} services successfully")

        return results

    async def stop_all_services(
        self,
        service_instances: Dict[str, Any],
        stop_order: Optional[list] = None
    ) -> Dict[str, bool]:
        """
        Stop all services in reverse dependency order

        Args:
            service_instances: Service instances to stop
            stop_order: Optional custom stop order (reverse of start order)

        Returns:
            Dict mapping service_id -> success status
        """
        results = {}

        # Use provided stop order or reverse of instance keys
        order = stop_order or list(reversed(list(service_instances.keys())))

        logger.info(f"🛑 Stopping {len(order)} services in order: {order}")

        for service_id in order:
            if service_id not in service_instances:
                logger.warning(f"⚠️  Skipping {service_id} (not loaded)")
                results[service_id] = False
                continue

            service_instance = service_instances[service_id]

            # Stop service
            success = await self.stop_internal_service(service_id, service_instance)
            results[service_id] = success

            if success:
                logger.info(f"✅ [{len([r for r in results.values() if r])}/{len(order)}] {service_id} stopped")
            else:
                logger.error(f"❌ [{len([r for r in results.values() if r])}/{len(order)}] {service_id} failed to stop")

            # Small delay between stops
            await asyncio.sleep(0.3)

        success_count = sum(1 for r in results.values() if r)
        logger.info(f"🎯 Stopped {success_count}/{len(order)} services successfully")

        return results

    async def health_check_service(
        self,
        service_id: str,
        service_instance: Any
    ) -> Dict[str, Any]:
        """
        Perform health check on service

        Args:
            service_id: Service identifier
            service_instance: Service instance

        Returns:
            Health check result dict
        """
        try:
            if hasattr(service_instance, 'health_check'):
                health = await service_instance.health_check()
                return {
                    "service": service_id,
                    "status": "healthy",
                    "details": health
                }
            else:
                return {
                    "service": service_id,
                    "status": "unknown",
                    "message": "No health_check method"
                }

        except Exception as e:
            logger.error(f"❌ Health check failed for {service_id}: {e}")
            return {
                "service": service_id,
                "status": "unhealthy",
                "error": str(e)
            }

    def get_lifecycle_status(self) -> Dict[str, Any]:
        """
        Get lifecycle manager status

        Returns:
            Status dictionary
        """
        return {
            "starting_services": list(self.starting_services),
            "stopping_services": list(self.stopping_services),
            "active_operations": len(self.starting_services) + len(self.stopping_services)
        }
