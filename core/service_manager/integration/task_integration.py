"""
Task Management Integration for Service Manager
Adds async task support while preserving existing structure
"""

from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from fastapi import Query, HTTPException
import sys
from pathlib import Path

# Add project root to path if not already there
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import task management
from src.core.task_manager import (
    get_task_manager,
    TaskType, TaskStatus, TaskPriority, TaskFilter
)


class TaskIntegration:
    """Integrates task management into existing ServiceManager"""

    def __init__(self, service_manager):
        self.manager = service_manager

        # Initialize task manager
        self.task_manager = get_task_manager(
            service_name="service-manager",
            max_workers=5,
            default_timeout=300,
            cleanup_after_hours=24
        )

        # Register handlers that use existing methods
        self._register_handlers()

    def _register_handlers(self):
        """Register task handlers that wrap existing ServiceManager methods"""

        def handle_start(task):
            """Wrapper for start_service"""
            service_id = task.metadata.target
            force = task.metadata.custom_data.get("force", False)

            self.task_manager.update_task_progress(
                task.id, current=1, total=2, message=f"Starting {service_id}"
            )

            result = self.manager.start_service(service_id, force)

            self.task_manager.update_task_progress(
                task.id, current=2, total=2, message="Service started"
            )

            return result

        def handle_stop(task):
            """Wrapper for stop_service"""
            service_id = task.metadata.target

            self.task_manager.update_task_progress(
                task.id, current=1, total=2, message=f"Stopping {service_id}"
            )

            result = self.manager.stop_service(service_id)

            self.task_manager.update_task_progress(
                task.id, current=2, total=2, message="Service stopped"
            )

            return result

        def handle_restart(task):
            """Wrapper for restart - uses existing stop and start"""
            service_id = task.metadata.target
            force = task.metadata.custom_data.get("force", False)

            self.task_manager.update_task_progress(
                task.id, current=1, total=3, message=f"Stopping {service_id}"
            )

            stop_result = self.manager.stop_service(service_id)

            self.task_manager.update_task_progress(
                task.id, current=2, total=3, message="Starting service"
            )

            import time
            time.sleep(2)
            start_result = self.manager.start_service(service_id, force)

            self.task_manager.update_task_progress(
                task.id, current=3, total=3, message="Service restarted"
            )

            return {
                "stop": stop_result,
                "start": start_result,
                "success": start_result.get("success", False)
            }

        def handle_validate_all(task):
            """Wrapper for validate_all_apis"""
            self.task_manager.update_task_progress(
                task.id, current=1, total=2, message="Starting validation"
            )

            result = self.manager._validate_all_apis_sync()

            self.task_manager.update_task_progress(
                task.id, current=2, total=2, message="Validation complete"
            )

            return result

        def handle_start_all(task):
            """Wrapper for start_all_services"""
            force = task.metadata.custom_data.get("force", False)

            # Skip service_manager itself to avoid self-termination
            services = [s for s in self.manager.services.keys() if s != "service_manager"]
            total = len(services)
            results = {}

            for idx, service_id in enumerate(services, 1):
                self.task_manager.update_task_progress(
                    task.id,
                    current=idx,
                    total=total,
                    message=f"Starting {service_id} ({idx}/{total})"
                )

                try:
                    results[service_id] = self.manager.start_service(service_id, force)
                except Exception as e:
                    results[service_id] = {"success": False, "error": str(e)}

            return results

        def handle_stop_all(task):
            """Wrapper for stop_all_services"""
            # Skip service_manager itself to avoid self-termination
            services = [s for s in self.manager.services.keys() if s != "service_manager"]
            total = len(services)
            results = {}

            for idx, service_id in enumerate(services, 1):
                self.task_manager.update_task_progress(
                    task.id,
                    current=idx,
                    total=total,
                    message=f"Stopping {service_id} ({idx}/{total})"
                )

                try:
                    results[service_id] = self.manager.stop_service(service_id)
                except Exception as e:
                    results[service_id] = {"success": False, "error": str(e)}

            return results

        def handle_restart_all(task):
            """Wrapper for restart_all_services"""
            force = task.metadata.custom_data.get("force", False)

            # Skip service_manager itself to avoid self-termination
            services = [s for s in self.manager.services.keys() if s != "service_manager"]
            total = len(services) * 2  # Stop + Start for each service
            results = {"stopped": {}, "started": {}}
            current = 0

            # Stop all services first
            for service_id in services:
                current += 1
                self.task_manager.update_task_progress(
                    task.id,
                    current=current,
                    total=total,
                    message=f"Stopping {service_id} ({current}/{total})"
                )

                try:
                    results["stopped"][service_id] = self.manager.stop_service(service_id)
                except Exception as e:
                    results["stopped"][service_id] = {"success": False, "error": str(e)}

            # Wait a bit for services to fully stop
            import time
            time.sleep(3)

            # Start all services
            for service_id in services:
                current += 1
                self.task_manager.update_task_progress(
                    task.id,
                    current=current,
                    total=total,
                    message=f"Starting {service_id} ({current}/{total})"
                )

                try:
                    results["started"][service_id] = self.manager.start_service(service_id, force)
                except Exception as e:
                    results["started"][service_id] = {"success": False, "error": str(e)}

            return results

        # Register all handlers
        self.task_manager.register_handler(TaskType.START, handle_start)
        self.task_manager.register_handler(TaskType.STOP, handle_stop)
        self.task_manager.register_handler(TaskType.RESTART, handle_restart)
        self.task_manager.register_handler(TaskType.VALIDATE_ALL, handle_validate_all)
        self.task_manager.register_handler(TaskType.START_ALL, handle_start_all)
        self.task_manager.register_handler(TaskType.STOP_ALL, handle_stop_all)
        self.task_manager.register_handler(TaskType.RESTART_ALL, handle_restart_all)

    def create_start_task(self, service_id: str, force: bool = False,
                          webhook_url: Optional[str] = None,
                          priority: int = 3) -> str:
        """Create async start task"""
        return self.task_manager.create_task(
            task_type=TaskType.START,
            priority=TaskPriority(priority),
            metadata={
                "target": service_id,
                "webhook_url": webhook_url,
                "custom_data": {"force": force}
            }
        )

    def create_stop_task(self, service_id: str,
                         webhook_url: Optional[str] = None,
                         priority: int = 3) -> str:
        """Create async stop task"""
        return self.task_manager.create_task(
            task_type=TaskType.STOP,
            priority=TaskPriority(priority),
            metadata={
                "target": service_id,
                "webhook_url": webhook_url
            }
        )

    def create_restart_task(self, service_id: str, force: bool = False,
                           webhook_url: Optional[str] = None) -> str:
        """Create async restart task"""
        return self.task_manager.create_task(
            task_type=TaskType.RESTART,
            priority=TaskPriority.HIGH,
            metadata={
                "target": service_id,
                "webhook_url": webhook_url,
                "custom_data": {"force": force}
            }
        )

    def create_validate_all_task(self, webhook_url: Optional[str] = None) -> str:
        """Create async validate all task"""
        return self.task_manager.create_task(
            task_type=TaskType.VALIDATE_ALL,
            priority=TaskPriority.LOW,
            metadata={"webhook_url": webhook_url}
        )

    def create_start_all_task(self, force: bool = True,
                             webhook_url: Optional[str] = None) -> str:
        """Create async start all task"""
        import logging
        logger = logging.getLogger("service-manager")
        logger.info(f"Creating START_ALL task with force={force}")

        try:
            task_id = self.task_manager.create_task(
                task_type=TaskType.START_ALL,
                priority=TaskPriority.HIGH,
                metadata={
                    "webhook_url": webhook_url,
                    "custom_data": {"force": force}
                }
            )
            logger.info(f"START_ALL task created successfully: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to create START_ALL task: {e}")
            raise

    def create_stop_all_task(self, webhook_url: Optional[str] = None) -> str:
        """Create async stop all task"""
        return self.task_manager.create_task(
            task_type=TaskType.STOP_ALL,
            priority=TaskPriority.HIGH,
            metadata={"webhook_url": webhook_url}
        )

    def create_restart_all_task(self, force: bool = True,
                               webhook_url: Optional[str] = None) -> str:
        """Create async restart all task"""
        return self.task_manager.create_task(
            task_type=TaskType.RESTART_ALL,
            priority=TaskPriority.HIGH,
            metadata={
                "webhook_url": webhook_url,
                "custom_data": {"force": force}
            }
        )


def add_async_endpoints(app, manager):
    """Add async endpoints to existing FastAPI app"""

    # Create task integration
    task_integration = TaskIntegration(manager)

    # Store reference in manager for use in endpoints
    manager.task_integration = task_integration

    # Add new async endpoints

    @app.post("/services/{service_id}/start")
    async def start_service_async(service_id: str,
                                  force: bool = False,
                                  webhook_url: Optional[str] = None,
                                  priority: int = 3):
        """Start service asynchronously - returns tracking ID immediately"""
        if service_id not in manager.services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        tracking_id = task_integration.create_start_task(
            service_id, force, webhook_url, priority
        )

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": f"Start task for {service_id} queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    @app.post("/services/{service_id}/stop")
    async def stop_service_async(service_id: str,
                                 webhook_url: Optional[str] = None,
                                 priority: int = 3):
        """Stop service asynchronously - returns tracking ID immediately"""
        if service_id not in manager.services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        tracking_id = task_integration.create_stop_task(
            service_id, webhook_url, priority
        )

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": f"Stop task for {service_id} queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    @app.post("/services/{service_id}/restart")
    async def restart_service_async(service_id: str,
                                   force: bool = False,
                                   webhook_url: Optional[str] = None):
        """Restart service asynchronously - returns tracking ID immediately"""
        if service_id not in manager.services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        tracking_id = task_integration.create_restart_task(
            service_id, force, webhook_url
        )

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": f"Restart task for {service_id} queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    # Note: validate-all endpoint is already async in main.py
    # It returns a tracking ID immediately, no need for validate-all-async

    @app.post("/services/start-all")
    async def start_all_async(force: bool = True,
                             webhook_url: Optional[str] = None):
        """Start all services asynchronously - returns tracking ID immediately"""
        import logging
        logger = logging.getLogger("service-manager")
        logger.info(f"start-all endpoint called with force={force}")

        try:
            tracking_id = task_integration.create_start_all_task(force, webhook_url)
            logger.info(f"Task created with tracking_id: {tracking_id}")
        except Exception as e:
            logger.error(f"Failed to create start-all task: {e}")
            raise

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": "Start all services task queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    @app.post("/services/stop-all")
    async def stop_all_async(webhook_url: Optional[str] = None):
        """Stop all services asynchronously - returns tracking ID immediately"""
        tracking_id = task_integration.create_stop_all_task(webhook_url)

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": "Stop all services task queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    @app.post("/services/restart-all")
    async def restart_all_async(force: bool = Query(True, description="Force restart even if services are running"),
                               webhook_url: Optional[str] = None):
        """Restart all services asynchronously - returns tracking ID immediately"""
        tracking_id = task_integration.create_restart_all_task(force, webhook_url)

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": "Restart all services task queued",
            "check_url": f"/tasks/{tracking_id}"
        }

    # Task management endpoints

    @app.get("/tasks/{tracking_id}")
    async def get_task_status(tracking_id: str):
        """Get status of a background task"""
        task_status = task_integration.task_manager.get_task_status(tracking_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {tracking_id} not found")
        return task_status

    @app.get("/tasks")
    async def list_tasks(
        status: Optional[str] = Query(None, description="Filter by status"),
        type: Optional[str] = Query(None, description="Filter by type"),
        limit: int = Query(100, description="Maximum number of tasks"),
        offset: int = Query(0, description="Offset for pagination")
    ):
        """List all tasks with optional filtering"""
        filter = TaskFilter(
            service_name="service-manager",
            status=TaskStatus(status) if status else None,
            type=TaskType(type) if type else None,
            limit=limit,
            offset=offset
        )

        tasks = task_integration.task_manager.list_tasks(filter)
        return {
            "tasks": tasks,
            "total": len(tasks),
            "statistics": task_integration.task_manager.get_statistics()
        }

    @app.delete("/tasks/{tracking_id}")
    async def cancel_task(tracking_id: str):
        """Cancel a pending or running task"""
        success = task_integration.task_manager.cancel_task(tracking_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not found or in terminal state)"
            )
        return {"message": f"Task {tracking_id} cancelled successfully"}

    print(f"✨ Async task endpoints added to Service Manager")
    return task_integration