"""
Debug endpoint for process visualization
Shows all registered processes, orphans, and zombies
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import ValidationError, BaseModel
import logging
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)


# Response models
class ProcessInfo(BaseModel):
    """Process information"""
    pid: Optional[int]
    name: str
    state: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    uptime_seconds: float
    is_healthy: bool
    restart_count: int
    memory_limit_mb: int
    cpu_limit_percent: int
    port: Optional[int] = None
    command: Optional[str] = None


class OrphanProcess(BaseModel):
    """Orphan process information"""
    pid: int
    name: str
    cmdline: str
    age_seconds: float
    memory_mb: float
    cpu_percent: float


class ZombieProcess(BaseModel):
    """Zombie process information"""
    pid: int
    name: str
    ppid: int
    parent_name: Optional[str]


class ProcessesSummary(BaseModel):
    """Summary statistics"""
    total_registered: int
    total_running: int
    total_stopped: int
    total_orphans: int
    total_zombies: int
    total_memory_mb: float
    launcher_type: str


class ProcessesDebugResponse(BaseModel):
    """Complete debug response"""
    registered_processes: Dict[str, ProcessInfo]
    orphan_processes: List[OrphanProcess]
    zombie_processes: List[ZombieProcess]
    summary: ProcessesSummary


def create_debug_processes_router(process_manager_adapter) -> APIRouter:
    """
    Create debug processes router

    Args:
        process_manager_adapter: ProcessManagerAdapter instance

    Returns:
        Configured router
    """
    router = APIRouter(prefix="/debug", tags=["debug"])

    @router.get("/processes", response_model=ProcessesDebugResponse)
    async def get_processes_debug():
        """
        Get detailed information about all processes

        Returns comprehensive view of:
        - All registered processes with status
        - Orphan processes (not in registry)
        - Zombie processes
        - Summary statistics
        """
        try:
            pm = process_manager_adapter.pm

            # Get all registered processes
            registered = {}
            running_count = 0
            stopped_count = 0
            total_memory = 0.0

            for name, status in pm.status_all().items():
                registered[name] = ProcessInfo(
                    pid=status.pid,
                    name=status.name,
                    state=status.state,
                    cpu_percent=status.cpu_percent,
                    memory_mb=status.memory_mb,
                    memory_percent=status.memory_percent,
                    uptime_seconds=status.uptime_seconds,
                    is_healthy=status.is_healthy,
                    restart_count=status.restart_count,
                    memory_limit_mb=status.memory_limit_mb,
                    cpu_limit_percent=status.cpu_limit_percent,
                    port=pm.services[name].port if name in pm.services else None,
                    command=' '.join(pm.services[name].command[:3]) + '...' if name in pm.services else None
                )

                if status.state == 'running':
                    running_count += 1
                    total_memory += status.memory_mb
                elif status.state == 'stopped':
                    stopped_count += 1

            # Get orphans
            orphans = []
            for orphan in pm.get_orphans():
                orphans.append(OrphanProcess(
                    pid=orphan.pid,
                    name=orphan.name,
                    cmdline=orphan.cmdline[:100] + '...' if len(orphan.cmdline) > 100 else orphan.cmdline,
                    age_seconds=orphan.age_seconds,
                    memory_mb=orphan.memory_mb,
                    cpu_percent=orphan.cpu_percent
                ))

            # Get zombies
            zombies = []
            for zombie in pm.get_zombies():
                zombies.append(ZombieProcess(
                    pid=zombie.pid,
                    name=zombie.name,
                    ppid=zombie.ppid,
                    parent_name=zombie.parent_name
                ))

            # Summary
            summary = ProcessesSummary(
                total_registered=len(registered),
                total_running=running_count,
                total_stopped=stopped_count,
                total_orphans=len(orphans),
                total_zombies=len(zombies),
                total_memory_mb=round(total_memory, 2),
                launcher_type=pm.launcher.get_launcher_type()
            )

            return ProcessesDebugResponse(
                registered_processes=registered,
                orphan_processes=orphans,
                zombie_processes=zombies,
                summary=summary
            )

        except Exception as e:
            logger.error(f"Error getting process debug info: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/processes/cleanup-orphans")
    async def cleanup_orphans(dry_run: bool = False):
        """
        Cleanup orphan processes

        Args:
            dry_run: If true, only show what would be cleaned

        Returns:
            Number of orphans cleaned
        """
        try:
            count = process_manager_adapter.cleanup_orphans(dry_run=dry_run)
            return {
                "dry_run": dry_run,
                "orphans_cleaned": count,
                "message": f"{'Would clean' if dry_run else 'Cleaned'} {count} orphan processes"
            }
        except Exception as e:
            logger.error(f"Error cleaning orphans: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/processes/{service_name}")
    async def get_process_details(service_name: str):
        """
        Get detailed information about a specific process

        Args:
            service_name: Service name

        Returns:
            Detailed process information
        """
        try:
            status_dict = process_manager_adapter.get_service_status(service_name)
            return status_dict
        except Exception as e:
            logger.error(f"Error getting process details for {service_name}: {e}")
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

    @router.post("/processes/{service_name}/restart")
    async def restart_process(service_name: str):
        """
        Restart a specific process

        Args:
            service_name: Service name

        Returns:
            Success message
        """
        try:
            success = process_manager_adapter.restart_service(service_name)
            if success:
                return {"message": f"Service {service_name} restarted successfully"}
            else:
                raise HTTPException(status_code=500, detail=f"Failed to restart {service_name}")
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
