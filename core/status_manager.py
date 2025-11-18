"""
Status Manager - Gestão centralizada de status de todos os módulos
Permite monitoramento independente de cada componente
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    """Status possíveis de um módulo"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ModuleHealth:
    """Health check result for a module"""
    name: str
    status: ModuleStatus
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStatus:
    """Status completo do sistema"""
    timestamp: datetime
    uptime_seconds: float
    total_requests: int
    active_sessions: int
    modules: Dict[str, ModuleHealth]
    system_health: bool
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'total_requests': self.total_requests,
            'active_sessions': self.active_sessions,
            'system_health': self.system_health,
            'warnings': self.warnings,
            'modules': {
                name: {
                    'name': health.name,
                    'status': health.status.value,
                    'is_healthy': health.is_healthy,
                    'last_check': health.last_check.isoformat(),
                    'response_time_ms': health.response_time_ms,
                    'error_message': health.error_message,
                    'metrics': health.metrics
                }
                for name, health in self.modules.items()
            }
        }


class StatusManager:
    """
    Gerenciador de status para todos os módulos
    Fornece visibilidade completa e independente
    """
    
    def __init__(self):
        """Initialize status manager"""
        self.modules = {}  # Registered modules
        self.start_time = time.time()
        self.total_requests = 0
        self.active_sessions = set()
        self._health_check_interval = 30  # seconds
        self._health_check_task = None
        self._module_status = {}  # Current status of each module
        
    def register_module(self, name: str, module: Any) -> None:
        """
        Register a module for status tracking
        
        Args:
            name: Module name
            module: Module instance (must have get_stats method)
        """
        if not hasattr(module, 'get_stats'):
            raise ValueError(f"Module {name} must have get_stats() method")
        
        self.modules[name] = module
        self._module_status[name] = ModuleStatus.NOT_INITIALIZED
        logger.info(f"📊 Registered module for status tracking: {name}")
    
    def unregister_module(self, name: str) -> None:
        """Unregister a module"""
        if name in self.modules:
            del self.modules[name]
            del self._module_status[name]
            logger.info(f"📊 Unregistered module: {name}")
    
    def update_module_status(self, name: str, status: ModuleStatus) -> None:
        """
        Update module status
        
        Args:
            name: Module name
            status: New status
        """
        if name in self._module_status:
            old_status = self._module_status[name]
            self._module_status[name] = status
            
            if old_status != status:
                logger.info(f"📊 Module {name} status changed: {old_status.value} → {status.value}")
    
    async def check_module_health(self, name: str, module: Any) -> ModuleHealth:
        """
        Check health of a single module
        
        Args:
            name: Module name
            module: Module instance
            
        Returns:
            ModuleHealth object
        """
        start_time = time.time()
        error_message = None
        is_healthy = False
        metrics = {}
        
        try:
            # Get module stats
            stats = module.get_stats()
            metrics = stats
            
            # Check if initialized
            is_initialized = stats.get('is_initialized', False)
            
            # Check for errors
            errors = stats.get('errors', 0)
            total_requests = stats.get('total_requests', 0)
            
            # Determine health
            if not is_initialized:
                status = ModuleStatus.NOT_INITIALIZED
            elif errors > 0 and total_requests > 0:
                error_rate = errors / total_requests
                if error_rate > 0.5:
                    status = ModuleStatus.ERROR
                    error_message = f"High error rate: {error_rate:.1%}"
                elif error_rate > 0.1:
                    status = ModuleStatus.DEGRADED
                    error_message = f"Elevated error rate: {error_rate:.1%}"
                else:
                    status = ModuleStatus.READY
                    is_healthy = True
            else:
                status = ModuleStatus.READY
                is_healthy = True
            
            # Update module status
            self._module_status[name] = status
            
        except Exception as e:
            status = ModuleStatus.ERROR
            error_message = str(e)
            logger.error(f"❌ Health check failed for {name}: {e}")
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return ModuleHealth(
            name=name,
            status=status,
            is_healthy=is_healthy,
            last_check=datetime.now(),
            response_time_ms=response_time_ms,
            error_message=error_message,
            metrics=metrics
        )
    
    async def get_system_status(self) -> SystemStatus:
        """
        Get complete system status
        
        Returns:
            SystemStatus object
        """
        # Check all modules
        module_health = {}
        health_checks = []
        
        for name, module in self.modules.items():
            health_checks.append(self.check_module_health(name, module))
        
        # Run health checks in parallel
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        for i, (name, _) in enumerate(self.modules.items()):
            if isinstance(results[i], Exception):
                module_health[name] = ModuleHealth(
                    name=name,
                    status=ModuleStatus.ERROR,
                    is_healthy=False,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message=str(results[i])
                )
            else:
                module_health[name] = results[i]
        
        # Determine system health
        system_health = all(h.is_healthy for h in module_health.values())
        
        # Collect warnings
        warnings = []
        for name, health in module_health.items():
            if health.status == ModuleStatus.DEGRADED:
                warnings.append(f"{name}: {health.error_message}")
            elif health.status == ModuleStatus.ERROR:
                warnings.append(f"{name}: ERROR - {health.error_message}")
        
        # Calculate uptime
        uptime = time.time() - self.start_time
        
        return SystemStatus(
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            total_requests=self.total_requests,
            active_sessions=len(self.active_sessions),
            modules=module_health,
            system_health=system_health,
            warnings=warnings
        )
    
    async def get_module_status(self, module_name: str) -> Optional[ModuleHealth]:
        """
        Get status of a specific module
        
        Args:
            module_name: Name of the module
            
        Returns:
            ModuleHealth or None if not found
        """
        if module_name not in self.modules:
            return None
        
        return await self.check_module_health(
            module_name, 
            self.modules[module_name]
        )
    
    def increment_requests(self) -> None:
        """Increment total request counter"""
        self.total_requests += 1
    
    def add_session(self, session_id: str) -> None:
        """Add active session"""
        self.active_sessions.add(session_id)
    
    def remove_session(self, session_id: str) -> None:
        """Remove active session"""
        self.active_sessions.discard(session_id)
    
    async def start_health_monitoring(self, interval: int = 30) -> None:
        """
        Start background health monitoring
        
        Args:
            interval: Check interval in seconds
        """
        self._health_check_interval = interval
        
        async def monitor():
            while True:
                try:
                    await asyncio.sleep(self._health_check_interval)
                    status = await self.get_system_status()
                    
                    if not status.system_health:
                        logger.warning(f"⚠️ System health check failed: {status.warnings}")
                    else:
                        logger.debug(f"✅ System healthy - {len(self.modules)} modules OK")
                        
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        self._health_check_task = asyncio.create_task(monitor())
        logger.info(f"🏥 Started health monitoring (interval: {interval}s)")
    
    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("🏥 Stopped health monitoring")
    
    def get_module_names(self) -> List[str]:
        """Get list of registered module names"""
        return list(self.modules.keys())
    
    def is_module_healthy(self, module_name: str) -> bool:
        """
        Quick check if module is healthy
        
        Args:
            module_name: Module name
            
        Returns:
            True if healthy, False otherwise
        """
        status = self._module_status.get(module_name, ModuleStatus.NOT_INITIALIZED)
        return status == ModuleStatus.READY
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics for all modules
        
        Returns:
            Dictionary with detailed metrics
        """
        metrics = {
            'system': {
                'uptime_seconds': time.time() - self.start_time,
                'total_requests': self.total_requests,
                'active_sessions': len(self.active_sessions),
                'modules_count': len(self.modules)
            },
            'modules': {}
        }
        
        for name, module in self.modules.items():
            try:
                stats = module.get_stats()
                metrics['modules'][name] = stats
            except Exception as e:
                metrics['modules'][name] = {
                    'error': str(e),
                    'status': self._module_status.get(name, ModuleStatus.ERROR).value
                }
        
        return metrics