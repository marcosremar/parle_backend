"""
Dependency Injection Container
Manages service registration and resolution with:
- Circular dependency detection
- Interface-based registration
- Scoped lifetime support
- Async factory support
"""

from typing import Any, Type, Callable, Dict, Optional, TypeVar, Generic, Set
import asyncio
import logging
import inspect
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected"""

    def __init__(self, dependency_chain: list):
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join(str(d) for d in dependency_chain)
        super().__init__(f"Circular dependency detected: {chain_str}")


class ServiceLifetime(Enum):
    """Service lifetime options"""
    SINGLETON = "singleton"      # Single instance for entire app lifetime
    TRANSIENT = "transient"      # New instance every time
    SCOPED = "scoped"           # New instance per scope (e.g., per request)


class ServiceDescriptor:
    """Describes a registered service"""
    
    def __init__(self,
                 interface: Type,
                 implementation: Optional[Type] = None,
                 factory: Optional[Callable] = None,
                 instance: Optional[Any] = None,
                 lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
        self.interface = interface
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        
        # Validate descriptor
        if not any([implementation, factory, instance]):
            raise ValueError("Must provide implementation, factory, or instance")


class DIContainer:
    """
    Dependency Injection Container
    Manages service registration and resolution with different lifetimes

    Features:
    - Automatic circular dependency detection
    - Interface-based registration (Protocol types)
    - Scoped lifetime support (per-request)
    - Async factory functions
    - Constructor dependency injection
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = asyncio.Lock()
        self._resolution_stack: Set[Type] = set()  # For circular dependency detection
        
    def register_singleton(self,
                          interface: Type[T],
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> None:
        """
        Register a singleton service

        Args:
            interface: Service interface/protocol (can be class or string)
            implementation: Implementation class
            factory: Factory function to create instance
            instance: Pre-created instance
        """
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[interface] = descriptor
        interface_name = interface if isinstance(interface, str) else interface.__name__
        logger.debug(f"Registered singleton: {interface_name}")
        
    def register_transient(self,
                          interface: Type[T],
                          implementation: Type[T]) -> None:
        """
        Register a transient service (new instance each time)

        Args:
            interface: Service interface/protocol
            implementation: Implementation class
        """
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        self._services[interface] = descriptor
        interface_name = interface if isinstance(interface, str) else interface.__name__
        logger.debug(f"Registered transient: {interface_name}")
        
    def register_scoped(self,
                       interface: Type[T],
                       implementation: Type[T]) -> None:
        """
        Register a scoped service (new instance per scope)

        Args:
            interface: Service interface/protocol
            implementation: Implementation class
        """
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED
        )
        self._services[interface] = descriptor
        interface_name = interface if isinstance(interface, str) else interface.__name__
        logger.debug(f"Registered scoped: {interface_name}")
        
    def resolve(self, interface: Type[T], scope_id: Optional[str] = None) -> T:
        """
        Resolve a service synchronously with circular dependency detection

        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier for scoped services

        Returns:
            Service instance

        Raises:
            CircularDependencyError: If circular dependency detected
            ValueError: If service not registered
        """
        if interface not in self._services:
            interface_name = interface if isinstance(interface, str) else interface.__name__
            raise ValueError(f"Service {interface_name} not registered")

        # Check for circular dependencies
        if interface in self._resolution_stack:
            chain = list(self._resolution_stack) + [interface]
            raise CircularDependencyError(chain)

        try:
            # Add to resolution stack
            self._resolution_stack.add(interface)

            descriptor = self._services[interface]

            # Handle based on lifetime
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                return self._resolve_singleton(descriptor)
            elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                return self._create_instance(descriptor)
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                return self._resolve_scoped(descriptor, scope_id)
            else:
                raise ValueError(f"Unknown lifetime: {descriptor.lifetime}")

        finally:
            # Remove from resolution stack
            self._resolution_stack.discard(interface)
            
    async def resolve_async(self, 
                           interface: Type[T], 
                           scope_id: Optional[str] = None) -> T:
        """
        Resolve a service asynchronously
        Useful when factory functions are async
        
        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier
            
        Returns:
            Service instance
        """
        async with self._lock:
            return self.resolve(interface, scope_id)
            
    def _resolve_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve a singleton service"""
        if descriptor.instance is not None:
            return descriptor.instance
            
        # Create instance if not exists
        instance = self._create_instance(descriptor)
        descriptor.instance = instance
        return instance
        
    def _resolve_scoped(self, 
                       descriptor: ServiceDescriptor,
                       scope_id: Optional[str]) -> Any:
        """Resolve a scoped service"""
        if scope_id is None:
            scope_id = "default"
            
        # Get or create scope
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
            
        scope = self._scoped_instances[scope_id]
        
        # Get or create instance in scope
        if descriptor.interface not in scope:
            scope[descriptor.interface] = self._create_instance(descriptor)
            
        return scope[descriptor.interface]
        
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of a service"""
        if descriptor.instance is not None:
            return descriptor.instance
            
        if descriptor.factory is not None:
            return descriptor.factory()
            
        if descriptor.implementation is not None:
            # Try to resolve constructor dependencies
            return self._create_with_dependencies(descriptor.implementation)
            
        raise ValueError(f"Cannot create instance for {descriptor.interface.__name__}")
        
    def _create_with_dependencies(self, implementation: Type) -> Any:
        """
        Create instance with automatic dependency injection
        Inspects constructor and resolves dependencies

        Features:
        - Resolves dependencies by type annotation
        - Supports Protocol interfaces (ILogger, IMetricsCollector, etc.)
        - Handles optional dependencies with defaults
        - Detects circular dependencies automatically
        """
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        kwargs = {}

        # Resolve each parameter
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Try to resolve by type annotation
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation

                # Handle Optional[T] annotations (extract T)
                if hasattr(param_type, '__origin__') and param_type.__origin__ is type(Optional):
                    # Extract the actual type from Optional[T]
                    import typing
                    param_type = typing.get_args(param_type)[0]

                # Check if this type is registered
                if param_type in self._services:
                    try:
                        kwargs[param_name] = self.resolve(param_type)
                        logger.debug(f"Resolved {param_name}: {param_type.__name__} for {implementation.__name__}")
                    except CircularDependencyError as e:
                        logger.error(f"Circular dependency while resolving {param_name}: {e}")
                        raise
                elif param.default == inspect.Parameter.empty:
                    # Required parameter but not registered - log warning but continue
                    # This allows for manual parameter passing if needed
                    logger.warning(
                        f"Cannot auto-resolve {param_name}: {param_type} for {implementation.__name__} "
                        f"(not registered in container)"
                    )

        return implementation(**kwargs)
        
    async def initialize_all(self) -> None:
        """
        Initialize all singleton services that have an initialize method
        """
        for descriptor in self._services.values():
            if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance:
                if hasattr(descriptor.instance, 'initialize'):
                    logger.info(f"Initializing {descriptor.interface.__name__}")
                    await descriptor.instance.initialize()
                    
    async def cleanup_all(self) -> None:
        """
        Cleanup all services that have a cleanup method
        """
        for descriptor in self._services.values():
            if descriptor.instance and hasattr(descriptor.instance, 'cleanup'):
                logger.info(f"Cleaning up {descriptor.interface.__name__}")
                await descriptor.instance.cleanup()
                
    def clear_scope(self, scope_id: str) -> None:
        """
        Clear all instances in a scope
        
        Args:
            scope_id: Scope identifier
        """
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
            logger.debug(f"Cleared scope: {scope_id}")
            
    def get_all_registered(self) -> Dict[str, str]:
        """
        Get all registered services for debugging

        Returns:
            Dictionary of interface names to lifetime
        """
        result = {}
        for iface, desc in self._services.items():
            iface_name = iface if isinstance(iface, str) else iface.__name__
            result[iface_name] = desc.lifetime.value
        return result

    def validate_dependencies(self) -> Dict[str, Any]:
        """
        Validate all registered services can be resolved

        Returns:
            Validation report with:
            - valid: List of services that can be resolved
            - invalid: List of services with missing dependencies
            - circular: List of circular dependency chains
        """
        report = {
            "valid": [],
            "invalid": [],
            "circular": [],
            "total": len(self._services)
        }

        for interface in self._services.keys():
            interface_name = interface if isinstance(interface, str) else interface.__name__

            try:
                # Try to resolve (will trigger dependency resolution)
                # For singletons, this will create the instance
                # For transient/scoped, we just check if it can be created
                descriptor = self._services[interface]

                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    if descriptor.instance is None:
                        # Not yet created - try to create
                        self.resolve(interface)
                    report["valid"].append(interface_name)
                else:
                    # For transient/scoped, just validate dependencies exist
                    if descriptor.implementation:
                        # Check constructor dependencies
                        sig = inspect.signature(descriptor.implementation.__init__)
                        for param_name, param in sig.parameters.items():
                            if param_name == 'self':
                                continue
                            if param.annotation != inspect.Parameter.empty:
                                param_type = param.annotation
                                if param_type in self._services:
                                    continue
                                elif param.default == inspect.Parameter.empty:
                                    report["invalid"].append({
                                        "service": interface_name,
                                        "missing_dependency": param_name,
                                        "type": str(param_type)
                                    })
                                    break
                        else:
                            report["valid"].append(interface_name)

            except CircularDependencyError as e:
                report["circular"].append({
                    "service": interface_name,
                    "chain": [str(t) for t in e.dependency_chain]
                })
            except Exception as e:
                report["invalid"].append({
                    "service": interface_name,
                    "error": str(e)
                })

        return report

    def get_dependency_graph(self) -> Dict[str, list]:
        """
        Get dependency graph for all registered services

        Returns:
            Dictionary mapping service names to their dependencies
        """
        graph = {}

        for interface, descriptor in self._services.items():
            interface_name = interface if isinstance(interface, str) else interface.__name__
            dependencies = []

            if descriptor.implementation:
                # Analyze constructor dependencies
                sig = inspect.signature(descriptor.implementation.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    if param.annotation != inspect.Parameter.empty:
                        param_type = param.annotation
                        if param_type in self._services:
                            dep_name = param_type if isinstance(param_type, str) else param_type.__name__
                            dependencies.append({
                                "parameter": param_name,
                                "type": dep_name,
                                "required": param.default == inspect.Parameter.empty
                            })

            graph[interface_name] = dependencies

        return graph