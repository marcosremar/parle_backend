"""Custom exceptions for process management"""


class ProcessManagementError(Exception):
    """Base exception for process management errors"""
    pass


class SystemdNotAvailableError(ProcessManagementError):
    """Raised when systemd is not available but required"""
    pass


class ServiceNotFoundError(ProcessManagementError):
    """Raised when service is not registered"""
    pass


class ServiceAlreadyRunningError(ProcessManagementError):
    """Raised when trying to start a service that's already running"""
    pass


class ServiceStartFailedError(ProcessManagementError):
    """Raised when service fails to start"""
    pass


class ServiceStopFailedError(ProcessManagementError):
    """Raised when service fails to stop"""
    pass


class RegistryLockError(ProcessManagementError):
    """Raised when PID registry lock cannot be acquired"""
    pass
