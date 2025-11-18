"""
Telemetry Decorators for BaseService Methods

Provides decorators to add automatic telemetry to service methods
that don't use FastAPI (e.g., BaseService subclasses).

Usage:
    from src.core.telemetry_decorators import telemetry

    class MyService(BaseService):
        @telemetry(operation="process_data")
        async def process_data(self, data: bytes) -> dict:
            # Your code here
            return result
"""

import time
import functools
import logging
import asyncio
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def telemetry(
    operation: str,
    log_args: bool = False,
    log_result: bool = False
):
    """
    Decorator to add telemetry to any function or method

    Args:
        operation: Name of the operation (e.g., "transcribe", "synthesize")
        log_args: Whether to log function arguments (default: False)
        log_result: Whether to log return value (default: False)

    Example:
        @telemetry(operation="transcribe")
        async def transcribe(self, audio: bytes) -> str:
            ...

    Logs:
        🔷 [SERVICE] Starting: operation
        ✅ [SERVICE] Completed: operation (123ms)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs) -> Any:
            # Get service name from self
            service_name = getattr(self, '__class__').__name__.replace('Service', '').upper()

            # ==========================================
            # TELEMETRY START
            # ==========================================
            start_time = time.time()

            # Log operation start
            log_msg = f"🔷 [{service_name}] Starting: {operation}"

            if log_args and args:
                # Log first arg size if it's bytes
                first_arg = args[0]
                if isinstance(first_arg, bytes):
                    log_msg += f" (input: {len(first_arg)} bytes)"
                elif isinstance(first_arg, str):
                    log_msg += f" (input: '{first_arg[:50]}...')"

            logger.info(log_msg)

            try:
                # Execute function
                result = await func(self, *args, **kwargs)

                # ==========================================
                # TELEMETRY END (SUCCESS)
                # ==========================================
                processing_time = (time.time() - start_time) * 1000

                log_msg = f"✅ [{service_name}] Completed: {operation} ({processing_time:.0f}ms)"

                if log_result and result:
                    if isinstance(result, dict):
                        log_msg += f" (result keys: {list(result.keys())[:5]})"
                    elif isinstance(result, bytes):
                        log_msg += f" (output: {len(result)} bytes)"
                    elif isinstance(result, str):
                        log_msg += f" (output: '{result[:50]}...')"

                logger.info(log_msg)

                # Add timing to result if it's a dict
                if isinstance(result, dict):
                    if 'metrics' not in result:
                        result['metrics'] = {}
                    result['metrics'][f'{service_name.lower()}_{operation}_ms'] = int(processing_time)

                return result

            except Exception as e:
                # ==========================================
                # TELEMETRY END (ERROR)
                # ==========================================
                processing_time = (time.time() - start_time) * 1000

                logger.error(
                    f"❌ [{service_name}] Failed: {operation} ({processing_time:.0f}ms) "
                    f"- Error: {str(e)}"
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs) -> Any:
            # Get service name from self
            service_name = getattr(self, '__class__').__name__.replace('Service', '').upper()

            # ==========================================
            # TELEMETRY START
            # ==========================================
            start_time = time.time()

            # Log operation start
            log_msg = f"🔷 [{service_name}] Starting: {operation}"

            if log_args and args:
                first_arg = args[0]
                if isinstance(first_arg, bytes):
                    log_msg += f" (input: {len(first_arg)} bytes)"
                elif isinstance(first_arg, str):
                    log_msg += f" (input: '{first_arg[:50]}...')"

            logger.info(log_msg)

            try:
                # Execute function
                result = func(self, *args, **kwargs)

                # ==========================================
                # TELEMETRY END (SUCCESS)
                # ==========================================
                processing_time = (time.time() - start_time) * 1000

                log_msg = f"✅ [{service_name}] Completed: {operation} ({processing_time:.0f}ms)"

                if log_result and result:
                    if isinstance(result, dict):
                        log_msg += f" (result keys: {list(result.keys())[:5]})"
                    elif isinstance(result, bytes):
                        log_msg += f" (output: {len(result)} bytes)"
                    elif isinstance(result, str):
                        log_msg += f" (output: '{result[:50]}...')"

                logger.info(log_msg)

                # Add timing to result if it's a dict
                if isinstance(result, dict):
                    if 'metrics' not in result:
                        result['metrics'] = {}
                    result['metrics'][f'{service_name.lower()}_{operation}_ms'] = int(processing_time)

                return result

            except Exception as e:
                # ==========================================
                # TELEMETRY END (ERROR)
                # ==========================================
                processing_time = (time.time() - start_time) * 1000

                logger.error(
                    f"❌ [{service_name}] Failed: {operation} ({processing_time:.0f}ms) "
                    f"- Error: {str(e)}"
                )
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience decorators for common operations
def telemetry_transcribe(func: Callable) -> Callable:
    """Telemetry decorator for transcribe operations"""
    return telemetry(operation="transcribe", log_args=True, log_result=True)(func)


def telemetry_synthesize(func: Callable) -> Callable:
    """Telemetry decorator for synthesize operations"""
    return telemetry(operation="synthesize", log_args=True, log_result=True)(func)


def telemetry_generate(func: Callable) -> Callable:
    """Telemetry decorator for generate operations"""
    return telemetry(operation="generate", log_args=True, log_result=True)(func)


def telemetry_process(func: Callable) -> Callable:
    """Telemetry decorator for general process operations"""
    return telemetry(operation="process", log_args=True, log_result=True)(func)
