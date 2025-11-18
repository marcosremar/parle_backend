"""
Configuration Management Endpoints

Configuration validation and reload functionality.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import yaml
from src.core.core_logging import setup_logging
from src.core.service_manager.endpoints.models import ConfigValidationResponse

# Setup logging
logger = setup_logging("endpoints-config", level="INFO")

# Create router
router = APIRouter(prefix="/config", tags=["Configuration"])

# Manager instance
_manager = None

def set_manager(manager):
    """Set the manager instance"""
    global _manager
    _manager = manager

def get_manager():
    """Get the manager instance"""
    return _manager


@router.get("/validate", response_model=ConfigValidationResponse)
async def validate_config():
    """
    Validate service_execution.yaml without applying changes

    Returns:
        Validation result with errors and warnings
    """
    try:
        from src.config.service_execution_config import get_service_execution_config

        manager = get_manager()
        # Dynamic config path detection (supports both /workspace and ~/.cache deployments)
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        config_path = project_root / "config" / "service_execution.yaml"

        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Configuration file not found")

        errors = []
        warnings = []

        # Try to load and parse YAML
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {str(e)}")
            return ConfigValidationResponse(
                valid=False,
                errors=errors,
                warnings=warnings
            )

        # Try to instantiate configuration
        try:
            test_config = get_service_execution_config(reload=True)
            services_count = len(test_config.services)

            # Check for locked services with changed execution mode
            if manager and hasattr(manager, 'execution_config'):
                for service_id, info in test_config.services.items():
                    if info.locked:
                        old_info = manager.execution_config.services.get(service_id)
                        if old_info and old_info.execution_mode != info.execution_mode:
                            errors.append(
                                f"Cannot change execution_mode for locked service: {service_id}"
                            )

            # Validate venv paths exist
            for service_id, info in test_config.services.items():
                if info.venv_path:
                    venv_path = Path(info.venv_path)
                    if not venv_path.exists():
                        warnings.append(
                            f"Virtual environment does not exist for {service_id}: {info.venv_path}"
                        )

        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
            return ConfigValidationResponse(
                valid=False,
                errors=errors,
                warnings=warnings
            )

        return ConfigValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_config():
    """
    Reload service_execution.yaml configuration

    Note: Does not restart running services, only updates configuration
    """
    try:
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="Manager not initialized")

        # Validate first
        validation = await validate_config()

        if not validation.valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Configuration validation failed",
                    "errors": validation.errors,
                    "warnings": validation.warnings
                }
            )

        # Reload configuration
        from src.config.service_execution_config import get_service_execution_config
        manager.execution_config = get_service_execution_config(reload=True)

        logger.info(f"✅ Configuration reloaded: {len(manager.execution_config.services)} services")

        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "services_count": len(manager.execution_config.services),
            "warnings": validation.warnings
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
