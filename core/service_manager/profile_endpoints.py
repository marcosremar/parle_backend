"""
Profile Management Endpoints
API endpoints for managing service execution profiles
"""

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError, BaseModel
from typing import List, Dict, Any, Optional
import logging
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/profiles", tags=["profiles"])


class ProfileInfo(BaseModel):
    """Profile information"""
    name: str
    description: str
    enabled_services_count: int
    restrictions: Dict[str, Any]


class ProfileDetailResponse(BaseModel):
    """Detailed profile information"""
    name: str
    description: str
    enabled_services: List[str]
    service_overrides: Dict[str, Dict[str, Any]]
    composite_services: Dict[str, Dict[str, Any]]
    restrictions: Dict[str, Any]


class ProfileActivateRequest(BaseModel):
    """Request to activate a profile"""
    restart_required: bool = True


class ValidationMessageResponse(BaseModel):
    """Validation message"""
    severity: str
    message: str
    service_id: Optional[str] = None


class ValidationResponse(BaseModel):
    """Profile validation result"""
    valid: bool
    messages: List[ValidationMessageResponse]


def setup_profile_endpoints(app, manager):
    """
    Setup profile management endpoints

    Args:
        app: FastAPI application
        manager: Service Manager instance
    """

    @router.get("", response_model=List[ProfileInfo])
    async def list_profiles():
        """List all available profiles"""
        try:
            profile_names = manager.profile_manager.list_profiles()
            profiles_info = []

            for name in profile_names:
                profile = manager.profile_manager.get_profile(name)
                if profile:
                    profiles_info.append(ProfileInfo(
                        name=profile.name,
                        description=profile.description,
                        enabled_services_count=len(profile.enabled_services),
                        restrictions={
                            "max_gpu_memory_mb": profile.restrictions.max_gpu_memory_mb,
                            "allow_remote_services": profile.restrictions.allow_remote_services,
                            "allow_gpu_services": profile.restrictions.allow_gpu_services,
                            "allow_composite": profile.restrictions.allow_composite
                        }
                    ))

            return profiles_info

        except Exception as e:
            logger.error(f"❌ Error listing profiles: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/active")
    async def get_active_profile():
        """Get currently active profile"""
        try:
            profile = manager.profile_manager.get_active_profile()
            if not profile:
                return {
                    "active": False,
                    "message": "No active profile"
                }

            return {
                "active": True,
                "name": profile.name,
                "description": profile.description,
                "enabled_services_count": len(profile.enabled_services)
            }

        except Exception as e:
            logger.error(f"❌ Error getting active profile: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/active/services")
    async def get_active_profile_services():
        """Get services enabled in active profile"""
        try:
            profile = manager.profile_manager.get_active_profile()
            if not profile:
                return {
                    "services": [],
                    "message": "No active profile"
                }

            return {
                "profile": profile.name,
                "enabled_services": profile.enabled_services,
                "total": len(profile.enabled_services)
            }

        except Exception as e:
            logger.error(f"❌ Error getting active profile services: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/{name}", response_model=ProfileDetailResponse)
    async def get_profile(name: str):
        """Get detailed information about a profile"""
        try:
            profile = manager.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            return ProfileDetailResponse(
                name=profile.name,
                description=profile.description,
                enabled_services=profile.enabled_services,
                service_overrides=profile.service_overrides,
                composite_services=profile.composite_services,
                restrictions={
                    "max_gpu_memory_mb": profile.restrictions.max_gpu_memory_mb,
                    "allow_remote_services": profile.restrictions.allow_remote_services,
                    "allow_gpu_services": profile.restrictions.allow_gpu_services,
                    "require_gpu_services_remote": profile.restrictions.require_gpu_services_remote,
                    "require_all_internal": profile.restrictions.require_all_internal,
                    "allow_composite": profile.restrictions.allow_composite,
                    "max_child_modules": profile.restrictions.max_child_modules,
                    "max_concurrent_services": profile.restrictions.max_concurrent_services
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting profile {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/{name}/activate")
    async def activate_profile(name: str, request: ProfileActivateRequest):
        """
        Activate a profile

        Note: Requires service restart to take effect
        """
        try:
            # Check if profile exists
            profile = manager.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            # Validate profile before activating
            validation = manager.profile_manager.validate_profile(name)
            if not validation.valid:
                error_messages = [msg.message for msg in validation.messages if msg.severity.value == "error"]
                return {
                    "success": False,
                    "error": "Profile validation failed",
                    "validation_errors": error_messages
                }

            # Set active profile and persist to YAML
            success = manager.profile_manager.set_active_profile(name, persist=True)

            if success:
                return {
                    "success": True,
                    "message": f"Profile {name} activated and saved to config",
                    "restart_required": request.restart_required,
                    "note": "Service Manager restart required for changes to take effect",
                    "persisted": True
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to activate profile"
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error activating profile {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/{name}/validate", response_model=ValidationResponse)
    async def validate_profile(name: str):
        """Validate a profile configuration"""
        try:
            # Check if profile exists
            profile = manager.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            # Validate
            validation = manager.profile_manager.validate_profile(name)

            messages = [
                ValidationMessageResponse(
                    severity=msg.severity.value,
                    message=msg.message,
                    service_id=msg.service_id
                )
                for msg in validation.messages
            ]

            return ValidationResponse(
                valid=validation.valid,
                messages=messages
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error validating profile {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/{name}/restrictions")
    async def get_profile_restrictions(name: str):
        """Get profile restrictions"""
        try:
            restrictions = manager.profile_manager.get_restrictions(name)
            if not restrictions:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            return {
                "profile": name,
                "restrictions": {
                    "max_gpu_memory_mb": restrictions.max_gpu_memory_mb,
                    "allow_remote_services": restrictions.allow_remote_services,
                    "allow_gpu_services": restrictions.allow_gpu_services,
                    "require_gpu_services_remote": restrictions.require_gpu_services_remote,
                    "require_all_internal": restrictions.require_all_internal,
                    "allow_composite": restrictions.allow_composite,
                    "max_child_modules": restrictions.max_child_modules,
                    "max_concurrent_services": restrictions.max_concurrent_services
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting restrictions for {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/{name}/activate-hot")
    async def activate_profile_hot_reload(name: str):
        """
        HOT RELOAD: Activate profile with immediate reload (no restart required)

        This endpoint:
        1. Validates the new profile
        2. Shuts down all running services
        3. Resets GlobalContext singleton
        4. Creates new GlobalContext with new profile
        5. Restarts all services with new profile

        ⚠️ WARNING: This causes brief downtime (~15-30 seconds)
        """
        try:
            # Check if profile exists
            profile = manager.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            # Validate profile before activating
            validation = manager.profile_manager.validate_profile(name)
            if not validation.valid:
                error_messages = [msg.message for msg in validation.messages if msg.severity.value == "error"]
                return {
                    "success": False,
                    "error": "Profile validation failed",
                    "validation_errors": error_messages
                }

            logger.info(f"🔥 HOT RELOAD: Activating profile '{name}'...")

            # Step 1: Shutdown all services
            logger.info("   1️⃣ Shutting down all services...")
            await manager.shutdown_all_services()

            # Step 2: Reset GlobalContext singleton
            logger.info("   2️⃣ Resetting GlobalContext singleton...")
            from src.core.context.global_context import GlobalContext
            GlobalContext.reset()

            # Step 3: Set new active profile
            logger.info(f"   3️⃣ Setting active profile to '{name}'...")
            success = manager.profile_manager.set_active_profile(name, persist=True)

            if not success:
                logger.error("   ❌ Failed to set active profile")
                return {
                    "success": False,
                    "error": "Failed to set active profile"
                }

            # Step 4: Create new GlobalContext with new profile
            logger.info(f"   4️⃣ Creating GlobalContext with profile '{name}'...")
            global_ctx = GlobalContext.get_instance(profile_name=name)
            await global_ctx.initialize()

            # Step 5: Restart all services with new profile
            logger.info("   5️⃣ Starting services with new profile...")
            await manager.start_all_services()

            logger.info(f"✅ HOT RELOAD complete: Profile '{name}' activated")

            return {
                "success": True,
                "message": f"Profile '{name}' activated with hot reload",
                "previous_profile": manager.profile_manager.get_active_profile().name,
                "new_profile": name,
                "downtime_seconds": "~15-30",
                "services_restarted": True,
                "globalcontext_reset": True,
                "steps_completed": [
                    "Services shutdown",
                    "GlobalContext reset",
                    "Profile changed",
                    "GlobalContext recreated",
                    "Services restarted"
                ]
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error in hot reload for profile {name}: {e}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Hot reload failed: {str(e)}")

    @router.post("/reload")
    async def reload_profiles():
        """
        Reload profiles from YAML without restarting Service Manager

        Note: This reloads profile definitions but does NOT restart services.
        To apply profile changes to running services, restart is required.
        """
        try:
            from src.core.managers.profile_manager import get_profile_manager

            # Reload profile manager
            pm = get_profile_manager(reload=True)

            # Update manager's reference
            manager.profile_manager = pm

            active_profile = pm.get_active_profile()

            return {
                "success": True,
                "message": "Profiles reloaded from disk",
                "profiles_count": len(pm.list_profiles()),
                "active_profile": active_profile.name if active_profile else None,
                "note": "Service restart required to apply changes to running services"
            }

        except Exception as e:
            logger.error(f"❌ Error reloading profiles: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/{name}/dependencies/validate")
    async def validate_dependencies(name: str):
        """
        Validate service dependencies for a profile

        Checks that all required service dependencies are satisfied
        """
        try:
            # Check if profile exists
            profile = manager.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

            # Get service dependencies from Service Manager
            service_dependencies = manager.SERVICE_DEPENDENCIES

            # Validate dependencies
            all_satisfied, missing = manager.profile_manager.validate_service_dependencies(
                service_dependencies, name
            )

            if all_satisfied:
                return {
                    "valid": True,
                    "profile": name,
                    "message": "All service dependencies satisfied"
                }
            else:
                return {
                    "valid": False,
                    "profile": name,
                    "message": f"Missing {len(missing)} dependencies",
                    "missing_dependencies": missing
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error validating dependencies for {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Include router in app
    app.include_router(router)
    logger.info("✅ Profile management endpoints added successfully")
    logger.info("   - GET  /profiles - List all profiles")
    logger.info("   - GET  /profiles/active - Get active profile")
    logger.info("   - GET  /profiles/{name} - Get profile details")
    logger.info("   - POST /profiles/{name}/activate - Activate profile (requires restart)")
    logger.info("   - POST /profiles/{name}/activate-hot - 🔥 HOT RELOAD (immediate, ~15s downtime)")
    logger.info("   - POST /profiles/{name}/validate - Validate profile")
    logger.info("   - POST /profiles/reload - Reload profiles from disk")
    logger.info("   - POST /profiles/{name}/dependencies/validate - Validate dependencies")
