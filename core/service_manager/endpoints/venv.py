"""
Virtual Environment Management Endpoints
Handles venv listing and bulk installation for services
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import logging

from .models import VenvListResponse, VenvInstallRequest

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/venvs", tags=["Virtual Environments"])


def create_venv_router(manager):
    """
    Create and configure venv management router with manager dependency.

    Args:
        manager: ServiceManager instance

    Returns:
        Configured APIRouter
    """

    @router.get("/list", response_model=VenvListResponse)
    async def list_all_venvs():
        """
        List all service virtual environments

        Returns:
            VenvListResponse with all venvs
        """
        try:
            venv_list = manager.venv_manager.list_venvs()

            venvs_details = []
            for service_id in venv_list:
                python_exe = manager.venv_manager.get_python_executable(service_id)
                venv_path = manager.venv_manager.get_venv_path(service_id)

                # Get venv size
                venv_size_mb = 0
                try:
                    if venv_path.exists():
                        venv_size_mb = sum(f.stat().st_size for f in venv_path.rglob('*') if f.is_file()) / (1024 * 1024)
                except Exception as e:
                    logger.debug(f"Could not calculate venv size for {service_id}: {e}")

                venvs_details.append({
                    "service_id": service_id,
                    "venv_path": str(venv_path),
                    "python_executable": str(python_exe) if python_exe else None,
                    "size_mb": round(venv_size_mb, 2)
                })

            return VenvListResponse(
                venvs=venvs_details,
                total=len(venvs_details)
            )

        except Exception as e:
            logger.error(f"❌ Error listing venvs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/install-all")
    async def install_all_venvs(request: VenvInstallRequest):
        """
        Install virtual environments for ALL services with venv_path configured

        Args:
            request: Installation configuration (applies to all services)

        Returns:
            Summary of installation results
        """
        try:
            logger.info("🚀 Installing venvs for all configured services")

            # Get all services with venv_path configured
            services_with_venv = []
            for service_id, info in manager.execution_config.services.items():
                if info.venv_path:
                    services_with_venv.append(service_id)

            if not services_with_venv:
                return {
                    "success": True,
                    "message": "No services configured with venv_path",
                    "total": 0,
                    "installed": 0,
                    "skipped": 0,
                    "failed": 0,
                    "results": []
                }

            logger.info(f"📋 Found {len(services_with_venv)} services with venv configured")

            results = []
            installed = 0
            skipped = 0
            failed = 0

            for service_id in services_with_venv:
                try:
                    # Check if venv exists and skip if not forcing
                    venv_exists = manager.venv_manager.venv_exists(service_id)

                    if venv_exists and not request.force:
                        logger.info(f"✓ {service_id}: Already exists (skipping)")
                        skipped += 1
                        results.append({
                            "service_id": service_id,
                            "status": "skipped",
                            "message": "Venv already exists (use force=true to reinstall)"
                        })
                        continue

                    # Remove if force and exists
                    if request.force and venv_exists:
                        logger.info(f"🗑️  {service_id}: Force reinstall - removing existing venv")
                        if not manager.venv_manager.remove_venv(service_id):
                            failed += 1
                            results.append({
                                "service_id": service_id,
                                "status": "failed",
                                "message": "Failed to remove existing venv"
                            })
                            continue

                    # Create venv
                    logger.info(f"🔨 {service_id}: Creating venv...")
                    if not manager.venv_manager.create_venv(service_id):
                        failed += 1
                        results.append({
                            "service_id": service_id,
                            "status": "failed",
                            "message": "Failed to create venv"
                        })
                        continue

                    # Install requirements if requested
                    install_msg = "Venv created successfully"
                    if request.install_deps:
                        requirements_file = None
                        if request.requirements_file:
                            requirements_file = Path(request.requirements_file)

                        if manager.venv_manager.install_requirements(service_id, requirements_file):
                            install_msg = "Venv created and requirements installed"
                        else:
                            install_msg = "Venv created (requirements installation failed)"

                    installed += 1
                    results.append({
                        "service_id": service_id,
                        "status": "installed",
                        "message": install_msg,
                        "venv_path": str(manager.venv_manager.get_venv_path(service_id))
                    })
                    logger.info(f"✅ {service_id}: {install_msg}")

                except Exception as e:
                    failed += 1
                    results.append({
                        "service_id": service_id,
                        "status": "failed",
                        "message": str(e)
                    })
                    logger.error(f"❌ {service_id}: {e}")

            return {
                "success": failed == 0,
                "message": f"Completed: {installed} installed, {skipped} skipped, {failed} failed",
                "total": len(services_with_venv),
                "installed": installed,
                "skipped": skipped,
                "failed": failed,
                "results": results
            }

        except Exception as e:
            logger.error(f"❌ Error installing all venvs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
