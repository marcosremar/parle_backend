"""
Tutoring Router - Student Model, Diagnostic, Pedagogical Policy, Learning Path
"""

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

try:
    from src.services.student_model.app_complete import app as student_model_app
    from src.services.diagnostic_module.app_complete import app as diagnostic_app
    from src.services.pedagogical_policy.app_complete import app as pedagogical_policy_app
    from src.services.learning_path.app_complete import app as learning_path_app
    
    logger.info("✅ Tutoring routers ready")
except ImportError as e:
    logger.warning(f"⚠️  Some tutoring routers not available: {e}")

@router.get("/health")
async def health():
    """Health check for tutoring services"""
    return {"status": "ok", "services": ["student_model", "diagnostic_module", "pedagogical_policy", "learning_path"]}
