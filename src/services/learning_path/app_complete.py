"""
Learning Path Navigator Service - Standalone FastAPI application
Navegação de caminhos de aprendizado e spaced repetition
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from loguru import logger
import aiohttp
from datetime import datetime
try:
    from dateutil.parser import parse
except ImportError:
    # Fallback if dateutil not available
    def parse(date_string):
        from datetime import datetime
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .models import NextSkillResponse, ReviewSkillsResponse, ReviewSkillResponse
from .navigator import LearningPathNavigator

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "learning_path",
        "port": 8970,
        "host": "0.0.0.0"
    }
}

def get_config():
    """Get service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("LEARNING_PATH_PORT", os.getenv("PORT", "8970")))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Learning Path Navigator Service",
    version="1.0.0",
    description="Service for learning path navigation and spaced repetition"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Service Instances
# ============================================================================

navigator = LearningPathNavigator()
student_model_url = os.getenv("STUDENT_MODEL_SERVICE_URL", "http://localhost:8900")

# ============================================================================
# Helper Functions
# ============================================================================

async def get_student_skills(user_id: str) -> list:
    """Buscar skills do estudante do Student Model service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{student_model_url}/api/student/{user_id}/skills") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("skills", [])
                return []
    except Exception as e:
        logger.error(f"Error fetching student skills: {e}")
        return []

async def get_student_profile(user_id: str) -> dict:
    """Buscar perfil do estudante"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{student_model_url}/api/student/{user_id}/profile") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"cefr_level": "A1"}
    except Exception as e:
        logger.error(f"Error fetching student profile: {e}")
        return {"cefr_level": "A1"}

# ============================================================================
# API Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "learning_path"}

@app.get("/api/path/{user_id}/next", response_model=NextSkillResponse)
async def get_next_skill(user_id: str):
    """
    Obter próxima habilidade recomendada para o estudante
    
    Prioridades:
    1. Habilidades com mastery baixo (< 30%)
    2. Habilidades na ZPD (prontas para aprender)
    3. Habilidades em aprendizado (30-70%)
    """
    try:
        # Buscar skills do estudante
        skill_masteries = await get_student_skills(user_id)
        
        # Buscar perfil para nível CEFR
        profile = await get_student_profile(user_id)
        cefr_level = profile.get("cefr_level", "A1")
        
        # Converter para formato esperado pelo navigator
        mastery_list = [
            {
                "skill_id": s["skill_id"],
                "mastery_probability": s["mastery_probability"],
                "skill_name": s.get("skill_name", s["skill_id"]),
                "category": s.get("category", "unknown"),
                "last_practiced": s.get("last_practiced"),
                "success_rate": s.get("successes", 0) / max(s.get("attempts", 1), 1)
            }
            for s in skill_masteries
        ]
        
        # Obter próxima habilidade
        next_skill = navigator.get_next_skill(
            user_id,
            mastery_list,
            cefr_level,
            all_skills=None  # TODO: Buscar do catálogo de skills
        )
        
        if not next_skill:
            raise HTTPException(status_code=404, detail="No next skill found")
        
        return NextSkillResponse(**next_skill)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting next skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/path/{user_id}/review", response_model=ReviewSkillsResponse)
async def get_review_skills(user_id: str, limit: int = 5):
    """
    Obter habilidades que devem ser revisadas (spaced repetition)
    """
    try:
        # Buscar skills do estudante
        skill_masteries = await get_student_skills(user_id)
        
        # Converter para formato esperado
        mastery_list = [
            {
                "skill_id": s["skill_id"],
                "mastery_probability": s["mastery_probability"],
                "skill_name": s.get("skill_name", s["skill_id"]),
                "last_practiced": s.get("last_practiced"),
                "success_rate": s.get("successes", 0) / max(s.get("attempts", 1), 1)
            }
            for s in skill_masteries
        ]
        
        # Obter habilidades para revisar
        review_skills = navigator.get_review_skills(mastery_list, limit)
        
        # Converter para formato de resposta
        review_list = [
            ReviewSkillResponse(**skill)
            for skill in review_skills
        ]
        
        return ReviewSkillsResponse(
            review_skills=review_list,
            total=len(review_list)
        )
    except Exception as e:
        logger.error(f"Error getting review skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    logger.info(f"🚀 Starting Learning Path Navigator Service on port {config['service']['port']}")
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=True
    )

