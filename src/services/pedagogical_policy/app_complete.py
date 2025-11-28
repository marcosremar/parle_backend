"""
Pedagogical Policy Service - Standalone FastAPI application
Motor de decisões pedagógicas e composição de prompts modulares
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .models import (
    ComposePromptRequest, ComposePromptResponse,
    StrategyInfo, Strategy, PromptContext
)
from .policy_engine import PolicyEngine
from .prompt_composer import PromptComposer

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "pedagogical_policy",
        "port": 8950,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("PEDAGOGICAL_POLICY_PORT", os.getenv("PORT", "8950")))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Pedagogical Policy Service",
    version="1.0.0",
    description="Service for pedagogical decision-making and prompt composition"
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

policy_engine = PolicyEngine()
prompt_composer = PromptComposer()

# ============================================================================
# API Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pedagogical_policy"}

@app.post("/api/prompt/compose", response_model=ComposePromptResponse)
async def compose_prompt(request: ComposePromptRequest):
    """
    Compor prompt pedagógico baseado no contexto
    
    Recebe contexto do estudante e retorna prompt completo
    adaptado ao nível, estratégia e estado emocional.
    """
    try:
        # Convert dict to PromptContext if needed
        if isinstance(request.context, dict):
            from .models import PromptContext, CEFRLevel, EmotionalState
            context = PromptContext(
                scenario=request.context.get("scenario"),
                cefr_level=CEFRLevel(request.context.get("cefr_level", "A1")),
                native_language=request.context.get("native_language", "en"),
                target_skill=request.context.get("target_skill"),
                mastery_probability=request.context.get("mastery_probability", 0.0),
                strategy=None,  # Will be decided by policy engine
                emotional_state=EmotionalState(request.context.get("emotional_state", "neutral")),
                conversation_history=request.context.get("conversation_history"),
                cefr_details=request.context.get("cefr_details") or {},
                interpretable_knowledge_state=request.context.get("interpretable_knowledge_state"),
                current_turn_analysis=request.context.get("current_turn_analysis"),
                session_analysis=request.context.get("session_analysis")
            )
        else:
            context = request.context
        
        result = prompt_composer.compose(context)
        return ComposePromptResponse(**result)
    except Exception as e:
        logger.error(f"Error composing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies", response_model=List[StrategyInfo])
async def get_strategies():
    """Listar todas as estratégias pedagógicas disponíveis"""
    strategies = [
        StrategyInfo(
            strategy=Strategy.TEACH,
            description="Ensino explícito para alunos iniciantes (mastery < 30%)",
            mastery_range="0% - 30%",
            use_cases=[
                "Aluno está aprendendo um conceito pela primeira vez",
                "Aluno demonstra dificuldade consistente",
                "Introdução de novo tópico"
            ]
        ),
        StrategyInfo(
            strategy=Strategy.REINFORCE,
            description="Reforço com prática guiada para alunos intermediários (mastery 30-70%)",
            mastery_range="30% - 70%",
            use_cases=[
                "Aluno está praticando um conceito conhecido",
                "Consolidação de conhecimento",
                "Aumento de fluência"
            ]
        ),
        StrategyInfo(
            strategy=Strategy.CHALLENGE,
            description="Desafio avançado para alunos proficientes (mastery > 70%)",
            mastery_range="70% - 100%",
            use_cases=[
                "Aluno domina o conceito básico",
                "Introdução de exceções e casos complexos",
                "Manutenção de proficiência"
            ]
        )
    ]
    return strategies

@app.post("/api/policy/decide")
async def decide_strategy(mastery_probability: float):
    """Decidir estratégia baseado em mastery probability"""
    try:
        strategy = policy_engine.decide_strategy(mastery_probability)
        instructions = policy_engine.get_strategy_instructions(strategy, mastery_probability)
        return {
            "strategy": strategy.value,
            "instructions": instructions,
            "mastery_probability": mastery_probability
        }
    except Exception as e:
        logger.error(f"Error deciding strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    logger.info(f"🚀 Starting Pedagogical Policy Service on port {config['service']['port']}")
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=True
    )

