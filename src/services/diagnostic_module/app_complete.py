"""
Diagnostic Module Service - Standalone FastAPI application
Análise de erros e complexidade linguística
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .models import (
    AnalyzeTurnRequest, AnalyzeTurnResponse,
    EstimateLevelRequest, EstimateLevelResponse,
    AnalyzeSessionRequest, AnalyzeSessionResponse,
    ExtractSkillsRequest, ExtractSkillsResponse,
    CalibrateRequest, CalibrateResponse,
    CEFRLevel
)
from .analyzers import (
    GrammarAnalyzer, VocabularyAnalyzer,
    ComplexityAnalyzer, ProgressAnalyzer,
    SessionAnalyzer, ErrorRateAnalyzer
)
from .analyzers.calibration_manager import CalibrationManager
from .analyzers.feedback_generator import FeedbackGenerator
from .analyzers.task_relevance_analyzer import TaskRelevanceAnalyzer
from .analyzers.asr_metadata_analyzer import ASRMetadataAnalyzer
from .llm_client import DiagnosticLLMClient

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "speech_grader",
        "port": 8960,
        "host": "0.0.0.0"
    }
}

def get_config():
    """Get service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("DIAGNOSTIC_MODULE_PORT", os.getenv("PORT", "8960")))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Diagnostic Module Service",
    version="1.0.0",
    description="Service for analyzing student errors and linguistic complexity"
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

llm_client = DiagnosticLLMClient()
grammar_analyzer = GrammarAnalyzer(llm_client)
vocabulary_analyzer = VocabularyAnalyzer()
complexity_analyzer = ComplexityAnalyzer(llm_client)
progress_analyzer = ProgressAnalyzer()
session_analyzer = SessionAnalyzer()

# Phase 1-3: New analyzers
error_rate_analyzer = ErrorRateAnalyzer()
calibration_manager = CalibrationManager()
feedback_generator = FeedbackGenerator()
task_relevance_analyzer = TaskRelevanceAnalyzer()
asr_metadata_analyzer = ASRMetadataAnalyzer()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    session = aiohttp.ClientSession()
    await llm_client.initialize(session)
    logger.info("✅ Diagnostic Module initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if llm_client.session:
        await llm_client.session.close()

# ============================================================================
# API Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "speech_grader"}

@app.post("/api/diagnostic/analyze_turn", response_model=AnalyzeTurnResponse)
async def analyze_turn(request: AnalyzeTurnRequest):
    """
    Análise completa de um turno de conversa
    
    Analisa erros, complexidade e progresso
    """
    try:
        # Analisar erros gramaticais (com SINKT e features linguísticas)
        grammar_result = await grammar_analyzer.analyze(
            request.user_text,
            request.ai_text,
            request.valid_skills
        )
        
        grammar_errors = grammar_result.get("errors", [])
        correct_skills = grammar_result.get("correct_skills", [])
        linguistic_features = grammar_result.get("linguistic_features", {})
        semantic_skill_mapping = grammar_result.get("semantic_skill_mapping", {})
        
        # Fallback: Se linguistic_features estiver vazio, tentar extrair heuristicamente
        if not linguistic_features or len(linguistic_features) == 0:
            try:
                from src.services.student_model.skill_registry import extract_linguistic_features
                # Tentar extrair features do primeiro skill identificado ou do texto geral
                skill_id_for_extraction = None
                if grammar_errors and len(grammar_errors) > 0:
                    # Se há erros, usar o skill_id do primeiro erro
                    first_error = grammar_errors[0]
                    skill_id_for_extraction = first_error.skill_id if hasattr(first_error, 'skill_id') else first_error.get("skill_id") if isinstance(first_error, dict) else None
                elif correct_skills and len(correct_skills) > 0:
                    # Se não há erros mas há skills corretas, usar a primeira
                    skill_id_for_extraction = correct_skills[0]
                
                if skill_id_for_extraction:
                    linguistic_features = extract_linguistic_features(skill_id_for_extraction, {
                        "user_text": request.user_text,
                        "ai_text": request.ai_text
                    })
                else:
                    # Extração genérica baseada no texto (tentar verb_conjugation_past como padrão)
                    linguistic_features = extract_linguistic_features("verb_conjugation_past", {
                        "user_text": request.user_text,
                        "ai_text": request.ai_text
                    })
            except Exception as e:
                logger.debug(f"Fallback linguistic feature extraction failed: {e}")
                linguistic_features = {}
        
        # Analisar vocabulário
        vocabulary_errors = vocabulary_analyzer.analyze(request.user_text)
        
        # Combinar erros
        all_errors = grammar_errors + vocabulary_errors
        
        # Analisar complexidade
        complexity = await complexity_analyzer.analyze(request.user_text)
        
        # Análise de progresso (por enquanto sem dados anteriores)
        progress = progress_analyzer.analyze(
            current_errors=[e.dict() if hasattr(e, 'dict') else e for e in all_errors],
            previous_errors=None
        )
        
        # Determinar se habilidade em foco foi usada corretamente
        # Por enquanto, assume que sim se não houver erros relacionados
        target_skill_correct = len(all_errors) == 0
        
        # Criar resumo
        error_count = len(all_errors)
        if error_count == 0:
            summary = "Nenhum erro detectado. Excelente!"
        elif error_count == 1:
            summary = f"1 erro detectado: {all_errors[0].error_type.value}"
        else:
            summary = f"{error_count} erros detectados. Principais tipos: {', '.join(set(e.error_type.value for e in all_errors[:3]))}"
        
        return AnalyzeTurnResponse(
            errors=all_errors,
            correct_skills=correct_skills,
            linguistic_features=linguistic_features,
            semantic_skill_mapping=semantic_skill_mapping,
            complexity=complexity,
            progress=progress,
            target_skill_correct=target_skill_correct,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Error analyzing turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnostic/estimate_level", response_model=EstimateLevelResponse)
async def estimate_level(request: EstimateLevelRequest):
    """
    Estimar nível CEFR baseado em um texto com suporte a múltiplos recursos.
    
    Suporta:
    - Análise textual básica (sempre)
    - Error-rate features (se asr_transcription + expected_text)
    - Acoustic features (se audio_path)
    - ASR metadata analysis (se asr_metadata)
    - Task relevance (se question)
    - AKT alignment (se user_id)
    """
    try:
        # Base complexity analysis
        analysis_dict = {}
        
        # Choose analysis method based on available data
        if request.audio_path:
            # Full analysis with audio
            analysis_dict = await complexity_analyzer.analyze_with_audio(
                request.text,
                request.audio_path,
                request.user_id
            )
        elif request.asr_transcription and request.expected_text:
            # Analysis with ASR error-rate features
            analysis_dict = await complexity_analyzer.analyze_with_asr(
                request.text,
                request.asr_transcription,
                request.expected_text,
                request.user_id
            )
        else:
            # Basic text analysis
            complexity = await complexity_analyzer.analyze(request.text, request.user_id)
            analysis_dict = {
                "estimated_cefr_level": complexity.estimated_cefr_level.value,
                "confidence": complexity.confidence,
                "vocabulary_complexity": complexity.vocabulary_complexity,
                "grammar_complexity": complexity.grammar_complexity,
                "sentence_length_avg": complexity.sentence_length_avg,
                "indicators": complexity.indicators,
                "reasoning": complexity.reasoning
            }
        
        # Extract base fields
        cefr_level_str = analysis_dict.get("estimated_cefr_level", "A1")
        try:
            cefr_level = CEFRLevel(cefr_level_str)
        except ValueError:
            cefr_level = CEFRLevel.A1
            logger.warning(f"Invalid CEFR level: {cefr_level_str}, defaulting to A1")
        
        # Phase 3: Add ASR metadata analysis if available
        asr_metadata_analysis = None
        if request.asr_metadata:
            asr_metadata_analysis = asr_metadata_analyzer.analyze_asr_metadata(
                request.asr_metadata
            )
        
        # Phase 3: Add task relevance if question provided
        task_relevance_data = None
        topic_coverage_data = None
        if request.question:
            task_relevance_data = task_relevance_analyzer.calculate_task_relevance(
                request.question,
                request.text,
                request.exemplar
            )
            topic_coverage_data = task_relevance_analyzer.calculate_topic_coverage(
                request.question,
                request.text
            )
        
        # Phase 3: Apply calibration if weights available
        breakdown = analysis_dict.get("breakdown")
        if breakdown and calibration_manager.has_weights():
            breakdown = calibration_manager.apply_calibration(breakdown)
            analysis_dict["breakdown"] = breakdown
        
        # Phase 3: Generate structured feedback
        feedback_data = None
        try:
            feedback_data = await feedback_generator.generate_structured_feedback(
                analysis_dict,
                request.user_id or ""
            )
        except Exception as e:
            logger.warning(f"Failed to generate feedback: {e}")
        
        # Phase 3: Get AKT alignment if user_id provided
        akt_alignment = None
        if request.user_id:
            try:
                akt_alignment = await progress_analyzer.get_akt_alignment(
                    request.user_id,
                    cefr_level_str
                )
            except Exception as e:
                logger.warning(f"Failed to get AKT alignment: {e}")
        
        # Build response
        response = EstimateLevelResponse(
            cefr_level=cefr_level,
            confidence=analysis_dict.get("confidence", 0.5),
            indicators=analysis_dict.get("indicators", []),
            reasoning=analysis_dict.get("reasoning", "Análise automática"),
            breakdown=breakdown,
            # Phase 1: Error-rate features
            error_rate_features=analysis_dict.get("error_rate_features"),
            pronunciation_score=analysis_dict.get("pronunciation_score"),
            error_positions=analysis_dict.get("error_positions"),
            # Phase 2: Acoustic features
            acoustic_features=analysis_dict.get("acoustic_features"),
            acoustic_metadata=analysis_dict.get("acoustic_metadata"),
            # Phase 3: ASR metadata
            asr_metadata_analysis=asr_metadata_analysis,
            # Phase 3: Task relevance
            task_relevance=task_relevance_data,
            topic_coverage=topic_coverage_data,
            # Phase 3: Structured feedback
            feedback=feedback_data,
            # AKT alignment
            akt_alignment=akt_alignment,
            # Additional features
            features=analysis_dict.get("features"),
            grammar_error_profile=analysis_dict.get("grammar_error_profile")
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error estimating level: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnostic/analyze_session", response_model=AnalyzeSessionResponse)
async def analyze_session(request: AnalyzeSessionRequest):
    """
    Análise agregada de uma sessão completa
    
    Agrega dados de múltiplos turnos para identificar:
    - Tendências de erro (melhorando/piorando)
    - Skills problemáticas na sessão
    - Padrões linguísticos recorrentes
    - Progresso geral da sessão
    - Session dynamics (consistency, trajectory, engagement)
    
    Args:
        request: Request com lista de turnos e suas análises
        
    Returns:
        Análise agregada da sessão com padrões e recomendações
    """
    try:
        result = session_analyzer.analyze_session(
            session_turns=request.session_turns,
            conversation_history=request.conversation_history
        )
        
        # Phase 3: Add session dynamics if user_id provided
        session_dynamics = None
        if request.user_id:
            try:
                session_dynamics = await session_analyzer.analyze_session_dynamics(
                    request.session_turns,
                    request.user_id
                )
            except Exception as e:
                logger.warning(f"Failed to analyze session dynamics: {e}")
        
        result["session_dynamics"] = session_dynamics
        
        return AnalyzeSessionResponse(**result)
    except Exception as e:
        logger.error(f"Error analyzing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnostic/extract_skills", response_model=ExtractSkillsResponse)
async def extract_skills(request: ExtractSkillsRequest):
    """
    Extração focada de skills do texto do aluno
    
    Este endpoint faz uma chamada dedicada ao LLM apenas para identificar
    skills usadas no texto, sem análise de erros. Retorna skill_ids com
    confidence scores e features linguísticas.
    """
    try:
        # Validar que valid_skills não está vazio (já validado pelo Pydantic, mas garantir)
        if not request.valid_skills or len(request.valid_skills) == 0:
            raise HTTPException(
                status_code=400,
                detail="valid_skills não pode estar vazio. Forneça pelo menos uma skill_id válida."
            )
        
        # Validar que todas as skills existem no SKILL_CEFR_MAP
        try:
            from src.services.student_model.skill_registry import SKILL_CEFR_MAP
            
            all_valid_skills = []
            for skills in SKILL_CEFR_MAP.values():
                all_valid_skills.extend(skills)
            
            invalid_skills = [s for s in request.valid_skills if s not in all_valid_skills]
            if invalid_skills:
                logger.warning(f"Invalid skills provided: {invalid_skills[:5]} (showing first 5)")
                # Filtrar skills inválidas
                request.valid_skills = [s for s in request.valid_skills if s in all_valid_skills]
                
                if not request.valid_skills:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"All provided skills are invalid. Invalid examples: {invalid_skills[:3]}"
                    )
        except ImportError:
            # Se não conseguir importar, continuar sem validação
            logger.debug("Could not import SKILL_CEFR_MAP for validation, skipping validation")
        
        # Chamar LLM para extração de skills
        result = await llm_client.extract_skills(
            user_text=request.user_text,
            valid_skills=request.valid_skills,
            ai_text=request.ai_text
        )
        
        # Converter skills para lista de SkillExtraction
        skills = [
            {
                "skill_id": skill.get("skill_id"),
                "confidence": skill.get("confidence", 0.0),
                "linguistic_features": skill.get("linguistic_features", {})
            }
            for skill in result.get("skills", [])
        ]
        
        return ExtractSkillsResponse(
            skills=skills,
            overall_linguistic_features=result.get("overall_linguistic_features", {}),
            summary=result.get("summary", "Extração de skills realizada")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        # Retornar resposta vazia em caso de erro (fallback)
        return ExtractSkillsResponse(
            skills=[],
            overall_linguistic_features={},
            summary=f"Erro na extração: {str(e)}"
        )

@app.post("/api/diagnostic/calibrate", response_model=CalibrateResponse)
async def calibrate(request: CalibrateRequest):
    """
    Calibrate model predictions with human annotations.
    Based on Byun et al. (2025) - LLM-as-a-Grader.
    
    Args:
        request: Validation dataset with human annotations
        
    Returns:
        Calibration results with learned weights
    """
    try:
        result = calibration_manager.calibrate(request.validation_dataset)
        
        return CalibrateResponse(
            status=result.get("status", "success"),
            num_samples=result.get("num_samples", 0),
            avg_cefr_error=result.get("avg_cefr_error"),
            avg_score_errors=result.get("avg_score_errors"),
            weights=result.get("weights")
        )
    except Exception as e:
        logger.error(f"Error calibrating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    logger.info(f"🚀 Starting Diagnostic Module Service on port {config['service']['port']}")
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=True
    )

