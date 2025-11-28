"""
Pydantic models for Diagnostic Module Service
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ErrorType(str, Enum):
    """Tipos de erros linguísticos"""
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    PRONUNCIATION = "pronunciation"
    SYNTAX = "syntax"
    SPELLING = "spelling"


class ErrorCategory(str, Enum):
    """Categorias específicas de erros"""
    VERB_CONJUGATION = "verb_conjugation"
    VERB_TENSE = "verb_tense"
    ARTICLE = "article"
    PREPOSITION = "preposition"
    WORD_ORDER = "word_order"
    VOCABULARY_CHOICE = "vocabulary_choice"
    PRONUNCIATION_SOUND = "pronunciation_sound"
    ACCENT = "accent"


class CEFRLevel(str, Enum):
    """Níveis CEFR"""
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class AnalyzeTurnRequest(BaseModel):
    """Request para análise de um turno"""
    user_text: str = Field(..., description="Texto do usuário")
    ai_text: Optional[str] = Field(None, description="Resposta do AI (para contexto)")
    language: str = Field("pt-BR", description="Idioma")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional")
    valid_skills: Optional[List[str]] = Field(None, description="Lista de skill_ids válidas para SINKT semantic tagging")


class ErrorAnalysis(BaseModel):
    """Análise de um erro específico"""
    error_type: ErrorType
    category: Optional[ErrorCategory] = None
    skill_id: Optional[str] = Field(None, description="ID da habilidade relacionada")
    original_text: str = Field(..., description="Texto com erro")
    corrected_text: Optional[str] = Field(None, description="Texto corrigido")
    explanation: Optional[str] = Field(None, description="Explicação do erro")
    severity: str = Field("medium", description="Severidade: low, medium, high")
    position: Optional[Dict[str, int]] = Field(None, description="Posição do erro no texto")
    linguistic_features: Optional[Dict[str, Any]] = Field(None, description="Features linguísticas extraídas (tense, person, number, register, domain, etc.)")


class ComplexityAnalysis(BaseModel):
    """Análise de complexidade linguística"""
    estimated_cefr_level: CEFRLevel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança na estimativa")
    vocabulary_complexity: str = Field("basic", description="basic, intermediate, advanced")
    grammar_complexity: str = Field("basic", description="basic, intermediate, advanced")
    sentence_length_avg: float = Field(..., description="Tamanho médio das frases")
    indicators: List[str] = Field(default_factory=list, description="Indicadores de nível")
    reasoning: str = Field("Análise automática", description="Justificativa textual da análise")


class ProgressAnalysis(BaseModel):
    """Análise de progresso temporal"""
    improved: bool = Field(..., description="Se melhorou desde última análise")
    regression: bool = Field(False, description="Se regrediu")
    improvement_areas: List[str] = Field(default_factory=list)
    regression_areas: List[str] = Field(default_factory=list)
    overall_trend: str = Field("stable", description="improving, stable, declining")


class AnalyzeTurnResponse(BaseModel):
    """Resposta completa da análise de turno"""
    errors: List[ErrorAnalysis] = Field(default_factory=list)
    correct_skills: List[str] = Field(default_factory=list, description="Lista de skills usadas corretamente")
    linguistic_features: Dict[str, Any] = Field(default_factory=dict, description="Features linguísticas extraídas do texto")
    semantic_skill_mapping: Dict[str, float] = Field(default_factory=dict, description="Mapeamento semântico SINKT (skill_id -> confidence score)")
    complexity: Optional[ComplexityAnalysis] = None
    progress: Optional[ProgressAnalysis] = None
    target_skill_correct: bool = Field(False, description="Se a habilidade em foco foi usada corretamente")
    summary: str = Field(..., description="Resumo da análise")


class EstimateLevelRequest(BaseModel):
    """Request para estimar nível CEFR"""
    text: str = Field(..., description="Texto para análise")
    language: str = Field("pt-BR", description="Idioma")
    user_id: Optional[str] = Field(None, description="ID do usuário (para integração com AKT)")
    question: Optional[str] = Field(None, description="Pergunta/tarefa (para análise de relevância)")
    exemplar: Optional[str] = Field(None, description="Exemplo de resposta ideal (para similaridade)")
    audio_path: Optional[str] = Field(None, description="Caminho para arquivo de áudio")
    asr_transcription: Optional[str] = Field(None, description="Transcrição ASR (para error-rate features)")
    expected_text: Optional[str] = Field(None, description="Texto esperado/correto (para error-rate features)")
    asr_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados ASR (timestamps, confidence)")


class EstimateLevelResponse(BaseModel):
    """Resposta com nível CEFR estimado"""
    cefr_level: CEFRLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Raciocínio por trás da estimativa")
    
    # Breakdown multi-aspecto
    breakdown: Optional[Dict[str, float]] = Field(None, description="Breakdown por aspecto (fluency, grammar, vocabulary, coherence)")
    
    # Phase 1: Error-Rate Features (Do et al., Interspeech 2024)
    error_rate_features: Optional[Dict[str, Any]] = Field(None, description="Error-rate features (ASR vs expected)")
    pronunciation_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pronunciation score from error-rate")
    error_positions: Optional[Dict[str, Any]] = Field(None, description="Specific error positions identified")
    
    # Phase 2: Acoustic Features (Lee et al., Interspeech 2024)
    acoustic_features: Optional[List[float]] = Field(None, description="Acoustic features from Wav2Vec multi-embedding")
    acoustic_metadata: Optional[Dict[str, Any]] = Field(None, description="Acoustic metadata (embedding shapes, model info)")
    
    # Phase 3: ASR Metadata (Mohammadi et al., 2025)
    asr_metadata_analysis: Optional[Dict[str, Any]] = Field(None, description="ASR metadata analysis (confidence, speech rate, pauses, fluency)")
    
    # Phase 3: Task Relevance (Lu et al., 2025; Reimers & Gurevych, 2019)
    task_relevance: Optional[Dict[str, float]] = Field(None, description="Task relevance and exemplar similarity (SBERT)")
    topic_coverage: Optional[Dict[str, Any]] = Field(None, description="Topic coverage analysis")
    
    # Phase 3: Structured Feedback (Lu et al., 2025; Xiao et al., 2024)
    feedback: Optional[Dict[str, Any]] = Field(None, description="Structured feedback (strengths, weaknesses, next steps, priority)")
    
    # Phase 3: AKT Integration (existing)
    akt_alignment: Optional[Dict[str, Any]] = Field(None, description="AKT cross-validation results")
    
    # Additional features
    features: Optional[Dict[str, Any]] = Field(None, description="Additional linguistic features")
    grammar_error_profile: Optional[Dict[str, int]] = Field(None, description="Grammar error profile by type")


class AnalyzeSessionRequest(BaseModel):
    """Request para análise agregada de uma sessão"""
    session_turns: List[Dict[str, Any]] = Field(..., description="Lista de turnos com suas análises")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Histórico completo da conversa")
    user_id: Optional[str] = Field(None, description="User ID for session dynamics analysis")


class AnalyzeSessionResponse(BaseModel):
    """Resposta com análise agregada da sessão"""
    session_summary: str = Field(..., description="Resumo da sessão")
    total_turns: int = Field(..., description="Total de turnos analisados")
    total_errors: int = Field(..., description="Total de erros na sessão")
    total_correct_skills: int = Field(..., description="Total de skills usadas corretamente")
    avg_errors_per_turn: float = Field(..., description="Média de erros por turno")
    avg_correct_skills_per_turn: float = Field(..., description="Média de skills corretas por turno")
    error_types_count: Dict[str, int] = Field(default_factory=dict, description="Contagem de tipos de erro")
    error_trends: List[Dict[str, Any]] = Field(default_factory=list, description="Tendência de erros ao longo da sessão")
    improving: bool = Field(False, description="Se o estudante está melhorando na sessão")
    skill_progress: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Progresso por skill na sessão")
    problematic_skills: List[str] = Field(default_factory=list, description="Skills problemáticas na sessão")
    mastered_skills: List[str] = Field(default_factory=list, description="Skills dominadas na sessão")
    linguistic_patterns: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Padrões linguísticos na sessão")
    recommendations: List[str] = Field(default_factory=list, description="Recomendações baseadas na sessão")
    
    # Phase 3: Session Dynamics (DynaEval, 2021)
    session_dynamics: Optional[Dict[str, Any]] = Field(None, description="Session dynamics (consistency, trajectory, engagement, anomalies, progress_rate)")


class CalibrateRequest(BaseModel):
    """Request para calibração com anotações humanas"""
    validation_dataset: List[Dict[str, Any]] = Field(..., description="Dataset de validação com anotações humanas")


class CalibrateResponse(BaseModel):
    """Resposta da calibração"""
    status: str = Field(..., description="Status da calibração")
    num_samples: int = Field(..., description="Número de amostras processadas")
    avg_cefr_error: Optional[float] = Field(None, description="Erro médio CEFR")
    avg_score_errors: Optional[Dict[str, float]] = Field(None, description="Erros médios por aspecto")
    weights: Optional[Dict[str, float]] = Field(None, description="Pesos de calibração aprendidos")


class SkillExtraction(BaseModel):
    """Extração de uma skill específica do texto"""
    skill_id: str = Field(..., description="ID da habilidade identificada")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança na identificação (0.0 a 1.0)")
    linguistic_features: Dict[str, Any] = Field(default_factory=dict, description="Features linguísticas específicas desta skill")


class ExtractSkillsRequest(BaseModel):
    """Request para extração de skills do texto"""
    user_text: str = Field(..., description="Texto do usuário")
    valid_skills: List[str] = Field(..., min_items=1, description="Lista obrigatória de skill_ids válidas")
    ai_text: Optional[str] = Field(None, description="Resposta do AI (opcional, para contexto)")


class ExtractSkillsResponse(BaseModel):
    """Resposta com skills extraídas do texto"""
    skills: List[SkillExtraction] = Field(default_factory=list, description="Lista de skills identificadas com confidence e features")
    overall_linguistic_features: Dict[str, Any] = Field(default_factory=dict, description="Features linguísticas gerais do texto")
    summary: str = Field(..., description="Resumo da extração de skills")
