"""
Student Model Service - Standalone FastAPI application
Rastreamento de conhecimento e progresso do estudante usando AKT (padrão) / BKT (fallback)
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Any, List
from loguru import logger
from sqlalchemy.orm import Session

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
from .database import (
    Base, User, Skill, SkillMastery, InteractionHistory,
    create_engine_and_session, init_database
)
from .models import (
    AssessRequest, AssessResponse,
    StudentProfileResponse, SkillsListResponse,
    SkillMasteryResponse,
    FocusAreaResponse, FocusAreasResponse,
    CreateUserRequest, CreateSkillRequest,
    CEFRLevel, CEFRProgressDetailedResponse,
    LinguisticErrorPatternResponse
)
from .knowledge_tracer import AttentiveKnowledgeTracer, BayesianKnowledgeTracer
from .skill_registry import get_bkt_params, get_akt_params, get_default_skills

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "student_model",
        "port": 8900,
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
    port = int(os.getenv("STUDENT_MODEL_PORT", os.getenv("PORT", "8900")))
    config["service"]["port"] = port
    return config

# ============================================================================
# Database Setup
# ============================================================================

engine, SessionLocal = create_engine_and_session()

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database on startup
def init_db():
    """Initialize database tables and seed default skills"""
    logger.info("📊 Initializing Student Model database...")
    init_database()
    
    # Seed default skills
    db = SessionLocal()
    try:
        default_skills = get_default_skills()
        for skill_data in default_skills:
            existing = db.query(Skill).filter(Skill.skill_id == skill_data["skill_id"]).first()
            if not existing:
                skill = Skill(**skill_data)
                db.add(skill)
        db.commit()
        logger.info(f"✅ Seeded {len(default_skills)} default skills")
    except Exception as e:
        logger.error(f"❌ Error seeding skills: {e}")
        db.rollback()
    finally:
        db.close()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Student Model Service",
    version="1.0.0",
    description="Service for tracking student knowledge and progress using AKT (default) / BKT (fallback)"
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
# Service Logic
# ============================================================================

class StudentModelService:
    """Service logic for student model"""
    
    @staticmethod
    def get_or_create_user(db: Session, user_id: str, cefr_level: str = "A1", native_language: str = "en") -> User:
        """Get or create a user"""
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(
                user_id=user_id,
                cefr_level=cefr_level,
                native_language=native_language
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    
    @staticmethod
    def get_or_create_skill_mastery(db: Session, user_id: str, skill_id: str) -> SkillMastery:
        """Get or create skill mastery record"""
        mastery = db.query(SkillMastery).filter(
            SkillMastery.user_id == user_id,
            SkillMastery.skill_id == skill_id
        ).first()
        
        if not mastery:
            # Get skill to ensure it exists
            skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
            if not skill:
                raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
            
            # Get AKT parameters for this skill (AKT is default, BKT is fallback)
            akt_params = get_akt_params(skill_id)
            tracer = AttentiveKnowledgeTracer(akt_params)
            
            mastery = SkillMastery(
                user_id=user_id,
                skill_id=skill_id,
                mastery_probability=tracer.get_mastery_probability(),
                attempts=0,
                successes=0
            )
            db.add(mastery)
            db.commit()
            db.refresh(mastery)
        
        return mastery
    
    @staticmethod
    def assess_response(
        db: Session,
        user_id: str,
        skill_id: str,
        correct: bool,
        context: Optional[Dict[str, Any]] = None,
        user_text: Optional[str] = None,
        ai_text: Optional[str] = None,
        difficulty: Optional[float] = None,
        linguistic_features: Optional[Dict[str, Any]] = None
    ) -> AssessResponse:
        """Assess a student response and update knowledge"""
        # Get or create user
        user = StudentModelService.get_or_create_user(db, user_id)
        
        # Get or create skill mastery
        mastery = StudentModelService.get_or_create_skill_mastery(db, user_id, skill_id)
        
        # Get AKT parameters (AKT is default, BKT is fallback)
        try:
            akt_params = get_akt_params(skill_id)
            tracer = AttentiveKnowledgeTracer(akt_params)
            tracer.mastery_probability = mastery.mastery_probability  # Restore current state
            
            # Restore interaction history if available (for AKT)
            # Load recent interactions from database
            from datetime import datetime, timedelta
            from datetime import timezone
            recent_interactions = db.query(InteractionHistory).filter(
                InteractionHistory.user_id == user_id,
                InteractionHistory.skill_id == skill_id,
                InteractionHistory.timestamp >= datetime.now(timezone.utc) - timedelta(days=30)
            ).order_by(InteractionHistory.timestamp.desc()).limit(50).all()
            
            # Get skill difficulty for IRT
            from .skill_registry import get_skill_difficulty
            skill_difficulty = get_skill_difficulty(skill_id)
            
            # Restore to tracer's history
            for interaction in reversed(recent_interactions):  # Reverse to get chronological order
                # Ensure timestamp is timezone-aware
                timestamp = interaction.timestamp
                if timestamp and timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                tracer.interaction_history.append({
                    'correct': interaction.correct,
                    'timestamp': timestamp,
                    'features': {
                        'difficulty': skill_difficulty,  # Use IRT difficulty
                        'complexity': 0.5,
                        'time_since_last': 1.0,
                        'error_type_severity': 0.5
                    },
                    'mastery_after': mastery.mastery_probability
                })
        except Exception as e:
            # Fallback to BKT if AKT fails
            logger.warning(f"AKT initialization failed for {skill_id}, falling back to BKT: {e}")
            bkt_params = get_bkt_params(skill_id)
            tracer = BayesianKnowledgeTracer(bkt_params)
            tracer.p_L = mastery.mastery_probability  # Restore current state
        
        # Update belief
        # Ensure context includes skill_id for IRT difficulty calculation
        if context is None:
            context = {}
        if 'skill_id' not in context:
            context['skill_id'] = skill_id
        
        # Add difficulty to context if provided
        if difficulty is not None:
            context['difficulty'] = difficulty
        elif 'difficulty' not in context:
            # Get difficulty from skill_registry
            from .skill_registry import get_skill_difficulty
            context['difficulty'] = get_skill_difficulty(skill_id)
        
        # Add linguistic_features to context if provided
        # Se não fornecidas, tentar extrair do skill_id e context usando extract_linguistic_features
        if linguistic_features:
            context['linguistic_features'] = linguistic_features
        else:
            # Tentar extrair features linguísticas como complemento/fallback
            try:
                from .skill_registry import extract_linguistic_features
                extracted_features = extract_linguistic_features(skill_id, {
                    "user_text": user_text,
                    "ai_text": ai_text,
                    **context
                })
                if extracted_features:
                    context['linguistic_features'] = extracted_features
            except Exception as e:
                logger.debug(f"Could not extract linguistic features: {e}")
        
        previous_prob = mastery.mastery_probability
        new_prob = tracer.update_belief(correct, context)
        
        # Update mastery record
        mastery.mastery_probability = new_prob
        mastery.attempts += 1
        if correct:
            mastery.successes += 1
        from datetime import datetime, timezone
        mastery.last_practiced = datetime.now(timezone.utc)
        mastery.updated_at = datetime.now(timezone.utc)
        
        # Save interaction history
        import json
        # Enriquecer semantic_features com mais metadata para análise futura
        semantic_features_data = {}
        if linguistic_features:
            semantic_features_data['linguistic_features'] = linguistic_features
        if context:
            semantic_features_data['difficulty'] = context.get('difficulty')
            semantic_features_data['complexity'] = context.get('complexity')
            semantic_features_data['error_type'] = context.get('error_type')
            semantic_features_data['error_severity'] = context.get('error_severity')
        
        semantic_features_json = json.dumps(semantic_features_data) if semantic_features_data else None
        
        interaction = InteractionHistory(
            user_id=user_id,
            skill_id=skill_id,
            correct=correct,
            context=str(context) if context else None,
            user_text=user_text,
            ai_text=ai_text,
            semantic_features=semantic_features_json
        )
        db.add(interaction)
        
        db.commit()
        db.refresh(mastery)
        
        # Determine mastery status
        if new_prob < 0.3:
            status_str = "beginner"
        elif new_prob < 0.7:
            status_str = "learning"
        else:
            status_str = "mastered"
        
        return AssessResponse(
            skill_id=skill_id,
            new_mastery_probability=new_prob,
            previous_mastery_probability=previous_prob,
            improvement=new_prob - previous_prob,
            mastery_status=status_str
        )
    
    @staticmethod
    def get_student_profile(db: Session, user_id: str) -> StudentProfileResponse:
        """Get complete student profile"""
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Get all skill masteries
        masteries = db.query(SkillMastery).filter(SkillMastery.user_id == user_id).all()
        
        total_skills = len(masteries)
        mastered = sum(1 for m in masteries if m.mastery_probability > 0.7)
        learning = sum(1 for m in masteries if 0.3 <= m.mastery_probability <= 0.7)
        beginner = sum(1 for m in masteries if m.mastery_probability < 0.3)
        
        return StudentProfileResponse(
            user_id=user.user_id,
            cefr_level=CEFRLevel(user.cefr_level),
            native_language=user.native_language,
            total_skills=total_skills,
            mastered_skills=mastered,
            learning_skills=learning,
            beginner_skills=beginner,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    @staticmethod
    def get_student_skills(db: Session, user_id: str) -> SkillsListResponse:
        """Get all skills for a student"""
        masteries = db.query(SkillMastery).filter(SkillMastery.user_id == user_id).all()
        
        skills_list = []
        from .skill_registry import get_skill_difficulty
        for mastery in masteries:
            skill = db.query(Skill).filter(Skill.skill_id == mastery.skill_id).first()
            if skill:
                skill_difficulty = get_skill_difficulty(mastery.skill_id)
                skills_list.append(SkillMasteryResponse(
                    skill_id=mastery.skill_id,
                    skill_name=skill.name,
                    mastery_probability=mastery.mastery_probability,
                    difficulty=skill_difficulty,
                    attempts=mastery.attempts,
                    successes=mastery.successes,
                    last_practiced=mastery.last_practiced
                ))
        
        return SkillsListResponse(skills=skills_list, total=len(skills_list))
    
    @staticmethod
    def get_focus_areas(db: Session, user_id: str, limit: int = 3) -> FocusAreasResponse:
        """Get top focus areas for a student"""
        masteries = db.query(SkillMastery).filter(SkillMastery.user_id == user_id).all()
        
        # Sort by priority: low mastery + recent practice
        focus_areas = []
        for mastery in masteries:
            skill = db.query(Skill).filter(Skill.skill_id == mastery.skill_id).first()
            if not skill:
                continue
            
            # Calculate priority score (lower mastery = higher priority)
            priority_score = 1.0 - mastery.mastery_probability
            
            # Determine priority level
            if mastery.mastery_probability < 0.3:
                priority = "high"
            elif mastery.mastery_probability < 0.5:
                priority = "medium"
            else:
                priority = "low"
            
            reason = f"Mastery at {mastery.mastery_probability:.0%}"
            if mastery.mastery_probability < 0.3:
                reason += " - Needs practice"
            elif mastery.mastery_probability < 0.7:
                reason += " - Continue practicing"
            else:
                reason += " - Ready for review"
            
            focus_areas.append(FocusAreaResponse(
                skill_id=mastery.skill_id,
                skill_name=skill.name,
                category=skill.category,
                mastery_probability=mastery.mastery_probability,
                priority=priority,
                reason=reason
            ))
        
        # Sort by priority score (descending)
        focus_areas.sort(key=lambda x: (1.0 - x.mastery_probability), reverse=True)
        
        return FocusAreasResponse(
            focus_areas=focus_areas[:limit],
            total_recommended=len(focus_areas)
        )

    @staticmethod
    def calculate_cefr_progress(db: Session, user_id: str) -> Dict[str, Any]:
        """
        Calcula o progresso agregado do aluno em cada nível CEFR
        baseado nas probabilidades do AKT.
        """
        from .skill_registry import SKILL_CEFR_MAP
        
        # 1. Buscar todas as masteries do aluno no banco (calculadas pelo AKT)
        user_masteries = db.query(SkillMastery).filter(
            SkillMastery.user_id == user_id
        ).all()
        
        # Criar mapa rápido: skill_id -> probability
        mastery_map = {m.skill_id: m.mastery_probability for m in user_masteries}
        
        cefr_status = {}
        overall_level = "A1"
        
        # 2. Iterar por cada nível CEFR
        levels_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        
        for level in levels_order:
            skills = SKILL_CEFR_MAP.get(level, [])
            if not skills:
                cefr_status[level] = {
                    "progress": 0.0,
                    "percentage": "0%",
                    "status": "locked",
                    "skills_breakdown": []
                }
                continue
                
            total_prob = 0.0
            count = len(skills)
            
            # Detalhes das skills deste nível
            level_skills_details = []
            
            for skill_id in skills:
                # Se o aluno tem mastery registrada pelo AKT, usa ela.
                # Se não tem, assume 0.0
                prob = mastery_map.get(skill_id, 0.0) 
                total_prob += prob
                
                level_skills_details.append({
                    "skill_id": skill_id,
                    "mastery": prob
                })
            
            # Média do nível (0.0 a 1.0)
            level_progress = total_prob / count if count > 0 else 0.0
            
            # Determine status
            if level_progress > 0.8:
                status = "mastered"
            elif level_progress > 0.2:
                status = "in_progress"
            else:
                status = "locked"
                
            # Special case: A1 is always at least in_progress if user exists
            if level == "A1" and status == "locked":
                status = "in_progress"
            
            cefr_status[level] = {
                "progress": round(level_progress, 2),
                "percentage": f"{int(level_progress * 100)}%",
                "status": status,
                "skills_breakdown": level_skills_details
            }
            
            # Lógica simples para definir o nível atual do aluno
            # Se ele completou > 80% deste nível, ele provavelmente está no próximo
            if level_progress > 0.8:
                current_idx = levels_order.index(level)
                if current_idx + 1 < len(levels_order):
                    overall_level = levels_order[current_idx + 1]

        # Calcular dimension_progress (grammar, vocabulary, pronunciation)
        # Use user_masteries que já foi definido acima
        dimension_avg = {}
        for mastery in user_masteries:
            skill = db.query(Skill).filter(Skill.skill_id == mastery.skill_id).first()
            if skill:
                category = skill.category.lower()
                if category not in dimension_avg:
                    dimension_avg[category] = []
                dimension_avg[category].append(mastery.mastery_probability)
        
        # Calcular média por dimensão
        dimension_progress = {}
        for dim, probs in dimension_avg.items():
            dimension_progress[dim] = sum(probs) / len(probs) if probs else 0.0
        
        return {
            "current_estimated_level": overall_level,
            "cefr_details": cefr_status,
            "dimension_progress": dimension_progress
        }
    
    @staticmethod
    def _analyze_linguistic_error_patterns(db: Session, user_id: str) -> Dict[str, Any]:
        """
        Analisa padrões de erro por features linguísticas agregando dados de todas as skills.
        
        Identifica quais features linguísticas (tense, person, number, register, domain)
        estão causando mais erros para o estudante.
        
        Args:
            db: Database session
            user_id: ID do estudante
            
        Returns:
            Dicionário com análise de padrões de erro por feature
        """
        import json
        from datetime import datetime, timedelta, timezone
        
        # Buscar interações dos últimos 90 dias
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        interactions = db.query(InteractionHistory).filter(
            InteractionHistory.user_id == user_id,
            InteractionHistory.timestamp >= cutoff_date
        ).all()
        
        # Agregar por feature linguística
        feature_stats: Dict[str, Dict[str, int]] = {}  # feature_key -> {correct_count, error_count}
        
        for interaction in interactions:
            # Parse semantic_features se disponível
            if not interaction.semantic_features:
                continue
            
            try:
                semantic_data = json.loads(interaction.semantic_features)
                linguistic_features = semantic_data.get("linguistic_features", {})
                
                if not linguistic_features:
                    continue
                
                # Para cada feature linguística presente
                linguistic_feature_keys = ['tense', 'person', 'number', 'register', 'domain']
                for key in linguistic_feature_keys:
                    if key in linguistic_features:
                        feature_value = linguistic_features[key]
                        pattern_key = f"{key}:{feature_value}"
                        
                        # Inicializar se não existir
                        if pattern_key not in feature_stats:
                            feature_stats[pattern_key] = {
                                'correct_count': 0,
                                'error_count': 0,
                                'total_attempts': 0
                            }
                        
                        # Atualizar contadores
                        feature_stats[pattern_key]['total_attempts'] += 1
                        if interaction.correct:
                            feature_stats[pattern_key]['correct_count'] += 1
                        else:
                            feature_stats[pattern_key]['error_count'] += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Failed to parse semantic_features for interaction {interaction.id}: {e}")
                continue
        
        # Calcular estatísticas e identificar padrões problemáticos
        feature_analysis = []
        for feature_key, stats in feature_stats.items():
            total = stats['total_attempts']
            if total < 3:  # Mínimo de 3 tentativas para considerar
                continue
            
            error_count = stats['error_count']
            correct_count = stats['correct_count']
            error_rate = error_count / total if total > 0 else 0.0
            success_rate = correct_count / total if total > 0 else 0.0
            
            feature_analysis.append({
                'feature_key': feature_key,
                'feature_type': feature_key.split(':')[0],  # tense, person, etc.
                'feature_value': feature_key.split(':')[1] if ':' in feature_key else feature_key,
                'total_attempts': total,
                'error_count': error_count,
                'correct_count': correct_count,
                'error_rate': round(error_rate, 3),
                'success_rate': round(success_rate, 3)
            })
        
        # Ordenar por taxa de erro (maior primeiro)
        feature_analysis.sort(key=lambda x: x['error_rate'], reverse=True)
        
        # Identificar features problemáticas (error_rate > 0.5 e >= 5 tentativas)
        problematic_features = [
            f for f in feature_analysis 
            if f['error_rate'] > 0.5 and f['total_attempts'] >= 5
        ]
        
        # Identificar features dominadas (error_rate < 0.2 e >= 5 tentativas)
        mastered_features = [
            f for f in feature_analysis 
            if f['error_rate'] < 0.2 and f['total_attempts'] >= 5
        ]
        
        # Gerar resumo human-readable
        summary_parts = []
        if problematic_features:
            top_problem = problematic_features[0]
            feature_name = top_problem['feature_key'].replace(':', ' ')
            summary_parts.append(
                f"Maior dificuldade: {feature_name} ({top_problem['error_rate']:.0%} de erros em {top_problem['total_attempts']} tentativas)"
            )
        if mastered_features:
            top_mastered = mastered_features[0]
            feature_name = top_mastered['feature_key'].replace(':', ' ')
            summary_parts.append(
                f"Melhor domínio: {feature_name} ({top_mastered['success_rate']:.0%} de acertos em {top_mastered['total_attempts']} tentativas)"
            )
        
        summary = ". ".join(summary_parts) if summary_parts else "Dados insuficientes para análise de padrões (mínimo 3 tentativas por feature)"
        
        return {
            'total_interactions_analyzed': len(interactions),
            'features_analyzed': len(feature_analysis),
            'problematic_features': problematic_features[:5],  # Top 5
            'mastered_features': mastered_features[:5],  # Top 5
            'all_features': feature_analysis[:10],  # Top 10 para análise detalhada
            'summary': summary
        }
    
    @staticmethod
    def get_interpretable_knowledge_state(db: Session, user_id: str) -> Dict[str, Any]:
        """
        Retorna estado de conhecimento interpretável com breakdown detalhado
        
        Inclui:
        - Progresso por dimensão (grammar, vocabulary, pronunciation)
        - Top-3 skills fortes e fracas por nível CEFR
        - Recomendações human-readable
        
        Args:
            db: Database session
            user_id: ID do estudante
            
        Returns:
            Dicionário com estado interpretável
        """
        from .skill_registry import SKILL_CEFR_MAP, get_skill_difficulty
        
        # Obter todas as masteries
        masteries = db.query(SkillMastery).filter(SkillMastery.user_id == user_id).all()
        mastery_map = {m.skill_id: m.mastery_probability for m in masteries}
        
        # Agregar por dimensão (category)
        dimension_progress = {
            "grammar": [],
            "vocabulary": [],
            "pronunciation": []
        }
        
        # Agregar skills por categoria
        for mastery in masteries:
            skill = db.query(Skill).filter(Skill.skill_id == mastery.skill_id).first()
            if skill:
                category = skill.category.lower()
                if category in dimension_progress:
                    dimension_progress[category].append(mastery.mastery_probability)
        
        # Calcular média por dimensão
        dimension_avg = {}
        for dim, probs in dimension_progress.items():
            dimension_avg[dim] = sum(probs) / len(probs) if probs else 0.0
        
        # Identificar top skills fortes e fracas por nível CEFR
        strong_skills = []
        weak_skills = []
        
        for level, skill_ids in SKILL_CEFR_MAP.items():
            level_strong = []
            level_weak = []
            
            for skill_id in skill_ids:
                mastery_prob = mastery_map.get(skill_id, 0.0)
                skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
                
                if skill:
                    skill_info = {
                        "skill_id": skill_id,
                        "skill_name": skill.name,
                        "mastery": mastery_prob,
                        "level": level,
                        "category": skill.category
                    }
                    
                    if mastery_prob >= 0.7:
                        level_strong.append(skill_info)
                    elif mastery_prob < 0.3:
                        level_weak.append(skill_info)
            
            # Ordenar e pegar top-3
            level_strong.sort(key=lambda x: x["mastery"], reverse=True)
            level_weak.sort(key=lambda x: x["mastery"])
            
            strong_skills.extend(level_strong[:3])
            weak_skills.extend(level_weak[:3])
        
        # Ordenar globalmente e pegar top-3
        strong_skills.sort(key=lambda x: x["mastery"], reverse=True)
        weak_skills.sort(key=lambda x: x["mastery"])
        strong_skills = strong_skills[:3]
        weak_skills = weak_skills[:3]
        
        # Gerar recomendações human-readable
        recommendations = []
        
        # Recomendação baseada em dimensão mais fraca
        weakest_dim = min(dimension_avg.items(), key=lambda x: x[1])
        if weakest_dim[1] < 0.5:
            recommendations.append(
                f"Você precisa praticar mais {weakest_dim[0]}. "
                f"Seu progresso atual é {weakest_dim[1]:.0%}."
            )
        
        # Recomendação baseada em skills fracas
        if weak_skills:
            weak_names = [s["skill_name"] for s in weak_skills[:2]]
            recommendations.append(
                f"Foque em praticar: {', '.join(weak_names)}. "
                "Essas habilidades precisam de mais atenção."
            )
        
        # Recomendação baseada em skills fortes
        if strong_skills:
            strong_names = [s["skill_name"] for s in strong_skills[:2]]
            recommendations.append(
                f"Excelente progresso em: {', '.join(strong_names)}! "
                "Continue praticando para manter o domínio."
            )
        
        # Recomendação geral baseada no nível CEFR
        cefr_progress = StudentModelService.calculate_cefr_progress(db, user_id)
        current_level = cefr_progress.get("current_estimated_level", "A1")
        recommendations.append(
            f"Você está no nível {current_level}. "
            "Continue praticando para avançar para o próximo nível."
        )
        
        # NOVO: Análise de padrões de erro por linguistic features
        # Agregar erros por feature linguística (tense, person, number, register, domain)
        linguistic_error_patterns = StudentModelService._analyze_linguistic_error_patterns(db, user_id)
        
        # Adicionar recomendações baseadas em padrões de erro
        if linguistic_error_patterns.get("problematic_features"):
            problematic = linguistic_error_patterns["problematic_features"][:2]  # Top 2
            for feature in problematic:
                feature_name = feature.get("feature_key", "").replace(":", " ")
                error_rate = feature.get("error_rate", 0.0)
                recommendations.append(
                    f"Você está tendo dificuldades com {feature_name} "
                    f"({error_rate:.0%} de erros). Pratique mais esta área."
                )
        
        return {
            "current_estimated_level": current_level,
            "dimension_progress": dimension_avg,
            "linguistic_error_patterns": linguistic_error_patterns,  # NEW
            "strong_skills": strong_skills,
            "weak_skills": weak_skills,
            "recommendations": recommendations,
            "cefr_details": cefr_progress.get("cefr_details", {})
        }


# ============================================================================
# API Routes
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    init_db()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "student_model"}

@app.post("/api/student/{user_id}/assess", response_model=AssessResponse)
async def assess_student_response(
    user_id: str,
    request: AssessRequest,
    db: Session = Depends(get_db)
):
    """Avaliar resposta do estudante e atualizar conhecimento"""
    try:
        result = StudentModelService.assess_response(
            db=db,
            user_id=user_id,
            skill_id=request.skill_id,
            correct=request.correct,
            context=request.context,
            user_text=request.user_text,
            ai_text=request.ai_text,
            difficulty=request.difficulty,
            linguistic_features=request.linguistic_features
        )
        return result
    except Exception as e:
        logger.error(f"Error assessing response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/profile", response_model=StudentProfileResponse)
async def get_student_profile(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Obter perfil completo do estudante"""
    try:
        return StudentModelService.get_student_profile(db, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/skills", response_model=SkillsListResponse)
async def get_student_skills(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Listar todas as habilidades do estudante"""
    try:
        return StudentModelService.get_student_skills(db, user_id)
    except Exception as e:
        logger.error(f"Error getting skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/focus_areas", response_model=FocusAreasResponse)
async def get_focus_areas(
    user_id: str,
    limit: int = 3,
    db: Session = Depends(get_db)
):
    """Obter top 3 áreas de foco para o estudante"""
    try:
        return StudentModelService.get_focus_areas(db, user_id, limit)
    except Exception as e:
        logger.error(f"Error getting focus areas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/cefr_progress")
async def get_cefr_progress(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Retorna o progresso detalhado do aluno nos níveis CEFR (A1-C2)
    baseado na agregação das probabilidades do AKT.
    """
    try:
        return StudentModelService.calculate_cefr_progress(db, user_id)
    except Exception as e:
        logger.error(f"Error calculating CEFR progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/interpretable_knowledge_state", response_model=CEFRProgressDetailedResponse)
async def get_interpretable_knowledge_state(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Retorna estado de conhecimento interpretável com breakdown detalhado:
    - Progresso por dimensão (grammar, vocabulary, pronunciation)
    - Top skills fortes e fracas
    - Recomendações human-readable
    - Análise de padrões de erro por feature linguística
    """
    try:
        result = StudentModelService.get_interpretable_knowledge_state(db, user_id)
        return CEFRProgressDetailedResponse(**result)
    except Exception as e:
        logger.error(f"Error getting interpretable knowledge state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/student/{user_id}/linguistic_error_patterns", response_model=LinguisticErrorPatternResponse)
async def get_linguistic_error_patterns(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Analisa padrões de erro por feature linguística usando dados históricos
    
    O LLM extrai linguistic_features de cada interação (tense, person, number, register, domain).
    Este endpoint agrega essas features ao longo do tempo para identificar:
    - Quais features linguísticas estão causando mais erros
    - Quais features o estudante já domina
    - Padrões de erro por tipo de feature (ex: sempre erra na 3ª pessoa)
    
    Retorna análise detalhada baseada nas últimas 90 dias de interações.
    """
    try:
        result = StudentModelService._analyze_linguistic_error_patterns(db, user_id)
        return LinguisticErrorPatternResponse(**result)
    except Exception as e:
        logger.error(f"Error analyzing linguistic error patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/student")
async def create_user(
    request: CreateUserRequest,
    db: Session = Depends(get_db)
):
    """Criar novo usuário"""
    try:
        user = StudentModelService.get_or_create_user(
            db,
            request.user_id,
            request.cefr_level.value if isinstance(request.cefr_level, CEFRLevel) else request.cefr_level,
            request.native_language
        )
        return {"user_id": user.user_id, "created": user.created_at == user.updated_at}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/skills")
async def create_skill(
    request: CreateSkillRequest,
    db: Session = Depends(get_db)
):
    """Criar nova habilidade"""
    try:
        existing = db.query(Skill).filter(Skill.skill_id == request.skill_id).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Skill {request.skill_id} already exists")
        
        skill = Skill(
            skill_id=request.skill_id,
            name=request.name,
            category=request.category.value if hasattr(request.category, 'value') else request.category,
            difficulty=request.difficulty,
            description=request.description
        )
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return {"skill_id": skill.skill_id, "created": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    logger.info(f"🚀 Starting Student Model Service on port {config['service']['port']}")
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=True
    )

