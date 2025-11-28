"""
Session Analyzer - Análise agregada de toda a sessão/conversa
Enhanced with session dynamics analysis.
Based on DynaEval (2021) and DKT/AKT.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """Analisador que agrega dados de múltiplos turnos para análise de sessão"""
    
    def __init__(self):
        pass
    
    def analyze_session(
        self,
        session_turns: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analisa toda a sessão agregando dados de múltiplos turnos
        
        Args:
            session_turns: Lista de turnos da sessão atual (cada turno tem analysis do diagnostic_module)
            conversation_history: Histórico completo da conversa (opcional, para contexto)
            
        Returns:
            Análise agregada da sessão com padrões identificados
        """
        if not session_turns:
            return {
                "session_summary": "Nenhum turno analisado ainda",
                "total_turns": 0,
                "error_trends": {},
                "skill_progress": {},
                "linguistic_patterns": {},
                "recommendations": []
            }
        
        # Agregar dados de todos os turnos
        total_errors = 0
        total_correct_skills = 0
        error_types_count = {}
        skills_used = {}  # skill_id -> {correct_count, error_count, total}
        linguistic_features_session = {}  # feature_key -> {count, error_rate}
        error_trends = []  # Lista de erros por turno (para identificar tendências)
        
        for i, turn in enumerate(session_turns):
            turn_analysis = turn.get("analysis", {})
            
            # Contar erros
            errors = turn_analysis.get("errors", [])
            total_errors += len(errors)
            
            # Agregar tipos de erro
            for error in errors:
                error_type = error.get("error_type", "unknown")
                error_types_count[error_type] = error_types_count.get(error_type, 0) + 1
            
            # Agregar skills
            correct_skills = turn_analysis.get("correct_skills", [])
            total_correct_skills += len(correct_skills)
            
            for skill_id in correct_skills:
                if skill_id not in skills_used:
                    skills_used[skill_id] = {"correct_count": 0, "error_count": 0, "total": 0}
                skills_used[skill_id]["correct_count"] += 1
                skills_used[skill_id]["total"] += 1
            
            # Agregar skills com erro
            for error in errors:
                error_skill_id = error.get("skill_id")
                if error_skill_id:
                    if error_skill_id not in skills_used:
                        skills_used[error_skill_id] = {"correct_count": 0, "error_count": 0, "total": 0}
                    skills_used[error_skill_id]["error_count"] += 1
                    skills_used[error_skill_id]["total"] += 1
            
            # Agregar features linguísticas
            linguistic_features = turn_analysis.get("linguistic_features", {})
            for key, value in linguistic_features.items():
                feature_key = f"{key}:{value}"
                if feature_key not in linguistic_features_session:
                    linguistic_features_session[feature_key] = {"count": 0, "errors": 0}
                linguistic_features_session[feature_key]["count"] += 1
                if len(errors) > 0:
                    linguistic_features_session[feature_key]["errors"] += 1
            
            # Trend: número de erros por turno
            error_trends.append({
                "turn": i + 1,
                "error_count": len(errors),
                "correct_skills_count": len(correct_skills)
            })
        
        # Calcular estatísticas agregadas
        total_turns = len(session_turns)
        avg_errors_per_turn = total_errors / total_turns if total_turns > 0 else 0
        avg_correct_skills_per_turn = total_correct_skills / total_turns if total_turns > 0 else 0
        
        # Identificar tendências (melhorando ou piorando?)
        improving = False
        if len(error_trends) >= 3:
            recent_errors = [t["error_count"] for t in error_trends[-3:]]
            earlier_errors = [t["error_count"] for t in error_trends[:3]]
            if earlier_errors:
                avg_recent = sum(recent_errors) / len(recent_errors)
                avg_earlier = sum(earlier_errors) / len(earlier_errors)
                improving = avg_recent < avg_earlier
        
        # Calcular error rate por skill
        skill_progress = {}
        for skill_id, stats in skills_used.items():
            if stats["total"] >= 2:  # Mínimo 2 usos para considerar
                error_rate = stats["error_count"] / stats["total"]
                skill_progress[skill_id] = {
                    "total_uses": stats["total"],
                    "correct_count": stats["correct_count"],
                    "error_count": stats["error_count"],
                    "error_rate": round(error_rate, 3),
                    "success_rate": round(1 - error_rate, 3)
                }
        
        # Calcular error rate por feature linguística
        linguistic_patterns = {}
        for feature_key, stats in linguistic_features_session.items():
            if stats["count"] >= 2:
                error_rate = stats["errors"] / stats["count"]
                linguistic_patterns[feature_key] = {
                    "count": stats["count"],
                    "errors": stats["errors"],
                    "error_rate": round(error_rate, 3)
                }
        
        # Identificar skills problemáticas na sessão
        problematic_skills = [
            skill_id for skill_id, progress in skill_progress.items()
            if progress["error_rate"] > 0.5 and progress["total_uses"] >= 3
        ]
        
        # Identificar skills dominadas na sessão
        mastered_skills = [
            skill_id for skill_id, progress in skill_progress.items()
            if progress["error_rate"] < 0.2 and progress["total_uses"] >= 3
        ]
        
        # Gerar recomendações baseadas na sessão
        recommendations = []
        
        if improving:
            recommendations.append("Você está melhorando! Continue praticando.")
        elif len(error_trends) >= 3 and not improving:
            recommendations.append("Foque nos erros mais comuns para melhorar mais rapidamente.")
        
        if problematic_skills:
            recommendations.append(
                f"Skills que precisam de mais atenção nesta sessão: {', '.join(problematic_skills[:3])}"
            )
        
        if mastered_skills:
            recommendations.append(
                f"Excelente progresso nestas skills: {', '.join(mastered_skills[:3])}"
            )
        
        # Resumo da sessão
        session_summary = f"Sessão com {total_turns} turnos. "
        session_summary += f"Média de {avg_errors_per_turn:.1f} erros por turno. "
        session_summary += f"Média de {avg_correct_skills_per_turn:.1f} skills corretas por turno. "
        if improving:
            session_summary += "Tendência: Melhorando."
        elif len(error_trends) >= 3:
            session_summary += "Tendência: Estável."
        
        return {
            "session_summary": session_summary,
            "total_turns": total_turns,
            "total_errors": total_errors,
            "total_correct_skills": total_correct_skills,
            "avg_errors_per_turn": round(avg_errors_per_turn, 2),
            "avg_correct_skills_per_turn": round(avg_correct_skills_per_turn, 2),
            "error_types_count": error_types_count,
            "error_trends": error_trends,
            "improving": improving,
            "skill_progress": skill_progress,
            "problematic_skills": problematic_skills,
            "mastered_skills": mastered_skills,
            "linguistic_patterns": linguistic_patterns,
            "recommendations": recommendations
        }
    
    async def analyze_session_dynamics(
        self,
        session_turns: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze session-level dynamics.
        Based on DynaEval (2021) and DKT/AKT.
        
        Args:
            session_turns: List of turn analyses
            user_id: User ID for progress tracking
            
        Returns:
            Dictionary with dynamics metrics:
            - consistency: Consistency across turns (0-1)
            - trajectory: Performance trajectory (improving/stable/degrading)
            - engagement: Engagement level (0-1)
            - anomalies: Detected anomalies
            - progress_rate: Rate of progress
        """
        if len(session_turns) < 2:
            return {
                "consistency": 1.0,
                "trajectory": "insufficient_data",
                "engagement": 0.5,
                "anomalies": [],
                "progress_rate": 0.0
            }
        
        # Extract scores from turns
        scores = []
        for turn in session_turns:
            analysis = turn.get("analysis", {})
            # Try to get CEFR numeric score
            cefr_level = analysis.get("estimated_cefr_level", "")
            if cefr_level:
                cefr_numeric = self._cefr_to_numeric(cefr_level)
                scores.append(cefr_numeric)
            else:
                # Fallback: use confidence or breakdown average
                confidence = analysis.get("confidence", 0.5)
                scores.append(confidence * 5)  # Scale to 0-5
        
        # Calculate consistency
        consistency = self._calculate_consistency(scores)
        
        # Analyze trajectory
        trajectory = self._analyze_trajectory(scores)
        
        # Measure engagement
        engagement = self._measure_engagement(session_turns)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(session_turns, scores)
        
        # Calculate progress rate
        progress_rate = self._calculate_progress_rate(session_turns, user_id)
        
        return {
            "consistency": float(consistency),
            "trajectory": trajectory,
            "engagement": float(engagement),
            "anomalies": anomalies,
            "progress_rate": float(progress_rate),
            "num_turns": len(session_turns),
            "score_range": {
                "min": float(np.min(scores)) if scores else 0.0,
                "max": float(np.max(scores)) if scores else 0.0,
                "mean": float(np.mean(scores)) if scores else 0.0,
                "std": float(np.std(scores)) if scores else 0.0
            }
        }
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """
        Measure consistency across turns.
        Lower std = higher consistency.
        
        Args:
            scores: List of scores per turn
            
        Returns:
            Consistency score (0-1)
        """
        if len(scores) < 2:
            return 1.0
        
        std_dev = np.std(scores)
        
        # Normalize: std of 0 = consistency 1.0
        # std of 2.0+ = consistency ~0.0
        consistency = 1.0 / (1.0 + std_dev)
        
        return float(consistency)
    
    def _analyze_trajectory(self, scores: List[float]) -> str:
        """
        Analyze performance trajectory.
        
        Args:
            scores: List of scores per turn
            
        Returns:
            "improving" | "stable" | "degrading" | "fluctuating"
        """
        if len(scores) < 3:
            return "insufficient_data"
        
        # Split into first half and second half
        mid = len(scores) // 2
        first_half = scores[:mid]
        second_half = scores[mid:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        diff = second_mean - first_mean
        threshold = 0.3  # Significant change threshold
        
        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "degrading"
        else:
            # Check for fluctuation
            if np.std(scores) > 0.5:
                return "fluctuating"
            else:
                return "stable"
    
    def _measure_engagement(self, session_turns: List[Dict[str, Any]]) -> float:
        """
        Measure engagement level.
        
        Args:
            session_turns: List of turn analyses
            
        Returns:
            Engagement score (0-1)
        """
        if not session_turns:
            return 0.5
        
        engagement_indicators = []
        
        for turn in session_turns:
            analysis = turn.get("analysis", {})
            
            # Length of response (longer = more engaged)
            text = turn.get("text", "")
            text_length = len(text.split())
            if text_length > 10:
                engagement_indicators.append(0.3)
            elif text_length > 5:
                engagement_indicators.append(0.2)
            else:
                engagement_indicators.append(0.1)
            
            # Number of skills attempted
            correct_skills = len(analysis.get("correct_skills", []))
            if correct_skills > 0:
                engagement_indicators.append(0.2)
            
            # Confidence level
            confidence = analysis.get("confidence", 0.5)
            engagement_indicators.append(confidence * 0.3)
        
        # Average engagement
        engagement = np.mean(engagement_indicators) if engagement_indicators else 0.5
        
        return float(engagement)
    
    def _detect_anomalies(
        self,
        session_turns: List[Dict[str, Any]],
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in session.
        
        Args:
            session_turns: List of turn analyses
            scores: List of scores per turn
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(scores) < 3:
            return anomalies
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Detect outliers (scores > 2 std from mean)
        for i, score in enumerate(scores):
            z_score = abs((score - mean_score) / std_score) if std_score > 0 else 0
            
            if z_score > 2.0:
                turn = session_turns[i]
                anomalies.append({
                    "turn": i + 1,
                    "type": "outlier_score",
                    "severity": "high" if z_score > 3.0 else "medium",
                    "score": float(score),
                    "z_score": float(z_score),
                    "description": f"Score {score:.2f} is {z_score:.2f} standard deviations from mean"
                })
        
        # Detect sudden drops
        for i in range(1, len(scores)):
            drop = scores[i-1] - scores[i]
            if drop > 1.5:  # Sudden drop of 1.5 points
                anomalies.append({
                    "turn": i + 1,
                    "type": "sudden_drop",
                    "severity": "high",
                    "drop": float(drop),
                    "from": float(scores[i-1]),
                    "to": float(scores[i]),
                    "description": f"Sudden drop from {scores[i-1]:.2f} to {scores[i]:.2f}"
                })
        
        return anomalies
    
    def _calculate_progress_rate(
        self,
        session_turns: List[Dict[str, Any]],
        user_id: Optional[str]
    ) -> float:
        """
        Calculate rate of progress.
        
        Args:
            session_turns: List of turn analyses
            user_id: User ID (for future AKT integration)
            
        Returns:
            Progress rate (0-1)
        """
        if len(session_turns) < 2:
            return 0.0
        
        # Extract scores
        scores = []
        for turn in session_turns:
            analysis = turn.get("analysis", {})
            cefr_level = analysis.get("estimated_cefr_level", "")
            if cefr_level:
                scores.append(self._cefr_to_numeric(cefr_level))
        
        if len(scores) < 2:
            return 0.0
        
        # Linear regression to get slope
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        # Normalize slope to 0-1 range
        # Positive slope = progress, negative = regression
        progress_rate = (slope + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        progress_rate = max(0.0, min(1.0, progress_rate))
        
        return float(progress_rate)
    
    def _cefr_to_numeric(self, cefr_level: str) -> float:
        """
        Convert CEFR level to numeric score.
        
        Args:
            cefr_level: CEFR level (A1, A2, B1, B2, C1, C2)
            
        Returns:
            Numeric score (0-5)
        """
        mapping = {
            "A1": 1.0,
            "A2": 2.0,
            "B1": 3.0,
            "B2": 4.0,
            "C1": 5.0,
            "C2": 5.5
        }
        return mapping.get(cefr_level.upper(), 2.5)

