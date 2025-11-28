"""
Turn Analysis Layer - Análise do turno atual e da sessão do estudante
"""

from typing import Optional, Dict, Any


class TurnAnalysisLayer:
    """Camada que fornece análise do turno atual (erros, features, skills) e da sessão completa"""
    
    def __init__(self, turn_analysis: Optional[Dict[str, Any]] = None, session_analysis: Optional[Dict[str, Any]] = None):
        self.turn_analysis = turn_analysis or {}
        self.session_analysis = session_analysis or {}
    
    def render(self) -> str:
        """Renderiza análise do turno atual"""
        if not self.turn_analysis:
            return "[Current Turn Analysis]\nNenhuma análise disponível para este turno."
        
        parts = ["[Current Turn Analysis]"]
        
        # Erros encontrados
        errors = self.turn_analysis.get("errors", [])
        if errors:
            parts.append(f"\nErros identificados ({len(errors)}):")
            for i, error in enumerate(errors[:3], 1):  # Top 3 erros
                error_type = error.get("error_type", "unknown")
                skill_id = error.get("skill_id", "unknown")
                explanation = error.get("explanation", "")
                severity = error.get("severity", "medium")
                
                parts.append(f"  {i}. {error_type} em '{skill_id}' ({severity})")
                if explanation:
                    parts.append(f"     → {explanation}")
        else:
            parts.append("\n✅ Nenhum erro detectado neste turno!")
        
        # Skills usadas corretamente
        correct_skills = self.turn_analysis.get("correct_skills", [])
        if correct_skills:
            parts.append(f"\n✅ Skills usadas corretamente ({len(correct_skills)}):")
            for skill_id in correct_skills[:5]:  # Top 5
                parts.append(f"  - {skill_id}")
        
        # Features linguísticas identificadas
        linguistic_features = self.turn_analysis.get("linguistic_features", {})
        if linguistic_features:
            parts.append("\n📊 Features linguísticas identificadas:")
            feature_parts = []
            if "tense" in linguistic_features:
                feature_parts.append(f"Tempo: {linguistic_features['tense']}")
            if "person" in linguistic_features:
                feature_parts.append(f"Pessoa: {linguistic_features['person']}")
            if "number" in linguistic_features:
                feature_parts.append(f"Número: {linguistic_features['number']}")
            if "register" in linguistic_features:
                feature_parts.append(f"Registro: {linguistic_features['register']}")
            if "domain" in linguistic_features:
                feature_parts.append(f"Domínio: {linguistic_features['domain']}")
            
            if feature_parts:
                parts.append("  " + ", ".join(feature_parts))
        
        # Semantic skill mapping (SINKT)
        semantic_mapping = self.turn_analysis.get("semantic_skill_mapping", {})
        if semantic_mapping:
            parts.append(f"\n🔍 Skills semanticamente relevantes ({len(semantic_mapping)}):")
            # Sort by confidence
            sorted_skills = sorted(semantic_mapping.items(), key=lambda x: x[1], reverse=True)
            for skill_id, confidence in sorted_skills[:3]:  # Top 3
                parts.append(f"  - {skill_id}: {confidence:.0%} confiança")
        
        # Resumo
        summary = self.turn_analysis.get("summary", "")
        if summary:
            parts.append(f"\n📝 Resumo: {summary}")
        
        # Instruções baseadas na análise
        if errors:
            parts.append("\n💡 Instruções pedagógicas:")
            parts.append("  - Foque nos erros identificados ao responder")
            parts.append("  - Use correção apropriada (implicit/explicit) baseada na estratégia")
            parts.append("  - Reforce as skills usadas corretamente")
        elif correct_skills:
            parts.append("\n💡 Instruções pedagógicas:")
            parts.append("  - Parabenize o estudante pelo uso correto das skills")
            parts.append("  - Continue praticando naturalmente")
        
        # Adicionar análise da sessão completa (padrões agregados)
        if self.session_analysis:
            parts.append("\n[Análise da Sessão Completa]")
            
            historical_patterns = self.session_analysis.get("historical_patterns", {})
            if historical_patterns:
                parts.append("Padrões históricos identificados (últimos 90 dias):")
                
                problematic_features = historical_patterns.get("problematic_features", [])
                if problematic_features:
                    parts.append("  Features problemáticas:")
                    for feature in problematic_features[:3]:
                        feature_name = feature.get("feature_key", "").replace(":", " ")
                        error_rate = feature.get("error_rate", 0.0)
                        parts.append(f"    - {feature_name}: {error_rate:.0%} de erros")
                
                mastered_features = historical_patterns.get("mastered_features", [])
                if mastered_features:
                    parts.append("  Features dominadas:")
                    for feature in mastered_features[:3]:
                        feature_name = feature.get("feature_key", "").replace(":", " ")
                        success_rate = feature.get("success_rate", 0.0)
                        parts.append(f"    - {feature_name}: {success_rate:.0%} de acertos")
                
                summary = historical_patterns.get("summary", "")
                if summary:
                    parts.append(f"  Resumo: {summary}")
            
            session_context = self.session_analysis.get("session_context", "")
            if session_context:
                parts.append(f"\n📊 Contexto da sessão: {session_context}")
        
        return "\n".join(parts)

